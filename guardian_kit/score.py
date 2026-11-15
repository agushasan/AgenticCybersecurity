"""
score.py  —  Score supervisory decisions against the predefined rubric and outcomes.
====================================================================================
Usage:
  python3 guardian_kit/score.py                       # baselines only
  python3 guardian_kit/score.py --runs runs.jsonl     # plus Guardian / ablation logs

runs.jsonl has one JSON object per (episode, repeat):
  {"episode_id": "...", "rep": 0, "method": "guardian" | "no_tool_llm",
   "decision": {"decision": "...", "autonomy_level": "L1", "explanation": "...",
                "claims": [{"tool": "...", "quantity": "...", "value": 1.23}]} | null,
   "tool_calls": [{"name": "...", "args": {...}, "result": {...}}],
   "latency_ms": 2400, "error": null}
Metrics (test split only): admissible-decision rate, malformed or late rate,
unsupported claims (claimed values absent from the tool results), numbers in
the explanation not traceable to any tool result, tool calls per episode,
latency median and p95, and closed-loop outcomes for the ambiguity episodes.
Uncertainty: 95% bootstrap intervals resampling simulator seeds (clusters).
"""
from __future__ import annotations
import os, sys, json, re, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import OUT, DECISIONS, LEVELS
from baselines import METHODS
from intervene import outcome, _load, MAX_LATENCY_S

REL, ABS = 0.05, 1e-3


def _numbers(obj):
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _numbers(v)]
    return []


def _match(v, pool):
    return any(abs(v - p) <= max(ABS, REL * abs(p)) for p in pool)


def faithfulness(decision, tool_calls):
    pool_by_tool = {}
    for c in tool_calls or []:
        pool_by_tool.setdefault(c.get("name"), []).extend(_numbers(c.get("result")))
    allpool = [x for v in pool_by_tool.values() for x in v]
    claims = decision.get("claims") or []
    unsup = sum(1 for c in claims if not isinstance(c.get("value"), (int, float))
                or not _match(float(c["value"]), pool_by_tool.get(c.get("tool"), [])))
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", decision.get("explanation", ""))]
    nums = [x for x in nums if abs(x) > 1e-9 and x not in (0.0, 1.0, 2.0, 3.0, 4.0)]   # ignore L-levels, counts
    untraced = sum(1 for x in nums if not _match(x, allpool))
    return dict(claims=len(claims), unsupported_claims=unsup, explanation_numbers=len(nums),
                untraceable_numbers=untraced)


def valid(dec):
    return isinstance(dec, dict) and dec.get("decision") in DECISIONS and dec.get("autonomy_level") in LEVELS


def boot_ci(seed_vals, B=2000, rng=np.random.default_rng(0)):
    seeds = sorted(seed_vals); vals = [np.mean(seed_vals[s]) for s in seeds]
    if len(vals) < 2:
        return [float(np.mean(vals)), float(np.mean(vals))] if vals else [None, None]
    bs = [np.mean(rng.choice(vals, len(vals))) for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=os.path.join(OUT, "evaluator", "scores.json")); a = ap.parse_args()
    labels = json.load(open(os.path.join(OUT, "evaluator", "labels.json")))
    adm = json.load(open(os.path.join(OUT, "evaluator", "admissible.json")))
    test = {e: l for e, l in labels.items() if l["split"] == "test"}
    packets = {os.path.basename(f)[:-5]: f for f in glob.glob(os.path.join(OUT, "packets", "test", "*.json"))}
    rows = []                                           # (method, eid, rep, decision, tool_calls, latency_s, error)
    for m, fn in METHODS.items():
        for eid in test:
            rows.append((m, eid, 0, fn(json.load(open(packets[eid]))), [], 0.0, None))
    ORACLE = {"clean": "monitor", "B1_fast_gnss": "endorse_isolation", "B2_all_anchors": "escalate_emergency",
              "B3_slow_gnss": "hypothesis_gnss_spoof", "B4_slow_camera": "endorse_isolation"}
    for eid, l in test.items():                          # label-informed reference, not a method
        rows.append(("oracle_reference", eid, 0, dict(decision=ORACLE[l["scenario"]], autonomy_level="L1",
                                                      explanation="", claims=[]), [], 0.0, None))
    if a.runs:
        for line in open(a.runs):
            r = json.loads(line)
            if r["episode_id"] in test:
                rows.append((r["method"], r["episode_id"], r.get("rep", 0), r.get("decision"),
                             r.get("tool_calls", []), (r.get("latency_ms") or 0) / 1000.0, r.get("error")))
    cache = _load(); res = {}
    for m in sorted(set(r[0] for r in rows)):
        R = [r for r in rows if r[0] == m]
        ok_by_seed, out_by_seed = {}, {}
        n_bad = n_late = 0; fa = []; tc = []; lat = []; by_group = {}
        for _, eid, rep, dec, calls, lat_s, err in R:
            l = test[eid]; key = f'{l["scenario"]}|{l["context"]}'
            good = valid(dec) and err is None; late = lat_s > MAX_LATENCY_S
            n_bad += (not good); n_late += late
            okv = float(good and not late and dec["decision"] in adm[key])
            ok_by_seed.setdefault(l["seed"], []).append(okv)
            by_group.setdefault(key, []).append(okv)
            if good:
                fa.append(faithfulness(dec, calls))
            tc.append(len(calls)); lat.append(lat_s)
            if l["scenario"] in ("B3_slow_gnss", "B4_slow_camera"):
                o = outcome(l, dec["decision"] if good else None, lat_s if good else None, cache)
                out_by_seed.setdefault(l["seed"], []).append(o["time_outside_envelope_s"])
        allok = [x for v in ok_by_seed.values() for x in v]
        res[m] = dict(n=len(R), admissible_rate=float(np.mean(allok)), admissible_ci95=boot_ci(ok_by_seed),
                      malformed=int(n_bad), late=int(n_late),
                      unsupported_claims=int(sum(f["unsupported_claims"] for f in fa)),
                      claims=int(sum(f["claims"] for f in fa)),
                      untraceable_numbers=int(sum(f["untraceable_numbers"] for f in fa)),
                      explanation_numbers=int(sum(f["explanation_numbers"] for f in fa)),
                      tool_calls_mean=float(np.mean(tc)), latency_median_s=float(np.median(lat)),
                      latency_p95_s=float(np.percentile(lat, 95)),
                      ambiguity_time_outside_mean_s=float(np.mean([x for v in out_by_seed.values() for x in v]))
                      if out_by_seed else None,
                      ambiguity_time_outside_ci95=boot_ci(out_by_seed) if out_by_seed else None,
                      admissible_by_group={k: float(np.mean(v)) for k, v in sorted(by_group.items())})
        print(f"{m:15s} admissible {res[m]['admissible_rate']:.3f} {np.round(res[m]['admissible_ci95'],3)}  "
              f"time-outside(ambig) {res[m]['ambiguity_time_outside_mean_s']}")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("scores ->", a.out)


if __name__ == "__main__":
    main()
