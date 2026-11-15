"""Benchmark reporting for the revision: decision categories, exits, seed-clustered CIs, replay margins,
threshold sensitivity.  Uses the recorded test packets and the cached intervention outcomes."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
import baselines as BL
from tools import GuardianTools
from intervene import outcome, _load
from score import boot_ci

CORRECT = {"clean": {"monitor"}, "B1_fast_gnss": {"endorse_isolation", "hypothesis_gnss_spoof"},
           "B2_all_anchors": {"escalate_emergency"}, "B3_slow_gnss": {"hypothesis_gnss_spoof"},
           "B4_slow_camera": {"endorse_isolation", "hypothesis_camera_spoof"}}
METHODS = {"monitor_only": BL.monitor_only, "evidence_rule": BL.rule_baseline,
           "context_rule": BL.context_rule, "replay_rule": BL.replay_rule}

def load():
    labels = json.load(open(os.path.join(OUT, "evaluator", "labels.json")))
    adm = json.load(open(os.path.join(OUT, "evaluator", "admissible.json")))
    test = {e: l for e, l in labels.items() if l["split"] == "test"}
    pk = {e: json.load(open(os.path.join(OUT, "packets", "test", f"{e}.json"))) for e in test}
    return test, adm, pk

def categorize(l, dec, adm):
    key = f'{l["scenario"]}|{l["context"]}'
    if dec not in adm[key]: return "incorrect"
    if dec in CORRECT[l["scenario"]]: return "correct"
    if dec == "withhold_request_evidence": return "abstain"
    return "other_admissible"

def evaluate(fn, test, adm, pk, cache):
    cat = {}; by_seed = {}; exits = 0; peak = 0.0; n_amb = 0; corr_seed = {}
    for eid, l in test.items():
        dec = fn(pk[eid])["decision"]; c = categorize(l, dec, adm)
        cat[c] = cat.get(c, 0) + 1
        by_seed.setdefault(l["seed"], []).append(float(c != "incorrect"))
        corr_seed.setdefault(l["seed"], []).append(float(c == "correct"))
        if l["scenario"] in ("B3_slow_gnss", "B4_slow_camera"):
            o = outcome(l, dec, 0.0, cache); n_amb += 1
            exits += int(o["time_outside_envelope_s"] > 0); peak = max(peak, o["peak_after_decision_m"])
    return dict(categories=cat, admissible=sum(v for k, v in cat.items() if k != "incorrect"),
                admissible_ci95=boot_ci(by_seed), correct_ci95=boot_ci(corr_seed),
                exits=exits, n_ambiguity=n_amb, peak_after_decision_m=peak)

def replay_stats(test, pk):
    seen = {}
    for eid, l in test.items():
        k = (l["scenario"], l["seed"])
        if k in seen: continue
        tl = GuardianTools(pk[eid])
        g, c = tl.replay_hypotheses([BL.GNSS_LAT, ["Lane-cam offset"]])["results"]
        seen[k] = dict(kG=g["consistency_score"], kC=c["consistency_score"], gap=c["consistency_score"] - g["consistency_score"])
    out = {}
    for sc in sorted(set(k[0] for k in seen)):
        gaps = [v["gap"] for k, v in seen.items() if k[0] == sc]
        out[sc] = dict(gap_min=min(gaps), gap_max=max(gaps), kG_mean=float(np.mean([v["kG"] for k, v in seen.items() if k[0] == sc])),
                       kC_mean=float(np.mean([v["kC"] for k, v in seen.items() if k[0] == sc])))
    return out, seen

def threshold_sweep(test, adm, pk, cache):
    res = {}; m0 = BL.REPLAY_MARGIN
    for thr in np.round(np.arange(0.02, 0.201, 0.01), 2):
        BL.REPLAY_MARGIN = float(thr)
        r = evaluate(BL.replay_rule, test, adm, pk, cache)
        res[float(thr)] = dict(admissible=r["admissible"], correct=r["categories"].get("correct", 0), exits=r["exits"])
    BL.REPLAY_MARGIN = m0
    return res

if __name__ == "__main__":
    test, adm, pk = load(); cache = _load()
    rep = dict(n_episodes=len(test), n_trajectories=len(set((l["scenario"], l["seed"]) for l in test.values())))
    rep["methods"] = {m: evaluate(fn, test, adm, pk, cache) for m, fn in METHODS.items()}
    rep["replay_gaps"], _ = replay_stats(test, pk)
    rep["threshold_sweep"] = threshold_sweep(test, adm, pk, cache)
    json.dump(rep, open(os.path.join(OUT, "evaluator", "report_v2.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "threshold_sweep"}, indent=1))
    for thr, v in rep["threshold_sweep"].items(): print("threshold", thr, v)
