"""
qwen_runner.py  —  Run the Guardian (or the no-tool ablation) on the evidence packets.
=====================================================================================
Talks to any OpenAI-compatible chat endpoint with tool calling, for example
  Ollama:  http://localhost:11434/v1     vLLM:  http://localhost:8000/v1
Each (episode, repeat) is a fresh session.  Writes one JSON line per run in the
format score.py expects.  Uses only the Python standard library.

Example (pin the exact checkpoint you report in the paper):
  python3 guardian_kit/qwen_runner.py --base-url http://localhost:11434/v1 \
      --model qwen2.5:14b-instruct-q4_K_M --method guardian --reps 20 --out runs.jsonl
  python3 guardian_kit/qwen_runner.py ... --method no_tool_llm --reps 20 --out runs.jsonl
  python3 guardian_kit/score.py --runs runs.jsonl

--mock runs a scripted stand-in (rule logic driven through the same tool loop)
to test the plumbing without a model.
"""
from __future__ import annotations
import os, sys, json, glob, time, re, argparse, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from tools import GuardianTools, TOOL_SCHEMAS

KIT = os.path.dirname(os.path.abspath(__file__))
SYS_G = open(os.path.join(KIT, "prompts", "system_guardian.txt")).read()
SYS_N = open(os.path.join(KIT, "prompts", "system_notool.txt")).read()


def user_message(pk):
    mc = pk["model_card"]
    return json.dumps(dict(episode_id=pk["episode_id"], decision_time_s=pk["decision_time_s"],
                           platform=mc["platform"], speed_mps=mc["speed_mps"], e_max_m=mc["e_max_m"],
                           lane_line_m=mc["lane_line_m"], channels=[c["name"] for c in mc["channels"]],
                           instruction="Inspect the evidence with the tools, then answer with the JSON object."))


def digest(pk):
    """Same evidence without derived quantities (no drift rate, rank or implied heading)."""
    s = pk["series"]; k = len(s["t"]) - 1; step = max(1, int(round(1.0 / pk["model_card"]["dt_s"])))
    idx = list(range(max(0, k - 5 * step), k + 1, step))
    return json.dumps(dict(
        episode_id=pk["episode_id"], decision_time_s=pk["decision_time_s"],
        speed_mps=pk["model_card"]["speed_mps"], e_max_m=pk["model_card"]["e_max_m"],
        events=pk["events"], flagged_now=[c for c, v in s["flags"].items() if v[-1]],
        residual_ratio_now={c: v[-1] for c, v in s["residual_ratio"].items()},
        dr_discrepancy_now=s["dr_discrepancy"][-1], dr_gate_now=s["dr_gate"][-1],
        last_5s_at_1Hz=dict(t=[s["t"][i] for i in idx],
                            **{c: [v[i] for i in idx] for c, v in s["measurement"].items()}),
        advisories=pk.get("context", [])))


def chat(base, model, messages, tools, seed, timeout):
    body = dict(model=model, messages=messages, temperature=0, seed=seed)
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none")})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    return json.loads(m.group(0)) if m else None


def mock_model(pk, tl, calls):
    """Scripted stand-in: queries tools in a realistic order, then decides like the rule baseline."""
    from baselines import rule_baseline
    for name, args in [("get_event_log", {}), ("get_dr_status", {}), ("lateral_disagreement_rate", {})]:
        calls.append(dict(name=name, args=args, result=tl.call(name, args)))
    d = rule_baseline(pk); dr = calls[2]["result"]["drift_rate_mps"]
    d.update(explanation=f"Lateral disagreement drifts at {dr} m/s.",
             claims=[dict(tool="lateral_disagreement_rate", quantity="drift rate", value=dr)])
    return d


def run_episode(pk, a, rep):
    tl = GuardianTools(pk); calls = []; t0 = time.time(); err = None; dec = None
    try:
        if a.mock:
            dec = mock_model(pk, tl, calls)
        elif a.method == "guardian":
            msgs = [dict(role="system", content=SYS_G), dict(role="user", content=user_message(pk))]
            for _ in range(a.max_rounds):
                msg = chat(a.base_url, a.model, msgs, TOOL_SCHEMAS, a.seed + rep, a.timeout)
                tcs = msg.get("tool_calls") or []
                if not tcs:
                    dec = parse_json(msg.get("content")); break
                msgs.append(dict(role="assistant", content=msg.get("content") or "", tool_calls=tcs))
                for tc in tcs:
                    fn = tc["function"]; args = fn.get("arguments") or "{}"
                    args = json.loads(args) if isinstance(args, str) else args
                    res = tl.call(fn["name"], args)
                    calls.append(dict(name=fn["name"], args=args, result=res))
                    msgs.append(dict(role="tool", tool_call_id=tc.get("id", fn["name"]), content=json.dumps(res)))
            else:
                err = "max_rounds"
        else:
            msg = chat(a.base_url, a.model, [dict(role="system", content=SYS_N),
                                             dict(role="user", content=digest(pk))], None, a.seed + rep, a.timeout)
            dec = parse_json(msg.get("content"))
            calls = [dict(name="digest", args={}, result=json.loads(digest(pk)))]
    except Exception as e:                                    # malformed or failed call is logged, not hidden
        err = f"{type(e).__name__}: {e}"
    return dict(episode_id=pk["episode_id"], rep=rep, method=("mock" if a.mock else a.method),
                model=dict(name=a.model, base_url=a.base_url, temperature=0, seed=a.seed + rep),
                decision=dec, tool_calls=calls, latency_ms=int(1000 * (time.time() - t0)), error=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1"); ap.add_argument("--model", default="")
    ap.add_argument("--method", choices=["guardian", "no_tool_llm"], default="guardian")
    ap.add_argument("--split", choices=["dev", "test"], default="test"); ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=120); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--out", default="runs.jsonl")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(OUT, "packets", a.split, "*.json")))[: (a.limit or None)]
    with open(a.out, "a") as f:
        for fp in files:
            pk = json.load(open(fp))
            for rep in range(a.reps):
                rec = run_episode(pk, a, rep); f.write(json.dumps(rec) + "\n"); f.flush()
                print(pk["episode_id"], rep, rec["decision"] and rec["decision"].get("decision"), rec["error"])


if __name__ == "__main__":
    main()
