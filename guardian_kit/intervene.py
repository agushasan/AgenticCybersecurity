"""
intervene.py  —  Apply a supervisory decision through Tier 0 and measure the outcome.
====================================================================================
Only a validated threat hypothesis changes Tier 0: it re-weights the SMT support
search (threat prior) from the decision step plus the measured latency onward,
on the same seeded run, which is equivalent to branching the simulator from the
saved state at that step.  All other decisions leave Tier 0 unchanged.  The
runtime monitor stays active in every branch and cannot be altered by the agent.
Results are cached in data/evaluator/outcomes_cache.json.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import car, scenarios, prior_weights, OUT, BASE, T
from simulate import simulate, SimConfig

CACHE = os.path.join(OUT, "evaluator", "outcomes_cache.json")
MAX_LATENCY_S = 5.0                      # later proposals are logged as late and ignored
_cm = _ap = _env = None


def _load():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def outcome(label, decision, latency_s=0.0, cache=None):
    global _cm, _ap, _env
    if _cm is None:
        _cm, _ap, _env = car()
    dt = _cm.p.dt
    late = latency_s is None or latency_s > MAX_LATENCY_S
    hyp = None if late else (decision if decision in ("hypothesis_gnss_spoof", "hypothesis_camera_spoof") else None)
    step = None if hyp is None else min(int(label["decision_step"] + round(latency_s / dt)), T - 1)
    key = f'{label["scenario"]}|{label["seed"]}|{hyp}|{step}'
    cache = _load() if cache is None else cache
    if key not in cache:
        cfg = SimConfig(T=T, s=4, seed=label["seed"], switch_step=step,
                        switch_weights=prior_weights(_cm, hyp) if hyp else None, **BASE)
        r = simulate(_cm, scenarios(_cm)[label["scenario"]](), cfg, configs=("secure",),
                     autopilot=_ap, rta=True, rta_env=_env)
        e = np.abs(r.x_true["secure"][:, 1]); kd = label["decision_step"]; m = r.rta_mode["secure"]
        cache[key] = dict(peak_after_decision_m=float(e[kd:].max()),
                          time_outside_envelope_s=float((e[kd:] > _env.e_max).sum() * dt),
                          rmse_after_decision_m=float(np.sqrt(np.mean(e[kd:] ** 2))),
                          failsafe_time_s=float((m[kd:] == 2).sum() * dt),
                          final_error_m=float(e[-1]), applied_hypothesis=hyp, late=bool(late))
        with open(CACHE, "w") as f:
            json.dump(cache, f, indent=1)
    return cache[key]
