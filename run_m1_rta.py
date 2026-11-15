#!/usr/bin/env python3
"""
run_m1_rta.py  —  Milestone M1: runtime-assurance safety filter evaluation
=========================================================================
Evaluates the runtime-assurance (RTA) safety filter of the agentic framework
(src/safety_filter.py) on the existing scenarios, with vs without the filter.

Demonstrates the central M1 guarantee: the RTA bounds the vessel's cross-track
excursion and yields ZERO safe-set violations across all scenarios — most
importantly turning the unrecoverable Scenario C from an open-loop runaway into
a bounded, safe degradation via the observability/dead-reckoning-triggered
fail-safe.  No agents are used; this is the verified backbone the agents sit on.

Outputs: figures/fig15_rta_containment.png, results/m1_rta.json.
Reproducible (seeded).
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ship_model import build_ship_model
from route import default_fairway
from attacks import (gnss_walkoff, single_channel_bias, single_channel_ramp,
                     combine, hopping_gnss_campaign, coordinated_consistent)
from simulate import simulate, SimConfig
from safety_filter import SafetyEnvelope
import analysis as AN

FIG = os.path.join(os.path.dirname(__file__), "figures")
RES = os.path.join(os.path.dirname(__file__), "results")
E_MAX = 70.0
ENV = SafetyEnvelope(e_max=E_MAX)


def peak_and_viol(res, path=False):
    e = res.x_true["secure"][:, 1] - (res.E_ref if path else 0.0)
    return float(np.max(np.abs(e))), int(np.sum(np.abs(e) > E_MAX))


def main():
    t0 = time.time()
    sm = build_ship_model()
    fairway = default_fairway(sm.p.U)

    scenarios = {
        "A": dict(atk=single_channel_bias(sm, 1, 120, bias=45.0),
                  cfg=SimConfig(T=320, s=1, seed=7)),
        "B": dict(atk=gnss_walkoff(sm, 120, ramp_rate_E=0.6),
                  cfg=SimConfig(T=460, s=4, seed=7)),
        "C": dict(atk=combine(gnss_walkoff(sm, 120, ramp_rate_E=0.6),
                              single_channel_ramp(sm, 5, 120, rate=0.6,
                                                  name="Radar-East ramp"),
                              name="all East anchors"),
                  cfg=SimConfig(T=420, s=5, seed=7)),
        "D": dict(atk=hopping_gnss_campaign(sm, t_start=100.0),
                  cfg=SimConfig(T=640, s=4, seed=7)),
        "E": dict(atk=coordinated_consistent(sm, 120),
                  cfg=SimConfig(T=400, s=3, seed=7)),
        "M": dict(atk=gnss_walkoff(sm, 180, ramp_rate_E=0.6),
                  cfg=SimConfig(T=440, s=4, seed=7), path=fairway),
    }

    print("=" * 72)
    print("M1 RUNTIME-ASSURANCE EVALUATION  (safe bound e_max = %.0f m)" % E_MAX)
    print("=" * 72)
    print(f"{'scen':5s} | {'peak no-RTA':>11s} {'viol':>5s} | "
          f"{'peak RTA':>9s} {'viol':>5s} | modes N/B/F")
    table = {}
    resC_off = resC_on = None
    peaks = {}
    for name, sc in scenarios.items():
        path = sc.get("path"); is_path = path is not None
        roff = simulate(sm, sc["atk"], sc["cfg"], configs=("secure",),
                        rta=False, path=path)
        ron = simulate(sm, sc["atk"], sc["cfg"], configs=("secure",),
                       rta=True, rta_env=ENV, path=path)
        p_off, v_off = peak_and_viol(roff, is_path)
        p_on, v_on = peak_and_viol(ron, is_path)
        summ = ron.rta_summary["secure"]
        print(f"{name:5s} | {p_off:9.1f} m {v_off:5d} | {p_on:7.1f} m {v_on:5d} | "
              f"{summ['nominal']}/{summ['backup']}/{summ['failsafe']}")
        table[name] = dict(peak_noRTA=p_off, viol_noRTA=v_off,
                           peak_RTA=p_on, viol_RTA=v_on, modes=summ)
        peaks[name] = (p_off, p_on)
        if name == "C":
            resC_off, resC_on = roff, ron

    # multi-seed safety check (zero violations with RTA)
    print("-" * 72)
    tot_viol_rta = 0
    for name, sc in scenarios.items():
        for sd in range(8):
            cfg = SimConfig(T=sc["cfg"].T, s=sc["cfg"].s, seed=sd)
            r = simulate(sm, sc["atk"], cfg, configs=("secure",), rta=True,
                         rta_env=ENV, path=sc.get("path"))
            _, v = peak_and_viol(r, sc.get("path") is not None)
            tot_viol_rta += v
    print(f"Multi-seed (6 scenarios x 8 seeds) safe-set violations WITH RTA: "
          f"{tot_viol_rta}  (target 0)")

    AN.fig_rta(resC_off, resC_on, E_MAX, peaks, attack_start=120.0,
               fname=f"{FIG}/fig15_rta_containment.png")
    with open(os.path.join(RES, "m1_rta.json"), "w") as f:
        json.dump(dict(e_max=E_MAX, scenarios=table,
                       multiseed_violations_with_rta=tot_viol_rta),
                  f, indent=2, default=str)
    print(f"\nFigure -> {FIG}/fig15_rta_containment.png")
    print(f"Runtime: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
