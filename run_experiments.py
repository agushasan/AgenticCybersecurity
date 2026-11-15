#!/usr/bin/env python3
"""
run_experiments.py
==================
Reproducible driver for the maritime secure-state-estimation (SSE) study.

Methodology: SMT-based secure state estimation (Multi-Modal Luenberger observer +
pseudo-Boolean conflict-driven attack isolation), after

    Y. Shoukry et al., "SMT-Based Observer Design for Cyber-Physical Systems
    under Sensor Attacks", ACM T. Cyber-Phys. Syst. 2(1):5, 2018.

applied to a linearised 3-DOF ship with a redundant navigation sensor suite.

The study comprises TWO simulation examples sharing one model, estimator and
visual-design system:

  EXAMPLE 1 — straight-transit sensor-attack study (Scenarios A–E)
    A  single-channel spoof (within certified s_max = 1)
    B  GNSS common-mode walk-off (s = 4)
    C  limitation: all cross-track anchors attacked (unobservable)
    D  adaptive multi-phase campaign (time-varying attack support)
    E  coordinated cross-consistent attack (GNSS position + gyro heading)

  EXAMPLE 2 — maneuvering trajectory-tracking study (Scenario M)
    the vessel follows a curving predefined fairway (S-bend + dog-leg) under
    line-of-sight guidance; a GNSS walk-off during the maneuver drags the naive-
    navigated hull out of the channel while the secure estimator holds the route.

  MC  Monte-Carlo detection study vs walk-off aggressiveness.

Figures carry no titles (descriptions live in the report captions).
Outputs: results/*.npz, figures/*.png, printed metrics. All seeded/reproducible.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ship_model import build_ship_model
from sparse_observability import analyze_suite
from attacks import (gnss_walkoff, single_channel_bias, single_channel_ramp,
                     combine, hopping_gnss_campaign, coordinated_consistent)
from route import default_fairway
from simulate import simulate, SimConfig
import analysis as AN
from style import PALETTE as P

FIG = os.path.join(os.path.dirname(__file__), "figures")
RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

HAZARD_E = 90.0
ATTACK_START = 120.0


def banner(s):
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)


def save_npz(name, res):
    d = dict(t=res.t, attacked=res.attacked, attack_truth=res.attack_truth,
             y_clean=res.y_clean, y_att=res.y_att, attack_name=res.attack_name,
             E_ref=res.E_ref if res.E_ref is not None else np.zeros_like(res.t))
    for c in res.x_true:
        d[f"xtrue_{c}"] = res.x_true[c]; d[f"xhat_{c}"] = res.x_hat[c]
        d[f"u_{c}"] = res.u[c]
    if res.smt_trace is not None:
        d["smt_R"] = res.smt_trace["R"]; d["smt_env"] = res.smt_trace["env"]
    np.savez_compressed(os.path.join(RES, name + ".npz"), **d)


def print_metrics(m):
    print(f"  Attack: {m['attack']}")
    for c in ("truth", "naive", "secure"):
        if c in m:
            mc = m[c]
            print(f"   [{c:6s}] xtrack est-RMSE={mc['crosstrack_est_rmse']:7.2f} m | "
                  f"TRUE max|δE|={mc['true_max_crosstrack']:7.2f} m | "
                  f"final|δE|={mc['true_final_crosstrack']:7.2f} m")
    if "detection" in m:
        d = m["detection"]
        print(f"   [detect] precision={d['precision']:.3f} recall={d['recall']:.3f} "
              f"FA(pre)={d['false_alarm_rate']:.3f} latency={d['latency_s']} s "
              f"attacked-ch={d['attacked_channels']}")


def route_tracking_metrics(res, attack_start):
    out = {"attack": res.attack_name}
    mask = res.t >= (attack_start + 10)
    for c in res.x_true:
        te = res.x_true[c][:, 1] - res.E_ref
        out[c] = {"track_rmse": float(np.sqrt(np.mean(te[mask] ** 2))),
                  "track_max": float(np.max(np.abs(te))),
                  "true_max_crosstrack": float(np.max(np.abs(te)))}
    if res.attacked is not None:
        out["detection"] = AN.compute_metrics(res, attack_start).get("detection", {})
    return out


def main():
    t0 = time.time()
    sm = build_ship_model()
    metrics_all = {}; impact = {}

    # ======================================================================= #
    #  SECURITY ANALYSIS                                                       #
    # ======================================================================= #
    banner("SPARSE-OBSERVABILITY SECURITY ANALYSIS")
    sec = analyze_suite(sm, verbose=True)
    with open(os.path.join(RES, "security_analysis.json"), "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in sec.items() if k != "device_report"},
                  f, indent=2, default=str)
    AN.fig_observability(sm, sec, fname=f"{FIG}/fig01_observability.png")

    # ======================================================================= #
    #  EXAMPLE 1 — straight transit (Scenarios A–E)                           #
    # ======================================================================= #
    banner("EXAMPLE 1 / SCENARIO A - single-channel spoof (within s_max = 1)")
    resA = simulate(sm, single_channel_bias(sm, 1, ATTACK_START, bias=45.0),
                    SimConfig(T=320, s=1, seed=7), verbose=True)
    mA = AN.compute_metrics(resA, ATTACK_START)
    metrics_all["A single-channel"] = mA; impact["A\nsingle-ch"] = mA
    print_metrics(mA); save_npz("scenarioA", resA)
    AN.fig_dashboard(resA, sm, HAZARD_E, ATTACK_START, fname=f"{FIG}/fig03_A_dashboard.png")

    banner("EXAMPLE 1 / SCENARIO B - GNSS common-mode walk-off (s = 4)")
    resB = simulate(sm, gnss_walkoff(sm, ATTACK_START, ramp_rate_E=0.6),
                    SimConfig(T=460, s=4, seed=7), verbose=True)
    mB = AN.compute_metrics(resB, ATTACK_START)
    metrics_all["B GNSS walk-off"] = mB; impact["B\nwalk-off"] = mB
    print_metrics(mB); save_npz("scenarioB", resB)
    AN.fig_birdseye(resB, sm, HAZARD_E, ATTACK_START, fname=f"{FIG}/fig02_B_birdseye.png")
    AN.fig_dashboard(resB, sm, HAZARD_E, ATTACK_START, fname=f"{FIG}/fig04_B_dashboard.png")
    AN.fig_sensor_streams(resB, sm, [1, 3, 5, 6], ATTACK_START,
                          fname=f"{FIG}/fig05_B_sensors.png")

    banner("EXAMPLE 1 / SCENARIO C - limitation (all East anchors attacked)")
    atkC = combine(gnss_walkoff(sm, ATTACK_START, ramp_rate_E=0.6),
                   single_channel_ramp(sm, 5, ATTACK_START, rate=0.6,
                                       name="Radar-East ramp"),
                   name="GNSS walk-off + radar-East (all East anchors)")
    resC = simulate(sm, atkC, SimConfig(T=420, s=5, seed=7), verbose=True)
    mC = AN.compute_metrics(resC, ATTACK_START)
    metrics_all["C all anchors"] = mC; impact["C\nall anchors"] = mC
    print_metrics(mC); save_npz("scenarioC", resC)
    AN.fig_dashboard(resC, sm, HAZARD_E, ATTACK_START, fname=f"{FIG}/fig06_C_dashboard.png")

    banner("EXAMPLE 1 / SCENARIO D - adaptive multi-phase campaign")
    resD = simulate(sm, hopping_gnss_campaign(sm, t_start=100.0),
                    SimConfig(T=640, s=4, seed=7), verbose=True)
    mD = AN.compute_metrics(resD, 100.0)
    metrics_all["D adaptive campaign"] = mD; impact["D\ncampaign"] = mD
    print_metrics(mD); save_npz("scenarioD", resD)
    phasesD = [(100, 170, "P1 probe", P["attack"]),
               (210, 310, "P2 walk-off", P["hazard"]),
               (370, 450, "P3 heading", P["accent"]),
               (510, 620, "P4 coordinated", P["believed"])]
    AN.fig_campaign(resD, sm, phasesD, attack_start=100.0,
                    fname=f"{FIG}/fig07_D_campaign.png")

    banner("EXAMPLE 1 / SCENARIO E - coordinated cross-consistent")
    resE = simulate(sm, coordinated_consistent(sm, ATTACK_START),
                    SimConfig(T=400, s=3, seed=7), verbose=True)
    mE = AN.compute_metrics(resE, ATTACK_START)
    metrics_all["E coordinated"] = mE; impact["E\ncoordinated"] = mE
    print_metrics(mE); save_npz("scenarioE", resE)
    AN.fig_dashboard(resE, sm, HAZARD_E, ATTACK_START, fname=f"{FIG}/fig08_E_dashboard.png")

    # ======================================================================= #
    #  EXAMPLE 2 — maneuvering trajectory tracking (Scenario M)               #
    # ======================================================================= #
    banner("EXAMPLE 2 / SCENARIO M - maneuvering fairway under GNSS walk-off")
    path = default_fairway(sm.p.U)
    print(f"  fairway length {path.total_length:.0f} m, "
          f"max course {path.max_course_deg:.1f} deg")
    M_START = 180.0
    resM = simulate(sm, gnss_walkoff(sm, M_START, ramp_rate_E=0.6),
                    SimConfig(T=440, s=4, seed=7), verbose=True, path=path)
    mM = route_tracking_metrics(resM, M_START)
    metrics_all["M maneuvering"] = mM; impact["M\nmaneuvering"] = mM
    print(f"  Attack: {mM['attack']}")
    for c in ("truth", "naive", "secure"):
        print(f"   [{c:6s}] tracking RMSE={mM[c]['track_rmse']:7.2f} m | "
              f"max off-route={mM[c]['track_max']:7.2f} m")
    if "detection" in mM and mM["detection"]:
        d = mM["detection"]
        print(f"   [detect] precision={d.get('precision'):.3f} "
              f"recall={d.get('recall'):.3f} latency={d.get('latency_s')} s")
    save_npz("scenarioM", resM)
    AN.fig_route_birdseye(resM, sm, hazard_E=70.0, attack_start=M_START,
                          fname=f"{FIG}/fig09_M_birdseye.png")
    AN.fig_route_performance(resM, sm, attack_start=M_START,
                             fname=f"{FIG}/fig10_M_performance.png")
    # nominal (no-attack) tracking baseline for the report
    from attacks import no_attack
    resM0 = simulate(sm, no_attack(sm), SimConfig(T=440, s=4, seed=7),
                     configs=("truth", "secure"), path=path)
    te0 = resM0.x_true["secure"][:, 1] - resM0.E_ref
    print(f"   [nominal] secure tracking RMSE (no attack) = "
          f"{np.sqrt(np.mean(te0[60:] ** 2)):.2f} m")

    # ======================================================================= #
    #  CROSS-EXAMPLE IMPACT SUMMARY + MONTE-CARLO                              #
    # ======================================================================= #
    AN.fig_impact_summary(impact, fname=f"{FIG}/fig11_impact_summary.png")

    banner("MONTE-CARLO - detection vs walk-off aggressiveness")
    rates = [0.15, 0.3, 0.5, 0.8, 1.2]; seeds = list(range(8))
    lat_m, lat_s, rec_m, rec_s = [], [], [], []
    for rate in rates:
        lats, recs = [], []
        for sd in seeds:
            r = simulate(sm, gnss_walkoff(sm, ATTACK_START, ramp_rate_E=rate),
                         SimConfig(T=300, s=4, seed=sd), configs=("secure",))
            d = AN.compute_metrics(r, ATTACK_START)["detection"]
            if not np.isnan(d["latency_s"]):
                lats.append(d["latency_s"])
            recs.append(d["recall"])
        lat_m.append(np.mean(lats) if lats else np.nan)
        lat_s.append(np.std(lats) if lats else 0.0)
        rec_m.append(np.nanmean(recs)); rec_s.append(np.nanstd(recs))
        print(f"  rate={rate:.2f} m/s: latency={lat_m[-1]:.1f}±{lat_s[-1]:.1f} s, "
              f"recall={rec_m[-1]:.2f}")
    mc = dict(rates=rates, latency_mean=lat_m, latency_std=lat_s,
              recall_mean=rec_m, recall_std=rec_s)
    np.savez_compressed(os.path.join(RES, "montecarlo.npz"), **mc)
    AN.fig_mc(mc, fname=f"{FIG}/fig12_MC_detection.png")

    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2, default=str)

    banner("DONE")
    print(f"Total runtime: {time.time()-t0:.1f} s")
    print(f"Figures -> {FIG}\nResults -> {RES}")


if __name__ == "__main__":
    main()
