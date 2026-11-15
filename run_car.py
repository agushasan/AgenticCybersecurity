"""
run_car.py
==========
Case study B: autonomous ground vehicle (highway lane keeping).

The same secure estimator, runtime-assurance filter and certificate used for
the surface vessel are applied, unchanged, to a linear single-track car model
with an RTK-GNSS / LiDAR / lane-camera / IMU / odometry suite.

  B1  fast common-mode GNSS spoof (2 m/s lateral drag-off): dynamically
      inconsistent with the trusted heading and lateral-velocity sensors, so
      sparse secure estimation isolates both GNSS lateral channels.
  B2  slow, stealthy common-mode GNSS spoof (0.3 m/s): two of the three lateral
      anchors agree, the drift is dynamically plausible within heading noise,
      and the minimum-cardinality explanation wrongly blames the honest lane
      camera.  The RTA dead-reckoning cross-check catches it.

Outputs: results/car.json, figures/fig18_car.png
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib.pyplot as plt

from vehicle_model import build_car_model, CarAutopilot, car_envelope, DEG
from attacks import gnss_walkoff, single_channel_ramp, combine, no_attack
from simulate import simulate, SimConfig
import certify_rta as CR
import style as S
from style import PALETTE as P

S.apply_style()
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

T0 = 15.0                     # attack onset [s]
T = 600                       # 60 s at 10 Hz
BASE = dict(tau=6, poles=(0.70, 0.73, 0.76, 0.80, 0.84, 0.88), proc_sigma=0.004,
            x0=(0, 0.3, 0, 0, 0, 0), weave_amp=(0.2 * DEG, 0.6 * DEG))
FAST, SLOW = 2.0, 0.3
MRM = 10.0                    # minimal-risk-maneuver window after onset [s]


def det_stats(r, t0):
    tr, dec, t = r.attack_truth, r.attacked, r.t
    post = t >= t0
    tp = np.sum(dec[post] & tr[post]); fp = np.sum(dec[post] & ~tr[post])
    fn = np.sum(~dec[post] & tr[post])
    ch = np.where(np.any(tr, 0))[0]
    lat = next((float(t[k] - t0) for k in np.where(post)[0] if np.any(dec[k, ch])), None)
    return dict(precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
                latency_s=lat, attacked=ch.tolist(),
                false_flag_channels=sorted(set(np.where(dec & ~tr)[1].tolist())))


def main():
    cm = build_car_model(); ap = CarAutopilot(model=cm); env = car_envelope()
    out = dict(e_max=env.e_max, fast_rate=FAST, slow_rate=SLOW)

    # --- B1: fast spoof ------------------------------------------------------
    r1 = simulate(cm, gnss_walkoff(cm, T0, ramp_rate_E=FAST),
                  SimConfig(T=T, s=4, seed=7, **BASE), autopilot=ap)
    out["B1"] = dict(peak={c: float(np.max(np.abs(r1.x_true[c][:, 1]))) for c in r1.x_true},
                     detection=det_stats(r1, T0))
    def allanch(rate):
        return combine(gnss_walkoff(cm, T0, ramp_rate_E=rate),
                       single_channel_ramp(cm, 5, T0, rate=rate, name="lane-camera ramp"),
                       name="all lateral anchors")
    def win_peak(r, t1):
        y = np.abs(r.x_true["secure"][:, 1]); k1 = int(t1 / cm.p.dt)
        return float(y[:k1].max())
    # --- B2: all lateral anchors spoofed (unrecoverable for SSE) -----------
    r2 = simulate(cm, allanch(FAST), SimConfig(T=T, s=5, seed=7, **BASE), configs=("secure",), autopilot=ap)
    r2r = simulate(cm, allanch(FAST), SimConfig(T=T, s=5, seed=7, **BASE), configs=("secure",),
                   autopilot=ap, rta=True, rta_env=env)
    m = r2r.rta_mode["secure"]
    out["B2"] = dict(peak_nortta=float(np.max(np.abs(r2.x_true["secure"][:, 1]))),
                     peak_rta_mrm=win_peak(r2r, T0 + MRM),
                     peak_rta_full=float(np.max(np.abs(r2r.x_true["secure"][:, 1]))),
                     failsafe_trip_s=float(np.argmax(m == 2) * cm.p.dt) if (m == 2).any() else None)
    # --- B3: slow stealthy GNSS spoof (identifiability limit) --------------
    r3 = simulate(cm, gnss_walkoff(cm, T0, ramp_rate_E=SLOW), SimConfig(T=T, s=4, seed=7, **BASE), autopilot=ap)
    r3r = simulate(cm, gnss_walkoff(cm, T0, ramp_rate_E=SLOW), SimConfig(T=T, s=4, seed=7, **BASE),
                   configs=("secure",), autopilot=ap, rta=True, rta_env=env)
    m3 = r3r.rta_mode["secure"]
    out["B3"] = dict(peak={c: float(np.max(np.abs(r3.x_true[c][:, 1]))) for c in r3.x_true},
                     detection=det_stats(r3, T0),
                     peak_rta=float(np.max(np.abs(r3r.x_true["secure"][:, 1]))),
                     time_to_lane_line_rta=float(np.argmax(np.abs(r3r.x_true["secure"][:, 1]) > 1.75) * cm.p.dt)
                         if (np.abs(r3r.x_true["secure"][:, 1]) > 1.75).any() else None,
                     failsafe_trip_s=float(np.argmax(m3 == 2) * cm.p.dt) if (m3 == 2).any() else None)

    # --- multi-seed RTA check (8 seeds) -------------------------------------
    ms = dict(clean_failsafe_runs=0, B1_viol_runs=0, B2_mrm_viol_runs=0, B2_full_max=0.0,
              B2_lane_line_runs=0, seeds=8)
    for sd in range(1, 9):
        cfg4 = SimConfig(T=T, s=4, seed=sd, **BASE); cfg5 = SimConfig(T=T, s=5, seed=sd, **BASE)
        rc = simulate(cm, no_attack(cm), cfg4, configs=("secure",), autopilot=ap, rta=True, rta_env=env)
        ms["clean_failsafe_runs"] += int((rc.rta_mode["secure"] == 2).any())
        rb1 = simulate(cm, gnss_walkoff(cm, T0, ramp_rate_E=FAST), cfg4, configs=("secure",),
                       autopilot=ap, rta=True, rta_env=env)
        ms["B1_viol_runs"] += int((np.abs(rb1.x_true["secure"][:, 1]) > env.e_max).any())
        rb2 = simulate(cm, allanch(FAST), cfg5, configs=("secure",), autopilot=ap, rta=True, rta_env=env)
        ms["B2_mrm_viol_runs"] += int(win_peak(rb2, T0 + MRM) > env.e_max)
        pk = float(np.max(np.abs(rb2.x_true["secure"][:, 1])))
        ms["B2_full_max"] = max(ms["B2_full_max"], pk); ms["B2_lane_line_runs"] += int(pk > 1.75)
    out["multiseed"] = ms

    # --- Theorem 1 certificate for the car ----------------------------------
    obj = CR.build(cm, env, None, autopilot=ap, proc_sigma=BASE["proc_sigma"])
    sup, hi, lo, info = CR.certify(obj, BASE["x0"][1], env.e_max)
    scale = CR.certifiable_scale(obj, BASE["x0"][1], env.e_max)
    out["certificate"] = dict(spectral_radius=obj["schur"],
                              E_half=dict(zip(["dX", "dY", "dpsi", "du", "dv", "dr"], obj["E_half"].tolist())),
                              sup=float(sup), tail_eps=info["tail_eps"], certified=bool(sup <= env.e_max),
                              inflation=float(scale), worst_removed=obj["worst_removed"])
    with open("results/car.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    figure(cm, env, r1, r2, r2r, hi, lo, sup)


def _clip_mark(a, t, y, ymax, label, color):
    """Mark where a trajectory leaves the axis and state its true peak."""
    k = np.where(np.abs(y) > ymax)[0]
    if k.size == 0:
        return
    k0 = k[0]; sgn = np.sign(y[k0]); pk = float(np.max(np.abs(y)))
    a.annotate(f"{label}: peak {pk:.0f} m (clipped)", xy=(t[k0], sgn * ymax * 0.97), xytext=(t[k0] + 9.0, sgn * ymax * 0.72),
               fontsize=8.0, color=color, ha="left", va="center",
               arrowprops=dict(arrowstyle="->", color=color, lw=1.0))


def figure(cm, env, r1, r2, r2r, hi, lo, sup):
    fig, ax = plt.subplots(2, 2, figsize=(12.6, 6.6))
    lane = 1.75
    def lanes(a, ymax):
        a.axhspan(-env.e_max, env.e_max, color=P["secure"], alpha=0.07, lw=0)
        for s_ in (1, -1):
            a.axhline(s_ * env.e_max, color=P["hazard"], ls="--", lw=1.2)
            a.axhline(s_ * lane, color=P["subtle"], ls=":", lw=1.1)
        a.set_ylim(-ymax, ymax)
    # (a) fast spoof
    a = ax[0, 0]; t = r1.t
    for c in ("truth", "naive", "secure"):
        a.plot(t, r1.x_true[c][:, 1], color=S.ROLE[c], lw=2.0 if c != "truth" else 1.4, label=S.LABEL[c])
    lanes(a, 4.0); a.axvline(T0, color=P["attack"], lw=1.2)
    _clip_mark(a, t, r1.x_true["naive"][:, 1], 4.0, "naive", S.ROLE["naive"])
    a.set_ylabel("lateral offset  $\\delta E$  [m]"); a.set_xlabel("time  [s]")
    a.legend(loc="upper right", fontsize=8.6); S.panel_label(a, "a", dx=-0.13, dy=1.04)
    # (b) isolation strip, fast spoof
    a = ax[0, 1]; tr, dec = r1.attack_truth, r1.attacked
    for k in range(cm.psensors):
        a.plot(t, np.where(tr[:, k], k, np.nan), color=P["naive_lt"], lw=6, solid_capstyle="butt")
        a.plot(t, np.where(dec[:, k], k, np.nan), color=P["hazard"], lw=2.4, solid_capstyle="butt")
    a.set_yticks(range(cm.psensors)); a.set_yticklabels(cm.names, fontsize=8.2)
    a.invert_yaxis(); a.set_xlabel("time  [s]"); a.set_xlim(0, t[-1]); a.axvline(T0, color=P["attack"], lw=1.2)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([0], [0], color=P["naive_lt"], lw=6, label="ground-truth attacked"),
                      Line2D([0], [0], color=P["hazard"], lw=2.4, label="flagged by estimator")],
             loc="lower right", fontsize=8.4)
    S.panel_label(a, "b", dx=-0.30, dy=1.04)
    # (c) slow spoof with/without RTA
    a = ax[1, 0]; t = r2.t
    a.plot(t, r2.x_true["secure"][:, 1], color=P["naive"], lw=2.0, label="secure estimator, no RTA")
    a.plot(r2r.t, r2r.x_true["secure"][:, 1], color=P["secure"], lw=2.2, label="secure estimator + RTA")
    a.axvline(T0 + MRM, color=P["ink"], lw=1.0, ls="-.", label="end of 10 s minimal-risk window")
    m = r2r.rta_mode["secure"]
    if (m == 2).any():
        a.axvspan(np.argmax(m == 2) * cm.p.dt, t[-1], color=P["gold"], alpha=0.12, label="fail-safe (heading hold)")
    lanes(a, 2.2); a.axvline(T0, color=P["attack"], lw=1.2)
    _clip_mark(a, t, r2.x_true["secure"][:, 1], 2.2, "no RTA", P["naive"])
    a.set_ylabel("lateral offset  $\\delta E$  [m]"); a.set_xlabel("time  [s]")
    a.legend(loc="upper right", fontsize=8.0, ncol=2); S.panel_label(a, "c", dx=-0.13, dy=1.04)
    # (d) exact Theorem-1 filter under the filter-aware adversary, B3 and B4 (certify/stress_cert.py)
    a = ax[1, 1]
    labs, es, sws = [], [], []
    for tag in ("_a", "_b"):
        tr = np.load(f"results/cert_traces{tag}.npz")
        labs += [str(x) for x in tr["label"]]; es += list(tr["e"]); sws += list(tr["switch"])
        ec = float(tr["e_cert"]); tt = tr["t"]; ts = float(tr["t_start"])
    a.axhspan(-ec, ec, color=P["gold"], alpha=0.12, lw=0)
    for s_ in (1, -1):
        a.axhline(s_ * ec, color=P["gold"], ls="-", lw=1.6, label=f"certified bound {ec:.2f} m" if s_ > 0 else None)
        a.axhline(s_ * env.e_max, color=P["hazard"], ls="--", lw=1.2)
        a.axhline(s_ * lane, color=P["subtle"], ls=":", lw=1.1)
    first = {"B3": True, "B4": True}
    for lb, e_, sw in zip(labs, es, sws):
        sc = lb.split("|")[0][:2]
        if sc not in first: continue
        col = P["secure"] if sc == "B3" else P["attack"]
        a.plot(tt, e_, color=col, lw=0.9, alpha=0.8, label=(("B3, GNSS-free predictor" if sc == "B3" else "B4, switch to camera-free") if first[sc] else None))
        first[sc] = False
        if sc == "B4" and sw > 0:
            a.plot(sw * 0.1, e_[sw], marker="|", color=P["ink"], ms=7, mew=1.2)
    a.axvline(T0, color=P["attack"], lw=1.2); a.axvline(ts, color=P["ink"], lw=1.0, ls="-.", label="adversary starts")
    a.set_ylim(-2.2, 2.2); a.set_xlim(0, tt[-1])
    a.set_ylabel("lateral offset  $\\delta E$  [m]"); a.set_xlabel("time  [s]")
    a.legend(loc="lower left", fontsize=7.8, ncol=2); S.panel_label(a, "d", dx=-0.13, dy=1.04)
    fig.tight_layout()
    S.finalize(fig, "figures/fig18_car.png")
    print("figure -> figures/fig18_car.png")


if __name__ == "__main__":
    main()
