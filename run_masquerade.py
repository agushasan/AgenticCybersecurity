"""
run_masquerade.py
=================
Example 2, Scenario M2 — the MANEUVER-MASQUERADE attack.

A turn-timed GNSS cross-track spoof, shaped so its drift accumulates only while
the vessel genuinely turns through the planned dog-leg, masquerading as a
slightly wider course.  It is designed to defeat a maneuver-consistency
plausibility check (apparent drift agrees with commanded turning) yet is still
isolated by secure state estimation through structural (2s-sparse) observability,
because the trusted heading channels and the independent radar fix do not support
the extra offset.  This is the attack Example 1 cannot pose: it hides inside
legitimate maneuvering dynamics.

Outputs: figures/fig17_masquerade.png, results/masquerade.json, results/scenarioM2.npz
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
import matplotlib.pyplot as plt

from ship_model import build_ship_model
from route import default_fairway
from attacks import maneuver_masquerade, gnss_walkoff
from simulate import simulate, SimConfig
import style as S
from style import PALETTE as P

def save_npz(name, res):
    np.savez_compressed(f"results/{name}.npz",
                        t=res.t, E_ref=res.E_ref,
                        **{f"xt_{c}": res.x_true[c] for c in res.x_true},
                        **{f"xh_{c}": res.x_hat[c] for c in res.x_hat},
                        attacked=res.attacked, attack_truth=res.attack_truth)

S.apply_style()
FIG = "figures"; os.makedirs(FIG, exist_ok=True)
os.makedirs("results", exist_ok=True)

DYN = [6, 7, 8, 9, 10, 11]   # heading(3) + rate + speed-log(2): continuously exercised
MSTART = 235.0               # onset at the dog-leg turn
T = 440


def main():
    sm = build_ship_model()
    path = default_fairway(sm.p.U)
    U = sm.p.U
    atk = maneuver_masquerade(sm, path, t_start=MSTART, T=T, offset_max=55.0)
    res = simulate(sm, atk, SimConfig(T=T, s=4, seed=7), verbose=False, path=path)

    t = res.t
    # --- tracking performance vs the planned route --------------------------
    track = {}
    for c in ("truth", "naive", "secure"):
        te = res.x_true[c][:, 1] - res.E_ref
        track[c] = dict(rmse=float(np.sqrt(np.mean(te[60:] ** 2))),
                        max=float(np.max(np.abs(te))))
    # nominal (no-attack) secure tracking baseline
    from attacks import no_attack
    res0 = simulate(sm, no_attack(sm), SimConfig(T=T, s=4, seed=7),
                    configs=("secure",), path=path)
    te0 = res0.x_true["secure"][:, 1] - res0.E_ref
    track["secure_nominal"] = dict(rmse=float(np.sqrt(np.mean(te0[60:] ** 2))))

    # --- detection / isolation ---------------------------------------------
    truth = res.attack_truth; dec = res.attacked; post = t >= MSTART
    atk_ch = np.where(np.any(truth, axis=0))[0]
    tp = np.sum(dec[post] & truth[post]); fp = np.sum(dec[post] & ~truth[post])
    fn = np.sum(~dec[post] & truth[post])
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    fa_dyn = float(np.mean(dec[:, DYN][post]))          # FA on dynamic channels, in-maneuver

    # --- maneuver-consistency contrast vs a plain walk-off ------------------
    rref = np.array([path.ref(U * tt)[2] for tt in t])
    def drift_rate(a):
        b = np.array([a(tt, np.zeros(sm.n), sm)[1] for tt in t])
        return np.gradient(b, t), b
    dm, bm = drift_rate(atk)
    dw, bw = drift_rate(gnss_walkoff(sm, MSTART, ramp_rate_E=0.6))
    win = t >= MSTART
    def corr(d):
        x, y = np.abs(rref[win]), np.abs(d[win])
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else float("nan")
    consistency = dict(masquerade=corr(dm), walkoff=corr(dw))

    results = dict(attack=res.attack_name, t_start=MSTART,
                   track=track, detection=dict(precision=float(prec), recall=float(rec),
                   attacked_channels=atk_ch.tolist(), false_alarm_dynamic=fa_dyn),
                   maneuver_consistency_corr=consistency)
    with open("results/masquerade.json", "w") as f:
        json.dump(results, f, indent=2)
    save_npz("scenarioM2", res)
    print(json.dumps(results, indent=2))

    make_figure(res, sm, path, bm, rref, consistency, prec, rec, fa_dyn, track)


def make_figure(res, model, path, bias, rref, consistency, prec, rec, fa_dyn, track):
    U = model.p.U; t = res.t; N = U * t; Eref = res.E_ref
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.32], height_ratios=[1, 1],
                          wspace=0.26, hspace=0.42)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # -- (a) bird's-eye -----------------------------------------------------
    hz = 70.0
    xlo, xhi = Eref.min() - hz - 150, Eref.max() + hz + 150
    axA.fill_betweenx(N, xlo, xhi, color=P["hazard"], alpha=0.08, lw=0)
    axA.fill_betweenx(N, Eref - hz, Eref + hz, color=P["secure"], alpha=0.085, lw=0,
                      label="navigable channel")
    axA.plot(Eref - hz, N, color=P["hazard"], lw=1.1, alpha=0.7)
    axA.plot(Eref + hz, N, color=P["hazard"], lw=1.1, alpha=0.7, label="channel edge")
    axA.plot(Eref, N, color=P["subtle"], lw=1.3, ls=(0, (6, 4)), label="planned route")
    axA.plot(res.x_true["secure"][:, 1], N, color=P["secure"], lw=2.6,
             label="secure: true hull", zorder=5)
    En = res.x_true["naive"][:, 1]; Eb = res.x_hat["naive"][:, 1]
    axA.fill_betweenx(N, En, Eb, color=P["naive"], alpha=0.12, lw=0)
    lc = S.time_colored_line(axA, En, N, t, cmap=S.CMAP_TIME, lw=3.0, zorder=6)
    axA.plot(Eb, N, color=P["believed"], lw=1.6, ls=(0, (2, 2)), zorder=5,
             label="naive: believed (ghost)")
    axA.scatter([En[-1]], [N[-1]], s=120, color=P["naive"], marker="X",
                edgecolor="white", linewidth=1.3, zorder=7, label="naive: final hull")
    ka = int(MSTART / model.p.dt)
    axA.scatter([Eref[ka]], [N[ka]], s=78, color=P["attack"], marker="D",
                edgecolor="white", linewidth=1.2, zorder=8)
    axA.annotate("spoof onset\n(dog-leg)", xy=(Eref[ka], N[ka]), xytext=(-96, -6),
                 textcoords="offset points", fontsize=9.2, color=P["attack"],
                 fontweight="bold", va="center",
                 arrowprops=dict(arrowstyle="->", color=P["attack"], lw=1.3))
    axA.set_xlim(xlo, xhi)
    axA.set_xlabel("cross-track East  E  [m]")
    axA.set_ylabel("along-track North  [m]")
    S.panel_label(axA, "a")
    S.style_legend(axA, loc="lower right", fontsize=8.6)
    cbar = fig.colorbar(lc, ax=axA, fraction=0.045, pad=0.02)
    cbar.set_label("time  [s]", fontsize=9.5); cbar.ax.tick_params(labelsize=8.5)

    # -- (b) the spoof rides the turn --------------------------------------
    turn_deg_s = np.degrees(rref)
    axB.fill_between(t, 0, turn_deg_s, color=P["secure"], alpha=0.16,
                     label="reference turn rate  r$_{ref}$", lw=0)
    axB.plot(t, turn_deg_s, color=P["secure"], lw=1.6)
    axB.set_ylabel("turn rate  [deg/s]", color=P["secure"])
    axB.tick_params(axis="y", labelcolor=P["secure"])
    axB.axhline(0, color=P["subtle"], lw=0.8)
    axBr = axB.twinx(); axBr.grid(False)
    axBr.plot(t, bias, color=P["attack"], lw=2.4, label="injected GNSS bias")
    axBr.set_ylabel("injected cross-track bias  [m]", color=P["attack"])
    axBr.tick_params(axis="y", labelcolor=P["attack"])
    axB.axvspan(235, 300, color=P["gold"], alpha=0.10)
    axB.annotate("bias accumulates only\nwhile turning  (corr $=%.2f$)" % consistency["masquerade"],
                 xy=(300, 0), xytext=(305, axB.get_ylim()[1]*0.60),
                 fontsize=9.0, color=P["ink"])
    axB.set_xlabel("time  [s]")
    axB.set_xlim(t[0], t[-1])
    S.panel_label(axB, "b")
    l1, la1 = axB.get_legend_handles_labels(); l2, la2 = axBr.get_legend_handles_labels()
    axB.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8.6, framealpha=0.95)

    # -- (c) isolation strip + readouts ------------------------------------
    names = model.names
    truth = res.attack_truth; dec = res.attacked
    rows = list(range(model.psensors))
    for r in rows:
        axC.plot(t, np.where(truth[:, r], r, np.nan), color=P["naive_lt"], lw=6,
                 solid_capstyle="butt")
        axC.plot(t, np.where(dec[:, r], r, np.nan), color=P["hazard"], lw=2.4,
                 solid_capstyle="butt")
    axC.axvspan(235, 300, color=P["gold"], alpha=0.10)
    axC.set_yticks(rows); axC.set_yticklabels([names[r] for r in rows], fontsize=8.4)
    axC.invert_yaxis()
    axC.set_xlabel("time  [s]"); axC.set_xlim(t[0], t[-1])
    axC.set_ylim(model.psensors - 0.4, -0.6)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=P["naive_lt"], lw=6, label="ground-truth attacked"),
               Line2D([0], [0], color=P["hazard"], lw=2.4, label="flagged by estimator")]
    axC.legend(handles=handles, loc="upper right", fontsize=8.4, framealpha=0.95)
    S.panel_label(axC, "c")
    txt = (f"secure isolates GNSS-East:  precision {prec:.2f},  recall {rec:.2f}\n"
           f"false alarms on heading/rate/log channels:  {fa_dyn:.3f}\n"
           f"naive dragged {track['naive']['max']:.0f} m off-route;  "
           f"secure RMSE {track['secure']['rmse']:.1f} m ({track['secure_nominal']['rmse']:.1f} m nominal)")
    axC.text(0.03, 0.045, txt, transform=axC.transAxes, fontsize=8.8,
             va="bottom", ha="left", color=P["ink"],
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=P["grid"], lw=1.0))

    S.finalize(fig, f"{FIG}/fig17_masquerade.png")
    print(f"figure -> {FIG}/fig17_masquerade.png")


if __name__ == "__main__":
    main()
