"""
agentic_figs.py
===============
Schematic figures for the agentic supervisory framework (design document):
  fig13  three-tier runtime-assured multi-agent architecture
  fig14  decision / runtime-assurance flow with the autonomy ladder
Consistent with the project visual-design system (style.py).  Titleless;
descriptions live in AGENTIC_FRAMEWORK.md captions.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import style as S
from style import PALETTE as P

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def _box(ax, x, y, w, h, text, fc, ec=None, fs=10.0, tc=None, lw=1.4,
         bold=True, round=0.02):
    ec = ec or P["ink"]
    tc = tc or P["ink"]
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={round}",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=4,
            wrap=True)
    return (x + w / 2, y + h / 2, x, y, w, h)


def _arrow(ax, p0, p1, color=None, lw=1.8, style="-|>", ls="-", rad=0.0,
           alpha=1.0):
    color = color or P["subtle"]
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15,
                        color=color, lw=lw, ls=ls, alpha=alpha,
                        connectionstyle=f"arc3,rad={rad}", zorder=2,
                        shrinkA=3, shrinkB=3)
    ax.add_patch(a)


def _tier_band(ax, y, h, color, label, sub):
    ax.add_patch(Rectangle((0, y), 100, h, fc=color, ec="none", alpha=0.06,
                           zorder=0))
    ax.text(1.4, y + h - 1.6, label, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color=color, zorder=1)
    ax.text(1.4, y + h - 4.6, sub, ha="left", va="top", fontsize=8.2,
            color=P["subtle"], zorder=1, style="italic")


# --------------------------------------------------------------------------- #
#  Fig 13 — architecture                                                      #
# --------------------------------------------------------------------------- #
def fig_architecture(fname):
    fig, ax = plt.subplots(figsize=(12.6, 9.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # tier bands
    _tier_band(ax, 66, 34, P["accent"], "TIER 2 · SUPERVISORY",
               "event-driven · seconds · owns safety/mission trade-off")
    _tier_band(ax, 30, 35, P["secure"], "TIER 1 · TACTICAL AGENTS",
               "≈0.2–1 Hz · goal-directed, tool-using, planning, coordinating")
    _tier_band(ax, 0, 29, P["clean"], "TIER 0 · REFLEXIVE (verified)",
               "control rate · safe on its own · the existing SSE + controller")

    # ---- Tier 0 ----
    veh = _box(ax, 6, 7, 17, 9, "Vessel\n+ sensors y(t)\n(may be attacked)",
               "#ffffff", ec=P["clean"], fs=9.2)
    sse = _box(ax, 30, 7, 17, 9, "Secure State\nEstimator (SSE)\nMML + SMT",
               S.PALETTE["secure_lt"], ec=P["secure"], fs=9.2)
    ctl = _box(ax, 54, 7, 15, 9, "Constrained\nreconfig.\ncontroller",
               "#ffffff", ec=P["clean"], fs=9.2)
    filt = _box(ax, 74, 7, 12, 9, "SAFETY\nFILTER\nCBF/Simplex",
                "#fdece0", ec=P["attack"], fs=9.0)
    act = _box(ax, 89, 7, 8.5, 9, "actu-\nators", "#ffffff", ec=P["clean"], fs=9.0)
    _arrow(ax, (23, 11.5), (30, 11.5), P["secure"])
    _arrow(ax, (47, 11.5), (54, 11.5), P["clean"])
    _arrow(ax, (69, 11.5), (74, 11.5), P["clean"])
    _arrow(ax, (86, 11.5), (89, 11.5), P["attack"])
    _arrow(ax, (93, 16), (93, 22), P["clean"])          # actuator -> vessel loop
    _arrow(ax, (93, 22), (14.5, 22), P["clean"])
    _arrow(ax, (14.5, 22), (14.5, 16), P["clean"])
    ax.text(50, 23.2, "closed loop (vessel motion)", ha="center", va="bottom",
            fontsize=8.0, color=P["subtle"], style="italic")

    # ---- Tier 1 agents ----
    ag_y = 46.5; aw, ah = 19.5, 10.5
    mon = _box(ax, 4, ag_y, aw, ah, "MONITOR\ncalibrated sensor\ntrust  pᵢ(t)",
               S.PALETTE["secure_lt"], ec=P["secure"], fs=9.0)
    dia = _box(ax, 27, ag_y, aw, ah, "DIAGNOSTICIAN\nattack type · set\nactive probing",
               S.PALETTE["secure_lt"], ec=P["secure"], fs=9.0)
    rsk = _box(ax, 50, ag_y, aw, ah, "RISK ASSESSOR\noperational\nrisk  ρ(t)",
               S.PALETTE["secure_lt"], ec=P["secure"], fs=9.0)
    mit = _box(ax, 73, ag_y, aw, ah, "MITIGATION\nPLANNER\nplan ∈ A_cert",
               S.PALETTE["secure_lt"], ec=P["secure"], fs=9.0)
    for a, b in [(mon, dia), (dia, rsk), (rsk, mit)]:
        _arrow(ax, (a[0] + aw / 2, ag_y + ah / 2), (b[0] - aw / 2, ag_y + ah / 2),
               P["secure"])
    # feedback request Planner -> Diagnostician (curved, above)
    _arrow(ax, (mit[0], ag_y + ah), (dia[0], ag_y + ah), P["believed"], lw=1.4,
           ls=(0, (4, 2)), rad=-0.30)
    ax.text(50, ag_y + ah + 4.4, "feedback: request sharper diagnosis / probe",
            ha="center", fontsize=7.6, color=P["believed"], style="italic")

    # blackboard (left-centre) + tools shelf (right), non-overlapping
    _box(ax, 4, 31.3, 55, 5.4,
         "SHARED BLACKBOARD   ·   trust · attack hypothesis · risk · candidate plan · UQ",
         "#eef2f7", ec=P["subtle"], fs=8.2, round=0.012)
    for a in (mon, dia, rsk):
        _arrow(ax, (a[0], ag_y), (min(a[0], 55), 36.7), P["subtle"], lw=1.1,
               ls=(0, (2, 2)))
    _box(ax, 62, 31.3, 35, 5.4,
         "TOOLS  ·  SSE · observability δ₂ₛ · predictor · risk/ENC · what-if",
         "#f6f1e6", ec=P["gold"], fs=7.8, round=0.012)
    _arrow(ax, (82, ag_y), (82, 36.7), P["gold"], lw=1.1, ls=(0, (2, 2)))
    ax.text(82, 44.6, "tool calls", ha="center", fontsize=7.4, color=P["gold"],
            style="italic")

    # ---- Tier 2 ----
    guard = _box(ax, 20, 73, 34, 12,
                 "SUPERVISORY / GUARDIAN AGENT\narbitration · autonomy ladder\nconflict resolution · explanation",
                 "#efe9f6", ec=P["accent"], fs=9.4)
    human = _box(ax, 64, 73, 24, 12, "HUMAN\nOPERATOR / SCC\n(authorization)",
                 "#ffffff", ec=P["ink"], fs=9.4)
    _arrow(ax, (guard[0] + 17, 79), (64, 79), P["accent"])
    _arrow(ax, (64, 76), (guard[0] + 17, 76), P["ink"], ls=(0, (3, 2)))
    ax.text(59, 80.4, "escalate / explain", ha="center", fontsize=7.6,
            color=P["accent"], style="italic")
    ax.text(59, 73.8, "authorize", ha="center", fontsize=7.6, color=P["ink"],
            style="italic")

    # cross-tier arrows
    _arrow(ax, (14.5, 16), (14, ag_y), P["secure"], lw=1.6)   # sensors/residuals up
    ax.text(10.5, 30, "residuals,\nstate", ha="center", fontsize=7.4,
            color=P["secure"], style="italic")
    _arrow(ax, (37, 57), (37, 73), P["accent"], lw=1.6)        # blackboard -> guardian
    ax.text(33.5, 65, "beliefs,\nrisk, plan", ha="center", fontsize=7.4,
            color=P["accent"], style="italic")
    # authorized mitigation: right-margin L-route Guardian -> safety filter
    _arrow(ax, (54, 76), (98.5, 76), P["attack"], lw=1.9, style="-")
    _arrow(ax, (98.5, 76), (98.5, 18), P["attack"], lw=1.9, style="-")
    _arrow(ax, (98.5, 18), (86, 16), P["attack"], lw=1.9)
    ax.text(99.3, 47, "authorized mitigation  (a ∈ A_cert)", rotation=90,
            ha="center", va="center", fontsize=7.8, color=P["attack"],
            style="italic")

    S.finalize(fig, fname)


# --------------------------------------------------------------------------- #
#  Fig 14 — decision / runtime-assurance flow + autonomy ladder               #
# --------------------------------------------------------------------------- #
def fig_decision_flow(fname):
    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # --- left: decision pipeline ---
    ax.text(2, 97, "Decision pipeline", fontsize=11, fontweight="bold",
            color=P["ink"])
    steps = [
        ("evidence\nR(t), y(t)", P["secure"]),
        ("belief\ntrust pᵢ(t) + UQ", P["secure"]),
        ("risk\nρ(t), time-to-hazard", P["gold"]),
        ("decision\na*, autonomy ℓ", P["accent"]),
    ]
    y = 78; xw = 30
    centers = []
    for i, (t, c) in enumerate(steps):
        b = _box(ax, 4, y, xw, 9, t, "#ffffff", ec=c, fs=9.0)
        centers.append((b[0], y))
        if i:
            _arrow(ax, (b[0], y + 9), centers[i-1] if False else (b[0], y + 9),
                   P["subtle"])
        y -= 15
    for i in range(1, len(centers)):
        _arrow(ax, (centers[i-1][0], centers[i-1][1]),
               (centers[i][0], centers[i][1] + 9), P["subtle"])

    # safety filter + actuator
    filt = _box(ax, 4, y - 2, xw, 9, "SAFETY FILTER\nveto / project to safe set",
                "#fdece0", ec=P["attack"], fs=9.0)
    _arrow(ax, (centers[-1][0], centers[-1][1]), (filt[0], y + 7), P["attack"])
    _arrow(ax, (filt[0], y - 2), (filt[0], y - 9), P["clean"])
    _box(ax, 4, y - 18, xw, 7, "actuators (provably safe u*)", "#ffffff",
         ec=P["clean"], fs=8.6)

    # escalation branch
    _arrow(ax, (4 + xw, centers[-1][1] + 4.5), (52, centers[-1][1] + 4.5),
           P["accent"], ls=(0, (4, 2)))
    ax.text(42, centers[-1][1] + 6.6, "escalate?", fontsize=8.0,
            color=P["accent"], style="italic")

    # --- right: autonomy ladder ---
    ax.text(55, 97, "Autonomy ladder  (authority gated by risk × irreversibility × uncertainty)",
            fontsize=10.5, fontweight="bold", color=P["ink"])
    rungs = [
        ("L4  FAIL-SAFE", "obs. lost / filter infeasible → safe state + inform human",
         P["hazard"], "auto + notify"),
        ("L3  AUTHORIZATION", "irreversible / major deviation → human must approve",
         P["ink"], "human"),
        ("L2  SIGNIFICANT", "station-keep · route revert → auto if confident, else advise",
         P["gold"], "auto / advise"),
        ("L1  REVERSIBLE", "reweight · derate · isolate (if observable) → autonomous",
         P["secure"], "autonomous"),
        ("L0  MONITOR", "report only, no action",
         P["clean"], "autonomous"),
    ]
    yy = 12
    for i, (name, desc, col, tag) in enumerate(rungs):
        b = _box(ax, 55, yy, 33, 12.5, "", "#ffffff", ec=col, lw=1.6)
        ax.text(57, yy + 8.6, name, fontsize=9.6, fontweight="bold", color=col,
                ha="left", va="center")
        ax.text(57, yy + 3.6, desc, fontsize=7.7, color=P["ink"], ha="left",
                va="center")
        chip_c = P["ink"] if tag == "human" else P["clean"]
        ax.add_patch(FancyBboxPatch((89.2, yy + 3.7), 9.5, 5,
                     boxstyle="round,pad=0.004,rounding_size=0.02",
                     fc=chip_c, ec="none", alpha=0.85, zorder=4))
        ax.text(93.95, yy + 6.2, tag, fontsize=6.8, color="white", ha="center",
                va="center", fontweight="bold", zorder=5)
        yy += 15.5
    ax.annotate("", xy=(53.3, 12), xytext=(53.3, 90),
                arrowprops=dict(arrowstyle="-|>", color=P["subtle"], lw=2))
    ax.text(51.6, 50, "increasing authority / risk", rotation=90, ha="center",
            va="center", fontsize=8.2, color=P["subtle"], style="italic")

    S.finalize(fig, fname)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    os.makedirs(FIG, exist_ok=True)
    fig_architecture(os.path.join(FIG, "fig13_agentic_architecture.png"))
    fig_decision_flow(os.path.join(FIG, "fig14_autonomy_ladder.png"))
    print("wrote fig13_agentic_architecture.png, fig14_autonomy_ladder.png")
