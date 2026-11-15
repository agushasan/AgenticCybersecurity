"""
style.py
========
A single, cohesive visual-design system for every figure in the study:
a curated palette, typographic/axis defaults, custom colormaps, and reusable
helpers (panel labels, event annotations, phase shading, time-coloured tracks,
hazard corridors).  Importing this module applies the global style.
"""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------- #
#  Palette  (colourblind-aware, consistent semantic roles)                    #
# --------------------------------------------------------------------------- #
PALETTE = dict(
    ink      = "#1b2330",   # text / axes
    subtle   = "#5b6675",   # secondary text
    grid     = "#d9dee6",
    panel    = "#f4f6f9",   # light panel fill
    truth    = "#263238",   # ground truth / ideal
    naive    = "#e5484d",   # naive KF (danger / compromised)
    naive_lt = "#f6b8ba",
    secure   = "#1f77b4",   # secure SSE (protected)
    secure_lt= "#b3d2ea",
    attack   = "#f08a24",   # attack signal / onset
    hazard   = "#c0202a",   # hazard zone
    clean    = "#2a9d8f",   # clean sensors
    believed = "#8e44ad",   # "believed" (ghost) position
    accent   = "#6b4fbb",
    gold     = "#e0a500",
)
ROLE = {"truth": PALETTE["truth"], "ideal": PALETTE["truth"],
        "naive": PALETTE["naive"], "secure": PALETTE["secure"]}
ROLE_LT = {"naive": PALETTE["naive_lt"], "secure": PALETTE["secure_lt"]}
LABEL = {"truth": "Ideal (true-state control)",
         "naive": "Naive Kalman filter",
         "secure": "Secure SMT estimator"}

# --------------------------------------------------------------------------- #
#  Custom colormaps                                                           #
# --------------------------------------------------------------------------- #
CMAP_RESID = LinearSegmentedColormap.from_list(
    "resid", ["#f4f6f9", "#fde9c8", "#f6b24b", "#e5693b", "#b11226"])
CMAP_TIME = LinearSegmentedColormap.from_list(
    "timeflow", ["#2c3e78", "#2f8fae", "#56c596", "#d7e05a"])
CMAP_COVER = LinearSegmentedColormap.from_list(
    "cover", ["#f4f6f9", "#b3d2ea", "#1f77b4", "#15314f"])


# --------------------------------------------------------------------------- #
#  Global rcParams                                                            #
# --------------------------------------------------------------------------- #
def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 12.0,
        "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlepad": 8, "axes.labelsize": 12.5,
        "axes.labelweight": "normal",
        "axes.labelcolor": PALETTE["ink"], "text.color": PALETTE["ink"],
        "axes.edgecolor": PALETTE["subtle"], "axes.linewidth": 1.0,
        "xtick.color": PALETTE["ink"], "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "axes.grid": True, "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8, "grid.alpha": 0.9, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.edgecolor": PALETTE["grid"], "legend.fontsize": 11.0,
        "legend.borderpad": 0.5, "lines.solid_capstyle": "round",
        "lines.antialiased": True, "patch.linewidth": 0.8,
        "mathtext.default": "regular",
    })


apply_style()


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def panel_label(ax, letter, dx=-0.02, dy=1.06, fs=13):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=fs,
            fontweight="bold", va="top", ha="right", color=PALETTE["ink"])


def style_legend(ax, loc="best", ncol=1, title=None, fontsize=8.8, **kw):
    leg = ax.legend(loc=loc, ncol=ncol, title=title, fontsize=fontsize,
                    fancybox=True, **kw)
    if leg and title:
        leg.get_title().set_fontsize(8.6)
        leg.get_title().set_fontweight("bold")
    if leg:
        leg.get_frame().set_linewidth(0.8)
    return leg


def event_marker(ax, x, text, color=None, y=0.98, align="left", lw=1.4,
                 ls="--", fs=8.4, box=True):
    color = color or PALETTE["attack"]
    ax.axvline(x, color=color, lw=lw, ls=ls, alpha=0.9, zorder=1)
    ha = "left" if align == "left" else "right"
    xoff = 4 if align == "left" else -4
    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.9,
                alpha=0.95) if box else None
    ax.annotate(text, xy=(x, y), xycoords=("data", "axes fraction"),
                xytext=(xoff, 0), textcoords="offset points",
                ha=ha, va="top", fontsize=fs, color=color, fontweight="bold",
                bbox=bbox, zorder=6)


def shade_phase(ax, t0, t1, color=None, alpha=0.10, label=None):
    color = color or PALETTE["attack"]
    ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0, zorder=0, label=label)


def hazard_corridor(ax, half_width, orientation="h", extent=600,
                    label="Channel edge"):
    """Shade the un-navigable region beyond +/- half_width."""
    c = PALETTE["hazard"]
    if orientation == "h":
        ax.axhspan(half_width, half_width + extent, color=c, alpha=0.10, lw=0)
        ax.axhspan(-half_width - extent, -half_width, color=c, alpha=0.10, lw=0)
        ax.axhline(half_width, color=c, lw=1.3, alpha=0.85)
        ax.axhline(-half_width, color=c, lw=1.3, alpha=0.85, label=label)
    else:
        ax.axvspan(half_width, half_width + extent, color=c, alpha=0.10, lw=0)
        ax.axvspan(-half_width - extent, -half_width, color=c, alpha=0.10, lw=0)
        ax.axvline(half_width, color=c, lw=1.3, alpha=0.85)
        ax.axvline(-half_width, color=c, lw=1.3, alpha=0.85, label=label)


def time_colored_line(ax, x, y, t, cmap=CMAP_TIME, lw=2.4, zorder=3,
                      alpha=1.0):
    """Draw a path whose colour encodes time (LineCollection)."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segs, cmap=cmap, norm=norm, lw=lw, zorder=zorder,
                        alpha=alpha, capstyle="round")
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


def finalize(fig, fname=None, pad=0.3):
    if fname:
        fig.savefig(fname, bbox_inches="tight", pad_inches=pad)
        plt.close(fig)
    return fig
