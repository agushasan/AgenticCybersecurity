"""
sparse_observability.py
=======================
Determines the *security margin* of a sensor suite for secure state estimation,
following the observability conditions of Shoukry et al. (2018), Def. 2.2:

  * A system is s-sparse observable if (A, C_Gamma) is observable for every
    sensor subset Gamma with |Gamma| = p - s  (i.e. after removing ANY s sensors).

  * For secure state estimation under up to s attacked sensors, 2s-sparse
    observability is necessary and sufficient (Fawzi 2014; Shoukry-Tabuada 2016;
    Chong 2014).  Hence the largest number of simultaneously attackable sensors
    the suite can provably tolerate is

        s_max = floor( smax_obs / 2 ),   where smax_obs = max s : s-sparse obs.

This module also returns the 2s-restricted eigenvalue delta_2s
(Proposition 6.1), which lower-bounds lambda_min(O_Gamma^T O_Gamma) over all
|Gamma| >= p - 2s and sets the robustness of the estimate to noise.

Crucially for the maritime GNSS-spoofing threat, we also compute the margin
under a *common-mode* constraint: a GNSS spoofer captures all GNSS-derived
channels at once, so the relevant adversary removes whole DEVICE groups, not
arbitrary scalar channels.  We therefore compute both:
    - s_max_channels : tolerance to arbitrary scalar-channel attacks
    - device-level recoverability of specific common-mode attacks
"""

from __future__ import annotations
import itertools
import numpy as np
from numpy.linalg import matrix_rank, eigvalsh


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])


def is_observable(A: np.ndarray, C_rows: np.ndarray, tol: float | None = None) -> bool:
    """C_rows: (q x n). Observable iff observability matrix has full column rank n."""
    n = A.shape[0]
    if C_rows.ndim == 1:
        C_rows = C_rows[None, :]
    if C_rows.shape[0] == 0:
        return False
    O = observability_matrix(A, C_rows)
    return matrix_rank(O, tol=tol) == n


def max_sparse_observability(A: np.ndarray, C: np.ndarray, s_cap: int | None = None):
    """
    Return the largest integer smax such that the system is smax-sparse observable,
    i.e. observable after removing ANY smax sensors.  Brute force over subsets.
    """
    p = C.shape[0]
    if s_cap is None:
        s_cap = p - 1
    smax = 0
    for s in range(1, s_cap + 1):
        keep = p - s
        ok = True
        for Gamma in itertools.combinations(range(p), keep):
            if not is_observable(A, C[list(Gamma)]):
                ok = False
                break
        if ok:
            smax = s
        else:
            break
    return smax


def restricted_eigenvalue(A: np.ndarray, C: np.ndarray, s: int,
                          tau: int | None = None) -> float:
    """
    delta_2s = min over |Gamma| >= p - 2s of lambda_min(O_Gamma^T O_Gamma),
    using the finite-window (tau-step) observability blocks O_i (Sec. 2.4).
    This is the 2s-restricted eigenvalue (Prop. 6.1) governing noise robustness.
    """
    n = A.shape[0]; p = C.shape[0]
    if tau is None:
        tau = n
    # per-sensor tau-step observability block O_i (tau x n)
    Oi = [np.vstack([C[i][None, :] @ np.linalg.matrix_power(A, k)
                     for k in range(tau)]) for i in range(p)]
    keep = max(p - 2 * s, 1)
    best = np.inf
    for Gamma in itertools.combinations(range(p), keep):
        OG = np.vstack([Oi[i] for i in Gamma])
        lam = eigvalsh(OG.T @ OG)[0]
        best = min(best, lam)
    return float(best)


def analyze_suite(model, verbose: bool = True):
    """Full security-margin analysis of a ShipModel-like object."""
    A, C = model.A, model.C
    p = C.shape[0]
    smax_obs = max_sparse_observability(A, C)
    s_max = smax_obs // 2
    delta_2s = restricted_eigenvalue(A, C, s_max) if s_max >= 1 else np.nan

    # Which single-channel removals break observability (fragile directions)?
    fragile = []
    for i in range(p):
        keep = [j for j in range(p) if j != i]
        if not is_observable(A, C[keep]):
            fragile.append(i)

    # Device-level common-mode analysis: if an adversary removes a whole device
    # group (e.g. all GNSS), is the remaining suite still observable, and how
    # many *channels* does that correspond to?
    devices = {}
    for i, d in enumerate(model.devices):
        devices.setdefault(d, []).append(i)
    device_report = {}
    for d, idxs in devices.items():
        keep = [j for j in range(p) if j not in idxs]
        device_report[d] = {
            "channels": idxs,
            "n_channels": len(idxs),
            "obs_after_removal": is_observable(A, C[keep]) if keep else False,
        }

    # Common-mode GNSS: remove ALL gnss channels at once
    gnss_idx = [i for i in range(p) if model.gnss_mask[i]]
    keep = [j for j in range(p) if j not in gnss_idx]
    gnss_common = {
        "channels": gnss_idx,
        "n_channels": len(gnss_idx),
        "obs_after_removal": is_observable(A, C[keep]) if keep else False,
    }

    res = dict(p=p, n=A.shape[0], smax_obs=smax_obs, s_max=s_max,
               delta_2s=delta_2s, fragile_channels=fragile,
               device_report=device_report, gnss_common=gnss_common)

    if verbose:
        print("=" * 68)
        print("SPARSE-OBSERVABILITY SECURITY ANALYSIS")
        print("=" * 68)
        print(f"  sensors p = {p},  states n = {A.shape[0]}")
        print(f"  max s-sparse observability   smax_obs = {smax_obs}")
        print(f"  => provably recoverable attacks  s_max = floor(smax_obs/2) "
              f"= {s_max}")
        print(f"  2s-restricted eigenvalue  delta_2s = {delta_2s:.4g}")
        if fragile:
            print(f"  FRAGILE single channels (removal breaks obs.): "
                  f"{[model.names[i] for i in fragile]}")
        else:
            print("  No single-channel removal breaks observability "
                  "(1-sparse observable).")
        print("-" * 68)
        print("  Device-group (common-mode) removal:")
        for d, r in device_report.items():
            print(f"    {d:7s}: {r['n_channels']} ch -> remaining suite "
                  f"observable = {r['obs_after_removal']}")
        print(f"  ALL-GNSS common-mode spoof ({gnss_common['n_channels']} "
              f"channels): remaining suite observable = "
              f"{gnss_common['obs_after_removal']}")
        print("=" * 68)
    return res


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ship_model import build_ship_model
    sm = build_ship_model()
    analyze_suite(sm)
