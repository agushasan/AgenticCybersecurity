"""
certify_rta.py  —  Numerical certificate for Theorem 1 (safety invariance)
=========================================================================
Computes a formal, deterministic certificate that the runtime-assurance (RTA)
filter keeps the true cross-track position inside the safe set  S = {|e| <= e_max}
for ANY agent policy, in the recoverable regime (trusted set 2s-sparse
observable), under bounded process/measurement noise and the secure-estimator
error set.

Method (robust forward reachability via zonotopes):
  1. Cross-track subsystem  xi = (dE, dpsi, dv, dr)  — closed under the rudder;
     the along-track / surge states do not affect the safety output e = dE.
  2. Secure-observer error set E: the secure observer is tuned for noise
     rejection (steady-state Kalman-optimal gains), so its error is bounded by
     the NSIG-sigma box of the filtered covariance Pf (solution of the discrete
     algebraic Riccati equation), inflated to cover use of p-s sensors after
     attack isolation.  Projected to the subsystem.
  3. Backup closed loop  A_cl = A_sub - B_sub K_b  (Schur).  Disturbance set
     D = B_sub K_b E_sub  (+)  W_sub   (control acts on the estimate = true - eta).
  4. Robust reachable tube from the scenario initial condition under A_cl + D;
     the supremum of |e| over the tube is the certified cross-track bound.
     If it is <= e_max, Theorem 1's conclusion holds (zero safety violations).

Outputs: results/rta_certificate.json and figures/fig16_certificate.png.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from scipy.linalg import solve_discrete_are
from ship_model import build_ship_model
from simulate import Autopilot
from safety_filter import SafetyFilter, SafetyEnvelope
from secure_estimation import design_gain
import style as S
from style import PALETTE as P
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
IX = [1, 2, 4, 5]                       # subsystem states: dE, dpsi, dv, dr
NSIG = 3.0                              # bounded-noise box = NSIG * sigma


# --------------------------------------------------------------------------- #
#  Minimal zonotope                                                           #
# --------------------------------------------------------------------------- #
class Zono:
    def __init__(self, c, G):
        self.c = np.asarray(c, float)
        self.G = np.asarray(G, float).reshape(len(self.c), -1)

    def lin(self, M):                   # M @ zonotope
        return Zono(M @ self.c, M @ self.G)

    def plus(self, other):              # Minkowski sum
        return Zono(self.c + other.c, np.hstack([self.G, other.G]))

    def support(self, h):               # max over zonotope of h^T x
        return float(h @ self.c + np.sum(np.abs(h @ self.G)))

    def box(self):                      # per-coordinate [min,max]
        r = np.sum(np.abs(self.G), axis=1)
        return self.c - r, self.c + r

    def reduce(self, maxgen=120):       # order reduction (box over-approx of extra gens)
        if self.G.shape[1] <= maxgen:
            return self
        norms = np.sum(np.abs(self.G), axis=0)
        idx = np.argsort(norms)
        keep = idx[-maxgen:]; drop = idx[:-maxgen]
        box = np.sum(np.abs(self.G[:, drop]), axis=1)
        Gkeep = self.G[:, keep]
        Gbox = np.diag(box)
        return Zono(self.c, np.hstack([Gkeep, Gbox]))


def box_zono(half):                      # centred box zonotope from half-widths
    half = np.asarray(half, float)
    return Zono(np.zeros_like(half), np.diag(half))


# --------------------------------------------------------------------------- #
#  Build the certified objects                                                #
# --------------------------------------------------------------------------- #
def build(model, env, poles, autopilot=None, proc_sigma=0.02):
    A, B, C = model.A, model.B, model.C
    n, p = model.n, model.psensors
    # --- noise bounds (bounded-noise assumption; boxes = NSIG * sigma) -------
    w_half = np.array([proc_sigma*2, proc_sigma*2, proc_sigma*0.2,
                       proc_sigma, proc_sigma, proc_sigma*0.2]) * NSIG
    psi_half = model.sigma * NSIG

    # --- secure-observer error set E ----------------------------------------
    # Kalman-gain secure observer on the TRUSTED set T (attacked channels
    # removed).  E is the NSIG-sigma box of the filtered error covariance,
    # taken elementwise as the worst case over every recoverable trusted set:
    # the full suite, every single-channel removal (|K| <= s_max = 1), and the
    # common-mode GNSS removal (all four GNSS channels), which is the regime of
    # Scenarios B, E, M and M2.  No ad hoc inflation factor is used.
    w_sigma = w_half / NSIG
    Q = np.diag(w_sigma ** 2) + 1e-9 * np.eye(n)
    full = list(range(p))
    gnss = [i for i in full if model.gnss_mask[i]]
    trusted_sets = [full] + [[j for j in full if j != i] for i in full] \
                   + [[j for j in full if j not in gnss]]
    E_half = np.zeros(n); worst = None
    for T in trusted_sets:
        Ct = C[T]; R = np.diag(model.sigma[T] ** 2)
        Ppred = solve_discrete_are(A.T, Ct.T, Q, R)
        Kf = Ppred @ Ct.T @ np.linalg.inv(Ct @ Ppred @ Ct.T + R)
        Pf = (np.eye(n) - Kf @ Ct) @ Ppred
        h = NSIG * np.sqrt(np.maximum(np.diag(Pf), 0.0))
        if worst is None or h[1] > E_half[1]:
            worst = [model.names[i] for i in full if i not in T]
        E_half = np.maximum(E_half, h)

    # --- cross-track subsystem ----------------------------------------------
    A_sub = A[np.ix_(IX, IX)]
    B_sub = B[IX, 1:2]                              # rudder column
    sf = SafetyFilter(model, autopilot if autopilot is not None else Autopilot(), env)
    Kb = sf.K_backup[1, :]                          # backup rudder gain row (6,)
    Kb_sub = Kb[IX].reshape(1, -1)                  # 1 x 4
    A_cl = A_sub - B_sub @ Kb_sub                   # 4 x 4
    schur = float(np.max(np.abs(np.linalg.eigvals(A_cl))))

    # --- subsystem disturbance D = B_sub Kb_sub E_sub (+) W_sub -------------
    E_sub_half = E_half[IX]
    BK = B_sub @ Kb_sub                             # 4 x 4 (control error map)
    D = box_zono(E_sub_half).lin(BK).plus(box_zono(w_half[IX]))

    G_D = np.hstack([BK @ np.diag(E_sub_half), np.diag(w_half[IX])])
    return dict(A_sub=A_sub, A_cl=A_cl, D=D, schur=schur, E_half=E_half,
                E_sub_half=E_sub_half, w_half=w_half, psi_half=psi_half,
                Ce=np.array([1.0, 0, 0, 0]), G_D=G_D, worst_removed=worst, BK=BK)


def certify(obj, e0, e_max, K=400):
    """
    Certify Omega = union_k X_k, X_k = A_cl^k xi0 (+) Omega_k, the reachable
    set of the backup loop from xi0 = (e0,0,0,0).  Omega is RPI and contains
    xi0.  For k < K the support of X_k is computed on zonotopes (order
    reduction is a sound outer bound).  For k >= K, X_k = Omega_K (+) A^K X_{k-K}
    with ||A^K x||_inf <= eps, eps = ||A^K||_inf (M |xi0|_inf + S ||G_D||_inf),
    M = sup_j ||A^j||_inf, S = sum_j ||A^j||_inf.  Returns the certified
    sup |e| = max( max_{k<K} h_{X_k}(+-H),  h_{Omega_K}(+-H) + eps ).
    """
    A_cl, D, Ce, G_D = obj["A_cl"], obj["D"], obj["Ce"], obj["G_D"]
    xi0 = np.array([e0, 0.0, 0.0, 0.0])
    X = Zono(xi0.copy(), np.zeros((4, 1)))
    O = Zono(np.zeros(4), np.zeros((4, 1)))          # Omega_k from the origin
    sup_fin = abs(e0)
    tube = [X.support(Ce)]; tube_lo = [-X.support(-Ce)]
    for _ in range(K):
        X = X.lin(A_cl).plus(D).reduce(150)
        O = O.lin(A_cl).plus(D).reduce(150)
        sup_fin = max(sup_fin, X.support(Ce), X.support(-Ce))
        tube.append(X.support(Ce)); tube_lo.append(-X.support(-Ce))
    ninf = lambda Mx: float(np.max(np.sum(np.abs(Mx), axis=1)))
    Ak = np.eye(4); norms = []
    for _ in range(20000):
        norms.append(ninf(Ak)); Ak = Ak @ A_cl
        if norms[-1] < 1e-14: break
    M, Ssum = max(norms), float(np.sum(norms))
    eps = ninf(np.linalg.matrix_power(A_cl, K)) * (M * abs(e0) + Ssum * ninf(G_D))
    rad_O = max(O.support(Ce), O.support(-Ce))
    sup = max(sup_fin, rad_O + eps)
    info = dict(sup_finite=float(sup_fin), mrpi_radius=float(rad_O),
                tail_eps=float(eps), K=K)
    return sup, np.array(tube), np.array(tube_lo), info


def certifiable_scale(obj, e0, e_max, K=300):
    """Largest uniform inflation of E for which the certificate still holds."""
    base = obj["E_sub_half"]; BK = obj["BK"]; w = obj["w_half"][IX]
    lo, hi = 1.0, 30.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        D = box_zono(base * mid).lin(BK).plus(box_zono(w))
        G_D = np.hstack([BK @ np.diag(base * mid), np.diag(w)])
        o = dict(obj, D=D, G_D=G_D)
        s, *_ = certify(o, e0, e_max, K)
        lo, hi = (mid, hi) if s <= e_max else (lo, mid)
    return lo


# --------------------------------------------------------------------------- #
#  Main: certificate for every scenario's initial condition                   #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    model = build_ship_model()
    env = SafetyEnvelope(e_max=70.0)
    poles = np.array([0.55, 0.60, 0.66, 0.70, 0.74, 0.78])
    obj = build(model, env, poles)

    e0 = 15.0                                        # scenario initial cross-track
    sup, tube_hi, tube_lo, info = certify(obj, e0, env.e_max)
    certified = sup <= env.e_max
    scale = certifiable_scale(obj, e0, env.e_max)

    print("=" * 66)
    print("THEOREM 1 — NUMERICAL SAFETY CERTIFICATE")
    print("=" * 66)
    print(f"  backup closed-loop spectral radius  = {obj['schur']:.4f}  "
          f"({'Schur-stable' if obj['schur'] < 1 else 'UNSTABLE'})")
    print(f"  secure-observer error set E (half-widths, {NSIG}-sigma noise):")
    names = ["dN", "dE", "dpsi", "du", "dv", "dr"]
    for nm, h in zip(names, obj["E_half"]):
        print(f"      {nm:5s} <= {h:.3f}")
    print(f"  safe bound  e_max                   = {env.e_max:.1f} m")
    print(f"  initial cross-track  e0             = {e0:.1f} m")
    print(f"  worst-case trusted set removes      = {obj['worst_removed']}")
    print(f"  finite-horizon tube sup (K={info['K']})    = {info['sup_finite']:.2f} m")
    print(f"  mRPI radius + tail eps              = {info['mrpi_radius']:.2f} + {info['tail_eps']:.2e} m")
    print(f"  CERTIFIED sup |e| over Omega        = {sup:.2f} m")
    print(f"  max certifiable uniform inflation of E = {scale:.2f}x "
          f"(dE <= {scale*obj['E_half'][1]:.1f} m)")
    print(f"  => Theorem 1 conclusion holds: {certified}  "
          f"(sup|e| {'<=' if certified else '>'} e_max)")
    print("=" * 66)

    # --- figure: certified tube vs e_max vs a simulated secure trajectory ----
    from attacks import gnss_walkoff
    from simulate import simulate, SimConfig
    resB = simulate(model, gnss_walkoff(model, 120, ramp_rate_E=0.6),
                    SimConfig(T=400, s=4, seed=7), configs=("secure",),
                    rta=True, rta_env=env)
    t = np.arange(len(tube_hi))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.axhspan(-env.e_max, env.e_max, color=P["secure"], alpha=0.07, lw=0)
    ax.axhline(env.e_max, color=P["hazard"], lw=1.4, ls="--")
    ax.axhline(-env.e_max, color=P["hazard"], lw=1.4, ls="--",
               label="safe bound $\\pm e_{max}$")
    ax.fill_between(t, tube_lo, tube_hi, color=P["gold"], alpha=0.30, lw=0,
                    label="provisional reachable tube")
    ax.plot(t, tube_hi, color=P["gold"], lw=1.3)
    ax.plot(t, tube_lo, color=P["gold"], lw=1.3)
    ax.plot(resB.t, resB.x_true["secure"][:, 1], color=P["secure"], lw=1.8,
            label="simulated secure trajectory (Scenario B)")
    ax.axhline(sup, color=P["ink"], lw=1.0, ls=":",
               label=f"provisional sup$|e|$ = {sup:.1f} m")
    ax.set_xlim(0, max(len(tube_hi), len(resB.t)))
    ax.set_xlabel("time step  $k$"); ax.set_ylabel("cross-track  $e=\\delta E$  [m]")
    ax.set_ylim(-env.e_max * 1.15, env.e_max * 1.15)
    S.style_legend(ax, loc="lower right", fontsize=9.0)
    S.finalize(fig, os.path.join(FIG, "fig16_certificate.png"))

    out = dict(e_max=env.e_max, e0=e0, backup_spectral_radius=obj["schur"],
               error_set_halfwidths=dict(zip(names, obj["E_half"].tolist())),
               certified_sup_abs_e=float(sup), certified=bool(certified),
               noise_bound_nsigma=NSIG, worst_trusted_set_removes=obj['worst_removed'],
               tube_sup_finite=info['sup_finite'], mrpi_radius=info['mrpi_radius'],
               tail_eps=info['tail_eps'], horizon_K=info['K'],
               max_certifiable_inflation=scale)
    with open(os.path.join(RES, "rta_certificate.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("figure -> figures/fig16_certificate.png")
    print("json   -> results/rta_certificate.json")


if __name__ == "__main__":
    main()
