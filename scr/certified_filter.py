"""
certified_filter.py  —  Exact implementation of the filter of Theorem 1 (version 2)
===================================================================================
Instantiates every object that Theorem 1 requires for the lateral subsystem
xi = (dE, dpsi, dv, dr) of the car, and implements the admissibility predicate (16)
as a sound support-function test.

Threat model of the certificate.  Attacks are confined to ONE device group at a
time, either GNSS (all four GNSS channels) or the lane camera (offset and heading).
The filter runs one predictor per group-free sensor set ("GNSS-free",
"camera-free"), each driven only by its own channels from k = 0, and uses the
predictor selected by the current threat hypothesis (default: GNSS-free).  A
predictor never consumes the excluded group, so an attack on that group can never
contaminate it: Assumption 2 holds at every step, including isolation transients,
whenever the hypothesis names the attacked group.

Offline design (CertifiedDesign):
  * Assumption 1: |nu_i| <= nsig sigma_i, |mu_j| <= nsig sigma_mu,j (the simulator
    clips its Gaussian draws at the same level).
  * Assumption 4 / initialization: each predictor starts from the weighted
    least-squares fit to the first measurement, so its initial error lies in the
    box |M_T| nsig sigma_T; the initial lateral state lies in the declared box X0.
  * Assumption 2 / Proposition 2: E_T = cl conv U_{k>=0} R_k^T (reachable error set
    from the initialization box), RPI by the argument of Lemma 1, outer-bounded by
    a box from exact zonotope supports plus a tail bound; E = E_T of the set in use.
  * Validity region X_valid: rear slip |alpha_r| <= alpha_max, front slip
    |alpha_f| <= alpha_max (input dependent), heading |dpsi| <= psi_max.
  * Assumption 3: backup u_b = -K_b xhat; on Omega (+) (-E) it satisfies
    |u_b| <= u_max and |alpha_f(xi, u_b)| <= alpha_max, enforced as
        |K_b xi| <= u_max - s_E,   |(K_b + c_f) xi| <= alpha_max - s_E,
    with s_E = max_T h_{E_T}(K_b) and alpha_f = u - c_f xi.
  * Omega: maximal RPI polytope of xi+ = A_cl xi + d, d in D = B K_b E (+) W, inside
    S = {|e| <= e_cert} and the validity and input constraints, computed by the
    Kolmanovsky-Gilbert recursion with LP redundancy removal and a strict margin.
  * X0 subset Omega is checked on the vertices of the box X0.
  * Lemma 1 hull Omega_R (reachable set of the backup loop from X0) is reported as
    the existence certificate (Omega_R subset Omega).

Runtime (CertifiedFilter):
  phi(xhat, u) = |u| <= u_max  and  |u - c_f xhat| <= alpha_max - h_E(c_f)  and
                 G_i (A xhat + B u) + h_{A E}(G_i) + h_W(G_i) <= g_i for all facets i.
  u = u_agent if phi holds, else u = -K_b xhat_T.
"""
from __future__ import annotations
import itertools
import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog

IX = [1, 2, 4, 5]
MARGIN = 1e-8                       # strict margin on every constraint of Omega
D_INFLATE = 1e-6                    # relative inflation of D in the recursion


def _ninf(M):
    return float(np.max(np.sum(np.abs(M), axis=1)))


def _tail_constants(F):
    norms = []; Aj = np.eye(F.shape[0])
    for _ in range(200000):
        norms.append(_ninf(Aj)); Aj = Aj @ F
        if norms[-1] < 1e-16:
            break
    return max(norms), float(np.sum(norms))


def predictor_kalman_gain(A, C, sigma, wsig, qs):
    n = A.shape[0]
    R = np.diag(sigma ** 2); Q = np.diag((qs * wsig) ** 2) + 1e-12 * np.eye(n)
    P = solve_discrete_are(A.T, C.T, Q, R)
    return A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + R)


def reach_box(F, Gd, e0_half, K=1500):
    """Half-widths of cl conv U_{k>=0} R_k, R_k = F^k E0 (+) sum_{j<k} F^j <0,Gd>, with tail bound."""
    n = F.shape[0]; acc = np.zeros(n); Fk = np.eye(n); Md = Gd.copy(); best = np.zeros(n)
    for _ in range(K + 1):
        best = np.maximum(best, np.abs(Fk) @ e0_half + acc)
        acc = acc + np.abs(Md).sum(axis=1); Md = F @ Md; Fk = F @ Fk
    M, S = _tail_constants(F)
    tail = _ninf(np.linalg.matrix_power(F, K)) * (M * np.max(e0_half) + S * _ninf(Gd))
    return best + tail, tail


@dataclass
class CertifiedDesign:
    model: object
    Kb: np.ndarray                         # (4,) backup steering gain on xi
    e_cert: float                          # certified lateral bound [m]
    u_max: float                           # steering actuator limit [rad]
    sets: dict                             # name -> trusted channel indices
    alpha_max: float = np.deg2rad(4.0)     # linear-tyre slip limit [rad]
    psi_max: float = np.deg2rad(5.0)       # small-angle heading limit [rad]
    x0_half: np.ndarray = None             # half-widths of the initial lateral box X0
    nsig: float = 3.0
    proc_sigma: float = 0.004
    qs: float = 30.0
    build_omega: bool = True
    L: dict = field(default_factory=dict)
    M0: dict = field(default_factory=dict)
    Ebox: dict = field(default_factory=dict)

    def __post_init__(self):
        m = self.model; A, B, C = m.A, m.B, m.C; p = m.p
        ps = self.proc_sigma
        self.wsig = np.array([ps * 2, ps * 2, ps * 0.2, ps, ps, ps * 0.2])
        self.w_half = self.nsig * self.wsig
        # small-angle linearization error of dE' = U sin(dpsi) + dv over |dpsi| <= psi_max, added to W (Assumption 1)
        self.lin_err = p.U * (self.psi_max - np.sin(self.psi_max)) * p.dt
        self.w_half = self.w_half + np.array([0.0, self.lin_err, 0, 0, 0, 0])
        self.v_half = self.nsig * m.sigma
        if self.x0_half is None:
            self.x0_half = np.array([0.3, np.deg2rad(0.5), 0.05, np.deg2rad(0.5)])
        self.Kb = np.asarray(self.Kb, float).ravel()
        self.Axi = A[np.ix_(IX, IX)]; self.Bxi = B[IX, 1]
        self.Acl = self.Axi - np.outer(self.Bxi, self.Kb)
        self.rho = float(np.max(np.abs(np.linalg.eigvals(self.Acl))))
        if self.rho >= 1.0:
            raise ValueError("backup not Schur")
        self.U = p.U
        self.cf = np.array([0.0, 0.0, 1.0 / p.U, p.lf / p.U])      # alpha_f = u - cf xi
        self.cr = np.array([0.0, 0.0, -1.0 / p.U, p.lr / p.U])     # alpha_r = cr xi
        self.tails, self.e0box = {}, {}
        for name, T in self.sets.items():
            CT, RT = C[T], np.diag(1.0 / m.sigma[T] ** 2)
            M0 = np.linalg.solve(CT.T @ RT @ CT, CT.T @ RT)          # first-sample WLS initializer
            e0 = np.abs(M0) @ self.v_half[T]
            L = predictor_kalman_gain(A, CT, m.sigma[T], self.wsig, self.qs)
            F = A - L @ CT
            Gd = np.hstack([np.diag(self.w_half), -L @ np.diag(self.v_half[T])])
            box, tail = reach_box(F, Gd, e0)
            self.L[name], self.M0[name], self.Ebox[name] = L, M0, box
            self.tails[name], self.e0box[name] = tail, e0
        self.Exi = {k: v[IX] for k, v in self.Ebox.items()}
        self.s_E = max(float(np.abs(self.Kb) @ e) for e in self.Exi.values())
        self.s_cf = max(float(np.abs(self.cf) @ e) for e in self.Exi.values())
        self.Wxi = self.w_half[IX]
        if self.s_E >= self.u_max or self.s_E >= self.alpha_max or self.s_cf >= self.alpha_max:
            raise ValueError("error set consumes the input or slip range")
        H = np.array([1.0, 0, 0, 0]); P = np.array([0, 1.0, 0, 0])
        rows = [H, P, self.cr, self.Kb, self.Kb + self.cf]
        bnds = [self.e_cert, self.psi_max, self.alpha_max, self.u_max - self.s_E, self.alpha_max - self.s_E]
        self.G0 = np.vstack([r for r in rows for _ in (0,)] + [-r for r in rows])
        self.g0 = np.array(bnds + bnds) - MARGIN
        self.constraint_names = ["e", "psi", "alpha_r", "u_backup", "alpha_f_backup"]
        self.lemma1 = self.lemma1_bound()
        if self.build_omega:
            self._build_omega()

    # ---- support functions --------------------------------------------------
    def h_D(self, c):
        return abs(c @ self.Bxi) * self.s_E + np.abs(c) @ self.Wxi

    def h_AE(self, c):
        ca = np.abs(c @ self.Axi)
        return max(float(ca @ e) for e in self.Exi.values())

    # ---- Lemma 1: reachable hull of the backup loop from X0 ---------------
    def lemma1_bound(self, K=600):
        """Supports of Omega_R = cl conv U_k (A_cl^k X0 (+) sum_{j<k} A_cl^j D) along every constraint row."""
        GD = np.hstack([self.Bxi[:, None] * self.s_E, np.diag(self.Wxi)])
        G0 = np.vstack([np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), self.cr, self.Kb, self.Kb + self.cf])
        X0 = np.diag(self.x0_half)
        sup = np.zeros(len(G0)); rad = np.zeros(len(G0)); Ak = np.eye(4); M = GD.copy()
        for k in range(K + 1):
            sup = np.maximum(sup, np.abs(G0 @ Ak @ X0).sum(axis=1) + rad)
            rad = rad + np.abs(G0 @ M).sum(axis=1); M = self.Acl @ M; Ak = self.Acl @ Ak
        Mx, S = _tail_constants(self.Acl)
        eps = _ninf(np.linalg.matrix_power(self.Acl, K)) * (Mx * np.max(self.x0_half) + S * _ninf(GD))
        sup = np.maximum(sup, rad + eps * np.abs(G0).sum(axis=1))
        return dict(zip(self.constraint_names, sup.tolist()), tail=eps, K=K)

    # ---- maximal RPI set (Kolmanovsky-Gilbert) -----------------------------
    def _build_omega(self, kmax=600):
        G0, g0 = self.G0, self.g0
        G, g = [G0], [g0]; Ak = np.eye(4); shrink = np.zeros(len(g0))
        self.k_star = None
        for k in range(1, kmax + 1):
            for i in range(len(g0)):
                shrink[i] += (1.0 + D_INFLATE) * self.h_D(G0[i] @ Ak) + 1e-9
            Ak = self.Acl @ Ak
            Gk = G0 @ Ak; gk = g0 - shrink
            if np.any(gk < 0):
                raise ValueError("maximal RPI set is empty")
            Gc = np.vstack(G); gc = np.concatenate(g)
            redundant = True
            for i in range(len(gk)):
                r = linprog(-Gk[i], A_ub=Gc, b_ub=gc, bounds=[(None, None)] * 4, method="highs")
                if r.status != 0 or -r.fun > gk[i] + 1e-12:
                    redundant = False; break
            if redundant:
                self.k_star = k - 1; break
            G.append(Gk); g.append(gk)
        if self.k_star is None:
            raise RuntimeError("not finitely determined within kmax")
        Gc = np.vstack(G); gc = np.concatenate(g); keep = []
        for i in range(len(gc)):
            others = [j for j in range(len(gc)) if j != i and (j in keep or j > i)]
            r = linprog(-Gc[i], A_ub=Gc[others], b_ub=gc[others], bounds=[(None, None)] * 4, method="highs")
            if r.status != 0 or -r.fun > gc[i] + 1e-9:
                keep.append(i)
        self.OG, self.Og = Gc[keep], gc[keep]
        self.rpi_margin = min(self.Og[i] - self.h_D(self.OG[i]) - self._support(self.OG[i] @ self.Acl)
                              for i in range(len(self.Og)))
        self.omega_support = {nm: self._support(r) for nm, r in zip(self.constraint_names, G0[:5])}
        # constant right-hand sides of (16): g_i - h_{A E}(G_i) - h_W(G_i)
        self.rhs = np.array([gi - self.h_AE(Gi) - np.abs(Gi) @ self.Wxi for Gi, gi in zip(self.OG, self.Og)])
        self.OGA, self.OGB = self.OG @ self.Axi, self.OG @ self.Bxi
        corners = [np.array(c) * self.x0_half for c in itertools.product((-1, 1), repeat=4)]
        self.x0_margin = min(float(np.min(self.Og - self.OG @ c)) for c in corners)
        self.x0_inside = self.x0_margin > 0

    def _support(self, c):
        r = linprog(-c, A_ub=self.OG, b_ub=self.Og, bounds=[(None, None)] * 4, method="highs")
        return -r.fun

    def contains(self, xi, tol=0.0):
        return bool(np.all(self.OG @ xi <= self.Og + tol))

    def in_valid(self, xi, u):
        return (abs(xi[1]) <= self.psi_max and abs(self.cr @ xi) <= self.alpha_max
                and abs(u - self.cf @ xi) <= self.alpha_max)

    # ---- predicate (16) ----------------------------------------------------
    def phi(self, xhat_xi, u):
        if abs(u) > self.u_max or abs(u - self.cf @ xhat_xi) > self.alpha_max - self.s_cf:
            return False
        return bool(np.all(self.OGA @ xhat_xi + self.OGB * u <= self.rhs))

    def phi_batch(self, xhat_xi, us):
        """Vectorized predicate over an array of candidate inputs."""
        us = np.asarray(us, float)
        ok = (np.abs(us) <= self.u_max) & (np.abs(us - self.cf @ xhat_xi) <= self.alpha_max - self.s_cf)
        lhs = (self.OGA @ xhat_xi)[:, None] + np.outer(self.OGB, us)
        return ok & np.all(lhs <= self.rhs[:, None], axis=0)

    def export(self):
        return dict(Kb=self.Kb.tolist(), e_cert=self.e_cert, u_max_rad=self.u_max, alpha_max_rad=self.alpha_max,
                    psi_max_rad=self.psi_max, nsig=self.nsig, process_sigma=self.wsig.tolist(), linearization_error=self.lin_err,
                    measurement_sigma=self.model.sigma.tolist(), qs=self.qs, x0_half=self.x0_half.tolist(),
                    sets={k: list(map(int, v)) for k, v in self.sets.items()},
                    observer_gain={k: v.tolist() for k, v in self.L.items()},
                    init_wls={k: v.tolist() for k, v in self.M0.items()},
                    init_error_half={k: v.tolist() for k, v in self.e0box.items()},
                    error_half={k: v.tolist() for k, v in self.Ebox.items()},
                    error_tail={k: float(v) for k, v in self.tails.items()},
                    s_E=self.s_E, s_cf=self.s_cf, rho=self.rho, Axi=self.Axi.tolist(), Bxi=self.Bxi.tolist(),
                    W_half=self.Wxi.tolist(), omega_G=self.OG.tolist(), omega_g=self.Og.tolist(),
                    k_star=self.k_star, rpi_margin=self.rpi_margin, x0_margin=self.x0_margin,
                    omega_support=self.omega_support, lemma1=self.lemma1,
                    tolerances=dict(constraint_margin=MARGIN, D_inflation=D_INFLATE, lp="HiGHS default"))


class CertifiedFilter:
    """Observer bank plus the filter (15)-(16) at run time; the set is chosen by the threat hypothesis."""

    def __init__(self, design: CertifiedDesign, default="GNSS-free"):
        self.d = design; m = design.model
        self.C = m.C; self.A = m.A; self.B = m.B
        self.xh = None
        self.active = default
        self.n_admit = self.n_backup = 0

    def initialize(self, y0):
        self.xh = {k: self.d.M0[k] @ y0[T] for k, T in self.d.sets.items()}

    def set_hypothesis(self, name):
        self.active = name

    def filter(self, attacked_mask, u_agent_steer):
        name = self.active; xi = self.xh[name][IX]
        if self.d.phi(xi, u_agent_steer):
            self.n_admit += 1
            return float(u_agent_steer), 0, name, "hyp"
        self.n_backup += 1
        return float(-self.d.Kb @ xi), 1, name, "hyp"

    def update(self, u, y):
        for k, T in self.d.sets.items():
            L = self.d.L[k]
            self.xh[k] = self.A @ self.xh[k] + self.B @ u + L @ (y[T] - self.C[T] @ self.xh[k])
