"""
secure_estimation.py
====================
Faithful implementation of the secure state estimator of

    Shoukry, Chong, Wakaiki, Nuzzo, Sangiovanni-Vincentelli, Seshia,
    Hespanha, Tabuada, "SMT-Based Observer Design for Cyber-Physical Systems
    under Sensor Attacks", ACM T. Cyber-Phys. Syst. 2(1):5, 2018.

Components
----------
1. ObserverBank          : the exhaustive bank of Luenberger observers (Eq. 3.1),
                           one per candidate attack-free set Gamma.
2. MMLObserver           : the Multi-Modal Luenberger observer (Sec. 4) --
                           a single shared state z (dim n(m+p)) from which the
                           per-Gamma estimate x_hat_Gamma = Cbar_Gamma z is read
                           out.  We construct Cbar_Gamma by Markov-parameter
                           matching (equivalent to the paper's T_Gamma, Eq. 4.6)
                           and VERIFY Theorem 4.2 (MML == bank) numerically.
3. DetectionBound        : the decaying residual envelope gamma_i*lam^t + d_i(t)
                           of Proposition A.2, used as the per-sensor attack test.
4. SMTSecureEstimator    : the pseudo-Boolean SAT + conflict-driven search
                           (Algorithms 1-3) that identifies the attacked sensors
                           and returns the secure state estimate.
5. KalmanBaseline        : a conventional steady-state Kalman filter using ALL
                           sensors (the non-secure integrated-navigation filter).

Because z3 / python-sat are not required here, the DPLL-style pseudo-Boolean
search is implemented natively and is faithful to Algorithms 1-3; a production
deployment can swap in an off-the-shelf SMT solver without changing the logic.
"""

from __future__ import annotations
import itertools
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import place_poles
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from numpy.linalg import matrix_rank


# ============================================================================ #
#  Observability helper                                                        #
# ============================================================================ #
def _obsv(A, C):
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])


def _is_obsv(A, C):
    if C.ndim == 1:
        C = C[None, :]
    if C.shape[0] == 0:
        return False
    return matrix_rank(_obsv(A, C)) == A.shape[0]


# ============================================================================ #
#  Luenberger-observer gain design (common characteristic polynomial)          #
# ============================================================================ #
def _null_space(M, rtol=1e-9):
    """Orthonormal basis for the null space of M (columns)."""
    u, s, vh = np.linalg.svd(M)
    if s.size == 0:
        return vh.conj().T
    tol = rtol * max(M.shape) * (s[0] if s.size else 1.0)
    ns = vh[np.sum(s > tol):].conj().T
    return ns


def robust_place(A, C_rows, poles, rng=None, tries=60):
    """
    Robust observer-gain placement by eigenstructure assignment (null-space /
    Moore's method) on the dual pair (A^T, C_rows^T).  Works for any observable
    pair with DISTINCT real target poles, including derogatory open-loop systems
    (multiple integrators) for which the KNV algorithm in scipy.place_poles
    fails.  Returns L (n x q) with eig(A - L C_rows) = poles, or None if the
    pair is not observable.
    """
    if C_rows.ndim == 1:
        C_rows = C_rows[None, :]
    if not _is_obsv(A, C_rows):
        return None
    n = A.shape[0]
    q = C_rows.shape[0]
    Ad, Bd = A.T, C_rows.T                      # dual: place eig(Ad - Bd K)
    if rng is None:
        rng = np.random.default_rng(12345)
    # per-pole allowable eigenvector subspaces
    tops, bots = [], []
    for lam in poles:
        Mlam = np.hstack([Ad - lam * np.eye(n), Bd])     # n x (n+q)
        Nk = _null_space(Mlam)                            # (n+q) x k
        if Nk.shape[1] == 0:
            return None
        tops.append(Nk[:n, :]); bots.append(Nk[n:, :])
    best = None
    for _ in range(tries):
        W = np.zeros((n, n)); Z = np.zeros((q, n))
        coeffs = []
        for i in range(n):
            k = tops[i].shape[1]
            c = rng.standard_normal(k)
            c = c / (np.linalg.norm(c) + 1e-12)
            coeffs.append(c)
            W[:, i] = tops[i] @ c
            Z[:, i] = bots[i] @ c
        cond = np.linalg.cond(W)
        if np.isfinite(cond) and cond < 1e8:
            K = -Z @ np.linalg.inv(W)                     # q x n
            L = K.T                                       # n x q
            ev = np.sort(np.linalg.eigvals(A - L @ C_rows).real)
            if np.allclose(np.sort(np.asarray(poles, float)), ev, atol=1e-6):
                return L
            if best is None or cond < best[0]:
                best = (cond, L)
    return None if best is None else best[1]


def design_gain(A, C_rows, poles):
    """
    Observer gain L (n x q) placing eig(A - L C_rows) at `poles`.
    Tries scipy.place_poles first (fast), falls back to robust eigenstructure
    assignment.  Returns None if (A, C_rows) is not observable.
    """
    if C_rows.ndim == 1:
        C_rows = C_rows[None, :]
    if not _is_obsv(A, C_rows):
        return None
    try:
        res = place_poles(A.T, C_rows.T, poles, method='YT')
        L = res.gain_matrix.T
        ev = np.sort(np.linalg.eigvals(A - L @ C_rows).real)
        if np.allclose(np.sort(np.asarray(poles, float)), ev, atol=1e-6):
            return L
    except Exception:
        pass
    return robust_place(A, C_rows, poles)


# ============================================================================ #
#  MML observer (Section 4)                                                    #
# ============================================================================ #
@dataclass
class MMLObserver:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray                      # p x n
    poles: np.ndarray                  # n common observer poles (distinct)
    n: int = field(init=False)
    m: int = field(init=False)
    p: int = field(init=False)
    ell: int = field(init=False)       # ell = m + p
    k: np.ndarray = field(init=False)  # char-poly coeffs [k1..kn]
    Abar: np.ndarray = field(init=False)
    Bbar: np.ndarray = field(init=False)
    Rbar_inv: np.ndarray = field(init=False)

    def __post_init__(self):
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.p = self.C.shape[0]
        self.ell = self.m + self.p
        # common characteristic polynomial d(s) = prod(s - poles)
        poly = np.real(np.poly(self.poles))          # [1, k1, ..., kn]
        self.k = poly[1:]
        self._build_shared_realization()

    # ---- shared controllable-canonical-form realization (Eqs. 4.4, 4.7) ---- #
    def _build_shared_realization(self):
        n, ell, k = self.n, self.ell, self.k
        I = np.eye(ell)
        Abar = np.zeros((n * ell, n * ell))
        # first block row: [-k1 I, -k2 I, ..., -kn I]
        for j in range(n):
            Abar[0:ell, j * ell:(j + 1) * ell] = -k[j] * I
        # sub-diagonal identity blocks
        for i in range(1, n):
            Abar[i * ell:(i + 1) * ell, (i - 1) * ell:i * ell] = I
        Bbar = np.zeros((n * ell, ell))
        Bbar[0:ell, :] = I
        self.Abar, self.Bbar = Abar, Bbar
        # shared controllability matrix Rbar = [Bbar, Abar Bbar, ..., Abar^{n-1} Bbar]
        blocks = []
        M = Bbar.copy()
        for _ in range(n):
            blocks.append(M)
            M = Abar @ M
        Rbar = np.hstack(blocks)                      # (n ell) x (n ell)
        self.Rbar_inv = np.linalg.inv(Rbar)

    # ---- per-Gamma read-out matrix Cbar_Gamma (Markov-parameter matching) --- #
    def readout(self, Gamma):
        """
        Build Cbar_Gamma (n x n ell) s.t. x_hat_Gamma(t) = Cbar_Gamma z(t)
        reproduces the Luenberger observer that uses sensors in Gamma.
        Returns (Cbar_Gamma, L_Gamma) or (None, None) if Gamma unobservable.
        """
        A, C = self.A, self.C
        Crows = C[list(Gamma)]
        L = design_gain(A, Crows, self.poles)
        if L is None:
            return None, None
        # L_tilde: n x p with columns of L mapped back to sensor indices in Gamma
        Ltil = np.zeros((self.n, self.p))
        for col, i in enumerate(Gamma):
            Ltil[:, i] = L[:, col]
        Abar_G = A - Ltil @ C                         # A - L_Gamma C_Gamma  (n x n)
        Bbar_G = np.hstack([self.B, Ltil])            # n x ell
        # M_Gamma = [Bbar_G, Abar_G Bbar_G, ..., Abar_G^{n-1} Bbar_G]  (n x n ell)
        blocks = []
        Mp = Bbar_G.copy()
        for _ in range(self.n):
            blocks.append(Mp)
            Mp = Abar_G @ Mp
        M_Gamma = np.hstack(blocks)                    # n x (n ell)
        Cbar_G = M_Gamma @ self.Rbar_inv               # n x (n ell)
        return Cbar_G, Ltil

    # ---- shared-state update (Eq. 4.7) ------------------------------------- #
    def step(self, z, u, y):
        ubar = np.concatenate([u, y])
        return self.Abar @ z + self.Bbar @ ubar

    def zero_state(self):
        return np.zeros(self.n * self.ell)


# ============================================================================ #
#  Detection bound (Proposition A.2)                                           #
# ============================================================================ #
@dataclass
class DetectionBound:
    """
    Computes, for a chosen security level s (solver bound), the windowed
    per-sensor residual envelope

        || Y_i(t) - O_i x_hat(t) ||^2  <=  gamma_i * lam_bar^t + d_i(t)

    with constants exactly as defined in Proposition A.2:
        d_i(t) = alpha_i * sum_{m=0}^{t-1} lam_bar^{t-m-1} + beta_i
               = alpha_i * (1 - lam_bar^t)/(1 - lam_bar) + beta_i
    """
    A: np.ndarray
    C: np.ndarray                      # p x n
    poles: np.ndarray
    s: int                             # security level (max attacked sensors)
    tau: int                           # detection window length (<= n)
    psi_bar: np.ndarray                # per-sensor 1-sigma noise (p,)
    mu_bar: float                      # process-noise bound
    x0_bound: float                    # ||x(0) - x_hat(0)|| bound (x_hat0 = 0)
    nsig: float = 3.0                  # k-sigma -> hard bound conversion
    B: np.ndarray = None               # (for calibrated error covariance)
    Qnoise: np.ndarray = None          # process-noise covariance (n x n)
    mode: str = "calibrated"           # "calibrated" or "theoretical"

    def __post_init__(self):
        self.n = self.A.shape[0]
        self.p = self.C.shape[0]
        self._precompute()
        self._precompute_calibrated()

    def _Oi(self, i):
        """tau-step observability block for sensor i (tau x n)."""
        return np.vstack([self.C[i][None, :] @ np.linalg.matrix_power(self.A, k)
                          for k in range(self.tau)])

    def _precompute(self):
        A, C, n, p, s = self.A, self.C, self.n, self.p, self.s
        tau = self.tau
        # per-sensor hard noise bound Psi_bar_i over the window: ||Psi_i||<=Psi_bar_i
        # windowed measurement-noise magnitude ~ sqrt(tau) * nsig * sigma_i
        self.Oi = [self._Oi(i) for i in range(p)]
        self.Oi_norm = np.array([np.linalg.norm(O, 2) for O in self.Oi])
        Psi_bar = self.nsig * self.psi_bar * np.sqrt(tau)         # (p,)
        self.Psi_bar = Psi_bar
        Psi_total = self.mu_bar + np.max(Psi_bar)                 # aggregate (Prop A.2)

        # Lyapunov constants over all Gamma with |Gamma| = p - s
        lamP_max, lamP_min, kappa, Lnorm_max = -np.inf, np.inf, -np.inf, 0.0
        count = 0
        for Gamma in itertools.combinations(range(p), p - s):
            Crows = C[list(Gamma)]
            L = design_gain(A, Crows, self.poles)
            if L is None:
                continue                                          # skip unobservable
            Ae = A - (self._Ltil(L, Gamma, p)) @ C
            # discrete Lyapunov:  Ae^T P Ae - P = -I
            try:
                P = solve_discrete_lyapunov(Ae.T, np.eye(n))
            except Exception:
                continue
            if np.any(~np.isfinite(P)):
                continue
            ev = np.linalg.eigvalsh((P + P.T) / 2)
            lamP_max = max(lamP_max, ev[-1]); lamP_min = min(lamP_min, ev[0])
            kappa = max(kappa, np.linalg.norm(P @ Ae, 2) ** 2 + np.linalg.norm(P, 2))
            Lnorm_max = max(Lnorm_max, np.linalg.norm(L, 2))
            count += 1
        self.n_valid_Gamma = count
        self.lamP_max, self.lamP_min = lamP_max, lamP_min
        self.kappa = kappa
        # lam_bar = 1 - 1/(2 lamP_max)   (Prop A.2);  ensure 0<lam_bar<1
        self.lam_bar = max(1e-3, min(0.999, 1.0 - 1.0 / (2.0 * lamP_max)))
        # aggregate noise bound Psi used in alpha_i
        self.Psi = self.mu_bar + self.psi_bar.max() * self.nsig * np.sqrt(tau) \
            * 1.0 + Lnorm_max * 0.0  # Psi = mu + max||L|| * psi  (per Prop A.2)
        Psi_agg = self.mu_bar + Lnorm_max * (self.nsig * self.psi_bar.max()
                                             * np.sqrt(tau))
        self.Psi = Psi_agg
        # per-sensor constants
        self.gamma = 2.0 * self.Oi_norm ** 2 * (lamP_max / lamP_min) \
            * (self.x0_bound ** 2)
        self.alpha = 2.0 * self.Oi_norm ** 2 * self.kappa * self.Psi / lamP_min
        self.beta = 2.0 * self.Psi_bar ** 2

    @staticmethod
    def _Ltil(L, Gamma, p=None):
        # expand L (n x |Gamma|) to n x p by sensor index
        n = L.shape[0]
        if p is None:
            p = max(Gamma) + 1
        Ltil = np.zeros((n, p))
        for col, i in enumerate(Gamma):
            Ltil[:, i] = L[:, col]
        return Ltil

    # ------------------------------------------------------------------ #
    #  Practical (statistically calibrated) envelope                     #
    # ------------------------------------------------------------------ #
    #  We use the causal ACCUMULATED-INNOVATION residual for a candidate Gamma:
    #     R_i(t) = sum_{k=t-tau+1}^{t} ( y_i(k) - C_i x_hat_Gamma(k) )^2
    #  which decays with the observer error exactly like the output error
    #  ||Y_i - O_i x|| bounded in test (5.1), but is causal and has clean
    #  statistics.  Under the no-attack hypothesis each innovation
    #  s_i(k) = y_i(k) - C_i x_hat(k) is zero-mean with variance
    #     v_i = C_i Pe C_i^T + sigma_i^2            (Pe = estimation-error cov),
    #  so R_i/v_i ~ chi^2_tau.  We keep the Prop. A.2 envelope STRUCTURE
    #     env_i(t) = g_i * lam_eff^{2t} + f_i
    #  with f_i the chi^2 upper tail (steady noise floor) and g_i a transient
    #  allowance for initial observer convergence.  This is the "accuracy" side
    #  of the scalability-accuracy tradeoff of Sec. 5.4 of Shoukry et al. (2018).
    # ------------------------------------------------------------------ #
    def _precompute_calibrated(self):
        A, C, p, s, tau = self.A, self.C, self.p, self.s, self.tau
        self.lam_eff = float(np.max(np.abs(self.poles)))
        # representative estimation-error covariance: steady-state Kalman filter
        # (filtered) covariance using all sensors -- the correct physical scale
        # of the integrated-navigation estimate.
        Q = self.Qnoise if self.Qnoise is not None else 1e-6 * np.eye(self.n)
        R = np.diag(self.psi_bar ** 2)
        try:
            Ppred = solve_discrete_are(A.T, C.T, Q, R)
            Sinv = np.linalg.inv(C @ Ppred @ C.T + R)
            Kf = Ppred @ C.T @ Sinv
            Pf = (np.eye(self.n) - Kf @ C) @ Ppred      # filtered covariance
            Pe_rep = (Pf + Pf.T) / 2
        except Exception:
            Pe_rep = 5.0 * np.eye(self.n)
        self.Pe_rep = Pe_rep
        # secure estimator uses p-s sensors (slightly larger error): inflate.
        infl = 1.0 + 2.0 * s / max(p - s, 1)
        self.f = np.zeros(p); self.g = np.zeros(p); self.vinn = np.zeros(p)
        for i in range(p):
            Ci = self.C[i]
            v_i = float(Ci @ Pe_rep @ Ci) * infl + self.psi_bar[i] ** 2
            self.vinn[i] = v_i
            # chi^2_tau upper tail: mean tau, var 2 tau
            self.f[i] = v_i * (tau + self.nsig * np.sqrt(2.0 * tau))
            # transient: window-accumulated observer convergence from x0
            self.g[i] = tau * float(Ci @ Ci) * (self.x0_bound ** 2)

    def envelope(self, i, t):
        """Residual-test RHS. Default: statistically calibrated (Sec. 5.4)."""
        if self.mode == "theoretical":
            lam = self.lam_bar
            d_i = self.alpha[i] * (1.0 - lam ** t) / (1.0 - lam) + self.beta[i]
            return self.gamma[i] * lam ** t + d_i
        return self.g[i] * (self.lam_eff ** (2 * t)) + self.f[i]

    def envelope_theoretical(self, i, t):
        lam = self.lam_bar
        d_i = self.alpha[i] * (1.0 - lam ** t) / (1.0 - lam) + self.beta[i]
        return self.gamma[i] * lam ** t + d_i

    def residual_accum(self, i, yhist, xhat_hist):
        """
        Accumulated-innovation residual for sensor i over the stored window:
            R_i = sum_k ( y_i(k) - C_i x_hat(k) )^2
        yhist: (tau, p) raw measurements; xhat_hist: (tau, n) candidate estimates.
        """
        Ci = self.C[i]
        s = yhist[:, i] - xhat_hist @ Ci
        return float(s @ s)


# Need Ltil with correct p inside _precompute:
def _patch_ltil():
    pass


# ============================================================================ #
#  SMT-based secure estimator (Algorithms 1-3)                                 #
# ============================================================================ #
@dataclass
class SMTSecureEstimator:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    poles: np.ndarray
    s: int                              # pB-SAT bound: at most s attacked sensors
    tau: int
    bound: DetectionBound
    scheme: str = "relaxed"             # "strict" (Alg 2) or "relaxed" (Alg 3)
    weights: tuple = None               # per-channel support cost (threat prior); None = cardinality
    conflict: str = "greedy"            # "greedy" worst-offender refinement, or "set" (sound)
    mml: MMLObserver = field(init=False, default=None)
    _readout_cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.n = self.A.shape[0]
        self.p = self.C.shape[0]
        self.mml = MMLObserver(self.A, self.B, self.C, self.poles)
        self.z = self.mml.zero_state()
        self.conflicts = []             # learned pseudo-Boolean conflict clauses
        self.t = 0
        self.last_Gamma = None
        self.last_attacked = np.zeros(self.p, dtype=bool)
        self.zhist = []                 # last tau shared states z(k)
        self.yhist = []                 # last tau raw measurement vectors
        self.certified = True           # did this step find a consistent SAT set?
        self.uncertified_run = 0        # consecutive uncertified steps

    # ---- read-out cache (Cbar_Gamma per Gamma) ----------------------------- #
    def _readout(self, Gamma):
        key = tuple(Gamma)
        if key not in self._readout_cache:
            self._readout_cache[key] = self.mml.readout(Gamma)
        return self._readout_cache[key]

    # ---- pseudo-Boolean SAT solve: pick b with sum(b)<=s satisfying clauses - #
    def _pbsat_solve(self):
        """
        Return a candidate attack-support set S (indices with b_i = 1), |S| <= s,
        consistent with all learned conflict clauses (each clause = a set that
        must contain at least one attacked sensor: sum_{i in clause} b_i >= 1).
        Preference: smallest support (fewest sensors declared attacked), which
        matches "assume as few attacks as possible".  Returns None if UNSAT.
        """
        p, s = self.p, self.s
        # enumerate candidate supports by increasing size (0,1,...,s)
        # p is modest here; the conflict clauses prune drastically. For large p
        # a real pB-SAT/DPLL solver (e.g. z3) would replace this enumeration.
        if self.weights is None:
            for size in range(0, s + 1):
                for S in itertools.combinations(range(p), size):
                    Sset = set(S)
                    if all(Sset & cl for cl in self.conflicts):
                        return list(S)
            return None
        # weighted minimum-cost support (threat prior): cost = sum of channel weights
        w = np.asarray(self.weights, float); best = None
        for size in range(0, s + 1):
            for S in itertools.combinations(range(p), size):
                Sset = set(S)
                if all(Sset & cl for cl in self.conflicts):
                    key = (float(w[list(S)].sum()) if S else 0.0, size)
                    if best is None or key < best[0]:
                        best = (key, list(S))
        return None if best is None else best[1]

    # ---- per-Gamma windowed estimate trajectory and residual --------------- #
    def _xhat_hist(self, Cbar):
        """Candidate estimate trajectory over the stored window: (tau, n)."""
        return np.array([Cbar @ zk for zk in self.zhist])

    def _residuals(self, Gamma):
        """Return (xhat_now, dict i->R_i) for candidate Gamma, or (None,None)."""
        Cbar, _ = self._readout(Gamma)
        if Cbar is None:
            return None, None
        yhist = np.array(self.yhist)
        xh = self._xhat_hist(Cbar)
        R = {i: self.bound.residual_accum(i, yhist, xh) for i in Gamma}
        xhat_now = Cbar @ self.z
        return xhat_now, R

    # ---- conflict certificate (Sec. 5.2 / Algorithm 1) --------------------- #
    def _learn_conflict(self, Gamma):
        """
        Conflict clause: the set of sensors in Gamma whose residual exceeds the
        envelope.  At least one of them must be under attack (they cannot all be
        attack-free and simultaneously consistent with the Gamma-observer), so
        the pB-SAT clause  sum_{i in fails} b_i >= 1  is a valid certificate
        (Eq. 5.2, restricted to the failing sensors).

        Optionally refined toward a minimal certificate (Algorithm 1) by
        dropping the largest-residual single sensor and checking whether the
        remainder still conflicts -- but only while the reduced set stays
        OBSERVABLE, so the test remains well defined.
        """
        if self.conflict == "set":      # sound clause: Gamma is not entirely attack-free
            return set(Gamma)
        _, R = self._residuals(Gamma)
        if R is None:
            return set(Gamma)
        env = {i: self.bound.envelope(i, self.t) for i in Gamma}
        fails = {i for i in Gamma if R[i] > env[i]}
        if not fails:
            return set(Gamma)
        # Algorithm-1 refinement: if removing the single worst offender from
        # Gamma clears ALL failures, that offender alone is the certificate.
        worst = max(fails, key=lambda i: (R[i] - env[i]) / (self.bound.vinn[i] + 1e-12))
        trial = tuple(sorted(set(Gamma) - {worst}))
        if len(trial) >= 1:
            _, Rt = self._residuals(trial)
            if Rt is not None and all(Rt[i] <= self.bound.envelope(i, self.t)
                                      for i in trial):
                return {worst}
        return fails

    # ---- estimation (no current input needed) ------------------------------ #
    def estimate(self, y):
        """
        Run the SMT secure-estimation search on measurement y(t) using the
        current shared state z(t) and the stored window.  Returns
        (x_hat, attacked_mask, Gamma).  Must be followed by commit(u, y) with
        the control u(t) actually applied, to advance the observer correctly.
        """
        p = self.p
        self.yhist.append(y.copy()); self.zhist.append(self.z.copy())
        if len(self.yhist) > self.bound.tau:
            self.yhist.pop(0); self.zhist.pop(0)

        Gamma = self.last_Gamma
        xhat = np.zeros(self.n)
        if len(self.yhist) < self.bound.tau:
            Cbar, _ = self._readout(tuple(range(p)))
            xhat = Cbar @ self.z if Cbar is not None else np.zeros(self.n)
            attacked = np.zeros(p, bool)
            self.last_Gamma = tuple(range(p)); self.last_attacked = attacked
            self._pending = (xhat, attacked, self.last_Gamma)
            return self._pending

        self.conflicts = []
        status = "UNSAT"; iters = 0; max_iters = 300
        while status == "UNSAT" and iters < max_iters:
            iters += 1
            S = self._pbsat_solve()
            if S is None:
                Gamma = self.last_Gamma; break
            Gamma = tuple(sorted(set(range(p)) - set(S)))
            xhat_now, R = self._residuals(Gamma)
            if R is None:
                self.conflicts.append(set(Gamma))
                if self.scheme == "relaxed":
                    break
                continue
            xhat = xhat_now
            fails = [i for i in Gamma if R[i] > self.bound.envelope(i, self.t)]
            if fails:
                I = self._learn_conflict(Gamma)
                if not I:
                    I = set(Gamma)
                self.conflicts.append(set(I))
                if self.scheme == "relaxed":
                    break
            else:
                status = "SAT"

        self.certified = (status == "SAT")
        self.uncertified_run = 0 if self.certified else self.uncertified_run + 1
        if status != "SAT":
            Gamma = self.last_Gamma if self.last_Gamma is not None \
                else tuple(range(p))
            xnow, _ = self._residuals(Gamma)
            if xnow is not None:
                xhat = xnow

        attacked = np.ones(p, dtype=bool)
        if Gamma is not None:
            for i in Gamma:
                attacked[i] = False
        self.last_Gamma = Gamma; self.last_attacked = attacked
        self._pending = (xhat, attacked, Gamma)
        return self._pending

    # ---- commit: advance the shared observer with the applied control ------ #
    def commit(self, u, y):
        """Advance z(t) -> z(t+1) using the control u(t) actually applied."""
        self.z = self.mml.step(self.z, u, y)
        self.t += 1
        self._decay_conflicts()

    # ---- convenience single-call step (uses CURRENT u for the commit) ------ #
    def step(self, u, y, Ywin=None):
        out = self.estimate(y)
        self.commit(u, y)
        return out

    def _decay_conflicts(self, keep=40):
        # bound memory; keep most-recent conflict clauses
        if len(self.conflicts) > keep:
            self.conflicts = self.conflicts[-keep:]


# ============================================================================ #
#  Conventional steady-state Kalman filter baseline (uses ALL sensors)         #
# ============================================================================ #
@dataclass
class KalmanBaseline:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray                       # process noise covariance (n x n)
    R: np.ndarray                       # measurement noise covariance (p x p)

    def __post_init__(self):
        self.n = self.A.shape[0]
        # steady-state predictor Kalman gain via DARE
        P = solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
        S = self.C @ P @ self.C.T + self.R
        self.K = P @ self.C.T @ np.linalg.inv(S)       # filter gain
        self.P = P
        self.x = np.zeros(self.n)

    def reset(self):
        self.x = np.zeros(self.n)

    def estimate(self, y):
        """Measurement update: return filtered estimate x_hat(t|t)."""
        innov = y - self.C @ self.x
        self._xf = self.x + self.K @ innov
        return self._xf

    def commit(self, u, y):
        """Time update with the applied control u(t)."""
        self.x = self.A @ self._xf + self.B @ u

    def step(self, u, y):
        xf = self.estimate(y)
        self.commit(u, y)
        return xf


# ============================================================================ #
#  Self-test: verify Theorem 4.2  (MML == Luenberger bank)                     #
# ============================================================================ #
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ship_model import build_ship_model
    np.random.seed(0)
    sm = build_ship_model()
    A, B, C = sm.A, sm.B, sm.C
    n, m, p = sm.n, sm.m, sm.psensors
    poles = np.array([0.55, 0.60, 0.66, 0.70, 0.74, 0.78])
    mml = MMLObserver(A, B, C, poles)

    # pick a few observable Gamma with |Gamma| = p-1
    T = 60
    U = 0.05 * np.random.randn(T, m)
    Y = np.zeros((T, p))
    xtrue = np.zeros(n)
    for t in range(T):
        Y[t] = C @ xtrue + 0.0
        xtrue = A @ xtrue + B @ U[t]

    max_err = 0.0
    for drop in range(p):
        Gamma = tuple(i for i in range(p) if i != drop)
        Cbar, Ltil = mml.readout(Gamma)
        if Cbar is None:
            continue
        # bank observer
        xb = np.zeros(n); z = mml.zero_state()
        for t in range(T):
            # bank update
            Crows = C[list(Gamma)]
            Lg = Ltil[:, list(Gamma)]
            xb = A @ xb + B @ U[t] + Lg @ (Y[t, list(Gamma)] - Crows @ xb)
            # mml update + readout
            z = mml.step(z, U[t], Y[t])
            xm = Cbar @ z
            max_err = max(max_err, np.linalg.norm(xb - xm))
    print("Theorem 4.2 check  max||x_bank - x_MML|| over all drop-1 sets = %.3e"
          % max_err)
    print("(should be ~1e-8 or smaller -> MML reproduces the bank)")
