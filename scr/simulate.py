"""
simulate.py
===========
Closed-loop maritime cyberattack simulation.

The TRUE vessel evolves under the discrete-time LTI error-state dynamics
(ship_model), driven by a line-of-sight-style track-keeping autopilot.  The
autopilot closes the loop on an ESTIMATE of the cross-track error and heading.
We run, from identical truth and noise realisations, three estimator
configurations so the effect of the attack and of the secure estimator is
isolated cleanly:

    'truth-fed'  : autopilot uses the (unknowable) true state  -> ideal ref
    'naive'      : autopilot uses a conventional Kalman filter  (all sensors)
    'secure'     : autopilot uses the SMT secure estimate       (attack-resilient)

For each configuration we propagate an INDEPENDENT true trajectory (same process
and measurement noise seeds, same attack) so that the control action computed
from each estimator feeds back into its own ship.  This exposes the operational
consequence: with the naive estimator the attack steers the *true* ship off
track; with the secure estimator the true ship stays on track.

The simulator also assembles, at each step and for each sensor, the windowed
"clean-of-input" output stack  Y_i(t) = [y_i(t-tau+1) ... y_i(t)] - F_i U  that
the detection test (Eq. 5.1, Sec. 2.4) consumes.  Here F_i U removes the known
deterministic input contribution, consistent with Y_tilde in Eq. (2.3).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from secure_estimation import (MMLObserver, DetectionBound, SMTSecureEstimator,
                               KalmanBaseline, design_gain)


# --------------------------------------------------------------------------- #
#  Track-keeping autopilot (LOS-style PD on cross-track + heading)            #
# --------------------------------------------------------------------------- #
@dataclass
class Autopilot:
    """
    Line-of-sight (LOS) track-keeping autopilot.  Commands rudder delta and
    thrust tau from the estimated error state x = [dN,dE,dpsi,du,dv,dr].

    Cross-track control uses a LOS desired-heading law
        psi_des = -Klos * dE            (small-angle LOS, lookahead 1/Klos)
    realised by a PD rudder law on heading error (dpsi - psi_des) and yaw-rate
    damping dr; surge holds service speed U via proportional thrust on du.

    The control is implemented as u = -K x with the gain matrix K assembled from
    (Kpsi, Kr, Klos, Ku).  The signs of the steering gains follow the rudder
    convention of the model (b_r < 0: positive rudder -> negative yaw), and the
    resulting closed loop A - B K is Schur-stable (verified in run_experiments).
    """
    Kpsi: float = 2.0          # heading proportional gain
    Kr: float = 8.0            # yaw-rate damping gain
    Klos: float = 1.0 / 150.0  # LOS gain: cross-track -> desired heading [rad/m]
    Ku: float = 0.8            # surge proportional gain -> thrust
    delta_max: float = 0.61    # rudder saturation (~35 deg)
    tau_max: float = 2.0       # thrust command saturation
    K: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        # u = -K x  (state = [dN,dE,dpsi,du,dv,dr])
        K = np.zeros((2, 6))
        K[0, 3] = self.Ku                                   # thrust on du
        K[1, 1] = -self.Kpsi * self.Klos                    # rudder on dE (LOS)
        K[1, 2] = -self.Kpsi                                # rudder on dpsi
        K[1, 5] = -self.Kr                                  # rudder on dr
        self.K = K

    def command(self, xhat, xref=None, delta_ff=0.0):
        """
        Track-keeping command u = -K (xhat - xref) + [0, delta_ff].
        xref: desired error-state (cross-track E_ref in [1], heading psi_ref in
        [2], yaw r_ref in [5]); None -> regulate to the straight centreline.
        delta_ff: steady-turn rudder feed-forward for the reference yaw rate.
        """
        e = xhat if xref is None else (xhat - xref)
        u = -self.K @ e
        u[1] += delta_ff
        u[0] = float(np.clip(u[0], -self.tau_max, self.tau_max))
        u[1] = float(np.clip(u[1], -self.delta_max, self.delta_max))
        return u


# --------------------------------------------------------------------------- #
#  Windowed-output buffer for the detection test                              #
# --------------------------------------------------------------------------- #
class WindowBuffer:
    """
    Maintains, per sensor i, the last tau residual-free windowed outputs
        Y_i(t) = stack_{k=0..tau-1} [ y_i(t-tau+1+k) ] - F_i U(t-tau+1)
    where F_i U removes the deterministic known-input contribution over the
    window so that  Y_i(t) ~ O_i x(t-tau+1) + noise  (Eq. 2.3).  We align the
    window so O_i multiplies the state at the START of the window.
    """
    def __init__(self, model, tau):
        self.m = model
        self.p = model.psensors
        self.tau = tau
        self.n = model.n
        self.A, self.B, self.C = model.A, model.B, model.C
        self.ybuf = []      # list of raw y vectors (p,)
        self.ubuf = []      # list of u vectors (m,)
        # F_i blocks: lower-triangular Markov param stacks (tau x tau*m) per i
        self.Fi = []
        for i in range(self.p):
            Ci = self.C[i][None, :]
            F = np.zeros((tau, tau * model.m))
            for r in range(tau):
                for c in range(r):
                    # contribution of u at window-step c to y_i at window-step r
                    F[r, c*model.m:(c+1)*model.m] = (
                        Ci @ np.linalg.matrix_power(self.A, r - c - 1) @ self.B)
            self.Fi.append(F)

    def push(self, y, u):
        self.ybuf.append(y.copy()); self.ubuf.append(u.copy())
        if len(self.ybuf) > self.tau:
            self.ybuf.pop(0); self.ubuf.pop(0)

    def ready(self):
        return len(self.ybuf) == self.tau

    def windows(self):
        """Return dict i -> Y_i(t) (tau,), residual-free (input removed)."""
        Ystack = np.array(self.ybuf)           # tau x p
        Ustack = np.concatenate(self.ubuf)     # (tau*m,)
        out = {}
        for i in range(self.p):
            yi = Ystack[:, i]                  # tau,
            out[i] = yi - self.Fi[i] @ Ustack
        return out


# --------------------------------------------------------------------------- #
#  Simulation configuration & result                                          #
# --------------------------------------------------------------------------- #
def calibrate_floors(model, poles, s, tau, mu_bar, proc_sigma, x0, nsig,
                     n_seeds=24, T=220, quantile=0.999, margin=1.5,
                     seed0=50000, autopilot=None, weave_amp=(0.08, 0.22)):
    """
    Empirically calibrate the per-sensor detection floor f_i from CLEAN
    (no-attack) Monte-Carlo runs.  For each run we drive the true ship with a
    truth-fed autopilot (so the trajectory is representative), feed the MML
    observer, and record the accumulated-innovation residual R_i(t) for the
    nominal all-sensors hypothesis Gamma = {1..p}.  The floor is set to a high
    quantile (x margin) of the observed clean residuals -- a principled,
    distribution-free threshold with a controlled false-alarm rate that
    accounts exactly for dynamic coupling and windowed-residual correlation.
    Returns (f_floors (p,), bound_object).
    """
    A, B, C = model.A, model.B, model.C
    n, m, p = model.n, model.m, model.psensors
    scale = np.array([proc_sigma*2, proc_sigma*2, proc_sigma*0.2,
                      proc_sigma, proc_sigma, proc_sigma*0.2])
    Qcov = np.diag(scale ** 2) + 1e-9 * np.eye(n)
    x0_bound = float(np.linalg.norm(x0)) + 5.0
    db = DetectionBound(A, C, poles, s=s, tau=tau, psi_bar=model.sigma,
                        mu_bar=mu_bar, x0_bound=x0_bound, nsig=nsig, B=B,
                        Qnoise=Qcov, mode="calibrated")
    ap = autopilot if autopilot is not None else Autopilot()
    Rsamples = [[] for _ in range(p)]
    Gamma_all = tuple(range(p))
    warmup = 50          # discard initial observer-convergence transient
    # calibration must exercise the SAME dynamics the attacked closed loop will:
    # the ship weaves (helm perturbations) so heading/yaw residuals reflect real
    # manoeuvring, and the observer is fed the lagged input exactly as in
    # operation.  This prevents spurious failures of the low-noise channels when
    # the real ship manoeuvres against a spoof.
    for sd in range(n_seeds):
        rng = np.random.default_rng(seed0 + sd)
        x = np.array(x0, float)
        mml = MMLObserver(A, B, C, poles)
        z = mml.zero_state()
        Cbar_all, _ = mml.readout(Gamma_all)
        zhist, yhist = [], []
        uprev = np.zeros(m)
        # random weaving reference (helm perturbation) per seed
        periods = rng.uniform(25, 60, 3); amps = rng.uniform(weave_amp[0], weave_amp[1], 3)
        phases = rng.uniform(0, 2*np.pi, 3)
        for t in range(T):
            y = C @ x + rng.normal(0, 1, p) * model.sigma
            zhist.append(z.copy()); yhist.append(y.copy())
            if len(zhist) > tau:
                zhist.pop(0); yhist.pop(0)
            if len(zhist) == tau and t >= warmup:
                xh = np.array([Cbar_all @ zk for zk in zhist])
                yy = np.array(yhist)
                for i in range(p):
                    Rsamples[i].append(db.residual_accum(i, yy, xh))
            # weaving helm perturbation on top of track-keeping (exercises yaw)
            dperturb = float(np.sum(amps * np.sin(2*np.pi*t/periods + phases)))
            u = ap.command(x) + np.array([0.0, dperturb])
            u[1] = float(np.clip(u[1], -ap.delta_max, ap.delta_max))
            z = mml.step(z, u, y)                     # observer fed CURRENT input
            w = rng.normal(0, 1, n) * scale
            x = A @ x + B @ u + w
    f = np.zeros(p)
    for i in range(p):
        arr = np.array(Rsamples[i])
        q = float(np.quantile(arr, quantile))
        mx = float(arr.max())
        # robust floor: generous over the observed clean tail, so transient
        # innovation spikes on low-noise channels do not cause spurious
        # non-convergence, while still far below any sustained attack residual.
        f[i] = max(q * margin, mx * 1.25) + 1e-9
    db.f = f                                          # override analytic floor
    db.f_empirical = f
    return f, db


@dataclass
class SimConfig:
    T: int = 420                 # steps (s, since dt=1)
    tau: int = 6                 # detection window
    s: int = 1                   # SMT security level (max attacked sensors)
    poles: tuple = (0.55, 0.60, 0.66, 0.70, 0.74, 0.78)
    mu_bar: float = 0.05         # process-noise bound (for detector)
    proc_sigma: float = 0.02     # actual process-noise 1-sigma (per state scaled)
    x0: tuple = (0, 15.0, 0.0, 0, 0, 0)   # initial error state (15 m off-track)
    seed: int = 7
    scheme: str = "strict"
    nsig: float = 4.0
    calib_margin: float = 1.5
    calib_quantile: float = 0.999
    channel_hazard_E: float = 90.0        # East distance to channel-edge hazard [m]
    weave_amp: tuple = (0.08, 0.22)       # calibration helm-weave amplitude range [rad]
    smt_weights: tuple = None             # per-channel attack-support costs (threat prior); None = cardinality
    smt_conflict: str = "greedy"           # "greedy" (worst-offender refinement) or "set" (sound whole-set clause)
    switch_step: int = None               # Tier-0 hypothesis switch: step at which new SMT settings apply
    switch_weights: tuple = None          # threat-prior weights applied from switch_step (None = cardinality)
    switch_conflict: str = "set"           # conflict-clause mode applied from switch_step
    noise_clip: float = None              # clip Gaussian noise draws at +-noise_clip sigma (bounded-noise runs)


@dataclass
class SimResult:
    cfg: SimConfig
    t: np.ndarray
    # per-config true trajectories and estimates
    x_true: dict          # config -> (T,n)
    x_hat: dict           # config -> (T,n)
    u: dict               # config -> (T,m)
    attacked: np.ndarray  # (T,p) bool  (secure estimator decisions)
    attack_truth: np.ndarray   # (T,p) bool  (ground-truth attacked channels)
    y_clean: np.ndarray   # (T,p)
    y_att: np.ndarray     # (T,p)
    attack_name: str
    detbound: object
    model: object
    smt_trace: dict = None    # {'t','R','env','attacked_channels'} under Gamma=all
    path: object = None       # ChannelPath for the maneuvering example (else None)
    E_ref: np.ndarray = None  # (T,) reference cross-track centreline
    psi_ref: np.ndarray = None
    rta_mode: dict = None     # config -> (T,) int mode {0 nominal,1 backup,2 failsafe}
    rta_summary: dict = None
    dr_trace: dict = None     # {"dE_dr","gate","discrepancy"} (secure config with rta)
    cert_trace: dict = None   # certified-filter trace (secure config with cert_design)


def _proc_noise(model, sigma, rng, clip=None):
    # scale process noise per state (positions get larger absolute noise)
    scale = np.array([sigma*2, sigma*2, sigma*0.2, sigma, sigma, sigma*0.2])
    z = rng.normal(0, 1, model.n)
    return (clip(z) if clip is not None else z) * scale


def simulate(model, attack, cfg: SimConfig, configs=("truth", "naive", "secure"),
             verbose=False, path=None, rta=False, rta_env=None, autopilot=None,
             proposal_fn=None, cert_design=None, cert_switch=None):
    rng_master = np.random.default_rng(cfg.seed)
    A, B, C = model.A, model.B, model.C
    n, m, p = model.n, model.m, model.psensors
    poles = np.array(cfg.poles)
    dt = model.p.dt
    U = model.p.U
    # reference trajectory (maneuvering example) -------------------------------
    if path is not None:
        from route import steady_turn_feedforward
        gff_delta, _ = steady_turn_feedforward(model)
        def reference(t):
            pp = U * t * dt
            E_r, psi_r, r_r = path.ref(pp)
            xref = np.array([0.0, E_r, psi_r, 0.0, 0.0, r_r])
            return xref, gff_delta * r_r
    else:
        def reference(t):
            return None, 0.0
    E_ref_log = np.zeros(cfg.T); psi_ref_log = np.zeros(cfg.T)

    # noise covariances for KF and detector
    Rcov = np.diag(model.sigma ** 2)
    scale = np.array([cfg.proc_sigma*2, cfg.proc_sigma*2, cfg.proc_sigma*0.2,
                      cfg.proc_sigma, cfg.proc_sigma, cfg.proc_sigma*0.2])
    Qcov = np.diag(scale ** 2) + 1e-9 * np.eye(n)

    # detection bound with EMPIRICALLY CALIBRATED per-sensor floors (clean MC)
    _, detb = calibrate_floors(model, poles, cfg.s, cfg.tau, cfg.mu_bar,
                               cfg.proc_sigma, cfg.x0, cfg.nsig,
                               quantile=cfg.calib_quantile, margin=cfg.calib_margin,
                               autopilot=autopilot, weave_amp=cfg.weave_amp)

    autopilot = autopilot if autopilot is not None else Autopilot()

    x_true_all, x_hat_all, u_all = {}, {}, {}
    attacked_log = None
    attack_truth = None
    smt_trace = None
    rta_mode_all = {}; rta_summary_all = {}; dr_trace_all = None; cert_trace = None
    _clip = (lambda z: np.clip(z, -cfg.noise_clip, cfg.noise_clip)) if cfg.noise_clip else (lambda z: z)
    _MODE = {None: 0}
    if rta:
        from safety_filter import SafetyFilter, SafetyEnvelope, ActionType
        _MODE = {ActionType.NOMINAL: 0, ActionType.BACKUP: 1,
                 ActionType.FAILSAFE: 2}
    y_clean_log = np.zeros((cfg.T, p)); y_att_log = np.zeros((cfg.T, p))

    for conf in configs:
        rng = np.random.default_rng(cfg.seed + 1000)   # same noise across configs
        x = np.array(cfg.x0, float)
        kf = KalmanBaseline(A, B, C, Qcov, Rcov)
        smt = SMTSecureEstimator(A, B, C, poles, s=cfg.s, tau=cfg.tau,
                                 bound=detb, scheme=cfg.scheme,
                                 weights=cfg.smt_weights, conflict=cfg.smt_conflict)
        win = WindowBuffer(model, cfg.tau)

        X = np.zeros((cfg.T, n)); Xhat = np.zeros((cfg.T, n)); Us = np.zeros((cfg.T, m))
        att_log = np.zeros((cfg.T, p), bool)
        att_truth = np.zeros((cfg.T, p), bool)
        R_log = np.full((cfg.T, p), np.nan); env_log = np.full((cfg.T, p), np.nan)
        mode_log = np.zeros(cfg.T, int)
        dr_log = np.full((cfg.T, 3), np.nan)
        cf = None
        if cert_design is not None and conf == "secure":
            from certified_filter import CertifiedFilter, IX as _CIX
            cf = CertifiedFilter(cert_design)
            c_mode = np.full(cfg.T, -1, int); c_set = [None] * cfg.T
            c_eta = np.full((cfg.T, 4), np.nan); c_in = np.zeros(cfg.T, bool); c_inE = np.zeros(cfg.T, bool)
            c_trusted_clean = np.zeros(cfg.T, bool); c_how = [None] * cfg.T; c_valid = np.zeros(cfg.T, bool)
        sf = None
        if rta and conf == "secure":
            sf = SafetyFilter(model, autopilot,
                              rta_env if rta_env is not None else SafetyEnvelope())

        xhat_prev = np.zeros(n)
        for t in range(cfg.T):
            tt = t * dt
            # --- measurements ---
            meas_noise = _clip(rng.normal(0, 1, p)) * model.sigma
            y_clean = C @ x + meas_noise
            a = attack(tt, x, model)
            y = y_clean + a
            if conf == configs[0]:
                y_clean_log[t] = C @ x + meas_noise
                y_att_log[t] = y
            att_truth[t] = np.abs(a) > 1e-9
            if conf == "secure" and cfg.switch_step is not None and t == cfg.switch_step:
                smt.weights = cfg.switch_weights; smt.conflict = cfg.switch_conflict

            # --- estimators ---
            # --- estimates (do NOT yet advance observers) ---
            xkf = kf.estimate(y)
            xsec, attacked, Gamma = smt.estimate(y)
            att_log[t] = attacked
            # log residual vs envelope under the all-sensors hypothesis (secure)
            if conf == "secure" and len(smt.zhist) == cfg.tau:
                _, Rall = smt._residuals(tuple(range(p)))
                if Rall is not None:
                    for i in range(p):
                        R_log[t, i] = Rall[i]
                        env_log[t, i] = detb.envelope(i, smt.t)

            # --- reference trajectory (maneuvering example) ---
            xref, delta_ff = reference(t)
            if xref is not None:
                E_ref_log[t] = xref[1]; psi_ref_log[t] = xref[2]

            # --- choose estimate feeding the autopilot ---
            if conf == "truth":
                xctrl = x.copy()
            elif conf == "naive":
                xctrl = xkf
            else:
                xctrl = xsec
            Xhat[t] = xctrl if conf != "truth" else x

            # --- control (track the reference, with turn feed-forward) ---
            u = autopilot.command(xctrl, xref=xref, delta_ff=delta_ff)
            if proposal_fn is not None and conf == "secure":     # supervisory proposal u^a (stress tests)
                up = proposal_fn(tt, xctrl)
                if up is not None:
                    u = np.asarray(up, float).copy()
            # --- exact Theorem-1 filter (certified bank), active from step k_a ---
            cert_handled = False
            if cf is not None:
                if t == 0:
                    cf.initialize(y)
                if cert_switch is not None and t == cert_switch[0]:
                    cf.set_hypothesis(cert_switch[1])
                steer, cmode, cname, chow = cf.filter(attacked, u[1])
                c_mode[t] = cmode; c_set[t] = cname; c_how[t] = chow
                xi_true = x[_CIX]; c_in[t] = cert_design.contains(xi_true, tol=1e-9)
                if cname is not None:
                    eta = (x - cf.xh[cname])[_CIX]; c_eta[t] = eta
                    c_inE[t] = bool(np.all(np.abs(eta) <= cert_design.Exi[cname] + 1e-12))
                    c_trusted_clean[t] = not bool(np.any(att_truth[t][cert_design.sets[cname]]))
                if cmode in (0, 1):
                    u = u.copy(); u[1] = steer; cert_handled = True
                    mode_log[t] = cmode
                c_valid[t] = cert_design.in_valid(x[_CIX], u[1])
            # --- runtime-assurance safety filter (secure config only) ---
            if sf is not None and not cert_handled:
                u, mode, info = sf.filter(
                    xsec, attacked, u, xref=xref, delta_ff=delta_ff,
                    estimator_certified=smt.certified,
                    uncertified_run=smt.uncertified_run)
                mode_log[t] = _MODE.get(mode, 0)
                if sf.dE_dr is not None:
                    gate = sf.dr_nsig * np.sqrt(sf.dr_var) + sf.env.margin if sf.dr_var > 0 else np.inf
                    dr_log[t] = (sf.dE_dr, gate, abs(sf.dE_dr - xsec[sf.env.cross_track_idx]))
            Us[t] = u
            X[t] = x

            # --- advance observers with the control u(t) actually applied ---
            kf.commit(u, y)
            smt.commit(u, y)
            if cf is not None:
                cf.update(u, y)

            # --- plant update ---
            w = _proc_noise(model, cfg.proc_sigma, rng, _clip)
            x = A @ x + B @ u + w

        x_true_all[conf] = X; x_hat_all[conf] = Xhat; u_all[conf] = Us
        rta_mode_all[conf] = mode_log
        if cf is not None:
            cert_trace = dict(mode=c_mode, set=c_set, how=c_how, eta=c_eta, in_omega=c_in, in_E=c_inE,
                              trusted_clean=c_trusted_clean, valid=c_valid, k_a=0)
        if sf is not None:
            rta_summary_all[conf] = sf.summary()
            dr_trace_all = dict(dE_dr=dr_log[:, 0], gate=dr_log[:, 1], discrepancy=dr_log[:, 2])
        if conf == "secure":
            attacked_log = att_log
            atk_ch = np.where(np.any(att_truth, axis=0))[0].tolist()
            smt_trace = dict(t=np.arange(cfg.T) * dt, R=R_log, env=env_log,
                             attacked_channels=atk_ch)
        attack_truth = att_truth
        if verbose:
            print(f"  [{conf}] done. final |dE|={abs(X[-1,1]):.1f} m")

    return SimResult(cfg=cfg, t=np.arange(cfg.T)*dt, x_true=x_true_all,
                     x_hat=x_hat_all, u=u_all, attacked=attacked_log,
                     attack_truth=attack_truth, y_clean=y_clean_log,
                     y_att=y_att_log, attack_name=attack.name,
                     detbound=detb, model=model, smt_trace=smt_trace,
                     path=path, E_ref=E_ref_log, psi_ref=psi_ref_log,
                     rta_mode=rta_mode_all, rta_summary=rta_summary_all,
                     dr_trace=dr_trace_all, cert_trace=cert_trace)
