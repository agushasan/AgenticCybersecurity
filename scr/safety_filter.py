"""
safety_filter.py  —  Milestone M1 of the agentic framework
===========================================================
Runtime assurance (RTA / Simplex-style) for the autonomous vessel: a verified
safety layer that bounds the vessel's behaviour **regardless of the (future)
agents' decisions**.  This is the trustworthiness backbone of the agentic
architecture (AGENTIC_FRAMEWORK.md, Tier 0 / §4.4).  No agents are used here.

Two mechanisms:

1. Predictive backup filter (recoverability check).
   The safe set is the channel  S = { |e| <= e_max }  (e = cross-track error,
   possibly about a curving reference E_ref).  A verified BACKUP controller
   (aggressive cross-track regulator on the SECURE estimate) defines a
   control-invariant containment behaviour.  At each step the nominal command is
   admitted only if, after applying it, the backup can still contain the vessel
   over a horizon H (recursive feasibility), with an uncertainty margin; else the
   backup's action is applied.  This keeps the TRUE state in S whenever the
   secure estimate is trustworthy.

2. Observability-triggered fail-safe (epistemic safety).
   Safety via (1) presupposes a trustworthy estimate.  When the attacked set
   destroys observability of the cross-track position from the surviving
   (un-flagged) sensors — the Scenario-C regime — the estimate of e is
   meaningless and (1) cannot help.  The RTA detects this using the system's
   observability structure and engages a FAIL-SAFE: it stops trusting the
   position channel, holds the last reference heading using the trusted
   heading/rate estimate (zero cross-track feedback), and raises an escalation
   flag.  The vessel then travels straight instead of being steered off by the
   corrupted estimate, bounding the excursion to (sway + residual-heading)·time
   rather than the open-loop runaway.

Guarantee (informal; ties to the SSE analysis).
   Under 2s-sparse observability of the trusted set, the secure estimate error is
   bounded (Shoukry et al., Thm 6.2 / our calibrated bound); the predictive
   filter then keeps the TRUE cross-track within  e_max + eps, eps = O(estimate
   error).  If that observability is lost, the fail-safe guarantees a bounded,
   predictable excursion instead of divergence.  The guarantee does NOT depend on
   any agent being correct — only on the filter's authority being final.

Certified action library A_cert (interface used later by the agents; M1 wires
the safety-relevant ones).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from secure_estimation import _obsv  # observability matrix helper


# --------------------------------------------------------------------------- #
#  Certified action library  A_cert                                           #
# --------------------------------------------------------------------------- #
class ActionType(Enum):
    NOMINAL   = "nominal"      # follow the performance controller
    BACKUP    = "backup"       # aggressive cross-track containment (RTA)
    FAILSAFE  = "failsafe"     # heading-hold, drop position feedback, escalate
    DERATE    = "derate"       # reduce speed (buy time)  [interface, M2]
    ISOLATE   = "isolate"      # exclude sensor set        [interface, M2]
    HOLD      = "hold"         # station-keep              [interface, M2]


@dataclass
class CertifiedAction:
    kind: ActionType
    params: dict = field(default_factory=dict)
    rationale: str = ""


# --------------------------------------------------------------------------- #
#  Safety envelope                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class SafetyEnvelope:
    e_max: float = 70.0        # cross-track safe bound [m] (inside the 90 m hazard)
    horizon: int = 25          # backup containment horizon [steps]
    margin: float = 8.0        # uncertainty inflation on predicted excursion [m]
    backup_gain_scale: float = 2.2   # backup cross-track aggressiveness vs nominal
    obs_tol: float = 1e-6      # observability rank tolerance
    cross_track_idx: int = 1   # state index of cross-track position (dE)
    # platform-specific limits and dead-reckoning parameters (ship defaults)
    est_error_bound: float = 12.0   # allowance for secure-estimate error [m]
    u_steer_max: float = 0.61       # steering actuator limit (rudder/wheel) [rad]
    u_thrust_max: float = 2.0       # longitudinal command limit
    dr_sigma_rate: float = 0.15     # trusted velocity/heading noise [m/s-eq]
    dr_nsig: float = 5.0            # k-sigma dead-reckoning gate
    dr_warmup: int = 60             # steps before the DR check arms
    dr_window: int = 0              # re-anchor DR every N steps when not alarmed (0 = never)


# --------------------------------------------------------------------------- #
#  Observability of the cross-track position from the trusted sensor set       #
# --------------------------------------------------------------------------- #
def crosstrack_observable(model, trusted_mask, env: SafetyEnvelope):
    """
    Is the cross-track position state observable from the currently trusted
    (un-flagged) sensors?  Returns (observable: bool, unobs_weight: float) where
    unobs_weight is the projection of the cross-track direction onto the
    unobservable subspace (0 = fully observable, 1 = fully unobservable).
    """
    A, C = model.A, model.C
    idx = np.where(trusted_mask)[0]
    if idx.size == 0:
        return False, 1.0
    O = _obsv(A, C[idx])
    # unobservable subspace = right-null space of O
    u, s, vh = np.linalg.svd(O)
    n = A.shape[0]
    tol = env.obs_tol * max(O.shape) * (s[0] if s.size else 1.0)
    rank = int(np.sum(s > tol))
    if rank == n:
        return True, 0.0
    null_basis = vh[rank:].conj().T            # n x (n-rank)
    e_dir = np.zeros(n); e_dir[env.cross_track_idx] = 1.0
    proj = null_basis @ (null_basis.T @ e_dir)
    w = float(np.linalg.norm(proj))            # 0..1
    return (w < 0.1), w


# --------------------------------------------------------------------------- #
#  Backup controller (aggressive cross-track containment)                      #
# --------------------------------------------------------------------------- #
def _backup_K(autopilot, scale):
    K = autopilot.K.copy()
    K[1, 1] *= scale          # stronger cross-track -> rudder
    K[1, 2] *= 1.3            # stronger heading authority
    return K


def _predict_excursion(model, K, x0, xref, delta_ff, H, env, u_rudder_max=0.61):
    """Max |e - e_ref| over H steps of the backup controller from x0."""
    A, B = model.A, model.B
    x = x0.copy(); worst = abs(x[env.cross_track_idx] - (xref[1] if xref is not None else 0.0))
    for _ in range(H):
        e = x if xref is None else (x - xref)
        u = -K @ e
        u[1] = float(np.clip(u[1] + delta_ff, -u_rudder_max, u_rudder_max))
        u[0] = float(np.clip(u[0], -env.u_thrust_max, env.u_thrust_max))
        x = A @ x + B @ u
        dev = abs(x[env.cross_track_idx] - (xref[1] if xref is not None else 0.0))
        worst = max(worst, dev)
    return worst


# --------------------------------------------------------------------------- #
#  The runtime-assurance safety filter                                         #
# --------------------------------------------------------------------------- #
@dataclass
class SafetyFilter:
    model: object
    autopilot: object
    env: SafetyEnvelope = field(default_factory=SafetyEnvelope)

    def __post_init__(self):
        self.K_backup = _backup_K(self.autopilot, self.env.backup_gain_scale)
        # fail-safe controller: heading-hold, NO cross-track position feedback
        self.K_failsafe = self.autopilot.K.copy()
        self.K_failsafe[1, 1] = 0.0            # drop (untrustworthy) cross-track term
        self.n_fail = 0; self.n_backup = 0; self.n_nominal = 0
        # independent dead-reckoning cross-check state (catches consistent
        # common-mode position spoofing that residuals cannot, as in Scenario C)
        self.dE_dr = None                      # dead-reckoned cross-track position
        self.dr_var = 0.0                      # accumulated DR variance [m^2]
        self.U = self.model.p.U
        self.dt = self.model.p.dt
        self.dr_sigma_rate = self.env.dr_sigma_rate
        self.dr_nsig = self.env.dr_nsig
        self.dr_warmup = self.env.dr_warmup
        self._k = 0                            # filter step counter
        self._dr_alarm = False                 # latched once the DR gate trips

    def filter(self, x_hat, attacked_mask, u_nom, xref=None, delta_ff=0.0,
               est_error_bound=None, estimator_certified=True, uncertified_run=0,
               uncertified_trip=6):
        """
        Return (u_safe, mode, info).  x_hat: secure estimate; attacked_mask:
        current isolation decision (flagged sensors are untrusted).
        estimator_certified / uncertified_run: whether the secure estimator could
        certify a consistent attack-free set this step — the robust epistemic
        signal that the estimate is untrustworthy (the unrecoverable regime).
        """
        env = self.env
        if est_error_bound is None:
            est_error_bound = env.est_error_bound
        trusted = ~np.asarray(attacked_mask, bool)

        # --- independent dead-reckoning cross-check ----------------------------
        # Propagate cross-track position from the TRUSTED velocity/heading
        # estimate (log + compass + IMU, physically un-attacked in the
        # common-mode position-spoof regime) and compare to the position the
        # estimator reports.  Divergence beyond the DR uncertainty gate reveals a
        # consistent position spoof that residual tests cannot (Scenario C), while
        # staying silent when the SSE already recovers the true position (B).
        self._k += 1
        rate = self.U * x_hat[2] + x_hat[4]            # U*dpsi + dv  (trusted)
        reanchor = (env.dr_window > 0 and not self._dr_alarm and self._k > self.dr_warmup
                    and (self._k - self.dr_warmup) % env.dr_window == 0)
        if self.dE_dr is None or self._k <= self.dr_warmup or reanchor:
            # settle (or sliding-window re-anchor while trusted): no gate this step
            self.dE_dr = float(x_hat[env.cross_track_idx]); self.dr_var = 0.0
            dr_spoof = False; dr_discrepancy = 0.0; dr_gate = np.inf
        else:
            self.dE_dr += self.dt * rate
            self.dr_var += (self.dt * self.dr_sigma_rate) ** 2
            dr_gate = self.dr_nsig * np.sqrt(self.dr_var) + env.margin
            dr_discrepancy = abs(self.dE_dr - x_hat[env.cross_track_idx])
            dr_spoof = dr_discrepancy > dr_gate
            if dr_spoof:
                self._dr_alarm = True

        # --- epistemic check: cross-track trustworthy? --------------------------
        # (a) estimator cannot certify a consistent set, (b) cross-track
        # unobservable from trusted sensors, or (c) dead-reckoning disagrees with
        # the reported position (consistent common-mode spoof).
        obs_ok, unobs_w = crosstrack_observable(self.model, trusted, env)
        epistemically_unsafe = (uncertified_run >= uncertified_trip) \
            or (not obs_ok) or dr_spoof
        if epistemically_unsafe:
            # FAIL-SAFE: hold heading on the reference, ignore position estimate
            e = x_hat if xref is None else (x_hat - xref)
            u = -self.K_failsafe @ e
            u[1] = float(np.clip(u[1] + delta_ff, -env.u_steer_max, env.u_steer_max))
            u[0] = float(np.clip(u[0], -env.u_thrust_max, env.u_thrust_max))
            self.n_fail += 1
            if dr_spoof:
                why = (f"dead-reckoning disagrees with reported position "
                       f"({dr_discrepancy:.0f} m > {dr_gate:.0f} m gate): "
                       f"consistent position spoof")
            elif uncertified_run >= uncertified_trip:
                why = "estimator uncertified (no consistent attack-free set)"
            else:
                why = "cross-track unobservable from trusted sensors"
            return u, ActionType.FAILSAFE, dict(unobs_w=unobs_w, escalate=True,
                action=CertifiedAction(ActionType.FAILSAFE,
                    rationale=f"{why}; holding last heading and escalating"))

        # --- predictive recoverability check on the secure estimate ----------
        # one nominal step, then can the backup still contain over the horizon?
        A, B = self.model.A, self.model.B
        x_next = A @ x_hat + B @ u_nom
        worst = _predict_excursion(self.model, self.K_backup, x_next, xref,
                                   delta_ff, env.horizon, env, u_rudder_max=env.u_steer_max)
        budget = env.e_max - env.margin - est_error_bound
        if worst <= budget:
            self.n_nominal += 1
            return u_nom, ActionType.NOMINAL, dict(worst=worst, budget=budget)
        # nominal would risk the envelope -> apply backup's first action
        e = x_hat if xref is None else (x_hat - xref)
        u = -self.K_backup @ e
        u[1] = float(np.clip(u[1] + delta_ff, -env.u_steer_max, env.u_steer_max))
        u[0] = float(np.clip(u[0], -env.u_thrust_max, env.u_thrust_max))
        self.n_backup += 1
        return u, ActionType.BACKUP, dict(worst=worst, budget=budget,
            action=CertifiedAction(ActionType.BACKUP,
                rationale="nominal command risks channel edge; engaging backup "
                          "containment controller"))

    def summary(self):
        tot = self.n_nominal + self.n_backup + self.n_fail
        return dict(nominal=self.n_nominal, backup=self.n_backup,
                    failsafe=self.n_fail, total=tot)
