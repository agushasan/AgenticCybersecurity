"""
route.py
========
Predefined maneuvering trajectory (a curving fairway centreline) and the
guidance reference it induces, for the trajectory-tracking study (Example 2).

To remain rigorously inside the validity of the error-state LTI model
(linearised about a due-North transit, small heading deviations), the planned
route is expressed as a smooth cross-track centreline  E_ref(N)  with GENTLE
bends, so the path-tangent heading
        psi_ref(N) = arctan(dE_ref/dN)
stays small (|psi_ref| <~ 20 deg).  The vessel performs line-of-sight (LOS)
track-keeping of this centreline with a turn-rate feed-forward, so it genuinely
manoeuvres (develops yaw rate and sway through the bends) while every signal
remains within the small-angle regime.  This is the standard marine
guidance-navigation setup (Fossen 2011, Ch. 10), cast so that the secure
state-estimation machinery applies unchanged.

The route is built from a sequence of (length, course) segments; the course
profile is rate-limited (a ship cannot turn instantaneously) and integrated to
give the centreline.  The induced reference is returned as functions of the
along-track coordinate p (= North distance travelled):
        E_ref(p)      desired cross-track position [m]
        psi_ref(p)    path-tangent heading deviation [rad]
        r_ref(p)      reference yaw rate  = U * dpsi_ref/dp  [rad/s]
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

DEG = np.pi / 180.0


@dataclass
class ChannelPath:
    segments: list          # list of (length_m, course_deg)
    U: float                # service speed [m/s]
    dt: float = 1.0
    turn_rate_deg_s: float = 1.1   # max course change rate [deg/s]
    dp: float = 2.0         # along-track resolution [m]

    def __post_init__(self):
        self._build()

    def _build(self):
        U, dp = self.U, self.dp
        total = sum(L for L, _ in self.segments)
        pgrid = np.arange(0.0, total + dp, dp)
        # desired course at each along-track station (piecewise constant)
        course_des = np.zeros_like(pgrid)
        edges = np.cumsum([L for L, _ in self.segments])
        for k, (L, c) in enumerate(self.segments):
            lo = 0 if k == 0 else edges[k - 1]
            course_des[(pgrid >= lo) & (pgrid <= edges[k])] = c * DEG
        # rate-limit the course vs TIME: max d(course)/dp = (rate per s)/U
        max_dpsi_dp = (self.turn_rate_deg_s * DEG) / U
        psi = np.zeros_like(pgrid)
        for i in range(1, len(pgrid)):
            d = course_des[i] - psi[i - 1]
            step = np.clip(d, -max_dpsi_dp * dp, max_dpsi_dp * dp)
            psi[i] = psi[i - 1] + step
        # centreline cross-track position E_ref = integral tan(psi) ~ integral psi
        E = np.concatenate([[0.0], np.cumsum(0.5 * (np.tan(psi[1:]) +
                                                    np.tan(psi[:-1])) * dp)])
        # reference yaw rate r_ref = U * dpsi/dp
        dpsi_dp = np.gradient(psi, pgrid)
        r_ref = U * dpsi_dp

        self.pgrid = pgrid
        self.E_ref_arr = E
        self.psi_ref_arr = psi
        self.r_ref_arr = r_ref
        self.total_length = total
        self.max_course_deg = float(np.max(np.abs(psi)) / DEG)

    def ref(self, p):
        """Return (E_ref, psi_ref, r_ref) at along-track coordinate p."""
        p = float(np.clip(p, self.pgrid[0], self.pgrid[-1]))
        E = np.interp(p, self.pgrid, self.E_ref_arr)
        psi = np.interp(p, self.pgrid, self.psi_ref_arr)
        r = np.interp(p, self.pgrid, self.r_ref_arr)
        return E, psi, r


def default_fairway(U, dt=1.0):
    """
    A realistic buoyed fairway with an S-bend then a dog-leg, total ~3 km,
    maximum course deviation ~18-20 deg (well within the linear regime).
    """
    segs = [
        (500.0,  0.0),    # initial straight leg, due North
        (450.0, 20.0),    # turn to starboard (NE)
        (500.0, -12.0),   # ease back to port (channel S-bend)
        (450.0, 15.0),    # dog-leg to starboard
        (700.0,  0.0),    # final straight run-in
    ]
    return ChannelPath(segs, U=U, dt=dt)


def steady_turn_feedforward(ship):
    """
    Feed-forward gain G_ff mapping a desired yaw rate r_ref to the rudder angle
    that sustains it in steady state (v_dot = r_dot = 0), from the continuous
    sway-yaw model.  Returns (G_ff_delta, G_ff_v) so that
        delta_ff = G_ff_delta * r_ref,   v_steady = G_ff_v * r_ref.
    """
    Ac, Bc = ship.Ac, ship.Bc
    a_vv, a_vr = Ac[4, 4], Ac[4, 5]
    a_rv, a_rr = Ac[5, 4], Ac[5, 5]
    b_v, b_r = Bc[4, 1], Bc[5, 1]
    # [a_vv b_v; a_rv b_r][v; delta] = -r_ref [a_vr; a_rr]
    M = np.array([[a_vv, b_v], [a_rv, b_r]])
    rhs = -np.array([a_vr, a_rr])
    sol = np.linalg.solve(M, rhs)       # [v_per_r, delta_per_r]
    return float(sol[1]), float(sol[0])


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ship_model import build_ship_model
    sm = build_ship_model()
    path = default_fairway(sm.p.U)
    print(f"fairway length = {path.total_length:.0f} m, "
          f"max course = {path.max_course_deg:.1f} deg")
    gff, gv = steady_turn_feedforward(sm)
    print(f"feed-forward: delta_ff = {gff:.2f} * r_ref,  v_ss = {gv:.2f} * r_ref")
    for p in [0, 400, 700, 1200, 1700, 2500]:
        E, psi, r = path.ref(p)
        print(f"  p={p:5d} m: E_ref={E:7.1f} m  psi_ref={psi/DEG:6.2f} deg  "
              f"r_ref={r/DEG:6.3f} deg/s")
