"""
ship_model.py
=============
Linearised 3-DOF horizontal-plane ship manoeuvring model (error-state form) and
a redundant shipboard navigation/monitoring sensor suite, cast as a discrete-time
LTI system

        x(t+1) = A x(t) + B u(t) + mu(t)
        y_i(t) = C_i x(t) [+ a_i(t)] + psi_i(t)

matching the system/attack model of

    Y. Shoukry, M. Chong, M. Wakaiki, P. Nuzzo, A. Sangiovanni-Vincentelli,
    S. A. Seshia, J. P. Hespanha, P. Tabuada,
    "SMT-Based Observer Design for Cyber-Physical Systems under Sensor Attacks",
    ACM Trans. Cyber-Physical Systems, 2(1), Art. 5, 2018.  (Eqs. 2.1-2.2)

--------------------------------------------------------------------------------
PHYSICAL MODEL
--------------------------------------------------------------------------------
We model a surface vessel transiting a fairway on a nominal straight-line track
(heading psi0 = 0, i.e. due North) at service speed U.  We use the *error state*
relative to that nominal transit, which makes the otherwise nonlinear kinematics
Linear-Time-Invariant -- a standard device in marine guidance/navigation
(see Fossen, "Handbook of Marine Craft Hydrodynamics and Motion Control", 2011,
Ch. 7 for the manoeuvring model and Ch. 10 for line-of-sight track keeping).

State (n = 6), all quantities are deviations from the nominal transit:
    x = [ dN,  dE,  dpsi,  du,  dv,  dr ]^T
      dN   : along-track (North) position deviation            [m]
      dE   : cross-track (East)  position deviation             [m]
      dpsi : heading deviation                                  [rad]
      du   : surge-speed deviation (u - U)                      [m/s]
      dv   : sway-speed deviation                               [m/s]
      dr   : yaw-rate deviation                                 [rad/s]

Kinematics linearised about (psi0 = 0, U):
      dN_dot   = du
      dE_dot   = U*dpsi + dv
      dpsi_dot = dr

Kinetics (linear manoeuvring model, surge decoupled from sway-yaw):
      du_dot = -(1/T_u) du                       + (1/T_u) tau_c        (thrust)
    [ dv_dot ; dr_dot ] = M^{-1}(-N) [dv;dr] + M^{-1} b * delta_c       (rudder)

Control input (m = 2):
      u = [ tau_c, delta_c ]
      tau_c  : commanded surge-speed set-point deviation (thrust channel) [m/s]
      delta_c: commanded rudder angle                                     [rad]

The sway-yaw hydrodynamic coefficients are taken from a Nomoto-equivalent
representation of a medium-size vessel so that the open-loop yaw response has a
realistic time constant (~ tens of seconds).  Parameters are documented inline
and are the single source of truth for reproducibility.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import cont2discrete


# ----------------------------------------------------------------------------- #
#  Physical / hydrodynamic parameters                                           #
# ----------------------------------------------------------------------------- #
@dataclass
class ShipParams:
    # --- transit operating point ------------------------------------------- #
    U: float = 6.0            # nominal surge speed [m/s]  (~11.7 kn)
    # --- surge dynamics ---------------------------------------------------- #
    T_u: float = 25.0         # surge speed time constant [s]
    # --- linear sway-yaw manoeuvring model (Fossen, linear model) ---------- #
    #   M [dv_dot; dr_dot] = -N [dv; dr] + b*delta
    #   M = [[m11, m12],[m21, m22]]   (inertia incl. added mass, normalised)
    #   N = [[n11, n12],[n21, n22]]   (linear damping + Coriolis at speed U)
    m11: float = 1.0          # sway inertia (normalised)
    m12: float = 0.0
    m21: float = 0.0
    m22: float = 1.0          # yaw inertia (normalised)
    n11: float = 0.18         # sway damping
    n12: float = 0.30         # sway-yaw coupling (incl. U-dependent Coriolis)
    n21: float = -0.08        # yaw-sway coupling
    n22: float = 0.12         # yaw damping
    b_v: float = 0.06         # rudder -> sway force
    b_r: float = -0.12        # rudder -> yaw moment (sign: +rudder -> -yaw by convention)
    # --- discretisation ---------------------------------------------------- #
    dt: float = 1.0           # sample period [s] (navigation update rate ~1 Hz)


# ----------------------------------------------------------------------------- #
#  Sensor suite definition                                                      #
# ----------------------------------------------------------------------------- #
# Each *scalar* measurement channel is one "sensor" i in the Shoukry et al.
# formulation (y_i = C_i x + ...).  We model a realistic redundant navigation /
# monitoring suite.  The 1-sigma noise levels are representative civilian-grade
# figures (GNSS ~ few m, radar fix ~ 10-25 m, gyrocompass ~ 0.2-0.5 deg,
# magnetic ~ 1 deg, MEMS rate gyro, Doppler/EM speed log).
#
# "device" groups channels that share a physical/common-mode failure surface:
# both GNSS position channels of one receiver spoof together; a GNSS *spoofer*
# captures ALL civilian GNSS receivers simultaneously (common service signals),
# which is the crux of the maritime spoofing threat.
DEG = np.pi / 180.0

@dataclass
class Sensor:
    name: str
    C: np.ndarray              # 1 x n measurement row
    sigma: float               # 1-sigma measurement noise [state units]
    device: str                # physical device / common-mode group
    gnss: bool = False         # True if derived from GNSS (spoofable in common mode)


def build_sensor_suite(n: int = 6) -> list[Sensor]:
    def row(idx, val=1.0):
        c = np.zeros(n); c[idx] = val; return c
    # state index map: 0:dN 1:dE 2:dpsi 3:du 4:dv 5:dr
    S = [
        # --- GNSS receiver A (independent antenna / constellation mix) ------ #
        Sensor("GNSS-A North", row(0), sigma=3.0,       device="GNSS_A", gnss=True),
        Sensor("GNSS-A East",  row(1), sigma=3.0,       device="GNSS_A", gnss=True),
        # --- GNSS receiver B ------------------------------------------------ #
        Sensor("GNSS-B North", row(0), sigma=3.5,       device="GNSS_B", gnss=True),
        Sensor("GNSS-B East",  row(1), sigma=3.5,       device="GNSS_B", gnss=True),
        # --- Radar positioning vs charted landmarks (GNSS-independent) ------ #
        Sensor("Radar North",  row(0), sigma=18.0,      device="RADAR",  gnss=False),
        Sensor("Radar East",   row(1), sigma=18.0,      device="RADAR",  gnss=False),
        # --- Heading references (3, all GNSS-independent) ------------------- #
        Sensor("Gyrocompass",  row(2), sigma=0.3*DEG,   device="GYROC",  gnss=False),
        Sensor("Mag compass",  row(2), sigma=1.0*DEG,   device="MAG",    gnss=False),
        Sensor("FOG heading",  row(2), sigma=0.2*DEG,   device="FOG",    gnss=False),
        # --- Inertial rate gyro --------------------------------------------- #
        Sensor("Rate gyro r",  row(5), sigma=0.05*DEG,  device="IMU",    gnss=False),
        # --- Dual-axis Doppler speed log (surge + sway) -------------------- #
        Sensor("Speed log u",  row(3), sigma=0.1,       device="LOG",    gnss=False),
        Sensor("Speed log v",  row(4), sigma=0.1,       device="LOG",    gnss=False),
    ]
    return S


# ----------------------------------------------------------------------------- #
#  Assembled discrete-time LTI model                                            #
# ----------------------------------------------------------------------------- #
@dataclass
class ShipModel:
    p: ShipParams
    sensors: list[Sensor]
    n: int
    m: int
    Ac: np.ndarray
    Bc: np.ndarray
    A: np.ndarray            # discrete-time A
    B: np.ndarray            # discrete-time B
    C: np.ndarray            # p x n  stacked sensor matrix
    sigma: np.ndarray        # p vector of 1-sigma noise
    names: list[str]
    devices: list[str]
    gnss_mask: np.ndarray

    @property
    def psensors(self) -> int:
        return self.C.shape[0]


def build_ship_model(p: ShipParams | None = None) -> ShipModel:
    if p is None:
        p = ShipParams()
    n, m = 6, 2

    # --- continuous-time A ---------------------------------------------------
    M = np.array([[p.m11, p.m12], [p.m21, p.m22]])
    N = np.array([[p.n11, p.n12], [p.n21, p.n22]])
    bvr = np.array([p.b_v, p.b_r])
    Minv = np.linalg.inv(M)
    A_vr = -Minv @ N                      # [dv;dr] dynamics matrix (2x2)
    B_vr = Minv @ bvr                     # rudder -> [dv;dr]

    Ac = np.zeros((n, n))
    # kinematics
    Ac[0, 3] = 1.0                        # dN_dot = du
    Ac[1, 2] = p.U                        # dE_dot = U dpsi + dv
    Ac[1, 4] = 1.0
    Ac[2, 5] = 1.0                        # dpsi_dot = dr
    # surge
    Ac[3, 3] = -1.0 / p.T_u              # du_dot = -(1/T_u) du + ...
    # sway-yaw
    Ac[4, 4] = A_vr[0, 0]; Ac[4, 5] = A_vr[0, 1]
    Ac[5, 4] = A_vr[1, 0]; Ac[5, 5] = A_vr[1, 1]

    Bc = np.zeros((n, m))
    Bc[3, 0] = 1.0 / p.T_u               # thrust channel
    Bc[4, 1] = B_vr[0]                    # rudder -> sway
    Bc[5, 1] = B_vr[1]                    # rudder -> yaw

    # --- discretise (zero-order hold) ---------------------------------------
    Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, np.eye(n), np.zeros((n, m))),
                                    p.dt, method="zoh")

    sensors = build_sensor_suite(n)
    C = np.vstack([s.C for s in sensors])
    sigma = np.array([s.sigma for s in sensors])
    names = [s.name for s in sensors]
    devices = [s.device for s in sensors]
    gnss_mask = np.array([s.gnss for s in sensors], dtype=bool)

    return ShipModel(p=p, sensors=sensors, n=n, m=m, Ac=Ac, Bc=Bc,
                     A=Ad, B=Bd, C=C, sigma=sigma, names=names,
                     devices=devices, gnss_mask=gnss_mask)


# ----------------------------------------------------------------------------- #
#  Quick self-test / sanity checks                                              #
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    sm = build_ship_model()
    print("=== Continuous A ===\n", sm.Ac)
    print("=== Discrete A (dt=%.2fs) ===\n" % sm.p.dt, sm.A)
    print("=== Discrete B ===\n", sm.B)
    print("\nOpen-loop continuous eigenvalues:", np.linalg.eigvals(sm.Ac))
    print("Open-loop discrete  eigenvalues:", np.linalg.eigvals(sm.A))
    print("|eig(A_d)| =", np.abs(np.linalg.eigvals(sm.A)))
    # full observability of (A, C)
    from numpy.linalg import matrix_rank
    O = np.vstack([sm.C @ np.linalg.matrix_power(sm.A, k) for k in range(sm.n)])
    print("\nFull-suite observability matrix rank = %d (n=%d)"
          % (matrix_rank(O), sm.n))
    print("\nSensor suite:")
    for i, s in enumerate(sm.sensors):
        print(f"  [{i:2d}] {s.name:14s} dev={s.device:7s} gnss={str(s.gnss):5s} "
              f"sigma={s.sigma:.4g}")
