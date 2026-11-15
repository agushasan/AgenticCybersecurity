"""
vehicle_model.py
================
Autonomous ground vehicle (passenger car) for the second case study.

A linear single-track ("bicycle") model at constant longitudinal speed U, written
in lane coordinates as an error state about the lane centre.  Under the usual
small-angle linearization its state has EXACTLY the structure of the vessel
model in ship_model.py,

    x = [ dX, dY, dpsi, du, dv, dr ]^T
      dX   : along-lane position deviation                  [m]
      dY   : lateral offset from the lane centre            [m]
      dpsi : heading relative to the lane                   [rad]
      du   : longitudinal speed deviation                   [m/s]
      dv   : lateral (body) velocity                        [m/s]
      dr   : yaw rate                                       [rad/s]

    dX_dot = du,   dY_dot = U dpsi + dv,   dpsi_dot = dr,
    du_dot = -(1/T_u) du + (1/T_u) a_c,
    [dv_dot; dr_dot] = A_vr [dv; dr] + b_vr * delta_f,

with front-wheel steering angle delta_f and the classical linear tyre model
(cornering stiffnesses C_f, C_r).  The model object reuses the ShipModel
container, so the secure estimator, the runtime-assurance filter and the
certificate apply unchanged; only parameters, sensors and the controller differ.

Sensor suite (p = 12), mirroring the vessel suite channel for channel:
    two RTK-GNSS receivers (along/lateral each)        common-mode spoofable
    LiDAR map-matching fix (along)                     GNSS-independent anchor
    lane camera lateral offset                         GNSS-independent anchor
    three heading references (magnetometer, LiDAR map heading, camera heading)
    IMU yaw-rate gyro, wheel odometry (u), optical velocity sensor (v)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are

from ship_model import ShipModel, Sensor

DEG = np.pi / 180.0


@dataclass
class CarParams:
    U: float = 20.0          # longitudinal speed [m/s] (72 km/h)
    T_u: float = 5.0         # speed-loop time constant [s]
    mass: float = 1500.0     # [kg]
    Iz: float = 2500.0       # yaw inertia [kg m^2]
    lf: float = 1.2          # CoG to front axle [m]
    lr: float = 1.4          # CoG to rear axle  [m]
    Cf: float = 80000.0      # front axle cornering stiffness [N/rad]
    Cr: float = 80000.0      # rear axle cornering stiffness  [N/rad]
    dt: float = 0.1          # navigation/control rate 10 Hz [s]


def build_car_sensor_suite(n: int = 6) -> list[Sensor]:
    def row(idx):
        c = np.zeros(n); c[idx] = 1.0; return c
    return [
        Sensor("GNSS-A along",   row(0), sigma=0.10,      device="GNSS_A", gnss=True),
        Sensor("GNSS-A lateral", row(1), sigma=0.10,      device="GNSS_A", gnss=True),
        Sensor("GNSS-B along",   row(0), sigma=0.12,      device="GNSS_B", gnss=True),
        Sensor("GNSS-B lateral", row(1), sigma=0.12,      device="GNSS_B", gnss=True),
        Sensor("LiDAR-map along", row(0), sigma=0.20,     device="LIDAR",  gnss=False),
        Sensor("Lane-cam offset", row(1), sigma=0.10,     device="CAM",    gnss=False),
        Sensor("Magnetometer",   row(2), sigma=2.0*DEG,   device="MAG",    gnss=False),
        Sensor("LiDAR heading",  row(2), sigma=0.3*DEG,   device="LIDAR",  gnss=False),
        Sensor("Lane-cam heading", row(2), sigma=0.5*DEG, device="CAM",    gnss=False),
        Sensor("IMU yaw rate",   row(5), sigma=0.05*DEG,  device="IMU",    gnss=False),
        Sensor("Wheel odometry", row(3), sigma=0.05,      device="ODO",    gnss=False),
        Sensor("Optical v",      row(4), sigma=0.05,      device="OPT",    gnss=False),
    ]


def build_car_model(p: CarParams | None = None) -> ShipModel:
    if p is None:
        p = CarParams()
    n, m = 6, 2
    U = p.U
    a_vv = -(p.Cf + p.Cr) / (p.mass * U)
    a_vr = (-p.lf * p.Cf + p.lr * p.Cr) / (p.mass * U) - U
    a_rv = (-p.lf * p.Cf + p.lr * p.Cr) / (p.Iz * U)
    a_rr = -(p.lf ** 2 * p.Cf + p.lr ** 2 * p.Cr) / (p.Iz * U)
    b_v = p.Cf / p.mass
    b_r = p.lf * p.Cf / p.Iz

    Ac = np.zeros((n, n))
    Ac[0, 3] = 1.0
    Ac[1, 2] = U; Ac[1, 4] = 1.0
    Ac[2, 5] = 1.0
    Ac[3, 3] = -1.0 / p.T_u
    Ac[4, 4] = a_vv; Ac[4, 5] = a_vr
    Ac[5, 4] = a_rv; Ac[5, 5] = a_rr
    Bc = np.zeros((n, m))
    Bc[3, 0] = 1.0 / p.T_u
    Bc[4, 1] = b_v
    Bc[5, 1] = b_r
    Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, np.eye(n), np.zeros((n, m))),
                                    p.dt, method="zoh")
    sensors = build_car_sensor_suite(n)
    return ShipModel(p=p, sensors=sensors, n=n, m=m, Ac=Ac, Bc=Bc, A=Ad, B=Bd,
                     C=np.vstack([s.C for s in sensors]),
                     sigma=np.array([s.sigma for s in sensors]),
                     names=[s.name for s in sensors],
                     devices=[s.device for s in sensors],
                     gnss_mask=np.array([s.gnss for s in sensors], dtype=bool))


@dataclass
class CarAutopilot:
    """
    Lane-keeping controller u = -K x designed by discrete LQR on the lateral
    subsystem (dY, dpsi, dv, dr) with a proportional speed loop on du.  It has the
    same interface as the vessel Autopilot (K, delta_max, tau_max, command).
    """
    model: object = None
    q_lat: tuple = (1 / 0.15 ** 2, 1 / (1.0 * DEG) ** 2, 1 / 0.5 ** 2, 1 / (5 * DEG) ** 2)
    r_steer: float = 1 / (2.0 * DEG) ** 2
    Ku: float = 0.8
    delta_max: float = 8.0 * DEG     # road-wheel steering limit at highway speed
    tau_max: float = 2.0
    K: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        A, B = self.model.A, self.model.B
        IX = [1, 2, 4, 5]
        As = A[np.ix_(IX, IX)]; Bs = B[IX, 1:2]
        Q = np.diag(self.q_lat); R = np.array([[self.r_steer]])
        P = solve_discrete_are(As, Bs, Q, R)
        Klat = np.linalg.solve(R + Bs.T @ P @ Bs, Bs.T @ P @ As)
        K = np.zeros((2, 6))
        K[0, 3] = self.Ku
        K[1, IX] = Klat.ravel()
        self.K = K

    def command(self, xhat, xref=None, delta_ff=0.0):
        e = xhat if xref is None else (xhat - xref)
        u = -self.K @ e
        u[1] += delta_ff
        u[0] = float(np.clip(u[0], -self.tau_max, self.tau_max))
        u[1] = float(np.clip(u[1], -self.delta_max, self.delta_max))
        return u


def car_envelope():
    """Safe set and filter parameters for highway lane keeping."""
    from safety_filter import SafetyEnvelope
    return SafetyEnvelope(e_max=0.8,             # body stays inside a 3.5 m lane
                          horizon=25, margin=0.10, backup_gain_scale=1.6,
                          est_error_bound=0.20, u_steer_max=8.0 * DEG,
                          u_thrust_max=2.0, dr_sigma_rate=0.20,
                          dr_nsig=5.0, dr_warmup=100, dr_window=50)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    cm = build_car_model()
    print("continuous eig:", np.round(np.linalg.eigvals(cm.Ac), 3))
    ap = CarAutopilot(model=cm)
    print("K lateral:", np.round(ap.K[1], 4))
    Acl = cm.A - cm.B @ ap.K
    print("closed-loop |eig| max:", np.max(np.abs(np.linalg.eigvals(Acl))))
    from secure_estimation import _obsv
    print("rank O:", np.linalg.matrix_rank(_obsv(cm.A, cm.C)))
