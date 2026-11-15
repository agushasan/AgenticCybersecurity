"""Vessel: search for a certificate with the GNSS-free predictor at 3 sigma (rudder limit, e_max = 70 m)."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('src', '', 'guardian_kit', 'certify'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import itertools, numpy as np
from scipy.linalg import solve_discrete_are
from ship_model import build_ship_model
from certified_filter import predictor_kalman_gain, reach_box, _tail_constants, _ninf, IX
m = build_ship_model(); full = list(range(m.psensors))
T = [i for i in full if not m.gnss_mask[i]]
ps = 0.02; wsig = np.array([ps * 2, ps * 2, ps * 0.2, ps, ps, ps * 0.2]); W = 3 * wsig[IX]
As, Bs = m.A[np.ix_(IX, IX)], m.B[IX, 1]; UMAX = 0.61; EMAX = 70.0; x0 = np.array([15.0, 0, 0, 0])
def lqr(q, r):
    P = solve_discrete_are(As, Bs[:, None], np.diag(q), np.array([[r]])); return (Bs @ P @ As) / (r + Bs @ P @ Bs)
best = None
for qs in (10.0, 30.0, 100.0, 300.0):
    L = predictor_kalman_gain(m.A, m.C[T], m.sigma[T], wsig, qs); F = m.A - L @ m.C[T]
    E, _ = reach_box(F, np.hstack([np.diag(3 * wsig), -L @ np.diag(3 * m.sigma[T])]), 3 * m.sigma[T].max() * np.ones(m.n) * 0 + np.abs(np.linalg.pinv(m.C[T])) @ (3 * m.sigma[T]))
    Ex = E[IX]; n_ok = 0
    for qy, qp, qv, qr, r in itertools.product(np.geomspace(1e-6, 1e-1, 8), np.geomspace(1e-1, 1e4, 8), [1e-3, 1.0, 1e2], [1e-1, 1e2, 1e4], np.geomspace(1e-2, 1e4, 8)):
        Kb = lqr([qy, qp, qv, qr], r); Acl = As - np.outer(Bs, Kb)
        if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1: continue
        sE = float(np.abs(Kb) @ Ex)
        if sE >= UMAX: continue
        GD = np.hstack([Bs[:, None] * sE, np.diag(W)]); H = np.vstack([[1.0, 0, 0, 0], Kb])
        sup = np.abs(H @ x0); rad = np.zeros(2); M = GD.copy(); c = x0.copy()
        for _ in range(1500):
            c = Acl @ c; rad += np.abs(H @ M).sum(axis=1); M = Acl @ M
            sup = np.maximum(sup, np.abs(H @ c) + rad)
        if sup[1] + sE <= UMAX:
            n_ok += 1
            if best is None or sup[0] < best[0]: best = (sup[0], qs, Ex[0], sE)
    print(f"qs {qs}: GNSS-free error bound {Ex[0]:.1f} m, rudder-feasible backups {n_ok}, best lateral bound so far", None if best is None else round(best[0], 1), flush=True)
print("RESULT:", "no certificate within 70 m" if best is None or best[0] > EMAX else f"certificate {best[0]:.1f} m")
