"""Grid search over LQ backup gains for the certificate with validity constraints (car, 3 sigma)."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('src', '', 'guardian_kit', 'certify'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import itertools, json, numpy as np
from scipy.linalg import solve_discrete_are
from vehicle_model import build_car_model, DEG
from certified_filter import CertifiedDesign, IX
cm = build_car_model(); full = list(range(12))
SETS = {"GNSS-free": [i for i in full if i not in (0, 1, 2, 3)], "camera-free": [i for i in full if i not in (5, 8)]}
As = cm.A[np.ix_(IX, IX)]; Bs = cm.B[IX, 1:2]
def lqr(q, r):
    P = solve_discrete_are(As, Bs, np.diag(q), np.array([[r]]))
    return np.linalg.solve(r + Bs.T @ P @ Bs, Bs.T @ P @ As).ravel()

def run(alpha_max=4.0, u_max=8.0, psi_max=5.0, qs=30.0):
    best = None; n_ok = 0
    for qy, qp, qv, qr, r in itertools.product(np.geomspace(1/2.0**2, 1/0.05**2, 7), np.geomspace(1/(8*DEG)**2, 1/(0.3*DEG)**2, 7),
                                               [1/0.05**2, 1/0.2**2, 1/1.0**2], [1/(1*DEG)**2, 1/(5*DEG)**2, 1/(20*DEG)**2],
                                               np.geomspace(1/(32*DEG)**2, 1/(0.5*DEG)**2, 7)):
        Kb = lqr([qy, qp, qv, qr], r)
        try:
            d = CertifiedDesign(cm, Kb=Kb, e_cert=10.0, u_max=u_max*DEG, sets=SETS, alpha_max=alpha_max*DEG,
                                psi_max=psi_max*DEG, qs=qs, build_omega=False)
        except ValueError:
            continue
        L = d.lemma1
        if L["psi"] <= d.psi_max and L["alpha_r"] <= d.alpha_max and L["u_backup"] <= d.u_max - d.s_E and L["alpha_f_backup"] <= d.alpha_max - d.s_E:
            n_ok += 1
            if best is None or L["e"] < best[0]:
                best = (L["e"], Kb.tolist(), dict(L), d.s_E / DEG)
    return best, n_ok

if __name__ == "__main__":
    out = {}
    for am, pm in ((4.0, 10.0), (4.0, 15.0), (5.0, 10.0), (5.0, 15.0)):
        b, n = run(alpha_max=am, psi_max=pm)
        out[f"{am}/{pm}"] = b; print(f"alpha_max {am} deg psi_max {pm} deg: feasible {n}, best Lemma-1 lateral bound", None if b is None else round(b[0], 3), "Kb", None if b is None else np.round(b[1], 4), "s_E", None if b is None else round(b[3], 2), flush=True)
    json.dump({str(k): v for k, v in out.items()}, open(_os.path.join(_ROOT, 'results', 'cert_search_v2.json'), 'w'), indent=1)
