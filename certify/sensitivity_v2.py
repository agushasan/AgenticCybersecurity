"""Certified lateral bound versus noise bound and lane-camera accuracy (validity constraints included)."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('src', '', 'guardian_kit', 'certify'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import copy, itertools, json, numpy as np
from search_v2 import cm, SETS, lqr, DEG, CertifiedDesign

def best_bound(model, nsig):
    d = CertifiedDesign(model, Kb=[0.02, 0.5, 0.006, 0.07], e_cert=10.0, u_max=8*DEG, sets=SETS, alpha_max=4*DEG,
                        psi_max=10*DEG, nsig=nsig, build_omega=False)
    best = None
    for qy, qp, qv, qr, r in itertools.product(np.geomspace(1/2.0**2, 1/0.05**2, 9), np.geomspace(1/(8*DEG)**2, 1/(0.3*DEG)**2, 9),
                                               [1/0.05**2, 1/0.2**2, 1/1.0**2], [1/(1*DEG)**2, 1/(5*DEG)**2, 1/(20*DEG)**2],
                                               np.geomspace(1/(32*DEG)**2, 1/(0.5*DEG)**2, 9)):
        Kb = lqr([qy, qp, qv, qr], r)
        Acl = d.Axi - np.outer(d.Bxi, Kb)
        if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1: continue
        d.Kb, d.Acl = Kb, Acl
        d.s_E = max(float(np.abs(Kb) @ e) for e in d.Exi.values())
        if d.s_E >= min(d.u_max, d.alpha_max): continue
        L = d.lemma1_bound()
        if L["psi"] <= d.psi_max and L["alpha_r"] <= d.alpha_max and L["u_backup"] <= d.u_max - d.s_E and L["alpha_f_backup"] <= d.alpha_max - d.s_E:
            if best is None or L["e"] < best: best = L["e"]
    return best

if __name__ == "__main__":
    res = {}
    for label, nsig, cam in (("3sigma", 3.0, 0.10), ("2sigma", 2.0, 0.10), ("3sigma_cam0.05", 3.0, 0.05), ("3sigma_cam0.03", 3.0, 0.03)):
        m = copy.deepcopy(cm); m.sigma = m.sigma.copy(); m.sigma[5] = cam
        res[label] = best_bound(m, nsig); print(label, res[label], flush=True)
    json.dump(res, open(_os.path.join(_ROOT, 'results', 'cert_sensitivity_v2.json'), 'w'), indent=1)
