"""Independent vertex-based check of the version-2 certificate (no support-function shortcuts)."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('src', '', 'guardian_kit', 'certify'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import itertools, numpy as np
from scipy.optimize import linprog
from vehicle_model import DEG
import stress_cert as SC
d = SC.d; OG, Og = d.OG, d.Og; rng = np.random.default_rng(0)
def vertex(c):
    return linprog(-c, A_ub=OG, b_ub=Og, bounds=[(None, None)] * 4, method="highs").x
V = np.unique(np.round([vertex(rng.normal(size=4)) for _ in range(5000)], 10), axis=0)
print("distinct vertices found", len(V))
Dv = [d.Bxi * a * d.s_E + np.array(w) * d.Wxi for a in (-1, 1) for w in itertools.product((-1, 1), repeat=4)]
print("RPI: max violation over vertices x D corners = %.3e" % max(float(np.max(OG @ (d.Acl @ v + dv) - Og)) for v in V for dv in Dv))
print("Omega in S: max |e| = %.6f (e_cert %.2f)" % (np.max(np.abs(V[:, 0])), d.e_cert))
print("Omega in X_valid: max |psi| = %.3f deg (limit %.1f), max |alpha_r| = %.3f deg (limit %.1f)" %
      (np.max(np.abs(V[:, 1])) / DEG, d.psi_max / DEG, np.max(np.abs(V @ d.cr)) / DEG, d.alpha_max / DEG))
corners = lambda e: [np.array(c) * e for c in itertools.product((-1, 1), repeat=4)]
ub = max(abs(float(d.Kb @ (v - c))) for v in V for e in d.Exi.values() for c in corners(e))
af = max(abs(float(-d.Kb @ (v - c) - d.cf @ v)) for v in V for e in d.Exi.values() for c in corners(e))
print("backup on Omega (+) (-E): max |u_b| = %.4f deg (limit %.1f), max |alpha_f| = %.4f deg (limit %.1f)" % (ub / DEG, d.u_max / DEG, af / DEG, d.alpha_max / DEG))
X0 = [np.array(c) * d.x0_half for c in itertools.product((-1, 1), repeat=4)]
print("X0 corners inside Omega:", all(np.all(OG @ x <= Og) for x in X0))
viol = tested = 0; slipv = 0
for _ in range(4000):
    v = V[rng.integers(len(V))] * rng.uniform(0, 1); u = rng.uniform(-8, 8) * DEG
    if not d.phi(v, u): continue
    tested += 1
    for e in d.Exi.values():
        for ce in corners(e):
            xt = v + ce
            slipv += int(abs(u - d.cf @ xt) > d.alpha_max + 1e-12)
            for cw in itertools.product((-1, 1), repeat=4):
                viol += int(np.any(OG @ (d.Axi @ xt + d.Bxi * u + np.array(cw) * d.Wxi) > Og + 1e-12))
print("predicate: admitted samples", tested, "| corner successors outside Omega:", viol, "| front-slip violations:", slipv)
