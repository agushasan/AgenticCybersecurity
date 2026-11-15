"""Verified-filter campaign (version 2): certificate with validity region, hypothesis-selected observer bank.

Scenarios: clean, B1 (fast GNSS), B3 (slow GNSS) run on the default GNSS-free predictor, which no GNSS attack
can contaminate; B4 (slow camera) switches to the camera-free predictor at the replay rule's decision step of
the matching benchmark episode; B2 (GNSS and camera) lies outside the single-group threat model.
Seven proposal policies from 16 s, ten seeds, noise clipped at 3 sigma.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ('src', '', 'guardian_kit', 'certify'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import json, math, time, numpy as np
from vehicle_model import DEG
from certified_filter import CertifiedDesign, IX
from simulate import simulate, SimConfig
from common import car, scenarios
from run_car import BASE, T

cm, ap, env = car(); full = list(range(12))
SETS = {"GNSS-free": [i for i in full if i not in (0, 1, 2, 3)], "camera-free": [i for i in full if i not in (5, 8)]}
KB = json.load(open(_os.path.join(_ROOT, 'results', 'cert_search_v2.json')))["4.0/10.0"][1]
ECERT = 1.70
d = CertifiedDesign(cm, Kb=KB, e_cert=ECERT, u_max=8 * DEG, sets=SETS, alpha_max=4 * DEG, psi_max=10 * DEG)
dmax = env.u_steer_max; T_START = 16.0; GRID = np.linspace(-dmax, dmax, 321)
LAB = json.load(open(_os.path.join(_ROOT, 'guardian_kit', 'data', 'evaluator', 'labels.json')))
DEC_STEP = {v['seed']: v['decision_step'] for v in LAB.values()
            if v['split'] == 'test' and v['scenario'] == 'B4_slow_camera' and v['context'] == 'none'}

def policies(seed):
    rng = np.random.default_rng(10_000 + seed)
    def greedy(t, x):                      # filter-aware: largest admissible steering toward the nearer edge
        xi = x[IX]; side = 1.0 if xi[0] >= 0 else -1.0
        ok = GRID[d.phi_batch(xi, GRID)]
        if ok.size == 0: return np.array([0.0, side * dmax])
        return np.array([0.0, ok.max() if side > 0 else ok.min()])
    state = {"u": dmax}
    def bang(t, x):
        if rng.random() < 0.2: state["u"] = -state["u"]
        return np.array([0.0, state["u"]])
    return {
        "hard_over_left": lambda t, x: np.array([0.0, dmax]),
        "hard_over_right": lambda t, x: np.array([0.0, -dmax]),
        "weave_0p5Hz": lambda t, x: np.array([0.0, dmax * math.sin(2 * math.pi * 0.5 * t)]),
        "push_to_edge": lambda t, x: np.array([0.0, float(np.clip(2.0 * np.sign(x[1] + 1e-9) * dmax, -dmax, dmax))]),
        "random_uniform": lambda t, x: np.array([0.0, rng.uniform(-dmax, dmax)]),
        "random_bang_bang": bang,
        "filter_aware_greedy": greedy,
    }

SCEN = ("clean", "B1_fast_gnss", "B3_slow_gnss", "B4_slow_camera", "B2_all_anchors")

def run_one(sname, seed, pname, fn):
    prop = lambda t, x, fn=fn: fn(t, x) if t >= T_START else None
    sw = (DEC_STEP[seed], "camera-free") if sname == "B4_slow_camera" else None
    r = simulate(cm, scenarios(cm)[sname](), SimConfig(T=T, s=4, seed=seed, noise_clip=3.0, **BASE), configs=("secure",),
                 autopilot=ap, rta=False, proposal_fn=prop, cert_design=d, cert_switch=sw)
    return r, sw

def summarize(sname, seed, pname, r, sw):
    ct = r.cert_trace; e = np.abs(r.x_true["secure"][:, 1]); m = ct["mode"]; k0 = int(T_START / 0.1)
    clean = ct["trusted_clean"]; cov = clean.copy()            # steps whose used predictor is attack-free
    first_bad = int(np.argmax(~clean)) if (~clean).any() else None
    return dict(scenario=sname, seed=seed, policy=pname, switch_step=None if sw is None else sw[0],
                peak_m=float(e.max()), peak_after_start_m=float(e[k0:].max()),
                admitted_after_start=float(np.mean(m[k0:] == 0)),
                contaminated_steps=int(np.sum(~clean)), first_contaminated_step=first_bad,
                in_omega_all=bool(ct["in_omega"].all()), in_omega_on_clean_steps=bool(ct["in_omega"][cov].all()),
                eta_in_E_on_clean_steps=bool(ct["in_E"][cov].all()), valid_all=bool(ct["valid"].all()),
                valid_on_clean_steps=bool(ct["valid"][cov].all()),
                in_omega_after_switch=None if sw is None else bool(ct["in_omega"][sw[0]:].all()),
                eta_in_E_after_switch=None if sw is None else bool(ct["in_E"][sw[0]:].all()),
                max_abs_steer_deg=float(np.max(np.abs(r.u["secure"][:, 1])) / DEG),
                max_abs_eta_dE=float(np.nanmax(np.abs(ct["eta"][:, 0]))))

def main(seeds=range(1, 11), scen=SCEN, tag=""):
    rows, traces = [], {}
    for sname in scen:
        for seed in seeds:
            for pname, fn in policies(seed).items():
                t0 = time.time(); r, sw = run_one(sname, seed, pname, fn)
                rows.append(summarize(sname, seed, pname, r, sw))
                if pname == "filter_aware_greedy":
                    traces[f"{sname}|{seed}"] = r.x_true["secure"][:, 1].copy()
                print(sname, seed, pname, {k: rows[-1][k] for k in ("peak_m", "admitted_after_start", "contaminated_steps", "in_omega_all", "eta_in_E_on_clean_steps", "valid_all")}, f"{time.time()-t0:.0f}s", flush=True)
    os = _os
    out = dict(design=d.export(), runs=rows)
    json.dump(out, open(os.path.join(_ROOT, 'results', f'cert_stress{tag}.json'), 'w'), indent=1)
    np.savez(os.path.join(_ROOT, 'results', f'cert_traces{tag}.npz'), label=np.array(list(traces)), e=np.array(list(traces.values())),
             t=np.arange(T) * 0.1, e_cert=ECERT, t_start=T_START,
             switch=np.array([DEC_STEP.get(int(k.split('|')[1]), -1) if k.startswith('B4') else -1 for k in traces]))

if __name__ == "__main__":
    import argparse
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--scen", default=",".join(SCEN)); ap_.add_argument("--tag", default="")
    a = ap_.parse_args(); main(scen=tuple(a.scen.split(",")), tag=a.tag)
