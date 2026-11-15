"""Replay-rule robustness: simulate clean, B3 and B4 with a perturbed plant (noise level or tyre stiffness),
while the replay tool keeps the nominal model and nominal noise weights.  Ten held-out seeds each."""
import os, sys, json, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import car, scenarios, TEST_SEEDS, BASE, T
from vehicle_model import build_car_model, CarAutopilot, CarParams
from simulate import simulate, SimConfig
import build_kit as BK
import baselines as BL
from tools import GuardianTools

def variant(name):
    if name == "noise_x1.5":
        cm = build_car_model(); cm.sigma = cm.sigma * 1.5; return cm
    if name == "stiffness_-20%":
        return build_car_model(CarParams(Cf=64000.0, Cr=64000.0))
    if name == "stiffness_+20%":
        return build_car_model(CarParams(Cf=96000.0, Cr=96000.0))
    if name == "speed_25":
        return build_car_model(CarParams(U=25.0))
    raise KeyError(name)

if __name__ == "__main__":
    _, _, env = car(); out = {}
    for vname in ("noise_x1.5", "stiffness_-20%", "stiffness_+20%"):
        cm = variant(vname); ap = CarAutopilot(model=build_car_model()); scen = scenarios(cm)
        res = {}
        for sname in ("clean", "B3_slow_gnss", "B4_slow_camera"):
            for sd in TEST_SEEDS:
                r = simulate(cm, scen[sname](), SimConfig(T=T, s=4, seed=sd, **BASE), configs=("secure",),
                             autopilot=ap, rta=True, rta_env=env)
                kd = BK.decision_index(r, cm.p.dt)
                pk = BK.packet(r, build_car_model(), env, kd, f"{vname}|{sname}|{sd}", BK.make_context(sname, "none", sd))
                dec = BL.replay_rule(pk)["decision"]
                g, c = GuardianTools(pk).replay_hypotheses([BL.GNSS_LAT, ["Lane-cam offset"]])["results"]
                res[f"{sname}|{sd}"] = dict(decision=dec, gap=c["consistency_score"] - g["consistency_score"],
                                            kG=g["consistency_score"], kC=c["consistency_score"])
                print(vname, sname, sd, dec, round(res[f"{sname}|{sd}"]["gap"], 4), flush=True)
        out[vname] = res
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "evaluator", "sens_replay.json"), "w"), indent=1)
