"""
build_kit.py  —  Export recorded evidence packets for the Guardian evaluation.
==============================================================================
Runs the car scenarios with the deployed Tier-0 stack (secure estimator plus
runtime monitor) for the held-out and development seeds, and writes

  data/packets/<split>/<episode_id>.json   agent-visible evidence up to the
                                           decision time (no future samples)
  data/evaluator/labels.json               evaluator-only ground truth
  data/evaluator/admissible.json           predefined admissible decisions

Decision time: 1.0 s after the first Tier-0 event (a channel flag, a monitor
fail-safe or an uncertified estimate) after the attack-free warm-up, or 30 s
if no event occurs.  Run from the repository root:  python3 guardian_kit/build_kit.py
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import (car, scenarios, model_hash, episode_id, OUT, TEST_SEEDS, DEV_SEEDS,
                    CONTEXTS, BASE, T, T0)
from simulate import simulate, SimConfig
from context_messages import make_context

WARMUP_S = 10.0
LAT = ["GNSS-A lateral", "GNSS-B lateral", "Lane-cam offset"]
HEAD = ["Magnetometer", "LiDAR heading", "Lane-cam heading"]


def decision_index(r, dt):
    att = r.attacked; mode = r.rta_mode["secure"]
    k0 = int(WARMUP_S / dt)
    ev = np.where(att[k0:].any(axis=1) | (mode[k0:] == 2))[0]
    kd = (k0 + int(ev[0]) + int(1.0 / dt)) if ev.size else int(30.0 / dt)
    return min(kd, len(mode) - 1)


def packet(r, cm, env, kd, eid, ctx):
    dt = cm.p.dt; names = list(cm.names); k1 = kd + 1
    R, E = r.smt_trace["R"], r.smt_trace["env"]
    ratio = np.where(np.isfinite(R) & (E > 0), R / np.where(E > 0, E, 1), np.nan)
    att, mode = r.attacked, r.rta_mode["secure"]
    events = []
    for k in range(1, k1):
        for i in np.where(att[k] != att[k - 1])[0]:
            events.append(dict(t=round(k * dt, 1), event=("flagged" if att[k, i] else "cleared"),
                               channel=names[i]))
        if mode[k] != mode[k - 1]:
            events.append(dict(t=round(k * dt, 1), event="monitor_mode",
                               value=["nominal", "backup", "failsafe"][int(mode[k])]))
    ix = list(range(len(names)))            # every channel the vehicle receives
    series = dict(
        t=[round(k * dt, 1) for k in range(k1)],
        residual_ratio={names[i]: [None if not np.isfinite(v) else round(float(v), 4)
                                   for v in ratio[:k1, i]] for i in range(len(names))},
        measurement={names[i]: [round(float(v), 5) for v in r.y_att[:k1, i]] for i in ix},
        flags={names[i]: [int(v) for v in att[:k1, i]] for i in range(len(names))},
        monitor_mode=[int(v) for v in mode[:k1]],
        applied_input={"speed_cmd": [round(float(v), 6) for v in r.u["secure"][:k1, 0]],
                       "steering_rad": [round(float(v), 6) for v in r.u["secure"][:k1, 1]]},
        estimate_lateral_offset=[round(float(v), 4) for v in r.x_hat["secure"][:k1, 1]],
        estimate_heading=[round(float(v), 6) for v in r.x_hat["secure"][:k1, 2]],
        dr_discrepancy=[None if not np.isfinite(v) else round(float(v), 4)
                        for v in r.dr_trace["discrepancy"][:k1]],
        dr_gate=[None if not np.isfinite(v) else round(float(v), 4) for v in r.dr_trace["gate"][:k1]],
    )
    return dict(
        episode_id=eid, kit_version="v1", decision_time_s=round(kd * dt, 1),
        model_card=dict(model_hash=model_hash(cm), platform="automated car, highway lane keeping",
                        speed_mps=cm.p.U, dt_s=dt, e_max_m=env.e_max, lane_line_m=1.75,
                        channels=[dict(name=s.name, sigma=float(s.sigma), device=s.device, gnss=bool(s.gnss))
                                  for s in cm.sensors],
                        notes="residual_ratio is the windowed residual under the all-sensors "
                              "hypothesis divided by its detection envelope (>1 fails); "
                              "monitor_mode 0 nominal, 1 backup, 2 failsafe (heading hold)."),
        events=events, context=ctx, series=series)


def main():
    cm, ap, env = car(); scen = scenarios(cm)
    os.makedirs(os.path.join(OUT, "evaluator"), exist_ok=True)
    labels, n = {}, 0
    for split, seeds in [("dev", DEV_SEEDS), ("test", TEST_SEEDS)]:
        os.makedirs(os.path.join(OUT, "packets", split), exist_ok=True)
        for name, mk in scen.items():
            for sd in seeds:
                r = simulate(cm, mk(), SimConfig(T=T, s=4, seed=sd, **BASE), configs=("secure",),
                             autopilot=ap, rta=True, rta_env=env)
                kd = decision_index(r, cm.p.dt)
                ctxs = CONTEXTS if name in ("B3_slow_gnss", "B4_slow_camera") else ["none"]
                for c in ctxs:
                    eid = episode_id(name, sd, c)
                    pk = packet(r, cm, env, kd, eid, make_context(name, c, sd))
                    with open(os.path.join(OUT, "packets", split, f"{eid}.json"), "w") as f:
                        json.dump(pk, f)
                    e = np.abs(r.x_true["secure"][:, 1])
                    labels[eid] = dict(split=split, scenario=name, seed=sd, context=c,
                                       decision_step=int(kd), decision_time_s=round(kd * cm.p.dt, 1),
                                       attacked_channels=[cm.names[i] for i in
                                                          np.where(r.attack_truth.any(axis=0))[0]],
                                       baseline_peak_m=float(e.max()),
                                       baseline_time_outside_s=float((e > env.e_max).sum() * cm.p.dt))
                    n += 1
                print(split, name, sd, "decision at", round(kd * cm.p.dt, 1), "s", flush=True)
    with open(os.path.join(OUT, "evaluator", "labels.json"), "w") as f:
        json.dump(labels, f, indent=1)
    from rubric import admissible_table
    with open(os.path.join(OUT, "evaluator", "admissible.json"), "w") as f:
        json.dump(admissible_table(), f, indent=1)
    print(f"wrote {n} episodes")


if __name__ == "__main__":
    main()
