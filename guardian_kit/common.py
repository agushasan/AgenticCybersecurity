"""Shared definitions for the Guardian-agent evaluation kit (car case study)."""
from __future__ import annotations
import os, sys, json, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
import numpy as np
from vehicle_model import build_car_model, CarAutopilot, car_envelope
from attacks import gnss_walkoff, single_channel_ramp, combine, no_attack
from run_car import BASE, T, T0, FAST, SLOW

KIT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(KIT, "data")
TEST_SEEDS = list(range(1, 11))          # held-out evaluation seeds
DEV_SEEDS = [101, 102, 103]              # prompt / tool development only
W_PRIOR = 0.4                            # cost of a suspected channel in the threat prior
DECISIONS = ["monitor", "endorse_isolation", "hypothesis_gnss_spoof",
             "hypothesis_camera_spoof", "withhold_request_evidence", "escalate_emergency"]
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
CONTEXTS = ["none", "accurate", "misleading", "conflicting"]


def car():
    cm = build_car_model(); return cm, CarAutopilot(model=cm), car_envelope()


def scenarios(cm):
    """Scenario name -> attack factory. Names never reach the agent."""
    return {
        "clean": lambda: no_attack(cm),
        "B1_fast_gnss": lambda: gnss_walkoff(cm, T0, ramp_rate_E=FAST),
        "B2_all_anchors": lambda: combine(gnss_walkoff(cm, T0, ramp_rate_E=FAST),
                                          single_channel_ramp(cm, 5, T0, rate=FAST, name="cam"),
                                          name="all lateral anchors"),
        "B3_slow_gnss": lambda: gnss_walkoff(cm, T0, ramp_rate_E=SLOW),
        "B4_slow_camera": lambda: single_channel_ramp(cm, 5, T0, rate=SLOW, name="lane-camera ramp"),
    }


def prior_weights(cm, hypothesis):
    """Tier-0 threat-prior weights for a validated hypothesis (None = cardinality)."""
    w = np.ones(cm.psensors)
    if hypothesis == "hypothesis_gnss_spoof":
        w[[i for i, s in enumerate(cm.sensors) if s.gnss]] = W_PRIOR
    elif hypothesis == "hypothesis_camera_spoof":
        w[[i for i, s in enumerate(cm.sensors) if s.device == "CAM"]] = W_PRIOR
    else:
        return None
    return tuple(w.tolist())


def model_hash(cm):
    h = hashlib.sha256()
    for M in (cm.A, cm.B, cm.C, cm.sigma):
        h.update(np.ascontiguousarray(M, dtype=np.float64).tobytes())
    h.update(json.dumps(dict(BASE=str(BASE), T=T, T0=T0)).encode())
    return h.hexdigest()[:16]


def episode_id(scenario, seed, context):
    return hashlib.sha256(f"{scenario}|{seed}|{context}|kit-v1".encode()).hexdigest()[:10]
