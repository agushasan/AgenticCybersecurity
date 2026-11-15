"""Deterministic comparison policies on the same packets and tools.

rule_baseline        uses Tier-0 evidence only, written to reproduce the pilot behaviour.
context_rule         additionally follows a single external advisory and abstains when
                     advisories conflict (a deliberately simple context-aware policy).
monitor_only         the original system: no supervisory action at all.
replay_rule          rule_baseline plus the hypothesis-replay tool, with thresholds tuned on
                     the development seeds only (attack-free gap < 0.05, attacked gap > 0.15).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import GuardianTools

GNSS_LAT = ["GNSS-A lateral", "GNSS-B lateral"]


def _evidence(tl):
    ev = tl.get_event_log(); dr = tl.get_dr_status(); d = tl.lateral_disagreement_rate()
    return ev, dr, d


def rule_baseline(packet):
    tl = GuardianTools(packet); ev, dr, d = _evidence(tl)
    flagged = ev["currently_flagged"]; dr_exc = dr["first_exceedance_s"] is not None
    if dr_exc and abs(d["drift_rate_mps"]) < 0.1:
        dec, lvl = "escalate_emergency", "L4"        # all anchors agree yet contradict dead reckoning
    elif dr_exc and "Lane-cam offset" in flagged:
        dec, lvl = "withhold_request_evidence", "L3"  # estimator blames camera, DR contradicts it
    elif flagged:
        dec, lvl = "endorse_isolation", "L1"
    else:
        dec, lvl = "monitor", "L0"
    return dict(decision=dec, autonomy_level=lvl, explanation="rule baseline", claims=[])


def context_rule(packet):
    out = rule_baseline(packet)
    adv = GuardianTools(packet).get_context()["advisories"]
    anomalous = out["decision"] not in ("monitor", "escalate_emergency")
    if anomalous and len(adv) == 1:
        gnss = adv[0].get("subject") == "GNSS receivers"
        out = dict(out, decision="hypothesis_gnss_spoof" if gnss else "hypothesis_camera_spoof",
                   autonomy_level="L1", explanation="context rule: follows the single advisory")
    elif anomalous and len(adv) > 1:
        out = dict(out, decision="withhold_request_evidence", autonomy_level="L3",
                   explanation="context rule: advisories conflict")
    return out


REPLAY_MARGIN = 0.08          # dev seeds: clean |gap| <= 0.045, slow attacks |gap| >= 0.155
ALL_BAD = 1.5                 # dev seeds: all-anchor attack scores 2.2 to 2.9, otherwise <= 1.21


def replay_rule(packet):
    out = rule_baseline(packet)
    if out["decision"] == "monitor":
        return out
    tl = GuardianTools(packet)
    g, c = tl.replay_hypotheses([GNSS_LAT, ["Lane-cam offset"]])["results"]
    gs, cs = g["consistency_score"], c["consistency_score"]
    if gs > ALL_BAD and cs > ALL_BAD:
        dec, lvl = "escalate_emergency", "L4"
    elif cs - gs > REPLAY_MARGIN:
        dec, lvl = "hypothesis_gnss_spoof", "L1"
    elif gs - cs > REPLAY_MARGIN:
        dec, lvl = ("endorse_isolation" if "Lane-cam offset" in tl.get_event_log()["currently_flagged"]
                    else "hypothesis_camera_spoof"), "L1"
    else:
        return out
    return dict(decision=dec, autonomy_level=lvl, explanation="replay rule", claims=[])


def monitor_only(packet):
    return dict(decision="monitor", autonomy_level="L0", explanation="no supervisor", claims=[])


METHODS = {"monitor_only": monitor_only, "rule_baseline": rule_baseline, "context_rule": context_rule,
           "replay_rule": replay_rule}
