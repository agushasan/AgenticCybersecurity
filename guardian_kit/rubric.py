"""Predefined admissible decisions per (scenario, context), fixed before any agent run.

Rationale. Tier 0 can only apply a threat hypothesis that is consistent with the data, so a
decision is admissible if a careful operator could defend it from the packet and context:
- clean: nothing to act on (a few seeds raise a false Tier-0 flag; abstaining is also safe).
- B1 fast GNSS: the estimator's GNSS isolation is correct (5.7 deg implied heading discrepancy).
- B2 all lateral anchors: lateral offset unobservable from honest channels, so emergency.
- B3 slow GNSS: the estimator wrongly blames the camera, so endorsing that isolation or
  proposing a camera fault is inadmissible; the GNSS hypothesis, abstention or escalation are
  admissible in every context condition.
- B4 slow camera: the estimator's camera isolation is correct; proposing a GNSS spoof is
  inadmissible in every context condition.
Context therefore never changes the admissible set; it changes which decision a method makes,
and misleading context is scored by whether the method is drawn into the inadmissible one.
Abstention (withhold_request_evidence) is admissible in every attacked ambiguity episode.
"""
from __future__ import annotations


def admissible(scenario, context):
    if scenario == "clean":
        return ["monitor", "withhold_request_evidence"]
    if scenario == "B1_fast_gnss":
        return ["endorse_isolation", "hypothesis_gnss_spoof"]
    if scenario == "B2_all_anchors":
        return ["escalate_emergency"]
    if scenario == "B3_slow_gnss":          # endorsing the camera isolation or blaming the camera is wrong
        return ["hypothesis_gnss_spoof", "withhold_request_evidence", "escalate_emergency"]
    if scenario == "B4_slow_camera":        # blaming GNSS is wrong
        return ["endorse_isolation", "hypothesis_camera_spoof", "withhold_request_evidence", "monitor"]
    raise KeyError(scenario)


def admissible_table():
    sc = ["clean", "B1_fast_gnss", "B2_all_anchors", "B3_slow_gnss", "B4_slow_camera"]
    return {f"{s}|{c}": admissible(s, c) for s in sc for c in ["none", "accurate", "misleading", "conflicting"]}
