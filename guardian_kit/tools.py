"""
tools.py  —  Read-only tools the Guardian may call on one evidence packet.
=========================================================================
Every tool returns a JSON-serialisable dict computed only from the packet
(and the public model), never from evaluator labels.  TOOL_SCHEMAS gives the
OpenAI-compatible function definitions; GuardianTools.call(name, args) runs one.
"""
from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import car, model_hash

_CM = None


def _model():
    global _CM
    if _CM is None:
        _CM = car()[0]
    return _CM


class GuardianTools:
    def __init__(self, packet):
        self.p = packet; s = packet["series"]
        self.t = np.array(s["t"]); self.dt = packet["model_card"]["dt_s"]
        self.U = packet["model_card"]["speed_mps"]
        cm = _model()
        if packet["model_card"]["model_hash"] != model_hash(cm):
            raise ValueError("packet was produced with a different model")
        self.names = list(cm.names)

    def _win(self, last_seconds):
        n = max(2, int(round(float(last_seconds) / self.dt)))
        return slice(max(0, len(self.t) - n), len(self.t))

    # ------------------------------------------------------------------ tools
    def get_event_log(self):
        s = self.p["series"]
        flagged = [c for c, v in s["flags"].items() if v[-1]]
        mode = ["nominal", "backup", "failsafe"][s["monitor_mode"][-1]]
        return dict(decision_time_s=self.p["decision_time_s"], events=self.p["events"],
                    currently_flagged=flagged, monitor_mode_now=mode)

    def get_residual_ratios(self, channels=None, last_seconds=5.0):
        w = self._win(last_seconds); out = {}
        for c in (channels or self.names):
            v = np.array([np.nan if x is None else x for x in self.p["series"]["residual_ratio"][c]])[w]
            v = v[np.isfinite(v)]
            out[c] = None if v.size == 0 else dict(latest=round(float(v[-1]), 3),
                                                   max=round(float(v.max()), 3), mean=round(float(v.mean()), 3))
        return dict(window_s=float(last_seconds), note="ratio > 1 fails the detection envelope", ratios=out)

    def get_measurement_trend(self, channels, last_seconds=5.0):
        w = self._win(last_seconds); out = {}
        for c in channels:
            if c not in self.p["series"]["measurement"]:
                out[c] = "not recorded (only lateral position and heading channels are)"; continue
            y = np.array(self.p["series"]["measurement"][c])[w]; tt = self.t[w]
            slope = float(np.polyfit(tt, y, 1)[0])
            out[c] = dict(latest=round(float(y[-1]), 4), mean=round(float(y.mean()), 4), slope_per_s=round(slope, 4))
        return dict(window_s=float(last_seconds), trends=out)

    def lateral_disagreement_rate(self, last_seconds=5.0):
        """Drift-rate estimate b_dot: slope of (mean GNSS lateral - camera offset)."""
        w = self._win(last_seconds); m = self.p["series"]["measurement"]
        g = 0.5 * (np.array(m["GNSS-A lateral"]) + np.array(m["GNSS-B lateral"]))
        d = (g - np.array(m["Lane-cam offset"]))[w]; tt = self.t[w]
        return dict(window_s=float(last_seconds), gnss_minus_camera_now_m=round(float(d[-1]), 4),
                    drift_rate_mps=round(float(np.polyfit(tt, d, 1)[0]), 4),
                    note="positive means GNSS reports the vehicle further left than the camera")

    def implied_heading_discrepancy(self, drift_rate_mps):
        """Heading offset a competing explanation needs, Delta psi ~ b_dot / U (Eq. 9)."""
        dpsi = float(drift_rate_mps) / self.U
        return dict(speed_mps=self.U, drift_rate_mps=float(drift_rate_mps),
                    implied_heading_deg=round(math.degrees(dpsi), 3),
                    best_single_heading_sigma_deg=0.3,
                    note="a diagnostic heuristic, not a detectability condition")

    def observability_rank(self, exclude_channels):
        cm = _model(); A, C = cm.A, cm.C
        keep = [i for i, nm in enumerate(self.names) if nm not in set(exclude_channels)]
        O = np.vstack([C[keep] @ np.linalg.matrix_power(A, k) for k in range(cm.n)])
        rank = int(np.linalg.matrix_rank(O))
        lat_anchor = any(C[i, 1] != 0 for i in keep)
        return dict(excluded=list(exclude_channels), rank=rank, full_rank=cm.n,
                    observable=rank == cm.n, lateral_offset_anchored=bool(lat_anchor),
                    note="observability of a subset does not establish that it is honest")

    def get_dr_status(self):
        s = self.p["series"]
        disc = [np.nan if x is None else x for x in s["dr_discrepancy"]]
        gate = [np.nan if x is None else x for x in s["dr_gate"]]
        exc = [i for i, (a, b) in enumerate(zip(disc, gate)) if np.isfinite(a) and np.isfinite(b) and a > b]
        return dict(discrepancy_now_m=None if not np.isfinite(disc[-1]) else round(float(disc[-1]), 4),
                    gate_now_m=None if not np.isfinite(gate[-1]) else round(float(gate[-1]), 4),
                    exceeded_now=bool(exc and exc[-1] == len(disc) - 1),
                    first_exceedance_s=None if not exc else round(float(self.t[exc[0]]), 1),
                    note="dead reckoning integrates trusted heading and lateral velocity; "
                         "an exceedance means the reported position disagrees with honest dynamics")

    def replay_hypotheses(self, candidates, last_seconds=3.0):
        """Batch weighted least-squares replay of the recent window under each candidate
        exclusion set: fit the window-initial state from the retained channels, applied
        inputs and the model, and report the normalised residual RMS of the retained
        channels (about 1 when the retained set is mutually consistent)."""
        cm = _model(); A, B, C, sig = cm.A, cm.B, cm.C, cm.sigma
        s = self.p["series"]; n = len(self.t); W = min(n, max(10, int(round(float(last_seconds) / self.dt))))
        Y = np.array([s["measurement"][c] for c in self.names]).T[n - W:]
        U = np.vstack([s["applied_input"]["speed_cmd"], s["applied_input"]["steering_rad"]]).T[n - W:]
        out = []
        for excl in candidates:
            keep = [i for i, c in enumerate(self.names) if c not in set(excl)]
            unknown = [c for c in excl if c not in self.names]
            rows, rhs, w = [], [], []
            Phi = np.eye(cm.n); forced = np.zeros(cm.n)
            for k in range(W):
                for i in keep:
                    rows.append(C[i] @ Phi); rhs.append(Y[k, i] - C[i] @ forced); w.append(1.0 / sig[i])
                forced = A @ forced + B @ U[k]; Phi = A @ Phi
            M = np.array(rows) * np.array(w)[:, None]; b = np.array(rhs) * np.array(w)
            x0, *_ = np.linalg.lstsq(M, b, rcond=None); r = b - M @ x0
            per = {self.names[i]: round(float(np.sqrt(np.mean(r[j::len(keep)] ** 2))), 3)
                   for j, i in enumerate(keep)}
            out.append(dict(excluded=list(excl), unknown_channels=unknown,
                            consistency_score=round(float(np.sqrt(np.mean(r ** 2))), 4),
                            per_channel_rms=per))
        return dict(window_s=round(W * self.dt, 1), results=out,
                    note="lower score = retained channels more consistent with the model; on "
                         "attack-free development runs both single-anchor exclusions score "
                         "about 1.05 to 1.11 and differ by less than 0.05")

    def get_context(self):
        return dict(advisories=self.p.get("context", []))

    # -------------------------------------------------------------- dispatch
    def call(self, name, args=None):
        fn = getattr(self, name, None)
        if name.startswith("_") or fn is None or name == "call":
            return dict(error=f"unknown tool {name}")
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return dict(error=f"bad arguments for {name}: {e}")


_CH = {"type": "array", "items": {"type": "string"}}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_event_log", "description":
        "Tier-0 events up to the decision time, currently flagged channels and monitor mode.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_residual_ratios", "description":
        "Residual-to-envelope ratios per channel over a recent window (>1 fails).",
        "parameters": {"type": "object", "properties": {"channels": _CH, "last_seconds": {"type": "number"}}}}},
    {"type": "function", "function": {"name": "get_measurement_trend", "description":
        "Latest value, mean and linear slope of lateral-position and heading channels.",
        "parameters": {"type": "object", "properties": {"channels": _CH, "last_seconds": {"type": "number"}},
                       "required": ["channels"]}}},
    {"type": "function", "function": {"name": "lateral_disagreement_rate", "description":
        "Drift rate of mean GNSS lateral position relative to the lane-camera offset.",
        "parameters": {"type": "object", "properties": {"last_seconds": {"type": "number"}}}}},
    {"type": "function", "function": {"name": "implied_heading_discrepancy", "description":
        "Heading offset needed to explain a lateral drift rate at the current speed.",
        "parameters": {"type": "object", "properties": {"drift_rate_mps": {"type": "number"}},
                       "required": ["drift_rate_mps"]}}},
    {"type": "function", "function": {"name": "observability_rank", "description":
        "Rank of the observability matrix after excluding the given channels.",
        "parameters": {"type": "object", "properties": {"exclude_channels": _CH}, "required": ["exclude_channels"]}}},
    {"type": "function", "function": {"name": "get_dr_status", "description":
        "Dead-reckoning cross-check discrepancy, gate and exceedance history.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "replay_hypotheses", "description":
        "Replay the recent window under candidate exclusion sets (e.g. both GNSS lateral channels, "
        "or the lane camera) and return a consistency score for the retained channels.",
        "parameters": {"type": "object", "properties": {
            "candidates": {"type": "array", "items": _CH}, "last_seconds": {"type": "number"}},
            "required": ["candidates"]}}},
    {"type": "function", "function": {"name": "get_context", "description":
        "External advisories available to the supervisor (may be absent, wrong or conflicting).",
        "parameters": {"type": "object", "properties": {}}}},
]
