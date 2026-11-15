"""
attacks.py
==========
Cyberattack models on shipboard navigation sensors, expressed as additive attack
signals a_i(t) injected into the measurement equation (Shoukry et al. 2018,
Eq. 2.2):     y_i(t) = C_i x(t) + a_i(t) + psi_i(t).

The attack signal a_i(t) is UNBOUNDED and arbitrary in the Shoukry et al. model
(no statistical assumptions).  We implement physically meaningful maritime
attacks:

1. GNSS spoofing "walk-off" (drag-off)
   A GNSS spoofer first aligns with the authentic signal, then slews the reported
   position smoothly away from truth so the victim receiver tracks the false
   solution without losing lock.  Because civilian GNSS services are open, ONE
   spoofer captures ALL GNSS receivers simultaneously -> a COMMON-MODE attack on
   every GNSS-derived channel.  We model a slowly ramped cross-track (East) bias
   (plus optional along-track), which is the maritime "lead the ship off the
   channel" threat (cf. the 2017 Black Sea mass-spoofing event and the 2013
   University of Texas yacht experiment).

2. GNSS meaconing / coordinated offset
   Constant or step position offset on all GNSS channels.

3. Single-sensor bias / ramp (e.g. tampered gyrocompass feed or one GNSS unit)
   A single corrupted channel, within the certified s_max = 1 margin.

4. Radar fix falsification
   Corruption of the radar-derived position fix (false landmark association),
   used to remove the surviving GNSS-independent position anchor.

Each attack is a callable a(t, x_true) -> attack vector (p,) added to the clean
measurement.  Attacks only act on their designated channels; all other entries
are zero.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Attack:
    name: str
    channels: list[int]
    t_start: float
    fn: callable                     # fn(t, x_true, model) -> full (p,) vector
    description: str = ""

    def __call__(self, t, x_true, model):
        return self.fn(t, x_true, model)


def _zeros(model):
    return np.zeros(model.psensors)


# --------------------------------------------------------------------------- #
#  1. GNSS spoofing walk-off (common-mode on all GNSS channels)               #
# --------------------------------------------------------------------------- #
def gnss_walkoff(model, t_start=120.0, ramp_rate_E=0.6, ramp_rate_N=0.0,
                 hold_after=None, align_first=True):
    """
    Smooth cross-track walk-off on all GNSS East (and optionally North) channels.
    ramp_rate_E : metres/second of injected East bias growth (e.g. 0.6 m/s).
    The injected bias is referenced so the spoofed position = truth + bias, i.e.
    a_i(t) = bias(t) on GNSS channels; the receiver reports y_i = C_i x + a_i.
    """
    gE = [i for i in range(model.psensors)
          if model.gnss_mask[i] and np.argmax(np.abs(model.C[i])) == 1]
    gN = [i for i in range(model.psensors)
          if model.gnss_mask[i] and np.argmax(np.abs(model.C[i])) == 0]

    def fn(t, x_true, m):
        a = _zeros(m)
        if t < t_start:
            return a
        dt = t - t_start
        if hold_after is not None and dt > hold_after:
            dt = hold_after
        biasE = ramp_rate_E * dt
        biasN = ramp_rate_N * dt
        for i in gE:
            a[i] = biasE
        for i in gN:
            a[i] = biasN
        return a

    return Attack("GNSS walk-off (common-mode)", gE + gN, t_start, fn,
                  description=f"All-GNSS cross-track drag-off, "
                              f"{ramp_rate_E} m/s East ramp from t={t_start}s")


# --------------------------------------------------------------------------- #
#  2. GNSS coordinated step offset (meaconing)                                #
# --------------------------------------------------------------------------- #
def gnss_step(model, t_start=120.0, offset_E=60.0, offset_N=0.0):
    g = [i for i in range(model.psensors) if model.gnss_mask[i]]

    def fn(t, x_true, m):
        a = _zeros(m)
        if t < t_start:
            return a
        for i in g:
            ax = np.argmax(np.abs(m.C[i]))
            a[i] = offset_E if ax == 1 else (offset_N if ax == 0 else 0.0)
        return a
    return Attack("GNSS step offset", g, t_start, fn,
                  description=f"All-GNSS step: E={offset_E} m, N={offset_N} m")


# --------------------------------------------------------------------------- #
#  3. Single-channel bias / ramp (within s_max = 1 margin)                    #
# --------------------------------------------------------------------------- #
def single_channel_ramp(model, channel, t_start=120.0, rate=0.01, cap=None,
                        name=None):
    def fn(t, x_true, m):
        a = _zeros(m)
        if t < t_start:
            return a
        val = rate * (t - t_start)
        if cap is not None:
            val = np.clip(val, -abs(cap), abs(cap))
        a[channel] = val
        return a
    nm = name or f"Single-channel ramp on {model.names[channel]}"
    return Attack(nm, [channel], t_start, fn,
                  description=f"{model.names[channel]} ramp {rate}/s from t={t_start}s")


def single_channel_bias(model, channel, t_start=120.0, bias=5.0, name=None):
    def fn(t, x_true, m):
        a = _zeros(m)
        if t >= t_start:
            a[channel] = bias
        return a
    nm = name or f"Single-channel bias on {model.names[channel]}"
    return Attack(nm, [channel], t_start, fn,
                  description=f"{model.names[channel]} bias {bias} from t={t_start}s")


# --------------------------------------------------------------------------- #
#  4. Combine attacks (e.g. GNSS walk-off + radar falsification)              #
# --------------------------------------------------------------------------- #
def combine(*attacks, name="combined"):
    chans = sorted(set(c for a in attacks for c in a.channels))
    t0 = min(a.t_start for a in attacks)

    def fn(t, x_true, m):
        a = _zeros(m)
        for atk in attacks:
            a = a + atk(t, x_true, m)
        return a
    return Attack(name, chans, t0, fn,
                  description=" + ".join(a.name for a in attacks))


# --------------------------------------------------------------------------- #
#  5. Intermittent / hopping multi-phase campaign (time-varying support)      #
# --------------------------------------------------------------------------- #
#  The Shoukry et al. model explicitly allows the attacked SET to change with
#  time (each sensor may be switched on/off).  A sophisticated adversary hops
#  between sensor groups to evade a static detector and to probe the suite.
#  A "phase" is (t_start, t_end, {channel: signal_fn(dt)}) where dt = t - t_start.
# --------------------------------------------------------------------------- #
def _ramp(rate, cap=None):
    def f(dt):
        v = rate * dt
        return float(np.clip(v, -abs(cap), abs(cap))) if cap is not None else v
    return f

def _const(val):
    return lambda dt: float(val)

def _ramp_hold(rate, t_hold):
    def f(dt):
        return rate * min(dt, t_hold)
    return f


def scripted_campaign(model, phases, name="scripted campaign", description=""):
    """
    phases: list of dicts, each
        {"t0": float, "t1": float, "sig": {channel_index: fn(dt)->bias}}
    Multiple phases may overlap; contributions add.
    """
    chans = sorted({c for ph in phases for c in ph["sig"]})
    t0 = min(ph["t0"] for ph in phases)

    def fn(t, x_true, m):
        a = _zeros(m)
        for ph in phases:
            if ph["t0"] <= t < ph["t1"]:
                dt = t - ph["t0"]
                for c, sig in ph["sig"].items():
                    a[c] += sig(dt)
        return a
    return Attack(name, chans, t0, fn, description=description)


def hopping_gnss_campaign(model, t_start=100.0):
    """
    A four-phase dynamic campaign demonstrating time-varying attack support:

      Phase 1  probe       : single GNSS-A East bias ramp        (within s_max=1)
      Phase 2  escalate    : full GNSS common-mode East walk-off (4 ch, s up)
      Phase 3  relocate    : GNSS released; radar-East falsified  (support hops)
      Phase 4  coordinated : GNSS walk-off + gyrocompass bias     (cross-sensor)

    Between phases the attacker goes quiet, so the detector must CONTINUALLY
    re-identify the (changing) attacked set.
    """
    iE = [i for i in range(model.psensors)
          if model.gnss_mask[i] and np.argmax(np.abs(model.C[i])) == 1]
    gnssA_E = next(i for i in iE if model.devices[i] == "GNSS_A")
    gyro = next(i for i in range(model.psensors) if model.devices[i] == "GYROC")
    mag = next(i for i in range(model.psensors) if model.devices[i] == "MAG")
    T0 = t_start
    DEG = np.pi / 180.0
    phases = [
        # Phase 1: probe with one GNSS channel (within s_max = 1)
        {"t0": T0, "t1": T0 + 70,
         "sig": {gnssA_E: _ramp_hold(0.9, 55)}},
        # Phase 2: escalate to full-GNSS common-mode cross-track walk-off
        {"t0": T0 + 110, "t1": T0 + 210,
         "sig": {i: _ramp(0.7) for i in iE}},
        # Phase 3: PIVOT to heading spoofing -- bias the gyrocompass (the
        # magnetic compass and fibre-optic gyro survive to expose it)
        {"t0": T0 + 270, "t1": T0 + 350,
         "sig": {gyro: _ramp_hold(0.22 * DEG, 55)}},
        # Phase 4: coordinated GNSS position + gyrocompass heading (cross-
        # consistent): the false heading 'explains' the cross-track motion
        {"t0": T0 + 410, "t1": T0 + 520,
         "sig": {**{i: _ramp(0.6) for i in iE},
                 gyro: _ramp(0.0022)}},   # ~0.13 deg/s heading drift
    ]
    return scripted_campaign(
        model, phases, name="Adaptive multi-phase campaign",
        description="probe -> GNSS common-mode walk-off -> heading-spoof pivot "
                    "-> coordinated GNSS+gyro (time-varying attack support)")


# --------------------------------------------------------------------------- #
#  6. Coordinated cross-consistent attack (defeats pairwise consistency)      #
# --------------------------------------------------------------------------- #
def coordinated_consistent(model, t_start=120.0, ramp_E=0.55, gyro_rate=0.0024):
    """
    The adversary spoofs the GNSS cross-track position AND biases the primary
    gyrocompass so that the two corrupted streams stay MUTUALLY CONSISTENT (the
    false heading 'explains' part of the apparent cross-track motion).  A naive
    pairwise GNSS-vs-gyro consistency check is thus fooled.  Secure estimation
    still isolates both because it relies on 2s-sparse OBSERVABILITY: the two
    remaining compasses (mag, FOG) and the radar position fix expose the
    inconsistency.  Attacks GNSS-East (x2) + gyrocompass = 3 channels.
    """
    iE = [i for i in range(model.psensors)
          if model.gnss_mask[i] and np.argmax(np.abs(model.C[i])) == 1]
    gyro = next(i for i in range(model.psensors) if model.devices[i] == "GYROC")

    def fn(t, x_true, m):
        a = _zeros(m)
        if t < t_start:
            return a
        dt = t - t_start
        for i in iE:
            a[i] = ramp_E * dt
        a[gyro] = gyro_rate * dt
        return a
    return Attack("Coordinated GNSS+gyro (cross-consistent)",
                  iE + [gyro], t_start, fn,
                  description="GNSS cross-track walk-off + consistent gyrocompass "
                              "bias; defeats pairwise consistency, not SSE")


# --------------------------------------------------------------------------- #
#  7. Stealthy detection-aware walk-off (rides just below the envelope)       #
# --------------------------------------------------------------------------- #
def stealthy_walkoff(model, t_start=120.0, slow_rate=0.12):
    """
    A patient adversary ramps the GNSS cross-track bias slowly enough to stay
    near/under the detection floor for a long time, trading effect for stealth.
    Demonstrates the fundamental stealth-vs-speed trade-off: the attack evades
    detection longer but its damage RATE is bounded by the detector's floor.
    """
    return gnss_walkoff(model, t_start=t_start, ramp_rate_E=slow_rate)


# --------------------------------------------------------------------------- #
#  8. Maneuver-masquerade spoof (hides inside a legitimate turn)              #
# --------------------------------------------------------------------------- #
def maneuver_masquerade(model, path, t_start=235.0, T=440, offset_max=55.0,
                        channels="gnssE"):
    """
    A turn-timed GNSS cross-track spoof SHAPED to masquerade as a legitimate
    course change.  The injected East bias accumulates *only while the vessel is
    genuinely turning* (positive reference yaw rate through the planned dog-leg),
    in lockstep with the real maneuver, then HOLDS on the following straight.  The
    spoofed position therefore stays consistent with "the vessel tracked a
    slightly wider dog-leg", defeating a plausibility check that tests whether the
    apparent position change agrees with the commanded/estimated maneuvering
    motion.  Crucially, the attacker does NOT touch the heading channels: the
    gyrocompass, magnetic compass and fibre-optic gyro report the TRUE (smaller)
    heading, and the independent radar fix reports the TRUE position, so the extra
    cross-track offset is unsupported by the trusted subset.  Secure state
    estimation isolates it via 2s-sparse OBSERVABILITY even though no residual
    "spike" at onset betrays it -- the bias rides the maneuver smoothly.

    Unlike the constant-rate walk-off (Scenario B/M), the injected drift here is
    correlated with, and proportional to, the genuine turn: a maneuver-consistency
    monitor keyed on "position drift vs commanded turning" is thereby fooled.
    """
    iE = [i for i in range(model.psensors)
          if model.gnss_mask[i] and np.argmax(np.abs(model.C[i])) == 1]
    U = model.p.U
    dt = model.p.dt
    tgrid = np.arange(T) * dt
    # reference path-tangent heading along the planned route
    psi = np.array([path.ref(U * tt)[1] for tt in tgrid])
    dpsi = np.gradient(psi, tgrid)                    # reference heading rate
    # accumulate ONLY positive heading change after t_start (the starboard dog-leg)
    inc = np.where((tgrid >= t_start) & (dpsi > 0.0), dpsi, 0.0)
    prof = np.cumsum(inc) * dt                        # ramps in turn, holds after
    if prof[-1] > 0:
        prof = prof / prof[-1] * offset_max           # scale to reach offset_max [m]

    def fn(t, x_true, m):
        a = _zeros(m)
        k = int(round(t / dt))
        k = min(max(k, 0), T - 1)
        b = prof[k]
        for i in iE:
            a[i] = b
        return a

    return Attack("Maneuver-masquerade GNSS spoof", iE, t_start, fn,
                  description="turn-shaped GNSS cross-track spoof that mimics a "
                              "wider dog-leg; stealthy to maneuver-consistency "
                              "checks, isolated by structural observability")


# --------------------------------------------------------------------------- #
#  No-attack                                                                  #
# --------------------------------------------------------------------------- #
def no_attack(model):
    return Attack("no attack", [], np.inf,
                  lambda t, x, m: _zeros(m), description="clean baseline")
