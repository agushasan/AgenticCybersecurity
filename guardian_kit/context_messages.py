"""Context advisories for the ambiguity episodes (identical for every context-aware method)."""
from __future__ import annotations

GNSS_ADV = dict(source="national road authority traffic bulletin", subject="GNSS receivers", age_min=12,
                text="GNSS interference reported on this motorway section; position "
                     "receivers may report shifted fixes. Lane markings are in good condition.")
CAM_ADV = dict(source="vehicle perception health monitor", subject="lane camera", age_min=1,
               text="Lane-camera confidence anomaly: possible lane-marking defacement or road "
                    "patch ahead. GNSS signal quality nominal.")


def make_context(scenario, kind, seed):
    """Return a list of advisories. 'accurate' points at the true attacked group,
    'misleading' at the other one, 'conflicting' gives both, 'none' gives nothing."""
    if kind == "none" or scenario not in ("B3_slow_gnss", "B4_slow_camera"):
        return []
    true_adv, false_adv = (GNSS_ADV, CAM_ADV) if scenario == "B3_slow_gnss" else (CAM_ADV, GNSS_ADV)
    return {"accurate": [true_adv], "misleading": [false_adv],
            "conflicting": [true_adv, false_adv] if seed % 2 else [false_adv, true_adv]}[kind]
