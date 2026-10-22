"""Route family and prior for the decision study (research example)."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .physics import MaterialParams, Schedule, Segment, isothermal

T_NOM = 933.0  # nominal spinodal temperature known to the decision maker (K)
AGING = ((T_NOM - 30.0, 1.0), (T_NOM - 60.0, 2.0), (T_NOM - 90.0, 3.0), (T_NOM - 120.0, 4.0))


def _aging() -> tuple[Segment, ...]:
    return tuple(isothermal(T, h, False) for T, h in AGING)


def route_family() -> list[Schedule]:
    """25 routes: field-off control, isothermal field anneals, continuous-cooling field anneals.

    Temperatures are relative to the nominal T_s; the true batch T_s is uncertain. Every route
    ends with the same gradual field-off step-down aging (10 h).
    """
    routes = [Schedule((isothermal(T_NOM - 5.0, 2.0, False),) + _aging(), label="nofield")]
    for off in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0):
        for h in (1.0, 2.0, 4.0):
            routes.append(Schedule((isothermal(T_NOM - off, h, True),) + _aging(),
                                   label=f"iso{off:+.0f}K-{h:.0f}h"))
    for rate in (10.0, 20.0, 40.0):
        hours = 40.0 / rate
        routes.append(Schedule((Segment(hours, T_NOM + 15.0, T_NOM - 25.0, True),) + _aging(),
                               label=f"ramp{rate:.0f}Kph"))
    return routes


def route_family_v2() -> list[Schedule]:
    """14 routes spanning the decision space (v2, after the box-artifact rerun decision):
    field-off control; 2 h field anneals at -5..+20 K offsets from the nominal T_s; 1 h and 4 h
    at +5 and +10 K; continuous cooling in field at 10, 20, 40 K/h through the window.
    """
    routes = [Schedule((isothermal(T_NOM - 5.0, 2.0, False),) + _aging(), label="nofield")]
    for off in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0):
        routes.append(Schedule((isothermal(T_NOM - off, 2.0, True),) + _aging(), label=f"iso{off:+.0f}K-2h"))
    for off in (5.0, 10.0):
        for h in (1.0, 4.0):
            routes.append(Schedule((isothermal(T_NOM - off, h, True),) + _aging(), label=f"iso{off:+.0f}K-{h:.0f}h"))
    for rate in (10.0, 20.0, 40.0):
        hours = 40.0 / rate
        routes.append(Schedule((Segment(hours, T_NOM + 15.0, T_NOM - 25.0, True),) + _aging(),
                               label=f"ramp{rate:.0f}Kph"))
    return routes


PRIOR = {
    "T_s": ("uniform", 913.0, 953.0),
    "c0": ("uniform", 0.48, 0.52),
    "D_ref": ("loguniform", 5e-19, 2e-17),
    "Q": ("uniform", 220e3, 280e3),
    "kappa": ("uniform", 0.6e-10, 2.0e-10),
    "B0": ("loguniform", 0.5e6, 4e6),
    "noise_rms": ("loguniform", 0.001, 0.02),
    "alpha_c": ("uniform", 0.25, 0.5),
    "R_coh_nm": ("uniform", 9.0, 18.0),
    "d_z_nm": ("loguniform", 5.0, 50.0),  # thermal-noise coarse-graining thickness (model form)
}


# Model form M2 (near-Curie magnetization): the batch Curie point of the homogeneous alloy
# relative to its spinodal, T_C(c0) - T_s, replaces B0 as the coupling parameter.
PRIOR_M2_EXTRA = {"dTc": ("uniform", -25.0, 5.0)}
TC_A = 1250.0


def sample_prior(n: int, namespace: str, model: str = "M1") -> list[dict]:
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    prior = dict(PRIOR)
    if model == "M2":
        prior.update(PRIOR_M2_EXTRA)
    out = []
    for _ in range(n):
        s = {}
        for k, (kind, lo, hi) in prior.items():
            s[k] = float(rng.uniform(lo, hi)) if kind == "uniform" else float(
                math.exp(rng.uniform(math.log(lo), math.log(hi))))
        s["ic_seed"] = int(rng.integers(1 << 31))
        if model == "M2":
            s["mag_model"] = "curie"
            s["Tc_a"] = TC_A
            target = s["T_s"] + s["dTc"]
            s["Tc_b"] = TC_A + (target - TC_A) / s["c0"]
        out.append(s)
    return out


def params_from(sample: dict) -> MaterialParams:
    keys = {k: v for k, v in sample.items() if k in MaterialParams.__dataclass_fields__}
    return MaterialParams(**keys)
