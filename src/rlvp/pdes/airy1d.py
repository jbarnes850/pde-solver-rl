"""Periodic linear Airy equation u_t + a*u_xxx = 0 on [0, L).

Training purpose: stiff third-order dispersion. With u = sum_k u_hat_k exp(i*k*x),
u_xxx -> (i*k)^3 u_hat = -i*k^3 u_hat, so d(u_hat)/dt = i*a*k^3*u_hat and
u_hat(t) = u_hat(0) * exp(i*a*k^3*t), k = 2*pi*m/L. The largest explicit eigenvalue is
|a|*(pi*N/L)^3, so explicit RK4 with a step sized for a second-order operator (dt ~ dx^2) is
unstable. Exact exponential / integrating-factor propagation, implicit Fourier-space schemes,
or explicit schemes with a dx^3-scaled step pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation
from rlvp.pdes.base import PDECase, make_case_record

from .periodic_linear import (
    IC_KINDS,
    TWO_PI,
    apply_stratum,
    check_form,
    draw_initial_condition,
    evaluate_initial,
    format_length,
    fourier_coefficients,
    matrix_exponential_trajectory,
    modal_trajectory,
    spectral_derivative,
    spectral_differentiation_matrices,
    time_coordinate,
    time_derivative,
    time_sentence,
)

FAMILY_ID = "airy1d"
DOMAIN_LENGTHS: dict[str, float] = {"1": 1.0, "2pi": TWO_PI}
# |a| per domain; the mode-1 phase |a|*(2*pi/L)^3*T is about 0.25, 0.5, 1.0, 1.5 rad on both.
COEFFICIENTS: dict[str, tuple[float, ...]] = {
    "1": (0.001, 0.002, 0.004, 0.006),
    "2pi": (0.25, 0.5, 1.0, 1.5),
}
FINAL_TIME = 1.0
GRID_SIZES = (64, 128)
TIME_COUNTS = (51, 101)
TIME_GRIDS = ("uniform", "uniform", "stretched")
STRATUM_KEYS = {
    "domain",
    "strength",
    "sign",
    "a",
    "final_time",
    "nx",
    "nt",
    "time_grid",
    "input_kind",
}
GRID_SENTENCE_P2 = (
    "Grid: u0_batch[:, i] is the value at x_i = i*L/N for i = 0, ..., N-1. The grid is periodic "
    "with spacing L/N, so x = L is the same point as x = 0."
)
PROPOSED_REFERENCE_LIMIT = 1.0e-8


def _float(case: PDECase, name: str) -> float:
    return float(case.parameters[name])


def x_coordinate(case: PDECase) -> np.ndarray:
    count = case.grid_shape[0]
    return np.arange(count, dtype=np.float64) * (_float(case, "L") / count)


def t_coordinate(case: PDECase) -> np.ndarray:
    if case.time_count is None or case.final_time is None:
        raise ValueError(f"{FAMILY_ID} requires time_count and final_time")
    return time_coordinate(str(case.parameters["time_grid"]), case.time_count, case.final_time)


def initial_condition(case: PDECase) -> np.ndarray:
    return evaluate_initial(
        x_coordinate(case) / _float(case, "L"), case.input_kind, case.parameters
    )


def make_case(
    *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    selected: dict[str, Any] = {
        "domain": str(rng.choice(sorted(DOMAIN_LENGTHS))),
        "strength": int(rng.integers(0, 4)),
        "sign": int(rng.choice((1, -1))),
        "final_time": FINAL_TIME,
        "nx": int(rng.choice(GRID_SIZES)),
        "nt": int(rng.choice(TIME_COUNTS)),
        "time_grid": str(rng.choice(TIME_GRIDS)),
        "input_kind": str(rng.choice(IC_KINDS)),
    }
    selected = apply_stratum(selected, stratum, STRATUM_KEYS)
    key = str(selected["domain"])
    if "a" in selected:
        coefficient = float(selected["a"])
    else:
        coefficient = int(selected["sign"]) * COEFFICIENTS[key][int(selected["strength"])]
    parameters: dict[str, int | float | str] = {
        "a": float(coefficient),
        "L": DOMAIN_LENGTHS[key],
        "domain": key,
        "time_grid": str(selected["time_grid"]),
        **draw_initial_condition(seed, str(selected["input_kind"])),
    }
    return make_case_record(
        family_id=FAMILY_ID,
        split=split,
        seed=seed,
        grid_shape=(int(selected["nx"]),),
        time_count=int(selected["nt"]),
        final_time=float(selected["final_time"]),
        input_kind=str(selected["input_kind"]),
        parameters=parameters,
    )


def prompt(case: PDECase, *, form: str = "generic") -> str:
    check_form(form)
    conditioning = ""
    if form in {"parameter", "parameter_ic"}:
        conditioning += (
            f"\nFor this task, a={_float(case, 'a')!r} and L={format_length(_float(case, 'L'))}, "
            "so the domain is [0, L)."
        )
    if form == "parameter_ic":
        conditioning += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Define exactly:
    def solver(u0_batch, t_coordinate, a, L):

Solve the one-dimensional linear Airy (dispersive) equation

    u_t + a * u_xxx = 0,  x in [0, L),

with periodic boundary conditions: u has period L in x. The domain length L
is a solver argument that changes between tasks; it is generally not 1, and a
may be negative.

{time_sentence()}{conditioning}
"""


def invocation(case: PDECase) -> SolverInvocation:
    times = t_coordinate(case)
    return SolverInvocation(
        argument_order=("u0_batch", "t_coordinate", "a", "L"),
        arrays={"u0_batch": initial_condition(case)[None, :], "t_coordinate": times},
        scalars={"a": _float(case, "a"), "L": _float(case, "L")},
        expected_shape=(1, len(times), case.grid_shape[0]),
    )


def reference_primary(case: PDECase) -> np.ndarray:
    """Exact modal solution of the continuous problem from closed-form IC coefficients."""
    length = _float(case, "L")
    modes, coefficients = fourier_coefficients(case.input_kind, case.parameters)
    wave = TWO_PI * modes / length
    return modal_trajectory(
        modes,
        coefficients,
        x_coordinate(case) / length,
        t_coordinate(case),
        1j * _float(case, "a") * wave**3,
    )


def reference_audit(case: PDECase) -> np.ndarray:
    """Independent route: expm(-a * t * D^3) with the physical-space spectral matrix D."""
    first, _ = spectral_differentiation_matrices(case.grid_shape[0], _float(case, "L"))
    operator = -_float(case, "a") * (first @ first @ first)
    return matrix_exponential_trajectory(operator, initial_condition(case), t_coordinate(case))


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    values = np.asarray(field, dtype=np.float64)
    times = t_coordinate(case)
    if values.shape != (1, len(times), case.grid_shape[0]):
        raise ValueError(f"{FAMILY_ID} residual received an invalid shape")
    third = spectral_derivative(values, _float(case, "L"), 3)
    residual = time_derivative(values, times) + _float(case, "a") * third
    return float(np.sqrt(np.mean(np.square(residual))))


class Airy1DEnvironment:
    family_id = FAMILY_ID
    seen_in_training = True
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != FAMILY_ID:
            raise ValueError(f"wrong family for {FAMILY_ID}: {case.family_id}")
        return case

    make_case = staticmethod(make_case)
    prompt = staticmethod(prompt)
    invocation = staticmethod(invocation)
    reference_primary = staticmethod(reference_primary)
    reference_audit = staticmethod(reference_audit)
    residual_norm = staticmethod(residual_norm)
