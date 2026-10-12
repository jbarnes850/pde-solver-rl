"""Periodic linear advection on a stated non-unit domain [x_min, x_max).

Training purpose: Fourier-symbol semantics. A correct spectral solver must use the wavenumber
scale k = 2*pi*fftfreq(N, d=L/N) with L = x_max - x_min, the sign exp(-i*k*beta*t), and
absolute output times (the time grid may be non-uniform). Semi-Lagrangian interpolation from
the initial state and high-order method-of-lines schemes also pass.
"""

from __future__ import annotations

import math
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
    significant,
    spectral_derivative,
    time_coordinate,
    time_derivative,
    time_sentence,
)

FAMILY_ID = "advection1d_domain"
DOMAINS: dict[str, tuple[float, float]] = {
    "0_2": (0.0, 2.0),
    "0_2pi": (0.0, TWO_PI),
    "0_half": (0.0, 0.5),
    "m1_1": (-1.0, 1.0),
}
# Domain traversals |beta| * T / L; beta is derived from these and rounded to 6 digits.
TRAVERSALS = (0.4, 0.7, 1.1, 1.5)
FINAL_TIMES = (1.0, 2.0)
GRID_SIZES = (64, 128)
TIME_COUNTS = (51, 101)
TIME_GRIDS = ("uniform", "uniform", "stretched")
STRATUM_KEYS = {
    "domain",
    "traversals",
    "direction",
    "final_time",
    "nx",
    "nt",
    "time_grid",
    "input_kind",
    "beta",
}
GRID_SENTENCE_P2 = (
    "Grid: u0_batch[:, i] is the value at x_i = x_min + i*(x_max - x_min)/N for i = 0, ..., "
    "N-1. The grid is periodic with spacing (x_max - x_min)/N, so x = x_max is the same point "
    "as x = x_min."
)
PROPOSED_REFERENCE_LIMIT = 1.0e-8


def _float(case: PDECase, name: str) -> float:
    return float(case.parameters[name])


def domain(case: PDECase) -> tuple[float, float, float]:
    lower, upper = _float(case, "x_min"), _float(case, "x_max")
    return lower, upper, upper - lower


def x_coordinate(case: PDECase) -> np.ndarray:
    lower, _, length = domain(case)
    count = case.grid_shape[0]
    return lower + np.arange(count, dtype=np.float64) * (length / count)


def t_coordinate(case: PDECase) -> np.ndarray:
    if case.time_count is None or case.final_time is None:
        raise ValueError(f"{FAMILY_ID} requires time_count and final_time")
    return time_coordinate(str(case.parameters["time_grid"]), case.time_count, case.final_time)


def initial_condition(case: PDECase) -> np.ndarray:
    lower, _, length = domain(case)
    return evaluate_initial(
        (x_coordinate(case) - lower) / length, case.input_kind, case.parameters
    )


def make_case(
    *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    selected: dict[str, Any] = {
        "domain": str(rng.choice(sorted(DOMAINS))),
        "traversals": float(rng.choice(TRAVERSALS)),
        "direction": int(rng.choice((1, -1))),
        "final_time": float(rng.choice(FINAL_TIMES)),
        "nx": int(rng.choice(GRID_SIZES)),
        "nt": int(rng.choice(TIME_COUNTS)),
        "time_grid": str(rng.choice(TIME_GRIDS)),
        "input_kind": str(rng.choice(IC_KINDS)),
    }
    selected = apply_stratum(selected, stratum, STRATUM_KEYS)
    lower, upper = DOMAINS[str(selected["domain"])]
    length = upper - lower
    final_time = float(selected["final_time"])
    if "beta" in selected:
        beta = float(selected["beta"])
    else:
        beta = int(selected["direction"]) * significant(
            float(selected["traversals"]) * length / final_time
        )
    parameters: dict[str, int | float | str] = {
        "beta": beta,
        "x_min": float(lower),
        "x_max": float(upper),
        "domain": str(selected["domain"]),
        "time_grid": str(selected["time_grid"]),
        **draw_initial_condition(seed, str(selected["input_kind"])),
    }
    return make_case_record(
        family_id=FAMILY_ID,
        split=split,
        seed=seed,
        grid_shape=(int(selected["nx"]),),
        time_count=int(selected["nt"]),
        final_time=final_time,
        input_kind=str(selected["input_kind"]),
        parameters=parameters,
    )


def prompt(case: PDECase, *, form: str = "generic") -> str:
    check_form(form)
    conditioning = ""
    if form in {"parameter", "parameter_ic"}:
        lower, upper, length = domain(case)
        conditioning += (
            f"\nFor this task, beta={_float(case, 'beta')!r}, x_min={lower!r} and "
            f"x_max={format_length(upper)}, so the domain is [x_min, x_max) with period "
            f"L = x_max - x_min = {format_length(length)}."
        )
    if form == "parameter_ic":
        conditioning += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Define exactly:
    def solver(u0_batch, t_coordinate, beta, x_min, x_max):

Solve the one-dimensional linear advection equation

    u_t + beta * u_x = 0,  x in [x_min, x_max),

with periodic boundary conditions: u has period L = x_max - x_min in x. The
domain endpoints x_min and x_max are solver arguments that change between
tasks; the domain is generally not [0, 1), and beta may be negative.

{time_sentence()}{conditioning}
"""


def invocation(case: PDECase) -> SolverInvocation:
    times = t_coordinate(case)
    return SolverInvocation(
        argument_order=("u0_batch", "t_coordinate", "beta", "x_min", "x_max"),
        arrays={"u0_batch": initial_condition(case)[None, :], "t_coordinate": times},
        scalars={
            "beta": _float(case, "beta"),
            "x_min": _float(case, "x_min"),
            "x_max": _float(case, "x_max"),
        },
        expected_shape=(1, len(times), case.grid_shape[0]),
    )


def reference_primary(case: PDECase) -> np.ndarray:
    """Exact characteristic shift of the continuous initial condition, evaluated in x."""
    lower, _, length = domain(case)
    x = x_coordinate(case)
    times = t_coordinate(case)
    departure = (x[None, :] - lower - _float(case, "beta") * times[:, None]) / length
    return evaluate_initial(departure, case.input_kind, case.parameters)[None, :, :]


def reference_audit(case: PDECase) -> np.ndarray:
    """Independent route: FFT of the sampled initial state, exact phase with scaled wavenumbers."""
    _, _, length = domain(case)
    count = case.grid_shape[0]
    times = t_coordinate(case)
    spectrum = np.fft.fft(initial_condition(case))
    wave = 2.0 * math.pi * np.fft.fftfreq(count, d=length / count)
    phase = np.exp(-1j * _float(case, "beta") * np.outer(times, wave))
    return np.fft.ifft(spectrum[None, :] * phase, axis=-1).real[None, :, :]


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    values = np.asarray(field, dtype=np.float64)
    times = t_coordinate(case)
    if values.shape != (1, len(times), case.grid_shape[0]):
        raise ValueError(f"{FAMILY_ID} residual received an invalid shape")
    _, _, length = domain(case)
    residual = time_derivative(values, times) + _float(case, "beta") * spectral_derivative(
        values, length, 1
    )
    return float(np.sqrt(np.mean(np.square(residual))))


class Advection1DDomainEnvironment:
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
