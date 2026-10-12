"""Periodic linear hyperdiffusion u_t = -nu*u_xxxx + D*u_xx on [0, L), nu > 0, D >= 0.

Training purpose: stiff fourth-order dissipation. Every Fourier mode decays,
u_hat(t) = u_hat(0) * exp(-(nu*k^4 + D*k^2)*t), k = 2*pi*m/L, so the problem is well posed
and has no growing band. The largest explicit eigenvalue is nu*(pi*N/L)^4, so explicit RK4 with
a step sized for a second-order operator is unstable, and a correctly limited explicit step
needs 1e5 to 1e7 steps. Exact exponential / integrating-factor / ETD propagation and implicit
schemes pass.
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

FAMILY_ID = "hyperdiffusion1d"
DOMAIN_LENGTHS: dict[str, float] = {"1": 1.0, "2": 2.0, "2pi": TWO_PI}
# nu per domain; the mode-1 hyperdiffusive decay nu*(2*pi/L)^4*T is about 0.3, 0.6, 1.2.
HYPER_COEFFICIENTS: dict[str, tuple[float, ...]] = {
    "1": (2.0e-4, 4.0e-4, 8.0e-4),
    "2": (0.003, 0.006, 0.012),
    "2pi": (0.3, 0.6, 1.2),
}
# D per domain; the mode-1 diffusive decay D*(2*pi/L)^2*T is 0, about 0.2, about 0.4.
DIFFUSION_COEFFICIENTS: dict[str, tuple[float, ...]] = {
    "1": (0.0, 0.005, 0.01),
    "2": (0.0, 0.02, 0.04),
    "2pi": (0.0, 0.2, 0.4),
}
FINAL_TIME = 1.0
GRID_SIZES = (64, 128)
TIME_COUNTS = (51, 101)
TIME_GRIDS = ("uniform", "uniform", "stretched")
STRATUM_KEYS = {
    "domain",
    "hyper_index",
    "diffusion_index",
    "nu",
    "D",
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
        "hyper_index": int(rng.integers(0, 3)),
        "diffusion_index": int(rng.integers(0, 3)),
        "final_time": FINAL_TIME,
        "nx": int(rng.choice(GRID_SIZES)),
        "nt": int(rng.choice(TIME_COUNTS)),
        "time_grid": str(rng.choice(TIME_GRIDS)),
        "input_kind": str(rng.choice(IC_KINDS)),
    }
    selected = apply_stratum(selected, stratum, STRATUM_KEYS)
    key = str(selected["domain"])
    nu = float(selected.get("nu", HYPER_COEFFICIENTS[key][int(selected["hyper_index"])]))
    diffusion = float(
        selected.get("D", DIFFUSION_COEFFICIENTS[key][int(selected["diffusion_index"])])
    )
    if nu <= 0.0 or diffusion < 0.0:
        raise ValueError("hyperdiffusion1d requires nu > 0 and D >= 0")
    parameters: dict[str, int | float | str] = {
        "nu": nu,
        "D": diffusion,
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
            f"\nFor this task, nu={_float(case, 'nu')!r}, D={_float(case, 'D')!r} and "
            f"L={format_length(_float(case, 'L'))}, so the domain is [0, L)."
        )
    if form == "parameter_ic":
        conditioning += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Define exactly:
    def solver(u0_batch, t_coordinate, nu, D, L):

Solve the one-dimensional linear hyperdiffusion equation

    u_t = -nu * u_xxxx + D * u_xx,  x in [0, L),

with periodic boundary conditions: u has period L in x. Here nu > 0 and D >= 0.
The domain length L is a solver argument that changes between tasks; it is
generally not 1.

{time_sentence()}{conditioning}
"""


def invocation(case: PDECase) -> SolverInvocation:
    times = t_coordinate(case)
    return SolverInvocation(
        argument_order=("u0_batch", "t_coordinate", "nu", "D", "L"),
        arrays={"u0_batch": initial_condition(case)[None, :], "t_coordinate": times},
        scalars={"nu": _float(case, "nu"), "D": _float(case, "D"), "L": _float(case, "L")},
        expected_shape=(1, len(times), case.grid_shape[0]),
    )


def reference_primary(case: PDECase) -> np.ndarray:
    """Exact modal decay of the continuous problem from closed-form IC coefficients."""
    length = _float(case, "L")
    modes, coefficients = fourier_coefficients(case.input_kind, case.parameters)
    wave = TWO_PI * modes / length
    rate = _float(case, "nu") * wave**4 + _float(case, "D") * wave**2
    return modal_trajectory(
        modes, coefficients, x_coordinate(case) / length, t_coordinate(case), -rate + 0j
    )


def reference_audit(case: PDECase) -> np.ndarray:
    """Independent route: expm(t * (-nu*D2@D2 + D*D2)) with physical-space spectral D2."""
    _, second = spectral_differentiation_matrices(case.grid_shape[0], _float(case, "L"))
    operator = -_float(case, "nu") * (second @ second) + _float(case, "D") * second
    return matrix_exponential_trajectory(operator, initial_condition(case), t_coordinate(case))


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    values = np.asarray(field, dtype=np.float64)
    times = t_coordinate(case)
    if values.shape != (1, len(times), case.grid_shape[0]):
        raise ValueError(f"{FAMILY_ID} residual received an invalid shape")
    length = _float(case, "L")
    residual = (
        time_derivative(values, times)
        + _float(case, "nu") * spectral_derivative(values, length, 4)
        - _float(case, "D") * spectral_derivative(values, length, 2)
    )
    return float(np.sqrt(np.mean(np.square(residual))))


class Hyperdiffusion1DEnvironment:
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
