from __future__ import annotations

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import integrate_to_times, time_grid

FAMILY_ID = "diffusion_sorption1d"

PAPER_DIFFUSIVITIES = (1.0e-4, 5.0e-4, 1.5e-3)
PAPER_FREUNDLICH_COEFFICIENTS = (1.0e-4, 3.5e-4, 1.0e-3)
PAPER_GRID_SIZES = (64, 128)
PAPER_TIME_COUNTS = (51, 101)
PAPER_FINAL_TIME = 500.0
PAPER_POROSITY = 0.29
PAPER_SOLID_DENSITY = 2880.0
PAPER_FREUNDLICH_EXPONENT = 0.874
PAPER_LEFT_CONCENTRATION = 1.0
PAPER_EPSILON = 1.0e-6

# The paper names low-, mid-, and high-concentration uniform fields but does not give their
# values. These explicit values are part of the source-derived contract, not paper facts.
LOCAL_INITIAL_VALUES = {
    "uniform_low": 0.0,
    "uniform_mid": 0.25,
    "uniform_high": 0.75,
}

DIFFUSION_SORPTION_METHOD_IDS = (
    "euler_fixed",
    "heun_fixed",
    "rk4_fixed",
    "backward_euler_lagged",
    "crank_nicolson_lagged",
    "backward_euler_newton_tridiag",
    "crank_nicolson_newton_tridiag",
    "mol_bdf_solve_ivp_mixedbc",
)


def _parameters(case: PDECase) -> tuple[float, float, float, float, float, float, float, float]:
    values = case.parameters
    return (
        float(values["D"]),
        float(values["phi"]),
        float(values["rho_s"]),
        float(values["kf"]),
        float(values["nf"]),
        float(values["sol"]),
        float(values["eps"]),
        float(values["initial_value"]),
    )


def _validate_case(case: PDECase) -> None:
    if case.family_id != FAMILY_ID:
        raise ValueError(f"expected {FAMILY_ID}, got {case.family_id}")
    if len(case.grid_shape) != 1 or case.grid_shape[0] not in PAPER_GRID_SIZES:
        raise ValueError("diffusion-sorption grid must use a paper grid size")
    if case.time_count not in PAPER_TIME_COUNTS:
        raise ValueError("diffusion-sorption time_count must use a paper value")
    if case.final_time != PAPER_FINAL_TIME:
        raise ValueError("diffusion-sorption final_time must match the paper")
    if case.input_kind not in LOCAL_INITIAL_VALUES:
        raise ValueError(f"unknown diffusion-sorption input kind: {case.input_kind}")
    diffusivity, porosity, density, coefficient, exponent, sol, epsilon, initial = _parameters(
        case
    )
    if diffusivity not in PAPER_DIFFUSIVITIES:
        raise ValueError("D must use a paper value")
    if coefficient not in PAPER_FREUNDLICH_COEFFICIENTS:
        raise ValueError("k_f must use a paper value")
    fixed = (porosity, density, exponent, sol, epsilon)
    expected = (
        PAPER_POROSITY,
        PAPER_SOLID_DENSITY,
        PAPER_FREUNDLICH_EXPONENT,
        PAPER_LEFT_CONCENTRATION,
        PAPER_EPSILON,
    )
    if fixed != expected:
        raise ValueError("fixed diffusion-sorption parameters must match the paper")
    if initial != LOCAL_INITIAL_VALUES[case.input_kind]:
        raise ValueError("initial_value does not match the source-derived input kind")


def _space_grid(case: PDECase) -> np.ndarray:
    count = case.grid_shape[0]
    return (np.arange(count, dtype=np.float64) + 0.5) / count


def _times(case: PDECase) -> np.ndarray:
    if case.time_count is None or case.final_time is None:
        raise ValueError("diffusion-sorption is time dependent")
    return time_grid(case.time_count, case.final_time)


def _initial_batch(case: PDECase) -> np.ndarray:
    _, _, _, _, _, _, _, value = _parameters(case)
    return np.full((1, case.grid_shape[0]), value, dtype=np.float64)


def _retardation(
    state: np.ndarray,
    *,
    porosity: float,
    density: float,
    coefficient: float,
    exponent: float,
    epsilon: float,
) -> np.ndarray:
    # All source-derived cases are nonnegative. The maximum supplies a deterministic numerical
    # extension for round-off near zero without changing the stated PDE on its intended domain.
    concentration = np.maximum(np.asarray(state, dtype=np.float64), 0.0) + epsilon
    scale = ((1.0 - porosity) / porosity) * density * coefficient * exponent
    return 1.0 + scale * np.power(concentration, exponent - 1.0)


def _laplacian_mixed(state: np.ndarray, *, spacing: float, sol: float) -> np.ndarray:
    """Cell-centered Laplacian with a left Dirichlet and right zero-flux face."""

    values = np.asarray(state, dtype=np.float64)
    result = np.empty_like(values)
    result[..., 1:-1] = values[..., :-2] - 2.0 * values[..., 1:-1] + values[..., 2:]
    # Reflection through the left face places the prescribed value at x=0.
    result[..., 0] = values[..., 1] - 3.0 * values[..., 0] + 2.0 * sol
    # Copying the right ghost cell enforces zero flux through x=1.
    result[..., -1] = values[..., -2] - values[..., -1]
    return result / spacing**2


def _rhs(state: np.ndarray, case: PDECase) -> np.ndarray:
    diffusivity, porosity, density, coefficient, exponent, sol, epsilon, _ = _parameters(case)
    retardation = _retardation(
        state,
        porosity=porosity,
        density=density,
        coefficient=coefficient,
        exponent=exponent,
        epsilon=epsilon,
    )
    spacing = 1.0 / case.grid_shape[0]
    return diffusivity * _laplacian_mixed(state, spacing=spacing, sol=sol) / retardation


def _reference_explicit(case: PDECase) -> np.ndarray:
    diffusivity, *_ = _parameters(case)
    spacing = 1.0 / case.grid_shape[0]
    # R(u) is at least one. This is below the one-dimensional RK4 diffusion limit even at the
    # largest effective diffusivity.
    max_step = 0.35 * spacing**2 / diffusivity
    trajectory = integrate_to_times(
        _initial_batch(case), _times(case), lambda u: _rhs(u, case), max_step
    )
    return np.moveaxis(trajectory, 0, 1)


def _solve_tridiagonal(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, right_hand_side: np.ndarray
) -> np.ndarray:
    lower_work = np.asarray(lower, dtype=np.float64).copy()
    diagonal_work = np.asarray(diagonal, dtype=np.float64).copy()
    upper_work = np.asarray(upper, dtype=np.float64).copy()
    rhs_work = np.asarray(right_hand_side, dtype=np.float64).copy()
    count = diagonal_work.shape[-1]
    for index in range(1, count):
        factor = lower_work[..., index] / diagonal_work[..., index - 1]
        diagonal_work[..., index] -= factor * upper_work[..., index - 1]
        rhs_work[..., index] -= factor * rhs_work[..., index - 1]
    result = np.empty_like(rhs_work)
    result[..., -1] = rhs_work[..., -1] / diagonal_work[..., -1]
    for index in range(count - 2, -1, -1):
        result[..., index] = (
            rhs_work[..., index] - upper_work[..., index] * result[..., index + 1]
        ) / diagonal_work[..., index]
    return result


def _crank_nicolson_step(state: np.ndarray, step: float, case: PDECase) -> np.ndarray:
    diffusivity, porosity, density, coefficient, exponent, sol, epsilon, _ = _parameters(case)
    spacing = 1.0 / case.grid_shape[0]
    count = case.grid_shape[0]
    boundary = np.zeros_like(state)
    boundary[..., 0] = 2.0 * sol / spacing**2

    def solve_with_mobility(mobility: np.ndarray) -> np.ndarray:
        factor = 0.5 * step * mobility / spacing**2
        lower = np.zeros_like(state)
        upper = np.zeros_like(state)
        diagonal = np.ones_like(state)
        lower[..., 1:] = -factor[..., 1:]
        upper[..., :-1] = -factor[..., :-1]
        diagonal[..., 1:-1] += 2.0 * factor[..., 1:-1]
        diagonal[..., 0] += 3.0 * factor[..., 0]
        diagonal[..., -1] += factor[..., -1]
        old_laplacian = _laplacian_mixed(state, spacing=spacing, sol=sol)
        rhs = state + 0.5 * step * mobility * old_laplacian
        rhs += 0.5 * step * mobility * boundary
        if count < 3:
            raise ValueError("diffusion-sorption reference requires at least three cells")
        return _solve_tridiagonal(lower, diagonal, upper, rhs)

    old_retardation = _retardation(
        state,
        porosity=porosity,
        density=density,
        coefficient=coefficient,
        exponent=exponent,
        epsilon=epsilon,
    )
    estimate = solve_with_mobility(diffusivity / old_retardation)
    # Two Picard corrections evaluate the nonlinear mobility near the temporal midpoint.
    for _ in range(2):
        midpoint = 0.5 * (state + estimate)
        midpoint_retardation = _retardation(
            midpoint,
            porosity=porosity,
            density=density,
            coefficient=coefficient,
            exponent=exponent,
            epsilon=epsilon,
        )
        estimate = solve_with_mobility(diffusivity / midpoint_retardation)
    return estimate


def _reference_crank_nicolson(case: PDECase) -> np.ndarray:
    times = _times(case)
    state = _initial_batch(case)
    output = np.empty((1, len(times), case.grid_shape[0]), dtype=np.float64)
    output[:, 0, :] = state
    now = float(times[0])
    for output_index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1.0e-14:
            step = min(0.1, float(target) - now)
            state = _crank_nicolson_step(state, step, case)
            now += step
        output[:, output_index, :] = state
    return output


def _residual_components(field: np.ndarray, case: PDECase) -> tuple[float, float, float]:
    values = np.asarray(field, dtype=np.float64)
    expected = (1, int(case.time_count or 0), case.grid_shape[0])
    if values.shape != expected:
        raise ValueError(f"expected field shape {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < -PAPER_EPSILON):
        return float("inf"), float("inf"), float("inf")

    diffusivity, porosity, density, coefficient, exponent, sol, epsilon, _ = _parameters(case)
    times = _times(case)
    spacing = 1.0 / case.grid_shape[0]
    time_derivative = np.gradient(values, times, axis=1, edge_order=2)[:, 1:-1, :]
    interior_states = values[:, 1:-1, :]
    retardation = _retardation(
        interior_states,
        porosity=porosity,
        density=density,
        coefficient=coefficient,
        exponent=exponent,
        epsilon=epsilon,
    )
    mobility = diffusivity / retardation
    equation = time_derivative - mobility * _laplacian_mixed(
        interior_states, spacing=spacing, sol=sol
    )

    # The output grid is cell centered. Quadratic extrapolation estimates the left face value,
    # and a quadratic one-sided derivative estimates the right-face flux. Boundary defects are
    # converted to the equation's concentration-per-time units before aggregation.
    left_value = (
        1.875 * interior_states[..., 0]
        - 1.25 * interior_states[..., 1]
        + 0.375 * interior_states[..., 2]
    )
    left_defect_rate = 2.0 * mobility[..., 0] * (left_value - sol) / spacing**2
    right_derivative = (
        2.0 * interior_states[..., -1] - 3.0 * interior_states[..., -2] + interior_states[..., -3]
    ) / spacing
    right_defect_rate = mobility[..., -1] * right_derivative / spacing
    return (
        float(np.sqrt(np.mean(np.square(equation)))),
        float(np.sqrt(np.mean(np.square(left_defect_rate)))),
        float(np.sqrt(np.mean(np.square(right_defect_rate)))),
    )


class DiffusionSorption1DEnvironment:
    family_id = FAMILY_ID
    seen_in_training = True
    method_ids = DIFFUSION_SORPTION_METHOD_IDS

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        _validate_case(case)
        return case

    @staticmethod
    def make_case(
        *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
    ) -> PDECase:
        rng = np.random.default_rng(seed)
        selected = dict(stratum or {})
        diffusivity = float(selected.pop("D", rng.choice(PAPER_DIFFUSIVITIES)))
        coefficient = float(selected.pop("kf", rng.choice(PAPER_FREUNDLICH_COEFFICIENTS)))
        grid_size = int(selected.pop("nx", rng.choice(PAPER_GRID_SIZES)))
        time_count = int(selected.pop("nt", rng.choice(PAPER_TIME_COUNTS)))
        input_kind = str(selected.pop("input_kind", rng.choice(tuple(LOCAL_INITIAL_VALUES))))
        final_time = float(selected.pop("final_time", PAPER_FINAL_TIME))
        if selected:
            raise ValueError(f"unknown diffusion-sorption stratum fields: {sorted(selected)}")
        if input_kind not in LOCAL_INITIAL_VALUES:
            raise ValueError(f"unknown diffusion-sorption input kind: {input_kind}")
        case = make_case_record(
            family_id=FAMILY_ID,
            split=split,
            seed=seed,
            grid_shape=(grid_size,),
            time_count=time_count,
            final_time=final_time,
            input_kind=input_kind,
            parameters={
                "D": diffusivity,
                "phi": PAPER_POROSITY,
                "rho_s": PAPER_SOLID_DENSITY,
                "kf": coefficient,
                "nf": PAPER_FREUNDLICH_EXPONENT,
                "sol": PAPER_LEFT_CONCENTRATION,
                "eps": PAPER_EPSILON,
                "initial_value": LOCAL_INITIAL_VALUES[input_kind],
            },
        )
        _validate_case(case)
        return case

    @staticmethod
    def prompt(case: PDECase, *, form: str = "generic") -> str:
        _validate_case(case)
        if form not in {"generic", "parameter", "parameter_ic"}:
            raise ValueError(f"unknown prompt form: {form}")
        diffusivity, porosity, density, coefficient, exponent, sol, epsilon, _ = _parameters(case)
        conditioning = ""
        if form in {"parameter", "parameter_ic"}:
            conditioning += f"\nUse D={diffusivity} and kf={coefficient}."
        if form == "parameter_ic":
            conditioning += f"\nThe initial-condition family is {case.input_kind}."
        return f"""Solve the one-dimensional nonlinear diffusion-sorption equation

    u_t = D * u_xx / R(u),  x in [0, 1],
    R(u) = 1 + ((1-phi)/phi) * rho_s * kf * nf * (u+eps)**(nf-1).

Use u(t, 0)=sol and u_x(t, 1)=0. Write this function:

    def solver(u0_batch, t_coordinate, D, kf, phi, rho_s, nf, sol, eps):
        ...

u0_batch has shape [B, N]. t_coordinate has shape [T]. Return a finite NumPy
array with shape [B, T, N]. Returned slice n must correspond to
t_coordinate[n]. Internal substeps are allowed. Use a stable numerical method.
Do not hard-code a solution array. The fixed paper values are phi={porosity},
rho_s={density}, nf={exponent}, sol={sol}, and eps={epsilon}.{conditioning}
"""

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        _validate_case(case)
        diffusivity, porosity, density, coefficient, exponent, sol, epsilon, _ = _parameters(case)
        return SolverInvocation(
            argument_order=(
                "u0_batch",
                "t_coordinate",
                "D",
                "kf",
                "phi",
                "rho_s",
                "nf",
                "sol",
                "eps",
            ),
            arrays={"u0_batch": _initial_batch(case), "t_coordinate": _times(case)},
            scalars={
                "D": diffusivity,
                "kf": coefficient,
                "phi": porosity,
                "rho_s": density,
                "nf": exponent,
                "sol": sol,
                "eps": epsilon,
            },
            expected_shape=(1, int(case.time_count or 0), case.grid_shape[0]),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        _validate_case(case)
        return _reference_explicit(case)

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        _validate_case(case)
        return _reference_crank_nicolson(case)

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        _validate_case(case)
        equation, left_boundary, right_boundary = _residual_components(field, case)
        return float(np.sqrt((equation**2 + left_boundary**2 + right_boundary**2) / 3.0))
