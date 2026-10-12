"""Source-derived held-out 1D advection-reaction-diffusion environment.

Paper fact: the transfer figure gives
``u_t + beta*u_x = nu*u_xx + rho*u*(1-u)`` and shows one front-like example with
``beta=1``, ``nu=0.02``, and ``rho=2``. The paper does not publish the domain,
boundary condition, front generator, parameter grid, case manifest, reference solver,
residual, or function signature. This module uses a periodic unit interval, a smooth
periodic front pair, and the explicit local interface below. These are source-derived
reproduction choices, not the authors' hidden environment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import integrate_to_times, periodic_first, periodic_grid, periodic_second, time_grid

LOCAL_GRID_SIZES = (64, 128)
LOCAL_TIME_COUNTS = (51, 101)
LOCAL_FINAL_TIME = 1.0
LOCAL_BETA_VALUES = (-1.0, -0.4, 0.4, 1.0)
LOCAL_NU_VALUES = (0.005, 0.02, 0.05)
LOCAL_RHO_VALUES = (1.0, 2.0, 5.0)


def make_case(*, seed: int, split: str = "eval", stratum: dict[str, Any] | None = None) -> PDECase:
    """Make one deterministic case from source-derived local parameter grids."""
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {
        "grid_shape": (int(rng.choice(LOCAL_GRID_SIZES)),),
        "time_count": int(rng.choice(LOCAL_TIME_COUNTS)),
        "final_time": LOCAL_FINAL_TIME,
        "input_kind": "periodic_front_pair",
        "beta": float(rng.choice(LOCAL_BETA_VALUES)),
        "nu": float(rng.choice(LOCAL_NU_VALUES)),
        "rho": float(rng.choice(LOCAL_RHO_VALUES)),
        "center": float(rng.uniform(0.30, 0.70)),
        "radius": float(rng.uniform(0.14, 0.24)),
        "width": float(rng.uniform(0.025, 0.055)),
        "low": float(rng.uniform(0.04, 0.12)),
        "height": float(rng.uniform(0.65, 0.82)),
    }
    if stratum:
        unknown = set(stratum) - set(values)
        if unknown:
            raise ValueError(
                f"unknown advection_reaction_diffusion1d stratum keys: {sorted(unknown)}"
            )
        values.update(stratum)
    parameters = {
        key: values[key]
        for key in (
            "beta",
            "nu",
            "rho",
            "center",
            "radius",
            "width",
            "low",
            "height",
        )
    }
    return make_case_record(
        family_id="advection_reaction_diffusion1d",
        split=split,
        seed=seed,
        grid_shape=tuple(int(item) for item in values["grid_shape"]),
        time_count=int(values["time_count"]),
        final_time=float(values["final_time"]),
        input_kind=str(values["input_kind"]),
        parameters=parameters,
    )


def _grid(case: PDECase) -> np.ndarray:
    return periodic_grid(case.grid_shape[0])


def _times(case: PDECase) -> np.ndarray:
    assert case.time_count is not None and case.final_time is not None
    return time_grid(case.time_count, case.final_time)


def initial_condition(case: PDECase) -> np.ndarray:
    """Return a smooth periodic plateau bounded between zero and one."""
    x = _grid(case)
    p = case.parameters
    distance = (x - float(p["center"]) + 0.5) % 1.0 - 0.5
    radius = float(p["radius"])
    width = float(p["width"])
    plateau = 0.5 * (np.tanh((distance + radius) / width) - np.tanh((distance - radius) / width))
    return (float(p["low"]) + float(p["height"]) * plateau)[None, :]


def _reaction_exact(state: np.ndarray, step: float, rho: float) -> np.ndarray:
    growth = np.exp(rho * step)
    return state * growth / (1.0 + state * (growth - 1.0))


def _spectral_advection_diffusion(
    state: np.ndarray, step: float, beta: float, nu: float
) -> np.ndarray:
    count = state.shape[-1]
    wave = 2.0 * np.pi * np.fft.fftfreq(count, d=1.0 / count)
    multiplier = np.exp((-nu * wave**2 - 1j * beta * wave) * step)
    return np.fft.ifft(np.fft.fft(state, axis=-1) * multiplier[None, :], axis=-1).real


def reference_strang(case: PDECase) -> np.ndarray:
    """Primary reference: exact reaction and spectral transport in Strang splitting."""
    times = _times(case)
    state = initial_condition(case)
    output = np.empty((1, len(times), case.grid_shape[0]), dtype=np.float64)
    output[:, 0, :] = state
    now = float(times[0])
    beta = float(case.parameters["beta"])
    nu = float(case.parameters["nu"])
    rho = float(case.parameters["rho"])
    for index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(0.002, float(target) - now)
            state = _reaction_exact(state, 0.5 * step, rho)
            state = _spectral_advection_diffusion(state, step, beta, nu)
            state = _reaction_exact(state, 0.5 * step, rho)
            now += step
        output[:, index, :] = state
    return output


def _finite_difference_rhs(case: PDECase) -> Callable[[np.ndarray], np.ndarray]:
    spacing = 1.0 / case.grid_shape[0]
    beta = float(case.parameters["beta"])
    nu = float(case.parameters["nu"])
    rho = float(case.parameters["rho"])

    def rhs(state: np.ndarray) -> np.ndarray:
        return (
            -beta * periodic_first(state, spacing)
            + nu * periodic_second(state, spacing)
            + rho * state * (1.0 - state)
        )

    return rhs


def reference_finite_difference(case: PDECase) -> np.ndarray:
    """Audit reference: centered finite differences with small-step RK4."""
    spacing = 1.0 / case.grid_shape[0]
    beta = abs(float(case.parameters["beta"]))
    nu = float(case.parameters["nu"])
    rho = float(case.parameters["rho"])
    limits = [0.01 / max(rho, 1e-12)]
    if beta > 0.0:
        limits.append(0.25 * spacing / beta)
    if nu > 0.0:
        limits.append(0.2 * spacing**2 / nu)
    trajectory = integrate_to_times(
        initial_condition(case),
        _times(case),
        _finite_difference_rhs(case),
        max_step=min(limits),
    )
    return np.swapaxes(trajectory, 0, 1)


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    """Compute a centered discrete RMS residual on interior output times."""
    candidate = np.asarray(field, dtype=np.float64)
    expected = (1, int(case.time_count or 0), case.grid_shape[0])
    if candidate.shape != expected:
        raise ValueError(f"expected field shape {expected}, got {candidate.shape}")
    times = _times(case)
    interior = candidate[:, 1:-1]
    derivative_t = (candidate[:, 2:] - candidate[:, :-2]) / (times[2:] - times[:-2])[None, :, None]
    spacing = 1.0 / case.grid_shape[0]
    residual = (
        derivative_t
        + float(case.parameters["beta"]) * periodic_first(interior, spacing)
        - float(case.parameters["nu"]) * periodic_second(interior, spacing)
        - float(case.parameters["rho"]) * interior * (1.0 - interior)
    )
    return float(np.sqrt(np.mean(np.square(residual))))


def prompt(case: PDECase, *, form: str = "generic") -> str:
    if form not in {"generic", "parameter", "parameter_ic"}:
        raise ValueError(f"unknown prompt form: {form}")
    details = ""
    if form in {"parameter", "parameter_ic"}:
        details += (
            f"\nFor this case beta={case.parameters['beta']}, nu={case.parameters['nu']}, "
            f"and rho={case.parameters['rho']}."
        )
    if form == "parameter_ic":
        details += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Solve the periodic one-dimensional advection-reaction-diffusion equation

    u_t + beta * u_x = nu * u_xx + rho * u * (1 - u),  x in [0, 1).

Write this function:

    def solver(u0_batch, t_coordinate, beta, nu, rho):
        ...

u0_batch has shape [B, N]. t_coordinate has shape [T]. Return a finite NumPy
array with shape [B, T, N]. Returned slice n must correspond to t_coordinate[n].
Internal substeps are allowed. Do not hard-code a solution array.{details}
"""


class AdvectionReactionDiffusion1DEnvironment:
    family_id = "advection_reaction_diffusion1d"
    seen_in_training = False
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "advection_reaction_diffusion1d":
            raise ValueError(f"wrong family for advection_reaction_diffusion1d: {case.family_id}")
        return case

    @staticmethod
    def make_case(
        *, seed: int, split: str = "eval", stratum: dict[str, Any] | None = None
    ) -> PDECase:
        return make_case(seed=seed, split=split, stratum=stratum)

    @staticmethod
    def prompt(case: PDECase, *, form: str = "generic") -> str:
        return prompt(case, form=form)

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("u0_batch", "t_coordinate", "beta", "nu", "rho"),
            arrays={"u0_batch": initial_condition(case), "t_coordinate": _times(case)},
            scalars={
                "beta": float(case.parameters["beta"]),
                "nu": float(case.parameters["nu"]),
                "rho": float(case.parameters["rho"]),
            },
            expected_shape=(1, int(case.time_count or 0), case.grid_shape[0]),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        return reference_strang(case)

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        return reference_finite_difference(case)

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        return residual_norm(field, case)
