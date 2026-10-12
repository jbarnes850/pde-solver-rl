"""Source-derived held-out 1D advection-diffusion environment.

Paper fact: ``advection-diffusion1d`` is one of the ten held-out families. The paper
does not give its equation, sign convention, domain, boundaries, coefficients, inputs,
case manifest, reference solver, residual, or function signature. This module uses the
canonical local choice ``u_t + beta*u_x = nu*u_xx`` on the periodic unit interval with
smooth Fourier inputs. These choices are explicit reproduction decisions, not author facts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import periodic_first, periodic_grid, periodic_second, time_grid

LOCAL_GRID_SIZES = (64, 128)
LOCAL_TIME_COUNTS = (51, 101)
LOCAL_FINAL_TIME = 1.0
LOCAL_BETA_VALUES = (-1.0, -0.4, 0.4, 1.0)
LOCAL_NU_VALUES = (0.002, 0.01, 0.05)


def make_case(*, seed: int, split: str = "eval", stratum: dict[str, Any] | None = None) -> PDECase:
    """Make one deterministic case from source-derived local parameter grids."""
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {
        "grid_shape": (int(rng.choice(LOCAL_GRID_SIZES)),),
        "time_count": int(rng.choice(LOCAL_TIME_COUNTS)),
        "final_time": LOCAL_FINAL_TIME,
        "input_kind": "two_periodic_modes",
        "beta": float(rng.choice(LOCAL_BETA_VALUES)),
        "nu": float(rng.choice(LOCAL_NU_VALUES)),
        "phase_1": float(rng.uniform(0.0, 1.0)),
        "phase_2": float(rng.uniform(0.0, 1.0)),
        "amplitude_1": float(rng.uniform(0.45, 0.9)),
        "amplitude_2": float(rng.uniform(0.15, 0.4)),
        "offset": float(rng.uniform(-0.15, 0.15)),
    }
    if stratum:
        unknown = set(stratum) - set(values)
        if unknown:
            raise ValueError(f"unknown advection_diffusion1d stratum keys: {sorted(unknown)}")
        values.update(stratum)
    parameters = {
        key: values[key]
        for key in (
            "beta",
            "nu",
            "phase_1",
            "phase_2",
            "amplitude_1",
            "amplitude_2",
            "offset",
        )
    }
    return make_case_record(
        family_id="advection_diffusion1d",
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
    """Return the local two-mode periodic input with shape ``[1, N]``."""
    x = _grid(case)
    p = case.parameters
    values = (
        float(p["offset"])
        + float(p["amplitude_1"]) * np.sin(2.0 * np.pi * (x - float(p["phase_1"])))
        + float(p["amplitude_2"]) * np.cos(4.0 * np.pi * (x - float(p["phase_2"])))
    )
    return values[None, :]


def reference_analytic(case: PDECase) -> np.ndarray:
    """Evaluate the closed-form evolution of both Fourier modes."""
    x = _grid(case)
    times = _times(case)
    p = case.parameters
    beta = float(p["beta"])
    nu = float(p["nu"])
    shifted = x[None, :] - beta * times[:, None]
    values = (
        float(p["offset"])
        + float(p["amplitude_1"])
        * np.exp(-nu * (2.0 * np.pi) ** 2 * times[:, None])
        * np.sin(2.0 * np.pi * (shifted - float(p["phase_1"])))
        + float(p["amplitude_2"])
        * np.exp(-nu * (4.0 * np.pi) ** 2 * times[:, None])
        * np.cos(4.0 * np.pi * (shifted - float(p["phase_2"])))
    )
    return values[None, :, :]


def reference_fft(case: PDECase) -> np.ndarray:
    """Independently evolve the sampled input through its discrete Fourier spectrum."""
    count = case.grid_shape[0]
    times = _times(case)
    wave = 2.0 * np.pi * np.fft.fftfreq(count, d=1.0 / count)
    generator = (
        -float(case.parameters["nu"]) * wave**2 - 1j * float(case.parameters["beta"]) * wave
    )
    modes = np.fft.fft(initial_condition(case), axis=-1)
    phase = np.exp(times[:, None] * generator[None, :])
    return np.fft.ifft(modes[:, None, :] * phase[None, :, :], axis=-1).real


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
    derivative_x = periodic_first(interior, spacing)
    derivative_xx = periodic_second(interior, spacing)
    residual = (
        derivative_t
        + float(case.parameters["beta"]) * derivative_x
        - float(case.parameters["nu"]) * derivative_xx
    )
    return float(np.sqrt(np.mean(np.square(residual))))


def prompt(case: PDECase, *, form: str = "generic") -> str:
    if form not in {"generic", "parameter", "parameter_ic"}:
        raise ValueError(f"unknown prompt form: {form}")
    details = ""
    if form in {"parameter", "parameter_ic"}:
        details += (
            f"\nFor this case beta={case.parameters['beta']} and nu={case.parameters['nu']}."
        )
    if form == "parameter_ic":
        details += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Solve the periodic one-dimensional advection-diffusion equation

    u_t + beta * u_x = nu * u_xx,  x in [0, 1).

Write this function:

    def solver(u0_batch, t_coordinate, beta, nu):
        ...

u0_batch has shape [B, N]. t_coordinate has shape [T]. Return a finite NumPy
array with shape [B, T, N]. Returned slice n must correspond to t_coordinate[n].
Internal substeps are allowed. Do not hard-code a solution array.{details}
"""


class AdvectionDiffusion1DEnvironment:
    family_id = "advection_diffusion1d"
    seen_in_training = False
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "advection_diffusion1d":
            raise ValueError(f"wrong family for advection_diffusion1d: {case.family_id}")
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
            argument_order=("u0_batch", "t_coordinate", "beta", "nu"),
            arrays={"u0_batch": initial_condition(case), "t_coordinate": _times(case)},
            scalars={
                "beta": float(case.parameters["beta"]),
                "nu": float(case.parameters["nu"]),
            },
            expected_shape=(1, int(case.time_count or 0), case.grid_shape[0]),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        return reference_analytic(case)

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        return reference_fft(case)

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        return residual_norm(field, case)
