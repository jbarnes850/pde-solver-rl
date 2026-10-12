"""Source-derived held-out 2D linear-advection environment.

Paper fact: the transfer figure gives ``u_t + beta_x*u_x + beta_y*u_y = 0``.
The paper does not publish the held-out domain, boundary conditions, parameter grid,
initial-condition generator, case manifest, reference solver, residual discretization, or
solver signature. This module therefore makes explicit local choices: the periodic unit
torus, smooth Fourier-mode inputs, fixed parameter grids, and the array interface below.
These choices reproduce the stated PDE family, not the authors' hidden artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import periodic_first, periodic_grid, time_grid

LOCAL_GRID_SIZES = (32, 48)
LOCAL_TIME_COUNTS = (31, 51)
LOCAL_FINAL_TIME = 1.0
LOCAL_BETA_VALUES = (-1.0, -0.4, 0.4, 1.0)


def make_case(*, seed: int, split: str = "eval", stratum: dict[str, Any] | None = None) -> PDECase:
    """Make one deterministic case from source-derived local parameter grids."""
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {
        "grid_shape": (int(rng.choice(LOCAL_GRID_SIZES)), int(rng.choice(LOCAL_GRID_SIZES))),
        "time_count": int(rng.choice(LOCAL_TIME_COUNTS)),
        "final_time": LOCAL_FINAL_TIME,
        "input_kind": "mixed_periodic_modes",
        "beta_x": float(rng.choice(LOCAL_BETA_VALUES)),
        "beta_y": float(rng.choice(LOCAL_BETA_VALUES)),
        "phase_x": float(rng.uniform(0.0, 1.0)),
        "phase_y": float(rng.uniform(0.0, 1.0)),
        "amplitude_x": float(rng.uniform(0.25, 0.65)),
        "amplitude_y": float(rng.uniform(0.20, 0.55)),
        "amplitude_xy": float(rng.uniform(0.10, 0.35)),
        "offset": float(rng.uniform(-0.15, 0.15)),
    }
    if stratum:
        unknown = set(stratum) - set(values)
        if unknown:
            raise ValueError(f"unknown advection2d stratum keys: {sorted(unknown)}")
        values.update(stratum)
    parameters = {
        key: values[key]
        for key in (
            "beta_x",
            "beta_y",
            "phase_x",
            "phase_y",
            "amplitude_x",
            "amplitude_y",
            "amplitude_xy",
            "offset",
        )
    }
    return make_case_record(
        family_id="advection2d",
        split=split,
        seed=seed,
        grid_shape=tuple(int(item) for item in values["grid_shape"]),
        time_count=int(values["time_count"]),
        final_time=float(values["final_time"]),
        input_kind=str(values["input_kind"]),
        parameters=parameters,
    )


def _coordinates(case: PDECase) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = case.grid_shape
    return periodic_grid(nx), periodic_grid(ny)


def _times(case: PDECase) -> np.ndarray:
    assert case.time_count is not None and case.final_time is not None
    return time_grid(case.time_count, case.final_time)


def initial_condition(case: PDECase) -> np.ndarray:
    """Return the local smooth periodic input with shape ``[1, Nx, Ny]``."""
    x, y = _coordinates(case)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    p = case.parameters
    values = (
        float(p["offset"])
        + float(p["amplitude_x"]) * np.sin(2.0 * np.pi * (xx - float(p["phase_x"])))
        + float(p["amplitude_y"]) * np.cos(2.0 * np.pi * (yy - float(p["phase_y"])))
        + float(p["amplitude_xy"])
        * np.sin(2.0 * np.pi * (xx + yy - float(p["phase_x"]) - float(p["phase_y"])))
    )
    return values[None, :, :]


def reference_analytic(case: PDECase) -> np.ndarray:
    """Evaluate the translated analytic Fourier modes at every requested time."""
    x, y = _coordinates(case)
    times = _times(case)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    p = case.parameters
    shifted_x = xx[None, :, :] - float(p["beta_x"]) * times[:, None, None]
    shifted_y = yy[None, :, :] - float(p["beta_y"]) * times[:, None, None]
    values = (
        float(p["offset"])
        + float(p["amplitude_x"]) * np.sin(2.0 * np.pi * (shifted_x - float(p["phase_x"])))
        + float(p["amplitude_y"]) * np.cos(2.0 * np.pi * (shifted_y - float(p["phase_y"])))
        + float(p["amplitude_xy"])
        * np.sin(2.0 * np.pi * (shifted_x + shifted_y - float(p["phase_x"]) - float(p["phase_y"])))
    )
    return values[None, :, :, :]


def reference_fft(case: PDECase) -> np.ndarray:
    """Independently translate the sampled input through its discrete Fourier spectrum."""
    nx, ny = case.grid_shape
    times = _times(case)
    modes = np.fft.fftn(initial_condition(case), axes=(-2, -1))
    wave_x = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    wave_y = 2.0 * np.pi * np.fft.fftfreq(ny, d=1.0 / ny)
    phase = np.exp(
        -1j
        * times[:, None, None]
        * (
            float(case.parameters["beta_x"]) * wave_x[:, None]
            + float(case.parameters["beta_y"]) * wave_y[None, :]
        )
    )
    return np.fft.ifftn(modes[:, None, :, :] * phase[None, :, :, :], axes=(-2, -1)).real


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    """Compute a centered discrete RMS residual on interior output times."""
    candidate = np.asarray(field, dtype=np.float64)
    expected = (1, int(case.time_count or 0), *case.grid_shape)
    if candidate.shape != expected:
        raise ValueError(f"expected field shape {expected}, got {candidate.shape}")
    times = _times(case)
    nx, ny = case.grid_shape
    interior = candidate[:, 1:-1]
    dt = (times[2:] - times[:-2])[None, :, None, None]
    derivative_t = (candidate[:, 2:] - candidate[:, :-2]) / dt
    derivative_x = periodic_first(interior, 1.0 / nx, axis=-2)
    derivative_y = periodic_first(interior, 1.0 / ny, axis=-1)
    residual = (
        derivative_t
        + float(case.parameters["beta_x"]) * derivative_x
        + float(case.parameters["beta_y"]) * derivative_y
    )
    return float(np.sqrt(np.mean(np.square(residual))))


def prompt(case: PDECase, *, form: str = "generic") -> str:
    if form not in {"generic", "parameter", "parameter_ic"}:
        raise ValueError(f"unknown prompt form: {form}")
    details = ""
    if form in {"parameter", "parameter_ic"}:
        details += (
            f"\nFor this case beta_x={case.parameters['beta_x']} and "
            f"beta_y={case.parameters['beta_y']}."
        )
    if form == "parameter_ic":
        details += f"\nThe initial-condition family is {case.input_kind}."
    return f"""Solve the periodic two-dimensional linear advection equation

    u_t + beta_x * u_x + beta_y * u_y = 0,  (x, y) in [0, 1)^2.

Write this function:

    def solver(u0_batch, t_coordinate, beta_x, beta_y):
        ...

u0_batch has shape [B, Nx, Ny]. t_coordinate has shape [T]. Return a finite
NumPy array with shape [B, T, Nx, Ny]. Returned slice n must correspond to
t_coordinate[n]. Internal substeps are allowed. Do not hard-code a solution array.{details}
"""


class Advection2DEnvironment:
    family_id = "advection2d"
    seen_in_training = False
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "advection2d":
            raise ValueError(f"wrong family for advection2d: {case.family_id}")
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
            argument_order=("u0_batch", "t_coordinate", "beta_x", "beta_y"),
            arrays={"u0_batch": initial_condition(case), "t_coordinate": _times(case)},
            scalars={
                "beta_x": float(case.parameters["beta_x"]),
                "beta_y": float(case.parameters["beta_y"]),
            },
            expected_shape=(1, int(case.time_count or 0), *case.grid_shape),
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
