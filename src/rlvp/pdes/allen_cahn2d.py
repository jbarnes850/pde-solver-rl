from __future__ import annotations

"""Source-derived held-out 2D Allen--Cahn environment.

ArXiv:2607.10474v1 reports that Allen--Cahn 2D is held out and captions one example
with a mixed-waves initial field, epsilon squared 0.005, and rho 2. The equation
convention, full parameter set, domain, boundary condition, grids, times, call
signature, references, and residual are not reported. All such settings below are
explicit source-derived choices and are excluded from SFT and RL training.
"""

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import integrate_to_times, periodic_second, time_grid

SOURCE_DERIVED_EPSILON_SQUARED = (0.001, 0.005, 0.01)
SOURCE_DERIVED_RHO = (1.0, 2.0, 5.0)
SOURCE_DERIVED_GRID_SIZES = (32, 48)
SOURCE_DERIVED_TIME_COUNTS = (31, 51)
SOURCE_DERIVED_FINAL_TIME = 1.0
SOURCE_DERIVED_INPUT_KINDS = ("mixed_waves", "smooth_random", "droplet")
SOURCE_DERIVED_PRIMARY_MAX_STEP = 0.005
SOURCE_DERIVED_AUDIT_MAX_STEP = 0.0025


def _required_time_data(case: PDECase) -> tuple[int, float]:
    if case.time_count is None or case.final_time is None:
        raise ValueError("allen_cahn2d requires time_count and final_time")
    return case.time_count, case.final_time


def times(case: PDECase) -> np.ndarray:
    count, final_time = _required_time_data(case)
    return time_grid(count, final_time)


def initial_state(case: PDECase) -> np.ndarray:
    n = case.grid_shape[0]
    coordinate = np.linspace(-1.0, 1.0, n, endpoint=False, dtype=np.float64)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    rng = np.random.default_rng(case.seed)
    if case.input_kind == "mixed_waves":
        phase = float(rng.uniform(-np.pi, np.pi))
        field = 0.42 * np.sin(np.pi * xx + phase) * np.cos(2.0 * np.pi * yy)
        field += 0.26 * np.sin(2.0 * np.pi * (xx + yy) - 0.5 * phase)
    elif case.input_kind == "smooth_random":
        noise = rng.standard_normal((n, n))
        spectrum = np.fft.fft2(noise)
        frequency = np.fft.fftfreq(n)
        radius_squared = frequency[:, None] ** 2 + frequency[None, :] ** 2
        field = np.fft.ifft2(spectrum * np.exp(-120.0 * radius_squared)).real
        field = 0.65 * field / (np.max(np.abs(field)) + 1e-12)
    elif case.input_kind == "droplet":
        radius = np.sqrt((xx - 0.15) ** 2 + (yy + 0.10) ** 2)
        field = np.tanh((0.42 - radius) / 0.08)
    else:
        raise ValueError(f"unknown allen_cahn2d input kind: {case.input_kind}")
    return field[None, :, :]


def _reaction(field: np.ndarray, rho: float) -> np.ndarray:
    return rho * (field - field**3)


def _fd_rhs(field: np.ndarray, *, epsilon_squared: float, rho: float, dx: float) -> np.ndarray:
    laplacian = periodic_second(field, dx, axis=0) + periodic_second(field, dx, axis=1)
    return epsilon_squared * laplacian + _reaction(field, rho)


def _primary_reference(case: PDECase) -> np.ndarray:
    field = initial_state(case)[0]
    dx = 2.0 / case.grid_shape[0]
    epsilon_squared = float(case.parameters["epsilon_squared"])
    rho = float(case.parameters["rho"])
    stable_step = min(
        SOURCE_DERIVED_PRIMARY_MAX_STEP,
        0.15 * dx**2 / epsilon_squared,
        0.1 / rho,
    )
    result = integrate_to_times(
        field,
        times(case),
        lambda value: _fd_rhs(value, epsilon_squared=epsilon_squared, rho=rho, dx=dx),
        stable_step,
    )
    return result[None, ...]


def _exact_reaction(field: np.ndarray, step: float, rho: float) -> np.ndarray:
    square = field**2
    denominator = np.sqrt(square + (1.0 - square) * np.exp(-2.0 * rho * step))
    return field / np.maximum(denominator, 1e-14)


def _exact_periodic_diffusion(
    field: np.ndarray, step: float, epsilon_squared: float
) -> np.ndarray:
    n = field.shape[0]
    wave = 2.0 * np.pi * np.fft.fftfreq(n, d=2.0 / n)
    laplacian_eigenvalues = -(wave[:, None] ** 2 + wave[None, :] ** 2)
    spectrum = np.fft.fft2(field)
    return np.fft.ifft2(spectrum * np.exp(epsilon_squared * step * laplacian_eigenvalues)).real


def _audit_reference(case: PDECase) -> np.ndarray:
    """Independent Strang path with exact local reaction and spectral diffusion."""
    field = initial_state(case)[0].copy()
    output_times = times(case)
    output = np.empty((len(output_times), *field.shape), dtype=np.float64)
    output[0] = field
    epsilon_squared = float(case.parameters["epsilon_squared"])
    rho = float(case.parameters["rho"])
    now = float(output_times[0])
    for output_index, target in enumerate(output_times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(SOURCE_DERIVED_AUDIT_MAX_STEP, float(target) - now)
            field = _exact_reaction(field, 0.5 * step, rho)
            field = _exact_periodic_diffusion(field, step, epsilon_squared)
            field = _exact_reaction(field, 0.5 * step, rho)
            now += step
        output[output_index] = field
    return output[None, ...]


def make_case(
    *, seed: int, split: str = "heldout", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {
        "epsilon_squared": float(rng.choice(SOURCE_DERIVED_EPSILON_SQUARED)),
        "rho": float(rng.choice(SOURCE_DERIVED_RHO)),
        "n": int(rng.choice(SOURCE_DERIVED_GRID_SIZES)),
        "time_count": int(rng.choice(SOURCE_DERIVED_TIME_COUNTS)),
        "final_time": SOURCE_DERIVED_FINAL_TIME,
        "input_kind": str(rng.choice(SOURCE_DERIVED_INPUT_KINDS)),
    }
    if stratum:
        unknown = set(stratum) - set(values)
        if unknown:
            raise ValueError(f"unknown allen_cahn2d stratum keys: {sorted(unknown)}")
        values.update(stratum)
    return make_case_record(
        family_id="allen_cahn2d",
        split=split,
        seed=seed,
        grid_shape=(int(values["n"]), int(values["n"])),
        time_count=int(values["time_count"]),
        final_time=float(values["final_time"]),
        input_kind=str(values["input_kind"]),
        parameters={
            "epsilon_squared": float(values["epsilon_squared"]),
            "rho": float(values["rho"]),
        },
    )


class AllenCahn2DEnvironment:
    family_id = "allen_cahn2d"
    seen_in_training = False
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        return PDECase.from_dict(value)

    @staticmethod
    def make_case(
        *, seed: int, split: str = "heldout", stratum: dict[str, Any] | None = None
    ) -> PDECase:
        return make_case(seed=seed, split=split, stratum=stratum)

    @staticmethod
    def prompt(case: PDECase, *, form: str = "generic") -> str:
        conditioning = ""
        if form in {"parameter", "parameter_ic"}:
            conditioning += (
                f"\nFor this task, epsilon_squared={case.parameters['epsilon_squared']} "
                f"and rho={case.parameters['rho']}."
            )
        if form == "parameter_ic":
            conditioning += f"\nThe initial-field family is {case.input_kind}."
        if form not in {"generic", "parameter", "parameter_ic"}:
            raise ValueError(f"unknown prompt form: {form}")
        return f"""Solve the periodic two-dimensional Allen--Cahn equation

    u_t = epsilon_squared * Laplacian(u) + rho * (u - u**3)

on (x, y) in [-1, 1)^2.

Write this function:

    def solver(u0_batch, t_coordinate, epsilon_squared, rho):
        ...

u0_batch has shape [B, Nx, Ny]. t_coordinate has shape [T]. Return a finite
NumPy array with shape [B, T, Nx, Ny], aligned with the requested times.
Internal substeps are allowed. Do not hard-code a solution array.{conditioning}
"""

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("u0_batch", "t_coordinate", "epsilon_squared", "rho"),
            arrays={"u0_batch": initial_state(case), "t_coordinate": times(case)},
            scalars={
                "epsilon_squared": float(case.parameters["epsilon_squared"]),
                "rho": float(case.parameters["rho"]),
            },
            expected_shape=(1, len(times(case)), *case.grid_shape),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        return _primary_reference(case)

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        return _audit_reference(case)

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        values = np.asarray(field, dtype=np.float64)
        output_times = times(case)
        edge_order = 2 if len(output_times) >= 3 else 1
        derivative = np.gradient(values, output_times, axis=1, edge_order=edge_order)
        dx = 2.0 / case.grid_shape[0]
        laplacian = periodic_second(values, dx, axis=2) + periodic_second(values, dx, axis=3)
        residual = derivative - (
            float(case.parameters["epsilon_squared"]) * laplacian
            + float(case.parameters["rho"]) * (values - values**3)
        )
        return float(np.sqrt(np.mean(residual**2)))
