"""Source-derived periodic Cahn-Hilliard held-out environment.

The paper gives the chemical potential and identifies this held-out family. It does not give
the case distribution, solver interface, reference method, or residual discretization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import periodic_grid, time_grid

# The paper gives the chemical potential but not the held-out instance distribution.
# Every value below is therefore an explicit source-derived choice, not an author artifact.
LOCAL_GRID_SIZES = (64, 96)
LOCAL_TIME_COUNTS = (41, 61)
LOCAL_FINAL_TIME = 0.05
LOCAL_EPSILON_SQUARED = (0.0025, 0.005, 0.01)
LOCAL_INPUT_KINDS = ("cosine_mix", "random_fourier", "periodic_pulse")


def _x(case: PDECase) -> np.ndarray:
    return periodic_grid(case.grid_shape[0])


def _t(case: PDECase) -> np.ndarray:
    if case.time_count is None or case.final_time is None:
        raise ValueError("Cahn-Hilliard cases require time_count and final_time")
    return time_grid(case.time_count, case.final_time)


def initial_condition(case: PDECase) -> np.ndarray:
    """Return one smooth periodic initial condition with shape (1, nx)."""
    x = _x(case)
    phase = float(case.parameters["phase"])
    amplitude = float(case.parameters["amplitude"])
    mean = float(case.parameters["mean"])
    mix = float(case.parameters["mix"])

    if case.input_kind == "cosine_mix":
        values = mean + amplitude * (
            np.cos(2.0 * np.pi * x + phase) + mix * np.cos(6.0 * np.pi * x - 0.5 * phase)
        )
    elif case.input_kind == "random_fourier":
        rng = np.random.default_rng(case.seed ^ 0xC4A4)
        coefficients = rng.normal(size=4)
        coefficients /= max(float(np.linalg.norm(coefficients)), np.finfo(float).eps)
        values = np.full_like(x, mean)
        for mode, coefficient in enumerate(coefficients, start=1):
            values += amplitude * coefficient * np.cos(2.0 * np.pi * mode * x + phase * (mode + 1))
    elif case.input_kind == "periodic_pulse":
        center = float(case.parameters["center"])
        width = float(case.parameters["width"])
        distance = np.minimum(np.abs(x - center), 1.0 - np.abs(x - center))
        pulse = np.exp(-0.5 * (distance / width) ** 2)
        values = mean + amplitude * (pulse - np.mean(pulse))
    else:
        raise ValueError(f"unknown Cahn-Hilliard initial condition: {case.input_kind}")
    return values[None, :]


def make_case(
    *, seed: int, split: str = "heldout", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    grid_shape = (int(rng.choice(LOCAL_GRID_SIZES)),)
    time_count = int(rng.choice(LOCAL_TIME_COUNTS))
    final_time = LOCAL_FINAL_TIME
    input_kind = str(rng.choice(LOCAL_INPUT_KINDS))
    parameters: dict[str, int | float | str] = {
        "epsilon_squared": float(rng.choice(LOCAL_EPSILON_SQUARED)),
        "phase": float(rng.uniform(-np.pi, np.pi)),
        "amplitude": float(rng.uniform(0.08, 0.22)),
        "mean": float(rng.uniform(-0.08, 0.08)),
        "mix": float(rng.uniform(0.2, 0.5)),
        "center": float(rng.uniform(0.2, 0.8)),
        "width": float(rng.uniform(0.06, 0.13)),
    }

    if stratum:
        if "grid_shape" in stratum:
            grid_shape = tuple(int(item) for item in stratum["grid_shape"])
        elif "nx" in stratum:
            grid_shape = (int(stratum["nx"]),)
        if "time_count" in stratum:
            time_count = int(stratum["time_count"])
        elif "nt" in stratum:
            time_count = int(stratum["nt"])
        if "final_time" in stratum:
            final_time = float(stratum["final_time"])
        if "input_kind" in stratum:
            input_kind = str(stratum["input_kind"])
        parameter_overrides = stratum.get("parameters", {})
        if not isinstance(parameter_overrides, dict):
            raise TypeError("stratum parameters must be a mapping")
        parameters.update(parameter_overrides)
        for name in parameters:
            if name in stratum:
                parameters[name] = stratum[name]

    if len(grid_shape) != 1 or grid_shape[0] < 8:
        raise ValueError("Cahn-Hilliard requires one periodic spatial dimension of size >= 8")
    if time_count < 3 or final_time <= 0.0:
        raise ValueError("Cahn-Hilliard requires at least three positive-time samples")
    if float(parameters["epsilon_squared"]) <= 0.0:
        raise ValueError("epsilon_squared must be positive")

    return make_case_record(
        family_id="cahn_hilliard1d",
        split=split,
        seed=seed,
        grid_shape=grid_shape,
        time_count=time_count,
        final_time=final_time,
        input_kind=input_kind,
        parameters=parameters,
    )


def _integrate_imex(case: PDECase, *, finite_difference_symbol: bool) -> np.ndarray:
    """Integrate with a stable semi-implicit periodic method.

    The primary uses the Fourier Laplacian. The audit uses the second-order finite-difference
    Laplacian symbol. FFTs only diagonalize the audit's periodic linear system.
    """
    state = initial_condition(case)[0]
    times = _t(case)
    nx = case.grid_shape[0]
    dx = 1.0 / nx
    epsilon_squared = float(case.parameters["epsilon_squared"])
    modes = np.fft.fftfreq(nx) * nx
    if finite_difference_symbol:
        q = 4.0 * np.sin(np.pi * modes / nx) ** 2 / dx**2
        maximum_step = 1.25e-4
    else:
        wave = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        q = wave**2
        maximum_step = 2.0e-4
    linear = q - epsilon_squared * q**2
    dealias = np.abs(modes) <= nx // 3

    output = np.empty((len(times), nx), dtype=np.float64)
    output[0] = state
    now = float(times[0])
    for output_index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(maximum_step, float(target) - now)
            state_hat = np.fft.fft(state)
            cubic_hat = np.fft.fft(state**3)
            cubic_hat[~dealias] = 0.0
            state_hat = (state_hat - step * q * cubic_hat) / (1.0 - step * linear)
            state = np.fft.ifft(state_hat).real
            now += step
        output[output_index] = state
    return output[None, :, :]


def _spectral_laplacian(values: np.ndarray, spacing: float) -> np.ndarray:
    wave = 2.0 * np.pi * np.fft.fftfreq(values.shape[-1], d=spacing)
    return np.fft.ifft(-(wave**2) * np.fft.fft(values, axis=-1), axis=-1).real


class CahnHilliard1DEnvironment:
    """Source-derived periodic realization of a paper-visible held-out PDE."""

    family_id = "cahn_hilliard1d"
    seen_in_training = False
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "cahn_hilliard1d":
            raise ValueError(f"wrong family for cahn_hilliard1d: {case.family_id}")
        return case

    @staticmethod
    def make_case(
        *, seed: int, split: str = "heldout", stratum: dict[str, Any] | None = None
    ) -> PDECase:
        return make_case(seed=seed, split=split, stratum=stratum)

    @staticmethod
    def prompt(case: PDECase, *, form: str = "generic") -> str:
        if form not in {"generic", "parameter", "parameter_ic"}:
            raise ValueError(f"unknown prompt form: {form}")
        conditioning = ""
        if form in {"parameter", "parameter_ic"}:
            conditioning += (
                f"\nFor this task, epsilon_squared={case.parameters['epsilon_squared']}."
            )
        if form == "parameter_ic":
            conditioning += f"\nThe initial-condition family is {case.input_kind}."
        return (
            "Write a robust numerical solver.\n"
            "Solve the one-dimensional periodic Cahn-Hilliard equation u_t = mu_xx on "
            "x in [0, 1), where mu = -epsilon_squared * u_xx + u**3 - u. "
            "Define solver(u0_batch, t_coordinate, epsilon_squared). u0_batch has shape "
            "(B, N). Return a finite NumPy array with shape (B, len(t_coordinate), N), "
            f"including the initial state. Do not hard-code a solution array.{conditioning}"
        )

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("u0_batch", "t_coordinate", "epsilon_squared"),
            arrays={"u0_batch": initial_condition(case), "t_coordinate": _t(case)},
            scalars={"epsilon_squared": float(case.parameters["epsilon_squared"])},
            expected_shape=(1, len(_t(case)), case.grid_shape[0]),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        return _integrate_imex(case, finite_difference_symbol=False)

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        return _integrate_imex(case, finite_difference_symbol=True)

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        values = np.asarray(field, dtype=np.float64)
        if values.shape != (1, len(_t(case)), case.grid_shape[0]):
            raise ValueError("field does not match the Cahn-Hilliard case shape")
        spacing = 1.0 / case.grid_shape[0]
        epsilon_squared = float(case.parameters["epsilon_squared"])
        times = _t(case)
        interior = values[:, 1:-1]
        temporal = (values[:, 2:] - values[:, :-2]) / (times[2:] - times[:-2])[None, :, None]
        laplacian = _spectral_laplacian(interior, spacing)
        chemical_potential = -epsilon_squared * laplacian + interior**3 - interior
        residual = temporal - _spectral_laplacian(chemical_potential, spacing)
        return float(np.sqrt(np.mean(residual**2)))
