"""Source-derived periodic KdV held-out environment.

The paper names KdV as held-out but does not publish its equation convention or case contract.
This module states and implements one local parameterized choice without attributing it to the
authors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import rk4_step, time_grid

# The paper names KdV as held-out but does not state its equation or instance distribution.
# This module locks one transparent local parameterization:
#     u_t + alpha * u * u_x + dispersion * u_xxx = 0,  x in [0, 1), periodic.
LOCAL_GRID_SIZES = (64, 96)
LOCAL_TIME_COUNTS = (41, 61)
LOCAL_FINAL_TIME = 0.1
LOCAL_NONLINEAR_COEFFICIENTS = (1.0, 3.0, 6.0)
LOCAL_DISPERSIONS = (0.00025, 0.0005, 0.001)
LOCAL_INPUT_KINDS = ("single_cosine", "two_mode", "periodic_sech_squared")


def _x(case: PDECase) -> np.ndarray:
    return np.linspace(0.0, 1.0, case.grid_shape[0], endpoint=False, dtype=np.float64)


def _t(case: PDECase) -> np.ndarray:
    if case.time_count is None or case.final_time is None:
        raise ValueError("KdV cases require time_count and final_time")
    return time_grid(case.time_count, case.final_time)


def initial_condition(case: PDECase) -> np.ndarray:
    x = _x(case)
    phase = float(case.parameters["phase"])
    amplitude = float(case.parameters["amplitude"])
    offset = float(case.parameters["offset"])
    mix = float(case.parameters["mix"])
    if case.input_kind == "single_cosine":
        values = offset + amplitude * np.cos(2.0 * np.pi * x + phase)
    elif case.input_kind == "two_mode":
        values = offset + amplitude * (
            np.cos(2.0 * np.pi * x + phase) + mix * np.sin(4.0 * np.pi * x - 0.5 * phase)
        )
    elif case.input_kind == "periodic_sech_squared":
        center = float(case.parameters["center"])
        width = float(case.parameters["width"])
        distance = np.minimum(np.abs(x - center), 1.0 - np.abs(x - center))
        values = offset + amplitude / np.cosh(distance / width) ** 2
    else:
        raise ValueError(f"unknown KdV initial condition: {case.input_kind}")
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
        "nonlinear_coefficient": float(rng.choice(LOCAL_NONLINEAR_COEFFICIENTS)),
        "dispersion": float(rng.choice(LOCAL_DISPERSIONS)),
        "phase": float(rng.uniform(-np.pi, np.pi)),
        "amplitude": float(rng.uniform(0.1, 0.3)),
        "offset": float(rng.uniform(-0.08, 0.08)),
        "mix": float(rng.uniform(0.2, 0.5)),
        "center": float(rng.uniform(0.2, 0.8)),
        "width": float(rng.uniform(0.04, 0.09)),
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
        raise ValueError("KdV requires one periodic spatial dimension of size >= 8")
    if time_count < 3 or final_time <= 0.0:
        raise ValueError("KdV requires at least three positive-time samples")
    if float(parameters["dispersion"]) <= 0.0:
        raise ValueError("dispersion must be positive")

    return make_case_record(
        family_id="kdv1d",
        split=split,
        seed=seed,
        grid_shape=grid_shape,
        time_count=time_count,
        final_time=final_time,
        input_kind=input_kind,
        parameters=parameters,
    )


def _spectral_first(values: np.ndarray, wave: np.ndarray) -> np.ndarray:
    return np.fft.ifft(1j * wave * np.fft.fft(values)).real


def _spectral_third(values: np.ndarray, wave: np.ndarray) -> np.ndarray:
    return np.fft.ifft((1j * wave) ** 3 * np.fft.fft(values)).real


def _integrate_spectral(case: PDECase) -> np.ndarray:
    """Strang-split spectral solver with exact linear dispersion."""
    state = initial_condition(case)[0]
    times = _t(case)
    nx = case.grid_shape[0]
    spacing = 1.0 / nx
    alpha = float(case.parameters["nonlinear_coefficient"])
    dispersion = float(case.parameters["dispersion"])
    wave = 2.0 * np.pi * np.fft.fftfreq(nx, d=spacing)
    modes = np.fft.fftfreq(nx) * nx
    dealias = np.abs(modes) <= nx // 3

    def nonlinear_rhs(values: np.ndarray) -> np.ndarray:
        squared_hat = np.fft.fft(values**2)
        squared_hat[~dealias] = 0.0
        return np.fft.ifft(-0.5j * alpha * wave * squared_hat).real

    output = np.empty((len(times), nx), dtype=np.float64)
    output[0] = state
    now = float(times[0])
    for output_index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1e-14:
            nonlinear_limit = 0.15 * spacing / max(alpha * np.max(np.abs(state)), 1e-12)
            step = min(1.0e-3, nonlinear_limit, float(target) - now)
            half_phase = np.exp(0.5j * dispersion * wave**3 * step)
            state = np.fft.ifft(np.fft.fft(state) * half_phase).real
            state = rk4_step(state, step, nonlinear_rhs)
            state = np.fft.ifft(np.fft.fft(state) * half_phase).real
            now += step
        output[output_index] = state
    return output[None, :, :]


def _periodic_first_fourth(values: np.ndarray, spacing: float) -> np.ndarray:
    return (
        np.roll(values, 2)
        - 8.0 * np.roll(values, 1)
        + 8.0 * np.roll(values, -1)
        - np.roll(values, -2)
    ) / (12.0 * spacing)


def _periodic_third(values: np.ndarray, spacing: float) -> np.ndarray:
    """Fourth-order seven-point periodic third derivative."""
    return (
        0.125 * np.roll(values, 3)
        - np.roll(values, 2)
        + 1.625 * np.roll(values, 1)
        - 1.625 * np.roll(values, -1)
        + np.roll(values, -2)
        - 0.125 * np.roll(values, -3)
    ) / spacing**3


def _integrate_finite_difference(case: PDECase) -> np.ndarray:
    """Independent second-order finite-difference and RK4 audit solver."""
    state = initial_condition(case)[0]
    times = _t(case)
    nx = case.grid_shape[0]
    spacing = 1.0 / nx
    alpha = float(case.parameters["nonlinear_coefficient"])
    dispersion = float(case.parameters["dispersion"])

    def rhs(values: np.ndarray) -> np.ndarray:
        nonlinear = -0.5 * alpha * _periodic_first_fourth(values**2, spacing)
        return nonlinear - dispersion * _periodic_third(values, spacing)

    output = np.empty((len(times), nx), dtype=np.float64)
    output[0] = state
    now = float(times[0])
    for output_index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1e-14:
            nonlinear_limit = 0.12 * spacing / max(alpha * np.max(np.abs(state)), 1e-12)
            dispersion_limit = 0.08 * spacing**3 / dispersion
            step = min(nonlinear_limit, dispersion_limit, float(target) - now)
            state = rk4_step(state, step, rhs)
            now += step
        output[output_index] = state
    return output[None, :, :]


class KdV1DEnvironment:
    """Transparent local KdV choice for a paper-named held-out family."""

    family_id = "kdv1d"
    seen_in_training = False
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "kdv1d":
            raise ValueError(f"wrong family for kdv1d: {case.family_id}")
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
                "\nFor this task, nonlinear_coefficient="
                f"{case.parameters['nonlinear_coefficient']} and dispersion="
                f"{case.parameters['dispersion']}."
            )
        if form == "parameter_ic":
            conditioning += f"\nThe initial-condition family is {case.input_kind}."
        return (
            "Write a robust numerical solver.\n"
            "Use this source-derived KdV convention on x in [0, 1): "
            "u_t + nonlinear_coefficient * u * u_x + dispersion * u_xxx = 0, with "
            "periodic boundaries. Define solver(u0_batch, t_coordinate, "
            "nonlinear_coefficient, dispersion). u0_batch has shape (B, N). Return a finite "
            "NumPy array with shape (B, len(t_coordinate), N), including the initial state. "
            f"Do not hard-code a solution array.{conditioning}"
        )

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=(
                "u0_batch",
                "t_coordinate",
                "nonlinear_coefficient",
                "dispersion",
            ),
            arrays={"u0_batch": initial_condition(case), "t_coordinate": _t(case)},
            scalars={
                "nonlinear_coefficient": float(case.parameters["nonlinear_coefficient"]),
                "dispersion": float(case.parameters["dispersion"]),
            },
            expected_shape=(1, len(_t(case)), case.grid_shape[0]),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        return _integrate_spectral(case)

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        return _integrate_finite_difference(case)

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        values = np.asarray(field, dtype=np.float64)
        if values.shape != (1, len(_t(case)), case.grid_shape[0]):
            raise ValueError("field does not match the KdV case shape")
        spacing = 1.0 / case.grid_shape[0]
        wave = 2.0 * np.pi * np.fft.fftfreq(case.grid_shape[0], d=spacing)
        alpha = float(case.parameters["nonlinear_coefficient"])
        dispersion = float(case.parameters["dispersion"])
        times = _t(case)
        interior = values[:, 1:-1]
        temporal = (values[:, 2:] - values[:, :-2]) / (times[2:] - times[:-2])[None, :, None]
        first = np.apply_along_axis(_spectral_first, -1, interior, wave)
        third = np.apply_along_axis(_spectral_third, -1, interior, wave)
        residual = temporal + alpha * interior * first + dispersion * third
        return float(np.sqrt(np.mean(residual**2)))
