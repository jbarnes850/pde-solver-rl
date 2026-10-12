"""Shared pieces for the periodic linear 1-D families advection1d_domain, airy1d, hyperdiffusion1d.

Every initial condition is defined on the normalized coordinate s = (x - x_min) / L in [0, 1)
and has closed-form Fourier coefficients. The primary references therefore evaluate the exact
solution of the continuous problem at the grid points, with no FFT of sampled data. The audit
references use independent routes: an FFT of the sampled initial state for advection, and the
matrix exponential of a physical-space spectral differentiation matrix (no hand-written Fourier
symbol) for the dispersive and fourth-order families.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.linalg import expm

IC_KINDS = ("fourier_series", "gaussian_bumps", "sine_sum")
TIME_GRID_KINDS = ("uniform", "stretched")
PROMPT_FORMS = ("generic", "parameter", "parameter_ic")

# Child stream for initial-condition draws, independent of the discrete selections.
IC_STREAM = 17
# Wrapped Gaussians sum images n = -3..3; with width <= 0.12 the first omitted image is
# below exp(-130).
GAUSSIAN_IMAGES = 3
GAUSSIAN_WIDTH_RANGE = (0.05, 0.12)
FOURIER_MODE_COUNTS = (3, 4, 5, 6)
SINE_EXTRA_MODES = (2, 3, 4)
TWO_PI = 2.0 * math.pi


def significant(value: float, digits: int = 6) -> float:
    return float(f"{value:.{digits}g}")


def draw_initial_condition(seed: int, kind: str) -> dict[str, int | float]:
    """Draw continuous initial-condition parameters from a seed-derived child stream."""
    rng = np.random.default_rng([seed, IC_STREAM])
    parameters: dict[str, int | float] = {"offset": float(rng.uniform(-0.3, 0.3))}
    if kind == "fourier_series":
        count = int(rng.choice(FOURIER_MODE_COUNTS))
        decay = float(rng.uniform(1.0, 2.0))
        scale = float(rng.uniform(0.6, 1.2))
        parameters["modes"] = count
        for mode in range(1, count + 1):
            parameters[f"c{mode}_amp"] = scale * mode**-decay * float(rng.uniform(0.6, 1.0))
            parameters[f"c{mode}_phase"] = float(rng.uniform(-math.pi, math.pi))
        return parameters
    if kind == "gaussian_bumps":
        count = int(rng.choice((1, 2, 3)))
        parameters["bumps"] = count
        for index in range(1, count + 1):
            sign = 1.0 if index == 1 else float(rng.choice((-1.0, 1.0)))
            parameters[f"b{index}_height"] = sign * float(rng.uniform(0.5, 1.2))
            parameters[f"b{index}_center"] = float(rng.uniform(0.0, 1.0))
            parameters[f"b{index}_width"] = float(rng.uniform(*GAUSSIAN_WIDTH_RANGE))
        return parameters
    if kind == "sine_sum":
        count = int(rng.choice((2, 3)))
        extra = [int(value) for value in rng.permutation(SINE_EXTRA_MODES)[: count - 1]]
        parameters["terms"] = count
        for index, mode in enumerate([1, *extra], start=1):
            low, high = (0.6, 1.2) if index == 1 else (0.2, 0.6)
            parameters[f"s{index}_mode"] = mode
            parameters[f"s{index}_amp"] = float(rng.uniform(low, high))
            parameters[f"s{index}_phase"] = float(rng.uniform(-math.pi, math.pi))
        return parameters
    raise ValueError(f"unknown initial-condition kind: {kind}")


def _p(parameters: dict[str, Any], name: str) -> float:
    return float(parameters[name])


def evaluate_initial(s: np.ndarray, kind: str, parameters: dict[str, Any]) -> np.ndarray:
    """Physical-space evaluation of the continuous initial condition at normalized s."""
    s = np.mod(np.asarray(s, dtype=np.float64), 1.0)
    values = np.full_like(s, _p(parameters, "offset"))
    if kind == "fourier_series":
        for mode in range(1, int(parameters["modes"]) + 1):
            values += _p(parameters, f"c{mode}_amp") * np.cos(
                TWO_PI * mode * s + _p(parameters, f"c{mode}_phase")
            )
        return values
    if kind == "gaussian_bumps":
        for index in range(1, int(parameters["bumps"]) + 1):
            height = _p(parameters, f"b{index}_height")
            center = _p(parameters, f"b{index}_center")
            width = _p(parameters, f"b{index}_width")
            for image in range(-GAUSSIAN_IMAGES, GAUSSIAN_IMAGES + 1):
                values += height * np.exp(-((s - center + image) ** 2) / (2.0 * width**2))
        return values
    if kind == "sine_sum":
        for index in range(1, int(parameters["terms"]) + 1):
            values += _p(parameters, f"s{index}_amp") * np.sin(
                TWO_PI * int(parameters[f"s{index}_mode"]) * s + _p(parameters, f"s{index}_phase")
            )
        return values
    raise ValueError(f"unknown initial-condition kind: {kind}")


def fourier_coefficients(kind: str, parameters: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Exact coefficients c_m with u0(s) = sum_m c_m exp(2*pi*i*m*s), from the closed forms."""
    coefficients: dict[int, complex] = {0: complex(_p(parameters, "offset"))}

    def add(mode: int, value: complex) -> None:
        coefficients[mode] = coefficients.get(mode, 0.0) + value

    if kind == "fourier_series":
        for mode in range(1, int(parameters["modes"]) + 1):
            amplitude = _p(parameters, f"c{mode}_amp")
            phase = _p(parameters, f"c{mode}_phase")
            add(mode, 0.5 * amplitude * np.exp(1j * phase))
            add(-mode, 0.5 * amplitude * np.exp(-1j * phase))
    elif kind == "gaussian_bumps":
        widths = [_p(parameters, f"b{i}_width") for i in range(1, int(parameters["bumps"]) + 1)]
        # exp(-2 pi^2 m^2 w^2) < 1e-20 beyond this cutoff for every bump.
        cutoff = max(8, math.ceil(1.6 / min(widths)))
        modes = np.arange(-cutoff, cutoff + 1)
        for index in range(1, int(parameters["bumps"]) + 1):
            height = _p(parameters, f"b{index}_height")
            center = _p(parameters, f"b{index}_center")
            width = _p(parameters, f"b{index}_width")
            values = (
                height
                * width
                * math.sqrt(TWO_PI)
                * np.exp(-2.0 * math.pi**2 * modes**2 * width**2)
                * np.exp(-1j * TWO_PI * modes * center)
            )
            for mode, value in zip(modes.tolist(), values.tolist(), strict=True):
                add(int(mode), complex(value))
    elif kind == "sine_sum":
        for index in range(1, int(parameters["terms"]) + 1):
            mode = int(parameters[f"s{index}_mode"])
            amplitude = _p(parameters, f"s{index}_amp")
            phase = _p(parameters, f"s{index}_phase")
            # sin(theta) = (e^{i theta} - e^{-i theta}) / (2 i)
            add(mode, amplitude * np.exp(1j * phase) / 2j)
            add(-mode, -amplitude * np.exp(-1j * phase) / 2j)
    else:
        raise ValueError(f"unknown initial-condition kind: {kind}")
    modes = np.array(sorted(coefficients), dtype=np.int64)
    return modes, np.array([coefficients[int(m)] for m in modes], dtype=np.complex128)


def time_coordinate(kind: str, count: int, final_time: float) -> np.ndarray:
    fraction = np.linspace(0.0, 1.0, count, dtype=np.float64)
    if kind == "uniform":
        return final_time * fraction
    if kind == "stretched":
        # Monotone, starts at 0, ends at final_time; the last interval is 3x the first.
        times = final_time * (0.5 * fraction + 0.5 * fraction**2)
        times[-1] = final_time
        return times
    raise ValueError(f"unknown time-grid kind: {kind}")


def modal_trajectory(
    modes: np.ndarray,
    coefficients: np.ndarray,
    s_grid: np.ndarray,
    times: np.ndarray,
    exponent: np.ndarray,
) -> np.ndarray:
    """Exact modal solution u(s_j, t_n) = Re sum_m c_m exp(exponent_m * t_n) exp(2 pi i m s_j)."""
    basis = np.exp(1j * TWO_PI * np.outer(modes, s_grid))
    evolved = coefficients[None, :] * np.exp(np.outer(times, exponent))
    return np.real(evolved @ basis)[None, :, :]


def spectral_differentiation_matrices(count: int, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Physical-space Fourier differentiation matrices (first and second derivative).

    Closed-form cotangent / cosecant entries (Trefethen, Spectral Methods in MATLAB, ch. 3)
    on an even periodic grid of period ``length``; no FFT and no hand-written symbol.
    """
    if count % 2:
        raise ValueError("spectral differentiation matrices require an even grid")
    spacing = TWO_PI / count
    index = np.arange(count)
    offset = index[:, None] - index[None, :]
    off_diagonal = offset != 0
    sign = np.where(offset % 2 == 0, 1.0, -1.0)
    first = np.zeros((count, count), dtype=np.float64)
    second = np.zeros((count, count), dtype=np.float64)
    half_angle = offset[off_diagonal] * spacing / 2.0
    first[off_diagonal] = 0.5 * sign[off_diagonal] / np.tan(half_angle)
    second[off_diagonal] = -0.5 * sign[off_diagonal] / np.sin(half_angle) ** 2
    second[~off_diagonal] = -(math.pi**2) / (3.0 * spacing**2) - 1.0 / 6.0
    scale = TWO_PI / length
    return first * scale, second * scale**2


def matrix_exponential_trajectory(
    operator: np.ndarray, initial: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """u(t_n) = expm(t_n * operator) @ u0 for every output time (no step accumulation)."""
    propagators = expm(times[:, None, None] * operator[None, :, :])
    return np.einsum("tij,j->ti", propagators, initial)[None, :, :]


def spectral_derivative(values: np.ndarray, length: float, order: int) -> np.ndarray:
    """Periodic Fourier derivative along the last axis with wavenumbers 2*pi*m/length."""
    count = values.shape[-1]
    wave = TWO_PI * np.fft.rfftfreq(count, d=length / count)
    factor = (1j * wave) ** order
    if order % 2 and count % 2 == 0:
        factor[-1] = 0.0
    return np.fft.irfft(np.fft.rfft(values, axis=-1) * factor, n=count, axis=-1)


def time_derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Second-order time derivative on a possibly non-uniform output grid."""
    return np.gradient(values, times, axis=1, edge_order=2)


def format_length(value: float) -> str:
    if math.isclose(value, TWO_PI, rel_tol=0.0, abs_tol=1e-15):
        return f"{value!r} (that is, 2*pi)"
    return repr(value)


def time_sentence() -> str:
    return (
        "u0_batch has shape [B, N] and holds u at time t_coordinate[0] = 0. t_coordinate has\n"
        "shape [T]; it is increasing but its spacing may be non-uniform, and each entry is an\n"
        "absolute time measured from the initial state. Return a finite NumPy array with\n"
        "shape [B, T, N]. Returned slice n must correspond to t_coordinate[n], and slice 0 is\n"
        "the initial state. Internal substeps are allowed. Use a stable numerical method.\n"
        "Do not hard-code a solution array."
    )


def check_form(form: str) -> None:
    if form not in PROMPT_FORMS:
        raise ValueError(f"unknown prompt form: {form}")


def apply_stratum(
    selected: dict[str, Any], stratum: dict[str, Any] | None, allowed: set[str]
) -> dict[str, Any]:
    if not stratum:
        return selected
    unknown = set(stratum) - allowed
    if unknown:
        raise ValueError(f"unsupported stratum keys: {sorted(unknown)}")
    merged = dict(selected)
    merged.update(stratum)
    return merged
