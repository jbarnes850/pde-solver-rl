from __future__ import annotations

from textwrap import dedent

METHOD_IDS = (
    "semi_lagrangian",
    "pdebench_fd2_midpoint_vanleer",
    "pdebench_rusanov2",
    "central4_rk4",
    "muscl_rusanov_ssprk3",
    "spectral_rk4",
    "fourier_exact_shift",
    "fourier_spectral_rk4",
)


_VALIDATION = r"""
import numpy as np


def _inputs(u0_batch, t_coordinate):
    u0 = np.asarray(u0_batch, dtype=np.float64)
    times = np.asarray(t_coordinate, dtype=np.float64)
    if u0.ndim != 2 or u0.shape[1] < 4:
        raise ValueError("u0_batch must have shape [B,N] with N >= 4")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("t_coordinate must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(u0)) or not np.all(np.isfinite(times)):
        raise ValueError("inputs must be finite")
    if times[0] < 0.0 or np.any(np.diff(times) < 0.0):
        raise ValueError("requested times must be nonnegative and nondecreasing")
    return u0, times
"""


def _render_mol(*, description: str, rhs: str, step: str, cfl: float) -> str:
    step_body = dedent(step).strip().replace("\n", "\n    ")
    driver = f"""
def _step(u, dt, beta, dx):
    {step_body}


def solver(u0_batch, t_coordinate, beta):
    u0, times = _inputs(u0_batch, t_coordinate)
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    dx = 1.0 / u0.shape[1]
    speed = abs(beta)
    max_step = np.inf if speed == 0.0 else {cfl!r} * dx / speed
    state = u0.copy()
    result = np.empty((u0.shape[0], times.size, u0.shape[1]), dtype=np.float64)
    now = 0.0
    for output_index, target in enumerate(times):
        span = float(target) - now
        count = 1 if span <= 0.0 else max(1, int(np.ceil(span / max_step)))
        dt = 0.0 if span <= 0.0 else span / count
        for _ in range(count):
            if dt:
                state = _step(state, dt, beta, dx)
        result[:, output_index, :] = state
        now = float(target)
    return result
"""
    return (
        _VALIDATION
        + f"\n# Source-derived clean-room method: {description}\n"
        + dedent(rhs).strip()
        + "\n\n"
        + driver
    )


def _render_flux(
    *, description: str, limiter: str, integrator: str, ambiguity_comment: str = ""
) -> str:
    limiter_source = {
        "vanleer": r"""
def _slope(u):
    backward = u - np.roll(u, 1, axis=-1)
    forward = np.roll(u, -1, axis=-1) - u
    slope = np.zeros_like(u)
    same_sign = backward * forward > 0.0
    np.divide(
        2.0 * backward * forward,
        backward + forward,
        out=slope,
        where=same_sign,
    )
    return slope
""",
        "minmod": r"""
def _slope(u):
    backward = u - np.roll(u, 1, axis=-1)
    forward = np.roll(u, -1, axis=-1) - u
    return np.where(
        backward * forward > 0.0,
        np.sign(backward) * np.minimum(np.abs(backward), np.abs(forward)),
        0.0,
    )
""",
    }[limiter]
    rhs = (
        (ambiguity_comment + "\n" if ambiguity_comment else "")
        + limiter_source
        + r"""

def _rhs(u, beta, dx):
    slope = _slope(u)
    left = u + 0.5 * slope
    right = np.roll(u, -1, axis=-1) - 0.5 * np.roll(slope, -1, axis=-1)
    flux = 0.5 * beta * (left + right) - 0.5 * abs(beta) * (right - left)
    return -(flux - np.roll(flux, 1, axis=-1)) / dx
"""
    )
    steps = {
        "midpoint": """
            k1 = _rhs(u, beta, dx)
            return u + dt * _rhs(u + 0.5 * dt * k1, beta, dx)
        """,
        "ssprk3": """
            first = u + dt * _rhs(u, beta, dx)
            second = 0.75 * u + 0.25 * (first + dt * _rhs(first, beta, dx))
            return (u + 2.0 * (second + dt * _rhs(second, beta, dx))) / 3.0
        """,
    }
    return _render_mol(
        description=description,
        rhs=rhs,
        step=steps[integrator],
        cfl=0.45,
    )


_SOURCES = {
    "semi_lagrangian": _VALIDATION
    + dedent(
        r"""

        # Periodic semi-Lagrangian characteristic backtrace with linear interpolation.
        # Direct backtrace from u0 avoids interpolation error accumulation between output times.
        def solver(u0_batch, t_coordinate, beta):
            u0, times = _inputs(u0_batch, t_coordinate)
            beta = float(beta)
            if not np.isfinite(beta):
                raise ValueError("beta must be finite")
            count = u0.shape[1]
            x = np.arange(count, dtype=np.float64) / count
            result = np.empty((u0.shape[0], times.size, count), dtype=np.float64)
            for output_index, target in enumerate(times):
                source_index = ((x - beta * float(target)) % 1.0) * count
                left_index = np.floor(source_index).astype(np.int64) % count
                fraction = source_index - np.floor(source_index)
                right_index = (left_index + 1) % count
                result[:, output_index, :] = (
                    (1.0 - fraction)[None, :] * u0[:, left_index]
                    + fraction[None, :] * u0[:, right_index]
                )
            return result
        """
    ),
    "pdebench_fd2_midpoint_vanleer": _render_flux(
        description="PDEBench-anchored MUSCL, Van Leer limiter, Rusanov flux, RK2 midpoint",
        limiter="vanleer",
        integrator="midpoint",
    ),
    "pdebench_rusanov2": _render_flux(
        description="PDEBench-anchored MUSCL, minmod limiter, Rusanov flux, RK2 midpoint",
        limiter="minmod",
        integrator="midpoint",
    ),
    "central4_rk4": _render_mol(
        description="fourth-order centered finite difference with classical RK4",
        rhs=r"""
def _rhs(u, beta, dx):
    derivative = (
        np.roll(u, 2, axis=-1)
        - 8.0 * np.roll(u, 1, axis=-1)
        + 8.0 * np.roll(u, -1, axis=-1)
        - np.roll(u, -2, axis=-1)
    ) / (12.0 * dx)
    return -beta * derivative
""",
        step="""
            k1 = _rhs(u, beta, dx)
            k2 = _rhs(u + 0.5 * dt * k1, beta, dx)
            k3 = _rhs(u + 0.5 * dt * k2, beta, dx)
            k4 = _rhs(u + dt * k3, beta, dx)
            return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        """,
        cfl=0.35,
    ),
    "muscl_rusanov_ssprk3": _render_flux(
        description="MUSCL, minmod limiter, Rusanov flux, SSPRK3",
        limiter="minmod",
        integrator="ssprk3",
        ambiguity_comment=(
            "# The paper does not name the limiter for this row. This clean-room variant uses "
            "minmod."
        ),
    ),
    "spectral_rk4": _render_mol(
        description="physical-state Fourier pseudospectral derivative with classical RK4",
        rhs=r"""
# The paper gives the same scheme description for spectral_rk4 and
# fourier_spectral_rk4. This variant applies the spectral derivative to physical states.
def _rhs(u, beta, dx):
    wave_number = 2.0 * np.pi * np.fft.fftfreq(u.shape[-1], d=dx)
    derivative = np.fft.ifft(
        1j * wave_number[None, :] * np.fft.fft(u, axis=-1), axis=-1
    ).real
    return -beta * derivative
""",
        step="""
            k1 = _rhs(u, beta, dx)
            k2 = _rhs(u + 0.5 * dt * k1, beta, dx)
            k3 = _rhs(u + 0.5 * dt * k2, beta, dx)
            k4 = _rhs(u + dt * k3, beta, dx)
            return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        """,
        cfl=0.20,
    ),
    "fourier_exact_shift": _VALIDATION
    + dedent(
        r"""

        # Exact phase evolution for the Fourier modes represented on the periodic grid.
        def solver(u0_batch, t_coordinate, beta):
            u0, times = _inputs(u0_batch, t_coordinate)
            beta = float(beta)
            if not np.isfinite(beta):
                raise ValueError("beta must be finite")
            count = u0.shape[1]
            wave_number = 2.0 * np.pi * np.fft.fftfreq(count, d=1.0 / count)
            initial_modes = np.fft.fft(u0, axis=-1)
            phase = np.exp(-1j * beta * times[:, None] * wave_number[None, :])
            shifted = np.fft.ifft(
                initial_modes[:, None, :] * phase[None, :, :], axis=-1
            ).real
            return shifted
        """
    ),
    "fourier_spectral_rk4": _VALIDATION
    + dedent(
        r"""

        # The paper gives the same scheme description for spectral_rk4 and this row.
        # This clean-room alias evolves Fourier coefficients directly, which keeps the
        # implementation distinct while retaining the paper-stated Fourier/RK4 method.
        def _rk4_modes(modes, dt, beta, wave_number):
            def rhs(value):
                return -1j * beta * wave_number[None, :] * value

            k1 = rhs(modes)
            k2 = rhs(modes + 0.5 * dt * k1)
            k3 = rhs(modes + 0.5 * dt * k2)
            k4 = rhs(modes + dt * k3)
            return modes + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


        def solver(u0_batch, t_coordinate, beta):
            u0, times = _inputs(u0_batch, t_coordinate)
            beta = float(beta)
            if not np.isfinite(beta):
                raise ValueError("beta must be finite")
            count = u0.shape[1]
            dx = 1.0 / count
            wave_number = 2.0 * np.pi * np.fft.fftfreq(count, d=dx)
            speed = abs(beta)
            max_step = np.inf if speed == 0.0 else 0.20 * dx / speed
            modes = np.fft.fft(u0, axis=-1)
            result = np.empty((u0.shape[0], times.size, count), dtype=np.float64)
            now = 0.0
            for output_index, target in enumerate(times):
                span = float(target) - now
                substeps = 1 if span <= 0.0 else max(1, int(np.ceil(span / max_step)))
                dt = 0.0 if span <= 0.0 else span / substeps
                for _ in range(substeps):
                    if dt:
                        modes = _rk4_modes(modes, dt, beta, wave_number)
                result[:, output_index, :] = np.fft.ifft(modes, axis=-1).real
                now = float(target)
            return result
        """
    ),
}


def render(codename: str) -> str:
    """Render one standalone solver response for the paper's advection interface."""

    try:
        source = _SOURCES[codename]
    except KeyError as exc:
        raise KeyError(f"unknown advection1d solver-bank codename: {codename}") from exc
    return dedent(source).strip() + "\n"
