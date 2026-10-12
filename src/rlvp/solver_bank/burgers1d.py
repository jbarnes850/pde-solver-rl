from __future__ import annotations

from textwrap import dedent

METHOD_IDS = (
    "godunov_euler",
    "godunov_ssprk2",
    "godunov_ssprk3",
    "lax_friedrichs_euler",
    "lax_friedrichs_ssprk2",
    "lax_friedrichs_ssprk3",
    "rusanov_euler",
    "rusanov_ssprk2",
    "rusanov_ssprk3",
    "muscl_rusanov_euler",
    "muscl_rusanov_rk2",
    "muscl_rusanov_ssprk2",
    "muscl_rusanov_ssprk3",
    "lax_wendroff",
    "maccormack",
    "imex_cn_fixed",
    "spectral_rk4_dealias",
    "spectral_if_rk4_dealias",
)


_VALIDATION = r"""
import numpy as np


def _inputs(u0_batch, t_coordinate, nu):
    u0 = np.asarray(u0_batch, dtype=np.float64)
    times = np.asarray(t_coordinate, dtype=np.float64)
    nu = float(nu)
    if u0.ndim != 2 or u0.shape[1] < 4:
        raise ValueError("u0_batch must have shape [B,N] with N >= 4")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("t_coordinate must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(u0)) or not np.all(np.isfinite(times)) or not np.isfinite(nu):
        raise ValueError("inputs must be finite")
    if nu < 0.0:
        raise ValueError("nu must be nonnegative")
    if times[0] < 0.0 or np.any(np.diff(times) < 0.0):
        raise ValueError("requested times must be nonnegative and nondecreasing")
    return u0, times, nu
"""


_DRIVER = r"""
def solver(u0_batch, t_coordinate, nu):
    u0, times, nu = _inputs(u0_batch, t_coordinate, nu)
    dx = 1.0 / u0.shape[1]
    kappa = nu / np.pi
    state = u0.copy()
    result = np.empty((u0.shape[0], times.size, u0.shape[1]), dtype=np.float64)
    now = 0.0
    for output_index, target in enumerate(times):
        target = float(target)
        while now < target - 1.0e-14:
            dt = min(_step_limit(state, dx, kappa), target - now)
            state = _step(state, dt, dx, kappa)
            now += dt
        result[:, output_index, :] = state
    return result
"""


def _integrator_source(name: str) -> str:
    return {
        "euler": r"""
def _step(u, dt, dx, kappa):
    return u + dt * _rhs(u, dx, kappa)
""",
        "rk2": r"""
def _step(u, dt, dx, kappa):
    first = _rhs(u, dx, kappa)
    return u + dt * _rhs(u + 0.5 * dt * first, dx, kappa)
""",
        "ssprk2": r"""
def _step(u, dt, dx, kappa):
    first = u + dt * _rhs(u, dx, kappa)
    return 0.5 * u + 0.5 * (first + dt * _rhs(first, dx, kappa))
""",
        "ssprk3": r"""
def _step(u, dt, dx, kappa):
    first = u + dt * _rhs(u, dx, kappa)
    second = 0.75 * u + 0.25 * (first + dt * _rhs(first, dx, kappa))
    return (u + 2.0 * (second + dt * _rhs(second, dx, kappa))) / 3.0
""",
    }[name]


def _render_finite_volume(*, flux: str, integrator: str, muscl: bool) -> str:
    flux_source = {
        "godunov": r"""
# Exact scalar Godunov flux for f(u)=u**2/2.
def _numerical_flux(left, right):
    left_flux = 0.5 * left * left
    right_flux = 0.5 * right * right
    rarefaction = left <= right
    rare_flux = np.where(left >= 0.0, left_flux, np.where(right <= 0.0, right_flux, 0.0))
    shock_flux = np.where(0.5 * (left + right) >= 0.0, left_flux, right_flux)
    return np.where(rarefaction, rare_flux, shock_flux)
""",
        "lax_friedrichs": r"""
# For a scalar conservation law, local Lax-Friedrichs and local Rusanov use the
# same max(|f'(u)|) interface speed. The paper lists both but gives no code-level
# distinction. This clean-room row therefore implements the stated local LF flux.
def _numerical_flux(left, right):
    speed = np.maximum(np.abs(left), np.abs(right))
    return 0.25 * (left * left + right * right) - 0.5 * speed * (right - left)
""",
        "rusanov": r"""
def _numerical_flux(left, right):
    speed = np.maximum(np.abs(left), np.abs(right))
    return 0.25 * (left * left + right * right) - 0.5 * speed * (right - left)
""",
    }[flux]
    reconstruction = (
        r"""
def _interface_states(u):
    backward = u - np.roll(u, 1, axis=-1)
    forward = np.roll(u, -1, axis=-1) - u
    slope = np.where(
        backward * forward > 0.0,
        np.sign(backward) * np.minimum(np.abs(backward), np.abs(forward)),
        0.0,
    )
    left = u + 0.5 * slope
    right = np.roll(u, -1, axis=-1) - 0.5 * np.roll(slope, -1, axis=-1)
    return left, right
"""
        if muscl
        else r"""
def _interface_states(u):
    return u, np.roll(u, -1, axis=-1)
"""
    )
    source = (
        _VALIDATION
        + "\n"
        + flux_source
        + "\n"
        + reconstruction
        + r"""

def _rhs(u, dx, kappa):
    left, right = _interface_states(u)
    flux = _numerical_flux(left, right)
    advection = -(flux - np.roll(flux, 1, axis=-1)) / dx
    laplacian = (np.roll(u, -1, axis=-1) - 2.0 * u + np.roll(u, 1, axis=-1)) / dx**2
    return advection + kappa * laplacian


def _step_limit(u, dx, kappa):
    max_speed = float(np.max(np.abs(u)))
    advective = np.inf if max_speed < 1.0e-14 else 0.35 * dx / max_speed
    diffusive = np.inf if kappa == 0.0 else 0.20 * dx**2 / kappa
    return min(advective, diffusive, 1.0)
"""
        + _integrator_source(integrator)
        + "\n"
        + _DRIVER
    )
    return dedent(source)


_SOURCES: dict[str, str] = {}
for _flux in ("godunov", "lax_friedrichs", "rusanov"):
    for _integrator in ("euler", "ssprk2", "ssprk3"):
        _SOURCES[f"{_flux}_{_integrator}"] = _render_finite_volume(
            flux=_flux, integrator=_integrator, muscl=False
        )
for _integrator in ("euler", "rk2", "ssprk2", "ssprk3"):
    _SOURCES[f"muscl_rusanov_{_integrator}"] = _render_finite_volume(
        flux="rusanov", integrator=_integrator, muscl=True
    )


_SOURCES["lax_wendroff"] = (
    _VALIDATION
    + dedent(
        r"""

    # Two-step Richtmyer form of Lax-Wendroff plus a centered viscous Laplacian.
    def _step_limit(u, dx, kappa):
        max_speed = float(np.max(np.abs(u)))
        advective = np.inf if max_speed < 1.0e-14 else 0.30 * dx / max_speed
        diffusive = np.inf if kappa == 0.0 else 0.18 * dx**2 / kappa
        return min(advective, diffusive, 1.0)


    def _step(u, dt, dx, kappa):
        flux = 0.5 * u * u
        right = np.roll(u, -1, axis=-1)
        right_flux = np.roll(flux, -1, axis=-1)
        half = 0.5 * (u + right) - 0.5 * dt * (right_flux - flux) / dx
        half_flux = 0.5 * half * half
        laplacian = (right - 2.0 * u + np.roll(u, 1, axis=-1)) / dx**2
        return u - dt * (half_flux - np.roll(half_flux, 1, axis=-1)) / dx + dt * kappa * laplacian


    """
    )
    + _DRIVER
)


_SOURCES["maccormack"] = (
    _VALIDATION
    + dedent(
        r"""

    # Periodic MacCormack predictor-corrector plus centered viscous diffusion.
    def _step_limit(u, dx, kappa):
        max_speed = float(np.max(np.abs(u)))
        advective = np.inf if max_speed < 1.0e-14 else 0.30 * dx / max_speed
        diffusive = np.inf if kappa == 0.0 else 0.18 * dx**2 / kappa
        return min(advective, diffusive, 1.0)


    def _laplacian(u, dx):
        return (np.roll(u, -1, axis=-1) - 2.0 * u + np.roll(u, 1, axis=-1)) / dx**2


    def _step(u, dt, dx, kappa):
        flux = 0.5 * u * u
        predicted = u - dt * (np.roll(flux, -1, axis=-1) - flux) / dx
        predicted += dt * kappa * _laplacian(u, dx)
        predicted_flux = 0.5 * predicted * predicted
        corrected = predicted - dt * (predicted_flux - np.roll(predicted_flux, 1, axis=-1)) / dx
        corrected += dt * kappa * _laplacian(predicted, dx)
        return 0.5 * (u + corrected)


    """
    )
    + _DRIVER
)


_SOURCES["imex_cn_fixed"] = (
    _VALIDATION
    + dedent(
        r"""

    # Explicit first-order Rusanov advection and a periodic Crank-Nicolson
    # diffusion solve. The paper does not publish its fixed substep size.
    def _rusanov_advection(u, dx):
        right = np.roll(u, -1, axis=-1)
        speed = np.maximum(np.abs(u), np.abs(right))
        flux = 0.25 * (u * u + right * right) - 0.5 * speed * (right - u)
        return -(flux - np.roll(flux, 1, axis=-1)) / dx


    def _step_limit(u, dx, kappa):
        del kappa
        max_speed = float(np.max(np.abs(u)))
        return 1.0 if max_speed < 1.0e-14 else min(0.30 * dx / max_speed, 1.0)


    def _step(u, dt, dx, kappa):
        count = u.shape[-1]
        mode = np.arange(count, dtype=np.float64)
        laplace_eigenvalue = -4.0 * np.sin(np.pi * mode / count) ** 2 / dx**2
        explicit = u + dt * _rusanov_advection(u, dx)
        right_hand_side = np.fft.fft(explicit, axis=-1)
        right_hand_side += 0.5 * dt * kappa * laplace_eigenvalue[None, :] * np.fft.fft(u, axis=-1)
        denominator = 1.0 - 0.5 * dt * kappa * laplace_eigenvalue
        return np.fft.ifft(right_hand_side / denominator[None, :], axis=-1).real


    """
    )
    + _DRIVER
)


_SOURCES["spectral_rk4_dealias"] = (
    _VALIDATION
    + dedent(
        r"""

    # Fourier pseudospectral flux derivative, two-thirds dealiasing, and RK4.
    def _spectral_data(count, dx):
        wave_number = 2.0 * np.pi * np.fft.fftfreq(count, d=dx)
        retained = np.abs(np.fft.fftfreq(count) * count) <= count / 3.0
        return wave_number, retained


    def _rhs(u, dx, kappa):
        wave_number, retained = _spectral_data(u.shape[-1], dx)
        flux_modes = np.fft.fft(0.5 * u * u, axis=-1)
        flux_modes[:, ~retained] = 0.0
        modes = np.fft.fft(u, axis=-1)
        return np.fft.ifft(
            -1j * wave_number[None, :] * flux_modes
            - kappa * wave_number[None, :] ** 2 * modes,
            axis=-1,
        ).real


    def _step_limit(u, dx, kappa):
        max_speed = float(np.max(np.abs(u)))
        advective = np.inf if max_speed < 1.0e-14 else 0.20 * dx / max_speed
        diffusive = np.inf if kappa == 0.0 else 0.10 * dx**2 / kappa
        return min(advective, diffusive, 1.0)


    def _step(u, dt, dx, kappa):
        k1 = _rhs(u, dx, kappa)
        k2 = _rhs(u + 0.5 * dt * k1, dx, kappa)
        k3 = _rhs(u + 0.5 * dt * k2, dx, kappa)
        k4 = _rhs(u + dt * k3, dx, kappa)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


    """
    )
    + _DRIVER
)


_SOURCES["spectral_if_rk4_dealias"] = (
    _VALIDATION
    + dedent(
        r"""

    # Lawson integrating-factor RK4: diffusion is exact in Fourier space and
    # the nonlinear flux uses the two-thirds dealiasing rule.
    def _nonlinear(modes, wave_number, retained):
        state = np.fft.ifft(modes, axis=-1).real
        flux_modes = np.fft.fft(0.5 * state * state, axis=-1)
        flux_modes[:, ~retained] = 0.0
        return -1j * wave_number[None, :] * flux_modes


    def _step_limit(u, dx, kappa):
        del kappa
        max_speed = float(np.max(np.abs(u)))
        return 1.0 if max_speed < 1.0e-14 else min(0.20 * dx / max_speed, 1.0)


    def _step(u, dt, dx, kappa):
        count = u.shape[-1]
        wave_number = 2.0 * np.pi * np.fft.fftfreq(count, d=dx)
        retained = np.abs(np.fft.fftfreq(count) * count) <= count / 3.0
        initial = np.fft.fft(u, axis=-1)
        exponential_half = np.exp(-0.5 * dt * kappa * wave_number**2)[None, :]
        exponential = exponential_half * exponential_half
        k1 = _nonlinear(initial, wave_number, retained)
        stage2 = exponential_half * (initial + 0.5 * dt * k1)
        k2 = _nonlinear(stage2, wave_number, retained)
        stage3 = exponential_half * initial + 0.5 * dt * k2
        k3 = _nonlinear(stage3, wave_number, retained)
        stage4 = exponential * initial + dt * exponential_half * k3
        k4 = _nonlinear(stage4, wave_number, retained)
        updated = exponential * initial + (dt / 6.0) * (
            exponential * k1
            + 2.0 * exponential_half * k2
            + 2.0 * exponential_half * k3
            + k4
        )
        updated[:, ~retained] = 0.0
        return np.fft.ifft(updated, axis=-1).real


    """
    )
    + _DRIVER
)


def render(codename: str) -> str:
    """Render one standalone solver response for the paper's Burgers interface."""

    try:
        source = _SOURCES[codename]
    except KeyError as exc:
        raise KeyError(f"unknown burgers1d solver-bank codename: {codename}") from exc
    return dedent(source).strip() + "\n"
