from __future__ import annotations

from textwrap import dedent

_METHODS = {
    "fd_explicit_euler": ("Forward Euler on the centered-FD diffusion and logistic-reaction RHS."),
    "fd_explicit_rk2": (
        "Explicit midpoint RK2 on the centered-FD diffusion and logistic-reaction RHS."
    ),
    "fd_explicit_rk4": ("Classical RK4 on the centered-FD diffusion and logistic-reaction RHS."),
    "split_strang_fd_rk2_pes": (
        "Strang splitting with exact logistic reaction and centered-FD RK2 diffusion. "
        "The paper does not define its PES reaction update, so this clean-room fallback "
        "uses the exact logistic subflow that the paper also permits."
    ),
    "spectral_strang_exact_reaction": (
        "Strang splitting with exact logistic reaction and exact Fourier diffusion."
    ),
    "spectral_imex_exact_diffusion": (
        "Fourier integrating-factor RK4 with exact diffusion factors. The paper does "
        "not define the reaction-diffusion coupling for this codename; integrating-factor "
        "RK4 is the documented clean-room fallback."
    ),
}


_SOURCE = r'''
import numpy as np

# Source-derived clean-room solver-bank method: __METHOD__
# __METHOD_NOTE__

_METHOD = __METHOD_LITERAL__


def _inputs(u0_batch, t_coordinate, nu, rho):
    u = np.asarray(u0_batch, dtype=np.float64)
    times = np.asarray(t_coordinate, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] < 3:
        raise ValueError("u0_batch must have shape [batch, space] with space >= 3")
    if times.ndim != 1 or times.size == 0:
        raise ValueError("t_coordinate must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(times)):
        raise ValueError("inputs must be finite")
    if times[0] < 0.0 or np.any(np.diff(times) < 0.0):
        raise ValueError("t_coordinate must be nonnegative and nondecreasing")
    nu = float(nu)
    rho = float(rho)
    if not np.isfinite(nu) or nu < 0.0:
        raise ValueError("nu must be finite and nonnegative")
    if not np.isfinite(rho) or rho < 0.0:
        raise ValueError("rho must be finite and nonnegative")
    return u.copy(), times, nu, rho


def _laplacian(u, dx):
    return (np.roll(u, -1, axis=-1) - 2.0 * u + np.roll(u, 1, axis=-1)) / (dx * dx)


def _rhs(u, nu, rho, dx):
    return nu * _laplacian(u, dx) + rho * u * (1.0 - u)


def _exact_reaction(u, dt, rho):
    if dt == 0.0 or rho == 0.0:
        return u.copy()
    growth = np.exp(np.clip(rho * dt, -700.0, 700.0))
    denominator = 1.0 + u * (growth - 1.0)
    tiny = np.finfo(np.float64).tiny
    safe_denominator = np.where(
        np.abs(denominator) < tiny,
        np.where(denominator < 0.0, -tiny, tiny),
        denominator,
    )
    return u * growth / safe_denominator


def _spectral_diffusion(u, dt, nu):
    if dt == 0.0 or nu == 0.0:
        return u.copy()
    n = u.shape[-1]
    wave_number = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    multiplier = np.exp(-nu * wave_number * wave_number * dt)
    return np.fft.ifft(np.fft.fft(u, axis=-1) * multiplier, axis=-1).real


def _explicit_step(u, dt, nu, rho, dx):
    if _METHOD == "fd_explicit_euler":
        return u + dt * _rhs(u, nu, rho, dx)
    if _METHOD == "fd_explicit_rk2":
        k1 = _rhs(u, nu, rho, dx)
        k2 = _rhs(u + 0.5 * dt * k1, nu, rho, dx)
        return u + dt * k2
    k1 = _rhs(u, nu, rho, dx)
    k2 = _rhs(u + 0.5 * dt * k1, nu, rho, dx)
    k3 = _rhs(u + 0.5 * dt * k2, nu, rho, dx)
    k4 = _rhs(u + dt * k3, nu, rho, dx)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _split_fd_step(u, dt, nu, rho, dx):
    # The unpublished PES formula is unavailable. Exact logistic half-steps are
    # the explicit fallback allowed by the paper's "exact or PES" description.
    half_reacted = _exact_reaction(u, 0.5 * dt, rho)
    k1 = nu * _laplacian(half_reacted, dx)
    k2 = nu * _laplacian(half_reacted + 0.5 * dt * k1, dx)
    diffused = half_reacted + dt * k2
    return _exact_reaction(diffused, 0.5 * dt, rho)


def _spectral_strang_step(u, dt, nu, rho):
    half_reacted = _exact_reaction(u, 0.5 * dt, rho)
    diffused = _spectral_diffusion(half_reacted, dt, nu)
    return _exact_reaction(diffused, 0.5 * dt, rho)


def _spectral_ifrk4_step(u, dt, nu, rho):
    # This integrating-factor RK4 composition is the declared fallback because
    # the paper does not give the coupling rule for spectral_imex_exact_diffusion.
    n = u.shape[-1]
    wave_number = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    linear = -nu * wave_number * wave_number
    e_half = np.exp(0.5 * dt * linear)
    e_full = e_half * e_half
    u_hat = np.fft.fft(u, axis=-1)

    def reaction_hat(values_hat):
        values = np.fft.ifft(values_hat, axis=-1).real
        return np.fft.fft(rho * values * (1.0 - values), axis=-1)

    k1 = reaction_hat(u_hat)
    stage2_hat = e_half * (u_hat + 0.5 * dt * k1)
    n2 = reaction_hat(stage2_hat)
    stage3_hat = e_half * u_hat + 0.5 * dt * n2
    n3 = reaction_hat(stage3_hat)
    stage4_hat = e_full * u_hat + dt * e_half * n3
    n4 = reaction_hat(stage4_hat)
    result_hat = (
        e_full * (u_hat + (dt / 6.0) * k1)
        + (dt / 3.0) * e_half * (n2 + n3)
        + (dt / 6.0) * n4
    )
    return np.fft.ifft(result_hat, axis=-1).real


def _step_limit(nu, rho, dx):
    reaction_limit = np.inf if rho == 0.0 else 0.1 / rho
    if _METHOD.startswith("fd_explicit_"):
        diffusion_factor = 0.65 if _METHOD == "fd_explicit_rk4" else 0.45
        diffusion_limit = np.inf if nu == 0.0 else diffusion_factor * dx * dx / nu
        return min(diffusion_limit, reaction_limit)
    if _METHOD == "split_strang_fd_rk2_pes":
        return np.inf if nu == 0.0 else 0.45 * dx * dx / nu
    # The spectral subflows are stable. This bound controls splitting or coupling error.
    return 0.05 / max(rho, 1.0)


def solver(u0_batch, t_coordinate, nu, rho):
    """Solve periodic u_t = nu*u_xx + rho*u*(1-u) on x in [0, 1).

    The return shape is [batch, requested_time, space]. The initial values are
    defined at t=0. Each requested nonnegative time is reached with internal
    substeps. Repeated requested times are supported.
    """
    u, times, nu, rho = _inputs(u0_batch, t_coordinate, nu, rho)
    dx = 1.0 / u.shape[-1]
    output = np.empty((u.shape[0], times.size, u.shape[1]), dtype=np.float64)
    current_time = 0.0
    max_step = _step_limit(nu, rho, dx)
    for time_index, target_time in enumerate(times):
        interval = float(target_time - current_time)
        if interval > 0.0:
            step_count = 1 if not np.isfinite(max_step) else max(1, int(np.ceil(interval / max_step)))
            dt = interval / step_count
            for _ in range(step_count):
                if _METHOD.startswith("fd_explicit_"):
                    u = _explicit_step(u, dt, nu, rho, dx)
                elif _METHOD == "split_strang_fd_rk2_pes":
                    u = _split_fd_step(u, dt, nu, rho, dx)
                elif _METHOD == "spectral_strang_exact_reaction":
                    u = _spectral_strang_step(u, dt, nu, rho)
                else:
                    u = _spectral_ifrk4_step(u, dt, nu, rho)
        output[:, time_index, :] = u
        current_time = float(target_time)
    return output
'''


def render(codename: str) -> str:
    """Render one standalone paper-named Fisher-KPP solver implementation."""

    try:
        method_note = _METHODS[codename]
    except KeyError as exc:
        raise KeyError(f"unknown reaction_diffusion1d solver method: {codename}") from exc
    return (
        dedent(_SOURCE)
        .replace("__METHOD_LITERAL__", repr(codename))
        .replace("__METHOD_NOTE__", method_note)
        .replace("__METHOD__", codename)
    )
