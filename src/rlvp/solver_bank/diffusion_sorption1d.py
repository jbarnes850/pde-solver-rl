from __future__ import annotations

"""Clean-room teacher renderers for the paper's diffusion-sorption solver bank.

The paper identifies the eight numerical methods but does not release their source. It also
does not state the solver signature, time-step caps, nonlinear tolerances, or fallback rules.
Each rendered program records the local choices that make those missing parts executable.
"""


METHOD_IDS = (
    "euler_fixed",
    "heun_fixed",
    "rk4_fixed",
    "backward_euler_lagged",
    "crank_nicolson_lagged",
    "backward_euler_newton_tridiag",
    "crank_nicolson_newton_tridiag",
    "mol_bdf_solve_ivp_mixedbc",
)


_COMMON = """import numpy as np


def _prepare(u0_batch, t_coordinate, D, kf, phi, rho_s, nf, sol, eps):
    state = np.asarray(u0_batch, dtype=np.float64).copy()
    times = np.asarray(t_coordinate, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 3:
        raise ValueError("u0_batch must have shape [B, N] with N >= 3")
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("t_coordinate must be a non-empty one-dimensional array")
    if abs(float(times[0])) > 1e-12 or np.any(np.diff(times) < 0.0):
        raise ValueError("t_coordinate must start at zero and be nondecreasing")
    parameters = (D, kf, phi, rho_s, nf, sol, eps)
    if not np.all(np.isfinite(np.asarray(parameters, dtype=np.float64))):
        raise ValueError("all scalar parameters must be finite")
    D, kf, phi, rho_s, nf, sol, eps = map(float, parameters)
    if D <= 0.0 or kf < 0.0 or not 0.0 < phi < 1.0 or eps <= 0.0:
        raise ValueError("invalid diffusion-sorption parameters")
    return state, times, D, kf, phi, rho_s, nf, sol, eps


def _retardation(values, kf, phi, rho_s, nf, eps):
    # The positive floor is a local guard for fractional powers if an iteration overshoots.
    concentration = np.maximum(values + eps, eps)
    coefficient = ((1.0 - phi) / phi) * rho_s * kf * nf
    return 1.0 + coefficient * concentration ** (nf - 1.0)


def _mobility(values, D, kf, phi, rho_s, nf, eps):
    return D / _retardation(values, kf, phi, rho_s, nf, eps)


def _laplacian_mixed(state, dx, sol):
    # Source-derived cell-centered grid: reflection through x=0 gives the Dirichlet ghost
    # value, and copying through x=1 gives the zero-flux ghost value.
    result = np.empty_like(state)
    result[:, 1:-1] = state[:, :-2] - 2.0 * state[:, 1:-1] + state[:, 2:]
    result[:, 0] = state[:, 1] - 3.0 * state[:, 0] + 2.0 * sol
    result[:, -1] = state[:, -2] - state[:, -1]
    return result / dx**2


def _rhs(state, dx, D, kf, phi, rho_s, nf, sol, eps):
    mobility = _mobility(state, D, kf, phi, rho_s, nf, eps)
    return mobility * _laplacian_mixed(state, dx, sol)


def _euler_step(state, step, dx, D, kf, phi, rho_s, nf, sol, eps):
    return state + step * _rhs(state, dx, D, kf, phi, rho_s, nf, sol, eps)


def _heun_step(state, step, dx, D, kf, phi, rho_s, nf, sol, eps):
    first = _rhs(state, dx, D, kf, phi, rho_s, nf, sol, eps)
    predictor = state + step * first
    second = _rhs(predictor, dx, D, kf, phi, rho_s, nf, sol, eps)
    return state + 0.5 * step * (first + second)


def _rk4_step(state, step, dx, D, kf, phi, rho_s, nf, sol, eps):
    arguments = (dx, D, kf, phi, rho_s, nf, sol, eps)
    k1 = _rhs(state, *arguments)
    k2 = _rhs(state + 0.5 * step * k1, *arguments)
    k3 = _rhs(state + 0.5 * step * k2, *arguments)
    k4 = _rhs(state + step * k3, *arguments)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _solve_tridiagonal(lower, diagonal, upper, right_hand_side):
    # Batched Thomas solve. The first lower and last upper entries are ignored.
    lower = np.asarray(lower, dtype=np.float64)
    diagonal = np.asarray(diagonal, dtype=np.float64).copy()
    upper = np.asarray(upper, dtype=np.float64)
    result = np.asarray(right_hand_side, dtype=np.float64).copy()
    count = diagonal.shape[1]
    for index in range(1, count):
        factor = lower[:, index] / diagonal[:, index - 1]
        diagonal[:, index] -= factor * upper[:, index - 1]
        result[:, index] -= factor * result[:, index - 1]
    result[:, -1] /= diagonal[:, -1]
    for index in range(count - 2, -1, -1):
        result[:, index] = (
            result[:, index] - upper[:, index] * result[:, index + 1]
        ) / diagonal[:, index]
    return result


def _lagged_step(state, step, theta, dx, D, kf, phi, rho_s, nf, sol, eps):
    previous = state.copy()
    mobility = _mobility(previous, D, kf, phi, rho_s, nf, eps)
    laplacian = _laplacian_mixed(previous, dx, sol)
    right_hand_side = previous + (1.0 - theta) * step * mobility * laplacian
    ratio = theta * step * mobility / dx**2
    lower = -ratio.copy()
    diagonal = 1.0 + 2.0 * ratio
    upper = -ratio.copy()
    lower[:, 0] = 0.0
    diagonal[:, 0] = 1.0 + 3.0 * ratio[:, 0]
    right_hand_side[:, 0] += 2.0 * ratio[:, 0] * sol
    # The copied right ghost cell removes one diagonal contribution.
    diagonal[:, -1] = 1.0 + ratio[:, -1]
    upper[:, -1] = 0.0
    return _solve_tridiagonal(lower, diagonal, upper, right_hand_side)


def _mobility_derivative(values, D, kf, phi, rho_s, nf, eps):
    concentration = np.maximum(values + eps, eps)
    coefficient = ((1.0 - phi) / phi) * rho_s * kf * nf
    retardation = 1.0 + coefficient * concentration ** (nf - 1.0)
    derivative = coefficient * (nf - 1.0) * concentration ** (nf - 2.0)
    return -D * derivative / retardation**2


def _newton_step(state, step, theta, dx, D, kf, phi, rho_s, nf, sol, eps):
    previous = state.copy()
    previous_flux = _mobility(
        previous, D, kf, phi, rho_s, nf, eps
    ) * _laplacian_mixed(previous, dx, sol)
    candidate = _lagged_step(
        previous, step, theta, dx, D, kf, phi, rho_s, nf, sol, eps
    )
    converged = False
    for _ in range(12):
        laplacian = _laplacian_mixed(candidate, dx, sol)
        mobility = _mobility(candidate, D, kf, phi, rho_s, nf, eps)
        derivative = _mobility_derivative(candidate, D, kf, phi, rho_s, nf, eps)
        residual = (
            candidate
            - previous
            - step * ((1.0 - theta) * previous_flux + theta * mobility * laplacian)
        )
        if float(np.max(np.abs(residual))) <= 1e-10:
            converged = True
            break
        ratio = theta * step * mobility / dx**2
        lower = -ratio.copy()
        upper = -ratio.copy()
        diagonal = 1.0 - theta * step * derivative * laplacian + 2.0 * ratio
        lower[:, 0] = 0.0
        diagonal[:, 0] = (
            1.0 - theta * step * derivative[:, 0] * laplacian[:, 0] + 3.0 * ratio[:, 0]
        )
        diagonal[:, -1] = (
            1.0 - theta * step * derivative[:, -1] * laplacian[:, -1] + ratio[:, -1]
        )
        upper[:, -1] = 0.0
        correction = _solve_tridiagonal(lower, diagonal, upper, residual)
        candidate -= correction
    if not converged:
        # Local fallback: the paper does not state Newton failure behavior. Use the named
        # lagged implicit counterpart instead of returning a non-converged nonlinear step.
        return _lagged_step(previous, step, theta, dx, D, kf, phi, rho_s, nf, sol, eps)
    return candidate
"""


_EXPLICIT_SOLVER = '''

def solver(u0_batch, t_coordinate, D, kf, phi, rho_s, nf, sol, eps):
    """Solve the mixed-boundary diffusion-sorption PDE with __METHOD_DESCRIPTION__.

    Source-derived choice: the paper omits the fixed step. This implementation uses the
    conservative cap 0.35*dx**2/D and only shortens a step to hit a requested output time.
    """
    state, times, D, kf, phi, rho_s, nf, sol, eps = _prepare(
        u0_batch, t_coordinate, D, kf, phi, rho_s, nf, sol, eps
    )
    output = np.empty((state.shape[0], len(times), state.shape[1]), dtype=np.float64)
    output[:, 0] = state
    dx = 1.0 / state.shape[1]
    max_step = 0.35 * dx**2 / D
    now = 0.0
    for output_index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(max_step, float(target) - now)
            state = __STEP_FUNCTION__(state, step, dx, D, kf, phi, rho_s, nf, sol, eps)
            now += step
        output[:, output_index] = state
    return output
'''


_IMPLICIT_SOLVER = '''

def solver(u0_batch, t_coordinate, D, kf, phi, rho_s, nf, sol, eps):
    """Solve the mixed-boundary diffusion-sorption PDE with __METHOD_DESCRIPTION__.

    Source-derived choice: the paper omits the implicit step and solve tolerance. This
    implementation uses steps no larger than 0.5 and the direct batched Thomas algorithm.
    """
    state, times, D, kf, phi, rho_s, nf, sol, eps = _prepare(
        u0_batch, t_coordinate, D, kf, phi, rho_s, nf, sol, eps
    )
    output = np.empty((state.shape[0], len(times), state.shape[1]), dtype=np.float64)
    output[:, 0] = state
    dx = 1.0 / state.shape[1]
    now = 0.0
    for output_index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(0.5, float(target) - now)
            state = __STEP_FUNCTION__(
                state, step, __THETA__, dx, D, kf, phi, rho_s, nf, sol, eps
            )
            now += step
        output[:, output_index] = state
    return output
'''


_BDF = '''import numpy as np
from scipy.integrate import solve_ivp


def solver(u0_batch, t_coordinate, D, kf, phi, rho_s, nf, sol, eps):
    """Solve by a mixed-boundary method of lines and SciPy's adaptive BDF method.

    Source-derived choices: the paper omits the grid placement, tolerances, maximum step,
    and failure behavior. This implementation uses a cell-centered grid with ghost-cell
    boundaries, rtol=1e-8, atol=1e-10, max_step=1.0, and raises on BDF failure.
    """
    initial = np.asarray(u0_batch, dtype=np.float64).copy()
    times = np.asarray(t_coordinate, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] < 3:
        raise ValueError("u0_batch must have shape [B, N] with N >= 3")
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("t_coordinate must be a non-empty one-dimensional array")
    if abs(float(times[0])) > 1e-12 or np.any(np.diff(times) < 0.0):
        raise ValueError("t_coordinate must start at zero and be nondecreasing")
    D, kf, phi, rho_s, nf, sol, eps = map(float, (D, kf, phi, rho_s, nf, sol, eps))
    output = np.empty((initial.shape[0], len(times), initial.shape[1]), dtype=np.float64)
    output[:, 0] = initial
    if len(times) == 1:
        return output
    dx = 1.0 / initial.shape[1]
    coefficient = ((1.0 - phi) / phi) * rho_s * kf * nf

    for batch_index, batch_initial in enumerate(initial):
        def right_hand_side(_time, state):
            laplacian = np.empty_like(state)
            laplacian[1:-1] = state[:-2] - 2.0 * state[1:-1] + state[2:]
            laplacian[0] = state[1] - 3.0 * state[0] + 2.0 * sol
            laplacian[-1] = state[-2] - state[-1]
            laplacian /= dx**2
            concentration = np.maximum(state + eps, eps)
            retardation = 1.0 + coefficient * concentration ** (nf - 1.0)
            return (D / retardation) * laplacian

        result = solve_ivp(
            right_hand_side,
            (0.0, float(times[-1])),
            batch_initial,
            method="BDF",
            t_eval=times,
            rtol=1e-8,
            atol=1e-10,
            max_step=1.0,
        )
        if not result.success:
            raise RuntimeError("diffusion-sorption BDF solve failed: " + result.message)
        output[batch_index] = result.y.T
    return output
'''


_EXPLICIT = {
    "euler_fixed": ("forward Euler", "_euler_step"),
    "heun_fixed": ("Heun / explicit RK2", "_heun_step"),
    "rk4_fixed": ("classical explicit RK4", "_rk4_step"),
}

_IMPLICIT = {
    "backward_euler_lagged": ("lagged-mobility backward Euler", "_lagged_step", "1.0"),
    "crank_nicolson_lagged": ("lagged-mobility Crank-Nicolson", "_lagged_step", "0.5"),
    "backward_euler_newton_tridiag": (
        "Newton backward Euler with a tridiagonal Jacobian",
        "_newton_step",
        "1.0",
    ),
    "crank_nicolson_newton_tridiag": (
        "Newton Crank-Nicolson with a tridiagonal Jacobian",
        "_newton_step",
        "0.5",
    ),
}


def render(codename: str) -> str:
    """Return one standalone clean-room teacher program for ``codename``."""

    if codename in _EXPLICIT:
        description, step_function = _EXPLICIT[codename]
        body = _EXPLICIT_SOLVER.replace("__METHOD_DESCRIPTION__", description).replace(
            "__STEP_FUNCTION__", step_function
        )
        return _COMMON + body
    if codename in _IMPLICIT:
        description, step_function, theta = _IMPLICIT[codename]
        body = (
            _IMPLICIT_SOLVER.replace("__METHOD_DESCRIPTION__", description)
            .replace("__STEP_FUNCTION__", step_function)
            .replace("__THETA__", theta)
        )
        return _COMMON + body
    if codename == "mol_bdf_solve_ivp_mixedbc":
        return _BDF
    known = ", ".join(METHOD_IDS)
    raise KeyError(f"unknown diffusion_sorption1d solver {codename!r}; expected one of: {known}")
