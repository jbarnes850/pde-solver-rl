from __future__ import annotations

from collections.abc import Callable

import numpy as np


def periodic_grid(count: int, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    return np.linspace(lower, upper, count, endpoint=False, dtype=np.float64)


def time_grid(count: int, final_time: float) -> np.ndarray:
    return np.linspace(0.0, final_time, count, dtype=np.float64)


def periodic_first(values: np.ndarray, spacing: float, axis: int = -1) -> np.ndarray:
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing)


def periodic_second(values: np.ndarray, spacing: float, axis: int = -1) -> np.ndarray:
    return (
        np.roll(values, -1, axis=axis) - 2.0 * values + np.roll(values, 1, axis=axis)
    ) / spacing**2


def neumann_second(values: np.ndarray, spacing: float, axis: int = -1) -> np.ndarray:
    padded = np.pad(
        values,
        [(1, 1) if index == axis % values.ndim else (0, 0) for index in range(values.ndim)],
        mode="edge",
    )
    slices = [slice(None)] * values.ndim
    left = slices.copy()
    center = slices.copy()
    right = slices.copy()
    left[axis] = slice(0, -2)
    center[axis] = slice(1, -1)
    right[axis] = slice(2, None)
    return (padded[tuple(left)] - 2.0 * padded[tuple(center)] + padded[tuple(right)]) / spacing**2


def rk4_step(state: np.ndarray, dt: float, rhs: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_to_times(
    initial: np.ndarray,
    times: np.ndarray,
    rhs: Callable[[np.ndarray], np.ndarray],
    max_step: float,
) -> np.ndarray:
    state = np.asarray(initial, dtype=np.float64).copy()
    output = np.empty((len(times), *state.shape), dtype=np.float64)
    output[0] = state
    now = float(times[0])
    for output_index, target in enumerate(times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(max_step, float(target) - now)
            state = rk4_step(state, step, rhs)
            now += step
        output[output_index] = state
    return output
