from __future__ import annotations

"""Source-derived held-out 2D Gray--Scott environment.

ArXiv:2607.10474v1 reports only that 2D Gray--Scott is one of ten held-out PDE
families. It does not report its equation convention, domain, boundary condition,
parameters, grids, times, initial fields, call signature, references, or residual.
Every scientific and numerical setting below is therefore a source-derived choice,
not a paper fact. This family is never an SFT target or an RL prompt.
"""

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import integrate_to_times, periodic_second, rk4_step, time_grid

SOURCE_DERIVED_DU = (2.0e-5, 4.0e-5)
SOURCE_DERIVED_DV = (1.0e-5, 2.0e-5)
SOURCE_DERIVED_FEED = (0.035, 0.05)
SOURCE_DERIVED_KILL = (0.06, 0.065)
SOURCE_DERIVED_GRID_SIZES = (32, 48)
SOURCE_DERIVED_TIME_COUNTS = (31, 51)
SOURCE_DERIVED_FINAL_TIME = 50.0
SOURCE_DERIVED_INPUT_KINDS = ("center_patch", "paired_spots", "noisy_ring")
SOURCE_DERIVED_PRIMARY_MAX_STEP = 0.1
SOURCE_DERIVED_AUDIT_MAX_STEP = 0.05


def _required_time_data(case: PDECase) -> tuple[int, float]:
    if case.time_count is None or case.final_time is None:
        raise ValueError("gray_scott2d requires time_count and final_time")
    return case.time_count, case.final_time


def times(case: PDECase) -> np.ndarray:
    count, final_time = _required_time_data(case)
    return time_grid(count, final_time)


def initial_state(case: PDECase) -> np.ndarray:
    n = case.grid_shape[0]
    coordinate = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    rng = np.random.default_rng(case.seed)
    u = np.ones((n, n), dtype=np.float64)
    v = np.zeros((n, n), dtype=np.float64)
    if case.input_kind == "center_patch":
        distance = np.maximum(np.abs(xx - 0.5), np.abs(yy - 0.5))
        envelope = 0.5 * (1.0 - np.tanh((distance - 0.10) / 0.02))
        u -= 0.50 * envelope
        v += 0.25 * envelope
    elif case.input_kind == "paired_spots":
        first = np.exp(-((xx - 0.38) ** 2 + (yy - 0.50) ** 2) / 0.004)
        second = np.exp(-((xx - 0.62) ** 2 + (yy - 0.50) ** 2) / 0.004)
        envelope = first + second
        u -= 0.45 * envelope
        v += 0.23 * envelope
    elif case.input_kind == "noisy_ring":
        radius = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        envelope = np.exp(-(((radius - 0.18) / 0.045) ** 2))
        noise = rng.standard_normal((n, n))
        u -= 0.42 * envelope
        v += 0.21 * envelope * (1.0 + 0.08 * noise)
    else:
        raise ValueError(f"unknown gray_scott2d input kind: {case.input_kind}")
    return np.stack((u, v), axis=-1)[None, ...]


def _reaction(state: np.ndarray, feed: float, kill: float) -> np.ndarray:
    u = state[..., 0]
    v = state[..., 1]
    uv2 = u * v**2
    return np.stack((-uv2 + feed * (1.0 - u), uv2 - (feed + kill) * v), axis=-1)


def _fd_rhs(
    state: np.ndarray,
    *,
    du: float,
    dv: float,
    feed: float,
    kill: float,
    dx: float,
) -> np.ndarray:
    laplacian = periodic_second(state, dx, axis=0) + periodic_second(state, dx, axis=1)
    diffusion = laplacian * np.array((du, dv), dtype=np.float64)
    return diffusion + _reaction(state, feed, kill)


def _primary_reference(case: PDECase) -> np.ndarray:
    state = initial_state(case)[0]
    dx = 1.0 / case.grid_shape[0]
    du = float(case.parameters["du"])
    dv = float(case.parameters["dv"])
    feed = float(case.parameters["feed"])
    kill = float(case.parameters["kill"])
    stable_step = min(SOURCE_DERIVED_PRIMARY_MAX_STEP, 0.15 * dx**2 / max(du, dv))
    result = integrate_to_times(
        state,
        times(case),
        lambda value: _fd_rhs(value, du=du, dv=dv, feed=feed, kill=kill, dx=dx),
        stable_step,
    )
    return result[None, ...]


def _exact_periodic_diffusion(state: np.ndarray, step: float, du: float, dv: float) -> np.ndarray:
    n = state.shape[0]
    # This Fourier path diagonalizes the same centered-difference Laplacian used by the
    # primary method. It independently checks time integration without changing the
    # source-derived discrete PDE target.
    dx = 1.0 / n
    frequency = np.fft.fftfreq(n) * n
    one_dimensional = -4.0 * np.sin(np.pi * frequency / n) ** 2 / dx**2
    laplacian_eigenvalues = one_dimensional[:, None] + one_dimensional[None, :]
    output = np.empty_like(state)
    for channel, diffusivity in enumerate((du, dv)):
        spectrum = np.fft.fft2(state[..., channel])
        output[..., channel] = np.fft.ifft2(
            spectrum * np.exp(diffusivity * step * laplacian_eigenvalues)
        ).real
    return output


def _audit_reference(case: PDECase) -> np.ndarray:
    """Independent Strang path with Fourier-exact FD diffusion and RK4 reaction."""
    state = initial_state(case)[0].copy()
    output_times = times(case)
    output = np.empty((len(output_times), *state.shape), dtype=np.float64)
    output[0] = state
    du = float(case.parameters["du"])
    dv = float(case.parameters["dv"])
    feed = float(case.parameters["feed"])
    kill = float(case.parameters["kill"])
    now = float(output_times[0])
    for output_index, target in enumerate(output_times[1:], start=1):
        while now < float(target) - 1e-14:
            step = min(SOURCE_DERIVED_AUDIT_MAX_STEP, float(target) - now)
            state = rk4_step(state, 0.5 * step, lambda value: _reaction(value, feed, kill))
            state = _exact_periodic_diffusion(state, step, du, dv)
            state = rk4_step(state, 0.5 * step, lambda value: _reaction(value, feed, kill))
            now += step
        output[output_index] = state
    return output[None, ...]


def make_case(
    *, seed: int, split: str = "heldout", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {
        "du": float(rng.choice(SOURCE_DERIVED_DU)),
        "dv": float(rng.choice(SOURCE_DERIVED_DV)),
        "feed": float(rng.choice(SOURCE_DERIVED_FEED)),
        "kill": float(rng.choice(SOURCE_DERIVED_KILL)),
        "n": int(rng.choice(SOURCE_DERIVED_GRID_SIZES)),
        "time_count": int(rng.choice(SOURCE_DERIVED_TIME_COUNTS)),
        "final_time": SOURCE_DERIVED_FINAL_TIME,
        "input_kind": str(rng.choice(SOURCE_DERIVED_INPUT_KINDS)),
    }
    if stratum:
        unknown = set(stratum) - set(values)
        if unknown:
            raise ValueError(f"unknown gray_scott2d stratum keys: {sorted(unknown)}")
        values.update(stratum)
    return make_case_record(
        family_id="gray_scott2d",
        split=split,
        seed=seed,
        grid_shape=(int(values["n"]), int(values["n"])),
        time_count=int(values["time_count"]),
        final_time=float(values["final_time"]),
        input_kind=str(values["input_kind"]),
        parameters={
            "du": float(values["du"]),
            "dv": float(values["dv"]),
            "feed": float(values["feed"]),
            "kill": float(values["kill"]),
        },
    )


class GrayScott2DEnvironment:
    family_id = "gray_scott2d"
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
                f"\nFor this task, du={case.parameters['du']}, dv={case.parameters['dv']}, "
                f"feed={case.parameters['feed']}, and kill={case.parameters['kill']}."
            )
        if form == "parameter_ic":
            conditioning += f"\nThe initial-field family is {case.input_kind}."
        if form not in {"generic", "parameter", "parameter_ic"}:
            raise ValueError(f"unknown prompt form: {form}")
        return f"""Solve the periodic two-dimensional Gray--Scott system

    u_t = du * Laplacian(u) - u * v**2 + feed * (1 - u)
    v_t = dv * Laplacian(v) + u * v**2 - (feed + kill) * v

on (x, y) in [0, 1)^2.

Write this function:

    def solver(uv0_batch, t_coordinate, du, dv, feed, kill):
        ...

uv0_batch has shape [B, Nx, Ny, 2], with channels (u, v). t_coordinate has
shape [T]. Return a finite NumPy array with shape [B, T, Nx, Ny, 2], aligned
with the requested times. Internal substeps are allowed. Do not hard-code a
solution array.{conditioning}
"""

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("uv0_batch", "t_coordinate", "du", "dv", "feed", "kill"),
            arrays={"uv0_batch": initial_state(case), "t_coordinate": times(case)},
            scalars={key: float(case.parameters[key]) for key in ("du", "dv", "feed", "kill")},
            expected_shape=(1, len(times(case)), *case.grid_shape, 2),
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
        dx = 1.0 / case.grid_shape[0]
        laplacian = periodic_second(values, dx, axis=2) + periodic_second(values, dx, axis=3)
        u = values[..., 0]
        v = values[..., 1]
        uv2 = u * v**2
        residual_u = derivative[..., 0] - (
            float(case.parameters["du"]) * laplacian[..., 0]
            - uv2
            + float(case.parameters["feed"]) * (1.0 - u)
        )
        residual_v = derivative[..., 1] - (
            float(case.parameters["dv"]) * laplacian[..., 1]
            + uv2
            - (float(case.parameters["feed"]) + float(case.parameters["kill"])) * v
        )
        return float(np.sqrt(np.mean(residual_u**2) + np.mean(residual_v**2)))
