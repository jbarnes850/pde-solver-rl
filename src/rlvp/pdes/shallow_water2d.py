from __future__ import annotations

from typing import Any, Literal

import numpy as np

from rlvp.contracts import SolverInvocation

from .base import PDECase, make_case_record
from .numerics import time_grid

PAPER_GRAVITIES = (1.0, 2.0)
PAPER_GRID_SIZES = (32, 48)
PAPER_TIME_COUNTS = (31, 51)
PAPER_FINAL_TIME = 1.0
PAPER_DAM_RADII = (0.35, 0.50, 0.65)
DOMAIN_LOWER = -2.5
DOMAIN_UPPER = 2.5

# The paper states only that the input is a radial dam-break state. It does not publish the
# two depths or the dam center. These values are explicit source-derived choices, not paper
# facts. They define a wet-bed radial dam break and avoid an undocumented dry-bed treatment.
SOURCE_DERIVED_INNER_DEPTH = 2.0
SOURCE_DERIVED_OUTER_DEPTH = 1.0
SOURCE_DERIVED_DAM_CENTER = (0.0, 0.0)
HEIGHT_FLOOR = 1.0e-8

SHALLOW_WATER_METHOD_IDS = (
    "fv_muscl_hll_ssprk3",
    "fv_muscl_rusanov_ssprk2",
    "fv_muscl_rusanov_ssprk3",
    "kt_central_upwind_muscl_ssprk3",
    "weno5_rusanov_ssprk3_pos",
    "roe_entropy_fix_muscl_ssprk3",
    "hllc_muscl_ssprk3_pos",
)


def _float(case: PDECase, name: str) -> float:
    return float(case.parameters[name])


def x_coordinate(case: PDECase) -> np.ndarray:
    return np.linspace(
        DOMAIN_LOWER, DOMAIN_UPPER, case.grid_shape[0], endpoint=True, dtype=np.float64
    )


def y_coordinate(case: PDECase) -> np.ndarray:
    return np.linspace(
        DOMAIN_LOWER, DOMAIN_UPPER, case.grid_shape[1], endpoint=True, dtype=np.float64
    )


def t_coordinate(case: PDECase) -> np.ndarray:
    if case.time_count is None or case.final_time is None:
        raise ValueError("shallow_water2d requires time_count and final_time")
    return time_grid(case.time_count, case.final_time)


def initial_condition(case: PDECase) -> np.ndarray:
    if case.input_kind != "radial_dam_break":
        raise ValueError(f"unknown shallow_water2d input kind: {case.input_kind}")
    xx, yy = np.meshgrid(x_coordinate(case), y_coordinate(case), indexing="ij")
    radius = np.hypot(xx - _float(case, "center_x"), yy - _float(case, "center_y"))
    height = np.where(
        radius <= _float(case, "dam_radius"),
        _float(case, "inner_depth"),
        _float(case, "outer_depth"),
    )
    return np.stack((height, np.zeros_like(height), np.zeros_like(height)), axis=-1)


def make_case(
    *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
) -> PDECase:
    rng = np.random.default_rng(seed)
    selected: dict[str, Any] = {
        "grav": float(rng.choice(PAPER_GRAVITIES)),
        "nx": int(rng.choice(PAPER_GRID_SIZES)),
        "ny": int(rng.choice(PAPER_GRID_SIZES)),
        "nt": int(rng.choice(PAPER_TIME_COUNTS)),
        "final_time": PAPER_FINAL_TIME,
        "dam_radius": float(rng.choice(PAPER_DAM_RADII)),
        "input_kind": "radial_dam_break",
    }
    if stratum:
        selected.update(stratum)
    parameters: dict[str, int | float | str] = {
        "grav": float(selected["grav"]),
        "dam_radius": float(selected["dam_radius"]),
        "inner_depth": SOURCE_DERIVED_INNER_DEPTH,
        "outer_depth": SOURCE_DERIVED_OUTER_DEPTH,
        "center_x": SOURCE_DERIVED_DAM_CENTER[0],
        "center_y": SOURCE_DERIVED_DAM_CENTER[1],
        "boundary": "zero_gradient_extrapolation",
        "local_choice": "centered_wet_bed_depth_ratio_2_to_1",
    }
    return make_case_record(
        family_id="shallow_water2d",
        split=split,
        seed=seed,
        grid_shape=(int(selected["nx"]), int(selected["ny"])),
        time_count=int(selected["nt"]),
        final_time=float(selected["final_time"]),
        input_kind=str(selected["input_kind"]),
        parameters=parameters,
    )


def prompt(case: PDECase, *, form: str = "generic") -> str:
    if form not in {"generic", "parameter", "parameter_ic"}:
        raise ValueError(f"unknown prompt form: {form}")
    conditioning = ""
    if form in {"parameter", "parameter_ic"}:
        conditioning += f"\nFor this task, grav={_float(case, 'grav')}."
    if form == "parameter_ic":
        conditioning += (
            "\nThe initial condition is a centered radial wet-bed dam break with "
            f"radius={_float(case, 'dam_radius')}."
        )
    return f"""Define exactly:
    def solver(h0_batch, hu0_batch, hv0_batch, t_coordinate, grav):

Solve the two-dimensional shallow-water equations in conservative variables
q=(h, hu, hv) on [-2.5, 2.5]^2:

    h_t  + (hu)_x + (hv)_y = 0,
    hu_t + (hu^2/h + grav*h^2/2)_x + (hu*hv/h)_y = 0,
    hv_t + (hu*hv/h)_x + (hv^2/h + grav*h^2/2)_y = 0.

Use zero-gradient extrapolation at all boundaries. Each initial array has shape
[B, Nx, Ny]. t_coordinate has shape [T]. Return one finite NumPy array with
shape [B, T, Nx, Ny, 3], ordered as (h, hu, hv). Returned slice n must
correspond to t_coordinate[n]. Keep h positive. Internal CFL substeps are
allowed. Do not hard-code a solution array.{conditioning}
"""


def invocation(case: PDECase) -> SolverInvocation:
    state = initial_condition(case)
    times = t_coordinate(case)
    return SolverInvocation(
        argument_order=("h0_batch", "hu0_batch", "hv0_batch", "t_coordinate", "grav"),
        arrays={
            "h0_batch": state[None, ..., 0],
            "hu0_batch": state[None, ..., 1],
            "hv0_batch": state[None, ..., 2],
            "t_coordinate": times,
        },
        scalars={"grav": _float(case, "grav")},
        expected_shape=(1, len(times), *case.grid_shape, 3),
    )


def _safe_primitives(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = np.maximum(state[..., 0], HEIGHT_FLOOR)
    return height, state[..., 1] / height, state[..., 2] / height


def _physical_flux(state: np.ndarray, grav: float, axis: int) -> np.ndarray:
    height, velocity_x, velocity_y = _safe_primitives(state)
    if axis == 0:
        return np.stack(
            (
                state[..., 1],
                state[..., 1] * velocity_x + 0.5 * grav * height**2,
                state[..., 1] * velocity_y,
            ),
            axis=-1,
        )
    if axis == 1:
        return np.stack(
            (
                state[..., 2],
                state[..., 2] * velocity_x,
                state[..., 2] * velocity_y + 0.5 * grav * height**2,
            ),
            axis=-1,
        )
    raise ValueError(f"invalid shallow-water flux axis: {axis}")


def _mc_slope(state: np.ndarray, axis: int) -> np.ndarray:
    """Componentwise monotonized-central slope with a positive-height limiter."""
    pad_width = [(0, 0)] * state.ndim
    pad_width[axis] = (1, 1)
    padded = np.pad(state, pad_width, mode="edge")
    center = [slice(None)] * state.ndim
    left = center.copy()
    right = center.copy()
    center[axis] = slice(1, -1)
    left[axis] = slice(0, -2)
    right[axis] = slice(2, None)
    backward = padded[tuple(center)] - padded[tuple(left)]
    forward = padded[tuple(right)] - padded[tuple(center)]
    centered = 0.5 * (backward + forward)
    same_sign = (np.sign(backward) == np.sign(forward)) & (np.sign(backward) == np.sign(centered))
    slope = np.where(
        same_sign,
        np.sign(centered)
        * np.minimum(np.abs(centered), np.minimum(2.0 * np.abs(backward), 2.0 * np.abs(forward))),
        0.0,
    )

    # Scale all conserved-variable slopes in a cell if either reconstructed depth would
    # cross the floor. This preserves velocity ratios better than clipping depth alone.
    height = state[..., 0]
    height_slope = slope[..., 0]
    admissible = np.maximum(height - HEIGHT_FLOOR, 0.0)
    theta = np.minimum(1.0, 2.0 * admissible / np.maximum(np.abs(height_slope), 1.0e-30))
    return slope * theta[..., None]


def _face_states(state: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    slope = _mc_slope(state, axis)
    minus = state - 0.5 * slope
    plus = state + 0.5 * slope
    if axis == 0:
        left = np.concatenate((state[0:1], plus), axis=0)
        right = np.concatenate((minus, state[-1:]), axis=0)
        return left, right
    if axis == 1:
        left = np.concatenate((state[:, 0:1], plus), axis=1)
        right = np.concatenate((minus, state[:, -1:]), axis=1)
        return left, right
    raise ValueError(f"invalid shallow-water reconstruction axis: {axis}")


def _normal_speed(state: np.ndarray, grav: float, axis: int) -> tuple[np.ndarray, np.ndarray]:
    height, velocity_x, velocity_y = _safe_primitives(state)
    velocity = velocity_x if axis == 0 else velocity_y
    return velocity, np.sqrt(grav * height)


def _rusanov_flux(left: np.ndarray, right: np.ndarray, grav: float, axis: int) -> np.ndarray:
    flux_left = _physical_flux(left, grav, axis)
    flux_right = _physical_flux(right, grav, axis)
    velocity_left, wave_left = _normal_speed(left, grav, axis)
    velocity_right, wave_right = _normal_speed(right, grav, axis)
    speed = np.maximum(np.abs(velocity_left) + wave_left, np.abs(velocity_right) + wave_right)
    return 0.5 * (flux_left + flux_right) - 0.5 * speed[..., None] * (right - left)


def _hlle_flux(left: np.ndarray, right: np.ndarray, grav: float, axis: int) -> np.ndarray:
    flux_left = _physical_flux(left, grav, axis)
    flux_right = _physical_flux(right, grav, axis)
    velocity_left, wave_left = _normal_speed(left, grav, axis)
    velocity_right, wave_right = _normal_speed(right, grav, axis)
    speed_left = np.minimum(velocity_left - wave_left, velocity_right - wave_right)
    speed_right = np.maximum(velocity_left + wave_left, velocity_right + wave_right)
    denominator = np.maximum(speed_right - speed_left, 1.0e-14)
    middle = (
        speed_right[..., None] * flux_left
        - speed_left[..., None] * flux_right
        + (speed_left * speed_right)[..., None] * (right - left)
    ) / denominator[..., None]
    return np.where(
        (speed_left >= 0.0)[..., None],
        flux_left,
        np.where((speed_right <= 0.0)[..., None], flux_right, middle),
    )


def _flux_divergence(
    state: np.ndarray,
    grav: float,
    spacing_x: float,
    spacing_y: float,
    kind: Literal["rusanov", "hlle"],
) -> np.ndarray:
    flux_function = _rusanov_flux if kind == "rusanov" else _hlle_flux
    left_x, right_x = _face_states(state, axis=0)
    left_y, right_y = _face_states(state, axis=1)
    flux_x = flux_function(left_x, right_x, grav, axis=0)
    flux_y = flux_function(left_y, right_y, grav, axis=1)
    return (flux_x[1:] - flux_x[:-1]) / spacing_x + (flux_y[:, 1:] - flux_y[:, :-1]) / spacing_y


def _enforce_positivity(state: np.ndarray) -> np.ndarray:
    output = np.asarray(state, dtype=np.float64).copy()
    invalid = output[..., 0] < HEIGHT_FLOOR
    output[..., 0] = np.maximum(output[..., 0], HEIGHT_FLOOR)
    output[..., 1] = np.where(invalid, 0.0, output[..., 1])
    output[..., 2] = np.where(invalid, 0.0, output[..., 2])
    return output


def _stable_step(
    state: np.ndarray, grav: float, spacing_x: float, spacing_y: float, cfl: float
) -> float:
    height, velocity_x, velocity_y = _safe_primitives(state)
    gravity_wave = np.sqrt(grav * height)
    spectral_radius = (np.abs(velocity_x) + gravity_wave) / spacing_x
    spectral_radius += (np.abs(velocity_y) + gravity_wave) / spacing_y
    return cfl / max(float(np.max(spectral_radius)), 1.0e-14)


def _integrate(
    case: PDECase,
    *,
    flux_kind: Literal["rusanov", "hlle"],
    integrator: Literal["ssprk2", "ssprk3"],
    cfl: float,
) -> np.ndarray:
    times = t_coordinate(case)
    grav = _float(case, "grav")
    spacing_x = (DOMAIN_UPPER - DOMAIN_LOWER) / (case.grid_shape[0] - 1)
    spacing_y = (DOMAIN_UPPER - DOMAIN_LOWER) / (case.grid_shape[1] - 1)
    state = initial_condition(case).astype(np.float64, copy=True)
    output = np.empty((1, len(times), *case.grid_shape, 3), dtype=np.float64)
    output[0, 0] = state
    now = float(times[0])

    def euler_stage(values: np.ndarray, step: float) -> np.ndarray:
        divergence = _flux_divergence(values, grav, spacing_x, spacing_y, flux_kind)
        return _enforce_positivity(values - step * divergence)

    for output_index, target in enumerate(times[1:], start=1):
        target_time = float(target)
        while now < target_time - 1.0e-14:
            step = min(_stable_step(state, grav, spacing_x, spacing_y, cfl), target_time - now)
            first = euler_stage(state, step)
            second_euler = euler_stage(first, step)
            if integrator == "ssprk2":
                state = _enforce_positivity(0.5 * state + 0.5 * second_euler)
            else:
                second = _enforce_positivity(0.75 * state + 0.25 * second_euler)
                third_euler = euler_stage(second, step)
                state = _enforce_positivity(state / 3.0 + 2.0 * third_euler / 3.0)
            now += step
        output[0, output_index] = state
    return output


def reference_primary(case: PDECase) -> np.ndarray:
    """Positivity-limited MUSCL-Rusanov finite-volume reference with SSPRK3."""
    return _integrate(case, flux_kind="rusanov", integrator="ssprk3", cfl=0.32)


def reference_audit(case: PDECase) -> np.ndarray:
    """Independent HLLE-flux, SSPRK2 audit trajectory at a lower CFL number."""
    return _integrate(case, flux_kind="hlle", integrator="ssprk2", cfl=0.24)


def _central_conservative_divergence(
    state: np.ndarray, grav: float, spacing_x: float, spacing_y: float
) -> np.ndarray:
    """Flux-form divergence without adding a primitive-variable residual."""
    physical_x = _physical_flux(state, grav, axis=0)
    physical_y = _physical_flux(state, grav, axis=1)
    faces_x = np.concatenate(
        (
            physical_x[0:1],
            0.5 * (physical_x[:-1] + physical_x[1:]),
            physical_x[-1:],
        ),
        axis=0,
    )
    faces_y = np.concatenate(
        (
            physical_y[:, 0:1],
            0.5 * (physical_y[:, :-1] + physical_y[:, 1:]),
            physical_y[:, -1:],
        ),
        axis=1,
    )
    return (faces_x[1:] - faces_x[:-1]) / spacing_x + (
        faces_y[:, 1:] - faces_y[:, :-1]
    ) / spacing_y


def residual_norm(field: np.ndarray, case: PDECase) -> float:
    values = np.asarray(field, dtype=np.float64)
    times = t_coordinate(case)
    expected = (1, len(times), *case.grid_shape, 3)
    if values.shape != expected:
        raise ValueError("shallow_water2d residual received an invalid shape")
    if len(times) < 3:
        raise ValueError("shallow_water2d residual needs at least three output times")
    temporal = np.gradient(values, times, axis=1, edge_order=2)
    spacing_x = (DOMAIN_UPPER - DOMAIN_LOWER) / (case.grid_shape[0] - 1)
    spacing_y = (DOMAIN_UPPER - DOMAIN_LOWER) / (case.grid_shape[1] - 1)
    divergence = np.empty_like(values)
    for time_index in range(len(times)):
        divergence[0, time_index] = _central_conservative_divergence(
            values[0, time_index], _float(case, "grav"), spacing_x, spacing_y
        )
    residual = temporal + divergence
    return float(np.sqrt(np.mean(np.square(residual))))


class ShallowWater2DEnvironment:
    family_id = "shallow_water2d"
    seen_in_training = True
    method_ids = SHALLOW_WATER_METHOD_IDS

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "shallow_water2d":
            raise ValueError(f"wrong family for shallow_water2d: {case.family_id}")
        return case

    make_case = staticmethod(make_case)
    prompt = staticmethod(prompt)
    invocation = staticmethod(invocation)
    reference_primary = staticmethod(reference_primary)
    reference_audit = staticmethod(reference_audit)
    residual_norm = staticmethod(residual_norm)
