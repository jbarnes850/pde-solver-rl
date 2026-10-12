from __future__ import annotations

import math

import numpy as np

from .contracts import ExecutionResult, PDEEnvironment, RewardComponents, SolverInvocation
from .pdes.advection1d import AdvectionCase

# Strict scientific success: valid output with range-normalized trajectory nRMSE at most this.
STRICT_TRAJECTORY_THRESHOLD = 0.01


def normalized_trajectory_loss(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    normalization: str = "reference_range",
    epsilon: float = 1e-12,
) -> float:
    error = float(np.sqrt(np.mean(np.square(candidate - reference))))
    if normalization == "reference_l2":
        denominator = float(np.sqrt(np.mean(np.square(reference))))
    elif normalization == "reference_range":
        denominator = float(np.max(reference) - np.min(reference))
        if denominator <= epsilon:
            denominator = float(np.sqrt(np.mean(np.square(reference))))
        if denominator <= epsilon:
            denominator = 1.0
    else:
        raise ValueError(f"unknown trajectory normalization: {normalization}")
    return error / (denominator + epsilon)


def advection_residual_norm(field: np.ndarray, case: AdvectionCase) -> float:
    dt = case.t[1] - case.t[0]
    dx = 1.0 / case.nx
    du_dt = np.gradient(field, dt, axis=1, edge_order=2)
    du_dx = (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)) / (2.0 * dx)
    residual = du_dt + case.beta * du_dx
    return float(np.sqrt(np.mean(np.square(residual))))


def evaluate_advection(
    execution: ExecutionResult,
    case: AdvectionCase,
    *,
    trajectory_temperature: float = 0.05,
    residual_temperature: float = 2.0,
    normalization: str = "reference_range",
    epsilon: float = 1e-12,
) -> RewardComponents:
    validity = execution.validity
    if not validity or execution.output is None:
        return RewardComponents(
            validity=0.0,
            trajectory_loss=None,
            trajectory_reward=0.0,
            residual_loss=None,
            residual_reward=0.0,
            full_reward=0.0,
            anchored_skill=0.0,
            anchored_reward=0.0,
        )

    reference = case.reference()
    candidate = np.asarray(execution.output, dtype=np.float64)
    loss_l2 = normalized_trajectory_loss(
        candidate, reference, normalization="reference_l2", epsilon=epsilon
    )
    loss_range = normalized_trajectory_loss(
        candidate, reference, normalization="reference_range", epsilon=epsilon
    )
    trajectory_loss = {
        "reference_l2": loss_l2,
        "reference_range": loss_range,
    }[normalization]
    if not math.isfinite(trajectory_loss):
        trajectory_loss = None
        trajectory_reward = 0.0
    else:
        trajectory_reward = math.exp(-trajectory_loss / trajectory_temperature)
    candidate_residual = advection_residual_norm(candidate, case)
    reference_residual = advection_residual_norm(reference, case)
    if not math.isfinite(reference_residual):
        raise RuntimeError("the trusted advection reference has a nonfinite residual")
    if not math.isfinite(candidate_residual):
        candidate_residual = None
        residual_loss = None
        residual_reward = 0.0
    else:
        residual_loss = abs(candidate_residual - reference_residual) / (
            reference_residual + epsilon
        )
        residual_reward = math.exp(-residual_loss / residual_temperature)
    return RewardComponents(
        validity=1.0,
        trajectory_loss=trajectory_loss,
        trajectory_reward=trajectory_reward,
        residual_loss=residual_loss,
        residual_reward=residual_reward,
        full_reward=trajectory_reward * residual_reward,
        candidate_residual=candidate_residual,
        reference_residual=reference_residual,
        trajectory_loss_reference_l2=loss_l2,
        trajectory_loss_reference_range=loss_range,
    )


def evaluate_case(
    execution: ExecutionResult,
    environment: PDEEnvironment,
    case: object,
    *,
    trajectory_temperature: float = 0.05,
    residual_temperature: float = 2.0,
    normalization: str = "reference_range",
    epsilon: float = 1e-12,
    best_trivial: float | None = None,
) -> RewardComponents:
    """Apply the paper reward equations to any registered PDE family.

    When ``best_trivial`` is supplied, the anchored training reward is also computed from the
    range-normalized trajectory error. The paper components are unchanged either way.
    """
    validity = execution.validity
    if not validity or execution.output is None:
        return RewardComponents(
            validity=0.0,
            trajectory_loss=None,
            trajectory_reward=0.0,
            residual_loss=None,
            residual_reward=0.0,
            full_reward=0.0,
            anchored_skill=0.0,
            anchored_reward=0.0,
        )
    reference = np.asarray(environment.reference_primary(case), dtype=np.float64)
    candidate = np.asarray(execution.output, dtype=np.float64)
    loss_l2 = normalized_trajectory_loss(
        candidate, reference, normalization="reference_l2", epsilon=epsilon
    )
    loss_range = normalized_trajectory_loss(
        candidate, reference, normalization="reference_range", epsilon=epsilon
    )
    trajectory_loss = {"reference_l2": loss_l2, "reference_range": loss_range}[normalization]
    if not math.isfinite(trajectory_loss):
        trajectory_loss = None
        trajectory_reward = 0.0
    else:
        trajectory_reward = math.exp(-trajectory_loss / trajectory_temperature)
    candidate_residual = environment.residual_norm(candidate, case)
    reference_residual = environment.residual_norm(reference, case)
    if reference_residual is not None and not math.isfinite(reference_residual):
        raise RuntimeError("the trusted reference has a nonfinite PDE residual")
    if candidate_residual is not None and not math.isfinite(candidate_residual):
        candidate_residual = None
        residual_loss = None
        residual_reward = 0.0
    elif candidate_residual is None or reference_residual is None:
        residual_loss = None
        residual_reward = 1.0
    else:
        residual_loss = abs(candidate_residual - reference_residual) / (
            reference_residual + epsilon
        )
        residual_reward = math.exp(-residual_loss / residual_temperature)
    return RewardComponents(
        validity=1.0,
        trajectory_loss=trajectory_loss,
        trajectory_reward=trajectory_reward,
        residual_loss=residual_loss,
        residual_reward=residual_reward,
        full_reward=trajectory_reward * residual_reward,
        candidate_residual=candidate_residual,
        reference_residual=reference_residual,
        trajectory_loss_reference_l2=loss_l2,
        trajectory_loss_reference_range=loss_range,
        best_trivial_error=best_trivial,
        anchored_skill=None if best_trivial is None else anchored_skill(loss_range, best_trivial),
        anchored_reward=None if best_trivial is None else anchored_reward(1.0, loss_range, best_trivial),
    )


def trivial_predictions(invocation: SolverInvocation) -> dict[str, np.ndarray]:
    """Input-only predictors that need no PDE solve: zeros, persistence and constant means.

    For time-dependent families the initial state is the first array argument, or, when the
    state arrives as separate component arrays (shallow water), those leading arrays stacked
    on a trailing channel axis.
    """
    shape = tuple(invocation.expected_shape)
    predictions = {"zeros": np.zeros(shape)}
    arrays = [
        np.asarray(invocation.arrays[name], dtype=np.float64)
        for name in invocation.argument_order
        if name in invocation.arrays and name != "t_coordinate"
    ]
    if "t_coordinate" in invocation.arrays:
        state_shape = shape[:1] + shape[2:]
        initial = None
        if arrays and arrays[0].shape == state_shape:
            initial = arrays[0]
        else:
            channels = state_shape[-1]
            leading = arrays[:channels]
            if len(leading) == channels and all(a.shape == state_shape[:-1] for a in leading):
                initial = np.stack(leading, axis=-1)
        if initial is not None:
            # [B, N] and [B, Nx, Ny] states are scalar fields; [B, Nx, Ny, C] keeps channels.
            spatial = tuple(range(1, initial.ndim - 1 if initial.ndim >= 4 else initial.ndim))
            predictions["persistence"] = np.repeat(initial[:, None], shape[1], axis=1)
            mean = initial.mean(axis=spatial, keepdims=True)
            predictions["mean"] = np.broadcast_to(mean[:, None], shape).copy()
    elif arrays and arrays[0].shape == shape:
        mean = arrays[0].mean(axis=tuple(range(1, arrays[0].ndim)), keepdims=True)
        predictions["mean_input"] = np.broadcast_to(mean, shape).copy()
    return predictions


def best_trivial_error(environment: PDEEnvironment, case: object) -> float:
    """Smallest range-normalized nRMSE achieved by any input-only trivial predictor."""
    reference = np.asarray(environment.reference_primary(case), dtype=np.float64)
    errors = [
        normalized_trajectory_loss(prediction, reference, normalization="reference_range")
        for prediction in trivial_predictions(environment.invocation(case)).values()
        if prediction.shape == reference.shape
    ]
    finite = [value for value in errors if math.isfinite(value)]
    if not finite:
        raise RuntimeError("no finite trivial-predictor error for this case")
    return float(min(finite))


def anchored_skill(
    trajectory_loss_range: float | None,
    best_trivial: float,
    *,
    strict: float = STRICT_TRAJECTORY_THRESHOLD,
) -> float:
    """Log-error skill: 0 at or above the best trivial predictor, 1 at or below ``strict``."""
    error = trajectory_loss_range
    if error is None or not math.isfinite(error):
        return 0.0
    if error <= strict:
        return 1.0
    if error >= best_trivial:
        return 0.0
    return math.log(best_trivial / error) / math.log(best_trivial / strict)


def anchored_reward(
    validity: float,
    trajectory_loss_range: float | None,
    best_trivial: float,
    *,
    strict: float = STRICT_TRAJECTORY_THRESHOLD,
) -> float:
    """Training reward: half graded skill over trivial predictors, half strict success.

    Invalid or nonfinite outputs score 0, outputs no better than a trivial predictor score 0,
    and the maximum 1 requires the strict accuracy criterion.
    """
    error = trajectory_loss_range
    if not validity or error is None or not math.isfinite(error):
        return 0.0
    return 0.5 * anchored_skill(error, best_trivial, strict=strict) + 0.5 * float(error <= strict)


def group_advantages(rewards: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    return (values - values.mean()) / (values.std(ddof=0) + epsilon)
