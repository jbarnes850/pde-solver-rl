from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

type Scalar = int | float


@dataclass(frozen=True)
class SolverInvocation:
    """Public inputs and call order exposed to one candidate solver."""

    argument_order: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    scalars: dict[str, Scalar]
    expected_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        names = set(self.arrays) | set(self.scalars)
        if names != set(self.argument_order):
            raise ValueError("argument_order must name every invocation input exactly once")
        if set(self.arrays) & set(self.scalars):
            raise ValueError("an invocation input cannot be both an array and a scalar")
        if not self.expected_shape or any(value <= 0 for value in self.expected_shape):
            raise ValueError("expected_shape must contain positive dimensions")


class PDEEnvironment(Protocol):
    """Family-owned numerical contract used by manifests and the verifier."""

    family_id: str
    seen_in_training: bool
    method_ids: tuple[str, ...]

    def case_from_dict(self, value: dict[str, Any]) -> Any: ...

    def make_case(
        self, *, seed: int, split: str = "train", stratum: dict[str, Any] | None = None
    ) -> Any: ...

    def prompt(self, case: Any, *, form: str = "generic") -> str: ...

    def invocation(self, case: Any) -> SolverInvocation: ...

    def reference_primary(self, case: Any) -> np.ndarray: ...

    def reference_audit(self, case: Any) -> np.ndarray: ...

    def residual_norm(self, field: np.ndarray, case: Any) -> float | None: ...


class RewardMode(StrEnum):
    VALIDITY = "validity"
    TRAJECTORY = "trajectory"
    FULL = "full"
    ANCHORED = "anchored"


@dataclass(frozen=True)
class ExecutionResult:
    executed: bool
    shape_valid: bool
    finite: bool
    output: np.ndarray | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] | None = None

    @property
    def validity(self) -> float:
        return float(self.executed and self.shape_valid and self.finite)


@dataclass(frozen=True)
class RewardComponents:
    validity: float
    trajectory_loss: float | None
    trajectory_reward: float
    residual_loss: float | None
    residual_reward: float
    full_reward: float
    candidate_residual: float | None = None
    reference_residual: float | None = None
    trajectory_loss_reference_l2: float | None = None
    trajectory_loss_reference_range: float | None = None
    best_trivial_error: float | None = None
    anchored_skill: float | None = None
    anchored_reward: float | None = None

    def scalar(self, mode: RewardMode) -> float:
        if mode is RewardMode.VALIDITY:
            return self.validity
        if mode is RewardMode.TRAJECTORY:
            return self.validity * self.trajectory_reward
        if mode is RewardMode.FULL:
            return self.full_reward
        if mode is RewardMode.ANCHORED:
            if self.anchored_reward is None:
                raise ValueError("anchored reward requires the case's best trivial-predictor error")
            return self.anchored_reward
        raise ValueError(f"unsupported reward mode: {mode}")
