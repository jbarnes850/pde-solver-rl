"""Source-derived RLVP research environment.

This package trains and evaluates a policy that writes numerical solver code.
It does not train a verifier and it does not establish physical truth.
"""

from .contracts import ExecutionResult, RewardComponents, RewardMode

__all__ = ["ExecutionResult", "RewardComponents", "RewardMode"]
