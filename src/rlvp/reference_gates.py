from __future__ import annotations

from types import MappingProxyType

# Fixed before policy evaluation. These are source-derived acceptance limits for the range
# nRMSE between the two independently coded reference routes. They are below the paper's
# 1e-2 solver-success threshold, except no family needs that exception in v1.
REFERENCE_RANGE_NRMSE_LIMITS = MappingProxyType(
    {
        "advection1d": 6.0e-3,
        "advection2d": 1.0e-10,
        "advection_diffusion1d": 1.0e-10,
        "advection_diffusion2d": 1.0e-10,
        "advection_reaction_diffusion1d": 9.0e-3,
        "allen_cahn2d": 9.0e-3,
        "burgers1d": 8.0e-3,
        "cahn_hilliard1d": 8.0e-3,
        "darcy2d": 1.0e-9,
        "darcy_reaction2d": 1.0e-9,
        "diffusion_sorption1d": 5.0e-5,
        "gray_scott2d": 1.0e-5,
        "heat1d": 1.0e-3,
        "incompressible_ns2d": 1.0e-2,
        "kdv1d": 5.0e-3,
        "reaction_diffusion1d": 1.0e-3,
        "reaction_diffusion2d": 1.0e-3,
        "shallow_water2d": 6.0e-3,
        # Extra training-only families: exact modal primary vs FFT or spectral-matrix
        # expm audit; observed max 7.2e-10 over 60 cases per family.
        "advection1d_domain": 1.0e-8,
        "airy1d": 1.0e-8,
        "hyperdiffusion1d": 1.0e-8,
    }
)


def validate_reference_gate_contract(family_ids: set[str]) -> None:
    configured = set(REFERENCE_RANGE_NRMSE_LIMITS)
    if configured != family_ids:
        raise ValueError(
            f"reference gate family mismatch; missing={sorted(family_ids - configured)}, "
            f"extra={sorted(configured - family_ids)}"
        )
    if any(not 0.0 < value <= 1.0e-2 for value in REFERENCE_RANGE_NRMSE_LIMITS.values()):
        raise ValueError("reference gates must be positive and at most the paper success limit")
