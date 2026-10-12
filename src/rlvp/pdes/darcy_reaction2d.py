from __future__ import annotations

from typing import Any

import numpy as np

from rlvp.contracts import SolverInvocation
from rlvp.pdes.base import PDECase, make_case_record
from rlvp.pdes.darcy2d import (
    PAPER_BETAS,
    PAPER_FIELD_KINDS,
    PAPER_GRID_SIZES,
    permeability_field,
    residual_norm,
    solve_audit,
    solve_primary,
)

# The paper names Darcy--reaction as held out but does not specify its equation, parameter grid,
# coefficient generator, interface, or boundary discretization. This explicit linear reaction term
# and grid are source-derived choices. They must not be reported as author-artifact exact.
SOURCE_DERIVED_REACTION_RATES = (0.1, 1.0, 10.0)


def make_case(*, seed: int, split: str = "test", stratum: dict[str, Any] | None = None) -> PDECase:
    rng = np.random.default_rng(seed)
    selected = dict(stratum or {})
    grid_size = int(selected.get("grid_size", rng.choice(PAPER_GRID_SIZES)))
    beta = float(selected.get("beta", rng.choice(PAPER_BETAS)))
    field_kind = str(selected.get("field_kind", rng.choice(PAPER_FIELD_KINDS)))
    reaction_rate = float(selected.get("reaction_rate", rng.choice(SOURCE_DERIVED_REACTION_RATES)))
    if grid_size not in PAPER_GRID_SIZES:
        raise ValueError(f"grid_size must be one of {PAPER_GRID_SIZES}")
    if beta not in PAPER_BETAS:
        raise ValueError(f"beta must be one of {PAPER_BETAS}")
    if field_kind not in PAPER_FIELD_KINDS:
        raise ValueError(f"field_kind must be one of {PAPER_FIELD_KINDS}")
    if reaction_rate not in SOURCE_DERIVED_REACTION_RATES:
        raise ValueError(f"reaction_rate must be one of {SOURCE_DERIVED_REACTION_RATES}")
    return make_case_record(
        family_id="darcy_reaction2d",
        split=split,
        seed=seed,
        grid_shape=(grid_size, grid_size),
        time_count=None,
        final_time=None,
        input_kind=field_kind,
        parameters={
            "beta": beta,
            "field_kind": field_kind,
            "reaction_rate": reaction_rate,
            "heldout_contract": "source-derived-linear-reaction-v1",
        },
    )


def _prompt(case: PDECase, *, form: str) -> str:
    if form not in {"generic", "parameter", "parameter_ic"}:
        raise ValueError(f"unsupported prompt form: {form}")
    details = ""
    if form in {"parameter", "parameter_ic"}:
        details += (
            f"\nFor this task:\n- beta = {float(case.parameters['beta']):g}"
            f"\n- reaction_rate = {float(case.parameters['reaction_rate']):g}"
        )
    if form == "parameter_ic":
        label = str(case.parameters["field_kind"]).replace("_", " ")
        details += f"\n- the permeability fields come from a {label} family"
    return f"""Write Python code only.

Define exactly:
    def solver(diffusion_batch, beta, reaction_rate):

Solve the PDE
    -div(a(x,y) grad u) + reaction_rate * u = beta
on (x,y) in [0,1]^2 with zero Dirichlet boundary conditions.{details}

Requirements:
- You may use NumPy and/or SciPy
- do not print anything
- do not access files or the network
- output only Python code
- Return a NumPy array of shape [B, Nx, Ny] containing the steady-state solution field.
- diffusion_batch has shape [B, Nx, Ny] and contains heterogeneous permeability fields a(x,y).
- beta is the scalar forcing term.
- reaction_rate is the nonnegative scalar linear reaction coefficient.
"""


class DarcyReaction2DEnvironment:
    family_id = "darcy_reaction2d"
    seen_in_training = False
    method_ids: tuple[str, ...] = ()

    @staticmethod
    def case_from_dict(value: dict[str, Any]) -> PDECase:
        case = PDECase.from_dict(value)
        if case.family_id != "darcy_reaction2d":
            raise ValueError("case does not belong to darcy_reaction2d")
        return case

    @staticmethod
    def make_case(
        *, seed: int, split: str = "test", stratum: dict[str, Any] | None = None
    ) -> PDECase:
        return make_case(seed=seed, split=split, stratum=stratum)

    @staticmethod
    def prompt(case: PDECase, *, form: str = "generic") -> str:
        return _prompt(case, form=form)

    @staticmethod
    def invocation(case: PDECase) -> SolverInvocation:
        return SolverInvocation(
            argument_order=("diffusion_batch", "beta", "reaction_rate"),
            arrays={"diffusion_batch": permeability_field(case)[None, :, :]},
            scalars={
                "beta": float(case.parameters["beta"]),
                "reaction_rate": float(case.parameters["reaction_rate"]),
            },
            expected_shape=(1, *case.grid_shape),
        )

    @staticmethod
    def reference_primary(case: PDECase) -> np.ndarray:
        return solve_primary(case, reaction_rate=float(case.parameters["reaction_rate"]))

    @staticmethod
    def reference_audit(case: PDECase) -> np.ndarray:
        return solve_audit(case, reaction_rate=float(case.parameters["reaction_rate"]))

    @staticmethod
    def residual_norm(field: np.ndarray, case: PDECase) -> float:
        return residual_norm(
            field,
            case,
            reaction_rate=float(case.parameters["reaction_rate"]),
        )
