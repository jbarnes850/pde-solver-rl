from __future__ import annotations

from typing import Any

from rlvp.contracts import PDEEnvironment

from .advection1d import Advection1DEnvironment
from .advection1d_domain import Advection1DDomainEnvironment
from .advection2d import Advection2DEnvironment
from .advection_diffusion1d import AdvectionDiffusion1DEnvironment
from .advection_diffusion2d import AdvectionDiffusion2DEnvironment
from .advection_reaction_diffusion1d import AdvectionReactionDiffusion1DEnvironment
from .airy1d import Airy1DEnvironment
from .allen_cahn2d import AllenCahn2DEnvironment
from .burgers1d import Burgers1DEnvironment
from .cahn_hilliard1d import CahnHilliard1DEnvironment
from .darcy2d import Darcy2DEnvironment
from .darcy_reaction2d import DarcyReaction2DEnvironment
from .diffusion_sorption1d import DiffusionSorption1DEnvironment
from .gray_scott2d import GrayScott2DEnvironment
from .heat1d import Heat1DEnvironment
from .hyperdiffusion1d import Hyperdiffusion1DEnvironment
from .incompressible_ns2d import IncompressibleNS2DEnvironment
from .kdv1d import KdV1DEnvironment
from .reaction_diffusion1d import ReactionDiffusion1DEnvironment
from .reaction_diffusion2d import ReactionDiffusion2DEnvironment
from .shallow_water2d import ShallowWater2DEnvironment

_ENVIRONMENTS: dict[str, PDEEnvironment] = {
    "advection1d": Advection1DEnvironment(),
    "advection2d": Advection2DEnvironment(),
    "advection_diffusion1d": AdvectionDiffusion1DEnvironment(),
    "advection_diffusion2d": AdvectionDiffusion2DEnvironment(),
    "advection_reaction_diffusion1d": AdvectionReactionDiffusion1DEnvironment(),
    "burgers1d": Burgers1DEnvironment(),
    "cahn_hilliard1d": CahnHilliard1DEnvironment(),
    "darcy2d": Darcy2DEnvironment(),
    "darcy_reaction2d": DarcyReaction2DEnvironment(),
    "diffusion_sorption1d": DiffusionSorption1DEnvironment(),
    "gray_scott2d": GrayScott2DEnvironment(),
    "heat1d": Heat1DEnvironment(),
    "incompressible_ns2d": IncompressibleNS2DEnvironment(),
    "kdv1d": KdV1DEnvironment(),
    "allen_cahn2d": AllenCahn2DEnvironment(),
    "reaction_diffusion1d": ReactionDiffusion1DEnvironment(),
    "reaction_diffusion2d": ReactionDiffusion2DEnvironment(),
    "shallow_water2d": ShallowWater2DEnvironment(),
    # Extra training-only families (rlvp.manifests.EXTRA_TRAIN_FAMILIES).
    "advection1d_domain": Advection1DDomainEnvironment(),
    "airy1d": Airy1DEnvironment(),
    "hyperdiffusion1d": Hyperdiffusion1DEnvironment(),
}


def register_environment(environment: PDEEnvironment) -> None:
    if environment.family_id in _ENVIRONMENTS:
        raise ValueError(f"duplicate PDE family: {environment.family_id}")
    _ENVIRONMENTS[environment.family_id] = environment


def get_environment(family_id: str) -> PDEEnvironment:
    try:
        return _ENVIRONMENTS[family_id]
    except KeyError as exc:
        raise KeyError(f"unknown PDE family: {family_id}") from exc


def environment_for_case(case: Any) -> PDEEnvironment:
    return get_environment(str(case.family_id))


def all_environments() -> tuple[PDEEnvironment, ...]:
    return tuple(_ENVIRONMENTS[key] for key in sorted(_ENVIRONMENTS))
