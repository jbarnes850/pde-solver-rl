from __future__ import annotations

from collections.abc import Callable

from . import advection1d as advection1d_bank
from . import burgers1d as burgers1d_bank
from . import darcy2d as darcy2d_bank
from . import diffusion_sorption1d as diffusion_sorption1d_bank
from . import incompressible_ns2d as incompressible_ns2d_bank
from . import reaction_diffusion1d as reaction_diffusion1d_bank
from . import reaction_diffusion2d as reaction_diffusion2d_bank
from . import shallow_water2d as shallow_water2d_bank
from .catalog import get_solver_method

_RENDERERS: dict[str, Callable[[str], str]] = {
    "advection1d": advection1d_bank.render,
    "burgers1d": burgers1d_bank.render,
    "reaction_diffusion1d": reaction_diffusion1d_bank.render,
    "diffusion_sorption1d": diffusion_sorption1d_bank.render,
    "reaction_diffusion2d": reaction_diffusion2d_bank.render,
    "darcy2d": darcy2d_bank.render,
    "incompressible_ns2d": incompressible_ns2d_bank.render,
    "shallow_water2d": shallow_water2d_bank.render,
}


def render_teacher(method_id: str) -> str:
    """Render one tested clean-room teacher from its family-scoped method ID."""
    method = get_solver_method(method_id)
    try:
        source = _RENDERERS[method.family_id](method.codename)
    except KeyError as exc:
        raise KeyError(f"no clean-room renderer for family: {method.family_id}") from exc
    compile(source, f"<{method.method_id}>", "exec")
    if "def solver(" not in source:
        raise ValueError(f"rendered teacher has no solver definition: {method.method_id}")
    return source.strip() + "\n"


def validate_all_renderers() -> None:
    from .catalog import SOLVER_METHODS

    for method in SOLVER_METHODS:
        render_teacher(method.method_id)
