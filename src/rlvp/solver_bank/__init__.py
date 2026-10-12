"""Metadata for the source-derived clean-room reconstruction of the paper solver bank."""

from .base import SolverMethod
from .catalog import (
    IMPLEMENTATION_VERSION,
    PAPER_FAMILY_COUNTS,
    SOLVER_METHODS,
    all_solver_methods,
    get_solver_method,
    methods_for_family,
    validate_catalog,
)
from .render import render_teacher, validate_all_renderers

__all__ = [
    "IMPLEMENTATION_VERSION",
    "PAPER_FAMILY_COUNTS",
    "SOLVER_METHODS",
    "SolverMethod",
    "all_solver_methods",
    "get_solver_method",
    "methods_for_family",
    "render_teacher",
    "validate_all_renderers",
    "validate_catalog",
]
