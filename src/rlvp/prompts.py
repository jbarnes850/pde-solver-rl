from __future__ import annotations

import re

from .pdes.advection1d import AdvectionCase

SYSTEM_PROMPT = """You write numerical PDE solver code.
Return only Python code. Do not use Markdown fences. Do not print. Do not use files,
the network, subprocesses, or environment variables. NumPy and SciPy are allowed.
The verifier will call the exact function signature in the user request."""


def advection_prompt(case: AdvectionCase, *, form: str = "generic") -> str:
    conditioning = ""
    if form in {"parameter", "parameter_ic"}:
        conditioning += f"\nUse beta={case.beta}."
    if form == "parameter_ic":
        conditioning += f"\nThe initial-condition family is {case.ic_kind}."
    return f"""Solve the periodic one-dimensional linear advection equation

    u_t + beta * u_x = 0,  x in [0, 1).

Write this function:

    def solver(u0_batch, t_coordinate, beta):
        ...

u0_batch has shape [B, N]. t_coordinate has shape [T]. Return a finite NumPy
array with shape [B, T, N]. Returned slice n must correspond to
t_coordinate[n]. Internal substeps are allowed. Use a stable numerical method.
Do not hard-code a solution array.{conditioning}
"""


def extract_python_code(completion: str) -> str:
    text = str(completion or "").strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()
