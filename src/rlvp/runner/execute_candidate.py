from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def main() -> None:
    work = Path("/work")
    spec = importlib.util.spec_from_file_location("candidate", work / "candidate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve = getattr(module, "solver", None)
    if not callable(solve):
        raise TypeError("candidate must define the callable solver requested by the prompt")
    call = json.loads((work / "call.json").read_text())
    array_names = set(call["array_names"])
    scalar_names = set(call["scalar_names"])
    with np.load(work / "input.npz", allow_pickle=False) as data:
        arguments = []
        for name in call["argument_order"]:
            if name in array_names:
                arguments.append(np.asarray(data[name]))
            elif name in scalar_names:
                arguments.append(np.asarray(data[name]).item())
            else:
                raise ValueError(f"unknown invocation argument: {name}")
        output = solve(*arguments)
    np.save(work / "output.npy", np.asarray(output), allow_pickle=False)


if __name__ == "__main__":
    main()
