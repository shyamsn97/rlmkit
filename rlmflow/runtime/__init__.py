from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DaytonaRuntime": "rlmflow.runtime.sandbox.daytona",
    "DockerRuntime": "rlmflow.runtime.docker",
    "E2BRuntime": "rlmflow.runtime.sandbox.e2b",
    "LocalRuntime": "rlmflow.runtime.local",
    "ModalRuntime": "rlmflow.runtime.sandbox.modal",
    "RemoteFileRuntime": "rlmflow.runtime.sandbox.remote",
    "Runtime": "rlmflow.runtime.runtime",
    "ToolDef": "rlmflow.runtime.runtime",
}

__all__ = [
    "DaytonaRuntime",
    "DockerRuntime",
    "E2BRuntime",
    "LocalRuntime",
    "ModalRuntime",
    "RemoteFileRuntime",
    "Runtime",
    "ToolDef",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
