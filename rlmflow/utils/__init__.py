"""Utility re-exports.

Code-parsing helpers from :mod:`rlmflow.utils.code` are cheap and imported
eagerly — the engine hot path (``rlm.py``, ``engine/transitions.py``) pulls
``find_code_blocks`` / ``check_wait_syntax`` on every turn.

The viewer/export helpers live in :mod:`rlmflow.utils.viewer`, which transitively
imports plotly (and, on demand, gradio). To keep that weight off the engine
path, those names are re-exported **lazily** via ``__getattr__`` — they only
import ``viewer`` when first accessed.
"""

from __future__ import annotations

from typing import Any

from rlmflow.utils.code import (
    check_wait_syntax,
    find_code_blocks,
    replace_code_block,
)

_LAZY_VIEWER = {
    "open_viewer",
    "render_html",
    "resolve_graphs",
    "save_gif",
    "save_html",
    "save_image",
    "save_steps",
}

__all__ = [
    "check_wait_syntax",
    "find_code_blocks",
    "open_viewer",
    "render_html",
    "resolve_graphs",
    "replace_code_block",
    "save_gif",
    "save_html",
    "save_image",
    "save_steps",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_VIEWER:
        from rlmflow.utils import viewer

        return getattr(viewer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
