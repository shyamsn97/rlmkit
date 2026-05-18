"""Backwards-compatible re-export of :class:`RLMConfig`.

The canonical home is :mod:`rlmflow.engine.config` — engine helpers
import it from there so they never need to reach back into
:mod:`rlmflow.rlm` for typing. This shim keeps existing
``from rlmflow.config import RLMConfig`` imports working.
"""

from __future__ import annotations

from rlmflow.engine.config import RLMConfig

__all__ = ["RLMConfig"]
