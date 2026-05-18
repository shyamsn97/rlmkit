"""Engine-level configuration.

Lives inside :mod:`rlmflow.engine` so engine helpers can import it
without ever reaching back up into :mod:`rlmflow.rlm` (which would
re-introduce the same circular-import shape this package was
restructured to avoid).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_max_concurrency() -> int:
    """Default to full parallelism across runnable agents.

    Uses the host's CPU count (or 1 if it can't be determined).
    Agent work is mostly LLM I/O — there's no real upside to gating
    below the thread count by default. Users with rate-limit
    concerns or single-flight requirements should set this
    explicitly.
    """
    return os.cpu_count() or 1


@dataclass
class RLMConfig:
    """Engine-level knobs."""

    max_depth: int = 5
    max_iterations: int = 30
    max_output_length: int = 12000
    max_messages: int | None = None
    max_concurrency: int | None = field(default_factory=_default_max_concurrency)
    child_max_iterations: int | None = None
    single_block: bool = True
    system_prompt: str | None = None
    max_budget: int | None = None


__all__ = ["RLMConfig"]
