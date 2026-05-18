"""Engine-level configuration.

Lives inside :mod:`rlmflow.engine` so engine helpers can import it
without ever reaching back up into :mod:`rlmflow.rlm` (which would
re-introduce the same circular-import shape this package was
restructured to avoid).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLMConfig:
    """Engine-level knobs."""

    max_depth: int = 5
    max_iterations: int = 30
    max_output_length: int = 12000
    max_messages: int | None = None
    max_concurrency: int | None = None
    child_max_iterations: int | None = None
    single_block: bool = True
    system_prompt: str | None = None
    max_budget: int | None = None


__all__ = ["RLMConfig"]
