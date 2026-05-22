"""Helpers the engine builds on.

:class:`~rlmflow.rlm.RLMFlow` is the engine — it owns state and the
loop, and every overridable seam lives there as a method. This
package is its toolbox: pure functions, pure data, and implementation
helpers called by the public ``RLMFlow`` methods.

- :mod:`~rlmflow.engine.actions` — :class:`Action` types
  (:class:`CallLLM` / :class:`Exec` / :class:`Resume`) and the pure
  projection ``Graph -> ActionPlan`` (:func:`act_one` / :func:`act`).
- :mod:`~rlmflow.engine.replay` — cold-start replay-of-one for
  rebuilding a suspended coroutine after a fork or process restart.
- :mod:`~rlmflow.engine.scheduler` — :class:`NodeScheduler`: pick the
  agents that can take a step right now (pure top-down walk over a
  :class:`~rlmflow.graph.Graph`).
- :mod:`~rlmflow.engine.scheduling` — implementation of the outer
  ``RLMFlow.step`` loop and async-child refill policy.
- :mod:`~rlmflow.engine.transitions` — implementation of action-to-state
  transition handlers behind ``RLMFlow.apply_one`` / ``step_exec`` /
  ``step_after_supervising``.
- :mod:`~rlmflow.engine.seq` — tiny pure helpers (sequence numbers,
  iteration counts, budget checks, output truncation/formatting, the
  pool factory).
- :mod:`~rlmflow.engine.config` — :class:`RLMConfig`. Pure data.

If something is a user-facing override seam, it stays as a method on
:class:`~rlmflow.rlm.RLMFlow`. Some method implementations delegate here
to keep the façade readable without hiding the public API.
"""
