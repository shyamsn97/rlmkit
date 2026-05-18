"""Pure helpers the engine builds on.

:class:`~rlmflow.rlm.RLMFlow` is the engine — it owns state and the
loop, and every overridable seam lives there as a method. This
package is its toolbox: free functions and pure data with no engine
state of their own.

- :mod:`~rlmflow.engine.actions` — :class:`Action` types
  (:class:`CallLLM` / :class:`Exec` / :class:`Resume`) and the pure
  projection ``Graph -> ActionPlan`` (:func:`act_one` / :func:`act`).
- :mod:`~rlmflow.engine.replay` — cold-start replay-of-one for
  rebuilding a suspended generator after a fork or process restart.
- :mod:`~rlmflow.engine.seq` — tiny pure helpers (sequence numbers,
  iteration counts, budget checks, output truncation/formatting, the
  pool factory).
- :mod:`~rlmflow.engine.config` — :class:`RLMConfig`. Pure data.

If something needs engine state to do its job, it's a method on
:class:`~rlmflow.rlm.RLMFlow`. If it's a pure function of its
arguments, it lives here. No middle category.
"""
