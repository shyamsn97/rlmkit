"""Engine internals.

The :class:`~rlmflow.rlm.RLMFlow` class in ``rlmflow/rlm.py`` is the
public orchestrator. The modules in this package implement its
mechanics:

- ``transitions`` — per-current-node-kind step functions plus the
  ``step_agent`` dispatch.
- ``code`` — calling the LLM, extracting code, recording an
  :class:`~rlmflow.graph.ActionNode`, and running / resuming code.
- ``replay`` — replay-of-one for cold-starting a suspended generator
  after a fork or process restart.
- ``sessions`` — per-agent runtime sessions: ``runtime_for``,
  ``create_runtime_session``, ``inject_env``, ``register_tools``.

Child spawning lives directly on :class:`~rlmflow.rlm.RLMFlow` as
``spawn_child`` — it touches enough engine state that splitting it
out would just mean passing ``self`` to it as the first argument.
- ``messages`` — building LLM message lists and the system prompt
  (tools section, status section, etc.).
- ``seq`` — small shared helpers (``append_node``, ``unique_child_id``,
  iteration counts, budget checks, output truncation, formatting).
"""
