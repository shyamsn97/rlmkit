# Changelog

All notable changes to **rlmflow** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is on `0.x`, breaking changes can land on minor bumps —
each one is called out under **Breaking** below.

## [Unreleased]

### Breaking

- `delegate(name, query, context, *, model=...)` — `context` is now
  **mandatory** and **positional**. The previous `context=None` keyword
  default is gone. Pass `""` for code-only delegations. This eliminates a
  silent footgun where children inherited the parent's payload by
  accident; every delegation now declares its child's input explicitly.
  Migration: `delegate("name", "query")` → `delegate("name", "query", "")`.
- `RLMFlow.start(query, *, context=None)` — `context` is keyword-only
  and optional (root agent gets `""` if omitted). No call-site changes
  needed for callers passing only a query.

### Added

- `CONTEXT.fork()` on `ContextVariable` — explicit snapshot of the
  parent's payload for handoff to a child (use only when the child
  genuinely needs the parent's view: reviewers, auditors, retry).
  Default delegations should still pass a fresh slice or `""`.
- "Inline first" strategy bias in the default prompt: when the parent
  can write a known multi-file artifact end-to-end itself, do not
  delegate per-file. Multi-file delegation example replaced with a
  parent-writes-everything-inline example. Sibling-interface guardrail
  added for the cases where delegation IS the right call (children must
  USE sibling names as-is and PRODUCE their own exports in the exact
  shape the contract declares).
- `tests/test_prompt_capabilities.py` — snapshot-style tests that pin
  the default prompt's required vocabulary so future trims don't drop
  load-bearing phrases.
- `tests/test_session_variable.py` — `SessionVariable` tree-navigation
  methods (`parent`, `ancestors`, `children`, `subtree`, `tree`)
  derived from real cross-agent edges.
- `examples/data/notebook-coding-agent/` — canonical saved trace shared
  by `coding_agent.ipynb` (generator), `node_basics.ipynb` (querying),
  and `viz_walkthrough.ipynb` (rendering).
- CI workflow (`.github/workflows/ci.yml`): ruff + pytest matrix on
  3.11 / 3.12 / 3.13, runs on every PR and `push: main`. Tag-driven
  publishing remains in `release.yml`.
- Coverage instrumentation: `pytest-cov` in `[dev]`, `--cov=rlmflow` in
  CI, `[tool.coverage.*]` config in `pyproject.toml`.
- OOLONG benchmark harness under `benchmarks/oolong/` — runnable
  flat-vs-RLM comparison adapted from Prime Intellect's reference
  environment.

### Changed

- Default system prompt rewritten end-to-end. Sections reordered to
  capabilities-first (Role → REPL → Strategy → Tools → Context →
  Recursion → Session → Guardrails → Examples → Status). Per-section
  prose tightened (~20% fewer tokens, zero outbound URLs in the
  shipped prompt). Examples reduced to five canonical patterns: small
  task, chunk-and-aggregate, self-contained multi-file (inline),
  cross-agent recovery, reviewer (`CONTEXT.fork()`).
- `[viewer]` extra now declares its `plotly` dependency directly. The
  unused `[viz]` extra was removed (`plotly` was previously declared
  there but only imported by the gated `[viewer]` code path).
- Python support clarified: `requires-python = ">=3.11"` matches the
  shipped classifiers (3.10 dropped — never tested in CI). Ruff target
  bumped to `py311`.
- Project status classifier: `Alpha` → `Beta`.

### Fixed

- Boids notebook regression: cross-file schema drift (`Boid.pos.x`
  vs flat `boid.x`) caused by an over-strict guardrail and an
  over-aggressive multi-file-delegation example. Repaired by adding
  the bidirectional contract guardrail and replacing the example.
- Notebook agent ids reflect filename sanitization (`root.index_html`,
  `root.styles_css`, etc.) — `.` is the agent-tree delimiter, so
  filenames with dots are sanitized to underscores. `node_basics.ipynb`
  and `viz_walkthrough.ipynb` updated.
- Static example payloads no longer use the deprecated 2-arg
  `delegate(...)` form (`view_demo.py`, `showcase.py`, `best_of_n.py`).

## [0.1.3] — 2026-04-29

- Engine refactor: graph-first replay path, deterministic stepping
  semantics tightened, additional integration tests.

## [0.1.2] — 2026-04-29

- Renamed package to `rlmflow`. Session and context layout consolidated
  under `Workspace` with explicit `fork()`. Major engine refactor toward
  the typed-node graph model.

## [0.1.1] — 2026-04-23

- `rlmflow` CLI shipped: `view`, `render`, `version` subcommands;
  `render -f` accepts mermaid / mermaid-flowchart / mermaid-sequence /
  dot / d2 / tree / ascii-boxes / gantt-html / report-md / code-log /
  error-summary / tokens.

## [0.1.0] — 2026-04-23

Initial release.

- Recursive `RLMFlow` engine with typed nodes (`QueryNode`,
  `ActionNode`, `ObservationNode`, `SupervisingNode`, `ResumeNode`,
  `ResultNode`, `ErrorNode`).
- Runtimes: `LocalRuntime`, `SubprocessRuntime`, `DockerRuntime`,
  `ModalRuntime`.
- `Workspace` with `Session` (event log) and `Context` (data payload)
  stores, both with `fork()`.
- Visualization: terminal `live` view, mermaid / dot / d2 / sequence
  exports, gantt HTML, code-log, error-summary, token sparkline,
  budget burndown, bench table, Markdown report, Slack/Discord
  webhooks, Gradio viewer.
- Optional extras: `[openai]`, `[anthropic]`, `[viewer]`, `[all]`,
  `[dev]`.

[Unreleased]: https://github.com/shyamsn97/rlmflow/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/shyamsn97/rlmflow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/shyamsn97/rlmflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/shyamsn97/rlmflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/shyamsn97/rlmflow/releases/tag/v0.1.0
