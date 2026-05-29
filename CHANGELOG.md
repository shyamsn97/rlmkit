# Changelog

All notable changes to **rlmflow** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is on `0.x`, breaking changes can land on minor bumps —
each one is called out under **Breaking** below.

## [Unreleased]

### Added

- **Two-function delegation surface: `launch_subagent` / `launch_subagents`.**
  Agents now delegate exclusively through two `async` launchers installed in
  the REPL namespace. `await launch_subagent(query, num_steps=None,
  context="", *, name="subagent", model="default")` spawns one child and
  returns its finish string; `await launch_subagents(specs)` spawns many in
  parallel (each spec a dict with `query` and optional
  `num_steps`/`context`/`name`/`model`, or a bare query string) and returns a
  `list[str]` in spec order. Both must be awaited. Sequential pipelines chain
  `launch_subagent` calls (threading each result into the next `context=`);
  parallel fanout uses a single `launch_subagents([...])`. The launchers are
  registered as real core tools and compose over `rlm_delegate` / `rlm_wait`, so
  they behave identically on local and remote runtimes.

### Breaking

- **`rlm_delegate` / `rlm_wait` are now internal primitives, not the
  agent-facing API.** The launchers compose over them; agent code, the default
  prompt, examples, and docs all use `launch_subagent` / `launch_subagents`
  instead. The AST check (`check_wait_syntax`) now permits `await
  launch_subagent(...)` / `await launch_subagents(...)` at action-block top
  level. Update any custom prompts or hand-scripted REPL fixtures.
- **`OrphanedDelegatesError` removed.** Because spawn and wait are fused inside
  the launchers, an un-awaited delegate is no longer expressible, so the
  orphaned-delegate detection, its `ErrorOutput(error="orphaned_delegates")`
  node, and the remote exception-injection path are gone.
- **`RLMConfig.async_children` renamed to `eager_children`.** Same semantics
  (work-conserving child drain once a parent is supervising); update config
  literals and any persisted `agent.json` fixtures.

## [0.3.2] — 2026-05-28

### Added

- **Per-agent LLM transcript log.** Every workspace now writes
  `session/<aid>/transcript.json` — a *single* document per agent
  that grows turn-by-turn. `messages` is the flat conversation as
  the LLM saw it across every turn so far
  (`[{role: system|user|assistant, content: ...}, ...]`);
  `metadata` is a parallel list with one dict per message. Each
  call appends only the new entries (any user nudges plus the
  assistant reply) — no duplicated prefix. Per-assistant metadata:
  `{ts, model, force_final, input_tokens, output_tokens,
  elapsed_s, after_node_id, after_seq}`. Every other message gets
  `{}`. Exact ground truth for "what did the LLM see?" — useful
  for debugging prompt issues, replaying a turn under a different
  model, or auditing context bloat. The `Session` ABC gains
  `read_transcript(agent_id)` and `write_transcript(agent_id,
  transcript)`; `FileSession` round-trips through `transcript.json`,
  `InMemorySession` keeps it in `agent_transcripts`. Transcript
  read/write failures are swallowed so persistence issues never
  break a run.

### Breaking

- **`delegate` / `wait` renamed to `rlm_delegate` / `rlm_wait`.** The
  two engine-bound REPL tools used names that were too generic and
  shadowed common identifiers in agent code. They are now namespaced as
  `rlm_delegate(*, name, query, context, max_iterations=None,
  model="default")` and `rlm_wait(*handles)`. The default system prompt,
  built-in examples, error messages (`OrphanedDelegatesError`,
  refusal strings), and the AST check that enforces `yield` before
  `wait` all use the new names. Update agent prompts, custom prompt
  builders, and any test fixtures that script REPL code by hand.
- **Node taxonomy expanded to a 9-leaf, 4-base-class hierarchy with
  strict obs → action alternation.** Every action is now followed by
  exactly one observation; outputs no longer share a node with the
  action that produced them. New leaf classes (and wire-format
  `type` tags): `UserQuery`, `LLMAction`, `LLMOutput`, `ExecAction`,
  `ExecOutput`, `SupervisingOutput`, `ErrorOutput`, `DoneOutput`,
  `ResumeAction`. Base classes (Python-only — not on the wire):
  `Node`, `ObservationNode`, `ActionNode`, `CodeObservation`. The
  old `outcome=` enum on `ExecAction`, the unified `ActionNode`
  (LLM-call) shape, and `SeedAction` / `ResultNode` / `ErrorNode` /
  `QueryNode` / `SupervisingNode` / `ResumeNode` are gone. Predicates
  follow the new taxonomy: `is_user_query`, `is_llm_output`,
  `is_llm_action`, `is_exec_output`, `is_exec_action`,
  `is_supervising`, `is_errored`, `is_done`, `is_resume_action`,
  `is_resumed`, `is_observation`, `is_action`, `is_code_observation`.
  `LLMOutput.code` is the source of truth for executed code;
  `ExecAction` / `ResumeAction` carry an optional echo only.
  Full spec: [`docs/internal/node_model.md`](docs/internal/node_model.md).
- **`ErrorOutput` (formerly `ErrorNode`) now distinguishes runtime
  exceptions from normal `ExecOutput`.** The REPL protocol surfaces
  an `errored` flag; `engine/transitions.py` writes `ErrorOutput`
  whenever the runtime reports an exception (including `SyntaxError`
  and the synthetic "no code block" case), instead of mixing
  tracebacks into `ExecOutput`.
- **LLM clients retry transient failures via `tenacity`.** The
  `chat` and `stream` methods on `OpenAIClient` and
  `AnthropicClient` retry on transient HTTP / protocol errors. The
  module-level `_`-prefixed helpers and constants in `rlmflow/llm.py`
  are now public.
- **Workspace step retracing.** `Workspace.load_steps()` returns the
  full history as a list of progressive `Graph` snapshots. The
  retrace simulates **unbounded `max_concurrency`**: every tick
  advances all currently ready agents in lockstep, producing one
  snapshot per tick. The viewer / `save_steps` / `save_gif` /
  `save_html` / `open_viewer` deduplicate consecutive frames that
  collapse to the same visualization (e.g. action nodes hidden by
  their paired observation), so the resulting slider/animation
  shows only visually distinct steps.
- **Viewer renames + node collapsing.** "Yielded" is now
  "supervising" everywhere in display surfaces. By default the
  figure renderer hides bookkeeping action nodes whose paired
  observation has already been written: `llm_action` collapses
  into `llm_output`, `exec_action` / `resume_action` collapse into
  `exec_output` or `supervising_output`. Terminal outcomes
  (`done_output`, `error_output`) are **never** collapsed — the
  preceding action stays visible so `... → exec → done` and
  `... → resume → errored` read explicitly. Action nodes that are
  the latest state on an agent also stay visible so progress is
  observable. The state-detail panel in `open_viewer` renders each
  state as a distinct, color-coded, type-labeled block.
- **Data model is now one recursive class.** `AgentMeta` is gone — its
  fields are flat on `Graph` itself (`graph.query`, `graph.config`,
  `graph.runtime`, `graph.workspace`, `graph.depth`, `graph.model`,
  `graph.system_prompt`, `graph.branch_id`, `graph.parent_agent_id`,
  `graph.parent_node_id`). `Graph` is a frozen `dataclass` with
  `states: tuple[Node, ...]` and `children: dict[str, Graph]` for
  sub-agents. Cross-agent navigation is `graph[other_aid]`;
  subtree views are `graph.agents`, `graph.nodes`, `graph.edges`.
- `Graph.from_agent_states(...)` is removed. Build `Graph` instances
  directly (frozen dataclass) or rely on `Session.load_graph()`.
- `Edge` no longer ships as a stored object on `Graph` — `graph.edges`
  derives `flows_to` from each agent's state order and `spawns` from
  each child's `parent_node_id`. The class survives as a `NamedTuple`
  for viz consumers.
- `Session.write_agent` now takes a `Graph` (not an `AgentMeta`).
  `Session.record_spawn` is removed; the parent link is captured on
  the child's `parent_node_id` field.
- `Graph.events` is now `Graph.states` — every `Node` represents the
  agent's *state* at one step in its trajectory, not a discrete event.
- `latest.json` writes `latest_node_id` instead of `latest_event_id`.

## [0.2.1] — 2026-05-10

### Changed

- Workspace persistence now uses per-call `session/<agent-id>/session.jsonl`
  logs plus a top-level `graph.json` manifest for graph structure and state
  ordering.
- Removed old workspace compatibility paths; `FileSession(path)` and
  `FileContext(path)` now treat `path` as the current workspace root layout.
- Removed the redundant `CONTEXT.fork()` REPL helper; pass `CONTEXT.read()` or
  a slice explicitly to `delegate(...)`.
- Added public prompt customization docs covering `PromptBuilder`,
  `RLMConfig.system_prompt`, and dynamic prompt overrides.

## [0.2.0] — 2026-05-08

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
- `rlmflow.utils.save_image(node, path, ...)` — render a node's
  graph to PNG/SVG/PDF. Markers, edges, and fonts auto-scale via
  `element_mult` so the tree stays visually balanced on the larger
  export canvas. Promoted from a one-off notebook helper.
- `rlmflow.utils.save_steps(states, dir, ...)` — multi-snapshot
  variant: writes one image per state under `dir`.
- `rlmflow.utils.render_html(states, ...)` /
  `rlmflow.utils.save_html(states, path, ...)` — single-file
  standalone stepper. Each slide pairs the Plotly graph for one
  snapshot with that snapshot's transcript and a node table; bottom
  nav has arrows + dots, plus keyboard left/right. Drop the file in
  a PR comment, attach to a CI artifact, or commit it next to the
  trace it came from. Promoted from
  `examples/blog_needle_graph.py:render_html_viewer`.
- `rlmflow.utils.save_gif(states, path, ...)` — animate a trace as
  an autoplay GIF. Renders each state to PNG with kaleido, then
  stitches frames with Pillow. Lazy-imports Pillow (raises a clear
  ImportError otherwise) so `[image]` stays focused on still
  exports.
- `Node.save_image(path, ...)` and `Node.save_html(path, ...)`
  shorthands for the helpers above.
- `Node.plot(..., element_mult=)` and
  `node_plot(..., element_mult=)` — scale markers/edges/fonts on
  the returned Plotly figure. Default `1.0` keeps the on-screen
  layout; bump for hi-res rendering.
- Split scaling on `node.plot()` / `save_image` / `save_steps` /
  `save_gif`: `marker_mult` and `text_mult` override
  `element_mult` separately, so labels can stay small (e.g. `2.2`)
  while marker dots get fat (`3.5`). Fixes label collisions on
  dense trees.
- `normalize_labels=` on `node.plot()` and the save helpers —
  forces every node label to `bottom center` so adjacent depths
  can't share the same vertical band. Default off for `node.plot`
  (on-screen alternation still looks fine), default on for
  `save_image` / `save_steps` / `save_gif` / `Node.save_image`.
- CLI: `rlmflow render <trace> -f steps -o frames/` gains
  `--marker-mult`, `--text-mult`, `--normalize-labels` /
  `--no-normalize-labels` flags (also work with `-f image`). One
  invocation now replaces the per-blog one-off scripts.
- `[image]` optional extra (`pip install rlmflow[image]`) — pulls
  `plotly` and `kaleido` for static image export.

### Changed

- Default system prompt rewritten end-to-end. Sections reordered to
  capabilities-first (Role → REPL → Strategy → Tools → Context →
  Recursion → Session → Guardrails → Examples → Status). Per-section
  prose tightened (~20% fewer tokens, zero outbound URLs in the
  shipped prompt). Examples reduced to five canonical patterns: small
  task, chunk-and-aggregate, self-contained multi-file (inline),
  cross-agent recovery, reviewer (`CONTEXT.read()`).
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

[Unreleased]: https://github.com/shyamsn97/rlmflow/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/shyamsn97/rlmflow/compare/v0.2.1...v0.3.2
[0.2.1]: https://github.com/shyamsn97/rlmflow/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/shyamsn97/rlmflow/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/shyamsn97/rlmflow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/shyamsn97/rlmflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/shyamsn97/rlmflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/shyamsn97/rlmflow/releases/tag/v0.1.0
