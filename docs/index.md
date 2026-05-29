# rlmflow docs

Pick the doc that matches what you're trying to do.

## Get oriented

- [Blog post](blog.md) — long-form pitch. Why recursive language
  models, why graphs over flat traces, full needle-in-a-haystack
  walkthrough with the same exports the CLI ships.
- [Positioning](positioning.md) — when to use rlmflow vs
  rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.

## Use rlmflow

- [Control](control.md) — step loop, workspace resume, rewind,
  forks, `CONTEXT.read()` / slices, delegation via
  `launch_subagent` / `launch_subagents`, inline-first strategy, custom tools.
- [Node injection](injections.md) — append typed controller events to a
  running graph, then commit them through `agent.step(graph)`.
- [Observability](observability.md) — querying the `Graph`,
  workspace layout, export helpers, live tree, gantt, topology
  exports, Gradio viewer, CLI.
- [Runtimes](runtimes.md) — `Runtime` protocol, shipped runtimes
  (Local / Docker / Modal / E2B / Daytona), writing your own.
- [Prompt customization](prompt_customization.md) — `PromptBuilder`
  sections, deriving from the default prompt, full replacement.
- [Security](security.md) — trust model, Docker isolation knobs,
  engine-level caps, proxied tools, approval gates.

## Extend rlmflow

- [**Internals**](internals.md) — engine architecture, step
  lifecycle (`act` → `apply_one`), the REPL `yield` protocol,
  resume semantics, cold-start replay, persistence, and the full
  `RLMFlow` override surface. **Start here if you want to subclass
  the engine.**
- [`internal/node_model.md`](internal/node_model.md) — full
  state-machine spec, every legal transition, simulation
  walkthroughs.
