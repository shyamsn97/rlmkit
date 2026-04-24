# Changelog

## 0.1.0 — initial release

First public release. A minimal state-machine library for recursive
LLM agents, with pluggable runtimes and a step-level control loop.

### Core

- `RLM` implements `LLMClient` — drop it in anywhere you'd use an LLM
  client. `chat(messages)` and `run(query)` run the full recursive
  agent loop under the hood.
- Immutable `RLMState` per `step()`. The full computation tree is one
  Pydantic object. `model_dump_json` / `model_validate_json` round-trip
  losslessly.
- Typed `StepEvent` hierarchy: `LLMReply`, `CodeExec`, `ResumeExec`,
  `NoCodeBlock`, `ChildStep`.
- Token tracking on every state. `tree_usage()` sums across the subtree.
  `RLMConfig.max_budget` auto-stops when a subtree exceeds a token cap.
- Session persistence via `Session` ABC. `FileSession` ships as the
  default; subclass for DB / remote backends.
- `ThreadPool`, `SequentialPool`, and user callables for child execution.
- Named model registry — `delegate(..., model="fast")` routes to a
  different LLM.

### Runtimes

- Two-method contract: `send(msg)` / `recv()`. Everything else
  (`execute`, `start_code`, `resume_code`, `inject`, proxy loop,
  workspace chdir) is in the `Runtime` base class.
- `LocalRuntime` — in-process Python execution.
- `SubprocessRuntime(argv)` — generic; spawn any command that runs
  `python -m rlmkit.runtime.repl` and speak JSON-over-stdio. Covers
  `docker exec`, `kubectl exec`, `ssh`, etc.
- `DockerRuntime` — `SubprocessRuntime` with an ergonomic
  `docker run` argv builder (mounts, env, network, cpus, memory,
  user, read-only, custom entrypoint).
- `ModalRuntime` — run the REPL inside a Modal container.
- Shipped `Dockerfile` and `rlmkit:local` image target.
- `Runtime.inject` handles three cases automatically: callables become
  host-backed proxies; Python literals are eval'd on the far side; any
  other object exposes its public methods as a method-proxy namespace.
- Proxied tool calls run with `os.getcwd()` set to the runtime's
  workspace, so relative paths behave the same everywhere.
- Exceptions raised by proxied tools are serialized and re-raised on
  the REPL side — the host stays alive and the agent sees a normal
  Python error.
- `from X import *` inside a code block that also `yield`s is handled
  by hoisting the star import to module level before wrapping the rest
  in a generator.

### Observability

- `state.tree()` — ASCII render of the full subtree.
- `save_trace` / `load_trace` / `view_trace` — JSON trace format, plus
  a Gradio-based viewer behind `rlmkit[viewer]`.
- Live terminal view via `rlmkit.utils.viz.live`.

### Packaging

- `pip install rlmkit[openai,anthropic,viewer,all]` extras.
- Top-level exports: `RLM`, `RLMConfig`, `RLMState`, `Status`,
  `LLMClient`, `LLMUsage`, `OpenAIClient`, `AnthropicClient`, plus all
  `StepEvent` subclasses, `ChildHandle`, `WaitRequest`.
- Python 3.11+.

### Tests

- 48 unit tests: REPL protocol, subprocess runtime end-to-end, code
  block parsing, nested delegation, release-contract smoke.
- 10 integration tests for the observability and control contracts.
- 3 Docker end-to-end tests (gated on `RLMKIT_DOCKER_TEST=1`) covering
  the regressions fixed during this cycle: object-proxy `inject`,
  `from X import *` with `yield`, workspace-relative tool writes.

### Docs

- `docs/positioning.md` — when to use rlmkit vs alternatives.
- `docs/observability.md` — state fields, events, traces, sessions.
- `docs/control.md` — step loop, checkpoint, fork, rewind, intervene,
  custom prompts / state / tools.
- `docs/runtimes.md` — `Runtime` protocol and shipped subclasses.
- `docs/security.md` — trust model, Docker isolation, approval gates.
