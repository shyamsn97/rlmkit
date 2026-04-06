# rlmkit

A state machine for [Recursive Language Model](https://github.com/alexzhang13/rlm-minimal) agents. Every agent — root and all descendants — advances one step at a time. The entire computation tree is a single immutable, serializable object at every step boundary.

```
pip install rlmkit
```

## The Idea

An LLM with a code REPL can solve problems by writing and executing code in a loop. When the problem is too big for one context window, it spawns sub-agents — each a fresh LLM+REPL with its own context — to handle pieces in parallel. Sub-agents can spawn their own. A tree grows:

```
root: "Find the security vulnerability in this repo"
├── scanner_auth: "Audit authentication in src/auth/"
│   ├── chunk_0: "Check login.py for injection"
│   ├── chunk_1: "Check session.py for fixation"
│   └── chunk_2: "Check tokens.py for leaks"
├── scanner_api: "Audit all API endpoints in src/api/"
│   ├── chunk_0: "Check users.py"
│   ├── chunk_1: "Check payments.py"
│   │   └── deep_scan: "Trace the payment flow end-to-end"
│   └── chunk_2: "Check admin.py"
└── scanner_db: "Audit database queries in src/db/"
    ├── chunk_0: "Check queries.py for SQLi"
    └── chunk_1: "Check migrations.py"
```

In most frameworks, this tree is a black box — you call `agent.run()` and wait. rlmkit makes it a **state machine**. You own the loop, you see every transition, you can checkpoint/fork/intervene between any two steps.

## Quick Start

```python
from rlmkit.llm import OpenAIClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime

agent = RLM(
    llm_client=OpenAIClient("gpt-5"),
    runtime=LocalRuntime(workspace="."),
    config=RLMConfig(max_depth=3, max_iterations=15, session="context"),
)

state = agent.start("Find and fix all type errors in src/")
while not state.finished:
    state = agent.step(state)
    print(state.event)
print(state.result)
```

Or: `result = agent.run("Find and fix all type errors in src/")`

## How It Works

Each `step(state) → state` is one atomic transition. Four statuses:

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
                WAITING ──── step_llm() ────→ HAS_REPLY       │
                  ▲  ▲                           │            │
                  │  │                     step_exec()        │
                  │  │                      ╱        ╲        │
                  │  └── no children ──────╴          ╲       │
                  │                                    ▼      │
                  │  done() ───────────────────→ FINISHED     │
                  │                                           │
                  │                              SUPERVISING ─┘
                  │                                  │
                  │                          step_supervise()
                  │                          flatten tree → step
                  └── children done ──────── leaves first → cascade
```

When code calls `delegate()` + `yield wait()`, the generator suspends. The engine flattens the entire tree, sorts deepest-first, and steps all leaves in parallel. Parents resume automatically when their children finish.

**Delegation:**

```python
h1 = delegate("searcher", "Find all TODOs in src/")
h2 = delegate("searcher", "Find all FIXMEs in src/")  # auto-suffixed: root.searcher_2
results = yield wait(h1, h2)
done(f"Found {len(results)} batches")
```

Re-delegating to a finished child resumes it with a new task — same REPL variables, fresh context window.

## What You Can Do

Because state is immutable and serializable, you get things for free that are hard in other frameworks:

- **Checkpoint & resume** — save `state` at any step, restore later
- **Fork** — branch the computation to try two approaches in parallel
- **Intervene** — inspect children between steps, kill bad branches, inject hints
- **Serialize** — `state.model_dump_json()` captures the entire tree; load on a different machine
- **Gym-style loop** — wrap `step()` for RL training with `(state, reward, done)` tuples
- **Time travel** — keep a history of states, rewind to any point

## `RLMState`

Frozen, recursive Pydantic model — the entire computation in one object:

```python
state.agent_id    # "root", "root.search_0", "root.search_0.chunk_2"
state.task        # the task string
state.status      # WAITING | HAS_REPLY | SUPERVISING | FINISHED
state.iteration   # current step count
state.event       # last StepEvent — LLMReply, CodeExec, ChildStep, or NoCodeBlock
state.messages    # full LLM message history
state.result      # final result (when finished)
state.children    # list[RLMState] — recursive
state.finished    # shorthand for status == FINISHED
```

## Core API

### `RLM`

```python
agent = RLM(
    llm_client=llm,            # LLMClient — or use OpenAIClient / AnthropicClient
    runtime=runtime,           # Runtime (LocalRuntime for in-process exec)
    config=RLMConfig(
        max_depth=5,           # recursion limit
        max_iterations=30,     # steps per agent
        max_concurrency=8,     # global parallel cap
        session="context",     # session persistence (str path, Session object, or None)
    ),
    pool=ThreadPool(8),        # execution pool (ThreadPool, SequentialPool, or custom)
    llm_clients={...},         # named model registry for delegate(model="fast")
)

state = agent.start("task")
state = agent.step(state)      # one transition
result = agent.run("task")     # run to completion
```

Override any method: `step`, `step_llm`, `step_exec`, `step_supervise`, `build_system_prompt`, `build_messages`, `create_child`.

### `LLMClient`

```python
from rlmkit.llm import OpenAIClient, AnthropicClient, LLMClient

llm = OpenAIClient("gpt-5")                    # lazy import — no hard dependency
llm = AnthropicClient("claude-sonnet-4-20250514")

class MyLLM(LLMClient):                        # or roll your own
    def chat(self, messages): ...
```

### `Runtime`

```python
from rlmkit.runtime.local import LocalRuntime

runtime = LocalRuntime(workspace=".")
# Builtins: read_file, write_file, edit_file, append_file, ls, grep
# Common modules pre-imported: re, os, json, math, etc.
```

Register custom tools:

```python
@runtime.tool("Search for a regex pattern across files.")
def search(pattern: str, path: str = ".") -> str:
    ...
```

### Sessions

Persist agent message histories so agents can read their own or each other's past:

```python
agent = RLM(..., config=RLMConfig(session="context/"))
# Tools: list_sessions(), read_history(agent_id=None, last_n=20)
```

Custom backends — subclass `Session` with `write`, `read`, `list_agents`, `exists`.

### Model Selection

```python
agent = RLM(
    llm_client=OpenAIClient("gpt-5"),
    llm_clients={
        "fast": {"model": OpenAIClient("gpt-5-mini"), "description": "Cheap, for simple tasks"},
    },
    ...
)
# Agent sees available models in its prompt and can: delegate("search", task, model="fast")
```

## Extending

```python
class SecurityAuditor(RLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_builder = (
            make_default_builder()
            .section("role", "You are a security auditor.", title="Role")
            .remove("examples")
        )

class ReviewState(RLMState):
    findings: list[str] = []

class CodeReviewer(RLM):
    state_cls = ReviewState

    def step_exec(self, state):
        new_state = super().step_exec(state)
        if isinstance(new_state.event, CodeExec) and "issue" in new_state.event.output.lower():
            return new_state.update(findings=state.findings + [new_state.event.output])
        return new_state
```

## Examples

All examples log a full step-by-step trace. Run with `--viz` for a live terminal UI.

| Example | What it shows |
|---------|--------------|
| [`agent.py`](examples/agent.py) | Interactive coding agent REPL — give it tasks, it writes and edits files |
| [`summarizer.py`](examples/summarizer.py) | Recursive map-reduce: chunk a 10k-line doc, summarize in parallel, combine results |
| [`basic.py`](examples/basic.py) | Parallel chunked search over 1M lines — the "hello world" of recursive agents |
| [`needle_haystack.py`](examples/needle_haystack.py) | 500-file search with `runtime_factory` — each child gets its own sandboxed runtime |
| [`custom_agent.py`](examples/custom_agent.py) | Subclassed RLM with custom state: a code reviewer that accumulates findings across steps |

## Project Structure

```
rlmkit/
├── rlm.py           # RLM engine, RLMConfig, step logic
├── state.py         # RLMState, Status, StepEvent hierarchy
├── pool.py          # Pool ABC, ThreadPool, SequentialPool
├── session.py       # Session ABC, FileSession
├── llm.py           # LLMClient ABC, OpenAIClient, AnthropicClient
├── utils.py         # @tool decorator, code block parsing
├── runtime/
│   ├── runtime.py   # Runtime ABC, ToolDef, builtins
│   ├── local.py     # LocalRuntime (in-process via Sandbox)
│   ├── sandbox.py   # Sandbox (code execution + JSON-over-stdio)
│   └── modal.py     # ModalRuntime (remote via Modal containers)
└── prompts/
    ├── builder.py   # PromptBuilder, Section
    └── default.py   # Default prompt sections
```

## References

- [Recursive Language Models](https://github.com/alexzhang13/rlm) — the original RLM paper
- [rlm-minimal](https://github.com/alexzhang13/rlm-minimal) — minimal single-file RLM reference
- [ypi](https://github.com/rawwerks/ypi) — recursive coding agent built on Pi

## License

See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{sudhakaran2025rlmkit,
  author = {Sudhakaran, Shyam},
  title = {rlmkit},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shyamsn97/rlmkit}},
}
```
