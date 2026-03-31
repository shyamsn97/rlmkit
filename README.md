# rlmkit

A **state-machine framework** for recursive LLM agents in ~1,200 lines of Python.

Every agent is a pure state machine: `state = agent.step(state)`. The entire computation — the agent, its children, their children — is a single immutable tree you can inspect, serialize, replay, or fork at any point. No hidden control flow. No callbacks. No magic.

```
pip install rlmkit
```

## Why

Most agent frameworks are black boxes. You call `.run()`, something happens inside, and you get a result. If it goes wrong, good luck debugging.

`rlmkit` inverts this. The agent **produces states**, you **drive the loop**. Every LLM call, every code execution, every child delegation is an explicit state transition you control. This means:

- **Full observability** — every step emits a typed event (`LLMReply`, `CodeExec`, `ChildStep`, ...)
- **Deterministic replay** — states are frozen Pydantic models, trivially serializable
- **Custom control flow** — pause, skip, inject, branch — it's just a while loop
- **Recursive by design** — sub-agents get fresh context windows and produce their own state trees, nested inside the parent

The recursive pattern is simple: **size up → search → delegate → combine**. Each sub-agent gets a fresh context budget, the same tools, and returns a compact result.

## Quick Start

```python
from rlmkit.llm import LLMClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime

class MyLLM(LLMClient):
    def chat(self, messages):
        ...  # plug in any provider

runtime = LocalRuntime(workspace=".")
agent = RLM(
    llm_client=MyLLM(),
    runtime=runtime,
    config=RLMConfig(max_depth=3, max_iterations=15),
)

# You drive the loop. Every step is visible.
state = agent.start("Find and fix all type errors in src/")
while not state.finished:
    state = agent.step(state)
    print(state.event)

print(state.result)
```

Or run to completion if you don't need step-level control:

```python
result = agent.run("Find and fix all type errors in src/")
```

## The State Machine

Each `step(state) → state` is a single atomic transition. The agent cycles through four statuses:

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
                WAITING ──── step_llm() ────→ HAS_REPLY      │
                  ▲  ▲                           │            │
                  │  │                     step_exec()        │
                  │  │                      ╱        ╲        │
                  │  └── no children ──────╴          ╲       │
                  │                                    ▼      │
                  │  done() ───────────────────→ FINISHED     │
                  │                                           │
                  │                              SUPERVISING ─┘
                  │                                  │
                  │                          step_children()
                  │                          (all children
                  └── children done ──────── stepped in parallel)
```

**States:**
- `WAITING` — ready for an LLM call
- `HAS_REPLY` — LLM responded, code block ready to execute
- `SUPERVISING` — exec suspended mid-block, waiting on child agents
- `FINISHED` — `done()` was called, result available

**Key mechanism:** when REPL code calls `delegate()` + `wait_all()`, the exec thread **suspends mid-block**. The stepper takes over, advancing all children in parallel via `step_children()`. Once children finish, exec resumes exactly where it left off — same stack, same locals, same line.

## `RLMState` — The Whole Truth

The state is a frozen, immutable, recursive Pydantic model. It IS the computation:

```python
state.agent_id    # "root", "root.1", "root.1.2", ...
state.status      # WAITING | HAS_REPLY | SUPERVISING | FINISHED
state.iteration   # current iteration count
state.event       # last StepEvent — what just happened
state.messages    # full LLM message history
state.result      # final result string (when finished)
state.children    # list[RLMState] — the full recursive tree
state.config      # dict of config + runtime info
state.context     # context file contents (if configured)
state.finished    # shorthand for status == FINISHED

new_state = state.update(iteration=5)  # immutable update
```

The state tree is recursive. `state.children[0].children[1]` gives you the grandchild's full state — its messages, events, result, everything. Serialize the root state and you've captured the entire multi-agent computation.

## StepEvents

Every `step()` attaches a typed event to the returned state:

| Event | Key Fields | Emitted When |
|-------|-----------|------|
| `LLMReply` | `text`, `code` | LLM responded |
| `CodeExec` | `code`, `output`, `suspended` | REPL block executed |
| `ChildStep` | `child_events[]`, `all_done` | Children were stepped (recursive — child events contain their own events) |
| `NoCodeBlock` | `text` | LLM forgot the ```repl``` block |

`ChildStep.child_events` is recursive — if a child was itself supervising grandchildren, you get the full event tree.

## Core API

### `RLMConfig`

```python
RLMConfig(
    max_depth=5,               # recursion limit
    max_iterations=30,         # loops per agent
    max_output_length=12_000,  # truncate REPL output
    single_block=True,         # only execute first ```repl``` block
    context_path=None,         # durable scratchpad file
    system_prompt=None,        # raw override (skips default builder)
)
```

### `RLM`

The agent engine. Subclass and override any step:

| Method | What it does |
|--------|-------------|
| `step(state)` | Dispatch to step_llm / step_exec / step_children |
| `step_llm(state)` | Call the LLM → HAS_REPLY |
| `step_exec(state)` | Execute code block → WAITING / SUPERVISING / FINISHED |
| `step_children(state)` | Step all active children in parallel |
| `build_system_prompt(state)` | Return the system prompt string |
| `build_messages(state)` | Assemble the LLM message list |
| `make_state(**fields)` | Construct initial state (override for custom state classes) |
| `create_child(agent_id, task)` | Full control over child construction |

### `LLMClient`

Implement `chat()`. Optionally override `stream()`:

```python
class LLMClient(ABC):
    def chat(self, messages: list[dict[str, str]]) -> str: ...
    def stream(self, messages) -> Iterator[str]:
        yield self.chat(messages)
```

### `Runtime`

Execution environment — two abstract methods:

```python
class Runtime(ABC):
    def execute(self, code: str, timeout=None) -> str: ...
    def inject(self, name: str, value: Any) -> None: ...
    def clone(self) -> Runtime: ...  # fresh namespace, same tools
```

`LocalRuntime` runs code via `exec()` with a persistent namespace. Builtin tools: `read_file`, `write_file`, `edit_file`, `append_file`, `ls`, `grep`.

Child agents get a **cloned runtime** — same workspace and tools, isolated namespace.

### Custom Tools

```python
@runtime.tool("Search for a regex pattern across files.")
def search(pattern: str, path: str = ".") -> str:
    ...

# or register manually
from rlmkit.utils import tool

@tool("Get the current timestamp.")
def now() -> str:
    return datetime.now(timezone.utc).isoformat()

runtime.register_tool(now)
```

## Examples

All examples save a full step-by-step trace to `examples/*_log.md`.

| Example | What it shows |
|---------|--------------|
| `basic.py` | 1M-line needle search with parallel sub-agent delegation |
| `needle_haystack.py` | 500-file search with `runtime_factory` for child isolation |
| `custom_agent.py` | Subclassed `RLM` + `RLMState` for a code reviewer with custom state |

## Project Structure

```
rlmkit/
├── rlm.py          # RLM engine, RLMConfig, ExecThread
├── state.py         # RLMState, Status, StepEvent hierarchy, ChildHandle
├── llm.py           # LLMClient ABC
├── utils.py         # @tool decorator, code block parsing
├── runtime/
│   ├── runtime.py   # Runtime ABC, ToolDef, builtins
│   └── local.py     # LocalRuntime (exec-based)
└── prompts/
    ├── builder.py   # PromptBuilder, Section
    └── default.py   # Default prompt sections
```

~1,200 lines of core code.

## License

See [LICENSE](LICENSE).
