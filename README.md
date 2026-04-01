# rlmkit

A **step-based state machine** for building [Recursive Language Model](https://github.com/alexzhang13/rlm-minimal) agents in ~1,400 lines of Python.

An RLM is an LLM with a REPL environment — it thinks in code, executes it, observes the output, and repeats. When a problem is too big for one context window, it spawns sub-agents (which are themselves RLMs) to handle pieces in parallel. Each sub-agent gets a fresh context window and the same tools — like forking a process, but for language models.

`rlmkit` makes this loop explicit. Every agent is a pure state machine: `state = agent.step(state)`. You drive the loop, you see every transition, you control the flow. The full computation — root agent, children, grandchildren — is a single immutable state tree.

```
pip install rlmkit
```

## The Idea

An RLM is analogous to a process in an operating system:

| Concept | OS Process | RLM Agent |
|---------|-----------|-----------|
| Execution | CPU runs instructions | LLM generates code, REPL executes it |
| Memory | Address space | Context window (finite, non-renewable) |
| Forking | `fork()` → child process | `delegate()` → child agent with fresh context |
| IPC | pipes, shared memory | `wait_all()` returns child results |
| Scheduling | OS scheduler | `step_children()` runs children in parallel |
| State | PCB (process control block) | `RLMState` (immutable, serializable) |

Just as an OS doesn't run one giant process — it decomposes work into many processes — an RLM decomposes work into many agents. The reason is the same: **bounded resources**. A single context window can't hold a million lines of text, just like a single process can't hold infinite memory. You fork, distribute, and combine.

`rlmkit` makes every step explicit: `state = agent.step(state)`. You see every LLM call, every code execution, every child spawn. Inspect, intervene, or terminate between any two steps. The state is immutable and recursive — the entire computation tree is a single serializable snapshot you can pause, resume, fork, or replay.

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
    config=RLMConfig(max_depth=3, max_iterations=15, context="context.md"),
)

# You drive the loop. Every step is visible.
state = agent.start("Find and fix all type errors in src/")
while not state.finished:
    state = agent.step(state)
    print(state.event)

print(state.result)
```

Or run to completion:

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
- `WAITING` — ready for an LLM call (like a process waiting for CPU)
- `HAS_REPLY` — LLM responded, code block ready to execute
- `SUPERVISING` — exec suspended mid-block, waiting on child agents (like a parent waiting on `waitpid`)
- `FINISHED` — `done()` was called, result available

**Key mechanism:** when REPL code calls `delegate()` + `wait_all()`, the exec thread **suspends mid-block**. The stepper takes over, advancing all children in parallel via `step_children()`. Once children finish, exec resumes exactly where it left off — same stack, same locals, same line.

## `RLMState`

The state is a frozen, immutable, recursive Pydantic model. It captures the entire computation:

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

The state tree is recursive. `state.children[0].children[1]` gives you the grandchild's full state — its messages, events, result, everything. Serialize the root state and you've captured the entire multi-agent computation tree.

## StepEvents

Every `step()` attaches a typed event to the returned state:

| Event | Key Fields | Emitted When |
|-------|-----------|------|
| `LLMReply` | `text`, `code` | LLM responded |
| `CodeExec` | `code`, `output`, `suspended` | REPL block executed |
| `ChildStep` | `child_events[]`, `all_done`, `exec_output` | Children were stepped |
| `NoCodeBlock` | `text` | LLM forgot the ```repl``` block |

`ChildStep.child_events` is recursive — if a child was itself supervising grandchildren, you get the full event tree. `exec_output` contains the resumed execution output after children complete.

## Core API

### `RLMConfig`

```python
RLMConfig(
    max_depth=5,               # recursion limit
    max_iterations=30,         # loops per agent
    max_output_length=12_000,  # truncate REPL output
    max_messages=None,         # keep last N messages (None = all)
    max_concurrent_children=8, # parallel child execution cap
    child_max_iterations=None, # override for child iteration limit
    single_block=True,         # only execute first ```repl``` block
    context=None,              # durable scratchpad: str path, Context object, or None
    system_prompt=None,        # raw override (skips default builder)
)
```

### `RLM`

The agent engine. Public attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `children` | `dict[str, RLM]` | Child engines, keyed by agent ID |
| `result` | `str \| None` | What `done()` was called with |
| `is_done` | `bool` | Whether `done()` has been called |
| `waiting_on` | `list[str]` | Agent IDs this engine is waiting on |
| `last_state` | `RLMState \| None` | The initial state from `start()` |

Subclass and override any method:

| Method | What it does |
|--------|-------------|
| `step(state)` | Dispatch to step_llm / step_exec / step_children |
| `step_llm(state)` | Call the LLM → HAS_REPLY |
| `step_exec(state)` | Execute code block → WAITING / SUPERVISING / FINISHED |
| `step_children(state)` | Step all active children in parallel |
| `execute_child_steps(active)` | Run child steps (override for custom executor) |
| `build_system_prompt(state)` | Return the system prompt string |
| `build_messages(state)` | Assemble the LLM message list |
| `make_state(**fields)` | Construct initial state (override for custom state classes) |
| `create_child(agent_id, task)` | Full control over child construction |
| `on_child_stepped(aid, state, n, total)` | Callback after each child completes a step |

### `LLMClient`

Implement `chat()`. Optionally override `stream()`:

```python
class LLMClient(ABC):
    def chat(self, messages: list[dict[str, str]]) -> str: ...
    def stream(self, messages) -> Iterator[str]:
        yield self.chat(messages)
```

### `Runtime`

The execution environment — analogous to a process's address space:

```python
class Runtime(ABC):
    def execute(self, code: str, timeout=None) -> str: ...
    def inject(self, name: str, value: Any) -> None: ...
    def clone(self) -> Runtime: ...  # fresh namespace, same tools
```

`LocalRuntime` runs code via `exec()` with a persistent namespace. Builtin tools: `read_file`, `write_file`, `edit_file`, `append_file`, `ls`, `grep`. Common modules (`re`, `os`, `json`, `math`, etc.) are pre-imported.

Child agents get a **cloned runtime** — same workspace and tools, isolated namespace. Like `fork()` — shared filesystem, separate memory.

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

### `PromptBuilder`

The default system prompt is assembled from named sections (`identity`, `recursion`, `examples`, `tools`, `guardrails`, `status`). Override or extend any section:

```python
from rlmkit.prompts.default import make_default_builder

builder = make_default_builder()
builder.update({"identity": "You are a code reviewer. Be thorough."})
```

Or bypass it entirely with `RLMConfig(system_prompt="...")`.

## Subclassing

The main extension point. Override methods to customize behavior, set `state_cls` for custom state.

### Custom state + custom prompt

```python
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.state import RLMState, CodeExec

class ReviewState(RLMState):
    findings: list[str] = []

class CodeReviewer(RLM):
    state_cls = ReviewState

    def make_state(self, **fields) -> ReviewState:
        return ReviewState(**fields, findings=[])

    def build_system_prompt(self, state: RLMState) -> str:
        return "You are a code reviewer. Focus on bugs and type safety. ..."

    def step_exec(self, state: ReviewState) -> ReviewState:
        new_state = super().step_exec(state)
        if isinstance(new_state.event, CodeExec) and "issue" in new_state.event.output.lower():
            return new_state.update(findings=state.findings + [new_state.event.output])
        return new_state
```

### Custom child execution (e.g. process pool)

```python
class RemoteRLM(RLM):
    def execute_child_steps(self, active):
        """Send child steps to a remote worker pool instead of local threads."""
        return dispatch_to_cluster(active)

    def create_child(self, agent_id, task, *, max_iterations=None):
        """Give children a sandboxed runtime."""
        child = super().create_child(agent_id, task, max_iterations=max_iterations)
        child.runtime = SandboxedRuntime(...)
        return child
```

### Navigating the engine tree

After running, the engine tree is fully inspectable:

```python
agent = RLM(llm_client=llm, runtime=runtime, config=config)
state = agent.start("Find the magic number in haystack.txt")

while not state.finished:
    state = agent.step(state)

# Engine tree (mutable — who did what)
for cid, child in agent.children.items():
    print(f"{cid}: result={child.result}, is_done={child.is_done}")
    for gcid, grandchild in child.children.items():
        print(f"  {gcid}: result={grandchild.result}")

# State tree (immutable — full computation history)
for cs in state.children:
    print(f"{cs.agent_id}: {cs.result}, {len(cs.messages)} messages")
```

The engine tree and state tree are parallel structures. The engine holds mutable resources (LLM client, runtime, threads). The state holds the immutable computation record (messages, events, results). Both are recursive.

### Mid-run intervention

Because you own the step loop, you can inspect children between steps and terminate them manually — override their result, skip bad branches, or inject corrections. This is hard to do in frameworks where the agent loop is a black box.

```python
from rlmkit.state import Status

state = agent.start("Analyze all log files")
while not state.finished:
    state = agent.step(state)

    # Check if a child is going off the rails
    for cs in state.children:
        if not cs.finished and "hallucinated" in (cs.last_reply or ""):
            # Kill it and inject a custom result
            agent.children[cs.agent_id].is_done = True
            agent.children[cs.agent_id].result = "Skipped: bad output detected"
            state = state.update(children=[
                c.update(status=Status.FINISHED, result="Skipped: bad output detected")
                if c.agent_id == cs.agent_id else c
                for c in state.children
            ])
```

Works at any depth — you can reach into `agent.children["root.1"].children["root.1.3"]` and terminate a grandchild the same way. The next `step()` call will see it as finished and move on.

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

~1,400 lines of core code.

## References

- [Recursive Language Models](https://github.com/alexzhang13/rlm) — the original RLM paper and implementation
- [rlm-minimal](https://github.com/alexzhang13/rlm-minimal) — minimal single-file RLM reference implementation
- [ypi](https://github.com/alexzhang13/ypi) — RLM-based coding agent built on rlm-minimal

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
