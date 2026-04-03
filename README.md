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
| Forking | `fork()` → child process | `delegate(name, task)` → named child agent with fresh context |
| IPC | pipes, shared memory | `yield wait()` returns child results |
| Scheduling | OS scheduler | Global pool steps all leaves in parallel |
| State | PCB (process control block) | `RLMState` (immutable, serializable) |

Just as an OS doesn't run one giant process — it decomposes work into many processes — an RLM decomposes work into many agents. The reason is the same: **bounded resources**. A single context window can't hold a million lines of text, just like a single process can't hold infinite memory. You fork, distribute, and combine.

`rlmkit` makes every step explicit: `state = agent.step(state)`. You see every LLM call, every code execution, every child spawn. Inspect, intervene, or terminate between any two steps. The state is immutable and recursive — the entire computation tree is a single serializable snapshot you can pause, resume, fork, or replay.

## Quick Start

```python
from rlmkit.llm import OpenAIClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime

runtime = LocalRuntime(workspace=".")
agent = RLM(
    llm_client=OpenAIClient("gpt-5"),
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

**States:**
- `WAITING` — ready for an LLM call (like a process waiting for CPU)
- `HAS_REPLY` — LLM responded, code block ready to execute
- `SUPERVISING` — exec suspended mid-block, waiting on child agents (like a parent waiting on `waitpid`)
- `FINISHED` — `done()` was called, result available

**Key mechanism:** when REPL code calls `delegate()` + `yield wait()`, the code generator **suspends at the yield point**. The engine flattens the entire agent tree, sorts deepest-first, and steps all leaves in parallel via a global pool. When children finish, parent generators resume automatically in the same pass — no nested threads, no wasted orchestration slots. One pool, one concurrency cap, regardless of tree depth.

## Architecture

The engine is a stateless stepper. The state is the single source of truth. A global pool handles all parallelism.

```
┌───────────────────────────────────────────────────────┐
│                       RLMState                        │
│           (immutable, serializable, recursive)        │
│                                                       │
│  task, status, messages, events, result, waiting_on,  │
│  context, config                                      │
│                                                       │
│  children: [RLMState, RLMState, ...]                  │
│             └── full recursive state tree             │
└───────────────────────────┬───────────────────────────┘
                            │
                  step(state) → new_state
                            │
┌───────────────────────────┴───────────────────────────┐
│                      RLM Engine                       │
│                  (stateless stepper)                   │
│                                                       │
│  immutable infra: llm_client, runtime, config, pool   │
│  ephemeral working memory: is_done, result            │
│    (set during a step, flushed to state, gone)        │
│                                                       │
│           new_state = f(state, infra)                 │
└───────────────────────────────────────────────────────┘
```

Between steps, the engine holds no meaningful computation state — everything is in `RLMState`. This means you can serialize, fork, rewind, or hand off a state to a completely different engine:

```python
# ── serialize at any step boundary ────────────────────
snapshot = state.model_dump_json()
state = RLMState.model_validate_json(snapshot)

# ── fork the computation ──────────────────────────────
branch_a = state.update(task="fix with approach A", status=Status.WAITING)
branch_b = state.update(task="fix with approach B", status=Status.WAITING)

# ── rewind to a checkpoint ────────────────────────────
history = []
while not state.finished:
    history.append(state)
    state = engine.step(state)
state = engine.step(history[3])  # back to step 3

# ── hand off to a different engine ────────────────────
engine2 = RLM(llm_client=other_llm, runtime=other_runtime)
while not state.finished:
    state = engine2.step(state)
```

### The Gym Analogy

The step API is structurally identical to OpenAI Gym's `reset → step → done` loop:

```python
# Gym                                    # rlmkit
env = gym.make("CartPole-v1")            agent = RLM(llm_client=..., runtime=...)
state, info = env.reset()                state = agent.start("Find the bug")
while not done:                          while not state.finished:
    action = policy(state)                   state = agent.step(state)
    state, reward, done, _, info = \
        env.step(action)
```

| Gym | rlmkit | |
|-----|--------|-|
| `env.reset()` | `agent.start(task)` | Returns initial state |
| `env.step(action)` | `agent.step(state)` | Returns next state |
| `state` | `RLMState` | Full observable snapshot |
| `done` | `state.finished` | Terminal flag |
| `reward` / `info` | `state.result` / `state.event` | Outcome + metadata |

In vanilla rlmkit, the LLM **is** the policy — it picks the action (code to run) internally. But because the state is immutable and `step()` is pure, you can **inject actions** between steps:

```python
while not state.finished:
    if stuck(state):
        state = state.update(context=state.context + "\nHint: try a different approach")
    state = agent.step(state)
```

The state you pass back *is* the action space — inject context, override the task, fork, rewind, or swap the engine entirely. A Gym-compatible wrapper is trivial:

```python
class RLMEnv:
    def __init__(self, agent): self.agent = agent

    def reset(self, task):
        self.state = self.agent.start(task)
        return self.state

    def step(self, action=None):
        if action:
            self.state = self.state.update(**action)
        self.state = self.agent.step(self.state)
        return self.state, self._reward(self.state), self.state.finished, {}
```

This makes rlmkit a natural fit for RL-based agent orchestration (train a meta-controller that decides when to hint, kill, or fork), evaluation harnesses, and human-in-the-loop workflows.

### Parallelism

When an agent delegates to children, the engine doesn't create nested thread pools per parent. Instead, it flattens the entire agent tree, sorts by depth (leaves first), and steps everything through a single global pool:

```
root (SUPERVISING)
├── child_A (SUPERVISING)
│   ├── gc_1 (WAITING)     ← step in pool
│   ├── gc_2 (WAITING)     ← step in pool
│   └── gc_3 (WAITING)     ← step in pool
├── child_B (WAITING)      ← step in pool
└── child_C (FINISHED)     ← skip
```

All 4 active leaves go into the pool. SUPERVISING parents don't take a thread — they're resolved for free when their children finish. With `max_concurrency=8`, at most 8 things run at a time across the **entire tree**, regardless of depth.

```python
from rlmkit.pool import ThreadPool, SequentialPool

agent = RLM(
    llm_client=llm,
    runtime=runtime,
    pool=ThreadPool(max_concurrency=8),   # default
)

# For debugging — run everything sequentially
agent = RLM(llm_client=llm, runtime=runtime, pool=SequentialPool())

# Or pass any function
agent = RLM(llm_client=llm, runtime=runtime, pool=my_custom_pool_fn)
```

The pool is shared across the entire tree — root and all descendants use the same pool instance. One pool, one concurrency cap.

## `RLMState`

The state is a frozen, immutable, recursive Pydantic model. It captures the entire computation:

```python
state.agent_id    # "root", "root.1", "root.1.2", ...
state.task        # the task string this agent is working on
state.status      # WAITING | HAS_REPLY | SUPERVISING | FINISHED
state.iteration   # current iteration count
state.event       # last StepEvent — what just happened
state.messages    # full LLM message history
state.result      # final result string (when finished)
state.waiting_on  # agent IDs currently waiting on (during SUPERVISING)
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
    max_concurrency=8,         # global parallel execution cap
    child_max_iterations=None, # override for child iteration limit
    single_block=True,         # only execute first ```repl``` block
    context=None,              # durable scratchpad: str path, Context object, or None
    system_prompt=None,        # raw override (skips default builder)
)
```

### `RLM`

The agent engine. Mostly stateless — ephemeral working memory is flushed to `RLMState` at each step boundary.

```python
agent = RLM(
    llm_client=llm,            # required — LLMClient instance
    runtime=runtime,           # required — Runtime instance
    config=RLMConfig(),        # optional — tuning knobs
    pool=ThreadPool(8),        # optional — execution pool (default: ThreadPool)
    runtime_factory=None,      # optional — factory for child runtimes
    llm_clients=None,          # optional — named model registry
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `children` | `dict[str, RLM]` | Child engine instances (infrastructure, not state) |
| `pool` | `Pool` | Shared execution pool for parallel stepping |
| `result` | `str \| None` | Ephemeral — set by `done()` during a step, flushed to state |
| `is_done` | `bool` | Ephemeral — set by `done()` during a step, flushed to state |
| `last_state` | `RLMState \| None` | The initial state from `start()` |
| `llm_clients` | `dict[str, LLMClient]` | Named model registry (always includes `"default"`) |

Subclass and override any method:

| Method | What it does |
|--------|-------------|
| `step(state)` | Dispatch to step_llm / step_exec / step_supervise |
| `step_llm(state)` | Call the LLM → HAS_REPLY |
| `step_exec(state)` | Execute code block → WAITING / SUPERVISING / FINISHED |
| `step_supervise(state)` | Flatten tree, step leaves in pool, cascade parent resumes |
| `flatten_all(state, depth)` | Collect all non-finished nodes as `(depth, state, engine)` |
| `apply_results(state, stepped)` | Bottom-up tree rebuild with parent resume cascading |
| `build_system_prompt(state)` | Return the system prompt string |
| `build_messages(state)` | Assemble the LLM message list |
| `make_state(**fields)` | Construct initial state (override for custom state classes) |
| `create_child(agent_id, *, max_iterations, llm_client)` | Full control over child construction |

### `LLMClient`

Built-in clients for OpenAI and Anthropic (imports are lazy — neither package is required unless you use it):

```python
from rlmkit.llm import OpenAIClient, AnthropicClient

llm = OpenAIClient("gpt-5")                   # or any OpenAI-compatible API
llm = AnthropicClient("claude-sonnet-4-20250514")
```

Or implement your own — just subclass `LLMClient` and implement `chat()`:

```python
from rlmkit.llm import LLMClient

class MyLLM(LLMClient):
    def chat(self, messages: list[dict[str, str]]) -> str: ...
    def stream(self, messages) -> Iterator[str]:  # optional
        yield self.chat(messages)
```

### `Runtime`

The execution environment — analogous to a process's address space:

```python
class Runtime(ABC):
    def execute(self, code: str, timeout=None) -> str: ...
    def inject(self, name: str, value: Any) -> None: ...
    def clone(self) -> Runtime: ...       # fresh namespace, same tools
    def start_code(self, code) -> tuple[bool, object]: ...   # generator execution
    def resume_code(self, send_value=None) -> tuple[bool, object]: ...
```

`start_code`/`resume_code` drive the generator-based execution that enables `yield wait()` suspension. Both `LocalRuntime` and `ModalRuntime` delegate to `Sandbox` — `LocalRuntime` uses it in-process, `ModalRuntime` communicates with it over JSON-over-stdio in a remote container.

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

### Context

Agents can have a **durable scratchpad** that persists across REPL turns. Pass a string path to auto-create a file-backed context, or implement your own `Context` subclass for other backends (database, in-memory, etc.):

```python
# String path → auto-creates FileContext
agent = RLM(llm_client=llm, runtime=runtime, config=RLMConfig(context="context.md"))

# Or provide a Context object directly
from rlmkit.context import FileContext, Context
agent = RLM(llm_client=llm, runtime=runtime, config=RLMConfig(context=FileContext("ctx.md", runtime)))
```

When context is configured, two tools are registered into the REPL:
- `read_context()` — returns the full context string
- `append_context(text)` — appends text to the context

Children get an **isolated clone** via `Context.clone(agent_id)`. For `FileContext`, this creates a sibling file under `{parent_dir}/{agent_id}/{filename}`.

To implement a custom context backend, subclass `Context` and implement `read()`, `append()`, `write()`, and `clone()`:

```python
from rlmkit.context import Context

class RedisContext(Context):
    def read(self) -> str: ...
    def append(self, text: str) -> None: ...
    def write(self, text: str) -> None: ...
    def clone(self, agent_id: str) -> RedisContext: ...
```

### Model Selection

Pass multiple models via `llm_clients` and let the agent choose per-delegation:

```python
agent = RLM(
    llm_client=OpenAIClient("gpt-5"),
    runtime=runtime,
    config=config,
    llm_clients={
        "fast": {"model": OpenAIClient("gpt-5-mini"), "description": "Cheap model for simple tasks"},
    },
)
```

The default model comes from `llm_client` and is always available as `"default"`. The agent sees available models in its system prompt and can select one per delegation:

```python
delegate("search", "Find the needle", model="fast")
```

### `PromptBuilder`

The system prompt is assembled from an ordered list of named sections. The default builder ships with: `role`, `repl`, `recursion`, `examples`, `tools`, `guardrails`, `status`.

Static sections (role, repl, recursion, examples, guardrails) are baked into the builder at construction. Dynamic sections (tools, status) are empty placeholders — the engine fills them at render time via keyword overrides to `build()`:

```python
from rlmkit.prompts import PromptBuilder, make_default_builder

# The default builder
builder = make_default_builder()
builder.names  # ['role', 'repl', 'recursion', 'examples', 'tools', 'guardrails', 'status']

# Render — dynamic sections passed as keyword args
prompt = builder.build(
    tools="- read_file(path): Read a file.\n- grep(pattern, path): Search.",
    status="You are at recursion depth 1 of 3.",
)
```

The builder is immutable during `build()` — overrides apply to that single render only.

**Customizing the builder:**

```python
builder = make_default_builder()

# Replace an existing section
builder = builder.section("role", "You are a security auditor. Find vulnerabilities.", title="Role")

# Add a new section after an existing one
builder = builder.section("methodology", "Always check OWASP Top 10.", title="Methodology", after="recursion")

# Remove a section
builder = builder.remove("examples")

# Insert before a section
builder = builder.section("constraints", "Only review .py files.", title="Constraints", before="guardrails")

# Build from scratch
builder = (
    PromptBuilder()
    .section("role", "You are a data analyst.", title="Role")
    .section("repl", REPL_TEXT, title="REPL")
    .section("tools", title="Tools")
)
```

The three core reusable sections — `role`, `repl`, `recursion` — are exported as constants from `rlmkit.prompts.default` (`ROLE_TEXT`, `REPL_TEXT`, `RECURSION_TEXT`). Most custom agents will keep `repl` and `recursion` as-is and just swap `role`.

Or bypass the builder entirely with `RLMConfig(system_prompt="...")`.

## Subclassing

Subclassing `RLM` is the main extension point. There are several reasons you'd subclass:

### Custom prompt via builder

The most common case — change what the agent is told to do. Override the role, add domain-specific sections, remove examples.

```python
from rlmkit.rlm import RLM
from rlmkit.prompts import make_default_builder
from rlmkit.prompts.default import ROLE_TEXT

class SecurityAuditor(RLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_builder = (
            make_default_builder()
            .section("role", f"You are a security auditor.\n\n{ROLE_TEXT}", title="Role")
            .section("methodology", METHODOLOGY, title="Methodology", after="recursion")
            .remove("examples")
        )
```

### Custom state

Track domain-specific data across steps. State is immutable — `update()` returns a new instance.

```python
from rlmkit.state import RLMState, CodeExec

class ReviewState(RLMState):
    findings: list[str] = []

class CodeReviewer(RLM):
    state_cls = ReviewState

    def make_state(self, **fields) -> ReviewState:
        return ReviewState(**fields, findings=[])

    def step_exec(self, state: ReviewState) -> ReviewState:
        new_state = super().step_exec(state)
        if isinstance(new_state.event, CodeExec) and "issue" in new_state.event.output.lower():
            return new_state.update(findings=state.findings + [new_state.event.output])
        return new_state
```

### Custom tools

Register domain-specific tools that all agents in the tree can use.

```python
from rlmkit.utils import tool

@tool("Query a SQL database. Returns results as CSV.")
def query_db(sql: str) -> str:
    return run_query(sql)

class DataAnalyst(RLM):
    def __init__(self, db_conn, **kwargs):
        super().__init__(**kwargs)
        self.runtime.register_tool(query_db)
```

### Step hooks — logging, metrics, early stopping

Intercept any step phase to add logging, collect metrics, or bail early.

```python
class MonitoredRLM(RLM):
    def step(self, state):
        new_state = super().step(state)
        self.metrics.record(state.agent_id, new_state.event)
        if self.budget_exceeded():
            return new_state.update(status=Status.FINISHED, result="Budget exceeded")
        return new_state
```

### Custom child construction

Control how children are created — different runtimes, different configs, rate limits.

```python
class SandboxedRLM(RLM):
    def create_child(self, agent_id, *, max_iterations=None, llm_client=None):
        child = super().create_child(agent_id, max_iterations=max_iterations, llm_client=llm_client)
        child.runtime = SandboxedRuntime(...)
        child.config.max_iterations = min(child.config.max_iterations, 8)
        return child
```

### Custom pools

The pool controls how agent steps are executed in parallel. Subclass `Pool` for custom scheduling, or pass a plain function:

```python
from rlmkit.pool import Pool

class RateLimitedPool(Pool):
    def __init__(self, max_concurrency=8, requests_per_second=10):
        self.max_concurrency = max_concurrency
        self.limiter = RateLimiter(requests_per_second)

    def execute(self, items):
        results = {}
        for cs, engine in items:
            self.limiter.wait()
            results[cs.agent_id] = engine.step(cs)
        return results

agent = RLM(llm_client=llm, runtime=runtime, pool=RateLimitedPool())
```

Or wrap any function — it gets auto-wrapped in `CallablePool`:

```python
def my_pool(items):
    return {cs.agent_id: engine.step(cs) for cs, engine in items}

agent = RLM(llm_client=llm, runtime=runtime, pool=my_pool)
```

### Navigating the engine tree

After running, the engine tree is fully inspectable:

```python
agent = RLM(llm_client=llm, runtime=runtime, config=config)
state = agent.start("Find the magic number in haystack.txt")

while not state.finished:
    state = agent.step(state)

# Engine tree (infrastructure — child engines)
for cid, child in agent.children.items():
    print(f"{cid}: result={child.result}, is_done={child.is_done}")
    for gcid, grandchild in child.children.items():
        print(f"  {gcid}: result={grandchild.result}")

# State tree (immutable — full computation history)
for cs in state.children:
    print(f"{cs.agent_id}: {cs.result}, {len(cs.messages)} messages")
```

The engine tree and state tree are parallel structures. The engine holds infrastructure (LLM client, runtime). The state holds the immutable computation record (messages, events, results). Both are recursive. Between steps, the state is the single source of truth — the engine's working memory is ephemeral.

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

Works at any depth — you can reach into `agent.children["root.search"].children["root.search.chunk_3"]` and terminate a grandchild the same way. The next `step()` call will see it as finished and move on.

## Examples

All examples save a full step-by-step trace to `examples/*_log.md`.

| Example | What it shows |
|---------|--------------|
| `basic.py` | 1M-line needle search with parallel sub-agent delegation |
| `needle_haystack.py` | 500-file search with `runtime_factory` for child isolation |
| `custom_agent.py` | Subclassed `RLM` + `RLMState` for a code reviewer with custom state |
| `summarizer.py` | Recursive summarization of a long document with custom prompt builder |

## Project Structure

```
rlmkit/
├── rlm.py           # RLM engine, RLMConfig, step logic
├── state.py         # RLMState, Status, StepEvent hierarchy, ChildHandle, WaitRequest
├── pool.py          # Pool ABC, ThreadPool, SequentialPool, CallablePool
├── context.py       # Context ABC, FileContext
├── llm.py           # LLMClient ABC, OpenAIClient, AnthropicClient
├── utils.py         # @tool decorator, code block parsing
├── runtime/
│   ├── runtime.py   # Runtime ABC, ToolDef, builtins
│   ├── local.py     # LocalRuntime (in-process execution via Sandbox)
│   ├── sandbox.py   # Sandbox (code execution + JSON-over-stdio for remote runtimes)
│   └── modal.py     # ModalRuntime (remote execution via Modal containers)
└── prompts/
    ├── builder.py   # PromptBuilder, Section
    └── default.py   # Default prompt sections
```

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
