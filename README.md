# rlmkit

A **step-based state machine** for [Recursive Language Model](https://github.com/alexzhang13/rlm-minimal) agents (~2,300 lines of Python).

```
pip install rlmkit
```

## The Problem

A single LLM can't solve hard problems alone. Its context window is finite — it can't hold a million-line codebase, a thousand documents, or a deep multi-step analysis all at once. The answer is **recursion**: the agent spawns sub-agents, each with a fresh context window focused on one piece. Sub-agents spawn their own sub-agents. A tree grows:

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

This is powerful — 11 agents working in parallel, each focused, each with full context on its piece. But in every existing framework, this tree is a **black box**. You call `agent.run()` and wait. You can't see which agents are running, what they're thinking, where they're stuck. You can't pause one branch and kill another. You can't checkpoint the computation and resume it tomorrow. You can't fork the tree to try two approaches. You get `result = agent.run(task)` and hope for the best.

## The Solution

rlmkit makes the recursive agent tree a **state machine**. Every agent — root and all descendants — advances one step at a time. The entire tree is captured in a single immutable object (`RLMState`) at every step boundary:

```
step 1:  root calls LLM → gets code with delegate() calls
step 2:  root executes → suspends at yield wait(), spawns 3 children
step 3:  [scanner_auth, scanner_api, scanner_db] call LLM (parallel)
step 4:  scanner_auth executes → spawns 3 grandchildren
         scanner_api executes → spawns 3 grandchildren
         scanner_db executes → spawns 2 grandchildren
step 5:  all 8 leaf agents call LLM in parallel
step 6:  8 leaves execute — 6 finish, 2 need another round
step 7:  2 remaining leaves call LLM
step 8:  both finish → parents resume → results cascade up
  ...
step 12: root resumes with all results, calls done()
```

At **any** of those step boundaries you can serialize the whole tree, fork it, rewind it, inspect any node, or hand it to a different engine. The tree isn't hidden inside a thread pool — it's data you own.

```python
state = agent.start("Find the security vulnerability in this repo")
while not state.finished:
    state = agent.step(state)
    print(f"step {state.iteration}: {state.event}")
print(state.result)
```

Every `step()` is one atomic transition. Between steps, the full computation — messages, events, results, the entire child tree — is frozen in `RLMState`. Here's what you can do with that:

### Checkpoint and resume

```python
# Save at any step boundary
history = []
while not state.finished:
    state = agent.step(state)
    history.append(state)

# Resume from step 5
state = agent.step(history[5])
```

### Fork the computation

```python
# Try two approaches in parallel
state = agent.step(state)  # agent proposes approach A

branch_a = state
branch_b = state.update(
    messages=state.messages + [{"role": "user", "content": "Try a different approach."}],
    status=Status.WAITING,
)

# Run both to completion
result_a = run_to_completion(agent, branch_a)
result_b = run_to_completion(agent, branch_b)
best = result_a if score(result_a) > score(result_b) else result_b
```

### Intervene mid-run

```python
while not state.finished:
    state = agent.step(state)

    # Kill a child that's going off the rails
    for cs in state.children:
        if not cs.finished and "hallucinated" in (cs.last_reply or ""):
            agent.children[cs.agent_id].is_done = True
            agent.children[cs.agent_id].result = "Skipped"
            state = state.update(children=[
                c.update(status=Status.FINISHED, result="Skipped")
                if c.agent_id == cs.agent_id else c
                for c in state.children
            ])

    # Inject a hint when stuck
    if state.iteration > 10 and not state.children:
        state = state.update(
            messages=state.messages + [{"role": "user", "content": "Hint: try delegating."}],
            status=Status.WAITING,
        )
```

### Serialize the entire computation

```python
# Save to disk
snapshot = state.model_dump_json()
Path("checkpoint.json").write_text(snapshot)

# Load on a different machine, different engine
state = RLMState.model_validate_json(Path("checkpoint.json").read_text())
engine2 = RLM(llm_client=other_llm, runtime=other_runtime)
while not state.finished:
    state = engine2.step(state)
```

### Use it as a Gym environment

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

Or run to completion:

```python
result = agent.run("Find and fix all type errors in src/")
```

## The State Machine

Each `step(state) → state` is a single atomic transition:

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
- `WAITING` — ready for an LLM call
- `HAS_REPLY` — LLM responded, code block ready to execute
- `SUPERVISING` — exec suspended at `yield wait()`, waiting on children
- `FINISHED` — `done()` was called, result available

**Key mechanism:** when REPL code calls `delegate()` + `yield wait()`, the generator **suspends at the yield point**. The engine flattens the entire agent tree, sorts deepest-first, and steps all leaves in parallel via a global pool. When children finish, parent generators resume automatically — no nested threads, no wasted orchestration slots.

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                       RLMState                        │
│           (immutable, serializable, recursive)        │
│                                                       │
│  task, status, messages, events, result, waiting_on,  │
│  config                                               │
│                                                       │
│  children: [RLMState, RLMState, ...]                  │
│             └── full recursive state tree             │
└───────────────────────────┬───────────────────────────┘
                            │
                  step(state) → new_state
                            │
┌───────────────────────────┴───────────────────────────┐
│                      RLM Engine                       │
│                  (stateless stepper)                  │
│                                                       │
│  immutable infra: llm_client, runtime, config, pool   │
│  ephemeral working memory: is_done, result            │
│    (set during a step, flushed to state, gone)        │
│                                                       │
│           new_state = f(state, infra)                 │
└───────────────────────────────────────────────────────┘
```

The engine holds infrastructure (LLM client, runtime, pool). The state holds the computation record (messages, events, results). Between steps, the engine holds no meaningful computation state — everything is in `RLMState`.

**REPL variables are engine-local.** Variables set in code blocks persist across steps within a single run, but are not part of `RLMState`. If you serialize and restore state to a different engine, the REPL namespace starts fresh. The agent can reconstruct working state from its message history via `read_history()`.

### Parallelism

When an agent delegates to children, the engine flattens the entire tree, sorts by depth (leaves first), and steps everything through a single global pool:

```
root (SUPERVISING)
├── child_A (SUPERVISING)
│   ├── gc_1 (WAITING)     ← step in pool
│   ├── gc_2 (WAITING)     ← step in pool
│   └── gc_3 (WAITING)     ← step in pool
├── child_B (WAITING)      ← step in pool
└── child_C (FINISHED)     ← skip
```

All 4 active leaves go into the pool. SUPERVISING parents don't take a thread — they're resolved for free when their children finish. One pool, one concurrency cap, regardless of tree depth.

```python
from rlmkit.pool import ThreadPool, SequentialPool

agent = RLM(llm_client=llm, runtime=runtime, pool=ThreadPool(max_concurrency=8))
agent = RLM(llm_client=llm, runtime=runtime, pool=SequentialPool())  # debugging
agent = RLM(llm_client=llm, runtime=runtime, pool=my_custom_pool_fn)
```

### Delegation and Re-delegation

Sub-agents are spawned with `delegate()` and awaited with `yield wait()`:

```python
h1 = delegate("searcher", "Find all TODOs in src/")
h2 = delegate("searcher", "Find all FIXMEs in src/")  # auto-suffixed: root.searcher_2
results = yield wait(h1, h2)
done(f"Found {len(results)} batches")
```

**Re-delegation:** if you delegate to a name that already finished, the same agent is resumed with a new task. It keeps its REPL variables, session history, and tools — but gets a fresh context window:

```python
h = delegate("analyst", "Read sales.csv and compute Q4 totals")
[q4] = yield wait(h)

# Same agent — still has the data in variables from round 1
h = delegate("analyst", f"Compare Q4={q4} to Q3. Which grew faster?")
[comparison] = yield wait(h)
done(comparison)
```

Rules:
- Child doesn't exist → create new
- Child exists and running → auto-suffix (parallel work)
- Child exists and finished → resume with new task

## `RLMState`

Frozen, immutable, recursive Pydantic model. The entire computation in one object:

```python
state.agent_id    # "root", "root.search_0", "root.search_0.chunk_2"
state.task        # the task string
state.status      # WAITING | HAS_REPLY | SUPERVISING | FINISHED
state.iteration   # current iteration count
state.event       # last StepEvent — what just happened
state.messages    # full LLM message history
state.result      # final result string (when finished)
state.waiting_on  # agent IDs being waited on (during SUPERVISING)
state.children    # list[RLMState] — the full recursive tree
state.config      # dict of config + runtime info
state.finished    # shorthand for status == FINISHED

new_state = state.update(iteration=5)  # immutable update
```

The tree is recursive: `state.children[0].children[1]` is a grandchild's full state. Serialize the root and you've captured the entire multi-agent computation.

## StepEvents

Every `step()` attaches a typed event to the returned state:

| Event | Key Fields | Emitted When |
|-------|-----------|------|
| `LLMReply` | `text`, `code` | LLM responded |
| `CodeExec` | `code`, `output`, `suspended` | REPL block executed |
| `ChildStep` | `child_events[]`, `all_done`, `exec_output` | Children were stepped |
| `NoCodeBlock` | `text` | LLM forgot the ```repl``` block |

`ChildStep.child_events` is recursive — if a child was itself supervising grandchildren, you get the full event tree.

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
    session=None,              # session persistence: str path, Session object, or None
    system_prompt=None,        # raw override (skips default builder)
)
```

### `RLM`

```python
agent = RLM(
    llm_client=llm,            # required — LLMClient instance
    runtime=runtime,           # required — Runtime instance
    config=RLMConfig(),        # optional — tuning knobs
    pool=ThreadPool(8),        # optional — execution pool (default: ThreadPool)
    prompt_builder=None,       # optional — custom PromptBuilder
    runtime_factory=None,      # optional — factory for child runtimes
    llm_clients=None,          # optional — named model registry
)
```

Subclass and override any method:

| Method | What it does |
|--------|-------------|
| `step(state)` | Dispatch to step_llm / step_exec / step_supervise |
| `step_llm(state)` | Call the LLM → HAS_REPLY |
| `step_exec(state)` | Execute code block → WAITING / SUPERVISING / FINISHED |
| `step_supervise(state)` | Flatten tree, step leaves in pool, cascade parent resumes |
| `build_system_prompt(state)` | Return the system prompt string |
| `build_messages(state)` | Assemble the LLM message list |
| `make_state(**fields)` | Construct initial state (override for custom state classes) |
| `create_child(agent_id, ...)` | Full control over child construction |

### `LLMClient`

```python
from rlmkit.llm import OpenAIClient, AnthropicClient

llm = OpenAIClient("gpt-5")
llm = AnthropicClient("claude-sonnet-4-20250514")

# Or implement your own
class MyLLM(LLMClient):
    def chat(self, messages: list[dict[str, str]]) -> str: ...
```

### `Runtime`

The execution environment — analogous to a process's address space:

```python
class Runtime(ABC):
    def execute(self, code: str, timeout=None) -> str: ...
    def inject(self, name: str, value: Any) -> None: ...
    def clone(self) -> Runtime: ...
    def start_code(self, code) -> tuple[bool, object]: ...
    def resume_code(self, send_value=None) -> tuple[bool, object]: ...
```

`LocalRuntime` runs code in-process with a persistent namespace. Builtin tools: `read_file`, `write_file`, `edit_file`, `append_file`, `ls`, `grep`. Common modules (`re`, `os`, `json`, `math`, etc.) are pre-imported. CWD is set to the workspace so `open("file.txt")` works naturally.

Variables persist across REPL turns — anything assigned in one code block is available in the next.

### Sessions

Agent message histories can be persisted to a **session store**. The engine writes after each step, and agents can read their own or other agents' sessions.

```python
agent = RLM(llm_client=llm, runtime=runtime, config=RLMConfig(session="context"))
```

When configured, two tools are registered:
- `list_sessions()` — list all agent IDs with sessions
- `read_history(agent_id=None, last_n=20)` — read any agent's message history

`FileSession` stores JSON files in a directory tree mirroring the agent hierarchy:

```
context/
├── session.json              ← root
├── search_0/
│   └── session.json          ← root.search_0
└── search_1/
    └── session.json
```

Custom backends (Redis, S3, database) — subclass `Session`:

```python
from rlmkit.session import Session

class RedisSession(Session):
    def write(self, agent_id: str, messages: list[dict]) -> None: ...
    def read(self, agent_id: str) -> list[dict]: ...
    def list_agents(self) -> list[str]: ...
    def exists(self, agent_id: str) -> bool: ...
```

### Model Selection

```python
agent = RLM(
    llm_client=OpenAIClient("gpt-5"),
    runtime=runtime,
    llm_clients={
        "fast": {"model": OpenAIClient("gpt-5-mini"), "description": "Cheap model for simple tasks"},
    },
)
# Agent can choose per delegation:
# delegate("search", "Find the needle", model="fast")
```

### `PromptBuilder`

System prompt assembled from ordered named sections. Default: `role`, `repl`, `recursion`, `examples`, `tools`, `guardrails`, `status`.

```python
from rlmkit.prompts import PromptBuilder, make_default_builder

builder = make_default_builder()

# Replace, add, remove sections
builder = builder.section("role", "You are a security auditor.", title="Role")
builder = builder.section("methodology", "Check OWASP Top 10.", title="Methodology", after="recursion")
builder = builder.remove("examples")

# Dynamic sections filled at render time
prompt = builder.build(tools="- grep(pattern, path): Search.", status="Depth 1 of 3.")
```

Or bypass entirely: `RLMConfig(system_prompt="...")`.

## Subclassing

### Custom prompt

```python
class SecurityAuditor(RLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_builder = (
            make_default_builder()
            .section("role", "You are a security auditor.", title="Role")
            .remove("examples")
        )
```

### Custom state

```python
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

### Step hooks

```python
class MonitoredRLM(RLM):
    def step(self, state):
        new_state = super().step(state)
        self.metrics.record(state.agent_id, new_state.event)
        if self.budget_exceeded():
            return new_state.update(status=Status.FINISHED, result="Budget exceeded")
        return new_state
```

### Custom pools

```python
from rlmkit.pool import Pool

class RateLimitedPool(Pool):
    def execute(self, items):
        results = {}
        for cs, engine in items:
            self.limiter.wait()
            results[cs.agent_id] = engine.step(cs)
        return results
```

## Examples

| Example | What it shows |
|---------|--------------|
| `basic.py` | 1M-line needle search with parallel sub-agent delegation |
| `needle_haystack.py` | 500-file search with `runtime_factory` for child isolation |
| `custom_agent.py` | Subclassed `RLM` + `RLMState` with custom state tracking |
| `summarizer.py` | Recursive summarization with custom prompt builder |

## Project Structure

```
rlmkit/
├── rlm.py           # RLM engine, RLMConfig, step logic
├── state.py         # RLMState, Status, StepEvent hierarchy
├── pool.py          # Pool ABC, ThreadPool, SequentialPool, CallablePool
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

- [Recursive Language Models](https://github.com/alexzhang13/rlm) — the original RLM paper and implementation
- [rlm-minimal](https://github.com/alexzhang13/rlm-minimal) — minimal single-file RLM reference
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
