# RLMFlow Internals

A deep reference for the engine's mechanics: data model, step
lifecycle, REPL/yield protocol, resume semantics, persistence, and
the extension seams on `RLMFlow`. If you want to subclass the
engine, debug a weird run, or write something on top of rlmflow,
this is the doc.

User-facing topic guides ([control](control.md), [observability](observability.md),
[runtimes](runtimes.md), [prompt_customization](prompt_customization.md),
[security](security.md)) cover the *what*. This doc covers the *how*.

---

## Architecture at a glance

```
                ┌────────────────────────────────────────────────┐
                │                    RLMFlow                     │  ← state + every overridable method
                │  (rlmflow/rlm.py — ~1k lines, one class)       │
                └──┬───────────────┬───────────────────┬────────┘
   pure helpers ───┼───────────────┼───────────────────┼─────────────────┐
                   ▼               ▼                   ▼                 ▼
            engine/actions    engine/replay      engine/scheduler   engine/seq
            (Action types,    (cold-start        (NodeScheduler:    (append_node,
            act/act_one)      replay-of-one)     runnable agents)   budget, ...)
                   │
                   ▼
                  Graph (recursive)  ←  Session / Workspace  ←  on disk
                  Node trajectory       (write_state, load_graph)
                  RuntimeRef        →   Runtime  (REPL: start_code, resume_code)
```

- **`RLMFlow`** owns state (sessions, runtimes, config, pool) and the
  loop. Every overridable seam — `step`, `apply_one`, `step_llm`,
  `step_exec`, `step_after_supervising`, `reply_to`, `call_llm`,
  `extract_code`, `build_messages`, `inject_env`, `spawn_child`, etc.
  — is a method on this class.
- **`engine/*`** is pure helpers. Anything in there is a free function
  or pure data with no engine state.
- **`Graph`** is the immutable data model. Every `start`/`step` returns
  a fresh snapshot reloaded from the session.
- **`Runtime`** runs the REPL. Send/receive a small JSON protocol;
  shipped variants are `LocalRuntime`, `SubprocessRuntime`,
  `DockerRuntime`, `ModalRuntime`.

The principle: **if something needs engine state to do its job, it's
a method on `RLMFlow`. If it's a pure function of its arguments, it's
in `engine/*`. No middle category, no kwarg bags.**

---

## The data model

A `Graph` is a recursive structure. `graph[other_aid]` returns the
`Graph` rooted at any descendant agent; per-agent invariants live as
flat fields on `Graph` itself; sub-agents live in `graph.children`;
the agent's trajectory lives in `graph.states`.

```python
graph.agent_id         # str
graph.depth            # int
graph.query            # str
graph.system_prompt    # str — snapshot from the first turn
graph.config           # dict — engine knobs at spawn
graph.workspace        # WorkspaceRef | None
graph.runtime          # RuntimeRef | None
graph.parent_agent_id  # str | None
graph.parent_node_id   # str | None — the ActionNode id that delegated us

graph.states           # tuple[Node, ...]  — this agent's trajectory
graph.children         # dict[str, Graph]  — direct sub-agents

# subtree views
graph.agents           # Mapping[agent_id, Graph]
graph.nodes            # every Node in the subtree (queryable)
graph.edges            # derived flows_to / spawns edges
```

### Node taxonomy

Trajectories are a strict alternation of **observations** (inputs the
system received) and **actions** (work the system did). Every
`ActionNode` is followed by exactly one `ObservationNode`. Nine leaf
types under four base classes:

| `type`               | Class               | Base                    | Carries                                            |
|----------------------|---------------------|-------------------------|----------------------------------------------------|
| `user_query`         | `UserQuery`         | `ObservationNode`       | initial task (root query / spawn prompt)           |
| `llm_action`         | `LLMAction`         | `ActionNode`            | "called the LLM" + model name                      |
| `llm_output`         | `LLMOutput`         | `ObservationNode`       | reply, extracted REPL code, token deltas           |
| `exec_action`        | `ExecAction`        | `ActionNode`            | "ran fresh code"                                   |
| `exec_output`        | `ExecOutput`        | `CodeObservation` (obs) | runtime stdout                                     |
| `supervising_output` | `SupervisingOutput` | `CodeObservation` (obs) | code yielded; `waiting_on` lists pending children  |
| `error_output`       | `ErrorOutput`       | `CodeObservation` (obs) | failure                                            |
| `done_output`        | `DoneOutput`        | `CodeObservation` (obs) | terminal answer from `done(...)`                   |
| `resume_action`      | `ResumeAction`      | `ActionNode`            | "supervisor resumed paused code"                   |

Every leaf in the persisted graph is an observation; intermediate
action nodes only exist *inside* a single `apply_one` call's writes.
See [`internal/node_model.md`](internal/node_model.md) for the full
state-machine spec.

---

## The step lifecycle

`step(graph) -> graph'` advances **one observation-to-observation
transition** per ready agent. A logical "reasoning turn" of an agent
(call the LLM, run the code it emitted) is therefore two `step()`
rounds:

1. **LLM half:** `obs → LLMAction → LLMOutput`.
2. **Exec half:** `LLMOutput → ExecAction → <CodeObservation>`.

Concurrency is finer-grained — one agent can be mid-LLM while another
is mid-exec.

### Two-phase split: `act` then `apply`

`step()` runs two phases each round:

```python
def step(self, graph):
    runnable = self.node_scheduler.runnable_agents(graph)
    plan = act(graph,
               config=self.config,
               runnable=runnable,
               terminate_requested=self.terminate_requested)
    tasks = [(aid, lambda a=action: self.apply_one(a))
             for aid, action in plan.items()]
    self.pool.execute(tasks)
    return self.session.load_graph()
```

- **Plan (pure).** `act(graph, ...) -> ActionPlan` in
  `engine/actions.py`. Pure projection `Graph -> {agent_id: Action}`.
  No I/O, no writes. For each runnable agent, it inspects the
  current observation and returns one of:
    - `CallLLM(agent_id, force_final=..., model=...)` — agent rests at
      a `UserQuery` / `ExecOutput` / `ErrorOutput`.
    - `Exec(agent_id)` — agent rests at an `LLMOutput`.
    - `Resume(agent_id)` — agent rests at a `SupervisingOutput`
      whose children have all settled.
    - `None` — terminal, empty, or stray half-written state.

- **Apply (I/O).** `self.apply_one(action)` materializes one
  `Action` against the persisted graph. Reloads the graph from
  `self.session`, enforces the global token budget, dispatches to
  one of three half-step handlers, and writes the resulting
  `(ActionNode, ObservationNode)` pair through the session.

  ```python
  def apply_one(self, action):
      graph = self.session.load_graph().agents[action.agent_id]
      over = budget_exceeded(graph, self.config.max_budget)
      if over is not None:
          append_node(self.session, graph,
                      DoneOutput(result=f"[budget exceeded: {over} tokens]"))
          return
      cur = graph.current()
      if isinstance(action, CallLLM):
          self.step_llm(graph, cur,
                        force_final=action.force_final, model=action.model)
      elif isinstance(action, Exec):
          self.step_exec(graph, cur)
      elif isinstance(action, Resume):
          self.step_after_supervising(graph, cur)
  ```

`Action` values are intentionally **lite** — just the policy intent
plus inputs that aren't recoverable from the graph (`force_final`,
model override). The persisted `ActionNode` written by the handlers
carries the full record of what happened.

### Scheduling

`NodeScheduler` (in `engine/scheduler.py`) is a pure top-down walk:

- Visit `graph.agent_id`.
- If finished or terminal, skip.
- If the agent rests at a `SupervisingOutput`:
    - If every child it's `waiting_on` is finished, the supervisor
      itself is runnable — return its id.
    - Otherwise recurse into the still-running children.
- Otherwise (rests at a normal observation), the agent is runnable.

The scheduler never produces side effects; `apply_one` re-checks
`can_resume` inside `step_after_supervising` before actually
resuming.

### The three half-step handlers

#### `step_llm` — LLM half

```python
def step_llm(self, graph, last, *, force_final, model=None):
    llm_model = model or graph.config.get("model", "default")
    llm_action = LLMAction(agent_id=graph.agent_id,
                           seq=last.seq + 1, model=llm_model)
    append_node(self.session, graph, llm_action)
    llm_output, usage = self.reply_to(graph, llm_action,
                                      force_final=force_final)
    self.record_usage(usage)
    append_node(self.session, graph, llm_output)
```

Writes `LLMAction` *first*, then calls `reply_to` to talk to the
model, then writes `LLMOutput`. The action-before-call ordering means
`build_messages` (called inside `reply_to`) sees the action in
`graph.states` for the in-progress turn — that's why the
`CONTINUE_ACTION` nudge gates on `LLMOutput` count, not `LLMAction`
count.

#### `step_exec` — exec half

```python
def step_exec(self, graph, llm_output):
    exec_action = ExecAction(agent_id=graph.agent_id,
                             seq=llm_output.seq + 1, code=llm_output.code)
    append_node(self.session, graph, exec_action)
    if not llm_output.code:
        append_node(self.session, graph,
                    ErrorOutput(content=NO_CODE_BLOCK, error="no_code_block"))
        return
    self._run_exec(graph, exec_action, llm_output.code)
```

`_run_exec` is the long branch that calls into the runtime, handles
`done(...)`, `rlm_delegate(...)` + `yield rlm_wait(...)`, orphaned delegates,
and exceptions. It produces exactly one of `ExecOutput`,
`SupervisingOutput`, `ErrorOutput`, or `DoneOutput`.

#### `step_after_supervising` — resume half

Runs after all `waiting_on` children settle. Writes a `ResumeAction`,
then either resumes the live generator or replays it (cold start; see
[Cold-start replay](#cold-start-replay)), then produces the next
observation just like `step_exec`.

---

## The REPL `await` protocol

The engine **only** intercepts top-level `await rlm_wait(*handles)`
values. Top-level `yield` is not part of the action language.

### The protocol

Each REPL block with top-level await is compiled with Python's
top-level-await flag so the engine can drive the resulting coroutine via
`send()`. The engine sends values into the coroutine and decides what to
do with each awaited value:

| What's awaited                          | Engine reaction                                    |
|-----------------------------------------|----------------------------------------------------|
| `rlm_wait(handle, …)` -> `WaitRequest`  | **Suspend** the agent until those children settle. |
| Anything else                           | Error; only `await rlm_wait(...)` is supported.    |
| `StopIteration`                         | Block done; return captured stdout.                |

The `WaitRequest` returned by `rlm_wait(...)` carries the child agent IDs
the engine needs to schedule on. Without it the engine has no
suspension target.

### What "top level" means

An await is top-level iff the `await` keyword sits in the REPL block
itself — *not* nested inside a `def` / `async def` / `lambda` / class
body / comprehension. The implementation walks the AST and stops
descending at nested boundaries.

```python
# ── TOP-LEVEL awaits ────────────────────────────────────────────────
# These count. The REPL compiles with top-level await.
result = await rlm_wait(h)
for h in handles:
    result, = await rlm_wait(h)

# ── NOT top level ───────────────────────────────────────────────────
# These don't count for engine suspension.
def squares(n):
    for i in range(n):
        yield i * i                  # belongs to `squares`
print(list(squares(5)))

xs = list(i * i for i in range(10))   # genexp yield is hidden

f = lambda: (yield 1)
print(next(f()))

class Counter:
    def __iter__(self):
        for i in range(3):
            yield i
```

### Allowed shapes

✅ **No suspension:**

```python
# Helpers, genexps, comprehensions — all plain Python.
print(sum(x * x for x in range(100)))

```

✅ **Suspension:**

```python
h1 = rlm_delegate(name="worker", query="...", context="")
h2 = rlm_delegate(name="worker", query="...", context="")
results = await rlm_wait(h1, h2)   # suspends, resumes with results
```

`results` is a list of strings (the children's `done(...)` payloads).

❌ **Doesn't do what you want:**

```python
rlm_wait(handle)                   # missing await
await some_other_coroutine()       # only rlm_wait is supported
```

These are errors. **Use `await rlm_wait(handle)`.**

### Why this design

The engine has exactly two decisions to make about a REPL block.

#### Decision 1 — wrap as a generator?

The REPL builds a synthetic function around the LLM's code:

```python
def __rlm_gen__():
    <code from the LLMOutput>
```

…and calls it. If `<code>` contains a `yield` at the top level,
calling `__rlm_gen__()` returns a generator object the engine drives
via `send()`. If there's no top-level yield, it just runs straight
through. **The wrapping decision hinges on top-level yields, not
yields anywhere in the block.**

This is what fixed the old "`def squares(n): yield i*i` crashes the
agent" bug. `ast.walk` was finding the inner yield and wrapping the
block, but `__rlm_gen__` itself had no top-level yield, so calling it
ran straight through and returned `None`. The engine then did
`None.agent_ids` and crashed. The fix:
`_has_top_level_yield(tree)` walks the AST but stops at function /
async function / lambda / class boundaries.

#### Decision 2 — suspend or pump?

Once the block is wrapped, every `gen.send(...)` returns whichever
value the next top-level yield produced. The engine handles it:

```python
result = gen.send(prev_value)
while not isinstance(result, WaitRequest):
    result = gen.send(None)        # pump past non-Wait yields
suspend(result.agent_ids)          # the thing we know how to do
```

So:

```python
yield rlm_wait(h)         # WaitRequest → suspend on h
yield 42              # int → pump, immediately resume
yield                 # None → pump, immediately resume
yield handle          # ChildHandle → pump, immediately resume
```

The agent only suspends on the thing the engine actually has a plan
for: a `WaitRequest`. Every other yield value is treated like a plain
Python generator yield — discard, resume, no engine involvement.

### Net effect on the graph

| Block did this                          | `<observation>` is                          | Graph effect                                                                                  |
|-----------------------------------------|---------------------------------------------|-----------------------------------------------------------------------------------------------|
| Ran to completion (no yield)            | `ExecOutput` (stdout)                       | Single `ExecAction → ExecOutput` pair.                                                        |
| Top-level non-Wait yields only          | `ExecOutput` (stdout)                       | **Same as above.** Non-Wait yields are invisible to the graph.                                |
| Top-level `yield rlm_wait(h1, h2)`          | `SupervisingOutput(waiting_on=[h1, h2])`    | Agent suspends. Children run. When they finish, a `ResumeAction` resumes.                     |
| `done(...)` called (any path)           | `DoneOutput`                                | Agent terminates.                                                                             |
| Exception                               | `ErrorOutput`                               | Surfaces in the next user message as a retry observation.                                     |

**Rule:** non-Wait yields never produce a new graph node. They're
internal to one `ExecAction`'s execution. Only `yield rlm_wait(...)`
introduces a `SupervisingOutput` and the eventual `ResumeAction`.

### Implementation

- `rlmflow/runtime/repl.py::_has_top_level_yield(tree)` — AST walk that
  skips `FunctionDef` / `AsyncFunctionDef` / `Lambda` / `ClassDef`
  bodies.
- `rlmflow/runtime/repl.py::REPL.advance()` — after `gen.send(...)`,
  loops sending `None` until either the generator yields a
  `WaitRequest` (return suspended) or raises `StopIteration` (done).
- Tests in `tests/test_repl_yield.py` cover the exhaustive matrix.

---

## Resume semantics

`yield rlm_wait(...)` is a Python generator suspension point. It is
**not** a request to copy child outputs into the next LLM prompt.

### The data path

For code like:

```python
handles = [rlm_delegate(name="a", query=q_a, context=c_a), rlm_delegate(name="b", query=q_b, context=c_b)]
results = yield rlm_wait(*handles)
print(len(results))
```

the flow is:

1. The parent REPL runs until `yield rlm_wait(*handles)`.
2. The generator suspends. The assignment to `results` has **not**
   happened yet.
3. The graph records a `SupervisingOutput(waiting_on=[a, b])`.
4. The scheduler runs the child agents until they finish.
5. When all children have a terminal `DoneOutput`, the runtime
   resumes the same parent generator and sends the child results
   list back into the suspended `yield`.
6. The line becomes equivalent to `results = child_results`.
7. The same stateful REPL continues and runs `print(len(results))`.
8. If the resumed code ends without `done(...)` or another
   `yield rlm_wait(...)`, the engine records an `ExecOutput`
   (`resumed_from=[...]`) and the next LLM turn continues in the same
   stateful REPL — variables assigned after the wait are still in
   scope.

**There is exactly one data path from children back to parent code:**

```
Child DoneOutput.result
  -> collected by step_after_supervising
  -> sent into runtime.resume_code(graph, results)
  -> assigned to parent REPL variable `results`
```

`ResumeAction` is **not** part of the data path. It's structured
graph metadata: "the suspended generator resumed and ran some more
code." The next prompt does **not** receive child result blobs or
"wait completed" text — the LLM inspects `results` directly because
the REPL is stateful.

### Worked walkthrough

Root receives:

```text
Research whether SDSK can reach $2000 by EOY 2026.
```

#### Step 0 — root query

```
root
  [0] UserQuery   content="Query: Research whether SDSK..."
```

#### Step 1 — root delegates and waits

The LLM emits one REPL block:

```python
contract = "Return JSON with claims, evidence, sources, contradictions."
jobs = [
    ("identity",     "Identify the listed security.",      contract),
    ("valuation",    "Find price, market cap, multiples.", contract),
    ("fundamentals", "Research growth, revenue, margins.", contract),
    ("analyst",      "Collect analyst targets.",           contract),
]
handles = [rlm_delegate(name=name, query=q, context=c, model="fast") for name, q, c in jobs]
results = yield rlm_wait(*handles)
print(f"got {len(results)} child results")
```

Runtime execution:

1. `rlm_delegate(...)` creates four child agents.
2. `yield rlm_wait(*handles)` suspends the root generator.
3. The graph records `SupervisingOutput(waiting_on=[...4 ids...])`.

```
root
  [0] UserQuery
  [1] LLMAction
  [2] LLMOutput          code=<the REPL block above>
  [3] ExecAction
  [4] SupervisingOutput  waiting_on=[root.identity, root.valuation,
                                     root.fundamentals, root.analyst]

root.identity      [0] UserQuery
root.valuation     [0] UserQuery
root.fundamentals  [0] UserQuery
root.analyst       [0] UserQuery
```

Data location: child outputs do not exist yet. Root REPL is suspended
at `yield rlm_wait(*handles)`. `handles` exists in the root REPL.
`results` does not.

#### Step 2 — children run

Each child runs to its own `DoneOutput`. After they all finish:

```
root.identity
  [0] UserQuery
  [1] LLMAction
  [2] LLMOutput
  [3] ExecAction
  [4] DoneOutput  result='{"topic":"identity",...}'
...
```

#### Step 3 — root resumes inline

`NodeScheduler.runnable_agents` sees all of root's `waiting_on`
children are terminal. `apply_one(Resume(root))` collects:

```python
child_results = [
    graph["root.identity"].result(),
    graph["root.valuation"].result(),
    graph["root.fundamentals"].result(),
    graph["root.analyst"].result(),
]
```

…and calls `runtime.resume_code(graph, child_results)`. The suspended
line:

```python
results = yield rlm_wait(*handles)
```

continues as if it were `results = child_results`. The same REPL
continues:

```python
print(f"got {len(results)} child results")  # prints "got 4 child results"
```

The generator ends. The agent isn't done yet (no `done(...)` was
called and no further yield). The engine records:

```
root
  [0] UserQuery
  [1] LLMAction
  [2] LLMOutput
  [3] ExecAction
  [4] SupervisingOutput
  [5] ResumeAction        resumed_from=[...4 ids...]
  [6] ExecOutput          output="got 4 child results\n"
                          resumed_from=[...4 ids...]
```

Data location: `results` lives in the root REPL.
`ResumeAction`/`ExecOutput` are trace/UI state — they're **not** a
prompt payload.

#### Step 4 — next LLM turn

`build_messages` does **not** inject child result blobs or
"wait completed" text. The next prompt just includes the normal
continue instruction (`CONTINUE_ACTION`) and any stdout from the
resumed block. The LLM knows the REPL is stateful and inspects
`results` directly:

```python
import json
sections = [json.loads(r) for r in results]
report = synthesize_report(sections)
done(report)
```

#### Final shape

```
root
  [0] UserQuery
  [1] LLMAction          delegate identity/valuation/fundamentals/analyst
  [2] LLMOutput
  [3] ExecAction
  [4] SupervisingOutput  waiting_on=[...]
  [5] ResumeAction       resumed root code ran
  [6] ExecOutput         resumed stdout
  [7] LLMAction          parse stateful `results`, synthesize, done(report)
  [8] LLMOutput
  [9] ExecAction
  [10] DoneOutput

root.{identity,valuation,fundamentals,analyst}
  [0] UserQuery
  [1] LLMAction
  [2] LLMOutput
  [3] ExecAction
  [4] DoneOutput
```

### Multi-yield in one block

A block can yield twice:

```python
h1 = rlm_delegate(name="a", query=..., context=...);  r1 = yield rlm_wait(h1)
h2 = rlm_delegate(name="b", query=..., context=...);  r2 = yield rlm_wait(h2)
done(combine(r1, r2))
```

Each yield/resume pair gets its own `(SupervisingOutput, ResumeAction)`
in the parent's trajectory. The REPL is the same generator both
times — variables persist.

### Multi-yield across blocks

```python
# Block 1
h = rlm_delegate(name="a", query=..., context=...)
yield rlm_wait(h)            # block ends right after the yield

# Block 2
done("p:" + results[0])
```

Same agent, two LLM turns. The first block's `LLMOutput` →
`ExecAction` → `SupervisingOutput`. After the child settles, the
resume produces an `ExecOutput`; then the agent runs another LLM
turn and the second block's code runs in a *fresh* REPL submission
(but the runtime keeps the same namespace — `rlm_delegate`,
`results`, and any prior assignment are in scope).

---

## Cold-start replay

When the engine is attached to a freshly-loaded workspace
(`Workspace.fork(...)` or just opening a saved run later), the
durable graph contains every `SupervisingOutput` that was recorded —
but the live Python generator that yielded inside the runtime is
gone.

`engine/replay.py::replay_to_yield(graph, target, runtime)` handles
this. It re-runs the action code with `rlm_delegate` in *replay mode*
(returns existing child handles instead of spawning new ones), so the
generator pauses again at the same yield. The regular resume path
then takes over.

```python
def step_after_supervising(self, graph, last):
    if not can_resume(graph, last):
        return                      # children still need to advance
    results = results_for_supervise(graph, last)
    resume_action = ResumeAction(...)
    append_node(self.session, graph, resume_action)
    runtime = self.inject_env(graph, resume_action)
    if not runtime.suspended:
        # Live generator is gone — process restart, fork, etc.
        # Replay action code with rlm_delegate in replay mode so the
        # generator pauses at the same yield we recorded.
        replay_to_yield(graph, last, runtime)
    # Now the runtime is suspended at the right yield; resume.
    suspended, raw, errored = runtime.resume_code(results)
    ...
```

`replay_to_yield` walks the trajectory back to the originating
`LLMOutput.code`, sets `runtime.env["_REPLAY_QUEUE"]` to the agent
IDs the original yield was waiting on, runs the code, and verifies
that the new yield's `WaitRequest.agent_ids` matches what we
recorded. On divergence, it raises.

For multi-yield blocks, it walks the prior `SupervisingOutput`/
`ResumeAction` chain in the same execution and replays each yield in
order.

---

## Persistence

A workspace is the durable run. It separates per-agent state logs,
the graph manifest, and task payloads:

```text
workspace/
  graph.json                  # workspace manifest: root + agent list + spawns edges
  session/
    root/
      agent.json              # per-agent invariants written once
      session.jsonl           # one Node per line, in seq order
      latest.json             # cached summary of the latest state
      transcript.json         # full LLM chat history for this agent
    root.child/
      agent.json
      session.jsonl
      latest.json
      transcript.json
  context/
    root/context.txt          # CONTEXT payload + metadata
    root.child/context.txt
```

### Reads and writes

- `session.write_state(node)` — append one immutable `Node` to
  `session.jsonl` and update `latest.json`. Called from
  `append_node()` in `engine/seq.py`, which assigns `agent_id` and
  `seq` deterministically.
- `session.write_agent(graph)` — write the per-agent invariants
  (`query`, `system_prompt`, `config`, `runtime`, etc.) to
  `agent.json`. Called on agent creation only.
- `session.write_transcript(agent_id, transcript)` — update the agent's
  flat LLM chat history. Called from `RLMFlow.transcript_recorder.record_turn()`
  (a `TranscriptRecorder` living in `rlmflow/engine/transcript.py`)
  inside `reply_to()`, and from `record_terminal()` when an agent
  finishes via `done(...)`. **Append-only**: each call adds just the
  new messages since the last call, never rewrites the prefix.
- `session.load_graph()` — rehydrate the persisted state as the same
  `Graph` shape the engine emits. `flows_to` edges are derived from
  state order; `spawns` edges come straight from `graph.json`.

### Transcripts

The transcript file is parallel-but-separate from the trajectory.
The trajectory tracks every action/observation; the transcript
tracks the flat conversation as the LLM saw it across turns. Each
turn appends only:

- the new user-side messages (any nudges between the previous turn
  and this one — typically `CONTINUE_ACTION` or an `ErrorOutput`'s
  `content`)
- the assistant reply
- per-assistant metadata (timestamp, model, force_final, token
  counts, elapsed_s, after_node_id, after_seq)

Transcript-write failures are swallowed: persistence should never
break a run.

### CONTEXT

`Workspace.context` stores optional payloads exposed inside the REPL
as `CONTEXT`. The root agent's payload is keyword-only and optional:

```python
graph = agent.start("answer from the payload", context=large_text)
```

Inside the REPL, agents see `CONTEXT` (read-only payload), `SESSION`
(read-only view of every other agent in the run), and the standard
filesystem tools. Sample, slice, or pass the full payload explicitly:

```python
CONTEXT.info()                  # {"chars": int, "lines": int}
sample  = CONTEXT.read(0, 2000) # char slice as str
window  = CONTEXT.lines(0, 50)  # line slice as list[str]
hits    = CONTEXT.grep(r"TODO") # lineno:line rows
full    = CONTEXT.read()        # full payload
```

`rlm_delegate(*, name, query, context)` writes the child's context payload
under `context/<child_id>/context.txt` before the child's first
`UserQuery` is appended.

---

## Runtime sessions

The engine keeps a separate runtime session per agent so each agent's
REPL state, suspended generator, and tool closures are isolated.

```python
self.runtime_sessions: dict[str, Runtime] = {ROOT_RUNTIME_ID: root_runtime}
```

Three methods own the lifecycle:

- `RLMFlow.runtime_for(ref)` — return the runtime bound to `ref`.
  Lazily restores on a fresh engine attached to a forked or reloaded
  workspace (the dict only holds `ROOT_RUNTIME_ID` after a cold
  start; everything else is materialized on demand by cloning the
  root or calling `runtime_factory`).
- `RLMFlow.create_runtime_session(parent_runtime, *, agent_id)` —
  allocate a fresh runtime session for a freshly-spawned child.
  Called from `spawn_child()`.
- `RLMFlow.inject_env(graph, node)` — clear and re-seed
  `runtime.env` and the REPL namespace before each code execution
  or resume. Pushes `AGENT_ID`, `DEPTH`, `MAX_DEPTH`,
  `PARENT_NODE_ID`, `DONE_RESULT`, `DELEGATED`, plus `CONTEXT` /
  `SESSION` proxies.

`register_tools(runtime)` binds the core `done` / `rlm_wait` /
`rlm_delegate` closures to `runtime.env`. The `rlm_delegate` closure
captures `self.spawn_child` so it can call back into engine state.

See [`runtimes.md`](runtimes.md) for the `Runtime` protocol and
shipped variants.

---

## RLMFlow — the overridable surface

Every public method on `RLMFlow` is an extension seam. Subclass and
override what you want; the default implementations call `super()`
or pure helpers from `engine/*`.

| Stage of the loop      | Methods                                                                                          |
|------------------------|--------------------------------------------------------------------------------------------------|
| Lifecycle              | `start`, `run`, `chat`, `step`, `terminate`                                                      |
| Per-step transitions   | `apply_one`, `step_llm`, `step_exec`, `step_after_supervising`                                   |
| LLM half-step          | `reply_to`, `call_llm`, `llm_client_for`, `extract_code`                                         |
| Messages / prompt      | `build_messages`, `build_system_prompt`, `build_system_prompt_for`, `build_tools_section`, `build_status_section` |
| Runtime / env          | `runtime_for`, `create_runtime_session`, `inject_env`, `register_tools`, `format_exec_output`    |
| Child spawning         | `spawn_child`                                                                                    |
| Bookkeeping            | `record_usage`, `node_config`                                                                    |

Every override actually works. The engine calls these through
`self`, so the dispatch goes through your subclass:

```python
class LoggingFlow(RLMFlow):
    def extract_code(self, text):
        code = super().extract_code(text)
        return None if code is None else PRELUDE + code

class RetryingFlow(RLMFlow):
    def call_llm(self, messages, *, client=None):
        for _ in range(3):
            try: return super().call_llm(messages, client=client)
            except TransientError: ...
        raise

class TracingFlow(RLMFlow):
    def apply_one(self, action):
        log.info("apply %s on %s", type(action).__name__, action.agent_id)
        return super().apply_one(action)

class CachedExecFlow(RLMFlow):
    def step_exec(self, graph, llm_output):
        if hit := self.cache.get(llm_output.code):
            return self._write_cached(graph, llm_output, hit)
        return super().step_exec(graph, llm_output)

class RoutedFlow(RLMFlow):
    def llm_client_for(self, graph):
        if graph.depth >= 2:
            return self.llm_clients["fast"]
        return super().llm_client_for(graph)

class GatedSpawnFlow(RLMFlow):
    def spawn_child(self, *args, **kwargs):
        if self.quota_exceeded(args[0]):
            return "[refused: quota exceeded]"
        return super().spawn_child(*args, **kwargs)
```

For prompt-builder customization (the common case), don't subclass —
use `PromptBuilder` sections instead. See
[`prompt_customization.md`](prompt_customization.md).

---

## Concurrency

`RLMFlow.step()` plans every runnable agent's next action, then
applies them through `self.pool`. The pool is a small abstraction
with one method:

```python
class Pool(ABC):
    @abstractmethod
    def execute(self, tasks: list[tuple[str, Callable[[], Any]]]) -> dict[str, Any]: ...
```

Three shipped pools (`rlmflow/utils/pool.py`):

- `ThreadPool(max_workers)` — `concurrent.futures.ThreadPoolExecutor`.
  Used by default when `RLMConfig.max_concurrency >= 2`.
- `SequentialPool` — runs one task at a time. Used when
  `max_concurrency` is `None`, `0`, or `1`. Useful for debugging
  and rate-limited setups.
- `CallablePool(fn)` — wraps a plain `def fn(tasks): ...` so users
  can plug in custom schedulers.

Default `max_concurrency = os.cpu_count()`. Agent work is mostly LLM
I/O, so the threadpool is essentially free; explicit `None` is the
opt-out.

You can also pass a custom `pool=` to `RLMFlow(...)`:

```python
agent = RLMFlow(..., pool=ThreadPool(max_workers=4))
agent = RLMFlow(..., pool=lambda tasks: my_async_scheduler(tasks))
```

---

## Action edge cases

A few edge cases that crop up in real runs:

### Orphaned delegates

If a block calls `rlm_delegate(...)` but never `yield rlm_wait(...)`'s on the
handle, the engine surfaces an `ErrorOutput(error="orphaned_delegates")`
and re-raises the corresponding exception inside the REPL on the next
turn so the LLM sees the failure inline. Children that were already
spawned stay in the graph (their work isn't discarded) but the parent
has to call `rlm_wait(...)` to consume their results.

### No code block

If the LLM reply has no parseable ```repl``` block, `extract_code`
returns `None`. `step_exec` writes an `ExecAction` with empty `code`
and an `ErrorOutput(content=NO_CODE_BLOCK, error="no_code_block")`.
The next round routes back to `step_llm` and the model is nudged to
retry.

### Max iterations

`act_one` checks `iteration_count(graph)` (count of `LLMAction`
nodes) against `graph.config["max_iterations"]`. When exhausted, it
emits `CallLLM(force_final=True)`. `build_messages` swaps the
trailing `CONTINUE_ACTION` for `FINAL_ANSWER_ACTION`, which strongly
nudges the model to call `done(...)`.

### Terminate

`agent.terminate(graph)` marks every still-running agent for a
final-answer turn:

```python
def terminate(self, graph):
    for aid in graph.agents:
        if not graph.agents[aid].finished:
            self.terminate_requested.add(aid)
    return self.session.load_graph()
```

`act_one` then propagates this into `CallLLM(force_final=True)` on
the next round.

### Budget

`apply_one` enforces `RLMConfig.max_budget` at the top of every call.
When exceeded, it writes
`DoneOutput(result=f"[budget exceeded: {n} tokens]")` and skips the
handler entirely.

### Max depth

`spawn_child` rejects new children once
`parent.depth >= config.max_depth`:

```python
if parent.depth >= self.config.max_depth:
    return f"[refused: max depth {self.config.max_depth}] Do this directly."
```

The string return is the documented refusal protocol — `rlm_delegate(...)`
in the REPL gets back a string instead of a `ChildHandle`, so the
parent's code can detect it (`isinstance(h, str)`) and recover.

---

## Where to read next

- [`internal/node_model.md`](internal/node_model.md) — full state-machine
  spec, every legal transition, simulation walkthroughs.
- [`control.md`](control.md) — user-facing step loop, forks, custom
  tools/prompts.
- [`observability.md`](observability.md) — querying the `Graph` API,
  viewer, exports.
- [`runtimes.md`](runtimes.md) — Runtime protocol, shipped backends.
- [`security.md`](security.md) — trust boundary, isolation knobs.
- [`prompt_customization.md`](prompt_customization.md) — building
  custom system prompts.
