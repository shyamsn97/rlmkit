# Engine Simplification Plan

`rlmflow/rlm.py` is one class doing too many jobs: lifecycle, scheduling, state
transitions, persistence, runtime tool wiring, prompt assembly, and child
spawning. This document locks in a small redesign that keeps the observable
runtime semantics but makes the state machine obvious.

The single biggest simplification: **state that the core tools need lives
inside the REPL namespace as env-style variables**, not on the host as a
context object. Each agent already owns its REPL, so per-agent state is just
namespace variables. No `ContextVar`. No per-step closure injection. No
`ActiveStep` object.

## Invariants

These do not change.

### Graph is the source of truth

Each step:

1. read latest snapshot from `Session`,
2. select runnable agents,
3. append new nodes,
4. return a freshly loaded `Graph`.

Engine never mutates `Graph.states` directly.

### Node meanings are strict

```text
QueryNode        user task
ActionNode       LLM-authored code block
ObservationNode  result of a fresh ActionNode running to completion
SupervisingNode  generator suspended at yield wait(...)
ResumeNode       result of resumed code running until next yield/done/end
ResultNode       done(...) was called
ErrorNode        engine/runtime/tool error
```

`ResumeNode` is the execution record after a wait resumes. It is not a child-
output injection.

### Resume executes code

For:

```python
results = yield wait(h1)
print("ok")
results2 = yield wait(h2)
done("done")
```

Root sequence:

```text
query action supervising resume supervising resume result
```

For split blocks (one `yield wait` per block):

```text
query action supervising resume action supervising resume action result
```

### Child results flow through the REPL only

`yield wait(...)` resumes with a list of child result strings. Python assigns
them. The next code block reads them from the REPL namespace.

The LLM prompt never contains a synthetic copy of child outputs.

### Prompt history rules

- `QueryNode`, `ObservationNode`, `ResumeNode` -> user messages.
- `ActionNode` -> assistant message.
- `SupervisingNode` -> skipped (engine control state).
- `ResultNode` -> skipped (terminal).

`ResumeNode.content` is normal `REPL output:\n...` only. No hint, no
`resumed_from`, no child summaries.

## Redesign

### One idea: all per-execution state lives in the REPL namespace

Every agent already gets its own `Runtime`, which owns its own `REPL`, which
owns its own `namespace` dict. Two parallel agents are two parallel
namespaces. Nothing is shared.

So the design is:

1. Each agent's namespace holds that agent's state.
2. `done`, `wait`, `delegate` are defined as Python source that runs **inside
   the REPL** at runtime startup. They live in the namespace alongside the
   state they read and write.
3. The host has zero per-agent state. The host only needs to inject env-style
   values before each execution and read result values back after.

This kills:

- `ActiveStep` / `ExecutionContext`,
- `ContextVar` machinery,
- `bind_step_tools` per-step closures,
- the placeholder `@tool def done/delegate/wait(self, ...): raise ...` stubs.

### REPL preamble (one-time, on runtime creation)

When a `Runtime` is created, the engine execs this preamble in the REPL
namespace once:

```python
from rlmflow.graph import ChildHandle, WaitRequest

# state. reset to defaults before every action/resume run.
AGENT_ID = ""
DEPTH = 0
MAX_DEPTH = 0
PARENT_NODE_ID = ""
_DONE_RESULT = None
_DELEGATED = []


def done(message):
    global _DONE_RESULT
    if _DONE_RESULT is None:
        _DONE_RESULT = str(message).strip()
        print(f"[done] {_DONE_RESULT}")
    return _DONE_RESULT


def wait(*handles):
    return WaitRequest(agent_ids=[h.agent_id for h in handles])


def delegate(name, query, context, *, max_iterations=None, model="default"):
    handle = _spawn_child(
        AGENT_ID, PARENT_NODE_ID, name, query, context,
        max_iterations=max_iterations, model=model,
    )
    if isinstance(handle, str):  # error string from the host
        return handle
    _DELEGATED.append(handle.agent_id)
    return handle
```

`done` and `wait` need no host call at all. `delegate` needs the host to write
files / spawn a child runtime, so it calls `_spawn_child(...)` — a single
stable host proxy registered alongside the preamble. The REPL stub passes
`AGENT_ID` and `PARENT_NODE_ID` from the namespace, so the host signature is
fully argument-driven and stateless.

`_DONE_RESULT` is a real name in the REPL namespace. `global _DONE_RESULT`
inside `done` refers to that namespace because the function was defined in it.
Nothing leaks to the host.

### Per-execution: inject env, run, read back

Before every action or resume run:

```python
runtime.inject("AGENT_ID", graph.agent_id)
runtime.inject("DEPTH", graph.depth)
runtime.inject("MAX_DEPTH", max_depth)
runtime.inject("PARENT_NODE_ID", node.id)
runtime.inject("CONTEXT", context_view(graph))   # object proxy
runtime.inject("SESSION", session_view(graph))   # object proxy
runtime.inject("_DONE_RESULT", None)
runtime.inject("_DELEGATED", [])
```

Then call `runtime.start_code(...)` or `runtime.resume_code(...)`.

After the call returns, read the per-execution mutables back:

```python
done_result = runtime.read("_DONE_RESULT")   # None or str
delegated   = runtime.read("_DELEGATED")     # list[str]
```

`runtime.read(name)` is a new primitive that mirrors `runtime.inject(name)`.
For `LocalRuntime` it is a dict access. For subprocess/docker/modal runtimes
it is a small JSON command (`{"cmd": "read", "name": ...}` returning
`{"value": ...}`). One method, one REPL command.

### The host has no per-agent state

The host-side `_spawn_child` is a stable function with this signature:

```python
def _spawn_child(parent_agent_id, parent_node_id, name, query, context, *,
                 max_iterations=None, model="default") -> ChildHandle | str:
```

It does not need to know "which runtime called me." Everything it needs is in
the args. Two parallel REPLs invoking it concurrently are just two function
calls with different args — they touch different parent graphs, write
different files, create different child runtimes.

### Multithreaded safety in one paragraph

Each agent has its own `Runtime` -> own `REPL` -> own `namespace`. State that
the tools touch (`AGENT_ID`, `_DONE_RESULT`, `_DELEGATED`) is namespace-local.
Tool functions are defined in that namespace, so `global` refers to that
namespace. The only shared host code is `_spawn_child`, which is stateless
(args-only) and writes to per-agent paths. There is no `ContextVar` to leak,
no host attribute to race on, no closure to rebuild per step.

### One execution function

```python
def execute(graph, node, *, mode):
    runtime = runtime_for(graph)
    inject_env(runtime, graph, node)
    if mode == "start":
        suspended, raw = runtime.start_code(node.code)
    else:
        suspended, raw = runtime.resume_code(child_results_for(graph, node))
    return ExecutionResult(
        suspended=suspended,
        wait_request=raw[0] if suspended else None,
        output=raw[1] if suspended else (raw if isinstance(raw, str) else ""),
        done_result=runtime.read("_DONE_RESULT"),
        delegated=tuple(runtime.read("_DELEGATED")),
    )


@dataclass
class ExecutionResult:
    suspended: bool
    output: str
    wait_request: WaitRequest | None
    done_result: str | None
    delegated: tuple[str, ...]
```

Used identically for actions and resumes. The transition recorder branches on
the fields, not on a separate code path.

### One append helper

Replace `_record_state` with a plain function:

```python
def append_node(session: Session, graph: Graph, node: Node) -> Node:
    next_seq = graph.states[-1].seq + 1 if graph.states else 0
    fields = node.model_dump(exclude={"id", "agent_id", "seq"}, mode="python")
    fixed = node.__class__(agent_id=graph.agent_id, seq=next_seq, **fields)
    session.write_state(fixed)
    return fixed
```

No class. Call sites never assign `agent_id`, `seq`, or `id`.

### One stepper

```python
class NodeStepper:
    def step(self, agent_id: str) -> None:
        graph = self.session.load_graph().agents[agent_id]
        cur = graph.current()
        if isinstance(cur, SupervisingNode):
            self._supervising(graph, cur)
        elif isinstance(cur, ObservationNode):  # query/observation/resume/error
            self._observation(graph, cur)

    def _observation(self, graph, last):
        action = self._llm_turn(graph, last)
        action = append_node(self.session, graph, action)
        if isinstance(action, ErrorNode):
            return
        graph = self._reload(graph.agent_id)
        result = execute(graph, action, mode="start")
        self._record_action(graph, action, result)

    def _supervising(self, graph, last):
        if not can_resume(graph, last):
            return
        result = execute(graph, last, mode="resume")
        self._record_resume(graph, last, result)

    def _record_action(self, graph, action, result):
        if result.delegated and not result.suspended and result.done_result is None:
            append_node(self.session, graph, ErrorNode(error="orphaned_delegates", ...))
        elif result.done_result is not None:
            append_node(self.session, graph, ResultNode(result=result.done_result))
        elif result.suspended:
            append_node(self.session, graph, SupervisingNode(
                code=action.code,
                output=result.output,
                waiting_on=list(result.wait_request.agent_ids),
            ))
        else:
            append_node(self.session, graph, ObservationNode(
                code=action.code, output=result.output,
                content=format_repl(result.output),
            ))

    def _record_resume(self, graph, supervising, result):
        append_node(self.session, graph, ResumeNode(
            code=supervising.code,
            output=result.output,
            content=format_repl(result.output),
            resumed_from=list(supervising.waiting_on),
        ))
        graph = self._reload(graph.agent_id)
        if result.done_result is not None:
            append_node(self.session, graph, ResultNode(result=result.done_result))
        elif result.suspended:
            append_node(self.session, graph, SupervisingNode(
                code=supervising.code,
                output=result.output,
                waiting_on=list(result.wait_request.agent_ids),
            ))
```

Action runs always end with one of: `Result | Supervising | Observation | Error`.
Resume always writes `ResumeNode` first, then optionally `Result | Supervising`.

### `RLMFlow` becomes a small facade

```python
class RLMFlow(LLMClient):
    def start(self, query, **kw): ...
    def run(self, query, **kw): ...
    def chat(self, messages, **kw): ...

    def step(self, graph):
        runnable = self.scheduler.runnable_agents(graph)
        if not runnable:
            return graph
        self.pool.execute([
            (aid, (lambda aid=aid: self.stepper.step(aid)))
            for aid in runnable
        ])
        return self.session.load_graph()
```

Composition only. No transition logic. No tool wiring.

## What This Eliminates

- `ActiveStep` dataclass.
- `bind_step_tools(...)`.
- per-step closures `done_for_step`, `delegate_for_step`, `wait_for_step`.
- `@tool def done(self, ...): raise RuntimeError(...)` placeholders.
- per-step `runtime.inject("delegate", closure)` calls.
- any "active execution" `ContextVar` or host-side context object.
- `_record_state` with hand-rolled seq numbers in multiple call sites.
- separate "raw return" unpacking of `(suspended, raw)` in two places.

## Exact File Changes

Every line/range below refers to current files. Each step is a self-contained
patch. Steps must land in order: tests first, then `append_node`, then
`ExecutionResult`, then the REPL preamble, then `NodeStepper`, then the file
split.

### `rlmflow/runtime/repl.py` — add `read` command

Existing handler is at `rlmflow/runtime/repl.py` lines 279–298.

Add one branch in `REPL.handle` (between `cmd == "resume"` and the `inject*`
branches):

```python
if cmd == "read":
    return {"value": serialize(self.namespace.get(msg["name"]))}
```

Also extend the docstring at lines 9–17 with:

```text
- ``{"cmd": "read", "name": N}``                                 returns ``{"value": namespace.get(N)}``
```

### `rlmflow/runtime/runtime.py` — add `Runtime.read(...)`

Insert directly after `resume_code` (currently lines 156–158). Same pattern
as `start_code`/`resume_code`:

```python
def read(self, name: str) -> Any:
    """Return the REPL-namespace value bound to ``name`` (``None`` if missing)."""
    return self.call({"cmd": "read", "name": name}).get("value")
```

### `rlmflow/runtime/local.py` — override `read` for in-process

`LocalRuntime` already overrides `inject` at lines 41–43. Add the symmetric
override:

```python
def read(self, name: str) -> Any:
    return self.repl.namespace.get(name)
```

### `rlmflow/runtime/runtime.py` — REPL preamble + `_spawn_child` proxy hook

Add a new method on `Runtime` (next to `register_tool` at line 263). This is
called once on every runtime (the engine's, every clone, every fork):

```python
PREAMBLE_SOURCE = """\
from rlmflow.graph import ChildHandle, WaitRequest

AGENT_ID = ""
DEPTH = 0
MAX_DEPTH = 0
PARENT_NODE_ID = ""
_DONE_RESULT = None
_DELEGATED = []


def done(message):
    global _DONE_RESULT
    if _DONE_RESULT is None:
        _DONE_RESULT = str(message).strip()
        print(f"[done] {_DONE_RESULT}")
    return _DONE_RESULT


def wait(*handles):
    return WaitRequest(agent_ids=[h.agent_id for h in handles])


def delegate(name, query, context, *, max_iterations=None, model="default"):
    handle = _spawn_child(
        AGENT_ID, PARENT_NODE_ID, name, query, context,
        max_iterations=max_iterations, model=model,
    )
    if isinstance(handle, str):
        return handle
    _DELEGATED.append(handle.agent_id)
    return handle
"""


def install_preamble(self) -> None:
    """Exec the preamble in the REPL namespace and register tool metadata."""
    self.call({"cmd": "exec_preamble"})
    for name, desc in (
        ("done", "Mark the current agent as finished."),
        ("delegate", "Delegate a subtask to a named child agent."),
        ("wait", "Wait for delegated children. Must be called with `yield`."),
    ):
        self.tools[name] = ToolDef(
            name=name,
            signature=_PREAMBLE_SIGNATURES[name],
            description=desc,
            fn=None,
            core=True,
        )
```

Add a matching command in `REPL.handle` (`rlmflow/runtime/repl.py`):

```python
if cmd == "exec_preamble":
    from rlmflow.engine.preamble import PREAMBLE_SOURCE
    exec(PREAMBLE_SOURCE, self.namespace)
    return {"ok": True}
```

For `LocalRuntime`, override `install_preamble` to avoid the JSON round-trip:

```python
def install_preamble(self) -> None:
    from rlmflow.engine.preamble import PREAMBLE_SOURCE
    exec(PREAMBLE_SOURCE, self.repl.namespace)
    Runtime.install_preamble(self)  # tool metadata only
```

`_spawn_child` is a host-side proxy registered per runtime (see Step 4
below).

### `rlmflow/rlm.py` — exact deletions, edits, additions

Targets: `/Users/shyam/Code/rlmkit/rlmflow/rlm.py` (901 lines).

#### Delete

| Lines     | What                                                             |
| --------- | ---------------------------------------------------------------- |
| 83–91     | `ActiveStep` dataclass                                           |
| 753–770   | `active_step(...)` and `ActiveStepScope`                         |
| 772–798   | `bind_step_tools(...)`                                           |
| 800–809   | `done` placeholder + `done_for_step`                             |
| 811–875   | `delegate` placeholder + `delegate_for_step`                     |
| 877–883   | `wait` placeholder + `wait_for_step`                             |

`unique_child_id` (885–897) survives but moves into `_spawn_child` (Step 4
below).

#### Replace `register_tools` (746–751)

```python
def register_tools(self, runtime: Runtime | None = None) -> None:
    runtime = runtime or self.runtime
    runtime.inject("OrphanedDelegatesError", OrphanedDelegatesError)
    runtime.install_preamble()
    runtime.register_tool(self._spawn_child, core=True)
```

#### Replace `_record_state` (485–495) → `append_node`

Move it out of the class. New module-level function at the top of the file
(after the `RLMConfig` block):

```python
def append_node(session: Session, graph: Graph, node: Node) -> Node:
    next_seq = (graph.states[-1].seq + 1) if graph.states else 0
    fields = node.model_dump(exclude={"id", "agent_id", "seq"}, mode="python")
    fixed = node.__class__(agent_id=graph.agent_id, seq=next_seq, **fields)
    session.write_state(fixed)
    return fixed
```

Then replace every call site (currently `self._record_state(...)`) with
`append_node(self.session, graph, ...)`. Call sites:

| Line | Call                                                          |
| ---- | ------------------------------------------------------------- |
| 306  | budget exceeded → `ResultNode`                                |
| 334  | observation reply → action / error                            |
| 376  | invalid yield → `ErrorNode`                                   |
| 395  | orphaned delegates → `ErrorNode`                              |
| 406  | action `done()` → `ResultNode`                                |
| 418  | action suspend → `SupervisingNode`                            |
| 424  | action complete → `ObservationNode`                           |
| 457  | resume → `ResumeNode`                                         |
| 469  | resume `done()` → `ResultNode`                                |
| 480  | resume re-suspend → `SupervisingNode`                         |

#### Replace `prepare_runtime` (535–553) → `inject_env`

```python
def inject_env(self, graph: Graph, node: Node) -> Runtime:
    runtime = self.runtime_for(graph.runtime)
    runtime.inject("AGENT_ID", graph.agent_id)
    runtime.inject("DEPTH", graph.depth)
    runtime.inject("MAX_DEPTH", self.config.max_depth)
    runtime.inject("PARENT_NODE_ID", node.id)
    runtime.inject(
        "SESSION",
        SessionVariable(
            self.session, agent_id=graph.agent_id,
            node_id=node.id, branch_id=graph.branch_id,
        ),
    )
    runtime.inject("CONTEXT", ContextVariable(self.context, agent_id=graph.agent_id))
    runtime.inject("_DONE_RESULT", None)
    runtime.inject("_DELEGATED", [])
    return runtime
```

#### Add `_spawn_child` (replaces `delegate_for_step` body)

Add as a `@tool` method on `RLMFlow`. It is the actual host-side child
spawner. Takes everything as args — no `ActiveStep`:

```python
@tool("Internal: host-side child spawner. Called from REPL `delegate(...)`.")
def _spawn_child(
    self,
    parent_agent_id: str,
    parent_node_id: str,
    name: str,
    query: str,
    context: str,
    *,
    max_iterations: int | None = None,
    model: str = "default",
) -> ChildHandle | str:
    parent = self.session.load_graph().agents[parent_agent_id]
    if parent.depth >= self.config.max_depth:
        return f"[refused: max depth {self.config.max_depth}] Do this directly."
    if model not in self.llm_clients:
        keys = ", ".join(sorted(self.llm_clients))
        return f"[error: unknown model {model!r}. available: {keys}]"

    existing = {c.agent_id for c in parent.children}
    child_aid = unique_child_id(parent_agent_id, name, existing)
    self.context.write("context", context, agent_id=child_aid)
    runtime_ref = self.create_runtime_session(
        self.runtime_for(parent.runtime), agent_id=child_aid
    )
    context_hint = CONTEXT_HINT_PRESENT if context else CONTEXT_HINT_ABSENT
    child_config = {**self.child_config(parent, max_iterations), "model": model}
    child_graph = Graph(
        agent_id=child_aid,
        branch_id=parent.branch_id,
        depth=parent.depth + 1,
        query=query,
        system_prompt=self.build_system_prompt_for(
            query=query, agent_id=child_aid,
            depth=parent.depth + 1, config=child_config,
        ),
        config=child_config,
        workspace=parent.workspace,
        runtime=runtime_ref,
        model=None,
        parent_agent_id=parent.agent_id,
        parent_node_id=parent_node_id,
    )
    self.session.write_agent(child_graph)
    self.session.write_state(QueryNode(
        agent_id=child_aid, seq=0,
        content=FIRST_ACTION.format(query=query, context_hint=context_hint),
    ))
    return ChildHandle(child_aid)
```

`unique_child_id` becomes a module-level function (no `delegated` arg
needed — read children from the parent graph).

#### Rewrite `_step_action` (373–431) and `_step_supervising` (435–481)

Both lose the `with self.active_step(...)` block, `step.done_result`,
`step.delegated`. They read those from the runtime namespace via
`runtime.read(...)`. Sketch for `_step_action`:

```python
def _step_action(self, graph: Graph, action: ActionNode) -> None:
    err = check_yield_errors(action.code)
    if err:
        append_node(self.session, graph, ErrorNode(
            code=action.code, content=err, error="invalid_yield",
        ))
        return

    runtime = self.inject_env(graph, action)
    suspended, raw = self._run_code(graph, action.code)
    done_result = runtime.read("_DONE_RESULT")
    delegated = list(runtime.read("_DELEGATED") or [])

    if delegated and not suspended and done_result is None:
        msg = ORPHANED_DELEGATES.format(names=", ".join(delegated))
        base = raw if isinstance(raw, str) else ""
        output = self._execute_code(graph, f"raise OrphanedDelegatesError({msg!r})")
        content = (base + "\n\n" + output).strip()
        append_node(self.session, graph, ErrorNode(
            code=action.code, content=self.format_exec_output(content),
            error="orphaned_delegates",
        ))
        return

    if done_result is not None:
        append_node(self.session, graph, ResultNode(result=done_result.strip()))
        return

    if suspended:
        request, pre_output = raw
        append_node(self.session, graph, SupervisingNode(
            code=action.code, output=pre_output,
            waiting_on=list(request.agent_ids),
        ))
        return

    output = raw if isinstance(raw, str) else ""
    if not output.strip():
        output = "(no output)"
    append_node(self.session, graph, ObservationNode(
        code=action.code, output=output,
        content=self.format_exec_output(output),
    ))
```

`_step_supervising` mirrors this — calls `inject_env(graph, last)`, calls
`_resume_code`, reads `_DONE_RESULT`, branches on `Result | Supervising`
after writing the `ResumeNode`. No `ActiveStep`.

#### `build_messages` (578–632) — already correct

Lines 599–604 already filter `ResultNode` and `SupervisingNode` and treat
`ObservationNode`/`ResumeNode` as user messages (because `ResumeNode`
inherits from `ObservationNode`). No change needed beyond confirming this in
a test (Step 1 below).

### `rlmflow/graph.py` (no changes expected)

`ChildHandle`, `WaitRequest`, all node types stay as-is. Confirmed by the
preamble import of `ChildHandle, WaitRequest` working unchanged.

### Tests

Add or update under `/Users/shyam/Code/rlmkit/tests/`.

#### `tests/test_runtime_read.py` (new)

- `LocalRuntime` round-trip: `inject("X", 5); assert runtime.read("X") == 5`.
- `LocalRuntime`: `inject("Y", None); assert runtime.read("Y") is None`.
- `SubprocessRuntime`: same two assertions, hits the new `cmd: read`.
- `runtime.read("missing")` returns `None` (not a `KeyError`).

#### `tests/test_preamble.py` (new)

- `Runtime.install_preamble(); start_code("done('hi')"); assert
  runtime.read("_DONE_RESULT") == "hi"`.
- `start_code("yield wait()")` returns suspended with empty `agent_ids`.
- `_DELEGATED` defaults to `[]` after `install_preamble`.

#### `tests/test_rlmflow_core.py` (extend)

Add:

- two parallel agents driven through `ThreadPool` do not see each other's
  `_DONE_RESULT` or `_DELEGATED` (assert by reading both runtimes after a
  parallel `step`).
- after a `delegate`, the child's `parent_node_id` matches the action node id
  that called `delegate`.
- `runtime.read("_DELEGATED")` is reset to `[]` between two consecutive
  action steps.

The existing assertions on `ResumeNode.content` (no resume hint, no child
result), node sequences, and prompt history stay unchanged. They were
already pinned in the previous fix.

#### `tests/test_step_ordering.py` (already updated)

Confirmed sequences:

```text
query action supervising resume result
query action supervising resume action result
query action supervising resume supervising resume result
query action supervising resume action supervising resume action result
```

### File Split (last step, mechanical)

After all the above lands and tests pass:

```text
rlmflow/engine/__init__.py        re-export everything below
rlmflow/engine/preamble.py        PREAMBLE_SOURCE constant
rlmflow/engine/scheduler.py       NodeScheduler                     (rlm.py 105–142)
rlmflow/engine/nodes.py           append_node                       (rlm.py new fn)
rlmflow/engine/executor.py        ExecutionResult, _run_code,
                                  _resume_code, _execute_code,
                                  inject_env                        (rlm.py 535–574)
rlmflow/engine/stepper.py         _step_agent, _step_observation,
                                  _step_action, _step_supervising,
                                  _can_resume, _iteration_count     (rlm.py 296–516)
rlmflow/engine/messages.py        build_messages,
                                  build_system_prompt*,
                                  build_tools_section,
                                  build_status_section              (rlm.py 578–715)
rlmflow/rlm.py                    RLMFlow facade only
                                  (start, run, chat, step, terminate,
                                   _spawn_child, runtime_for,
                                   create_runtime_session,
                                   register_tools)
```

Top-level `rlmflow/__init__.py` re-exports stay identical — public surface
unchanged.

## Migration Plan

### Step 1 — Pin behavior with tests

Lock the node sequences before changing code:

```text
# done in same block after wait
query action supervising resume result

# verify on next turn
query action supervising resume action result

# multiple waits in same block
query action supervising resume supervising resume result

# multiple waits in separate blocks
query action supervising resume action supervising resume action result
```

Plus:

- `ResumeNode.content` has only `REPL output:`.
- `ResumeNode.content` does not contain child results or `resumed_from`.
- `SupervisingNode` does not appear in prompt history.
- `ResumeNode` appears as a user message in prompt history.
- variables assigned by resume are visible to later blocks.
- two agents stepped in parallel do not see each other's `_DONE_RESULT` or
  `_DELEGATED` (proven by a parallel-step test).
- `delegate` records the parent node id of the running action/supervising
  node (proven by inspecting `child.parent_node_id` after spawn).
- `runtime.read("_DONE_RESULT")` and `runtime.read("_DELEGATED")` work on the
  local runtime and on at least one remote runtime backend.

Most of these tests already exist; add the parallel-isolation test and the
namespace-based delegate test.

### Step 2 — Replace `_record_state` with `append_node`

No behavior change. Single function. All call sites use it.

### Step 3 — Introduce `ExecutionResult` and `execute(...)`

Normalize start and resume to one shape. Removes `(suspended, raw)`
unpacking from the transition methods.

### Step 4 — Move core tools into the REPL namespace

- add `Runtime.read(name)` and a `cmd: read` handler in `REPL.handle`;
- add a one-time preamble exec that defines `AGENT_ID`, `DEPTH`, `MAX_DEPTH`,
  `PARENT_NODE_ID`, `_DONE_RESULT`, `_DELEGATED`, and the `done`, `wait`,
  `delegate` functions in the REPL namespace;
- register a single host proxy `_spawn_child(parent_agent_id, parent_node_id,
  name, query, context, **opts)` for the host work `delegate` needs done;
- replace `prepare_runtime` with `inject_env(runtime, graph, node)` that resets
  the per-execution names before each `execute(...)` call.

Delete:

- `ActiveStep`,
- `active_step`,
- `bind_step_tools`,
- `done_for_step`, `delegate_for_step`, `wait_for_step`,
- the placeholder `@tool def done/delegate/wait(self, ...)` stubs.

`@tool` metadata for prompt docs lives on the REPL preamble functions (or a
small registry alongside it) — not on host placeholders.

### Step 5 — Extract `NodeStepper` and `MessageBuilder`

`NodeStepper` owns transitions. `MessageBuilder` owns prompt assembly with
explicit node filtering (skip `SupervisingNode`, skip `ResultNode`).

### Step 6 — Move files

Only after tests still pass:

```text
rlmflow/engine/scheduler.py   NodeScheduler
rlmflow/engine/nodes.py       append_node
rlmflow/engine/preamble.py    REPL preamble source + _spawn_child host proxy
rlmflow/engine/executor.py    execute, ExecutionResult, inject_env
rlmflow/engine/stepper.py     NodeStepper
rlmflow/engine/messages.py    MessageBuilder
rlmflow/rlm.py                RLMFlow facade
```

Mechanical move only. No semantic change.

## Risks

### Generator state is not persisted across processes

`SupervisingNode` records that the graph is paused, but the live Python
generator only exists in the running runtime. A workspace reload reconstructs
the graph but not the suspended generator.

If `current()` is `SupervisingNode` and the runtime has no live generator, the
engine should write a clear `ErrorNode` instead of silently producing
malformed state.

### REPL preamble must run before any action

The preamble defines `done`, `wait`, `delegate`, and the env names. If the
runtime is forked or cloned (`Runtime.clone()`, `Runtime.fork()`), the new
runtime must re-run the preamble. Make this part of `Runtime.__init__` /
`clone()` so it can never be skipped.

### `runtime.read(name)` is a new primitive

Every backend (local, subprocess, docker, modal) must implement it. Local is
trivial. Remote backends need one extra JSON command. Add it once and add
tests against each backend.

### Public subclass hooks

`docs/control.md` suggests overriding `step_observation`, `step_action`,
`build_messages`, `delegate_for_step`, etc. After this refactor:

- `build_messages` stays public on `RLMFlow`.
- transition hooks move to `NodeStepper` methods.
- `delegate_for_step` is removed; subclassers should override the host-side
  `_spawn_child` instead.

Update docs in the same PR that ships the refactor.

## Desired End State

```python
class RLMFlow(LLMClient):
    def start(...): ...
    def run(...): ...
    def chat(...): ...
    def step(graph):
        runnable = scheduler.runnable_agents(graph)
        pool.execute([(aid, lambda aid=aid: stepper.step(aid)) for aid in runnable])
        return session.load_graph()
```

State machine: `NodeStepper`. Tools: stable, installed once, read env vars
from the REPL. Persistence: `append_node`. Prompt: `MessageBuilder`.

If the node sequence is not obvious from reading `NodeStepper`, the refactor
is not done.
