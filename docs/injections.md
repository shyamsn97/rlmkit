# Node Injection

Node injection lets an external controller edit a running agent graph by
appending typed nodes, then committing that graph through `agent.step(graph)`.
It is useful for budget controls, human/controller feedback, forced finalization,
and repair nudges that should be represented in the same trace as normal model
and runtime events.

The primitive is:

```python
graph2 = graph.inject(target="root", node=ExecOutput(...))
graph3 = agent.step(graph2)
```

`graph.inject(...)` returns a new graph value. It does not write to the active
session. `agent.step(graph2)` is the commit point: it persists the appended
nodes as ordinary graph states, updates the transcript/message projection, and
then continues normal scheduling.

For a runnable offline demo, see [`examples/injections.py`](../examples/injections.py).

## Inject A Controller Observation

Inject an `ExecOutput` when you want the next LLM turn to see controller-authored
feedback without pretending the model wrote a REPL block.

```python
from rlmflow import ExecOutput

graph = agent.start("Wait for a controller note, then finish.")

graph = graph.inject(
    target="root",
    node=ExecOutput(
        output="Injected controller observation: submit your final answer now.",
        content="Injected controller observation: submit your final answer now.",
    ),
)

graph = agent.step(graph)  # persists the observation, then calls the LLM
```

If several observations are adjacent, `build_messages()` coalesces them into one
user-role message so providers with strict role alternation still accept the
prompt.

## Inject An Action

Inject an `ExecAction` when the controller wants to run a specific REPL action
through the normal runtime path. The common case is immediate finalization:

```python
from rlmflow import ExecAction

graph = graph.inject(
    target="root.worker",
    node=ExecAction(code='done("message budget exhausted; best available answer")'),
)

graph = agent.step(graph)  # executes the injected action and writes DoneOutput
```

The resulting `DoneOutput` is produced by the usual runtime machinery, so it is
persisted and visible like any model-generated `done(...)`.

## Targeting

`target` may be:

```python
graph.inject(target="root.worker", node=...)      # exact agent id
graph.inject(target=r"root\.chunk_\d+$", node=...) # regex over agent ids
graph.inject(target=lambda g: g.leaves(), node=...) # callable returning agents
```

Useful traversal helpers include `graph.leaves()`, `graph.where(fn)`,
`graph.match(pattern)`, `graph.children_of(agent_id)`, and
`graph.descendants_of(agent_id)`.

## Rules

- Injection is append-only in the public API today.
- Injected nodes are stored as ordinary node rows; there is no per-node
  injection metadata.
- Do not inject into a finished agent.
- Multiple adjacent observation nodes are allowed.
- Only one pending injected action per agent is allowed; queue another action
  after the first one has been committed with `agent.step(graph)`.
- Stale graphs are rejected rather than overwriting newer session state.
