# Observability

Everything you need to debug a run lives in the typed node graph.

## Node Fields

```python
node.id
node.type                 # query | action | observation | supervising | resume | result | error
node.agent_id             # "root", "root.search_0", ...
node.depth                # root = 0, children = 1, ...
node.children             # successor ids or live child nodes
node.query                # original task for this agent
node.system_prompt        # prompt snapshot for this agent
node.config               # node-local max_depth, max_iterations, model, ...
node.workspace            # serializable WorkspaceRef
node.runtime              # RuntimeRef for REPL continuity
node.total_input_tokens
node.total_output_tokens
node.tree_usage()         # (in, out) across the subtree
node.tree_tokens          # subtree total
node.tree()               # ASCII tree render
```

Action nodes additionally store `reply`, `code`, `model`, and turn token usage.
Result nodes store `result`. Supervising nodes store `waiting_on` and child
leaves.

## Node Types

Each `step()` returns a new node:

| Node | Meaning |
|---|---|
| `QueryNode` | First user/task input for an agent. |
| `ActionNode` | Raw LLM reply plus extracted REPL code. |
| `ObservationNode` | REPL output after code execution. |
| `SupervisingNode` | Action suspended on `yield wait(...)`. |
| `ResumeNode` | Parent resumed after children finished. |
| `ErrorNode` | Failure observation. |
| `ResultNode` | Terminal answer from `done(...)`. |

## Save & load

A single node checkpoint — save, load, and continue later:

```python
node.save("checkpoint.json")
node = Node.load("checkpoint.json")
```

A full run — every step in order:

```python
from rlmflow.utils.trace import save_trace, load_trace

save_trace(states, "traces/run1")
save_trace(states, "traces/run1", metadata={"model": "gpt-5"})

t = load_trace("traces/run1")
t.states          # list[Node] — typed events preserved
t.metadata
```

Traces are plain JSON — grep-able, diff-able.

## Session And Context

`Workspace` separates message history from task payloads:

```text
workspace/
  session/
    nodes.jsonl
    agents/root.json
    agents/root.child.json
  context/
    context.txt
  trace/
```

`Workspace.session` persists the node/message graph. `Workspace.context`
persists optional payload data exposed in the REPL as `CONTEXT`.

Messages are derived from `Session.chain_to(node)`, not stored as a second
source of truth.

## Live terminal

```python
from rlmflow.utils.viz import live
for node in live(agent, agent.start(query)):
    pass
```

Or just `print(node.tree())` in a step loop.

## Gantt swimlane

One row per agent, one column per step, colored by node type. Makes
parallelism and critical path obvious at a glance.

```python
from rlmflow.utils.viz import gantt, gantt_html

gantt(states)                            # print to terminal (Rich)
Path("run.html").write_text(gantt_html(states, title="run 1"))
```

## Topology exports

Static renders of the tree, for READMEs, issues, and post-mortems.

```python
from rlmflow.utils.export import to_mermaid, to_dot

print(to_mermaid(states[-1]))            # stateDiagram-v2 — paste into GitHub
Path("run.dot").write_text(to_dot(states[-1]))
# $ dot -Tsvg run.dot -o run.svg
```

## Viewer

```python
from rlmflow.utils.viewer import open_viewer

open_viewer(states)                      # from an in-memory run
```

Requires `rlmflow[viewer]`.

## CLI

The same helpers are reachable from a shell. `view` and `render`
auto-detect trace directories, `trace.json` files, and single state
checkpoints.

```
rlmflow view   traces/run1/
rlmflow view   workspace/checkpoint.json --port 7861
rlmflow render traces/run1/   -f gantt-html -o run1.html
rlmflow render checkpoint.json -f mermaid          # stdout
rlmflow render checkpoint.json -f dot -o graph.dot
rlmflow render checkpoint.json -f tree
rlmflow version
```
