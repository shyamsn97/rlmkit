# Observability

Everything you need to debug a run lives on `RLMState`.

## State fields

```python
state.agent_id            # "root", "root.search_0", ...
state.status              # READY | EXECUTING | SUPERVISING | FINISHED
state.iteration
state.event               # last StepEvent
state.messages            # full LLM history
state.system_prompt       # resolved prompt for the last call
state.last_reply
state.result              # set when FINISHED
state.children            # list[RLMState], recursive
state.waiting_on          # agent_ids this node is blocked on
state.total_input_tokens
state.total_output_tokens
state.total_tokens        # property — this agent only (in + out)
state.tree_usage()        # (in, out) across the subtree
state.tree_tokens         # property — subtree total (in + out)
state.tree()              # ASCII tree render
```

## Step events

Each `step()` attaches one typed event:

| Event | Emitted when |
|---|---|
| `LLMReply` | LLM returned. `text`, `code`, token counts. |
| `CodeExec` | Code block ran. `code`, `output`, `suspended`. |
| `ResumeExec` | Generator resumed after children finished. |
| `NoCodeBlock` | Reply had no ` ```repl ``` ` block. |

## Save & load

A single state tree — checkpoint, fork, rehydrate later:

```python
state.save("checkpoint.json")
state = RLMState.load("checkpoint.json")
```

A full run — every step in order:

```python
from rlmkit.utils.trace import save_trace, load_trace

save_trace(states, "traces/run1")
save_trace(states, "traces/run1", metadata={"model": "gpt-5"})

t = load_trace("traces/run1")
t.states          # list[RLMState] — typed events preserved
t.metadata
```

Per-turn queries already live on each `state.query`, so multi-turn
sessions accumulate into a single trace without losing any context.
Traces are plain JSON — grep-able, diff-able.

## Sessions

Pass `session="context"` (or a `Session` instance) and the engine
writes `state.messages` to disk after every step, using one directory
per agent. Subclass `Session` to plug in a different backend.

Traces persist **states**. Sessions persist **messages**.

## Live terminal

```python
from rlmkit.utils.viz import live
for state in live(agent, agent.start(query)):
    pass
```

Or just `print(state.tree())` in a step loop.

## Gantt swimlane

One row per agent, one column per step, colored by status. Makes
parallelism and critical path obvious at a glance.

```python
from rlmkit.utils.viz import gantt, gantt_html

gantt(states)                            # print to terminal (Rich)
Path("run.html").write_text(gantt_html(states, title="run 1"))
```

## Topology exports

Static renders of the tree, for READMEs, issues, and post-mortems.

```python
from rlmkit.utils.export import to_mermaid, to_dot

print(to_mermaid(states[-1]))            # stateDiagram-v2 — paste into GitHub
Path("run.dot").write_text(to_dot(states[-1]))
# $ dot -Tsvg run.dot -o run.svg
```

## Viewer

```python
from rlmkit.utils.viewer import view_trace, open_viewer

open_viewer(states, query=query)         # from an in-memory run
view_trace("traces/run1")                # from disk
```

Requires `rlmkit[viewer]`. Hit ▶ to auto-advance through the trace;
the speed slider controls steps/sec.

## CLI

The same helpers are reachable from a shell. `view` and `render`
auto-detect trace directories, `trace.json` files, and single state
checkpoints.

```
rlmkit view   traces/run1/
rlmkit view   workspace/checkpoint.json --port 7861
rlmkit render traces/run1/   -f gantt-html -o run1.html
rlmkit render checkpoint.json -f mermaid          # stdout
rlmkit render checkpoint.json -f dot -o graph.dot
rlmkit render checkpoint.json -f tree
rlmkit version
```
