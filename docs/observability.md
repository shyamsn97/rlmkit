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
state.total_tokens        # property
state.tree_usage()        # (in, out) across the subtree
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

## Traces

```python
from rlmkit.utils.viewer import save_trace, load_trace, view_trace

save_trace(states, "traces/run1", query=query)
states, query, meta = load_trace("traces/run1")
view_trace("traces/run1")   # requires rlmkit[viewer]
```

A trace is a JSON list of `state.model_dump()` payloads — grep-able,
diff-able.

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
