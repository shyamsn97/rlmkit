# Control

`step(state) -> state'` is the only transition. Because state is
immutable, checkpoint, fork, rewind, and intervention are all just
list operations.

## Step loop

```python
state = agent.start(query)
while not state.finished:
    state = agent.step(state)
```

`agent.run(query)` does the same thing and returns `state.result`.
`agent.chat(messages)` is the `LLMClient` interface — same loop, last
user message becomes the query.

## Checkpoint / resume

```python
state.save("ckpt.json")

state = agent.restore(RLMNode.load("ckpt.json"))
while not state.finished:
    state = agent.step(state)
```

## Fork

Keep every state in a list and resume any one of them with a fresh
agent:

```python
states = [agent.start(query)]
while not states[-1].finished:
    states.append(agent.step(states[-1]))

alt_agent = RLM(...)
alt = alt_agent.restore(states[3])   # diverge from step 3
```

## Rewind

```python
state = states[-5]   # drop five steps, keep stepping from here
```

## Intervene

Between steps, patch any field with `state.update(**changes)`:

```python
state = state.update(
    children=[c for c in state.children if c.agent_id != "root.bad_branch"],
    waiting_on=[aid for aid in state.waiting_on if aid != "root.bad_branch"],
)
```

Use this to kill runaway children, inject hints into `messages`, or
force `status=Status.FINISHED` with a preset `result`.

## Custom runtime

Subclass `Runtime` and implement two methods:

```python
class MyRuntime(Runtime):
    def send(self, msg: dict) -> None: ...
    def recv(self) -> dict: ...
```

See [`runtimes.md`](runtimes.md).

## Custom tools

```python
@runtime.tool("Search files for a regex.")
def search(pattern: str, path: str = ".") -> str:
    ...
```

Or pass a list to `runtime.register_tools([...])`.

## Custom prompt

```python
from rlmkit.prompts.default import DEFAULT_BUILDER, GUARDRAILS_TEXT

agent = RLM(..., prompt_builder=(
    DEFAULT_BUILDER
    .section("role", "You are a security auditor.", title="Role")
    .section("guardrails", GUARDRAILS_TEXT, title="Guardrails", after="recursion")
))
```

Or subclass `RLM` and override `build_system_prompt`, `build_messages`,
`extract_code`, or `create_child`.

## Custom state

```python
class ReviewState(RLMNode):
    findings: list[str] = []

class CodeReviewer(RLM):
    node_cls = ReviewState
```

The engine preserves your extra fields through `update()`.

See [`showcase.py`](../examples/showcase.py) for a runnable walkthrough
of everything above.
