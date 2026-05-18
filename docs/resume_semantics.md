# Resume Semantics

This note defines what `yield wait(...)` and `ResumeNode` should mean in the
runtime.

## Mental Model

`yield wait(...)` is a Python generator suspension point. It is not a request to
copy child outputs into the next LLM prompt.

For code like:

```python
handles = [delegate("a", query_a, context_a), delegate("b", query_b, context_b)]
results = yield wait(*handles)
print(len(results))
```

the intended flow is:

1. The parent REPL runs until `yield wait(*handles)`.
2. The generator suspends. The assignment to `results` has not happened yet.
3. The scheduler runs the child agents until they finish.
4. The runtime resumes the same parent generator and sends the child result list
   back into the suspended `yield`.
5. The line becomes equivalent to `results = child_results`.
6. The same stateful REPL continues and runs `print(len(results))`.
7. If the generator ends without `done(...)` or another `yield wait(...)`, the
   engine records a `ResumeNode` trace marker and the next LLM turn continues
   in the same stateful REPL.

The important point: child outputs already flow through the REPL value returned
from `yield wait(...)`. The REPL is stateful, so variables assigned after resume
are available to the next code block.

## Full Multi-Delegation Walkthrough

Assume the root agent receives:

```text
Research whether SDSK can reach $2000 by the end of 2026.
```

### Step 0: Root query

The graph starts with one root agent and one node:

```text
root
  [0] QueryNode
      content = "Query: Research whether SDSK can reach $2000..."
```

No code has run yet.

### Step 1: Root delegates and waits

The LLM writes one REPL block:

```python
contract = "Return JSON with claims, evidence, sources, contradictions."
jobs = [
    ("identity", "Identify the listed security and ticker.", contract),
    ("valuation", "Find price, market cap, valuation multiples.", contract),
    ("fundamentals", "Research growth, revenue, margins, balance sheet.", contract),
    ("analyst", "Collect analyst targets and consensus.", contract),
]

handles = [
    delegate(name, query, context, model="fast")
    for name, query, context in jobs
]
results = yield wait(*handles)
print(f"got {len(results)} child results")
```

Runtime execution:

1. `delegate(...)` creates four child agents.
2. `yield wait(*handles)` suspends the root generator.
3. The assignment to `results` has not happened yet.
4. The root state becomes `SupervisingNode(waiting_on=[...])`.

Node shape after this step:

```text
root
  [0] QueryNode
  [1] ActionNode
      code = "contract = ...; handles = ...; results = yield wait(*handles); ..."
  [2] SupervisingNode
      waiting_on = [
        "root.identity",
        "root.valuation",
        "root.fundamentals",
        "root.analyst",
      ]

root.identity
  [0] QueryNode

root.valuation
  [0] QueryNode

root.fundamentals
  [0] QueryNode

root.analyst
  [0] QueryNode
```

Data location at this point:

- Child outputs do not exist yet.
- Root REPL is suspended at `yield wait(*handles)`.
- `handles` exists in the root REPL.
- `results` does not exist yet because the suspended line has not received a
  value.

### Step 2: Children run

The scheduler runs each child. Each child is a normal agent with its own REPL,
context, and state log.

Example child result:

```python
done('{"topic": "identity", "claims": [...], "sources": [...]}')
```

After all children finish:

```text
root.identity
  [0] QueryNode
  [1] ActionNode
  [2] ResultNode
      result = '{"topic": "identity", ...}'

root.valuation
  [0] QueryNode
  [1] ActionNode
  [2] ResultNode
      result = '{"topic": "valuation", ...}'

root.fundamentals
  [0] QueryNode
  [1] ActionNode
  [2] ResultNode
      result = '{"topic": "fundamentals", ...}'

root.analyst
  [0] QueryNode
  [1] ActionNode
  [2] ResultNode
      result = '{"topic": "analyst", ...}'
```

Data location at this point:

- Child outputs live in each child's `ResultNode.result`.
- Root is still suspended.
- Root still has no `results` variable.

### Step 3: Root REPL resumes inline

The scheduler sees all children in `waiting_on` are finished. It collects:

```python
child_results = [
    graph["root.identity"].result(),
    graph["root.valuation"].result(),
    graph["root.fundamentals"].result(),
    graph["root.analyst"].result(),
]
```

Then it resumes the suspended root generator by sending that list back into the
`yield wait(...)` expression.

The suspended line:

```python
results = yield wait(*handles)
```

continues as if it were:

```python
results = child_results
```

Then the same root REPL continues:

```python
print(f"got {len(results)} child results")
```

Output is:

```text
got 4 child results
```

The root generator reaches the end of the code block without `done(...)` and
without another `yield wait(...)`. So the agent is not finished, and the engine
records a `ResumeNode`.

Preferred node shape:

```text
root
  [0] QueryNode
  [1] ActionNode
  [2] SupervisingNode
  [3] ResumeNode
      resumed_from = [
        "root.identity",
        "root.valuation",
        "root.fundamentals",
        "root.analyst",
      ]
      output = "got 4 child results\n"  # trace/UI only, not prompt input
      content = ""
```

Data location after resume:

- `results` exists in the root REPL.
- `results[0]` is the identity child's JSON string.
- `results[1]` is the valuation child's JSON string.
- `results[2]` is the fundamentals child's JSON string.
- `results[3]` is the analyst child's JSON string.
- `ResumeNode.output`, if kept, is trace/UI-only stdout/stderr from resumed
  root code.
- `ResumeNode.content` is empty and is not serialized into the prompt.

### Step 4: Root gets the next LLM turn

The next LLM turn should not receive child result blobs, wait-completed text, or
resumed stdout in the transcript. It only needs the normal continue instruction.
The LLM knows it is in the same stateful REPL, so it can inspect `results`
directly.

The LLM writes:

```python
import json

sections = [json.loads(r) for r in results]
assert len(sections) == 4
assert all("claims" in s for s in sections)

report = synthesize_report(sections)
done(report)
```

Node shape after finalization:

```text
root
  [0] QueryNode
  [1] ActionNode
  [2] SupervisingNode
  [3] ResumeNode
  [4] ActionNode
      code = "import json; sections = ...; done(report)"
  [5] ResultNode
      result = "<final report>"
```

### Complete graph

The complete run is:

```text
root
  [0] QueryNode
  [1] ActionNode          # delegates identity, valuation, fundamentals, analyst
  [2] SupervisingNode     # waiting_on child agent ids
  [3] ResumeNode          # resumed root code ran; no child result injection
  [4] ActionNode          # parses stateful `results`, verifies, synthesizes
  [5] ResultNode

root.identity
  [0] QueryNode
  [1] ActionNode
  [2] ResultNode

root.valuation
  [0] QueryNode
  [1] ActionNode
  [2] ResultNode

root.fundamentals
  [0] QueryNode
  [1] ActionNode
  [2] ResultNode

root.analyst
  [0] QueryNode
  [1] ActionNode
  [2] ResultNode
```

There is exactly one data path from children back to parent code:

```text
Child ResultNode.result
  -> collected by _step_supervising
  -> sent into _resume_code(graph, results)
  -> assigned to parent REPL variable `results`
```

`ResumeNode` is not part of that data path. It is trace/UI state.

## What `ResumeNode` Is

`ResumeNode` is not what resumes the REPL. The REPL has already resumed by the
time the node is written.

`ResumeNode` is graph/session metadata: "the suspended generator resumed, ran
some more code, and did not finish the agent."

It should not be used as a prompt payload. The next LLM turn is triggered by the
normal step loop and continue instruction, not by injecting resume text.

## Previous Problem

The old resume observation included full child results:

```text
Children finished:
  root.a: <full child result>
  root.b: <full child result>

Generator resumed. Output:
...
```

That duplicates data already returned by `yield wait(...)`. It has bad effects:

- Bloats the next prompt with potentially large child outputs.
- Creates two sources of truth: REPL variables and transcript injection.
- Encourages the model to synthesize from injected text instead of inspecting
  the stateful REPL values it explicitly assigned.
- Makes deep research and other fan-out tasks look like context stuffing rather
  than structured REPL control flow.

## Proposed Semantics

Child results should be returned only through the generator resume value:

```python
results = yield wait(*handles)
```

`ResumeNode` should be structured trace/UI state, not a prompt payload. The LLM
does not need child outputs, child IDs, stdout, or a synthetic "wait completed"
message in its transcript. The next LLM turn can inspect the stateful REPL
variables that were assigned by resumed code.

No child results, resume status text, or resumed stdout should be injected into
`ResumeNode.content`.

## Actual Changes Needed

1. **Keep the real data path unchanged.**

   `RLMFlow._step_supervising` should still collect child results and pass them
   into the suspended parent generator:

   ```python
   results = [
       child.result() if isinstance(child.current(), ResultNode) else ""
       for child in children
   ]
   suspended, raw = self._resume_code(graph, results)
   ```

   This is the only parent-facing data path for child outputs.

2. **Stop building prompt text from child results.**

   In `RLMFlow._step_supervising`, delete the `child_summary` construction and
   stop writing child results into `ResumeNode.content`.

   Target shape:

   ```python
   self._record_state(
       graph,
       ResumeNode(
           output=output,  # optional trace/UI stdout only
           content="",
           resumed_from=list(last.waiting_on),
       ),
   )
   ```

3. **Do not serialize `ResumeNode` into chat messages.**

   In `RLMFlow.build_messages`, handle `ResumeNode` before the generic
   `ObservationNode` branch and skip it:

   ```python
   if isinstance(state, ResumeNode):
       continue
   if isinstance(state, ObservationNode):
       msgs.append({"role": "user", "content": state.content})
   ```

   The follow-up LLM turn still happens because `build_messages(...)` appends
   the normal continue instruction when the agent is unfinished.

4. **Update prompt wording and examples.**

   The prompt should not say the next turn sees `Children finished...` or
   `Generator resumed...`. It should say variables persist across turns and the
   results assigned from `yield wait(...)` are available in the REPL.

   Good example:

   ```python
   # Block 1
   results = yield wait(*handles)
   ```

   ```python
   # Block 2
   parsed = [json.loads(r) for r in results]
   done(synthesize(parsed))
   ```

5. **Update viewers/traces only if they depend on `content`.**

   Graph viewers should display `ResumeNode.resumed_from` and optionally
   `ResumeNode.output`. They should not require `content` to contain a synthetic
   narrative.

6. **Add tests for no prompt injection.**

   Tests should assert:

   - Child result strings are passed into the parent REPL variable assigned from
     `yield wait(...)`.
   - `ResumeNode.content == ""`.
   - `build_messages(...)` does not include child result strings from a
     `ResumeNode`.
   - The next action can still reference `results` from the stateful REPL.

## Implementation Sketch

In `RLMFlow._step_supervising`, keep this behavior:

```python
results = [
    child.result() if isinstance(child.current(), ResultNode) else ""
    for child in children
]

with self.active_step(graph, last) as step:
    suspended, raw = self._resume_code(graph, results)
```

That is the real data path. Do not change it.

Change only how the resumed fall-through is represented. The node can still
carry structured fields for graph/viewer/debugging, but `ResumeNode` should not
be used as a data channel into the prompt.

Old prompt-injecting shape:

```python
content = (
    f"Children finished:\n{child_summary}\n\n"
    f"Generator resumed. Output:\n{output or '(no output)'}"
)
```

Preferred shape:

```python
self._record_state(
    graph,
    ResumeNode(
        output=output,
        content="",
        resumed_from=list(last.waiting_on),
    ),
)
```

Then update message construction so `ResumeNode` is not serialized back into the
LLM transcript as an observation. It is enough for `build_messages(...)` to append
the normal continue instruction. The next code block can inspect REPL state
directly:

```python
print(len(results))
parsed = [json.loads(r) for r in results]
```

This makes `ResumeNode` cosmetic/structural: useful for the graph, viewer,
session log, and debugging, but not a hidden prompt injection channel.

## Prompt Guidance

Prompt examples should model this pattern:

```python
# Block 1
results = yield wait(*handles)
print(f"got {len(results)} results")
```

```python
# Block 2
parsed = [json.loads(r) for r in results]
assert parsed
done(synthesize(parsed))
```

The first block may do lightweight assignment or logging after the wait. The
second block does reasoning, validation, and final synthesis using the stateful
REPL variables from the resumed generator.

Avoid examples that rely on child results appearing in the observation text.
That is an implementation detail we should remove.

