# REPL `yield` semantics

This doc spells out exactly when a `yield` in an agent's REPL block
suspends the agent versus when it behaves like a normal Python yield.
Short version:

> The engine **only** intercepts top-level `yield wait(*handles)`
> values. Every other `yield` — including nested generators, generator
> expressions, helpers consumed inside the block — is plain Python.

## The protocol

Each REPL block is wrapped in a synthetic generator function
`__rlm_gen__` so the engine can drive it via `send()`. The engine
sends values into the generator and decides what to do with each
yielded value:

| What's yielded                          | Engine reaction                                    |
|-----------------------------------------|----------------------------------------------------|
| `wait(handle, …)` → `WaitRequest`       | **Suspend** the agent until those children settle. |
| Anything else (any value, or bare yield)| **Pump through.** Send `None` back, keep advancing.|
| `StopIteration`                         | Block done; return captured stdout.                |

The `WaitRequest` returned by `wait(...)` carries the child agent IDs
the engine needs to schedule on. Without it the engine has no
suspension target — so any other yielded value is treated like a
normal Python generator yield: discard the value, resume immediately.

## What this means in practice

### ✅ Allowed (no suspension)

Plain Python that happens to use `yield`:

```python
# Helper generator defined and consumed at the same level.
def squares(n):
    for i in range(n):
        yield i * i

print(list(squares(5)))                # [0, 1, 4, 9, 16]
print(sum(squares(100)))               # 328350

# Generator expressions
total = sum(x * x for x in range(100)) # works

# yield from inside a helper
def chained():
    yield from range(3)
    yield from (10, 20)

print(list(chained()))                 # [0, 1, 2, 10, 20]
```

These never suspend the agent. The wrapping decision is based on
**top-level** yields only — yields inside `def`, `async def`,
`lambda`, or `class` bodies don't count.

A top-level yield of a non-Wait value also doesn't suspend — the
engine just pumps the generator past it:

```python
yield 42                              # discarded, resume
yield "hello"                         # discarded, resume
print("still running")                # this prints
```

### ✅ Allowed (suspension)

```python
h1 = delegate("worker", "...")
h2 = delegate("worker", "...")
results = yield wait(h1, h2)          # suspends here, resumes with results
```

`results` is a list of strings (the children's `done(...)` payloads).

### ❌ Doesn't do what you want

```python
yield delegate("worker", "...")       # yields a ChildHandle, not a Wait
                                      # → engine pumps past it, child runs
                                      # but you never collected the result.

yield handle                          # same problem.
```

These don't crash anymore — they're treated as plain non-Wait yields
and silently pump through. But you also won't get a result back.
**Use `yield wait(handle)`.**

## Why this design

The engine has exactly two decisions to make about a REPL block.

### Decision 1: should we wrap the block in a generator?

The REPL builds a synthetic function around your code:

```python
def __rlm_gen__():
    <your code goes here>
```

…and calls it. If `<your code>` contains a `yield` *at the top
level*, calling `__rlm_gen__()` returns a generator object that the
engine drives via `send()`. If there's no top-level yield, calling it
just runs the code straight through. So **the wrapping decision
hinges on top-level yields, not yields anywhere in the block.**

What "top level" means by example. Forget `wait` and `delegate`
exist — top-level just means "is the `yield` keyword at the
indentation of the block itself, vs. nested inside a `def` /
`lambda` / generator expression / class method".

```python
# ── TOP-LEVEL yields ────────────────────────────────────────────────
# These count: the `yield` keyword sits at module level of the block.
# The REPL wraps the whole block in a generator function.

yield 1                              # bare value yield, top level
yield                                # bare yield, top level
x = yield 5                          # top level (yield as expression)
for i in range(3):
    yield i                          # still top level — the loop is
                                     # at module level, not inside a def

# ── NOT top level ───────────────────────────────────────────────────
# These don't count: the `yield` belongs to a nested scope, so the
# REPL won't wrap the block. `def squares` is just a normal Python
# generator; calling list(squares(5)) iterates it like any other.

def squares(n):
    for i in range(n):
        yield i * i                  # belongs to `squares`, not the block
print(list(squares(5)))               # [0, 1, 4, 9, 16]

# Generator expressions are syntactically a hidden function, so the
# `yield` Python emits internally is not at top level either.
xs = list(i * i for i in range(10))   # works

# Lambdas count as a function boundary too.
f = lambda: (yield 1)
print(next(f()))                      # 1

# Class method.
class Counter:
    def __iter__(self):
        for i in range(3):
            yield i
print(list(Counter()))                # [0, 1, 2]
```

All four "NOT top level" cases run as ordinary Python — the engine
doesn't know or care that there's a `yield` somewhere inside.

### What this means in the agent's graph

Every REPL block the LLM emits becomes one `ExecAction` node, which
the engine runs via `start()`. The result of running the block
controls what observation node lands next in the graph:

```
LLMOutput → ExecAction → <observation>
```

Where `<observation>` depends on whether the block suspended:

| Block did this                    | `<observation>` is               | Graph effect                                                                                  |
|-----------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------|
| Ran to completion (no yield)      | `ExecOutput` (stdout)            | Single `ExecAction → ExecOutput` pair. Agent advances to the next LLM turn.                   |
| Top-level `yield 1`, `yield`, etc.| `ExecOutput` (stdout)            | **Same as above.** Non-Wait yields are pumped through inside a single `ExecAction`; nothing extra shows up in the graph. |
| Top-level `yield wait(h1, h2)`    | `SupervisingOutput(waiting_on=[h1, h2])` | Agent suspends. Children run. When they finish, a `ResumeAction` resumes the generator and produces the next observation. |
| `done(...)` called (any yield path)| `DoneOutput`                    | Agent terminates with the result string.                                                      |

Concretely:

```python
# Block A — no yields: one ExecAction → one ExecOutput.
for i in range(3):
    print(i)

# Block B — top-level yields, none are Wait: STILL one ExecAction →
# one ExecOutput. The yields are invisible to the graph.
yield 1
yield "noise"
print("done")

# Block C — top-level yield wait(...): ExecAction → SupervisingOutput,
# waits for the child, then ResumeAction → next observation.
h = delegate("worker", "...")
result = yield wait(h)
print("got", result)

# Block D — yield inside a helper: helper is just a plain Python
# generator. One ExecAction → one ExecOutput. No suspension.
def squares(n):
    for i in range(n):
        yield i * i
print(list(squares(5)))
```

So the rule for graph-watchers: **non-Wait yields never produce a
new graph node.** They're internal to a single `ExecAction`'s
execution. Only `yield wait(...)` introduces a `SupervisingOutput`
and the eventual `ResumeAction`.

The bug we used to have: `ast.walk` was finding the yield inside
`def squares` and wrapping the block. But `__rlm_gen__` itself didn't
have a top-level yield — so calling it ran straight through and
returned `None`. The engine then did `None.agent_ids` and crashed.

The fix: `_has_top_level_yield(tree)` walks the AST but stops at
function / async function / lambda / class boundaries. Only yields
*outside* every nested function gate the wrapping.

### Decision 2: when the wrapped block yields, suspend or pump?

Once the block is wrapped, every `gen.send(...)` returns whichever
value the next top-level yield produced. The engine handles that
value like this:

```python
result = gen.send(prev_value)
while not isinstance(result, WaitRequest):
    result = gen.send(None)        # pump past non-Wait yields
suspend(result.agent_ids)          # this is the thing we know how to do
```

So:

```python
yield wait(h)         # WaitRequest → suspend on h
yield 42              # int → pump, immediately resume
yield                 # None → pump, immediately resume
yield handle          # ChildHandle → pump, immediately resume
                      #   (this is "forgot to wrap in wait(...)";
                      #    nothing breaks but no result comes back)
```

The agent only suspends on the thing the engine actually has a plan
for: a `WaitRequest`. Every other yield value is treated like a
plain Python generator yield — discard, resume, no engine
involvement.

### Net effect

The engine's special-case is exactly: "top-level
`yield wait(*handles)`". Every other shape of `yield` — bare,
valued, inside helpers, inside genexps, inside lambdas, inside
classes — is plain Python and behaves the way Python behaves.

## Implementation

- `rlmflow/runtime/repl.py::_has_top_level_yield(tree)` — walks the
  AST but skips `FunctionDef` / `AsyncFunctionDef` / `Lambda` /
  `ClassDef` bodies. Decides whether to wrap the block.
- `rlmflow/runtime/repl.py::REPL.advance()` — after `gen.send(...)`,
  loops sending `None` until either the generator yields a
  `WaitRequest` (return suspended) or raises `StopIteration` (return
  done). Non-Wait yields never reach the engine's suspension logic.

## Tests

See `tests/test_repl_yield.py` for the exhaustive matrix:

- top-level `yield wait(...)` suspends with the right `agent_ids`;
- top-level non-Wait yields (`yield`, `yield 1`, `yield handle`) do
  not suspend and do not crash;
- helper generators defined inside the block (`def squares()`,
  `yield from`, generator expressions) execute normally and produce
  expected values;
- nested-generator yields don't trigger wrapping (the block executes
  as straight code).
