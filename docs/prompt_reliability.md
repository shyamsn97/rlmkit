# Prompt Reliability Notes

This note summarizes prompt changes that would make recursive coding agents more reliable, based on the boids simulation traces.

The pattern so far is not one single bug. We have seen three related failure modes:

- The root implements every file inline instead of delegating.
- The root delegates, but passes children full file bodies to recover from context, turning them into fragile parsers.
- The root verifies incorrectly or stops with `done("Missing files: ...")` instead of repairing.

The base prompt should teach the protocol and recovery loop. The coding prompt should teach artifact ownership.

## Latest Failure: Delegation Collapse

The latest boids run did not delegate at all. The root wrote `index.html`, `styles.css`, `js/vector.js`, `js/boid.js`, and `js/main.js` directly in one large REPL block, then called `done(...)`.

That means the current prompt language is too weak:

```text
Prefer delegating naturally separable work...
Bias toward delegation for separable work...
```

For a weaker model, "prefer" and "bias" still leave inline implementation as a valid choice. Because the user asked for exactly one REPL block, the model chooses the path it can complete in one block: write all files itself. If we want delegation in examples like boids, the prompt needs behavioral triggers, not just preference language.

## General Delegation Behaviors

The goal is not a boids-specific instruction. The prompt should teach general habits that make delegation the obvious move when a task has separable ownership.

Useful behavior patterns:

- **Trigger rules:** Give the model a concrete condition where delegation should happen.
- **Role split:** Define what the parent is allowed to own and what children should own.
- **Completion gate:** Prevent the parent from calling `done(...)` until delegated artifacts are verified.
- **Anti-pattern callout:** Name the bad behavior directly: writing all separable files inline is a delegation failure.
- **Repair loop:** If delegation produces missing/broken work, retry the responsible child instead of restarting or finishing.
- **Self-audit:** Before writing code, make the parent answer "what can be delegated?" in its own manifest.

Brainstormed prompt lines:

```text
Before implementing, identify separable ownership boundaries. If two or more
parts can be written or checked independently, delegate those parts.
```

```text
For multi-artifact work, the parent coordinates. It writes the manifest,
shared contract, and verification. Children write independently-owned
artifacts or checks.
```

```text
Do not satisfy a clearly multi-file request by writing every file inline when
recursion depth is available. Delegation is part of the expected solution
strategy for separable work.
```

```text
If the task has three or more separable files/components/checks, delegate at
least two of them before implementing the rest directly.
```

```text
Prefer a small parent block that delegates, waits, then verifies over one
large parent block that writes every artifact itself.
```

```text
When a child owns a file/component, pass the path, interface, constraints, and
acceptance checks. Do not pass a complete file body unless the child task is
explicit copy/transform work.
```

```text
After `await launch_subagents([...])`, verify child-owned outputs from disk. Missing or
broken outputs should trigger a targeted child repair, not parent takeover of
the whole task.
```

These are intentionally more directive than the base prompt. The base prompt can keep the general "delegate separable work" rule; the coding prompt should carry the stronger multi-artifact behavior because the coding agent is explicitly demonstrating recursive implementation.

## Why It Still Fails

The later boids runs prove this is not only a missing prompt sentence. The root
system prompt included:

```text
For multi-file/component work with depth available, the first REPL block should
create a contract, delegate owned pieces, and end at `await launch_subagents([...])`; do
not write all artifact files inline.
```

The model still wrote every file inline. That suggests a few overlapping causes.

### 1. The Model Optimizes For A Complete One-Shot Answer

The old user query wording ended with:

```text
Respond with exactly one ```repl``` code block.
```

That was meant to mean "one code block per assistant turn", but a weaker model
could read it as "finish the whole job in this one block." Once it started
writing file bodies, the shortest path to success was to keep writing more
bodies.

Possible prompt fix:

```text
Use exactly one ```repl``` block per assistant message.
```

This belongs in the base REPL section, because it clarifies the protocol.

### 2. "Should" Still Leaves Inline As A Choice

Even direct text like "should create a contract, delegate..." did not constrain
the action. The model treated the instruction as strategy advice, not the main
shape of the solution.

Possible clearer wording:

```text
For multi-file/component work with depth available, start by delegating owned
pieces. The parent should write the contract and verification, not all final
file bodies.
```

This keeps the instruction behavioral instead of adding enforcement logic.

### 3. The Example Competes With A Strong Inline Pattern

The model knows a very common solution pattern for simple browser apps: put
HTML/CSS/JS in Python strings and call `write_file(...)` repeatedly. That path
is familiar, compact, and likely to finish in one model turn.

Delegation asks the model to do extra planning before the obvious implementation
move. If the example prompt does not make delegation feel like the natural
solution shape, the model falls back to the memorized file-dump pattern.

### 4. The Example Is Too Abstract

The base example uses generic `artifact/part_a.txt` files. It teaches the
mechanics of delegation, but not the common coding pattern:

- parent defines imports/exports and paths
- children write concrete files
- parent verifies integration

A more concrete coding example may help more than more rules:

```python
files = [
    ("html", "index.html", "entry point that loads styles.css and js/main.js"),
    ("model", "js/model.js", "exports the data/model API"),
    ("main", "js/main.js", "imports model API and starts the app"),
]
contract = '''
Shared interface:
- Browser app with index.html as entry point.
- Use ES modules consistently.
- js/model.js exports createState().
- js/main.js imports createState() and starts the runtime.
- Each child writes only its assigned path and verifies with read_file().
'''
results = await launch_subagents([
    {"name": name, "query": f"Write only {path}: {task}", "context": contract}
    for name, path, task in files
])
```

This probably belongs in the coding example docs/notebook rather than the base
prompt, to keep the base small.

### 5. The Task Itself Invites File Dumping

The boids prompt asks for a finished artifact with several obvious files. The
model has memorized this shape: emit HTML/CSS/JS strings and call `write_file`.
Delegation is a meta-strategy that competes with a very familiar coding
completion pattern.

Possible demo-task fix:

```text
Use recursion for this task: the root must delegate at least the HTML/CSS,
boid/model logic, and simulation loop as separate child tasks before final
verification.
```

This is not a general solution, but it is useful for a demo whose purpose is to
show recursive coding.

## Stronger Options

### Option A: Clarify The One-Block Protocol

Add a short clarification to the REPL/base prompt:

```text
Use exactly one ```repl``` block per assistant message.
```

This addresses the direct ambiguity in the user's task wrapper.

### Option B: Make The Coding Example More Concrete

Replace the generic `artifact/part_a.txt` example with a small coding-shaped
example. The model may follow examples more reliably than abstract rules:

```python
files = [
    ("html", "index.html", "entry point that loads styles.css and js/main.js"),
    ("styles", "styles.css", "full viewport app styling"),
    ("model", "js/model.js", "exports createState()"),
    ("main", "js/main.js", "imports createState() and starts the runtime"),
]

contract = '''
Shared interface:
- Browser app with index.html as entry point.
- Use ES modules consistently.
- js/model.js exports createState().
- js/main.js imports createState().
- Each child writes only its assigned path and verifies with read_file().
'''
results = await launch_subagents([
    {"name": name, "query": f"Write only {path}: {task}", "context": contract}
    for name, path, task in files
])
```

### Option C: Add A `build_file_specs(...)` Helper

Instead of asking the model to hand-roll the spec list, give it a small
helper in the coding example that returns `launch_subagents` specs:

```python
specs = build_file_specs(files, contract)   # returns a list of launch_subagents specs
results = await launch_subagents(specs)
```

This reduces the friction of delegation. The model often avoids delegation
because the inline path is simpler Python.

### Option D: Change The Demo Query

For notebook/demo reliability, make the task explicitly about recursion:

```text
Use child agents for separable files. The root should not write the final HTML,
CSS, or JS file bodies directly; it should delegate those pieces and verify
after waiting.
```

This proves the system feature, but it is less satisfying as a general-purpose
coding-agent prompt.

## Recommendation

Prefer prompt and example changes:

1. Clarify the turn protocol: use exactly one ```repl``` block per assistant
   message.
2. Make the coding example's delegation pattern concrete, not `part_a.txt`.
3. Keep the coding prompt short, but make the first-turn shape explicit:
   contract, delegate, `await launch_subagents([...])`, then verify on resume.
4. Consider a small `delegate_files(...)` helper so delegation is easier than
   writing every file body inline.

## Parent vs Child Ownership

The parent should not pre-write full file bodies and pass them to children. That produces "delegation theater": the parent did the implementation, and the child only copies or extracts text.

Use this split instead:

- Parent owns the manifest: paths, interfaces, dependencies, acceptance checks.
- Children own file/component bodies.
- Parent waits, reads files from disk, checks integration, and repairs only failed pieces.

Example child query shape:

```text
Write only `js/boid.js`.

Contract:
- export class `Boid`
- import `Vec` from `./vector.js`
- keep speed magnitude constant independent of size
- expose `flock(...)`, `update(...)`, and `draw(...)`
- write the file, read it back, and return JSON:
  {"path": "js/boid.js", "status": "wrote", "checked": true}
```

This gives the child real ownership without making the parent depend on a child parsing a giant context blob.

## Example Prompt Pattern

The multi-file example in the base prompt should show the pattern we actually want:

```python
files = [
    ("html", "index.html", "canvas shell that loads CSS and js/main.js"),
    ("styles", "styles.css", "full viewport canvas styling"),
    ("vector", "js/vector.js", "2D vector helper API"),
    ("boid", "js/boid.js", "Boid class using Vec"),
    ("main", "js/main.js", "simulation loop and canvas rendering"),
]

contract = '''
Shared contract:
- This is one runnable browser artifact.
- `index.html` is the entry point.
- JS modules must agree on imports/exports.
- Each child writes exactly its assigned path, reads it back, and returns JSON.
'''

results = await launch_subagents([
    {"name": name, "query": f"Write only {path}: {task}.", "context": contract}
    for name, path, task in files
])
```

Then the resumed block should verify exact paths with `read_file(path)`.

## Do Not Finish With Known Missing Requirements

The current example pattern includes:

```python
if missing:
    done("Missing files: " + ", ".join(missing))
```

That teaches early termination. Replace it with targeted repair:

```python
missing = []
for path in expected:
    try:
        assert read_file(path).strip()
    except Exception:
        missing.append(path)

if missing:
    results = await launch_subagents([
        {
            "name": path_to_child[path],
            "query": f"Repair only {path}. It is missing or empty. Write it, read it back, then return JSON.",
            "context": contract,
        }
        for path in missing
    ])
```

Only call `done(...)` after the expected contract passes, or when reporting a true unrecoverable blocker.

## Verification Guidance

For expected artifacts, verify exact paths:

```python
files = {path: read_file(path) for path in expected}
assert all(src.strip() for src in files.values())
```

Avoid bare `list_files()` for verification. It is useful for discovery, but exact-path reads are harder for the model to misuse.

## Exact Additions

Add only a few lines. The best next prompt change is not another long rubric; it is one hard delegation trigger plus one hard recovery rule.

### Add To The Base Prompt

Add this under the launcher wait:

```text
- After a wait, inspect child results and verify expected outputs directly.
  Missing or broken outputs should trigger targeted repair of those pieces,
  not a full restart.
```

Add this under `done(answer)`:

```text
- Do not call `done(...)` with known missing or broken requirements unless
  you are reporting an unrecoverable blocker.
```

Also fix the base multi-file example. Replace its missing-file branch with a repair branch. Do not show `done("Missing files: ...")` as the example path.

### Add To The Coding Prompt

Replace the softer delegation bullets with this exact wording:

```text
- **Plan the contract.** For non-trivial tasks, sketch owners, shared
  interfaces, and acceptance checks before writing.
- **Delegate separable work.** For multi-file/component work with depth
  available, the first REPL block should create a contract, delegate owned
  pieces, and end at `await launch_subagents([...])`; do not write all artifact files
  inline.
- **Pass contracts, not bodies.** Give children paths, interfaces,
  constraints, and checks; avoid full file bodies unless copying or
  transforming exact text.
- **Honor and verify the artifact.** Preserve the requested
  runtime/API/behavior; after children finish, read files from disk, verify
  shared interfaces, and run a real check when possible.
- **Repair by owner.** Retry only the responsible child/path for missing or
  broken outputs before `done()`.
```

This keeps the coding extension short enough for examples while preserving the important behavior: parent contracts, child ownership, interface verification, and targeted repair.

## Base Prompt Changes

Keep the base prompt short:

```text
- If verification fails, repair the specific failure before `done(...)`;
  only report a blocker when no useful repair step remains.
```

After the launcher wait:

```text
- After a wait, inspect child results and verify expected outputs directly.
  Missing/broken outputs should trigger targeted repair, not a full restart.
```

Under `done(...)`:

```text
- Do not call `done(...)` with known unmet requirements unless you are
  reporting an unrecoverable blocker.
```

## Coding Prompt Changes

The coding prompt should be more directive than the base prompt:

```text
- Plan the contract: owners, shared interfaces, and checks.
- For multi-file/component work, first create a contract, delegate owned
  pieces, and end at `await launch_subagents([...])`.
- Pass contracts to children, not full file bodies.
- Verify the artifact from disk, including shared interfaces and behavior.
- Repair only the responsible child/path before final `done()`.
```

## Non-Prompt Improvements

Prompting may still be unreliable if inline completion is always available. A stronger runtime option would help:

- Add a coding-agent mode or config that requires delegation for multi-file tasks when depth is available.
- Add a helper like `verify_files(paths)` so models do not hand-roll missing-file checks.
- Consider changing `list_files()` default from `"*.txt"` to `"*"`, because the current default is surprising for coding artifacts.

## Recommended Next Patch

1. Strengthen the coding prompt with an explicit multi-file delegation trigger.
2. Replace the base prompt example's missing-file `done(...)` branch with targeted repair.
3. Change examples to verify expected files with `read_file(path)`.
4. Keep the base prompt general, but add one rule: failed verification means repair before final `done(...)`.
