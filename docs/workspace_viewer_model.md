# Workspace, Loading, And Viewer Model

This note proposes a cleaner persistence and visualization model for rlmflow.

The core principle: **the workspace is the run.**

Users should not have to think about separate snapshot files, viewer files, and
graph exports during normal use. A workspace should be enough to resume a run,
inspect it, and open a viewer.

## Current Problem

Today we have too many overlapping ways to explain the same run:

- `Workspace.session` is the real durable state:
  - `graph.json`
  - `session/<agent-id>/agent.json`
  - `session/<agent-id>/session.jsonl`
  - `session/<agent-id>/latest.json`
- Examples sometimes keep manual `graphs = []` lists only to feed a viewer.
- `live_view()` is useful for current-process display, but it should feel like
  the same product surface as "open this saved workspace."

This makes users ask: which artifact is authoritative? Can I reopen the viewer
later from the workspace?

The answer should be simpler:

```python
workspace = Workspace.create("runs/deep_research")
agent = RLMFlow(..., workspace=workspace)
graph = agent.run(...)

open_viewer(workspace)
```

or from the CLI:

```bash
rlmflow view runs/deep_research
```

## User-Facing Mental Model

### Workspace

A `Workspace` is the durable run directory. It stores:

- agent metadata,
- every node/state emitted by every agent,
- context payloads,
- runtime/work files,
- optional UI/export artifacts.

If a run used a workspace, that workspace is the saved run. Users should be
able to stop the process and later do:

```python
workspace = Workspace.create("runs/deep_research")
graph = workspace.load_graph()
```

No extra save artifact should be needed for normal workflows.

### Graph

A `Graph` is a snapshot/view of workspace session state.

Use it for:

- programmatic inspection,
- plotting a current run state,
- tests.

### Viewer

The viewer should open from:

- a `Workspace`,
- a workspace path,
- a `Graph`,
- or a `list[Graph]` when the caller intentionally kept in-memory history.

Preferred:

```python
from rlmflow.utils.viewer import open_viewer

open_viewer("runs/deep_research")
```

This should load the workspace session and render the run.

For normal runs:

- no manual `graphs.append(graph)`,
- no separate viewer HTML unless the user asks for an export.

If we still want time-travel/step playback, record a lightweight event stream in
the workspace session.

## Proposed Workspace Layout

Keep the existing durable layout:

```text
workspace/
  graph.json
  session/
    root/
      agent.json
      session.jsonl
      latest.json
    root.child/
      agent.json
      session.jsonl
      latest.json
  context/
  ...
```

Add one optional global event log:

```text
workspace/
  events.jsonl
```

Each event is small:

```json
{"i": 0, "agent_id": "root", "node_id": "n_...", "type": "query"}
{"i": 1, "agent_id": "root", "node_id": "n_...", "type": "action"}
{"i": 2, "agent_id": "root.child", "node_id": "n_...", "type": "query"}
```

The per-agent session logs remain the source of truth for node payloads.
`events.jsonl` only provides global ordering for replay/viewer stepping.

This gives workspace-backed interactive playback:

- current graph: `workspace.session.load_graph()`;
- historical playback: rebuild prefixes by event index;
- static export: generate HTML/GIF/frames from workspace events on demand.

If we do not care about historical playback, `events.jsonl` is not required.
The viewer can still render the latest graph from `workspace.session.load_graph()`.

## Proposed Public API

### Workspace

Add convenience methods:

```python
class Workspace:
    @classmethod
    def create(cls, path: str | Path, ...): ...

    @classmethod
    def open_path(cls, path: str | Path, *, branch_id: str = "main") -> "Workspace":
        ...

    def load_graph(self) -> Graph:
        return self.session.load_graph()

    def open_viewer(self, **kwargs):
        from rlmflow.utils.viewer import open_viewer
        return open_viewer(self, **kwargs)
```

Naming note: existing `Workspace.open(ref)` currently takes a `WorkspaceRef`.
We can either overload it to accept `str | Path | WorkspaceRef`, or add
`Workspace.from_path(...)` / `Workspace.open_path(...)`. The user-facing API
should make opening a path obvious.

### Viewer

Change viewer inputs from "graphs only" to "viewable source":

```python
ViewSource = Workspace | str | Path | Graph | list[Graph]

def open_viewer(source: ViewSource, **launch_kwargs): ...
def render_html(source: ViewSource, **kwargs) -> str: ...
def save_html(source: ViewSource, path: str | Path, **kwargs) -> Path: ...
def save_image(source: ViewSource, path: str | Path, **kwargs) -> Path: ...
def save_gif(source: ViewSource, path: str | Path, **kwargs) -> Path: ...
```

Resolution rules:

```python
def resolve_view_source(source) -> list[Graph]:
    if isinstance(source, Workspace):
        return source.history() or [source.load_graph()]
    if isinstance(source, (str, Path)) and looks_like_workspace(source):
        workspace = Workspace.create(source)
        return workspace.history() or [workspace.load_graph()]
    if isinstance(source, Graph):
        return [source]
    if isinstance(source, list):
        return source
```

`history()` depends on whether we add `events.jsonl`. Without event history,
workspace viewing falls back to a single latest graph, which is still useful.

### CLI

Make workspace paths first-class:

```bash
rlmflow view runs/deep_research
rlmflow export runs/deep_research --format html --out viewer.html
rlmflow export runs/deep_research --format png --out graph.png
```

The CLI should teach workspace paths. Standalone graph JSON is an explicit
export/import path, not a primary workflow.

## Example Cleanup

Current example style:

```python
graph = agent.start(query)
graphs = [graph]
while not graph.finished:
    graph = agent.step(graph)
    graphs.append(graph)

save_html(graphs, workspace / "viewer.html")
```

Proposed style:

```python
graph = agent.start(query)
while not graph.finished:
    graph = agent.step(graph)

print(graph.result())
print(f"Workspace saved to {workspace.root}")
```

Optional live viewer:

```python
from rlmflow.utils.viz import live_view

with live_view() as view:
    view(graph)
    while not graph.finished:
        graph = agent.step(graph)
        view(graph)
```

Optional saved viewer:

```python
from rlmflow.utils.viewer import save_html

save_html(workspace, workspace.path("viewer.html"))
```

Optional interactive viewer after the run:

```python
from rlmflow.utils.viewer import open_viewer

open_viewer(workspace)
```

## Migration Plan

### Phase 1: Make workspace loading ergonomic

- Add `Workspace.load_graph()`.
- Add `Workspace.open_path(...)` or overload `Workspace.open(...)`.
- Add `open_viewer(workspace_or_path)` support.
- Keep existing `Graph` and `list[Graph]` support as programmatic inputs.

This alone lets users reopen a saved run from a workspace.

### Phase 2: Make examples workspace-first

- Remove manual `graphs` lists unless the example specifically demonstrates
  step playback.
- Print `Workspace saved to ...`.
- Use `open_viewer(workspace)` or `save_html(workspace, ...)`.

### Phase 3: Add event history if playback matters

If the viewer needs a time slider for workspace-backed runs, add `events.jsonl`
to `FileSession.write_state(...)`.

Implementation sketch:

```python
def write_state(self, state: Node) -> None:
    self.store.append_jsonl(f"session/{agent}/session.jsonl", state)
    self.store.append_jsonl(
        "events.jsonl",
        {
            "agent_id": state.agent_id,
            "node_id": state.id,
            "seq": state.seq,
            "type": state.type,
        },
    )
```

Then add:

```python
class Session:
    def load_history(self) -> list[Graph]: ...
```

`load_history()` can rebuild graph prefixes from the event stream.

## Design Decisions

### What should users save?

The workspace directory.

That is the unit users should zip, move, upload, reopen, view, fork, and resume.

## Target User Stories

### Reopen a finished run

```python
workspace = Workspace.open_path("runs/deep_research")
graph = workspace.load_graph()
print(graph.result())
open_viewer(workspace)
```

### Continue an interrupted run

```python
workspace = Workspace.open_path("runs/deep_research")
agent = RLMFlow(..., workspace=workspace)
graph = workspace.load_graph()

while not graph.finished:
    graph = agent.step(graph)
```

### Export a viewer

```python
save_html("runs/deep_research", "viewer.html")
```

### Run with live visualization

```python
graph = agent.start(query)
with live_view() as view:
    view(graph)
    while not graph.finished:
        graph = agent.step(graph)
        view(graph)
```

The workspace is the saved run. No separate artifact unless explicitly exporting.

