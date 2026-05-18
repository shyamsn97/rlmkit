# Graph features

Tiny self-contained scripts that show what a `Graph` can do. No LLM keys
needed — every example builds its own graph by hand or runs a one-line
mock LLM, so they finish in milliseconds and can be read top-to-bottom.

| script | what it shows |
|---|---|
| `01_query.py` | flat views (`graph.nodes`, `.agents`, `.edges`), filters (`.where`, `.queries()`, `.actions()`, `.errors()`), `find()`, `tokens()`, `result()` |
| `02_navigate.py` | `graph[aid]`, dotted paths, `walk()` / `subtree()`, parent ↔ child links, `len(graph)` |
| `03_mutate.py` | mutating editors (`add_state`, `replace_state`, `update_state`, `remove_state`, `add_child`, `remove_child`, `update`) and `graph.copy()` |
| `04_save_load.py` | `Graph.save()` / `Graph.load()` JSON round-trip + `Workspace.load_graph()` |
| `05_replay.py` | `graph.history()` — one snapshot per engine `step()` round via the `iteration` stamp on `Node` |
| `06_fork.py` | `Workspace.fork()` — branch a run, diverge, compare |
| `07_render.py` | `graph.tree()`, `graph.session()`, `graph.transcript()`, `graph.save_html(...)` |

Run any of them directly:

```bash
python examples/graph-features/01_query.py
python examples/graph-features/05_replay.py
python examples/graph-features/06_fork.py
```

Most scripts just print to stdout. `07_render.py` writes a viewer HTML
file you can open in your browser.
