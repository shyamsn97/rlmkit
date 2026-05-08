"""Generate the blog's needle-in-haystack RLM graph example.

This is an offline, deterministic demo: it uses a tiny scripted LLM to
force the nested delegate/wait shape from the blog post, then writes a
markdown artifact with:

- the collapsed "normal RLM" view, where children are just recursive
  calls that return strings;
- a sequence view of delegation and returns;
- selected steppable graph snapshots, rendered as Mermaid;
- an HTML stepper that renders those graph snapshots like a tiny viewer.

Usage:
    python examples/blog_needle_graph.py
    python examples/blog_needle_graph.py --output docs/generated/needle_graph.md
    python examples/blog_needle_graph.py --trace-output docs/generated/needle_trace
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from rlmflow.llm import LLMClient, LLMUsage
from rlmflow.node import Node
from rlmflow.rlm import RLMConfig, RLMFlow
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils.export import to_mermaid_flowchart
from rlmflow.utils.trace import save_trace
from rlmflow.utils.viz import session_events
from rlmflow.workspace import Workspace

ANSWER = "84721"


class BlogNeedleLLM(LLMClient):
    """Scripted LLM that produces one nested RLM run."""

    model = "blog-needle-demo"

    def chat(self, messages, *args, **kwargs) -> str:
        del args, kwargs
        self.last_usage = LLMUsage(input_tokens=80, output_tokens=24)
        last = messages[-1]["content"].lower()
        full = "\n".join(message["content"].lower() for message in messages)

        if "inspect candidate window a" in last:
            return '```repl\ndone("decoy: the code is not 12345")\n```'

        if "inspect candidate window b" in last:
            return f'```repl\ndone("needle: the secret code is {ANSWER}")\n```'

        if "verify candidate code" in last:
            return f'```repl\ndone("{ANSWER} matches the requested needle")\n```'

        if "scan first third" in last:
            return (
                "```repl\n"
                'hits = CONTEXT.grep(r"secret|code|passcode|needle")\n'
                'print(hits or "no matches")\n'
                'done("not found")\n'
                "```"
            )

        if "scan middle third" in last:
            return (
                "```repl\n"
                'hits = CONTEXT.grep(r"secret|code|passcode|needle")\n'
                'print(hits or "no matches")\n'
                'done("decoy, no code")\n'
                "```"
            )

        if "scan final third" in last:
            return (
                "```repl\n"
                'hits = CONTEXT.grep(r"secret|code|passcode|needle")\n'
                'lines = hits.splitlines()\n'
                'a = delegate("candidate_a", "Inspect candidate window A.", "\\n".join(lines[:1]))\n'
                'b = delegate("candidate_b", "Inspect candidate window B.", "\\n".join(lines[1:]))\n'
                "candidate_results = yield wait(a, b)\n"
                f'done("candidate code {ANSWER}")\n'
                "```"
            )

        if "children finished:" in full and "root.verify" in full:
            return f'```repl\ndone("{ANSWER}")\n```'

        if "children finished:" in full and "root.chunk_2" in full:
            return (
                "```repl\n"
                f'v = delegate("verify", "Verify candidate code {ANSWER} against the original question.", "")\n'
                "verdict = yield wait(v)\n"
                f'done("{ANSWER}")\n'
                "```"
            )

        if "what secret code is hidden" in full:
            return (
                "```repl\n"
                "n = CONTEXT.line_count()\n"
                'h0 = delegate("chunk_0", "Scan first third for the hidden secret code.", CONTEXT.lines(0, n // 3))\n'
                'h1 = delegate("chunk_1", "Scan middle third for the hidden secret code.", CONTEXT.lines(n // 3, 2 * n // 3))\n'
                'h2 = delegate("chunk_2", "Scan final third for the hidden secret code.", CONTEXT.lines(2 * n // 3, n))\n'
                "chunk_results = yield wait(h0, h1, h2)\n"
                "print(chunk_results)\n"
                "```"
            )

        return '```repl\ndone("not found")\n```'


def haystack() -> str:
    """Return a small haystack whose final third contains a decoy and needle."""
    lines: list[str] = []
    for idx in range(90):
        if idx == 45:
            lines.append("A decoy note says the code is not 12345.")
        elif idx == 77:
            lines.append(f"The needle sentence: the secret code is {ANSWER}.")
        else:
            lines.append(f"{idx:03d}: alpha bravo charlie delta")
    return "\n".join(lines)


def run_demo(workspace_root: Path) -> tuple[list[Node], RLMFlow]:
    if workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace = Workspace.create(workspace_root)
    runtime = LocalRuntime(workspace=workspace)
    agent = RLMFlow(
        llm_client=BlogNeedleLLM(),
        runtime=runtime,
        workspace=workspace,
        config=RLMConfig(max_depth=3, max_iterations=8),
    )

    state = agent.start(
        "What secret code is hidden in the haystack?",
        context=haystack(),
    )
    states = [state]
    while not state.finished:
        state = agent.step(state)
        states.append(state)
        if len(states) > 20:
            raise RuntimeError("Demo did not terminate within 20 steps")
    return states, agent


def latest_results(states: list[Node]) -> dict[str, str]:
    results: dict[str, str] = {}
    for state in states:
        for node in state.walk():
            result = getattr(node, "result", "")
            if node.type == "result" and result:
                results[node.agent_id] = result
    return results


def collapsed_view(states: list[Node]) -> str:
    results = latest_results(states)
    return "\n".join(
        [
            "root",
            f'  call_llm("scan first third")  -> {results.get("root.chunk_0", "?")}',
            f'  call_llm("scan middle third") -> {results.get("root.chunk_1", "?")}',
            f'  call_llm("scan final third")  -> {results.get("root.chunk_2", "?")}',
            f'  call_llm("verify candidate")  -> {results.get("root.verify", "?")}',
            f"  final answer                  -> {results.get('root', '?')}",
        ]
    )


def pick_phase_states(states: list[Node]) -> list[tuple[str, Node]]:
    """Pick stable snapshots that match the four blog slideshow phases."""

    def has(agent_id: str, node_type: str, state: Node) -> bool:
        return any(
            node.agent_id == agent_id and node.type == node_type
            for node in state.walk()
        )

    phases: list[tuple[str, Node]] = []
    for title, predicate in [
        (
            "1. Root parks after spawning parallel children",
            lambda s: has("root", "supervising", s)
            and has("root.chunk_0", "query", s)
            and has("root.chunk_2", "query", s),
        ),
        (
            "2. First children finish while chunk_2 keeps working",
            lambda s: has("root.chunk_0", "result", s)
            and has("root.chunk_1", "result", s)
            and has("root.chunk_2", "supervising", s),
        ),
        (
            "3. chunk_2 resumes from candidate readers",
            lambda s: has("root.chunk_2.candidate_a", "result", s)
            and has("root.chunk_2.candidate_b", "result", s)
            and has("root.chunk_2", "result", s),
        ),
        (
            "4. Root resumes and returns the answer",
            lambda s: has("root", "result", s),
        ),
    ]:
        match = next((state for state in states if predicate(state)), states[-1])
        phases.append((title, match))
    return phases


def parent_agent(agent_id: str) -> str:
    return agent_id.rsplit(".", 1)[0] if "." in agent_id else "root"


def participant_id(agent_id: str) -> str:
    return agent_id.replace(".", "_").replace("-", "_")


def truncate(text: str, limit: int = 46) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def sequence_diagram(states: list[Node]) -> str:
    """Render delegate/return events from the actual recorded node trace."""
    participants = ["root"]
    seen_participants = {"root"}
    seen_delegates: set[str] = set()
    seen_results: set[str] = set()
    events: list[str] = []

    def add_participant(agent_id: str) -> None:
        if agent_id not in seen_participants:
            seen_participants.add(agent_id)
            participants.append(agent_id)

    for state in states:
        for node in state.walk():
            agent_id = node.agent_id
            if agent_id != "root" and node.type == "query" and agent_id not in seen_delegates:
                parent = parent_agent(agent_id)
                add_participant(parent)
                add_participant(agent_id)
                seen_delegates.add(agent_id)
                events.append(
                    f"    {participant_id(parent)}->>+{participant_id(agent_id)}: "
                    f"delegate {truncate(node.query)}"
                )

            result = getattr(node, "result", "")
            if node.type == "result" and result and agent_id not in seen_results:
                seen_results.add(agent_id)
                add_participant(agent_id)
                if agent_id == "root":
                    events.append(
                        f"    root-->>root: done {truncate(str(result), 34)}"
                    )
                else:
                    parent = parent_agent(agent_id)
                    add_participant(parent)
                    events.append(
                        f"    {participant_id(agent_id)}-->>-{participant_id(parent)}: "
                        f"{truncate(str(result), 34)}"
                    )

    lines = ["sequenceDiagram"]
    for agent_id in participants:
        lines.append(f"    participant {participant_id(agent_id)} as {agent_id}")
    lines.extend(events)
    return "\n".join(lines)


def render_markdown(states: list[Node]) -> str:
    parts = [
        "# Generated Needle Graph Example",
        "",
        "## Collapsed RLM View",
        "",
        "This is what the run looks like if recursive calls collapse to strings:",
        "",
        "```text",
        collapsed_view(states),
        "```",
        "",
        "## Sequence View",
        "",
        "This is the same run as calls and returns:",
        "",
        "```mermaid",
        sequence_diagram(states),
        "```",
        "",
        "## Steppable Graph Snapshots",
        "",
    ]
    for title, state in pick_phase_states(states):
        parts.extend(
            [
                f"### {title}",
                "",
                "```mermaid",
                to_mermaid_flowchart(state),
                "```",
                "",
            ]
        )
    return "\n".join(parts)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def node_table(state: Node) -> str:
    rows = []
    for node in state.walk():
        result = getattr(node, "result", "") or ""
        waiting = ", ".join(getattr(node, "waiting_on", []) or [])
        detail = result or waiting or getattr(node, "query", "") or ""
        rows.append(
            "<tr>"
            f"<td><code>{_escape_html(node.agent_id)}</code></td>"
            f"<td><span class='pill kind-{_escape_html(node.type)}'>{_escape_html(node.type)}</span></td>"
            f"<td>{_escape_html(detail[:120])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_html_viewer(states: list[Node]) -> str:
    slides = []
    buttons = []
    session_path = _session_path(states)
    for idx, state in enumerate(states, start=1):
        title = f"{state.agent_id} · {state.type}"
        active = " active" if idx == 1 else ""
        buttons.append(
            f"<button class='dot{active}' data-step='{idx}' aria-label='Step {idx}'></button>"
        )
        include_plotly = "cdn" if idx == 1 else False
        slides.append(
            f"""
<section class="slide{active}" data-step="{idx}">
  <div class="slide-head">
    <span class="step">Step {idx} / {len(states)}</span>
    <h2>{_escape_html(title)}</h2>
  </div>
  <div class="viewer-grid">
    <div class="graph-card">
      {state.plot_html("graph", states=states, session=session_path, step=idx - 1, mode="events", height=500, title=f"step {idx} / {len(states)}", include_plotlyjs=include_plotly)}
    </div>
    <aside class="detail-card">
      <h3>Transcript: <code>{_escape_html(state.agent_id)}</code></h3>
      <pre>{_escape_html(state.transcript(include_system=False))}</pre>
      <h3>Visible Nodes</h3>
      <table>
        <thead><tr><th>Agent</th><th>Type</th><th>Detail</th></tr></thead>
        <tbody>{node_table(state)}</tbody>
      </table>
    </aside>
  </div>
</section>
""".strip()
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Needle Graph Stepper</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --muted: #8b949e;
      --blue: #58a6ff;
      --green: #3fb950;
      --purple: #bc8cff;
      --yellow: #d29922;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 2rem;
    }}
    main {{
      margin: 0 auto;
      max-width: 1180px;
    }}
    h1, h2, h3 {{ margin-top: 0; }}
    .collapsed, .sequence, .slide {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      margin: 1rem 0;
      padding: 1rem;
    }}
    pre {{
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: auto;
      padding: 1rem;
    }}
    .slide {{ display: none; }}
    .slide.active {{ display: block; }}
    .slide-head {{
      align-items: baseline;
      display: flex;
      gap: 0.75rem;
      justify-content: space-between;
    }}
    .step {{
      color: var(--muted);
      white-space: nowrap;
    }}
    .viewer-grid {{
      display: grid;
      gap: 1rem;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.8fr);
    }}
    .graph-card, .detail-card {{
      min-width: 0;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 0.45rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    code {{ color: var(--blue); }}
    .pill {{
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      display: inline-block;
      padding: 0.1rem 0.45rem;
    }}
    .kind-result {{ color: var(--green); border-color: var(--green); }}
    .kind-supervising {{ color: var(--purple); border-color: var(--purple); }}
    .kind-action {{ color: var(--yellow); border-color: var(--yellow); }}
    .kind-query, .kind-observation {{ color: var(--blue); border-color: var(--blue); }}
    .nav {{
      align-items: center;
      display: flex;
      gap: 0.75rem;
      justify-content: center;
      margin-top: 1rem;
    }}
    .arrow, .dot {{
      cursor: pointer;
    }}
    .arrow {{
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--text);
      height: 2.2rem;
      width: 2.2rem;
    }}
    .dot {{
      background: #30363d;
      border: 1px solid #484f58;
      border-radius: 999px;
      height: 0.7rem;
      width: 0.7rem;
    }}
    .dot.active {{
      background: var(--blue);
      border-color: var(--blue);
    }}
    @media (max-width: 900px) {{
      .viewer-grid {{ grid-template-columns: 1fr; }}
      body {{ padding: 1rem; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>Needle Graph Stepper</h1>
  <p>Generated from an actual deterministic rlmflow run using <code>delegate(...)</code> and <code>yield wait(...)</code>. Each slide is one recorded node snapshot rendered with <code>state.plot()</code>.</p>

  <section class="collapsed">
    <h2>Collapsed RLM View</h2>
    <p>What a normal recursive-call view reduces the run to:</p>
    <pre>{_escape_html(collapsed_view(states))}</pre>
  </section>

  <section class="sequence">
    <h2>Sequence View</h2>
    <pre class="mermaid">{_escape_html(sequence_diagram(states))}</pre>
  </section>

  {"".join(slides)}

  <div class="nav">
    <button class="arrow" id="prev" aria-label="Previous step">&larr;</button>
    {"".join(buttons)}
    <button class="arrow" id="next" aria-label="Next step">&rarr;</button>
  </div>
</main>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
  mermaid.initialize({{ startOnLoad: true, theme: "dark" }});

  const slides = [...document.querySelectorAll(".slide")];
  const dots = [...document.querySelectorAll(".dot")];
  let current = 0;

  function show(index) {{
    current = (index + slides.length) % slides.length;
    slides.forEach((slide, idx) => slide.classList.toggle("active", idx === current));
    dots.forEach((dot, idx) => dot.classList.toggle("active", idx === current));
  }}

  document.getElementById("prev").addEventListener("click", () => show(current - 1));
  document.getElementById("next").addEventListener("click", () => show(current + 1));
  dots.forEach((dot, idx) => dot.addEventListener("click", () => show(idx)));
  document.addEventListener("keydown", (event) => {{
    if (event.key === "ArrowLeft") show(current - 1);
    if (event.key === "ArrowRight") show(current + 1);
  }});
</script>
</body>
</html>
"""


def _session_path(states: list[Node]) -> Path | None:
    for state in states:
        ws = getattr(state, "workspace", None)
        root = getattr(ws, "root", None) if ws else None
        if root:
            path = Path(root) / "session"
            if (path / "nodes.jsonl").exists():
                return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated/needle_graph_example.md"),
        help="Markdown file to write.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("docs/generated/needle_graph_stepper.html"),
        help="Interactive HTML stepper to write.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=Path("docs/generated/needle_trace"),
        help="Trace directory/file to write with every graph snapshot.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Optional workspace directory to keep after the run.",
    )
    args = parser.parse_args()

    if args.workspace is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir) / "workspace"
            states, _ = run_demo(workspace_root)
            markdown = render_markdown(states)
            html = render_html_viewer(states)
            events = session_events(workspace_root / "session")
    else:
        args.workspace.mkdir(parents=True, exist_ok=True)
        states, _ = run_demo(args.workspace)
        markdown = render_markdown(states)
        html = render_html_viewer(states)
        events = session_events(args.workspace / "session")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(html, encoding="utf-8")
    trace_path = save_trace(
        states, args.trace_output, metadata={"answer": ANSWER}, events=events
    )
    if args.workspace is not None:
        save_trace(
            states,
            args.workspace / "trace",
            metadata={"answer": ANSWER},
            events=events,
        )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.html_output}")
    print(f"Wrote {trace_path}")
    print(f"Final answer: {latest_results(states).get('root')}")


if __name__ == "__main__":
    main()
