"""Visualization & export tests: ``Graph.plot``, ``render_html``, ``save_*``.

Replaces ``test_save_api.py`` and the viz/transcript tests that used to live
in ``test_rlmflow_core.py``.
"""

from __future__ import annotations

import importlib.util

import pytest

from rlmflow import Graph, LLMClient, LLMUsage, RLMConfig, RLMFlow, Workspace
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils import (
    render_html,
    resolve_graphs,
    save_gif,
    save_html,
    save_image,
    save_steps,
)
from rlmflow.utils.viz import code_log, report_md, token_sparkline
from rlmflow.utils.viewer import _scale_figure_elements

KALEIDO_INSTALLED = importlib.util.find_spec("kaleido") is not None
PIL_INSTALLED = importlib.util.find_spec("PIL") is not None
PLOTLY_INSTALLED = importlib.util.find_spec("plotly") is not None


# ── shared run fixtures ──────────────────────────────────────────────


class _DelegatingLLM(LLMClient):
    """Tiny scripted LLM that produces a 1-child run."""

    ROOT = (
        "```repl\n"
        "h = delegate('child', 'do the thing', '')\n"
        "results = yield wait(h)\n"
        "done('root:' + results[0])\n"
        "```"
    )
    CHILD = "```repl\ndone('child-answer')\n```"

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=10, output_tokens=5)
        for message in messages:
            if "do the thing" in (message.get("content") or ""):
                return self.CHILD
        return self.ROOT


def _run_steps() -> list[Graph]:
    agent = RLMFlow(
        llm_client=_DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    graph = agent.start("kick off")
    graphs = [graph]
    while not graph.finished:
        graph = agent.step(graph)
        graphs.append(graph)
        assert len(graphs) < 25
    return graphs


def _run_workspace(tmp_path) -> Workspace:
    workspace = Workspace.create(tmp_path / "workspace")
    agent = RLMFlow(
        llm_client=_DelegatingLLM(),
        workspace=workspace,
        config=RLMConfig(max_depth=2),
    )
    graph = agent.start("kick off")
    while not graph.finished:
        graph = agent.step(graph)
    return workspace


def _final() -> Graph:
    return _run_steps()[-1]


# ── Graph.plot: every supported format ───────────────────────────────


def test_plot_static_formats_render_each_kind():
    g = _final()
    assert g.plot("tree").startswith("● root")
    assert g.plot("mermaid").startswith("stateDiagram-v2")
    assert g.plot("flowchart").startswith("flowchart TD")
    assert g.plot("dot").startswith("digraph rlmflow")
    assert "root" in g.plot("d2")


def test_plot_gantt_html_is_self_contained():
    html = _final().plot("gantt", title="sample gantt")
    assert "<html>" in html
    assert "sample gantt" in html


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_plot_returns_plotly_figure_with_title():
    fig = _final().plot(title="sample")
    assert fig.layout.title.text.startswith("<b>sample</b>")
    assert len(fig.data) >= 2


# ── transcripts / sessions (text views over Graph) ───────────────────


def test_transcript_renders_query_action_result_chain():
    class _OneShot(LLMClient):
        def chat(self, *a, **kw):
            return '```repl\ndone("ok")\n```'

    agent = RLMFlow(_OneShot(), runtime=LocalRuntime(), config=RLMConfig(max_iterations=2))
    graph = agent.start("say ok")
    while not graph.finished:
        graph = agent.step(graph)

    transcript = graph.transcript(include_system=False)
    assert "--- query ---\nsay ok" in transcript
    assert '--- assistant ---\n```repl\ndone("ok")\n```' in transcript
    assert "--- result ---\nok" in transcript


def test_session_view_flattens_every_agent_in_tree():
    g = _final()
    session = g.session(include_system=False)
    assert "[root] query" in session
    assert "[root.child] query" in session
    assert "[root.child] result" in session


# ── model labels ─────────────────────────────────────────────────────


def test_model_label_combines_routing_key_and_actual_model():
    g = Graph(agent_id="root.fast", config={"model": "fast"}, model="gpt-5-mini")
    assert g.model_label == "fast:gpt-5-mini"


def test_model_label_falls_back_to_routing_key_when_unbound():
    g = Graph(agent_id="root", config={"model": "default"})
    assert g.model_label == "default"


# ── resolve_graphs ───────────────────────────────────────────────────


def test_resolve_graphs_loads_workspace_or_path(tmp_path):
    workspace = _run_workspace(tmp_path)
    from_workspace = resolve_graphs(workspace)
    from_path = resolve_graphs(workspace.root)
    # Workspaces / workspace paths fan out into one Graph per state
    # append (replayed in execution order) so the viewer slider can
    # scrub through the run.
    assert len(from_workspace) >= 1
    assert from_workspace[-1].result() == "root:child-answer"
    assert from_path[-1].tree() == from_workspace[-1].tree()
    assert len(from_path) == len(from_workspace)


# ── render_html / save_html ──────────────────────────────────────────


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_contains_one_slide_per_step():
    graphs = _run_steps()
    html = render_html(graphs, title="trace test")
    assert html.startswith("<!doctype html>")
    assert "<title>trace test</title>" in html
    for i in range(1, len(graphs) + 1):
        cls = "slide active" if i == 1 else "slide"
        assert f'<section class="{cls}" data-step="{i}"' in html
    assert 'id="prev"' in html and 'id="next"' in html
    assert html.count("https://cdn.plot.ly/plotly") <= 1


def test_render_html_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one graph"):
        render_html([])


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_accepts_workspace(tmp_path):
    workspace = _run_workspace(tmp_path)
    html = render_html(workspace, title="workspace")
    assert "<title>workspace</title>" in html
    assert html.count('<section class="slide') >= 1


def test_resolve_graphs_accepts_workspace_path_with_graph_json(tmp_path):
    workspace = _run_workspace(tmp_path)
    assert Workspace.check_path(workspace.root)

    graphs = resolve_graphs(workspace.root)

    assert graphs
    assert graphs[-1].result() == "root:child-answer"


def test_resolve_graphs_accepts_graph_dump_directory(tmp_path):
    graph_dir = tmp_path / "graph-dir"
    _final().save(graph_dir / "graph.json")
    assert not Workspace.check_path(graph_dir)

    graphs = resolve_graphs(graph_dir)

    assert len(graphs) == 1
    assert graphs[0].result() == "root:child-answer"


def test_viz_helpers_accept_workspace_and_path(tmp_path):
    workspace = _run_workspace(tmp_path)

    assert "tok over" in token_sparkline(workspace)
    assert "## Result" in report_md(workspace.root)
    assert "delegate('child', 'do the thing', '')" in code_log(workspace.root)


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_save_html_writes_file_and_creates_parent_dirs(tmp_path):
    out = save_html(_run_steps(), tmp_path / "nested" / "trace.html", title="t")
    assert out == tmp_path / "nested" / "trace.html"
    assert out.exists()
    assert out.parent.is_dir()
    assert "<title>t</title>" in out.read_text(encoding="utf-8")


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_graph_save_html_shorthand_writes_single_slide(tmp_path):
    out = _final().save_html(tmp_path / "trace.html")
    assert out.exists()
    assert out.read_text().count('<section class="slide') == 1


# ── plot scaling knobs (single test for all knobs) ───────────────────


def _marker_text_sizes(fig):
    """Return (marker.size, annotation_font_size) for the node trace.

    The figure renders nodes as a markers-only Scatter (state kinds are
    in the legend, not on the markers) and agent-name labels as
    layout annotations. ``text_mult`` therefore scales annotation
    fonts, not trace text.
    """
    sizes = None
    for trace in fig.data:
        mode = getattr(trace, "mode", "") or ""
        marker = getattr(trace, "marker", None)
        if (
            "markers" in mode
            and marker is not None
            and getattr(marker, "size", None) is not None
            and getattr(trace, "showlegend", True) is False
        ):
            sizes = marker.size
            break
    annotations = getattr(fig.layout, "annotations", ()) or ()
    font_size = None
    for ann in annotations:
        font = getattr(ann, "font", None)
        if font is not None and getattr(font, "size", None) is not None:
            font_size = font.size
            break
    return sizes, font_size


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_plot_marker_and_text_mults_scale_independently():
    g = _final()
    base_markers, base_font = _marker_text_sizes(g.plot())
    big_markers, big_font = _marker_text_sizes(g.plot(marker_mult=4.0, text_mult=2.0))
    assert base_markers and big_markers
    assert all(b == n * 4.0 for n, b in zip(base_markers, big_markers))
    if base_font is not None:
        assert big_font == base_font * 2.0
    # element_mult shorthand is a uniform scale
    elt_markers, _ = _marker_text_sizes(g.plot(element_mult=2.0))
    assert all(e == n * 2.0 for n, e in zip(base_markers, elt_markers))


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_plot_marker_size_is_uniform_not_token_weighted():
    sizes, _ = _marker_text_sizes(_final().plot())
    assert sizes is not None
    assert len(set(sizes)) == 1


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_plot_normalize_labels_avoids_top_positions():
    fig = _final().plot(normalize_labels=True)
    seen_top = False
    for trace in fig.data:
        if "text" not in (getattr(trace, "mode", "") or ""):
            continue
        positions = getattr(trace, "textposition", None)
        if positions is None:
            continue
        if isinstance(positions, str):
            seen_top |= positions.startswith("top")
        else:
            seen_top |= any((p or "").startswith("top") for p in positions)
    assert not seen_top


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_scale_figure_elements_is_noop_when_mults_are_one():
    fig = _final().plot()

    def snapshot(f):
        out = []
        for t in f.data:
            marker = getattr(t, "marker", None)
            size = getattr(marker, "size", None) if marker is not None else None
            if isinstance(size, (list, tuple)):
                out.append(("list", tuple(size)))
            elif isinstance(size, (int, float)):
                out.append(("scalar", float(size)))
            else:
                out.append(("none", None))
        return out

    before = snapshot(fig)
    _scale_figure_elements(fig, 1.0, 1.0)
    assert snapshot(fig) == before


# ── save_image / save_steps / save_gif (binary outputs) ──────────────


@pytest.mark.skipif(not KALEIDO_INSTALLED, reason="kaleido not installed")
def test_save_image_writes_png_with_magic_bytes(tmp_path):
    out = save_image(_final(), tmp_path / "snap.png", width=400, height=300, scale=1.0)
    assert out.exists() and out.stat().st_size > 0
    with open(out, "rb") as fh:
        assert fh.read(4) == b"\x89PNG"


@pytest.mark.skipif(not KALEIDO_INSTALLED, reason="kaleido not installed")
def test_save_steps_writes_one_frame_per_graph(tmp_path):
    graphs = _run_steps()
    out_dir = save_steps(graphs, tmp_path / "frames", width=400, height=300, scale=1.0)
    files = sorted(p.name for p in out_dir.glob("step_*.png"))
    assert len(files) == len(graphs)
    assert files[0] == "step_00.png"


def test_save_steps_empty_returns_empty_dir(tmp_path):
    out = save_steps([], tmp_path / "empty")
    assert out == tmp_path / "empty"
    assert list(out.iterdir()) == []


@pytest.mark.skipif(
    not (KALEIDO_INSTALLED and PIL_INSTALLED),
    reason="needs both kaleido and pillow",
)
def test_save_gif_writes_gif_with_magic_bytes(tmp_path):
    out = save_gif(_run_steps(), tmp_path / "trace.gif", duration=200, width=300, height=240, scale=1.0)
    with open(out, "rb") as fh:
        assert fh.read(6) in (b"GIF87a", b"GIF89a")


def test_save_gif_rejects_empty_graphs(tmp_path):
    with pytest.raises(ValueError, match="at least one graph"):
        save_gif([], tmp_path / "empty.gif")


def test_save_image_raises_helpful_error_without_kaleido(tmp_path, monkeypatch):
    pio = pytest.importorskip("plotly.io")

    def _boom(*_a, **_kw):
        raise ValueError("Image export requires the kaleido package.")

    monkeypatch.setattr(pio, "write_image", _boom)
    with pytest.raises(ImportError, match="kaleido"):
        save_image(_final(), tmp_path / "fail.png", width=200, height=150)
