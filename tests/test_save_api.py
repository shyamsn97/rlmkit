"""Tests for the public save_image / save_steps / save_html / render_html APIs."""

from __future__ import annotations

import importlib.util

import pytest

from rlmflow import LLMClient, LLMUsage, Node, RLMConfig, RLMFlow
from rlmflow.runtime.local import LocalRuntime
from rlmflow.utils import (
    render_html,
    save_gif,
    save_html,
    save_image,
    save_steps,
)
from rlmflow.utils.viewer import _scale_figure_elements

KALEIDO_INSTALLED = importlib.util.find_spec("kaleido") is not None
PIL_INSTALLED = importlib.util.find_spec("PIL") is not None
PLOTLY_INSTALLED = importlib.util.find_spec("plotly") is not None


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


def _run() -> list[Node]:
    agent = RLMFlow(
        llm_client=_DelegatingLLM(),
        runtime=LocalRuntime(),
        config=RLMConfig(max_depth=2),
    )
    node = agent.start("kick off")
    states = [node]
    while not node.finished:
        node = agent.step(node)
        states.append(node)
        assert len(states) < 25
    return states


# ── render_html / save_html ──────────────────────────────────────────────────


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_contains_one_slide_per_state():
    states = _run()
    html = render_html(states, title="trace test")

    assert html.startswith("<!doctype html>")
    assert "<title>trace test</title>" in html
    # one slide per state, identified by the unique data-step attribute on
    # the <section> element.
    section_steps = [
        '<section class="slide active" data-step="1"',
        *[f'<section class="slide" data-step="{i}"' for i in range(2, len(states) + 1)],
    ]
    for marker in section_steps:
        assert marker in html, f"missing slide marker {marker!r}"
    # navigation present
    assert 'id="prev"' in html and 'id="next"' in html
    # plotly script embedded once (CDN link)
    assert html.count("https://cdn.plot.ly/plotly") <= 1


def test_render_html_rejects_empty_states():
    with pytest.raises(ValueError, match="at least one state"):
        render_html([])


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_save_html_writes_file(tmp_path):
    states = _run()
    out = save_html(states, tmp_path / "trace.html", title="t")

    assert out == tmp_path / "trace.html"
    assert out.exists()
    contents = out.read_text(encoding="utf-8")
    assert "<title>t</title>" in contents


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_save_html_creates_parent_dirs(tmp_path):
    states = _run()
    out = save_html(states, tmp_path / "nested" / "deep" / "trace.html")

    assert out.exists()
    assert out.parent.is_dir()


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_node_save_html_method(tmp_path):
    states = _run()
    out = states[-1].save_html(tmp_path / "trace.html", states=states)

    assert out.exists()
    contents = out.read_text(encoding="utf-8")
    # One <section class="slide..."> per state.
    section_count = contents.count('<section class="slide')
    assert section_count == len(states)


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_normalize_labels_default_strips_top_positions():
    """The HTML stepper should default to bottom-only labels (matches save_image)."""
    states = _run()
    html_default = render_html(states)
    html_alt = render_html(states, normalize_labels=False)

    # Plotly serialises textposition into the embedded JSON. Without
    # normalize_labels the alternating layout produces "top center"
    # entries; with it (the default) every position becomes "bottom *".
    assert "top center" not in html_default, (
        "render_html should default to normalize_labels=True so labels "
        "don't collide across depths"
    )
    assert "top center" in html_alt, (
        "render_html(normalize_labels=False) should keep the alternating "
        "top/bottom layout"
    )


@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly not installed")
def test_render_html_marker_mult_shows_up_in_embedded_json():
    """marker_mult should propagate all the way through render_html."""
    states = _run()
    html_base = render_html(states, marker_mult=1.0)
    html_big = render_html(states, marker_mult=5.0)

    # marker_mult only affects the labelled scatter trace's marker.size
    # array; the document size grows because each rendered size is
    # ~5x bigger (e.g. "55" vs "11"). We just need to know the knob did
    # something — exact matching against Plotly's JSON output is brittle.
    assert html_big != html_base
    # And the legend stub markers (size 11, hard-coded) shouldn't change
    # — sanity check that we didn't accidentally over-scale.
    assert html_base.count('"size":11') >= 1


# ── element_mult scaling ──────────────────────────────────────────────────────


def _marker_text_sizes(fig):
    """Return (marker_sizes, font_size) from the labeled marker trace."""
    for trace in fig.data:
        mode = getattr(trace, "mode", "") or ""
        if "markers" in mode and "text" in mode and trace.marker.size is not None:
            font = getattr(trace, "textfont", None)
            return trace.marker.size, getattr(font, "size", None)
    return None, None


def test_node_plot_element_mult_scales_markers():
    pytest.importorskip("plotly.graph_objects")
    states = _run()
    fig_normal = states[-1].plot(states=states, element_mult=1.0)
    fig_big = states[-1].plot(states=states, element_mult=2.0)

    n_size, _ = _marker_text_sizes(fig_normal)
    b_size, _ = _marker_text_sizes(fig_big)
    assert n_size is not None and b_size is not None
    # element_mult=2 should give exactly 2x markers + fonts (uniform).
    assert all(b == n * 2 for n, b in zip(n_size, b_size))


def test_node_plot_uses_uniform_base_marker_sizes():
    """Node marker size should not encode token counts."""
    pytest.importorskip("plotly.graph_objects")
    states = _run()
    fig = states[-1].plot(states=states, element_mult=1.0)

    sizes, _ = _marker_text_sizes(fig)
    assert sizes is not None
    assert len(set(sizes)) == 1


def test_node_plot_split_marker_text_mult():
    """marker_mult and text_mult override element_mult independently."""
    pytest.importorskip("plotly.graph_objects")
    states = _run()
    fig_base = states[-1].plot(states=states)
    fig_split = states[-1].plot(states=states, marker_mult=4.0, text_mult=2.0)

    base_markers, base_font = _marker_text_sizes(fig_base)
    split_markers, split_font = _marker_text_sizes(fig_split)
    assert base_markers and split_markers
    assert all(s == b * 4.0 for b, s in zip(base_markers, split_markers))
    if base_font is not None:
        assert split_font == base_font * 2.0


def test_node_plot_normalize_labels():
    """normalize_labels forces every textposition to bottom center."""
    pytest.importorskip("plotly.graph_objects")
    states = _run()
    fig = states[-1].plot(states=states, normalize_labels=True)

    seen_top = False
    seen_bottom = False
    for trace in fig.data:
        mode = getattr(trace, "mode", "") or ""
        if "text" not in mode:
            continue
        positions = getattr(trace, "textposition", None)
        if positions is None:
            continue
        if isinstance(positions, str):
            seen_top |= positions.startswith("top")
            seen_bottom |= positions.startswith("bottom")
        else:
            seen_top |= any((p or "").startswith("top") for p in positions)
            seen_bottom |= any((p or "").startswith("bottom") for p in positions)

    assert not seen_top, "normalize_labels should erase every 'top *' position"
    assert seen_bottom, "expected at least one 'bottom *' label"


def test_scale_figure_elements_noop_when_mult_one():
    pytest.importorskip("plotly.graph_objects")
    states = _run()
    fig = states[-1].plot(states=states)

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
    after = snapshot(fig)
    assert before == after


# ── save_image / save_steps (kaleido-gated) ───────────────────────────────────


@pytest.mark.skipif(not KALEIDO_INSTALLED, reason="kaleido not installed")
def test_save_image_writes_png(tmp_path):
    states = _run()
    out = save_image(
        states[-1],
        tmp_path / "snap.png",
        states=states,
        # Keep the canvas tiny so kaleido is fast even on hot path.
        width=400,
        height=300,
        scale=1.0,
        element_mult=1.5,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    # PNG magic number.
    with open(out, "rb") as fh:
        assert fh.read(4) == b"\x89PNG"


@pytest.mark.skipif(not KALEIDO_INSTALLED, reason="kaleido not installed")
def test_save_steps_writes_one_per_state(tmp_path):
    states = _run()
    out_dir = save_steps(
        states,
        tmp_path / "frames",
        width=400,
        height=300,
        scale=1.0,
        element_mult=1.5,
    )
    assert out_dir == tmp_path / "frames"
    files = sorted(p.name for p in out_dir.glob("step_*.png"))
    assert len(files) == len(states)
    assert files[0] == "step_00.png"


@pytest.mark.skipif(not KALEIDO_INSTALLED, reason="kaleido not installed")
def test_node_save_image_method(tmp_path):
    states = _run()
    out = states[-1].save_image(
        tmp_path / "shorthand.png",
        states=states,
        width=400,
        height=300,
        scale=1.0,
    )
    assert out.exists()
    assert out.suffix == ".png"


def test_save_steps_empty_returns_dir(tmp_path):
    out = save_steps([], tmp_path / "empty")
    assert out == tmp_path / "empty"
    assert out.is_dir()
    assert list(out.iterdir()) == []


@pytest.mark.skipif(
    not (KALEIDO_INSTALLED and PIL_INSTALLED),
    reason="needs both kaleido and pillow",
)
def test_save_gif_writes_gif(tmp_path):
    states = _run()
    out = save_gif(
        states,
        tmp_path / "trace.gif",
        duration=200,
        width=300,
        height=240,
        scale=1.0,
        element_mult=1.5,
    )
    assert out.exists()
    with open(out, "rb") as fh:
        magic = fh.read(6)
    # GIF89a / GIF87a magic
    assert magic in (b"GIF87a", b"GIF89a")


def test_save_gif_empty_states_raises(tmp_path):
    with pytest.raises(ValueError, match="at least one state"):
        save_gif([], tmp_path / "empty.gif")


def test_save_image_raises_helpful_error_without_kaleido(tmp_path, monkeypatch):
    """Even with kaleido installed, simulate its absence to lock the message in."""
    pio = pytest.importorskip("plotly.io")

    states = _run()

    def _boom(*_args, **_kwargs):  # pragma: no cover - failure path
        raise ValueError("Image export requires the kaleido package.")

    monkeypatch.setattr(pio, "write_image", _boom)
    with pytest.raises(ImportError, match="kaleido"):
        save_image(
            states[-1],
            tmp_path / "fail.png",
            states=states,
            width=200,
            height=150,
        )
