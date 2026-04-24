"""Interactive Gradio state viewer for RLM traces.

Usage::

    from rlmkit.utils.viewer import open_viewer, view_trace

    open_viewer(states)               # from a live run
    view_trace("traces/my_run")       # from disk

Per-step queries are read from ``state.query`` and shown in the detail
panel — multi-turn sessions display correctly. Saving and loading
traces lives in :mod:`rlmkit.utils.trace`.

Requires: ``pip install gradio``.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmkit.state import _dump_state

if TYPE_CHECKING:
    from rlmkit.state import RLMState


def view_trace(path: str | Path, **launch_kwargs: Any) -> None:
    """Load a trace from disk and open the viewer."""
    from rlmkit.utils.trace import load_trace

    open_viewer(load_trace(path).states, **launch_kwargs)


# ── Helpers ──────────────────────────────────────────────────────────


def _flatten(state: dict, out: dict | None = None) -> dict:
    if out is None:
        out = {}
    out[state.get("agent_id") or "root"] = state
    for c in state.get("children") or []:
        _flatten(c, out)
    return out


def _all_ids(state: dict) -> list[str]:
    ids = [state.get("agent_id") or "root"]
    for c in state.get("children") or []:
        ids.extend(_all_ids(c))
    return ids


def _changed_ids(prev: dict | None, curr: dict) -> set[str]:
    if prev is None:
        return set()
    pf, cf = _flatten(prev), _flatten(curr)
    changed = set()
    for aid, c in cf.items():
        p = pf.get(aid)
        if p is None or (
            p.get("status") != c.get("status")
            or p.get("iteration") != c.get("iteration")
            or p.get("result") != c.get("result")
            or len(p.get("children") or []) != len(c.get("children") or [])
        ):
            changed.add(aid)
    return changed


def _event_type(ev: dict | None) -> str:
    if ev is None:
        return ""
    if "_type" in ev:
        return ev["_type"]
    if "code" in ev and "output" in ev:
        return "CodeExec"
    if "text" in ev and "code" in ev:
        return "LLMReply"
    if "text" in ev:
        return "NoCodeBlock"
    if "output" in ev:
        return "ResumeExec"
    return "Unknown"


def _esc(s: str) -> str:
    return _html.escape(s)


def _trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


# ── Radio-choice builder ─────────────────────────────────────────────

STATUS_MARK = {
    "ready": "○",
    "executing": "●",
    "supervising": "◆",
    "finished": "✔",
}


def _short_id(aid: str) -> str:
    return aid.rsplit(".", 1)[-1] if "." in aid else aid


def _radio_choices(
    state: dict,
    changed: set[str],
    depth: int = 0,
    parent_is_last: list[bool] | None = None,
) -> list[tuple[str, str]]:
    """Build (label, value) pairs for gr.Radio with box-drawing tree lines."""
    if parent_is_last is None:
        parent_is_last = []

    aid = state.get("agent_id") or "root"
    status = state.get("status", "ready")
    mark = STATUS_MARK.get(status, "○")
    name = _short_id(aid)
    chg = " *" if aid in changed else ""

    if depth == 0:
        prefix = ""
    else:
        prefix = ""
        for is_last in parent_is_last[:-1]:
            prefix += "   " if is_last else "│  "
        prefix += "└─ " if parent_is_last[-1] else "├─ "

    label = f"{prefix}{mark} {name} [{status}]{chg}"

    choices = [(label, aid)]
    children = state.get("children") or []
    for i, c in enumerate(children):
        is_last = i == len(children) - 1
        choices.extend(
            _radio_choices(c, changed, depth + 1, parent_is_last + [is_last])
        )
    return choices


# ── Detail HTML builders ─────────────────────────────────────────────

_STATUS_CSS = {
    "ready": "color:#58a6ff",
    "executing": "color:#d29922",
    "supervising": "color:#bc8cff",
    "finished": "color:#3fb950",
}


def _pill(status: str) -> str:
    style = _STATUS_CSS.get(status, "")
    return f'<span style="font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;background:rgba(255,255,255,.06);{style}">{status}</span>'


def build_info(state: dict | None) -> str:
    """Compact summary card — goes below the tree on the left."""
    if state is None:
        return '<p style="color:#8b949e">Select a node</p>'

    aid = state.get("agent_id") or "root"
    status = state.get("status", "ready")
    model = (state.get("config") or {}).get("model", "—")
    iteration = state.get("iteration", 0)
    query = state.get("query") or state.get("task") or "—"
    result = state.get("result")

    h = ['<div style="font-size:13px;line-height:1.7">']
    h.append(f'<h3 style="margin:0 0 6px;color:#e6edf3">{_esc(aid)}</h3>')
    h.append(
        f"<b>Status:</b> {_pill(status)} &nbsp; <b>Model:</b> {_esc(model)} &nbsp; <b>Iter:</b> {iteration}"
    )
    h.append(f"<br><b>Query:</b> {_esc(_trunc(query, 200))}")
    if result is not None:
        h.append(
            f'<br><b style="color:#3fb950">Result:</b> {_esc(_trunc(result, 200))}'
        )
    h.append("</div>")
    return "".join(h)


_PRE = 'style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;font-size:12px;overflow-x:auto;max-height:240px;overflow-y:auto;color:#e6edf3"'
_PRE_DIM = 'style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;font-size:12px;max-height:240px;overflow-y:auto;color:#8b949e"'


def build_event(state: dict | None) -> str:
    """Event + children details — goes on the right."""
    if state is None:
        return ""

    h = ['<div style="font-size:13px;line-height:1.6">']

    ev = state.get("event")
    evtype = _event_type(ev)
    if ev:
        h.append(f"<b>Event:</b> <code>{evtype}</code>")
        if evtype == "CodeExec":
            if ev.get("suspended"):
                h.append(
                    '<br><span style="color:#bc8cff">⏸ suspended — waiting on children</span>'
                )
            h.append(f'<br><b>Code:</b><pre {_PRE}>{_esc(ev.get("code", ""))}</pre>')
            h.append(
                f'<b>Output:</b><pre {_PRE_DIM}>{_esc(ev.get("output", "") or "(empty)")}</pre>'
            )
        elif evtype == "LLMReply":
            code = ev.get("code")
            if code:
                h.append(f"<pre {_PRE}>{_esc(code)}</pre>")
            h.append(
                f'<details><summary style="cursor:pointer;color:#8b949e;font-size:12px">Full reply ({len(ev.get("text", ""))} chars)</summary><pre style="font-size:12px;white-space:pre-wrap;color:#e6edf3;max-height:300px;overflow-y:auto">{_esc(ev.get("text", ""))}</pre></details>'
            )
        elif evtype == "ResumeExec":
            output = ev.get("output", "")
            if output:
                h.append(f"<pre {_PRE_DIM}>{_esc(output)}</pre>")
        elif evtype == "NoCodeBlock":
            h.append(
                '<br><span style="color:#f85149">⚠ LLM did not produce a code block</span>'
            )
            h.append(
                f'<pre style="font-size:12px;color:#8b949e;max-height:200px;overflow-y:auto">{_esc(_trunc(ev.get("text", ""), 400))}</pre>'
            )
    else:
        h.append('<span style="color:#8b949e">No event at this step</span>')

    children = state.get("children") or []
    if children:
        h.append('<hr style="border-color:#30363d;margin:10px 0">')
        h.append(f"<b>Children ({len(children)}):</b><br>")
        for c in children:
            cstatus = c.get("status", "?")
            cresult = c.get("result")
            if cresult:
                h.append(
                    f'{_pill(cstatus)} <code>{_esc(c.get("agent_id", "?"))}</code> → {_esc(_trunc(cresult, 60))}<br>'
                )
            else:
                h.append(
                    f'{_pill(cstatus)} <code>{_esc(c.get("agent_id", "?"))}</code><br>'
                )
        waiting = state.get("waiting_on") or []
        if waiting:
            h.append(
                f'<span style="color:#8b949e;font-size:12px">waiting_on: {", ".join(f"<code>{_esc(w)}</code>" for w in waiting)}</span>'
            )

    h.append("</div>")
    return "".join(h)


def build_messages(state: dict | None) -> str:
    if state is None:
        return ""
    messages = state.get("messages") or []
    sys_prompt = state.get("system_prompt")
    if not messages and not sys_prompt:
        return '<p style="color:#8b949e;font-size:12px">No messages</p>'

    role_colors = {"system": "#8b949e", "user": "#58a6ff", "assistant": "#3fb950"}

    h = [f'<div style="font-size:12px"><b>{len(messages)} messages</b>']
    if sys_prompt:
        h.append(
            '<details style="border:1px solid #30363d;border-left:3px solid #8b949e;border-radius:6px;margin:6px 0;overflow:hidden">'
        )
        h.append(
            f'<summary style="padding:5px 10px;cursor:pointer;font-size:10px;font-weight:700;color:#8b949e;text-transform:uppercase;background:rgba(255,255,255,.02)">SYSTEM PROMPT <span style="font-weight:400">({len(sys_prompt)} chars)</span></summary>'
        )
        h.append(
            f'<pre style="padding:8px 10px;margin:0;font-size:11px;white-space:pre-wrap;word-break:break-word;color:#e6edf3;max-height:300px;overflow-y:auto;border-top:1px solid #30363d">{_esc(sys_prompt)}</pre>'
        )
        h.append("</details>")
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2)

        is_exec = role == "user" and content.startswith("Code executed:")
        is_resume = role == "user" and content.startswith("Children finished:")
        if is_exec:
            color = "#bc8cff"
            label = "execution"
            bg = "rgba(188,140,255,.04)"
        elif is_resume:
            color = "#d29922"
            label = "resume"
            bg = "rgba(210,153,34,.04)"
        else:
            color = role_colors.get(role, "#8b949e")
            label = role
            bg = "rgba(255,255,255,.02)"

        short = len(content) <= 600
        h.append(
            f'<div style="border:1px solid #30363d;border-left:3px solid {color};border-radius:6px;margin:6px 0;overflow:hidden">'
        )
        h.append(
            f'<div style="padding:5px 10px;font-size:10px;font-weight:700;color:{color};text-transform:uppercase;border-bottom:1px solid #30363d;background:{bg}">{label} <span style="font-weight:400;color:#8b949e">({len(content)} chars)</span></div>'
        )
        max_h = "none" if short else "180px"
        h.append(
            f'<pre style="padding:8px 10px;margin:0;font-size:11px;white-space:pre-wrap;word-break:break-word;color:#e6edf3;max-height:{max_h};overflow-y:auto">{_esc(content)}</pre>'
        )
        h.append("</div>")
    h.append("</div>")
    return "".join(h)


def build_diff(prev: dict | None, curr: dict) -> str:
    if prev is None:
        return '<p style="color:#8b949e;font-size:12px">First step — no diff</p>'
    changed = _changed_ids(prev, curr)
    if not changed:
        return '<p style="color:#8b949e;font-size:12px">No changes</p>'

    pf, cf = _flatten(prev), _flatten(curr)
    h = ['<div style="font-size:12px">']
    for aid in sorted(changed):
        c = cf[aid]
        p = pf.get(aid)
        if p is None:
            h.append(
                f'<div style="color:#58a6ff;padding:2px 0">⊕ <code>{_esc(aid)}</code> — new ({c.get("status")})</div>'
            )
        elif c.get("status") == "finished" and p.get("status") != "finished":
            res = c.get("result")
            extra = f" → {_esc(_trunc(res, 50))}" if res else ""
            h.append(
                f'<div style="color:#3fb950;padding:2px 0">✓ <code>{_esc(aid)}</code> — {p.get("status")} → finished{extra}</div>'
            )
        else:
            parts = []
            if p.get("status") != c.get("status"):
                parts.append(f'{p.get("status")} → {c.get("status")}')
            if p.get("iteration") != c.get("iteration"):
                parts.append(f'iter {p.get("iteration")} → {c.get("iteration")}')
            h.append(
                f'<div style="color:#d29922;padding:2px 0">⟳ <code>{_esc(aid)}</code> — {", ".join(parts)}</div>'
            )
    h.append("</div>")
    return "".join(h)


# ── Gradio app ───────────────────────────────────────────────────────

VIEWER_CSS = """
/* ── global ── */
.gradio-container { max-width: 100% !important; }
footer { display: none !important; }

/* ── panels ── */
.sv-panel {
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    background: #161b22 !important;
    padding: 14px 16px !important;
}

/* ── slider ── */
#sv-step { border: 1px solid #30363d !important; border-radius: 10px !important; background: #161b22 !important; padding: 10px 16px !important; }
#sv-step input[type=range] { accent-color: #58a6ff !important; }

/* ── tree radio ── */
#sv-tree { padding: 0 !important; border: 1px solid #30363d !important; border-radius: 10px !important; background: #161b22 !important; overflow: hidden !important; }
#sv-tree .wrap {
    gap: 0 !important; padding: 6px 0 !important;
    flex-direction: column !important; flex-wrap: nowrap !important;
}
#sv-tree label {
    border: none !important; border-radius: 0 !important;
    border-left: 3px solid transparent !important;
    padding: 5px 14px !important; margin: 0 !important;
    background: transparent !important;
    font-family: 'SF Mono', 'Menlo', 'Consolas', 'Liberation Mono', monospace !important;
    font-size: 12.5px !important; white-space: pre !important;
    cursor: pointer !important; transition: all .12s !important;
    min-height: unset !important; width: 100% !important;
    line-height: 1.5 !important;
}
#sv-tree label:hover { background: rgba(255,255,255,.04) !important; }
#sv-tree label.selected {
    background: rgba(88,166,255,.08) !important;
    border-left-color: #58a6ff !important;
}
#sv-tree label { animation: sv-fade-in .18s ease-out; }
@keyframes sv-fade-in {
    from { opacity: 0; transform: translateX(-4px); }
    to   { opacity: 1; transform: translateX(0); }
}
#sv-tree label input[type="radio"] { display: none !important; }
#sv-tree label .icon { display: none !important; }

/* ── info card ── */
#sv-info { border: 1px solid #30363d !important; border-radius: 10px !important; background: #161b22 !important; padding: 12px 16px !important; }

/* ── right column panels ── */
#sv-event { border: 1px solid #30363d !important; border-radius: 10px !important; background: #161b22 !important; padding: 14px 16px !important; }
#sv-msgs  { border: 1px solid #30363d !important; border-radius: 10px !important; background: #161b22 !important; padding: 14px 16px !important; }
"""


def open_viewer(
    states: list[RLMState] | list[dict],
    **launch_kwargs: Any,
) -> None:
    """Launch the Gradio viewer. Blocks until closed."""
    import gradio as gr

    steps: list[dict] = [
        _dump_state(s) if hasattr(s, "model_dump") else s for s in states
    ]
    if not steps:
        raise ValueError("No states to display")

    max_step = len(steps) - 1

    def _get_view(step_idx: int, node_id: str | None):
        curr = steps[step_idx]
        prev = steps[step_idx - 1] if step_idx > 0 else None
        changed = _changed_ids(prev, curr)
        flat = _flatten(curr)
        ids = _all_ids(curr)
        choices = _radio_choices(curr, changed)

        node_val = node_id if node_id and node_id in flat else ids[0]
        selected = flat[node_val]
        return choices, node_val, selected, prev, curr

    def on_step(step_idx: int, node_id: str | None):
        step_idx = int(step_idx)
        choices, node_val, selected, prev, curr = _get_view(step_idx, node_id)
        return (
            gr.Radio(choices=choices, value=node_val),
            build_info(selected),
            build_event(selected),
            build_messages(selected),
            build_diff(prev, curr),
        )

    def on_node(step_idx: int, node_id: str):
        step_idx = int(step_idx)
        choices, node_val, selected, prev, curr = _get_view(step_idx, node_id)
        return (
            gr.Radio(choices=choices, value=node_val),
            build_info(selected),
            build_event(selected),
            build_messages(selected),
        )

    init_choices, init_val, init_state, _, _ = _get_view(0, None)

    with gr.Blocks() as app:
        gr.Markdown("## RLM State Viewer")

        with gr.Row(elem_id="sv-controls"):
            play_btn = gr.Button("▶ Play", scale=0, min_width=100)
            speed = gr.Slider(
                minimum=0.25,
                maximum=4.0,
                step=0.25,
                value=1.0,
                label="speed (steps/sec)",
                scale=1,
            )
            step_slider = gr.Slider(
                minimum=0,
                maximum=max_step,
                step=1,
                value=0,
                label=f"Step (0 – {max_step})",
                elem_id="sv-step",
                scale=3,
            )

        playing = gr.State(False)
        timer = gr.Timer(1.0, active=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=260):
                node_radio = gr.Radio(
                    choices=init_choices,
                    value=init_val,
                    label="Nodes",
                    elem_id="sv-tree",
                )
                info_html = gr.HTML(value=build_info(init_state), elem_id="sv-info")

            with gr.Column(scale=3):
                event_html = gr.HTML(value=build_event(init_state), elem_id="sv-event")
                messages_html = gr.HTML(
                    value=build_messages(init_state), elem_id="sv-msgs"
                )

        with gr.Accordion("Diff from previous step", open=False):
            diff_html = gr.HTML(
                value='<p style="color:#8b949e;font-size:12px">First step — no diff</p>'
            )

        step_slider.change(
            fn=on_step,
            inputs=[step_slider, node_radio],
            outputs=[node_radio, info_html, event_html, messages_html, diff_html],
        )
        node_radio.change(
            fn=on_node,
            inputs=[step_slider, node_radio],
            outputs=[node_radio, info_html, event_html, messages_html],
        )

        def toggle_play(is_playing: bool, step_idx: float):
            # Clicking Play at the end restarts from 0.
            if not is_playing and int(step_idx) >= max_step:
                return True, gr.Button(value="⏸ Pause"), 0
            new = not is_playing
            return new, gr.Button(value="⏸ Pause" if new else "▶ Play"), step_idx

        play_btn.click(
            fn=toggle_play,
            inputs=[playing, step_slider],
            outputs=[playing, play_btn, step_slider],
        )

        def sync_timer(is_playing: bool, steps_per_sec: float):
            return gr.Timer(value=max(0.05, 1.0 / steps_per_sec), active=is_playing)

        playing.change(fn=sync_timer, inputs=[playing, speed], outputs=[timer])
        speed.change(fn=sync_timer, inputs=[playing, speed], outputs=[timer])

        def tick(step_idx: float):
            step_idx = int(step_idx)
            if step_idx >= max_step:
                return (
                    max_step,
                    False,
                    gr.Button(value="▶ Play"),
                    gr.Timer(active=False),
                )
            return step_idx + 1, True, gr.Button(value="⏸ Pause"), gr.Timer(active=True)

        timer.tick(
            fn=tick,
            inputs=[step_slider],
            outputs=[step_slider, playing, play_btn, timer],
        )

    kwargs = {"share": False, "show_error": True, "css": VIEWER_CSS}
    kwargs.update(launch_kwargs)
    app.launch(**kwargs)
