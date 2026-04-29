"""Small Gradio viewer for typed RLMFlow traces."""

from __future__ import annotations

import json
from typing import Any

from rlmkit.node import Node, parse_node_obj


def _as_node(value: Node | dict) -> Node:
    return value if isinstance(value, Node) else parse_node_obj(value)


def _summary(node: Node) -> str:
    result = getattr(node, "result", None)
    lines = [
        f"agent: {node.agent_id}",
        f"node: {node.id}",
        f"type: {node.type}",
        f"model: {node.model_label}",
        f"terminal: {node.terminal}",
    ]
    if result:
        lines.append(f"result: {result}")
    return "\n".join(lines)


def open_viewer(states: list[Node] | list[dict], **launch_kwargs: Any):
    import gradio as gr

    nodes = [_as_node(state) for state in states]
    choices = [
        f"{i}: {node.agent_id} [{node.type}] {{{node.model_label}}}"
        for i, node in enumerate(nodes)
    ]

    def render(choice: str):
        idx = int(choice.split(":", 1)[0]) if choice else 0
        node = nodes[idx]
        return (
            node.tree(color=False),
            _summary(node),
            json.dumps(node.to_dict(), indent=2),
        )

    with gr.Blocks(title="RLMFlow Viewer") as demo:
        gr.Markdown("## RLMFlow Viewer")
        step = gr.Dropdown(
            choices=choices, value=choices[-1] if choices else None, label="Step"
        )
        tree = gr.Code(label="Tree", language="text")
        summary = gr.Code(label="Summary", language="text")
        raw = gr.Code(label="Raw Node", language="json")
        step.change(render, inputs=step, outputs=[tree, summary, raw])
        if choices:
            demo.load(render, inputs=step, outputs=[tree, summary, raw])
    return demo.launch(**launch_kwargs)


__all__ = ["open_viewer"]
