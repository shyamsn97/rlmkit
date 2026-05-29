"""Generate a synthetic Graph trace and open the viewer.

No LLM or runtime needed. This builds fake RLMFlow graph snapshots to
demonstrate the viewer UI.

    python examples/view_demo.py
"""

from __future__ import annotations

from rlmflow.graph import (
    DoneOutput,
    ErrorOutput,
    Graph,
    LLMOutput,
    Node,
    SupervisingOutput,
    UserQuery,
)
from rlmflow.utils.viewer import open_viewer

QUERY = "Create a boids simulation in plain HTML + JS"
CFG = {"model": "gpt-5", "max_depth": 3, "max_iterations": 8}


# ── agent metadata (immutable, used to assemble Graphs per snapshot) ──


META: dict[str, dict] = {
    "root": dict(agent_id="root", depth=0, query=QUERY, config=CFG),
    "root.index_html": dict(
        agent_id="root.index_html",
        depth=1,
        query="Write index.html",
        config=CFG,
        parent_agent_id="root",
    ),
    "root.style_css": dict(
        agent_id="root.style_css",
        depth=1,
        query="Write style.css",
        config=CFG,
        parent_agent_id="root",
    ),
    "root.script_js": dict(
        agent_id="root.script_js",
        depth=1,
        query="Write script.js",
        config=CFG,
        parent_agent_id="root",
    ),
    "root.script_js.boids_core": dict(
        agent_id="root.script_js.boids_core",
        depth=2,
        query="Core boids",
        config=CFG,
        parent_agent_id="root.script_js",
    ),
    "root.script_js.renderer": dict(
        agent_id="root.script_js.renderer",
        depth=2,
        query="Canvas renderer",
        config=CFG,
        parent_agent_id="root.script_js",
    ),
    "root.script_js.controls": dict(
        agent_id="root.script_js.controls",
        depth=2,
        query="UI controls",
        config=CFG,
        parent_agent_id="root.script_js",
    ),
}


# ── helpers ──────────────────────────────────────────────────────────


def snapshot(
    agent_states: dict[str, list[Node]],
    spawn_states: dict[str, str] | None = None,
) -> Graph:
    """Build a recursive :class:`Graph` from per-agent state lists.

    ``spawn_states`` maps a child agent id → the parent state id that
    spawned it (used to render the spawn edge on the parent's timeline).
    """
    spawn_states = spawn_states or {}

    def build(aid: str) -> Graph:
        meta = META[aid]
        states = tuple(agent_states.get(aid, ()))
        kids = {
            cid: build(cid)
            for cid in agent_states
            if META[cid].get("parent_agent_id") == aid
        }
        return Graph.from_meta_dict(
            {**meta, "parent_node_id": spawn_states.get(aid)},
            states=states,
            children=kids,
        )

    return build("root")


# ── snapshot 0: root just got its query ──────────────────────────────


root_q = UserQuery(
    agent_id="root",
    seq=0,
    content="Create a boids simulation in plain HTML + JS.",
)

g0 = snapshot({"root": [root_q]})


# ── snapshot 1: root spawned three children, now waiting ─────────────


root_action = LLMOutput(
    agent_id="root",
    seq=1,
    reply="I'll split this into files and delegate each part.",
    code=(
        "results = await launch_subagents([\n"
        '    {"name": "index_html", "query": "Write index.html"},\n'
        '    {"name": "style_css", "query": "Write style.css"},\n'
        '    {"name": "script_js", "query": "Write script.js with boids logic"},\n'
        "])\n"
        'done("\\n".join(results))'
    ),
)
root_sup = SupervisingOutput(
    agent_id="root",
    seq=2,
    waiting_on=[
        "root.index_html",
        "root.style_css",
        "root.script_js",
    ],
)
child_index_q = UserQuery(agent_id="root.index_html", seq=0, content="Write index.html")
child_style_q = UserQuery(agent_id="root.style_css", seq=0, content="Write style.css")
child_script_q = UserQuery(agent_id="root.script_js", seq=0, content="Write script.js")

FIRST_SPAWNS = {
    "root.index_html": root_action.id,
    "root.style_css": root_action.id,
    "root.script_js": root_action.id,
}

g1 = snapshot(
    {
        "root": [root_q, root_action, root_sup],
        "root.index_html": [child_index_q],
        "root.style_css": [child_style_q],
        "root.script_js": [child_script_q],
    },
    spawn_states=FIRST_SPAWNS,
)


# ── snapshot 2: two simple children done, script.js spawns sub-agents ──


child_index_done = DoneOutput(
    agent_id="root.index_html",
    seq=1,
    result="Created index.html with canvas element",
)
child_style_done = DoneOutput(
    agent_id="root.style_css",
    seq=1,
    result="Created style.css with dark theme",
)
script_action = LLMOutput(
    agent_id="root.script_js",
    seq=1,
    reply="Splitting into core/renderer/controls.",
    code=(
        "await launch_subagents([\n"
        '    {"name": "boids_core", "query": "Core boids"},\n'
        '    {"name": "renderer", "query": "Canvas renderer"},\n'
        '    {"name": "controls", "query": "UI controls"},\n'
        "])"
    ),
)
script_sup = SupervisingOutput(
    agent_id="root.script_js",
    seq=2,
    waiting_on=[
        "root.script_js.boids_core",
        "root.script_js.renderer",
        "root.script_js.controls",
    ],
)
sub_core_q = UserQuery(agent_id="root.script_js.boids_core", seq=0, content="Core boids")
sub_render_q = UserQuery(agent_id="root.script_js.renderer", seq=0, content="Canvas renderer")
sub_controls_q = UserQuery(agent_id="root.script_js.controls", seq=0, content="UI controls")

SECOND_SPAWNS = {
    **FIRST_SPAWNS,
    "root.script_js.boids_core": script_action.id,
    "root.script_js.renderer": script_action.id,
    "root.script_js.controls": script_action.id,
}

g2 = snapshot(
    {
        "root": [root_q, root_action, root_sup],
        "root.index_html": [child_index_q, child_index_done],
        "root.style_css": [child_style_q, child_style_done],
        "root.script_js": [child_script_q, script_action, script_sup],
        "root.script_js.boids_core": [sub_core_q],
        "root.script_js.renderer": [sub_render_q],
        "root.script_js.controls": [sub_controls_q],
    },
    spawn_states=SECOND_SPAWNS,
)


# ── snapshot 3: leaf agents finish (one errors mid-stream) ───────────


sub_core_done = DoneOutput(
    agent_id="root.script_js.boids_core",
    seq=1,
    result="Implemented separation, alignment, and cohesion",
)
sub_render_done = DoneOutput(
    agent_id="root.script_js.renderer",
    seq=1,
    result="Implemented requestAnimationFrame renderer",
)
sub_controls_err = ErrorOutput(
    agent_id="root.script_js.controls",
    seq=1,
    error="no_code_block",
    content="Previous reply did not include a repl block.",
)

g3 = snapshot(
    {
        "root": [root_q, root_action, root_sup],
        "root.index_html": [child_index_q, child_index_done],
        "root.style_css": [child_style_q, child_style_done],
        "root.script_js": [child_script_q, script_action, script_sup],
        "root.script_js.boids_core": [sub_core_q, sub_core_done],
        "root.script_js.renderer": [sub_render_q, sub_render_done],
        "root.script_js.controls": [sub_controls_q, sub_controls_err],
    },
    spawn_states=SECOND_SPAWNS,
)


# ── snapshot 4: script.js retries, finishes, then root completes ─────


sub_controls_done = DoneOutput(
    agent_id="root.script_js.controls",
    seq=2,
    result="Implemented UI controls",
)
script_done = DoneOutput(
    agent_id="root.script_js",
    seq=3,
    result="Created script.js by combining core, renderer, and controls",
)
root_done = DoneOutput(
    agent_id="root",
    seq=3,
    result="Created boids simulation: index.html, style.css, script.js",
)

g4 = snapshot(
    {
        "root": [root_q, root_action, root_sup],
        "root.index_html": [child_index_q, child_index_done],
        "root.style_css": [child_style_q, child_style_done],
        "root.script_js": [child_script_q, script_action, script_sup, script_done],
        "root.script_js.boids_core": [sub_core_q, sub_core_done],
        "root.script_js.renderer": [sub_render_q, sub_render_done],
        "root.script_js.controls": [sub_controls_q, sub_controls_err, sub_controls_done],
    },
    spawn_states=SECOND_SPAWNS,
)

g5 = snapshot(
    {
        "root": [root_q, root_action, root_sup, root_done],
        "root.index_html": [child_index_q, child_index_done],
        "root.style_css": [child_style_q, child_style_done],
        "root.script_js": [child_script_q, script_action, script_sup, script_done],
        "root.script_js.boids_core": [sub_core_q, sub_core_done],
        "root.script_js.renderer": [sub_render_q, sub_render_done],
        "root.script_js.controls": [sub_controls_q, sub_controls_err, sub_controls_done],
    },
    spawn_states=SECOND_SPAWNS,
)


graphs = [g0, g1, g2, g3, g4, g5]


if __name__ == "__main__":
    print(f"Generated {len(graphs)} graph snapshots. Launching viewer...")
    open_viewer(graphs)
