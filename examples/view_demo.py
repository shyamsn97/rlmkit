"""Generate a synthetic typed-node trace and open the viewer.

No LLM or runtime needed. This builds fake RLMFlow nodes to demonstrate the
viewer UI.

    python examples/view_demo.py
"""

from rlmflow.node import ActionNode, ErrorNode, QueryNode, ResultNode, SupervisingNode
from rlmflow.utils.viewer import open_viewer

QUERY = "Create a boids simulation in plain HTML + JS"
CFG = {"model": "gpt-5", "max_depth": 3, "max_iterations": 8}


s0 = QueryNode(
    agent_id="root",
    depth=0,
    query=QUERY,
    config=CFG,
    content="Create a boids simulation in plain HTML + JS.",
)

root_action = s0.successor(
    ActionNode,
    reply="I'll split this into files and delegate each part.",
    code=(
        'h1 = delegate("index_html", "Write index.html")\n'
        'h2 = delegate("style_css", "Write style.css")\n'
        'h3 = delegate("script_js", "Write script.js with boids logic")\n'
        "results = yield wait(h1, h2, h3)\n"
        'done("\\n".join(results))'
    ),
)

children = [
    QueryNode(agent_id="root.index_html", depth=1, query="Write index.html", config=CFG),
    QueryNode(agent_id="root.style_css", depth=1, query="Write style.css", config=CFG),
    QueryNode(agent_id="root.script_js", depth=1, query="Write script.js", config=CFG),
]

s1 = root_action.successor(
    SupervisingNode,
    output="",
    waiting_on=[child.agent_id for child in children],
    children=children,
)

s2_children = [
    children[0].successor(ResultNode, result="Created index.html with canvas element"),
    children[1].successor(ResultNode, result="Created style.css with dark theme"),
    children[2].successor(
        SupervisingNode,
        output="",
        waiting_on=[
            "root.script_js.boids_core",
            "root.script_js.renderer",
            "root.script_js.controls",
        ],
        children=[
            QueryNode(agent_id="root.script_js.boids_core", depth=2, query="Core boids"),
            QueryNode(agent_id="root.script_js.renderer", depth=2, query="Canvas renderer"),
            QueryNode(agent_id="root.script_js.controls", depth=2, query="UI controls"),
        ],
    ),
]
s2 = s1.update(children=s2_children)

script_supervisor = s2_children[2]
assert isinstance(script_supervisor, SupervisingNode)
s3 = s2.update(
    children=[
        s2_children[0],
        s2_children[1],
        script_supervisor.update(
            children=[
                script_supervisor.children[0].successor(
                    ResultNode,
                    result="Implemented separation, alignment, and cohesion",
                ),
                script_supervisor.children[1].successor(
                    ResultNode,
                    result="Implemented requestAnimationFrame renderer",
                ),
                script_supervisor.children[2].successor(
                    ErrorNode,
                    error="no_code_block",
                    content="Previous reply did not include a repl block.",
                ),
            ]
        ),
    ]
)

script_done = script_supervisor.successor(
    ResultNode,
    result="Created script.js by combining core, renderer, and controls",
)
s4 = s3.update(children=[s2_children[0], s2_children[1], script_done])

s5 = s4.successor(
    ResultNode,
    result="Created boids simulation: index.html, style.css, script.js",
)

states = [s0, s1, s2, s3, s4, s5]

print(f"Generated {len(states)} steps. Launching viewer...")
open_viewer(states)
