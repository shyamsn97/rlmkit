"""Generate a synthetic trace with nested delegation and open the state viewer.

No LLM or runtime needed — this builds fake RLMState snapshots
to demonstrate the viewer UI.

    python examples/view_demo.py
"""

from rlmkit.state import CodeExec, LLMReply, NoCodeBlock, ResumeExec, RLMState, Status
from rlmkit.utils.viewer import open_viewer

QUERY = "Create a boids simulation in plain HTML + JS"
CFG = {"model": "gpt-5", "max_depth": 3}
CHILD_CFG = {"model": "gpt-5", "depth": 1}
GRANDCHILD_CFG = {"model": "gpt-4.1-mini", "depth": 2}

# ── helpers ──────────────────────────────────────────────────────────

def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ═══════════════════════════════════════════════════════════════════════
# Step 0: root is READY, no children
# ═══════════════════════════════════════════════════════════════════════
s0 = RLMState(
    agent_id="root", query=QUERY, status=Status.READY, config=CFG,
    messages=[_msg("user", "Create a boids simulation in plain HTML + JS.")],
)

# ═══════════════════════════════════════════════════════════════════════
# Step 1: root LLM replied → delegates 4 children → SUPERVISING
# ═══════════════════════════════════════════════════════════════════════
s1_children = [
    RLMState(agent_id="root.index_html", query="Write index.html", status=Status.READY,
             config=CHILD_CFG, messages=[_msg("user", "Write index.html for boids sim")]),
    RLMState(agent_id="root.style_css", query="Write style.css", status=Status.READY,
             config=CHILD_CFG, messages=[_msg("user", "Write style.css for boids sim")]),
    RLMState(agent_id="root.script_js", query="Write script.js with boids logic",
             status=Status.READY, config=CHILD_CFG,
             messages=[_msg("user", "Write script.js implementing the boids algorithm")]),
    RLMState(agent_id="root.readme", query="Write README.md", status=Status.READY,
             config=CHILD_CFG, messages=[_msg("user", "Write README.md for boids sim")]),
]

s1 = RLMState(
    agent_id="root", query=QUERY, status=Status.SUPERVISING, iteration=1, config=CFG,
    event=CodeExec(
        agent_id="root", iteration=1, suspended=True,
        code='h1 = delegate("index_html", "Write index.html")\nh2 = delegate("style_css", "Write style.css")\nh3 = delegate("script_js", "Write script.js with boids logic")\nh4 = delegate("readme", "Write README.md")\nyield wait([h1, h2, h3, h4])',
        output="",
    ),
    messages=[
        _msg("user", "Create a boids simulation in plain HTML + JS."),
        _msg("assistant", "I'll split this into 4 files and delegate each.\n\n```repl\nh1 = delegate(...)\nyield wait([h1, h2, h3, h4])\n```"),
        _msg("user", "[code executed — suspended, waiting on children]"),
    ],
    last_reply="I'll split this into 4 files and delegate each.",
    children=s1_children,
    waiting_on=["root.index_html", "root.style_css", "root.script_js", "root.readme"],
)

# ═══════════════════════════════════════════════════════════════════════
# Step 2: index_html and style_css finished; script_js got LLM reply;
#         readme finished
# ═══════════════════════════════════════════════════════════════════════
s2_children = [
    RLMState(
        agent_id="root.index_html", query="Write index.html", status=Status.FINISHED,
        iteration=1, config=CHILD_CFG,
        event=CodeExec(agent_id="root.index_html", iteration=1,
                       code='write_file("index.html", html)\ndone("Created index.html")',
                       output="Created index.html"),
        result="Created index.html with canvas element and script/style links",
        messages=[
            _msg("user", "Write index.html for boids sim"),
            _msg("assistant", '```repl\nwrite_file("index.html", ...)\ndone("Created index.html")\n```'),
        ],
    ),
    RLMState(
        agent_id="root.style_css", query="Write style.css", status=Status.FINISHED,
        iteration=1, config=CHILD_CFG,
        event=CodeExec(agent_id="root.style_css", iteration=1,
                       code='write_file("style.css", css)\ndone("Created style.css")',
                       output="Created style.css"),
        result="Created style.css with dark theme and full-bleed canvas",
        messages=[
            _msg("user", "Write style.css for boids sim"),
            _msg("assistant", '```repl\nwrite_file("style.css", ...)\ndone("Created style.css")\n```'),
        ],
    ),
    RLMState(
        agent_id="root.script_js", query="Write script.js with boids logic",
        status=Status.EXECUTING, iteration=1, config=CHILD_CFG,
        event=LLMReply(
            agent_id="root.script_js", iteration=1,
            text="This is a complex file — I'll break it into modules and delegate.\n\n```repl\nh1 = delegate(\"boids_core\", \"Write core boids algorithm\")\nh2 = delegate(\"renderer\", \"Write canvas renderer\")\nh3 = delegate(\"controls\", \"Write UI controls\")\nyield wait([h1, h2, h3])\n```",
            code='h1 = delegate("boids_core", "Write core boids algorithm")\nh2 = delegate("renderer", "Write canvas renderer")\nh3 = delegate("controls", "Write UI controls")\nyield wait([h1, h2, h3])',
        ),
        messages=[
            _msg("user", "Write script.js implementing the boids algorithm"),
            _msg("assistant", "This is complex — I'll break it into modules and delegate."),
        ],
    ),
    RLMState(
        agent_id="root.readme", query="Write README.md", status=Status.FINISHED,
        iteration=1, config=CHILD_CFG,
        event=CodeExec(agent_id="root.readme", iteration=1,
                       code='write_file("README.md", readme)\ndone("Created README.md")',
                       output="Created README.md"),
        result="Created README.md with project description and usage instructions",
        messages=[
            _msg("user", "Write README.md for boids sim"),
            _msg("assistant", '```repl\nwrite_file("README.md", ...)\ndone("Created README.md")\n```'),
        ],
    ),
]

s2 = s1.update(children=s2_children, event=None)

# ═══════════════════════════════════════════════════════════════════════
# Step 3: script_js is now SUPERVISING with 3 grandchildren (all READY)
# ═══════════════════════════════════════════════════════════════════════
grandchildren = [
    RLMState(agent_id="root.script_js.boids_core", query="Write core boids algorithm",
             status=Status.READY, config=GRANDCHILD_CFG,
             messages=[_msg("user", "Implement separation, alignment, cohesion functions")]),
    RLMState(agent_id="root.script_js.renderer", query="Write canvas renderer",
             status=Status.READY, config=GRANDCHILD_CFG,
             messages=[_msg("user", "Write the canvas drawing loop for boids")]),
    RLMState(agent_id="root.script_js.controls", query="Write UI controls",
             status=Status.READY, config=GRANDCHILD_CFG,
             messages=[_msg("user", "Add sliders for separation/alignment/cohesion weights")]),
]

s3_script = s2_children[2].update(
    status=Status.SUPERVISING, iteration=1,
    event=CodeExec(
        agent_id="root.script_js", iteration=1, suspended=True,
        code='h1 = delegate("boids_core", "Write core boids algorithm")\nh2 = delegate("renderer", "Write canvas renderer")\nh3 = delegate("controls", "Write UI controls")\nyield wait([h1, h2, h3])',
        output="",
    ),
    children=grandchildren,
    waiting_on=["root.script_js.boids_core", "root.script_js.renderer", "root.script_js.controls"],
)

s3_children = [s2_children[0], s2_children[1], s3_script, s2_children[3]]
s3 = s2.update(children=s3_children)

# ═══════════════════════════════════════════════════════════════════════
# Step 4: grandchildren execute — boids_core finished, renderer executing,
#         controls got a NoCodeBlock (LLM forgot code fence)
# ═══════════════════════════════════════════════════════════════════════
gc4 = [
    grandchildren[0].update(
        status=Status.FINISHED, iteration=1,
        event=CodeExec(agent_id="root.script_js.boids_core", iteration=1,
                       code='boids_js = """\nfunction separation(boid, others) { ... }\nfunction alignment(boid, others) { ... }\nfunction cohesion(boid, others) { ... }\nfunction update(boid, others) {\n  let s = separation(boid, others);\n  let a = alignment(boid, others);\n  let c = cohesion(boid, others);\n  boid.vel.x += s.x + a.x + c.x;\n  boid.vel.y += s.y + a.y + c.y;\n}\n"""\ndone(boids_js)',
                       output="<boids core functions>"),
        result="Implemented separation, alignment, cohesion, and update functions",
        messages=[
            _msg("user", "Implement separation, alignment, cohesion functions"),
            _msg("assistant", "```repl\nboids_js = ...\ndone(boids_js)\n```"),
        ],
    ),
    grandchildren[1].update(
        status=Status.EXECUTING, iteration=1,
        event=LLMReply(
            agent_id="root.script_js.renderer", iteration=1,
            text="I'll create a canvas-based renderer with requestAnimationFrame.\n\n```repl\nrenderer_js = ...\ndone(renderer_js)\n```",
            code='renderer_js = """\nconst canvas = document.getElementById(\'boids-canvas\');\nconst ctx = canvas.getContext(\'2d\');\nfunction drawBoid(boid) {\n  ctx.beginPath();\n  ctx.arc(boid.x, boid.y, 3, 0, Math.PI * 2);\n  ctx.fillStyle = boid.color;\n  ctx.fill();\n}\nfunction render(boids) {\n  ctx.clearRect(0, 0, canvas.width, canvas.height);\n  boids.forEach(drawBoid);\n  requestAnimationFrame(() => render(boids));\n}\n"""\ndone(renderer_js)',
        ),
        messages=[
            _msg("user", "Write the canvas drawing loop for boids"),
            _msg("assistant", "I'll create a canvas-based renderer with requestAnimationFrame."),
        ],
    ),
    grandchildren[2].update(
        status=Status.EXECUTING, iteration=1,
        event=NoCodeBlock(
            agent_id="root.script_js.controls", iteration=1,
            text="I'll add range sliders for the three boid parameters. Each slider will control the weight multiplier. Here's my plan:\n\n1. Create a controls div\n2. Add three labeled range inputs\n3. Wire up event listeners to update the global weights",
        ),
        messages=[
            _msg("user", "Add sliders for separation/alignment/cohesion weights"),
            _msg("assistant", "I'll add range sliders for the three boid parameters..."),
        ],
    ),
]

s4_script = s3_script.update(children=gc4)
s4_children = [s3_children[0], s3_children[1], s4_script, s3_children[3]]
s4 = s3.update(children=s4_children)

# ═══════════════════════════════════════════════════════════════════════
# Step 5: all grandchildren finished — controls retried and succeeded
# ═══════════════════════════════════════════════════════════════════════
gc5 = [
    gc4[0],  # boids_core already done
    gc4[1].update(
        status=Status.FINISHED, iteration=1,
        event=CodeExec(agent_id="root.script_js.renderer", iteration=1,
                       code='renderer_js = "..."\ndone(renderer_js)',
                       output="<canvas renderer>"),
        result="Canvas renderer with requestAnimationFrame loop and boid drawing",
    ),
    gc4[2].update(
        status=Status.FINISHED, iteration=2,
        event=CodeExec(
            agent_id="root.script_js.controls", iteration=2,
            code='controls_js = """\nfunction createSlider(label, min, max, value, onChange) {\n  const div = document.createElement(\'div\');\n  div.innerHTML = `<label>${label}: <input type="range" min="${min}" max="${max}" value="${value}"></label>`;\n  div.querySelector(\'input\').addEventListener(\'input\', e => onChange(+e.target.value));\n  return div;\n}\nconst panel = document.getElementById(\'controls\');\npanel.append(createSlider(\'Separation\', 0, 5, 1.5, v => weights.sep = v));\npanel.append(createSlider(\'Alignment\', 0, 5, 1.0, v => weights.ali = v));\npanel.append(createSlider(\'Cohesion\', 0, 5, 1.0, v => weights.coh = v));\n"""\ndone(controls_js)',
            output="<controls code>",
        ),
        result="Slider controls for separation, alignment, and cohesion weights",
        messages=[
            _msg("user", "Add sliders for separation/alignment/cohesion weights"),
            _msg("assistant", "I'll add range sliders for the three boid parameters..."),
            _msg("user", "⚠ No code block found. Please include a ```repl block."),
            _msg("assistant", "```repl\ncontrols_js = ...\ndone(controls_js)\n```"),
        ],
    ),
]

s5_script = s4_script.update(children=gc5)
s5_children = [s4_children[0], s4_children[1], s5_script, s4_children[3]]
s5 = s4.update(children=s5_children)

# ═══════════════════════════════════════════════════════════════════════
# Step 6: script_js resumes, assembles the file, finishes
# ═══════════════════════════════════════════════════════════════════════
s6_script = s5_script.update(
    status=Status.FINISHED, iteration=1,
    event=ResumeExec(agent_id="root.script_js", iteration=1,
                     output='write_file("script.js", boids_core + renderer + controls)\ndone("Created script.js")'),
    result="Created script.js — assembled boids core + renderer + controls",
    waiting_on=[],
)

s6_children = [s5_children[0], s5_children[1], s6_script, s5_children[3]]
s6 = s5.update(children=s6_children)

# ═══════════════════════════════════════════════════════════════════════
# Step 7: root resumes — all children done → root FINISHED
# ═══════════════════════════════════════════════════════════════════════
s7 = s6.update(
    status=Status.FINISHED, iteration=1,
    event=ResumeExec(agent_id="root", iteration=1,
                     output='done("All files created: index.html, style.css, script.js, README.md")'),
    result="Created boids simulation: index.html, style.css, script.js, README.md",
    waiting_on=[],
    messages=s1.messages + [
        _msg("user", "Children finished. Results:\n- index_html: Created index.html\n- style_css: Created style.css\n- script_js: Created script.js (assembled from 3 sub-modules)\n- readme: Created README.md"),
        _msg("assistant", '```repl\ndone("All files created: index.html, style.css, script.js, README.md")\n```'),
    ],
)

states = [s0, s1, s2, s3, s4, s5, s6, s7]

print(f"Generated {len(states)} steps. Launching viewer...")
open_viewer(states, query=QUERY)
