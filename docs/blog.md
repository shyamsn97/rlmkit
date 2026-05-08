# Recursive Language Models: A graph approach

> [GitHub](https://github.com/shyamsn97/rlmflow) ·
> [PyPI](https://pypi.org/project/rlmflow/) ·
> [Examples](https://github.com/shyamsn97/rlmflow/tree/main/examples) ·
> [Changelog](https://github.com/shyamsn97/rlmflow/blob/main/CHANGELOG.md)

![Hero animation: an rlmflow run unfolding from a single root agent into a tree of typed nodes](rlm_animation.gif)

```bash
pip install rlmflow
```

## tldr

**rlmflow** turns [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) into inspectable execution graphs. It's a Python library for writing RLM agents where every query, action, observation, delegation, wait, resume, and result is a typed, immutable Pydantic node, and a run is just the tree of those snapshots.

The whole engine is one transition: `step(node) → node'`. The trace and the execution are the same data structure — there is no separate "tracing mode" to enable — so the same run renders as a Rich live tree, a Mermaid diagram, a Gantt swimlane, or a Gradio step-through viewer, all from one-line projections of the graph.

That graph allows you to **inspect** each subagent, **replay** from a checkpoint, **fork** from any node, and **edit** a branch before continuing. We'll walk through those moves on a real coding-agent run shipped with the repo.

## Introduction

**Context rot** is the failure mode every practitioner has hit: a
Claude Code session that "gets dumber", a Cursor chat that forgets
the file you opened thirty messages ago, a research agent that can
quote your prompt back but can't *use* it. Anthropic
[defines it](https://www.anthropic.com/news/context-rot) as recall
degrading as the context window grows. Frontier models advertise
200k–1M tokens and in practice degrade long before that — the
tokens fit, the model just can't reason over them all at once. The
easy benchmarks miss this: needle-in-a-haystack tests like RULER are
constant-complexity and frontier models score 90%+, but
[Chroma](https://research.trychroma.com/context-rot),
[OOLONG](https://github.com/oolong-bench/oolong), and
[lost-in-the-middle](https://arxiv.org/abs/2307.03172) all show
real degradation well below the nominal limit.

Existing fixes all bake some decomposition decision into the harness
before the model sees the data. Bigger windows and better positional
encodings (ALiBi, YaRN, ring attention) buy headroom without
addressing rot. Retrieval (vector DBs, BM25, top-k) picks the chunks
for you and falls over on multi-hop. Summarization — Claude Code's
auto-summarize, LangChain's `ConversationSummaryMemory`,
[MemGPT](https://github.com/cpacker/MemGPT) — picks the lossy
compression. The recent **context-folding** thread — [Scaling
Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967),
[AgentFold](https://arxiv.org/abs/2510.24803), [Agentic Context
Engineering](https://arxiv.org/abs/2510.04618) — picks the
branch/return policy. 

Each of these approaches work in practice, but each one is a decomposition
strategy chosen by the system designer in advance, rather than by
the model at run time. This is the pattern [Sutton's *Bitter
Lesson*](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
describes: methods that hard-code human structure win in the short
run and lose in the long run to general methods that scale with
compute. As model capability improves, a fixed decomposition
strategy becomes the ceiling.

[Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/)
flip that. The setup is small: an LLM sits in a Python REPL with
the long context bound **as a variable**, and a single extra
primitive — **`delegate`** — lets it spawn a fresh sub-agent with
its own context window. From there, the model decides for itself
how to peek at the context, slice it, regex through it, or hand a
chunk to a recursive sub-call. Nothing is summarized or delegated
unless the model chooses to. RAG retrieves; RLMs *investigate*. And
because the surface is so small, the strategy itself becomes
another scalar reward — the same RL machinery that taught models to
reason can teach them to manage their own context.

The empirical case is strong: Alex shows RLM(GPT-5-mini) beats raw
GPT-5 drastically on a tough long-context benchmark at roughly the
same API cost, and holds up at 10M+ token corpora that don't fit
any direct baseline. See the
[post](https://alexzhang13.github.io/blog/2025/rlm/),
[paper](https://arxiv.org/abs/2512.24601), and the
[`rlm-minimal`](https://github.com/alexzhang13/rlm-minimal) and
[`verifiers`](https://www.primeintellect.ai/blog/rlm) reference
implementations for the case.

However, as the number of sub-agents grows, that tree becomes much
harder to observe and control: parents spawn children, children
spawn more children, results bubble back up, and a flat transcript
hides almost everything you'd want to ask of the run. But what if there was a better way?

That's where <b>rlmflow</b> comes in -- Representing sprawling trees of recursive agents as inspectable and controllable graphs.

## RLMs are graphs

To better understand what this means, start with the canonical RLM
demo: needle-in-a-haystack. The context is a huge synthetic document,
and the question is simple: what secret code is hidden inside it?

The root agent looks at the document and decides not to read the
whole thing itself. It splits the haystack across a few sub-agents:

- one child scans the first third for the needle phrase,
- another scans the middle third,
- a third child scans the final third, finds several near-matches,
  and spawns two smaller children to inspect the candidate windows,
- a verifier child checks the candidate code against the original
  question,
- the root agent returns the final code.

That is still a small run, but it is already recursive: the root has
children, and one of those children has children of its own. The
important detail is that those children are not single black-box API
calls. Each child is an agent with its own little loop: inspect the
context, run a search, read a passage, maybe delegate again, then
return.

In a minimal RLM-style implementation, every
<span class="rlm-hl-del">delegate(name, query, ctx)</span> call is the
LLM call: it spins up a fresh sub-LLM with its own REPL — bound to
`ctx` as `CONTEXT` — runs that sub-LLM's agent loop until it calls
`done(value)`, and hands the value back as a `str`. A child's REPL
can call <span class="rlm-hl-del">delegate</span> again, and so on.
The parent never sees any of it. Click through:

<style>
.rlm-slides {
  border: 1px solid #30363d;
  border-radius: 12px;
  margin: 1rem 0;
  overflow: hidden;
}
.rlm-slides input {
  display: none;
}
.rlm-slide {
  display: none;
  flex-direction: column;
  min-height: 320px;
  padding: 1rem;
}
.rlm-slide pre {
  margin: 0;
  overflow-x: auto;
}
.rlm-slides-code .rlm-slide {
  background: #0d1117;
  height: 215px;
  padding: 0.5rem 1rem 0.35rem;
}
.rlm-slides-code .rlm-slide h4 {
  border-bottom: 1px solid #21262d;
  margin-bottom: 0.35rem;
  padding-bottom: 0.4rem;
}
.rlm-slides-code .rlm-slide pre {
  background: transparent;
  flex: 0 1 auto;
  font-size: 0.85rem;
  line-height: 1.45;
  margin: auto 0;
  min-height: 0;
  overflow: auto;
  padding: 0;
}
.rlm-slides-code .rlm-slide-nav {
  border-top-color: #21262d;
  margin-top: 0;
  padding-top: 0.4rem;
}
#code-phase-1:checked ~ .rlm-slide-1,
#code-phase-2:checked ~ .rlm-slide-2,
#code-phase-3:checked ~ .rlm-slide-3,
#code-phase-4:checked ~ .rlm-slide-4,
#graph-phase-1:checked ~ .rlm-slide-1,
#graph-phase-2:checked ~ .rlm-slide-2,
#graph-phase-3:checked ~ .rlm-slide-3,
#graph-phase-4:checked ~ .rlm-slide-4,
#graph-phase-5:checked ~ .rlm-slide-5,
#graph-phase-6:checked ~ .rlm-slide-6,
#graph-phase-7:checked ~ .rlm-slide-7,
#graph-phase-8:checked ~ .rlm-slide-8,
#graph-phase-9:checked ~ .rlm-slide-9 {
  display: flex;
}
.rlm-slides-graph {
  left: 50%;
  margin-left: 0;
  margin-right: 0;
  max-width: none;
  position: relative;
  transform: translateX(-50%);
  width: 94vw;
}
.rlm-slides-graph .rlm-slide {
  padding: 1rem 0.75rem 0.5rem;
}
.rlm-slides-graph .rlm-slide h4 {
  padding-left: 0.5rem;
  padding-right: 0.5rem;
}
.rlm-slides-graph .rlm-slide img {
  display: block !important;
  height: auto !important;
  margin: 0.5rem auto !important;
  max-height: none !important;
  max-width: none !important;
  width: 100% !important;
}
.rlm-slide h4 {
  margin-top: 0;
}
.rlm-slide-nav {
  align-items: center;
  border-top: 1px solid #30363d;
  display: flex;
  gap: 0.75rem;
  justify-content: center;
  margin-top: auto;
  padding-top: 0.75rem;
}
.rlm-slide-arrow {
  align-items: center;
  border: 1px solid #30363d;
  border-radius: 999px;
  color: #8b949e;
  cursor: pointer;
  display: inline-flex;
  height: 2rem;
  justify-content: center;
  width: 2rem;
}
.rlm-slide-dots {
  display: flex;
  gap: 0.5rem;
}
.rlm-slide-dot {
  background: #30363d;
  border: 1px solid #484f58;
  border-radius: 999px;
  cursor: pointer;
  height: 0.65rem;
  width: 0.65rem;
}
.rlm-slide-arrow:hover,
.rlm-slide-dot:hover {
  border-color: #58a6ff;
}
.rlm-hl-del {
  background: rgba(88, 166, 255, 0.18);
  border-radius: 4px;
  color: #79c0ff;
  font-weight: 600;
  padding: 0 0.15rem;
}
.rlm-hl-frame {
  background: rgba(255, 180, 84, 0.14);
  border-radius: 4px;
  color: #ffb454;
  font-weight: 600;
  padding: 0 0.2rem;
}
#code-phase-1:checked ~ .rlm-slide .rlm-slide-dot[for="code-phase-1"],
#code-phase-2:checked ~ .rlm-slide .rlm-slide-dot[for="code-phase-2"],
#code-phase-3:checked ~ .rlm-slide .rlm-slide-dot[for="code-phase-3"],
#code-phase-4:checked ~ .rlm-slide .rlm-slide-dot[for="code-phase-4"],
#graph-phase-1:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-1"],
#graph-phase-2:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-2"],
#graph-phase-3:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-3"],
#graph-phase-4:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-4"],
#graph-phase-5:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-5"],
#graph-phase-6:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-6"],
#graph-phase-7:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-7"],
#graph-phase-8:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-8"],
#graph-phase-9:checked ~ .rlm-slide .rlm-slide-dot[for="graph-phase-9"] {
  background: #58a6ff;
  border-color: #58a6ff;
}
</style>

<div class="rlm-slides rlm-slides-code">
  <input checked id="code-phase-1" name="code-slides" type="radio">
  <input id="code-phase-2" name="code-slides" type="radio">
  <input id="code-phase-3" name="code-slides" type="radio">
  <input id="code-phase-4" name="code-slides" type="radio">

  <div class="rlm-slide rlm-slide-1">
    <h4>1. What the root LLM emits in its REPL block</h4>
    <pre><span class="rlm-hl-frame"># In the root delegate — CONTEXT is the haystack, bound as a variable.</span>
n = CONTEXT.line_count()
chunk_0 = <span class="rlm-hl-del">delegate</span>("chunk_0", "scan first third",  CONTEXT.lines(0, n // 3))
chunk_1 = <span class="rlm-hl-del">delegate</span>("chunk_1", "scan middle third", CONTEXT.lines(n // 3, 2 * n // 3))
chunk_2 = <span class="rlm-hl-del">delegate</span>("chunk_2", "scan final third",  CONTEXT.lines(2 * n // 3, n))
done(extract_code([chunk_0, chunk_1, chunk_2]))   # all three are plain str</pre>
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="code-phase-4">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Code phase 1" class="rlm-slide-dot" for="code-phase-1"></label>
        <label aria-label="Code phase 2" class="rlm-slide-dot" for="code-phase-2"></label>
        <label aria-label="Code phase 3" class="rlm-slide-dot" for="code-phase-3"></label>
        <label aria-label="Code phase 4" class="rlm-slide-dot" for="code-phase-4"></label>
      </div>
      <label class="rlm-slide-arrow" for="code-phase-2">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-2">
    <h4>2. A child's REPL can recursively delegate(...) too</h4>
    <pre><span class="rlm-hl-frame"># In the delegate("chunk_2", ...) — its sub-LLM is now the one writing REPL.</span>
hits   = CONTEXT.grep(r"secret|code|passcode|needle").splitlines()
cand_a = <span class="rlm-hl-del">delegate</span>("candidate_a", "Inspect candidate window A.", hits[0])
cand_b = <span class="rlm-hl-del">delegate</span>("candidate_b", "Inspect candidate window B.", hits[1])
done("candidate code 84721")   # the root never sees this code ran</pre>
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="code-phase-1">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Code phase 1" class="rlm-slide-dot" for="code-phase-1"></label>
        <label aria-label="Code phase 2" class="rlm-slide-dot" for="code-phase-2"></label>
        <label aria-label="Code phase 3" class="rlm-slide-dot" for="code-phase-3"></label>
        <label aria-label="Code phase 4" class="rlm-slide-dot" for="code-phase-4"></label>
      </div>
      <label class="rlm-slide-arrow" for="code-phase-3">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-3">
    <h4>3. ...so the call stack nests delegate frames, with no fixed depth</h4>
    <pre><span class="rlm-hl-frame"># Live Python stack while candidate_b's sub-LLM is reasoning:</span>
<span class="rlm-hl-del">delegate</span>("root",         "What secret code is hidden in the haystack?", haystack)
└── <span class="rlm-hl-del">delegate</span>("chunk_2",      "Scan final third...",         final_third)
    └── <span class="rlm-hl-del">delegate</span>("candidate_b", "Inspect candidate window B.", line_77)
# 3 LLM agent loops live at once, each with its own messages and CONTEXT.
# nothing on an inner frame is visible to any frame above it.</pre>
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="code-phase-2">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Code phase 1" class="rlm-slide-dot" for="code-phase-1"></label>
        <label aria-label="Code phase 2" class="rlm-slide-dot" for="code-phase-2"></label>
        <label aria-label="Code phase 3" class="rlm-slide-dot" for="code-phase-3"></label>
        <label aria-label="Code phase 4" class="rlm-slide-dot" for="code-phase-4"></label>
      </div>
      <label class="rlm-slide-arrow" for="code-phase-4">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-4">
    <h4>4. All the root's REPL sees back is three str</h4>
    <pre><span class="rlm-hl-frame"># Back in the root delegate — every delegate(...) above returned a str.</span>
chunk_0 == "not found"
chunk_1 == "decoy, no code"
chunk_2 == "candidate code 84721"
# 6 hidden <span class="rlm-hl-del">delegate</span> frames and dozens of LLM iterations
# have collapsed into 3 strings. if chunk_2 is wrong, the root has no way
# to ask which inner sub-LLM screwed up, or what its CONTEXT even was.</pre>
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="code-phase-3">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Code phase 1" class="rlm-slide-dot" for="code-phase-1"></label>
        <label aria-label="Code phase 2" class="rlm-slide-dot" for="code-phase-2"></label>
        <label aria-label="Code phase 3" class="rlm-slide-dot" for="code-phase-3"></label>
        <label aria-label="Code phase 4" class="rlm-slide-dot" for="code-phase-4"></label>
      </div>
      <label class="rlm-slide-arrow" for="code-phase-1">&rarr;</label>
    </div>
  </div>
</div>

That's the core observability problem with vanilla RLMs: a single
`delegate()` call can hide an entire recursive subtree of LLM work,
and **nothing about that subtree survives the return**. Children can
delegate to children can delegate to children — and all the parent
ever gets is a `list[str]`. When the answer is wrong, you can't tell
*which* level of the recursion went off the rails; when the answer is
right, you can't tell whether it was right for the right reason. The
abstraction is too clean: the act of delegating throws away exactly
the structure you'd want to debug, evaluate, or steer.

rlmflow keeps that structure — every recursive call is a node in an
execution graph that you can step through, inspect, and replay:

<div class="rlm-slides rlm-slides-graph">
  <input checked id="graph-phase-1" name="graph-slides" type="radio">
  <input id="graph-phase-2" name="graph-slides" type="radio">
  <input id="graph-phase-3" name="graph-slides" type="radio">
  <input id="graph-phase-4" name="graph-slides" type="radio">
  <input id="graph-phase-5" name="graph-slides" type="radio">
  <input id="graph-phase-6" name="graph-slides" type="radio">
  <input id="graph-phase-7" name="graph-slides" type="radio">
  <input id="graph-phase-8" name="graph-slides" type="radio">
  <input id="graph-phase-9" name="graph-slides" type="radio">

  <div class="rlm-slide rlm-slide-1">
    <h4>Step 1 / 9 — root receives the query</h4>
    <img alt="root receives the query" src="static/needle_trace_images/step_00.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-9">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-2">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-2">
    <h4>Step 2 / 9 — root delegates 3 chunks and parks in supervising</h4>
    <img alt="root delegates 3 chunks and parks in supervising" src="static/needle_trace_images/step_01.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-1">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-3">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-3">
    <h4>Step 3 / 9 — chunks run; chunk_2 sub-delegates two candidates</h4>
    <img alt="chunks run; chunk_2 sub-delegates two candidates" src="static/needle_trace_images/step_02.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-2">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-4">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-4">
    <h4>Step 4 / 9 — both candidate readers finish</h4>
    <img alt="both candidate readers finish" src="static/needle_trace_images/step_03.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-3">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-5">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-5">
    <h4>Step 5 / 9 — chunk_2 returns "candidate code 84721"</h4>
    <img alt="chunk_2 returns candidate code 84721" src="static/needle_trace_images/step_04.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-4">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-6">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-6">
    <h4>Step 6 / 9 — root resumes with all three chunk results</h4>
    <img alt="root resumes with all three chunk results" src="static/needle_trace_images/step_05.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-5">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-7">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-7">
    <h4>Step 7 / 9 — root delegates to verify and parks again</h4>
    <img alt="root delegates to verify and parks again" src="static/needle_trace_images/step_06.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-6">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-8">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-8">
    <h4>Step 8 / 9 — verify confirms 84721 matches the question</h4>
    <img alt="verify confirms the answer" src="static/needle_trace_images/step_07.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-7">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-9">&rarr;</label>
    </div>
  </div>

  <div class="rlm-slide rlm-slide-9">
    <h4>Step 9 / 9 — root returns 84721</h4>
    <img alt="root returns 84721" src="static/needle_trace_images/step_08.png">
    <div class="rlm-slide-nav">
      <label class="rlm-slide-arrow" for="graph-phase-8">&larr;</label>
      <div class="rlm-slide-dots">
        <label aria-label="Step 1" class="rlm-slide-dot" for="graph-phase-1"></label>
        <label aria-label="Step 2" class="rlm-slide-dot" for="graph-phase-2"></label>
        <label aria-label="Step 3" class="rlm-slide-dot" for="graph-phase-3"></label>
        <label aria-label="Step 4" class="rlm-slide-dot" for="graph-phase-4"></label>
        <label aria-label="Step 5" class="rlm-slide-dot" for="graph-phase-5"></label>
        <label aria-label="Step 6" class="rlm-slide-dot" for="graph-phase-6"></label>
        <label aria-label="Step 7" class="rlm-slide-dot" for="graph-phase-7"></label>
        <label aria-label="Step 8" class="rlm-slide-dot" for="graph-phase-8"></label>
        <label aria-label="Step 9" class="rlm-slide-dot" for="graph-phase-9"></label>
      </div>
      <label class="rlm-slide-arrow" for="graph-phase-1">&rarr;</label>
    </div>
  </div>
</div>

This is the same run, but now the children are not opaque recursive
calls. The root reaches a supervising node and stops; at that moment
the runnable frontier is `root.chunk_0`, `root.chunk_1`, and
`root.chunk_2`. Those children can advance independently, so the graph
shows parallel work without pretending it is one conversation.

Then `root.chunk_2` reaches its own supervising node. The frontier
changes again: now `root.chunk_2.a` and `root.chunk_2.b` are runnable
while both `root` and `root.chunk_2` are parked. When those candidate
readers finish, `root.chunk_2` resumes, returns `84721`, and only then
can `root` resume and verify the final code.

That is the step-by-step execution state. You can pause after any
node, inspect exactly what one child saw, fork from the candidate
reader, or replace a bad child result before the parent resumes. The
flat recursive-call view tells you what returned. The graph tells you
how the answer moved through the run.

rlmflow stores the run in that shape from the beginning. The graph is
not a visualization recovered from a log after the fact. It is the
data model. Every meaningful moment in the run is stored as a typed
node: the initial question, a model step, a tool result, a paused
parent, a resumed parent, a final answer, or an error. The run is the
tree of those snapshots.

The whole engine is one transition:

```python
node = agent.start(query)
while not node.terminal:
    node = agent.step(node)
```

That loop works because a node is a complete checkpoint. It contains
enough information to continue the run from that point, inspect what
led there, or compare it with another branch.

That gives rlmflow its main operations:

- **Inspect** one agent without rereading every sibling's messages.
- **Replay** from a saved node instead of starting the whole run over.
- **Fork** from one point and try a different model, prompt, or
  workspace.
- **Edit** a branch by replacing a bad child result and continuing
  from the parent.

Those operations are hard to bolt onto a flat transcript because the
transcript has already thrown away the structure you need. It can tell
you what happened first and second. It cannot easily tell you which
subtree produced the result the parent used, or where to restart if
only that subtree was wrong.

Recursive agents also fail in graph-shaped ways. One child drifts
from the schema the parent expected. Another solves the right task
against the wrong slice of context. A parent combines two sibling
answers that were never compatible. These are not just bad messages
in a chat log; they are bad edges in a computation.

That is the bet behind rlmflow: once a model can create recursive
work, the run should be stored as recursive state. The graph is not
decoration around the agent. It is the object the agent is building.


## Acknowledgements

Alex Zhang and Omar Khattab for the RLM paper and post — without
which there is no rlmflow. The
[`rlm-minimal`](https://github.com/alexzhang13/rlm-minimal) and
[`ypi`](https://github.com/rawwerks/ypi) codebases for being
readable, hackable, and right; most of the prompt structure was
learned from them. The OOLONG authors and Prime Intellect's
`verifiers` team for the benchmark environments we wrap. Anthropic's
engineering blog for the harness/session/sandbox vocabulary. And
early users who filed the boids-simulation regressions, the
schema-drift confusions, and the "where is `CONTEXT.fork()`
documented?" issues — those reports are what produced the failure-
shapes section.

---

## Citation

```bibtex
@misc{sudhakaran2026rlmflow,
  author       = {Sudhakaran, Shyam},
  title        = {Recursive Language Models are Graphs},
  year         = {2026},
  howpublished = {\url{https://github.com/shyamsn97/rlmflow}}
}
```

— shyam
