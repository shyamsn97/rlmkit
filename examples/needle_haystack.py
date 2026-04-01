"""Needle in a haystack across many files.

Generates 500 files of random noise in a temp directory. One file
contains a magic string. The agent uses custom tools registered
with @runtime.tool to find it, delegating the search in parallel.

Uses the step-based API for full observability.
"""

from __future__ import annotations

import random
import string
import sys
import tempfile
from pathlib import Path

from rlmkit.llm import OpenAIClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import RLMState


# ── Generate the haystack ───────────────────────────────────────────

def generate_haystack(directory: Path, num_files: int = 500, lines_per_file: int = 200):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    answer = "".join(random.choices(string.digits, k=7))
    needle_file = random.randint(0, num_files - 1)
    needle_line = random.randint(0, lines_per_file - 1)

    for i in range(num_files):
        lines = []
        for j in range(lines_per_file):
            if i == needle_file and j == needle_line:
                lines.append(f"The magic number is {answer}")
            else:
                n = random.randint(3, 8)
                lines.append(" ".join(random.choice(words) for _ in range(n)))
        (directory / f"file_{i:04d}.txt").write_text("\n".join(lines))

    print(f"Needle in file_{needle_file:04d}.txt line {needle_line}")
    return answer


# ── Runtime with custom tools ───────────────────────────────────────

def setup_runtime(workspace: Path) -> LocalRuntime:
    rt = LocalRuntime(workspace=workspace)

    @rt.tool("List files matching a glob pattern.")
    def list_files(pattern: str = "*.txt") -> list[str]:
        return sorted(str(p.relative_to(rt.workspace)) for p in rt.workspace.glob(pattern))

    @rt.tool("Count files matching a glob pattern.")
    def count_files(pattern: str = "*.txt") -> int:
        return len(list(rt.workspace.glob(pattern)))

    @rt.tool("Grep for a regex across files. Pass a list of filenames to scope the search.")
    def grep(pattern: str, files: list[str] | None = None, max_results: int = 20) -> str:
        import re
        regex = re.compile(pattern)
        if files is None:
            targets = sorted(str(p.relative_to(rt.workspace)) for p in rt.workspace.rglob("*.txt"))
        else:
            targets = files
        matches = []
        for f in targets:
            try:
                for i, line in enumerate((rt.workspace / f).read_text().splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{f}:{i}: {line}")
                        if len(matches) >= max_results:
                            return "\n".join(matches)
            except (OSError, UnicodeDecodeError):
                continue
        return "\n".join(matches)

    @rt.tool("Read a file's contents.")
    def read_file(path: str) -> str:
        return (rt.workspace / path).read_text()

    return rt


# ── Step logger ──────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

LOG_PATH = Path(__file__).parent / "needle_haystack_log.md"

def _fmt_event(ev, indent=0):
    """Format a single StepEvent for terminal + markdown. Returns (term_lines, md_lines)."""
    from rlmkit.state import LLMReply, CodeExec, ChildStep, NoCodeBlock
    pad = "  " * indent
    term, md = [], []
    tag = f"{CYAN}{ev.agent_id}{RESET}"
    md_prefix = "  " * indent

    if isinstance(ev, LLMReply):
        term.append(f"{pad}{tag} {BOLD}LLM reply{RESET} (iter {ev.iteration})")
        md.append(f"{md_prefix}**`{ev.agent_id}`** LLM reply (iter {ev.iteration})")
        if ev.code:
            term.append(f"{pad}  {YELLOW}```repl{RESET}")
            for line in ev.code.splitlines():
                term.append(f"{pad}  {line}")
            term.append(f"{pad}  {YELLOW}```{RESET}")
            md.append(f"```repl\n{ev.code}\n```")
        else:
            preview = ev.text[:200].replace("\n", " ")
            term.append(f"{pad}  {DIM}{preview}{'...' if len(ev.text) > 200 else ''}{RESET}")
            md.append(f"{md_prefix}> {preview}{'...' if len(ev.text) > 200 else ''}")

    elif isinstance(ev, CodeExec):
        label = f"{RED}suspended{RESET}" if ev.suspended else f"{GREEN}done{RESET}"
        term.append(f"{pad}{tag} {BOLD}exec{RESET} → {label}")
        md.append(f"{md_prefix}**`{ev.agent_id}`** exec → {'suspended' if ev.suspended else 'done'}")
        out = ev.output.strip()
        if out and out != "(no output)":
            for line in out.splitlines()[:10]:
                term.append(f"{pad}  {DIM}│{RESET} {line}")
            if len(out.splitlines()) > 10:
                term.append(f"{pad}  {DIM}│ ...({len(out.splitlines())} lines){RESET}")
            md.append(f"```\n{out[:2000]}\n```")

    elif isinstance(ev, NoCodeBlock):
        term.append(f"{pad}{tag} {YELLOW}no code block{RESET}")
        md.append(f"{md_prefix}**`{ev.agent_id}`** no code block")

    elif isinstance(ev, ChildStep):
        if ev.agent_finished:
            status = f"{GREEN}finished{RESET}"
            md_status = "agent finished"
        elif ev.all_done:
            status = f"{GREEN}all done{RESET}"
            md_status = "all done"
        else:
            status = "in progress"
            md_status = "in progress"
        term.append(f"{pad}{tag} {BOLD}children{RESET} {status}")
        md.append(f"{md_prefix}**`{ev.agent_id}`** children — {md_status}")
        if ev.exec_output and ev.exec_output.strip():
            for line in ev.exec_output.strip().splitlines()[:10]:
                term.append(f"{pad}  {DIM}│{RESET} {line}")
            md.append(f"```\n{ev.exec_output.strip()[:2000]}\n```")
        for ce in ev.child_events:
            t, m = _fmt_event(ce, indent + 1)
            term.extend(t)
            md.extend(m)

    return term, md

def log_step(step: int, state: RLMState):
    import sys
    from rlmkit.state import ChildStep
    ev = state.event
    if not ev:
        return

    term_lines, md_lines = _fmt_event(ev)
    header = f"{DIM}── step {step} ──{RESET}"
    print(f"\n{header}")
    for line in term_lines:
        print(line)

    if isinstance(ev, ChildStep) and state.children:
        n_done = sum(1 for c in state.children if c.finished)
        total = len(state.children)
        print(f"  {DIM}[{n_done}/{total} done]{RESET}")
        for cs in state.children:
            if cs.finished and cs.result:
                preview = cs.result[:120].replace("\n", " ")
                print(f"  {GREEN}✓{RESET} {CYAN}{cs.agent_id}{RESET} → {preview}")
                md_lines.append(f"- ✓ `{cs.agent_id}` → {cs.result[:200]}")
            elif cs.finished:
                print(f"  {GREEN}✓{RESET} {DIM}{cs.agent_id}{RESET}")
            else:
                print(f"  {YELLOW}·{RESET} {DIM}{cs.agent_id} [{cs.status.value}]{RESET}")

    md = [f"### Step {step}"] + md_lines

    if state.finished and state.result:
        print(f"\n{GREEN}{BOLD}result:{RESET} {state.result[:200]}")
        md.append(f"\n---\n**Result:** {state.result}")

    with open(LOG_PATH, "a") as f:
        f.write("\n".join(md) + "\n\n")
    sys.stdout.flush()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    llm = OpenAIClient()

    LOG_PATH.write_text("# needle_haystack.py trace\n\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        answer = generate_haystack(workspace, num_files=500)
        print(f"Generated 500 files in {workspace}")

        runtime = setup_runtime(workspace)
        agent = RLM(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15, context="context.md"),
            runtime_factory=lambda: setup_runtime(workspace),
        )

        def _progress(aid, child_state, done, total):
            status = child_state.status.value
            msg = f"  {DIM}[{done}/{total}]{RESET} {aid} → {status}"
            print(msg)
            with open(LOG_PATH, "a") as f:
                f.write(f"- [{done}/{total}] `{aid}` → {status}\n")
            sys.stdout.flush()

        agent.on_child_stepped = _progress

        state = agent.start(
            "There are 500 text files in the workspace (file_0000.txt to file_0499.txt). "
            "Exactly one line across all files says 'The magic number is XXXXXXX'. "
            "Find it. There are too many files to grep all at once — "
            "split them into batches and delegate each batch to a sub-agent."
        )

        step = 0
        while not state.finished:
            state = agent.step(state)
            step += 1
            log_step(step, state)

        print(f"\n{'='*40}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in (state.result or '')}")

        print(f"Trace saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
