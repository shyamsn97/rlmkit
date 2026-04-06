"""Shared step logger for examples."""

from __future__ import annotations

import sys
from pathlib import Path

from rlmkit.state import ChildStep, CodeExec, LLMReply, NoCodeBlock, RLMState, Status

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"


def _fmt_event(ev, indent=0):
    pad = "  " * indent
    term, md = [], []
    tag = f"{CYAN}{ev.agent_id}{RESET}"
    md_pfx = "  " * indent

    if isinstance(ev, LLMReply):
        term.append(f"{pad}{tag} {BOLD}LLM reply{RESET} (iter {ev.iteration})")
        md.append(f"{md_pfx}**`{ev.agent_id}`** LLM reply (iter {ev.iteration})")
        if ev.code:
            lines = ev.code.splitlines()
            term.append(f"{pad}  {YELLOW}```repl{RESET}")
            for line in lines[:15]:
                term.append(f"{pad}  {line}")
            if len(lines) > 15:
                term.append(f"{pad}  {DIM}...({len(lines)} lines){RESET}")
            term.append(f"{pad}  {YELLOW}```{RESET}")
            md.append(f"```repl\n{ev.code}\n```")
        else:
            preview = ev.text[:200].replace("\n", " ")
            term.append(f"{pad}  {DIM}{preview}{'...' if len(ev.text) > 200 else ''}{RESET}")
            md.append(f"{md_pfx}> {preview}")

    elif isinstance(ev, CodeExec):
        if ev.suspended:
            label = f"{MAGENTA}spawning children{RESET}"
            md_label = "spawning children"
        else:
            label = f"{GREEN}done{RESET}"
            md_label = "done"
        term.append(f"{pad}{tag} {BOLD}exec{RESET} → {label}")
        md.append(f"{md_pfx}**`{ev.agent_id}`** exec → {md_label}")
        out = ev.output.strip()
        if out and out != "(no output)":
            out_lines = out.splitlines()
            for line in out_lines[:10]:
                term.append(f"{pad}  {DIM}│{RESET} {line}")
            if len(out_lines) > 10:
                term.append(f"{pad}  {DIM}│ ...({len(out_lines)} lines){RESET}")
            md.append(f"```\n{out[:2000]}\n```")

    elif isinstance(ev, NoCodeBlock):
        term.append(f"{pad}{tag} {YELLOW}no code block{RESET}")
        md.append(f"{md_pfx}**`{ev.agent_id}`** no code block")

    elif isinstance(ev, ChildStep):
        if ev.agent_finished:
            status, md_status = f"{GREEN}finished{RESET}", "finished"
        elif ev.all_done:
            status, md_status = f"{GREEN}children done → resuming{RESET}", "children done → resuming"
        else:
            status, md_status = f"{MAGENTA}supervising{RESET}", "supervising"
        term.append(f"{pad}{tag} {BOLD}supervising{RESET} → {status}")
        md.append(f"{md_pfx}**`{ev.agent_id}`** supervising → {md_status}")
        if ev.exec_output and ev.exec_output.strip():
            for line in ev.exec_output.strip().splitlines()[:10]:
                term.append(f"{pad}  {DIM}│{RESET} {line}")
            md.append(f"```\n{ev.exec_output.strip()[:2000]}\n```")
        for ce in ev.child_events:
            t, m = _fmt_event(ce, indent + 1)
            term.extend(t)
            md.extend(m)

    return term, md


def _status_label(state: RLMState) -> str:
    """Short human-readable label for what the next step will do."""
    n_children = len(state.children)
    match state.status:
        case Status.WAITING:
            return f"{DIM}thinking...{RESET}"
        case Status.HAS_REPLY:
            return f"{DIM}executing...{RESET}"
        case Status.SUPERVISING:
            active = sum(1 for c in state.children if not c.finished)
            return f"{MAGENTA}supervising {active}/{n_children} children...{RESET}"
        case _:
            return ""


class StepLogger:
    """Logs each step to terminal (ANSI) and a markdown file."""

    def __init__(self, log_path: str | Path, live: bool = False):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"# {self.path.stem} trace\n\n")
        self.live = live

    def status(self, state: RLMState):
        """Print a live status line (overwritten by the next call)."""
        if not self.live:
            return
        label = _status_label(state)
        if label:
            sys.stderr.write(f"{CLEAR_LINE}  {label}")
            sys.stderr.flush()

    def _clear_status(self):
        if self.live:
            sys.stderr.write(CLEAR_LINE)
            sys.stderr.flush()

    def log(self, step: int, state: RLMState, extra_term: str = ""):
        self._clear_status()

        ev = state.event
        if not ev:
            return

        term_lines, md_lines = _fmt_event(ev)

        if extra_term:
            term_lines.insert(0, extra_term)

        print(f"\n{DIM}── step {step} ──{RESET}")
        for line in term_lines:
            print(line)

        if isinstance(ev, ChildStep) and state.children:
            n_done = sum(1 for c in state.children if c.finished)
            total = len(state.children)
            print(f"  {DIM}[{n_done}/{total} children done]{RESET}")
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
            print(f"\n{GREEN}{BOLD}result:{RESET} {state.result[:500]}")
            md.append(f"\n---\n**Result:** {state.result}")

        with open(self.path, "a") as f:
            f.write("\n".join(md) + "\n\n")
        sys.stdout.flush()
