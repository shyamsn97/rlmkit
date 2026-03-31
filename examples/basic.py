"""Needle-in-a-haystack with recursive chunking.

Generates 1M lines of noise with a single hidden "magic number."
The agent can't fit it all in one context window, so it chunks
the data and delegates each chunk to a sub-agent in parallel.

Tools are added dynamically — the runtime starts bare, and we
register only what the agent needs for this specific task.

Uses the step-based API to show full state after each step.

Inspired by https://github.com/alexzhang13/rlm-minimal/blob/main/main.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

from rlmkit.llm import LLMClient
from rlmkit.rlm import RLM, RLMConfig
from rlmkit.runtime.local import LocalRuntime
from rlmkit.state import RLMState
from rlmkit.utils import tool


# ── LLM client (pick one) ───────────────────────────────────────────

try:
    from openai import OpenAI

    class OpenAIClient(LLMClient):
        def __init__(self, model: str = "gpt-4o"):
            self.client = OpenAI()
            self.model = model

        def chat(self, messages: list[dict[str, str]]) -> str:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
            )
            return resp.choices[0].message.content or ""

except ImportError:
    OpenAIClient = None  # type: ignore[misc,assignment]


try:
    import anthropic

    class AnthropicClient(LLMClient):
        def __init__(self, model: str = "claude-sonnet-4-20250514"):
            self.client = anthropic.Anthropic()
            self.model = model

        def _split_messages(self, messages):
            system = ""
            chat_msgs = []
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                else:
                    chat_msgs.append(m)
            return system, chat_msgs

        def chat(self, messages: list[dict[str, str]]) -> str:
            system, chat_msgs = self._split_messages(messages)
            resp = self.client.messages.create(
                model=self.model, max_tokens=4096,
                system=system, messages=chat_msgs,
            )
            return resp.content[0].text

        def stream(self, messages: list[dict[str, str]]):
            system, chat_msgs = self._split_messages(messages)
            with self.client.messages.stream(
                model=self.model, max_tokens=4096,
                system=system, messages=chat_msgs,
            ) as s:
                yield from s.text_stream

except ImportError:
    AnthropicClient = None  # type: ignore[misc,assignment]


# ── Generate the haystack ───────────────────────────────────────────

def generate_haystack(num_lines: int = 1_000_000) -> tuple[str, str]:
    """Return (haystack_text, answer). The answer is hidden on one line."""
    words = ["blah", "random", "text", "data", "content", "noise", "sample", "filler"]
    answer = str(random.randint(1_000_000, 9_999_999))

    lines = []
    for _ in range(num_lines):
        n = random.randint(3, 8)
        lines.append(" ".join(random.choice(words) for _ in range(n)))

    position = random.randint(num_lines // 3, 2 * num_lines // 3)
    lines[position] = f"The magic number is {answer}"
    print(f"Needle at line {position:,} of {num_lines:,}")

    return "\n".join(lines), answer


# ── Step logger ──────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

LOG_PATH = Path(__file__).parent / "basic_log.md"

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
        status = f"{GREEN}all done{RESET}" if ev.all_done else "in progress"
        term.append(f"{pad}{tag} {BOLD}children{RESET} {status}")
        md.append(f"{md_prefix}**`{ev.agent_id}`** children — {'all done' if ev.all_done else 'in progress'}")
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
    if OpenAIClient is not None:
        llm = OpenAIClient()
    elif AnthropicClient is not None:
        llm = AnthropicClient()
    else:
        raise RuntimeError("pip install openai anthropic")

    LOG_PATH.write_text("# basic.py trace\n\n")
    haystack, answer = generate_haystack(num_lines=1_000_000)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        haystack_path = workspace / "haystack.txt"
        haystack_path.write_text(haystack)
        print(f"Wrote {len(haystack):,} chars to haystack.txt")

        runtime = LocalRuntime(workspace=workspace)

        @tool("Read lines start:end (0-indexed) from a file. Returns the text.")
        def read_lines(path: str, start: int, end: int) -> str:
            p = workspace / path
            lines = p.read_text().splitlines()
            return "\n".join(lines[start:end])

        @tool("Count the number of lines in a file.")
        def line_count(path: str) -> int:
            p = workspace / path
            return len(p.read_text().splitlines())

        @tool("Search for a regex pattern in a line range. Returns matching lines.")
        def search_lines(path: str, pattern: str, start: int = 0, end: int = -1) -> str:
            import re
            p = workspace / path
            lines = p.read_text().splitlines()
            chunk = lines[start:end] if end > 0 else lines[start:]
            regex = re.compile(pattern)
            matches = []
            for i, line in enumerate(chunk, start=start):
                if regex.search(line):
                    matches.append(f"line {i}: {line}")
            return "\n".join(matches)

        runtime.register_tool(read_lines)
        runtime.register_tool(line_count)
        runtime.register_tool(search_lines)

        agent = RLM(
            llm_client=llm,
            runtime=runtime,
            config=RLMConfig(max_depth=3, max_iterations=15, context_path="context.md"),
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
            "I'm looking for a magic number in haystack.txt. What is it? You can only read around 50000 lines at a time."
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
