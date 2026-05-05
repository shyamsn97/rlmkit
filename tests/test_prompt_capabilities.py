"""The default and baseline prompts must always document core capabilities.

``CONTEXT`` and ``SESSION`` are both **always** documented in the system
prompt — they are core RLM concepts and the API must be visible whether
or not the current agent has a non-empty payload. The runtime always
injects both variables; on agents whose data payload is empty,
``CONTEXT.info()["chars"]`` will read 0 but the API still exists.
"""

from __future__ import annotations

from rlmflow.prompts.default import BASELINE_BUILDER, DEFAULT_BUILDER


def test_default_prompt_section_order_is_capabilities_first():
    assert DEFAULT_BUILDER.names == [
        "role",
        "repl",
        "strategy",
        "tools",
        "context",
        "recursion",
        "session",
        "guardrails",
        "core_examples",
        "status",
    ]


def test_baseline_prompt_section_order_drops_recursion_and_session():
    assert BASELINE_BUILDER.names == [
        "role",
        "repl",
        "strategy",
        "tools",
        "context",
        "guardrails",
        "core_examples",
        "status",
    ]


def test_default_prompt_documents_context_unconditionally():
    """The CONTEXT API is core to RLMs — the section is always rendered."""
    prompt = DEFAULT_BUILDER.build(tools="- read_file: ...", status="depth 0")

    assert "## Context" in prompt
    for needle in (
        "CONTEXT.info()",
        "CONTEXT.line_count()",
        "CONTEXT.read(",
        "CONTEXT.lines(",
        "CONTEXT.grep(",
    ):
        assert needle in prompt, f"missing CONTEXT API: {needle!r}"


def test_default_prompt_documents_session_unconditionally():
    prompt = DEFAULT_BUILDER.build(tools="- read_file: ...", status="depth 0")

    for needle in (
        "SESSION.list_agents()",
        "SESSION.read(",
        "SESSION.grep(",
        "SESSION.parent(",
        "SESSION.ancestors(",
        "SESSION.children(",
        "SESSION.subtree(",
        "SESSION.tree()",
    ):
        assert needle in prompt, f"missing SESSION API: {needle!r}"


def test_default_prompt_emphasizes_done_as_final_answer():
    prompt = DEFAULT_BUILDER.build(tools="- read_file: ...", status="depth 0")
    assert "Final answer:" in prompt
    assert "done(answer)" in prompt
    assert "No `done`, no result." in prompt


def test_default_prompt_teaches_size_up_strategy():
    prompt = DEFAULT_BUILDER.build(tools="- read_file: ...", status="depth 0")
    assert "Inline first" in prompt
    assert "Size up" in prompt or "size up" in prompt.lower()
    assert "Search before solving" in prompt
    assert "Delegate" in prompt
    assert "Combine in the parent" in prompt


def test_baseline_prompt_documents_context_and_done():
    prompt = BASELINE_BUILDER.build(tools="- read_file: ...", status="depth 0")
    assert "CONTEXT.info()" in prompt
    assert "Final answer:" in prompt
    assert "done(answer)" in prompt
    # No delegation in baseline.
    assert "delegate(" not in prompt
    assert "SESSION." not in prompt


