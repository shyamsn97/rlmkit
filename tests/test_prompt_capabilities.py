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
        "tools",
        "context",
        "recursion",
        "session",
        "core_examples",
        "status",
    ]


def test_baseline_prompt_section_order_drops_recursion_and_session():
    assert BASELINE_BUILDER.names == [
        "role",
        "repl",
        "tools",
        "context",
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
    # The prompt must make it clear that `done(answer)` is the only way the
    # parent / user sees the result. Don't pin the exact wording.
    assert (
        "what the parent/user sees" in prompt
        or "what the parent / user sees" in prompt
        or "what the user sees" in prompt
    )


def test_default_prompt_teaches_recursion():
    prompt = DEFAULT_BUILDER.build(tools="- read_file: ...", status="depth 0").lower()
    # The default prompt must motivate recursion ("why recurse?") and document the
    # core delegate / wait / verify protocol. We don't pin a dedicated Strategy
    # section anymore — the same ideas live in role/repl/recursion.
    for needle in (
        "delegate(",          # the API
        "yield wait(",        # the suspension protocol
        "context",            # context contract
        "verify",             # verify-before-done discipline
        "fresh context",      # the "why recurse?" framing
        "inline",             # inline is the alternative to delegate
    ):
        assert needle in prompt, f"prompt missing landmark: {needle!r}"


def test_baseline_prompt_documents_context_and_done():
    prompt = BASELINE_BUILDER.build(tools="- read_file: ...", status="depth 0")
    assert "CONTEXT.info()" in prompt
    assert "Final answer:" in prompt
    assert "done(answer)" in prompt
    # No delegation in baseline.
    assert "delegate(" not in prompt
    assert "SESSION." not in prompt


