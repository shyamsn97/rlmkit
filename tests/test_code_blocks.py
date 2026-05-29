"""Test find_code_blocks and replace_code_block edge cases."""

from rlmflow.utils.code import check_wait_syntax, find_code_blocks, replace_code_block


def test_standard_fence():
    text = "text\n\n```repl\nx = 42\ndone(x)\n```\n"
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "done(x)" in blocks[0]
    print("  standard fence: OK")


def test_glued_fence():
    """Closing ``` on the same line as last code (no preceding newline)."""
    text = "text\n\n```repl\ncontent = 'hi'\ndone('path')```\n"
    blocks = find_code_blocks(text)
    assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
    assert "done('path')" in blocks[0]
    print("  glued fence: OK")


def test_glued_fence_eof():
    """Closing ``` glued to last line, no trailing newline."""
    text = "text\n\n```repl\ndone('ok')```"
    blocks = find_code_blocks(text)
    assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
    assert "done('ok')" in blocks[0]
    print("  glued fence EOF: OK")


def test_nested_backticks():
    """Inner markdown fences inside a Python string."""
    text = (
        "```repl\n"
        'content = """\n'
        "```bash\n"
        "pip install foo\n"
        "```\n"
        '"""\n'
        "done(content)\n"
        "```\n"
    )
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert "done(content)" in blocks[0]
    assert "```bash" in blocks[0]
    print("  nested backticks: OK")


def test_fence_at_eof():
    """Closing fence at end of string with no trailing newline."""
    text = "```repl\nx = 1\n```"
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0] == "x = 1"
    print("  fence at EOF: OK")


def test_no_blocks():
    assert find_code_blocks("just some text") == []
    print("  no blocks: OK")


def test_bare_repl_label_is_not_a_code_block():
    assert find_code_blocks("repl\nprint('missing fences')") == []
    print("  bare repl label rejected: OK")


def test_replace_standard():
    text = "Here:\n\n```repl\nold_code()\n```\n\nMore text.\n"
    result = replace_code_block(text, "new_code()")
    assert "new_code()" in result
    assert "More text" not in result
    print("  replace standard: OK")


def test_replace_glued():
    text = "Here:\n\n```repl\nold()```\n"
    result = replace_code_block(text, "new()")
    assert "new()" in result
    print("  replace glued: OK")


def test_replace_no_block():
    text = "no block here"
    assert replace_code_block(text, "x") == text
    print("  replace no block: OK")


def test_wait_check_accepts_direct_await():
    assert check_wait_syntax("x = await rlm_wait(h)") is None
    assert check_wait_syntax("await rlm_wait(*handles)") is None
    print("  await direct call: OK")


def test_wait_check_accepts_conditional():
    code = "results = await rlm_wait(*handles) if handles else []"
    assert check_wait_syntax(code) is None
    print("  await ternary: OK")


def test_wait_check_rejects_yield():
    err = check_wait_syntax("x = yield rlm_wait(h)")
    assert err is not None and "top-level `yield` is not supported" in err
    print("  yield rejected: OK")


def test_wait_check_rejects_naked_wait():
    err = check_wait_syntax("results = rlm_wait(h)")
    assert err is not None and "must be awaited" in err
    err = check_wait_syntax("results = rlm_wait(*handles) if handles else []")
    assert err is not None
    print("  naked wait rejected: OK")


def test_wait_check_rejects_wait_in_comprehension():
    err = check_wait_syntax("[await rlm_wait(h) for h in handles]")
    assert err is not None
    print("  comprehension wait rejected: OK")


if __name__ == "__main__":
    print("test_code_blocks:")
    test_standard_fence()
    test_glued_fence()
    test_glued_fence_eof()
    test_nested_backticks()
    test_fence_at_eof()
    test_no_blocks()
    test_bare_repl_label_is_not_a_code_block()
    test_replace_standard()
    test_replace_glued()
    test_replace_no_block()
    test_wait_check_accepts_direct_await()
    test_wait_check_accepts_conditional()
    test_wait_check_rejects_yield()
    test_wait_check_rejects_naked_wait()
    test_wait_check_rejects_wait_in_comprehension()
    print("\nAll tests passed.")
