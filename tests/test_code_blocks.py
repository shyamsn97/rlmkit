"""Test find_code_blocks and replace_code_block edge cases."""

from rlmkit.utils.utils import find_code_blocks, replace_code_block


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


if __name__ == "__main__":
    print("test_code_blocks:")
    test_standard_fence()
    test_glued_fence()
    test_glued_fence_eof()
    test_nested_backticks()
    test_fence_at_eof()
    test_no_blocks()
    test_replace_standard()
    test_replace_glued()
    test_replace_no_block()
    print("\nAll tests passed.")
