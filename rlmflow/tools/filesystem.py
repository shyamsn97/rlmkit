"""Built-in filesystem tools.

These are pure functions that operate relative to the current working
directory.  The caller is responsible for ``os.chdir(workspace)`` or
setting the cwd on the runtime before registering them.

Usage::

    from rlmflow.tools import FILE_TOOLS

    runtime.register_tools(FILE_TOOLS)
"""

from __future__ import annotations

import re
from pathlib import Path

from rlmflow.tools import tool


@tool("Read a file and return its contents.")
def read_file(path: str) -> str:
    return Path(path).read_text()


@tool("Write content to a file, creating directories if needed.")
def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


@tool("Append content to a file.")
def append_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(content)
    return f"Appended {len(content)} bytes to {path}"


@tool("Find-and-replace edits. Each edit is (old, new).")
def edit_file(path: str, *edits: tuple[str, str]) -> str:
    p = Path(path)
    text = p.read_text()
    count = 0
    for old, new in edits:
        if old in text:
            text = text.replace(old, new, 1)
            count += 1
    p.write_text(text)
    return f"Applied {count}/{len(edits)} edits to {path}"


@tool("List files and directories.")
def ls(path: str = ".") -> list[str]:
    p = Path(path)
    if p.is_file():
        return [p.name]
    return sorted(entry.name for entry in p.iterdir())


@tool("Read lines start:end (0-indexed, exclusive) from a file.")
def read_lines(path: str, start: int, end: int) -> str:
    return "\n".join(Path(path).read_text().splitlines()[start:end])


@tool("Count the number of lines in a file.")
def line_count(path: str) -> int:
    return len(Path(path).read_text().splitlines())


@tool("List files matching a glob pattern.")
def list_files(pattern: str = "*.txt") -> list[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


@tool("Count files matching a glob pattern.")
def count_files(pattern: str = "*.txt") -> int:
    return len(list(Path(".").glob(pattern)))


@tool("Search for lines matching a regex pattern.")
def grep(pattern: str, path: str = ".", *, max_results: int = 50) -> str:
    p = Path(path)
    regex = re.compile(pattern)
    matches: list[str] = []
    files = [p] if p.is_file() else sorted(p.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        try:
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{f}:{i}: {line}")
                    if len(matches) >= max_results:
                        return "\n".join(matches)
        except (UnicodeDecodeError, PermissionError):
            continue
    return "\n".join(matches)


FILE_TOOLS = [
    read_file,
    write_file,
    append_file,
    edit_file,
    ls,
    read_lines,
    line_count,
    list_files,
    count_files,
    grep,
]
