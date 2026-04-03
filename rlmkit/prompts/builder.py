"""Prompt builder: ordered list of named sections with fluent API.

Sections are rendered top-to-bottom. Empty sections are skipped.
Dynamic content is passed as keyword overrides to ``build()``.

Usage::

    builder = (
        PromptBuilder()
        .section("role", "You are a helpful agent.", title="Role")
        .section("tools", title="Tools")  # placeholder
    )

    prompt = builder.build(tools="- read_file(path): Read a file.")
"""

from __future__ import annotations

import re


class Section:
    """One named section of a prompt."""

    __slots__ = ("name", "body", "title", "level")

    def __init__(
        self,
        name: str,
        body: str = "",
        *,
        title: str | None = None,
        level: int = 2,
    ) -> None:
        self.name = name
        self.body = body
        self.title = title
        self.level = level

    def render(self, body_override: str | None = None) -> str:
        text = (body_override if body_override is not None else self.body).strip()
        if not text:
            return ""
        if self.title:
            heading = "#" * max(self.level, 1) + " " + self.title
            return heading + "\n\n" + text
        return text


class PromptBuilder:
    """Ordered list of sections with a fluent API.

    Sections are stored in insertion order. ``build()`` renders them
    top-to-bottom, skipping any that produce empty output. Pass keyword
    arguments to ``build()`` to override section bodies for that single
    render without mutating the builder.
    """

    def __init__(self) -> None:
        self._sections: list[Section] = []

    def section(
        self,
        name: str,
        body: str = "",
        *,
        title: str | None = None,
        level: int = 2,
        before: str | None = None,
        after: str | None = None,
    ) -> PromptBuilder:
        """Add or replace a named section.

        If a section with *name* already exists, it is replaced in-place
        (preserving its position).  Otherwise it is appended, or inserted
        relative to *before* / *after* if given.
        """
        new = Section(name, body, title=title, level=level)

        for i, s in enumerate(self._sections):
            if s.name == name:
                self._sections[i] = new
                return self

        if before:
            for i, s in enumerate(self._sections):
                if s.name == before:
                    self._sections.insert(i, new)
                    return self
        if after:
            for i, s in enumerate(self._sections):
                if s.name == after:
                    self._sections.insert(i + 1, new)
                    return self

        self._sections.append(new)
        return self

    def remove(self, name: str) -> PromptBuilder:
        """Remove a section by name. No-op if not found."""
        self._sections = [s for s in self._sections if s.name != name]
        return self

    @property
    def names(self) -> list[str]:
        """Section names in current order."""
        return [s.name for s in self._sections]

    def get(self, name: str) -> Section | None:
        for s in self._sections:
            if s.name == name:
                return s
        return None

    def build(self, **overrides: str) -> str:
        """Render all sections in order, skip empties.

        Keyword arguments override section bodies for this call only —
        the builder itself is not mutated.
        """
        parts = []
        for s in self._sections:
            override = overrides.get(s.name)
            rendered = s.render(override)
            if rendered.strip():
                parts.append(rendered)
        text = "\n\n".join(parts)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text + "\n" if text else ""


__all__ = ["PromptBuilder", "Section"]
