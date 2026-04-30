"""Prompt builder: ordered list of named sections, immutable fluent API.

Every mutation method (``.section()``, ``.remove()``) returns a **new**
``PromptBuilder`` — the original is never modified.  This makes it safe
to keep a module-level ``DEFAULT_BUILDER`` and derive from it.

Usage::

    builder = (
        PromptBuilder()
        .section("role", "You are a helpful agent.", title="Role")
        .section("tools", title="Tools")  # placeholder
    )

    prompt = builder.build(tools="- read_file(path): Read a file.")

    # Derive without mutating the original:
    custom = builder.section("role", "You are a security auditor.", title="Role")
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
    """Ordered list of sections with an immutable fluent API.

    ``.section()`` and ``.remove()`` return a **new** builder — the
    original is never mutated.  ``build()`` renders sections
    top-to-bottom, skipping empties.  Pass keyword arguments to
    ``build()`` to override section bodies for that single render.
    """

    def __init__(self) -> None:
        self._sections: list[Section] = []

    def _copy(self) -> PromptBuilder:
        new = PromptBuilder()
        new._sections = list(self._sections)
        return new

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
        """Add or replace a named section. Returns a new builder."""
        out = self._copy()
        new = Section(name, body, title=title, level=level)

        for i, s in enumerate(out._sections):
            if s.name == name:
                out._sections[i] = new
                return out

        if before:
            for i, s in enumerate(out._sections):
                if s.name == before:
                    out._sections.insert(i, new)
                    return out
        if after:
            for i, s in enumerate(out._sections):
                if s.name == after:
                    out._sections.insert(i + 1, new)
                    return out

        out._sections.append(new)
        return out

    def remove(self, name: str) -> PromptBuilder:
        """Remove a section by name. Returns a new builder."""
        out = self._copy()
        out._sections = [s for s in out._sections if s.name != name]
        return out

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

        Keyword arguments override section bodies for this call only.
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
