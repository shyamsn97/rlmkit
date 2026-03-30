"""Tiny markdown prompt builder built around sections and ordering.

Override points for subclasses:
- `Section.render()` — change how a single section becomes text
- `PromptBuilder.render_section()` — change how any section value is rendered
- `PromptBuilder.normalize()` — change final whitespace cleanup
- `PromptBuilder.build()` — change the full assembly pipeline
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import re
import textwrap
from typing import Any, Optional


SectionBody = Optional[str]
SectionBodyConstructFn = Callable[[Mapping[str, Any]], Optional[str]]


def markdown_heading(title: str, level: int = 2) -> str:
    return "#" * max(level, 1) + " " + title


def markdown_section(title: str, body: str, level: int = 2) -> str:
    body = body.strip()
    heading = markdown_heading(title, level)
    if not body:
        return heading
    return heading + "\n\n" + body


class Section:
    """One named section of a prompt.

    Subclass and override `render()` to change how a section produces text.
    """

    def __init__(
        self,
        name: str,
        body: SectionBody = None,
        *,
        body_construct_fn: SectionBodyConstructFn | None = None,
        title: Optional[str] = None,
        level: int = 2,
    ) -> None:
        if body is not None and body_construct_fn is not None:
            raise ValueError("Provide either `body` or `body_construct_fn`, not both.")
        self.name = name
        self.body = body
        self.body_construct_fn = body_construct_fn
        self.title = title
        self.level = level

    def render(self, context: Mapping[str, Any]) -> str:
        value = self.resolve_body(context)
        if not self.title:
            return value
        return markdown_section(self.title, value, level=self.level)

    def resolve_body(self, context: Mapping[str, Any]) -> str:
        if self.body_construct_fn is not None:
            return (self.body_construct_fn(context) or "").strip()
        return (self.body or "").strip()


class PromptBuilder:
    """Build markdown from ordered placeholders like `{role}`.

    Subclass and override:
    - `render_section()` to change how individual sections render
    - `normalize()` to change whitespace cleanup
    - `build()` to change the full pipeline
    """

    def __init__(
        self,
        order: str = "",
        sections: Mapping[str, Section | SectionBody | SectionBodyConstructFn] | None = None,
    ) -> None:
        self._order = textwrap.dedent(order).strip()
        self._sections: dict[str, Section | SectionBody | SectionBodyConstructFn] = dict(
            sections or {}
        )

    @property
    def order(self) -> str:
        return self._order

    @property
    def sections(self) -> dict[str, Section | SectionBody | SectionBodyConstructFn]:
        return dict(self._sections)

    def set_order(self, order: str) -> "PromptBuilder":
        self._order = textwrap.dedent(order).strip()
        return self

    def section(
        self,
        name: str,
        body: SectionBody = None,
        *,
        body_construct_fn: SectionBodyConstructFn | None = None,
        title: Optional[str] = None,
        level: int = 2,
    ) -> "PromptBuilder":
        self._sections[name] = Section(
            name, body, body_construct_fn=body_construct_fn, title=title, level=level,
        )
        return self

    def raw(self, name: str, value: SectionBody | SectionBodyConstructFn) -> "PromptBuilder":
        self._sections[name] = value
        return self

    def file(
        self,
        name: str,
        path: str | Path,
        *,
        title: Optional[str] = None,
        level: int = 2,
        encoding: str = "utf-8",
    ) -> "PromptBuilder":
        def _read_file(context: Mapping[str, Any]) -> str:
            base_dir = context.get("base_dir")
            resolved = Path(base_dir) / path if base_dir else Path(path)
            return resolved.read_text(encoding=encoding).strip()

        return self.section(name, body_construct_fn=_read_file, title=title, level=level)

    def update(
        self,
        sections: Mapping[str, Section | SectionBody | SectionBodyConstructFn],
    ) -> "PromptBuilder":
        self._sections.update(sections)
        return self

    def remove(self, name: str) -> "PromptBuilder":
        self._sections.pop(name, None)
        return self

    def clone(self) -> "PromptBuilder":
        return self.__class__(order=self._order, sections=dict(self._sections))

    def build(self, context: Mapping[str, Any] | None = None) -> str:
        ctx = context or {}
        rendered = {
            name: self.render_section(name, value, ctx)
            for name, value in self._sections.items()
        }
        text = self._render_order(rendered)
        return self.normalize(text)

    def render_section(
        self,
        name: str,
        value: Section | SectionBody | SectionBodyConstructFn,
        context: Mapping[str, Any],
    ) -> str:
        if isinstance(value, Section):
            return value.render(context)
        if callable(value):
            return (value(context) or "").strip()
        return (value or "").strip()

    def normalize(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text:
            return text + "\n"
        return ""

    def _render_order(self, rendered: Mapping[str, str]) -> str:
        def _replace(match: re.Match[str]) -> str:
            return rendered.get(match.group(1), "").strip()
        return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _replace, self._order)

    @classmethod
    def from_sections(cls, *names: str) -> "PromptBuilder":
        order = "\n\n".join(f"{{{name}}}" for name in names)
        return cls(order=order)


__all__ = [
    "PromptBuilder",
    "Section",
    "SectionBody",
    "SectionBodyConstructFn",
    "markdown_heading",
    "markdown_section",
]
