"""Shared helpers for provider-backed sandbox runtimes."""

from __future__ import annotations

from collections.abc import Callable


def optional_dependency_error(provider: str, extra: str) -> str:
    return (
        f"{provider} requires an optional dependency. "
        f"Install it with `pip install rlmflow[{extra}]`."
    )


def command_output(
    result: object,
    provider_name: str,
    *,
    stdout_getter: Callable[[object], str] | None = None,
    stderr_getter: Callable[[object], str] | None = None,
) -> str:
    """Return stdout or raise a consistent provider command error."""

    stdout = stdout_getter(result) if stdout_getter else _string_attr(result, "stdout")
    stderr = (
        stderr_getter(result)
        if stderr_getter
        else _first_string_attr(
            result,
            ("stderr", "error"),
        )
    )
    exit_code = getattr(result, "exit_code", 0)
    if exit_code:
        raise RuntimeError(
            f"{provider_name} command failed ({exit_code}): {stderr or stdout}"
        )
    return stdout


def _first_string_attr(result: object, attrs: tuple[str, ...]) -> str:
    for attr in attrs:
        value = _string_attr(result, attr)
        if value:
            return value
    return ""


def _string_attr(result: object, attr: str) -> str:
    value = getattr(result, attr, "")
    if value is None:
        return ""
    return str(value)


__all__ = ["command_output", "optional_dependency_error"]
