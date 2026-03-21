# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from git import Diff as _Diff


__all__ = ["GitDelta"]


@dcls.dataclass(frozen=True)
class GitDelta:
    "GitDelta is a `git.Diff` wrapper object that exposes the API to `problolm`."

    diff: _Diff

    def __str__(self) -> str:
        return self._as_string(rich_color=False)

    def __rich__(self) -> str:
        return self._as_string(rich_color=True)

    @property
    def older_path(self):
        return self.diff.a_path

    @property
    def newer_path(self):
        return self.diff.b_path

    def _as_string(self, rich_color: bool) -> str:
        "Convert `self` to string. If `rich_color` is given, color using `rich` syntax."

        sb = []

        if self.older_path:
            sb.append(f"--- {self.older_path}")

        if self.newer_path:
            sb.append(f"+++ {self.newer_path}")

        sb.extend(self._maybe_color_line_diffs(color=rich_color))
        return "\n".join(sb)

    def _maybe_color_line_diffs(self, color: bool):
        text = _decode(self.diff.diff)
        render = _color_line if color else lambda x: x

        for line in text.splitlines():
            yield render(line)


def _decode(item: Any) -> str:
    match item:
        case str():
            return item

        case bytes():
            return item.decode()

        case _:
            return str(item)


def _wrap_style(text: str, style: str | None) -> str:
    if style is None:
        return text

    return f"[{style}] {text} [/{style}]"


def _get_line_style(modifier: str):
    match modifier:
        case "+":
            return "green"
        case "-":
            return "red"
        case _:
            return None


def _color_line(line: str):
    color = _get_line_style(line[0])
    return _wrap_style(line, color)
