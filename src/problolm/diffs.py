# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
import typing
from collections.abc import Generator
from typing import Any

if typing.TYPE_CHECKING:
    from git import Diff as _Diff

    from .commits import Commit

__all__ = ["CommitDiff"]


@dcls.dataclass(frozen=True)
class Delta:
    diff: "_Diff"

    def __str__(self) -> str:
        sb = []
        sb.append(f"--- {self.diff.a_path}")
        sb.append(f"+++ {self.diff.b_path}")
        sb.extend(_diff_lines(_decode(self.diff.diff)))
        return "\n".join(sb)


@dcls.dataclass(frozen=True)
class CommitDiff:
    newer: "Commit"
    older: "Commit"

    def __iter__(self) -> Generator[Delta]:
        for delta in self._git:
            yield Delta(delta)

    @property
    def _git(self):
        return self.newer._git.diff(self.older._git, create_patch=True)


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


def _diff_lines(diff: str) -> Generator[str]:
    for line in diff.splitlines():
        yield _color_line(line)
