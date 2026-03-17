# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
import typing
from collections.abc import Generator
from typing import Any

from .commits import Commit

if typing.TYPE_CHECKING:
    from git import Diff as _Diff


__all__ = ["CommitDiff"]


@dcls.dataclass(frozen=True)
class Delta:
    diff: "_Diff"

    def __str__(self) -> str:
        return self._as_string(color=False)

    def __rich__(self):
        return self._as_string(color=True)

    def _as_string(self, color: bool) -> str:
        sb = []
        sb.append(f"--- {self.diff.a_path}")
        sb.append(f"+++ {self.diff.b_path}")
        sb.extend(self._maybe_color_line_diffs(color=color))
        return "\n".join(sb)

    def _maybe_color_line_diffs(self, color: bool):
        text = _decode(self.diff.diff)
        render = _color_line if color else lambda x: x

        for line in text.splitlines():
            yield render(line)


@dcls.dataclass(frozen=True)
class CommitDiff:
    newer: str
    older: str
    repo: str = "."

    def __iter__(self) -> Generator[Delta]:
        for delta in self._git:
            yield Delta(delta)

    @property
    def _git(self):
        return (
            Commit(self.newer).git().diff(Commit(self.older).git(), create_patch=True)
        )

    def _newer_commit(self):
        return Commit(sha=self.newer, repo=self.repo)


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
