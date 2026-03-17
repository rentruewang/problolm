# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
import typing
from typing import Any

from .commits import Commit
from .shas import Sha

if typing.TYPE_CHECKING:
    from git import Diff as _Diff


__all__ = ["CommitDiff"]


@dcls.dataclass(frozen=True)
class Delta:
    diff: "_Diff"

    def __str__(self) -> str:
        return self._as_string(color=False)

    def __rich__(self) -> str:
        return self._as_string(color=True)

    @property
    def original_path(self):
        return self.diff.a_path

    @property
    def updated_path(self):
        return self.diff.b_path

    def _as_string(self, color: bool) -> str:
        sb = []

        if self.original_path:
            sb.append(f"--- {self.original_path}")

        if self.updated_path:
            sb.append(f"+++ {self.updated_path}")

        sb.extend(self._maybe_color_line_diffs(color=color))
        return "\n".join(sb)

    def _maybe_color_line_diffs(self, color: bool):
        text = _decode(self.diff.diff)
        render = _color_line if color else lambda x: x

        for line in text.splitlines():
            yield render(line)


@dcls.dataclass(frozen=True)
class CommitDiff:
    "The commit diff."

    newer: Sha
    """
    The LHS of the ``-`` equation.
    """

    older: Sha
    """
    The RHS of the ``-`` equation.
    """

    def __str__(self):
        newer = Commit(self.newer)
        older = Commit(self.older)
        return f"{older}..{newer}"

    def __repr__(self):
        newer = Commit(self.newer)
        older = Commit(self.older)
        num_changes = len(self.git)
        return f"CommitDiff[{num_changes}]({older!r}..{newer!r})"

    def __len__(self) -> int:
        return len(self.git)

    def __getitem__(self, idx: int) -> Delta:
        return Delta(self.git[idx])

    @property
    def git(self):
        newer = Commit(self.newer)
        return newer.git.diff(str(self.older), create_patch=True)


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
