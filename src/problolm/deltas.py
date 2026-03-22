# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
import difflib
from collections.abc import Sequence
from typing import NoReturn
import re
from problolm.fs import File, Folder

from .commits import Commit
from .fs import TrieNode

__all__ = ["Delta"]


class _ReadLines(TrieNode.Visitor[list[str]]):
    def visit_file(self, file: File) -> list[str]:
        try:
            return file.read().splitlines()
        except UnicodeDecodeError:
            raise RuntimeError

    def visit_folder(self, folder: Folder) -> NoReturn:
        raise ValueError(f"Does not handle {folder=}")


@dcls.dataclass(frozen=True)
class Delta:
    """
    `Delta` is a `git.Diff` wrapper object, representing the diff between modes.
    """

    older: Commit
    "The older commit."

    older_path: str | None
    "The file in older commit. If none, means that it was newly created."

    newer: Commit
    "The newer commit."

    newer_path: str | None

    def __str__(self) -> str:
        return self._as_string(rich=False)

    def __rich__(self) -> str:
        return self._as_string(rich=True)

    @property
    def is_created(self) -> bool:
        return self.older_path is None

    @property
    def is_deleted(self) -> bool:
        return self.newer_path is None

    def _older_text(self) -> Sequence[str]:
        return _read_lines_from_commit_path(self.older, self.older_path)

    def _newer_text(self) -> Sequence[str]:
        return _read_lines_from_commit_path(self.newer, self.newer_path)

    def _as_string(self, rich: bool) -> str:
        "Convert `self` to string. If `rich_color` is given, color using `rich` syntax."

        return "\n".join(self._maybe_color_line_diffs(color=rich))

    def _maybe_color_line_diffs(self, color: bool):
        text = difflib.unified_diff(
            a=self._older_text(),
            b=self._newer_text(),
            fromfile=self.older_path or "",
            tofile=self.newer_path or "",
        )
        render = _color_line if color else lambda x: x

        for line in text:
            yield render(line)


def _wrap_style(text: str, style: str | None) -> str:
    if style is None:
        return text

    return f"[{style}] {text} [/{style}]"


_ADD_REGEX = re.compile(r"(\+|\+\+\+ ).*")
_SUB_REGEX = re.compile(r"(\-|\-\-\- ).*")
_HUNK_REGEX = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")


def _get_line_style(line: str):
    if _ADD_REGEX.match(line):
        return "green"

    if _SUB_REGEX.match(line):
        return "red"

    if _HUNK_REGEX.match(line):
        return "cyan"

    return None


def _color_line(line: str):
    color = _get_line_style(line)
    return _wrap_style(line, color)


def _read_lines_from_commit_path(commit: Commit, path: str | None) -> Sequence[str]:
    """
    Read text lines from commit and path.
    If the file at the path is binary or not exist, return `()`.

    Raises:
        ValueError: If the path corresponds to a folder.
    """

    if path is None:
        return ()

    # This raises `ValueError` if the path resolves to a folder.
    # This means it would propagate up if it's a folder.
    read_lines = _ReadLines().visit

    try:
        file = commit.fs() / path
        return read_lines(file)
    except RuntimeError:
        return ()
