# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff between files information."

import dataclasses as dcls
import pathlib
import re
import typing
from collections import abc as cabc

from rich import markup, syntax

from problolm import diffs

from . import commits, fs

__all__ = ["Delta"]


class _ReadLines(fs.TrieNode.Visitor[list[str]]):
    def visit_file(self, file: fs.File) -> list[str]:
        try:
            return file.read().splitlines()
        except UnicodeDecodeError:
            raise RuntimeError

    def visit_folder(self, folder: fs.Folder) -> typing.NoReturn:
        raise ValueError(f"Does not handle {folder=}")


@dcls.dataclass(frozen=True)
class Delta:
    """
    `Delta` is a `git.Diff` wrapper object, representing the diff between modes.
    """

    older: commits.Commit
    "The older commit."

    older_path: str | None
    "The file in older commit. If none, means that it was newly created."

    newer: commits.Commit
    "The newer commit."

    newer_path: str | None

    def __str__(self) -> str:
        return self.__as_string(rich=False)

    def __rich__(self) -> str:
        return self.__as_string(rich=True)

    @property
    def extension(self):
        path = self.older_path or self.newer_path or ""
        return pathlib.Path(path).suffix[1:]

    @property
    def is_created(self) -> bool:
        return self.older_path is None

    @property
    def is_deleted(self) -> bool:
        return self.newer_path is None

    def _older_text(self) -> cabc.Sequence[str]:
        return _read_lines_from_commit_path(self.older, self.older_path)

    def _newer_text(self) -> cabc.Sequence[str]:
        return _read_lines_from_commit_path(self.newer, self.newer_path)

    def __as_string(self, rich: bool) -> str:
        "Convert `self` to string. If `rich_color` is given, color using `rich` syntax."
        return "\n".join(self.maybe_color_line_diffs(color=rich))

    def maybe_color_line_diffs(self, color: bool):
        text = diffs.unified_diff(
            a=self._older_text(),
            b=self._newer_text(),
            fromfile=self.older_path or "",
            tofile=self.newer_path or "",
        )
        render = self._color_line if color else lambda x: x

        for line in text:
            yield render(line)

    def _color_line(self, line: str):
        if m := _ADD_REGEX.match(line):
            return self.__split_modifier("green", *m.groups())

        if m := _SUB_REGEX.match(line):
            return self.__split_modifier("red", *m.groups())

        if _HUNK_REGEX.match(line):
            return f"[cyan]{markup.escape(line)}[/cyan]"

        return self.__highlight(line)

    def __split_modifier(self, color: str, modifier: str, rest: str) -> str:
        rest = markup.escape(rest)
        return f"[{color}]{modifier}[/{color}]" + self.__highlight(rest)

    def __highlight(self, code: str):
        s = syntax.Syntax(code, lexer=self.extension)
        highlight = str(s.highlight(code))
        return highlight.rstrip("\n")


_ADD_REGEX = re.compile(r"(\+\+\+ |\+)(.*)")
_SUB_REGEX = re.compile(r"(\-\-\- |\-)(.*)")
_HUNK_REGEX = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")


def _read_lines_from_commit_path(
    commit: commits.Commit, path: str | None
) -> cabc.Sequence[str]:
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
