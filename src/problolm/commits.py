# Copyright (c) ProBloLM Authors - All Rights Reserved

"The commits class."

import contextlib as ctxl
import functools
import logging
import typing
from collections.abc import Generator
from enum import StrEnum
from enum import auto as Auto
from typing import Self

import fire
import rich
from git import BadName

from problolm.fs import File

from . import repos

if typing.TYPE_CHECKING:
    from .ranges import CommitRange

__all__ = ["Commit", "CommitType"]

LOGGER = logging.getLogger(__name__)


_SHORT_SHA_LEN: int = 7
"The number of short SHA characters. Default to 7 (same as git)."

_SHA_LEN = 40
"The length of SHA."


class CommitType(StrEnum):
    ROOT = Auto()
    LINEAR = Auto()
    MERGE = Auto()


class Commit:
    "The object for the commits."

    __match_args__ = ("sha",)

    def __init__(self, sha_like: str) -> None:
        """
        Args:
            sha_like: Things that can be converted to a sha. e.g. HEAD etc. Parsed by `git`.
        """

        try:
            self._long_sha = repos.working_git_repo().commit(sha_like).hexsha
            "The sha of the commit."
        except BadName as bn:
            raise ValueError from bn

        assert len(self._long_sha) == _SHA_LEN

    def __str__(self) -> str:
        return self.short_sha

    def __repr__(self) -> str:
        return f"Commit({self!s})"

    def __sub__(self, other: str | Self):
        from .ranges import CommitRange

        match other:
            # Is a sha.
            case str():
                return CommitRange(newer=self, older=Commit(other))

            # Must be in the same repo.
            case Commit():
                return CommitRange(newer=self, older=other)

        raise ValueError(f"{self=!r} incompatible with {other=!r}")

    def __rsub__(self, other: str):
        match other:
            case str():
                return Commit(other) - self

        raise NotImplementedError(type(other))

    def __eq__(self, sha: object) -> bool:
        match sha:
            # Check if it's prefix.
            case str():
                return self._long_sha.startswith(sha)

            case Commit():
                return self._long_sha == sha._long_sha

        return NotImplemented

    def list_files(self) -> Generator[File]:
        file_system = self.fs()
        yield from file_system.list_files()

    @property
    def git(self):
        commit = repos.working_git_repo().commit(str(self.sha))
        assert self == commit.hexsha
        return commit

    def fs(self):
        "Return the folder structure at the specific commit."

        return self.__fs

    @functools.cached_property
    def __fs(self):
        from . import fs

        return fs.consume(self.git.tree)

    @property
    def parents(self) -> list[Self]:
        return [type(self)(sha_like=p.hexsha) for p in self.git.parents]

    @property
    def parent(self) -> Self:
        if self.type != CommitType.LINEAR:
            raise ValueError(f"{self.type} should not be a merge commit.")

        [parent] = self.parents
        return parent

    def ancestors(self):
        commit = self

        while not commit.is_root:
            yield commit
            commit = commit.parent

        yield commit

    def descendant_of(self, other: str | Commit) -> bool:
        """
        If `self` is descendant of `other`.
        """

        for commit in self.ancestors():
            if commit == other:
                return True

        return False

    def ancestor_of(self, other: str | Commit) -> bool:
        """
        If `self` is ancestor of `other`.
        """

        other = Commit(other) if isinstance(other, str) else other

        return other.descendant_of(self)

    def same_lineage(self, other: str | Commit) -> bool:
        return self.descendant_of(other) or self.ancestor_of(other)

    @property
    def is_root(self) -> bool:
        return self.type == CommitType.ROOT

    def _diff_parent(self) -> CommitRange:
        return self - self.parent

    def show(self) -> None:
        LOGGER.debug("Parsing commit hash: %s", self)

        rich.print(self._before_diff())
        for delta in self._diff_parent():
            rich.print(delta)

    def _before_diff(self) -> str:
        commit = self.git
        sb: list[str] = []
        sb.append(f"")
        sb.append(f"Author: {commit.author.name} <{commit.author.email}>")
        sb.append(f"Date: {commit.committed_datetime}")
        sb.append("")
        sb.append("Message:")
        sb.append(f"{commit.message!s}")

        # Diff (like git show)
        sb.append("")
        sb.append("Diff:")
        return "\n".join(sb)

    @property
    def type(self) -> CommitType:
        match len(self.parents):
            case 0:
                return CommitType.ROOT
            case 1:
                return CommitType.LINEAR
            case _:
                return CommitType.MERGE

    @property
    def sha(self) -> str:
        return self._long_sha

    @property
    def short_sha(self) -> str:
        return self._long_sha[:_SHORT_SHA_LEN]

    @staticmethod
    def set_short_size(size: int):
        return set_short_sha_size(size)


def head_commit() -> Commit:
    "Get the commit at the HEAD."
    return Commit(repos.working_git_repo().head.commit.hexsha)


def git_show_cmd() -> None:
    "The show command that is exposed publically via [project.scripts]."

    def show(sha: str = ""):
        commit = Commit(sha) if sha else head_commit()
        commit.show()

    fire.Fire(show)


@ctxl.contextmanager
def set_short_sha_size(size: int):
    """
    Set the size of short sha.

    Args:
        size: The length of the short sha.
    """

    global _SHORT_SHA_LEN

    if size <= 0 or size > _SHA_LEN:
        raise ValueError(f"The inequality 0 < {size=} < {_SHA_LEN} should be upheld.")

    old_val = _SHORT_SHA_LEN

    try:
        _SHORT_SHA_LEN = size
        yield
    finally:
        _SHORT_SHA_LEN = old_val
