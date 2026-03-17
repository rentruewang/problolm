# Copyright (c) ProBloLM Authors - All Rights Reserved

"The commits class."

import abc
import contextlib as ctxl
import logging
import typing
from abc import ABC
from enum import StrEnum
from enum import auto as Auto
from typing import Self

import fire
import rich

from . import repos

if typing.TYPE_CHECKING:
    from .diffs import CommitDiff

__all__ = ["CommitLike", "Commit", "CommitRange", "CommitType"]

LOGGER = logging.getLogger(__name__)


_SHORT_SHA_LEN: int = 7
"The number of short SHA characters. Default to 7 (same as git)."

_SHA_LEN = 40
"The length of SHA."


class CommitType(StrEnum):
    ROOT = Auto()
    LINEAR = Auto()
    MERGE = Auto()


class CommitLike(ABC):
    @abc.abstractmethod
    def diff(self) -> "CommitDiff": ...


class Commit(CommitLike):
    "The object for the commits."

    __match_args__ = ("short_sha",)

    def __init__(self, sha: str) -> None:
        self._long_sha = repos.global_repo().commit(sha).hexsha
        "The sha of the commit."

        assert len(self._long_sha) == _SHA_LEN

    def __str__(self) -> str:
        return self.short_sha

    def __sub__(self, other: str | Self):
        from .diffs import CommitDiff

        match other:
            # Is a sha.
            case str():
                return CommitDiff(newer=self, older=Commit(other))

            # Must be in the same repo.
            case Commit():
                return CommitDiff(newer=self, older=other)

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

    def __repr__(self):
        return f"Commit({self!s})"

    def __str__(self):
        return self.short_sha

    @property
    def git(self):
        commit = repos.global_repo().commit(str(self.sha))
        assert self == commit.hexsha
        return commit

    @property
    def parents(self) -> list[Self]:
        return [type(self)(sha=p.hexsha) for p in self.git.parents]

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

    @property
    def is_root(self) -> bool:
        return self.type == CommitType.ROOT

    @typing.override
    def diff(self) -> "CommitDiff":
        return self - self.parent

    def show(self) -> None:
        LOGGER.debug("Parsing commit hash: %s", self)

        rich.print(self._before_diff())
        for delta in self.diff():
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


class CommitRange(CommitLike):
    "The commit range."

    def __init__(self, begin: str, until: str):
        self._begin = Commit(begin)
        "The start commit. Exclusive."

        self._until = Commit(until)
        "The end commit. Inclusive."

    @typing.override
    def diff(self) -> "CommitDiff":
        return self.until - self.begin

    @property
    def begin(self) -> Commit:
        return self._begin

    @property
    def until(self) -> Commit:
        return self._until


def head_commit() -> Commit:
    "Get the commit at the HEAD."
    return Commit(repos.global_repo().head.commit.hexsha)


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
