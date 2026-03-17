# Copyright (c) ProBloLM Authors - All Rights Reserved

"The commits class."

import logging
from enum import StrEnum
from enum import auto as Auto
from typing import Self

import fire
import rich

from . import repos, shas
from .shas import Sha, ShaLike

__all__ = ["Commit", "CommitType", "short_sha_chars"]

LOGGER = logging.getLogger(__name__)


class CommitType(StrEnum):
    ROOT = Auto()
    LINEAR = Auto()
    MERGE = Auto()


class Commit:
    "The object for the commits."

    __match_args__ = ("sha",)

    def __init__(self, sha: ShaLike):
        self._sha = shas.sha(sha)

    def __repr__(self) -> str:
        return f"Commit({self._sha!s})"

    def __sub__(self, other: str | Self):
        from .diffs import CommitDiff

        match other:
            # Is a sha.
            case str():
                return CommitDiff(self.sha, older=shas.sha(other))

            # Must be in the same repo.
            case Commit(sha=sha):
                return CommitDiff(newer=self.sha, older=sha)

        raise ValueError(f"{self=!r} incompatible with {other=!r}")

    def __rsub__(self, other: str):
        match other:
            case str():
                return Commit(other) - self

        raise NotImplementedError(type(other))

    @property
    def sha(self) -> Sha:
        return self._sha

    @property
    def git(self):
        commit = repos.global_repo().commit(str(self.sha))
        assert self._sha == commit.hexsha
        return commit

    @property
    def parents(self) -> list[Self]:
        return [type(self)(sha=p.hexsha) for p in self.git.parents]

    @property
    def parent(self) -> Self:
        if self.type != CommitType.LINEAR:
            raise ValueError(f"{self.type} should not be a merge commit.")

        parent, *_ = self.parents
        return parent

    @property
    def diff(self):
        return self - self.parent

    def show(self):
        LOGGER.debug("Parsing commit hash: %s", self.sha)

        rich.print(self._before_diff())
        for diff in self.diff:
            rich.print(diff)

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


def head_commit() -> Commit:
    "Get the commit at the HEAD."
    return Commit(repos.global_repo().head.commit.hexsha)


def git_show_cmd() -> None:
    "The show command that is exposed publically via [project.scripts]."

    def show(sha: str = ""):
        commit = Commit(sha) if sha else head_commit()
        commit.show()

    fire.Fire(show)
