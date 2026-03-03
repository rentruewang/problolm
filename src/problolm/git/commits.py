# Copyright (c) ProBloLM Authors - All Rights Reserved

"The commits class."

import dataclasses as dcls
import functools
import logging
from enum import StrEnum
from enum import auto as Auto
from typing import Any

import fire
import rich

from . import repos

__all__ = ["Commit"]

LOGGER = logging.getLogger(__name__)


class CommitType(StrEnum):
    ROOT = Auto()
    LINEAR = Auto()
    MERGE = Auto()


@dcls.dataclass(frozen=True)
class _RepoBase:
    repo = "."
    "The path for the repo"

    @functools.cached_property
    def _repo(self):
        return repos.repo(self.repo)


@dcls.dataclass(frozen=True)
class _CommitBase:

    sha: str
    "The sha of the commit."


@dcls.dataclass(frozen=True)
class Commit(_CommitBase, _RepoBase):
    "The object for the commits."

    @functools.cached_property
    def git(self):
        return self._repo.commit(self.sha)

    @property
    def parent(self):
        if self.type != CommitType.LINEAR:
            raise ValueError(f"{self.type} should not be a merge commit.")

        return self.git.parents[0]

    @property
    def diff(self):
        return self.parent.diff(self.git, create_patch=True)

    def show(self):
        LOGGER.debug("Parsing commit hash: %s", self.sha)

        commit = self.git

        sb: list[str] = []
        sb.append(f"Commit: {commit.hexsha}")
        sb.append(f"Author: {commit.author.name} <{commit.author.email}>")
        sb.append(f"Date: {commit.committed_datetime}")
        sb.append("")
        sb.append("Message:")
        sb.append(f"{commit.message!s}")

        # Diff (like git show)
        sb.append("")
        sb.append("Diff:")

        for diff in self.diff:
            sb.append("")
            sb.append(f"--- {diff.a_path} -> {diff.b_path}")
            sb.append(_decode(diff.diff))

        rich.print("\n".join(sb))

    @property
    def type(self) -> CommitType:
        match len(self.git.parents):
            case 0:
                return CommitType.ROOT
            case 1:
                return CommitType.LINEAR
            case _:
                return CommitType.MERGE


@dcls.dataclass(frozen=True)
class _CommitRangeBase:
    sha_start: str
    sha_end: str


@dcls.dataclass(frozen=True)
class CommitRange(_CommitRangeBase, _RepoBase):
    pass


def head_commit() -> Commit:
    "Get the commit at the HEAD."
    return Commit(repos.repo().head.commit.hexsha)


def git_show_cmd() -> None:
    "The show command that is exposed publically via [project.scripts]."

    def show(sha: str = ""):
        commit = Commit(sha) if sha else head_commit()
        commit.show()

    fire.Fire(show)


def _decode(item: Any) -> str:
    match item:
        case str():
            return item

        case bytes():
            return item.decode()

        case _:
            return str(item)
