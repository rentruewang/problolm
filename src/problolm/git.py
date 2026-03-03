# Copyright (c) ProBloLM Authors - All Rights Reserved

"Custom git objects for easier use with git."

import dataclasses as dcls
import functools
import logging
from collections.abc import Sequence
from enum import StrEnum
from enum import auto as Auto

from git import Commit, Repo

__all__ = ["GitRepo"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class GitRepo:
    "The git repo object."

    path: str
    "The repository."

    @functools.cached_property
    def git(self) -> Repo:
        return Repo(self.path)


class CommitType(StrEnum):
    ROOT = Auto()
    LINEAR = Auto()
    MERGE = Auto()


@dcls.dataclass(frozen=True)
class GitCommit:
    "The object representing each commit."

    repo: GitRepo
    "The repository this commit belongs to."

    sha: str
    "The sha of the commit."

    @functools.cached_property
    def git(self) -> Commit:
        return self.repo.git.commit(self.sha)

    @property
    def parents(self) -> Sequence[Commit]:
        return self.git.parents

    @property
    def parent(self):
        if self.type != CommitType.LINEAR:
            raise AttributeError(f"Not a linear commit. {len(self.parents)=}")

        return self.parents[0]

    @property
    def type(self) -> CommitType:
        match len(self.parents):
            case 0:
                return CommitType.ROOT
            case 1:
                return CommitType.LINEAR
            case _:
                return CommitType.MERGE

    def delta(self):
        return self.parent.diff(self.sha)
