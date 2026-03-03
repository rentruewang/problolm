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
class Commit:
    "The object for the commits."

    sha: str
    "The sha of the commit."

    repo = "."
    "The path for the repo"

    @functools.cached_property
    def git(self):
        return self._repo.commit(self.sha)

    @functools.cached_property
    def _repo(self):
        return repos.repo(self.repo)

    def show(self):
        LOGGER.debug("Parsing commit hash: %s", self.sha)

        commit = self.git

        rich.print(f"Commit: {commit.hexsha}")
        rich.print(f"Author: {commit.author.name} <{commit.author.email}>")
        rich.print(f"Date: {commit.committed_datetime}")
        rich.print("\nMessage:")
        rich.print(commit.message)

        # Diff (like git show)
        rich.print("\nDiff:")

        for parent in commit.parents:
            diffs = commit.diff(parent, create_patch=True)
            for diff in diffs:
                rich.print(f"\n--- {diff.a_path} -> {diff.b_path}")
                rich.print(_decode(diff.diff))

    @property
    def type(self) -> CommitType:
        match len(self.git.parents):
            case 0:
                return CommitType.ROOT
            case 1:
                return CommitType.LINEAR
            case _:
                return CommitType.MERGE


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
