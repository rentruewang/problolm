# Copyright (c) ProBloLM Authors - All Rights Reserved

"Do the show command."

import logging
import typing
from argparse import ArgumentParser
from typing import Any, Protocol

import rich

from . import repo

__all__ = ["git_show"]

LOGGER = logging.getLogger(__name__)


def git_show(sha: str) -> None:
    LOGGER.debug("Parsing commit hash: %s", sha)

    repository = repo.repo()
    commit = repository.commit(sha)

    rich.print(f"Commit: {commit.hexsha}")
    rich.print(f"Author: {commit.author.name} <{commit.author.email}>")
    rich.print(f"Date: {commit.committed_datetime}")
    rich.print("\nMessage:")
    rich.print(commit.message)

    # Diff (like git show)
    rich.print("\nDiff:")

    for parent in commit.parents:
        diffs = parent.diff(commit, create_patch=True)
        for diff in diffs:
            rich.print(f"\n--- {diff.a_path} -> {diff.b_path}")
            rich.print(_decode(diff.diff))


def _decode(item: Any) -> str:
    match item:
        case str():
            return item

        case bytes():
            return item.decode()

        case _:
            return str(item)


if __name__ == "__main__":

    class Args(Protocol):
        sha: str

    @typing.no_type_check
    def parse_args() -> Args:
        parser = ArgumentParser()
        parser.add_argument("sha", type=str)
        return parser.parse_args()

    args = parse_args()
    git_show(args.sha)
