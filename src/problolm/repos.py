# Copyright (c) ProBloLM Authors - All Rights Reserved

"Get the repo."

import logging
import typing
from argparse import ArgumentParser
from pathlib import Path
from typing import Protocol

from git import Repo

__all__ = ["repo", "set_global_repo", "global_repo"]

LOGGER = logging.getLogger(__name__)


def repo(folder: str | Path = "."):
    """
    Find the git repo that the repository is located.
    """

    LOGGER.info("Lookup up `%s`'s parents to find the git repository.", folder)

    for path in _yield_parents_until_root(folder=folder):
        LOGGER.debug("Visiting %s", path)
        if (path / ".git").exists():
            LOGGER.info("Git repository found: %s", path)
            return Repo(path)

    raise FileNotFoundError("Cannot find git directory from")


def global_repo() -> Repo:
    if REPO is not None:
        return REPO

    raise RuntimeError("You must call `set_global_repo` first!")


def set_global_repo(path: str | Path) -> None:
    "Set the global default repo to the path."

    global REPO

    REPO = repo(path)


def _yield_parents_until_root(folder: str | Path):
    folder = Path(".")
    yield folder

    while not _folder_is_root(folder):
        folder = folder.parent
        yield folder


def _folder_is_root(path: Path):
    return path == path.parent


if __name__ == "__main__":

    class Args(Protocol):
        path: str

    @typing.no_type_check
    def parse_args() -> Args:
        parser = ArgumentParser()
        parser.add_argument("path", type=str)
        return parser.parse_args()

    args = parse_args()
    print(repo(args.path))


REPO: Repo = repo(".")
"The global default repo."
