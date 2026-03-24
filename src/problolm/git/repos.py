# Copyright (c) ProBloLM Authors - All Rights Reserved

"The global default repo. Since `problolm` will only work on 1 repo, it's global."

import contextlib as ctxl
import logging
import re
import tempfile
import typing
from argparse import ArgumentParser
from collections.abc import Generator
from datetime import datetime as DateTime
from pathlib import Path
from typing import Protocol

from git import Repo
from git.repo.base import Repo

__all__ = ["init_repo", "set_git_repo", "working_git_repo", "RepoIsDirty"]

LOGGER = logging.getLogger(__name__)


_current_git_repo: Repo | None = None
"The global default repo."


class RepoIsDirty(NotImplementedError):
    "Dirty repo is not yet handled."


def init_repo(loc: str = ".") -> Repo:
    """
    Find the git repo that the repository is located.

    Note that this may perform a clone and is thus expensive.
    """

    repo = _init_repo(loc=loc)

    if repo.is_dirty():
        raise RepoIsDirty("You have un-committed changes. This may cause problems.")

    return repo


def _init_repo(loc: str) -> Repo:

    # Local path.
    if Path(loc).exists():
        return _local_repo(loc)
    else:
        LOGGER.info("%s does not exist locally.")

    # Remote path.
    try:
        return _try_clone_remote_repo(loc)
    except Exception:
        LOGGER.info("%s is not a remote git url.")

    raise ValueError(
        f"We cannot handle {loc=}! Must be either a URL or exists on local folder."
    )


_GIT_URL = re.compile(r"(\w+://)(.+@)*([\w\d\.]+)(:[\d]+){0,1}/*(.*)")
"Adapted from https://stackoverflow.com/a/2514986"


def _try_clone_remote_repo(url: str):
    "Try cloning remote repo. Raise error if fail."
    if not (m := _GIT_URL.match(url)):
        raise RuntimeError

    repo_and_host = m.groups()[-1]
    assert isinstance(repo_and_host, str)
    repo_name = repo_and_host.split("/")[-1]

    # Repeated cloning might fail. This way it only fails in 1s window.
    now = DateTime.now().strftime("%Y-%M-%d-%H-%m-%S")
    return Repo.clone_from(url, Path(tempfile.gettempdir()) / f"{repo_name}-{now}")


def _local_repo(folder: str) -> Repo:
    "Find the for the folder. Matches parent directory until `.git` is found."

    LOGGER.info("Lookup up `%s`'s parents to find the git repository.", folder)

    for path in _yield_parents_until_root(folder):
        LOGGER.debug("Visiting %s", path)
        if (path / ".git").exists():
            LOGGER.info("Git repository found: %s", path)
            return Repo(path)

    raise FileNotFoundError("Cannot find git directory from")


def working_git_repo() -> Repo:
    "The repo that we are currently working on."
    if _current_git_repo is not None:
        return _current_git_repo

    raise RuntimeError("You must use `set_git_repo` context manager!")


@ctxl.contextmanager
def set_git_repo(path: str | Path | Repo):
    "Context manager to set the git repo to the target."

    global _current_git_repo

    # The rest of the code all uses `str` for flexibility.
    if isinstance(path, Path):
        path = str(path.resolve())

    original = _current_git_repo

    try:
        _current_git_repo = init_repo(path) if not isinstance(path, Repo) else path
        yield
    finally:
        _current_git_repo = original


def _yield_parents_until_root(path: str, /) -> Generator[Path]:
    folder = Path(path)
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
    print(init_repo(args.path))
