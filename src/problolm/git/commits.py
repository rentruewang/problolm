# Copyright (c) ProBloLM Authors - All Rights Reserved

"The commits class."

from git import Commit as _Commit
import dataclasses as dcls, functools
from . import repo
from git import Repo


@dcls.dataclass(frozen=True)
class Commit:
    "The object for the commits."

    sha: str
    "The sha of the commit."

    repo = "."
    "The path for the repo"

    @functools.cached_property
    def git(self) -> _Commit:
        return self._repo.commit(self.sha)

    @functools.cached_property
    def _repo(self) -> Repo:
        return repo.repo(self.repo)
