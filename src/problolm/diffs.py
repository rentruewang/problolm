# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls

from .commits import Commit

__all__ = ["CommitDiff"]


@dcls.dataclass(frozen=True)
class CommitDiff:
    newer: Commit
    older: Commit

    @property
    def git(self):
        return self.newer.git.diff(self.older.git, create_patch=True)
