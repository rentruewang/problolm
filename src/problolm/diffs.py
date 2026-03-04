# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

from .commits import Commit


class Diff:
    source: Commit
    target: Commit

    @property
    def git(self):
        return self.target.git.diff(self.source.git)
