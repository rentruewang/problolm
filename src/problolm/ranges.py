# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
import logging

from .commits import Commit
from .deltas import Delta

__all__ = ["CommitRange"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class CommitRange:
    "The commit diff."

    newer: Commit
    """
    The LHS of the `-` equation. Inclusive.
    """

    older: Commit
    """
    The RHS of the `-` equation. Exclusive.
    """

    def __str__(self) -> str:
        return f"{self.older!s}..{self.newer!s}"

    def __repr__(self) -> str:
        num_changes = len(self.git)
        return f"CommitDiff[{num_changes}]({self.older!s}..{self.newer!s})"

    def __len__(self) -> int:
        return len(self.git)

    def __getitem__(self, idx: int) -> Delta:
        diff = self.git[idx]
        return Delta(
            older=self.older,
            newer=self.newer,
            older_path=diff.a_path,
            newer_path=diff.b_path,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other: object):
        match other:
            case CommitRange():
                return self.newer == other.newer and self.older == other.older
            case Commit():
                return self.newer == other and self.older == other.parent

        return NotImplemented

    @property
    def git(self):
        return self.newer.git.diff(self.older.git, create_patch=True)

    @property
    def original_paths(self) -> set[str]:
        return {delta.older_path for delta in self if delta.older_path}

    @property
    def updated_paths(self) -> set[str]:
        return {delta.newer_path for delta in self if delta.newer_path}

    @property
    def is_linear(self):
        """
        Return if the begin..end commits are linear.
        """

        return self.newer.same_lineage(self.older)
