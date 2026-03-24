# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
from collections.abc import Sequence
from difflib import SequenceMatcher

import numpy as np
from numpy.typing import NDArray

__all__ = ["normal_diff", "DnaDiffer"]


def normal_diff(a: Sequence[str], b: Sequence[str], fromfile: str, tofile: str):
    matcher = SequenceMatcher(None, a, b)
    yield f"--- {fromfile}"
    yield f"+++ {tofile}"

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        yield ""
        yield _gen_hunk(i1, i2, j1, j2)

        for line in a[i1:i2]:
            yield f"-{line}"

        for line in b[j1:j2]:
            yield f"+{line}"


def _gen_hunk(i1: int, i2: int, j1: int, j2: int) -> str:
    return f"@@ -{i1+1},{i2-i1} +{j1+1},{j2-j1} @@"


@dcls.dataclass(frozen=True)
class DnaDiffResult[T]:
    left: Sequence[T]
    right: Sequence[T]
    score: NDArray


@dcls.dataclass(frozen=True)
class DnaDiffer:
    equal: float = 1
    diff: float = -1
    gap: float = -1

    def align[T](self, left: Sequence[T], right: Sequence[T]) -> DnaDiffResult[T]:
        score = self.lcs(left, right)
        result_left, result_right = self.traceback(left, right, score)
        return DnaDiffResult(result_left, result_right, score)

    def lcs[T](self, left: Sequence[T], right: Sequence[T]) -> NDArray:
        n, m = len(left), len(right)

        score = np.zeros((n + 1, m + 1), dtype=float)

        # Initialize first row/column
        for i in range(1, n + 1):
            score[i, 0] = score[i - 1, 0] + self.gap

        for j in range(1, m + 1):
            score[0, j] = score[0, j - 1] + self.gap

        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                matching = self.equal if left[i - 1] == right[j - 1] else self.diff

                score[i, j] = max(
                    [
                        score[i - 1, j - 1] + matching,
                        score[i - 1, j] + self.gap,
                        score[i, j - 1] + self.gap,
                    ]
                )

        return score

    def traceback[T](self, seq1: Sequence[T], seq2: Sequence[T], score: NDArray):
        i, j = len(seq1), len(seq2)
        align_left: list[T] = []
        align_right: list[T] = []

        while i > 0 and j > 0:
            matching = self.equal if seq1[i - 1] == seq2[j - 1] else self.diff

            if score[i, j] == score[i - 1, j - 1] + matching:
                align_left.append(seq1[i - 1])
                align_right.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif score[i, j] == score[i - 1, j] + self.gap:
                align_left.append(seq1[i - 1])
                i -= 1
            else:
                align_right.append(seq2[j - 1])
                j -= 1

        return list(reversed(align_left)), list(reversed(align_right))
