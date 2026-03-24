# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
from collections.abc import Generator, Sequence
from difflib import SequenceMatcher
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

__all__ = ["unified_diff", "DnaDiffer"]


class OpCode(StrEnum):
    REPLACE = "replace"
    DELETE = "delete"
    INSERT = "insert"
    EQUAL = "equal"


@dcls.dataclass(frozen=True)
class LineRange:
    "The line range of a string."

    start: int
    finish: int

    def __post_init__(self) -> None:
        assert self.start >= 0
        assert self.finish >= 0
        assert self.start <= self.finish

    def __len__(self) -> int:
        return self.finish - self.start

    def slice(self, text: Sequence[str]):
        assert self.finish < len(text)
        idx = slice(self.start, self.finish)
        return text[idx]


@dcls.dataclass(frozen=True)
class Op:
    """
    `Op` is an easier to work with representation of `get_opcodes()` output.

    It replaces the 5-tuples with 3 objects:

    `OpCode` enum to show the op.
    Source lines and target lines for easier slicing and validation.
    """

    code: OpCode
    source: LineRange
    target: LineRange

    def __post_init__(self) -> None:
        match self.code:
            # len(lhs) = len(rhs).
            case OpCode.EQUAL:
                if len(self.source) != len(self.target):
                    raise ValueError(f"{self.code=} but {self.source=}, {self.target=}")

            # len(rhs) = 0.
            case OpCode.DELETE:
                if len(self.target):
                    raise ValueError(f"{self.code=} but {self.target=}.")

            # len(lhs) = 0.
            case OpCode.INSERT:
                if len(self.source):
                    raise ValueError(f"{self.code=} but {self.source=}.")

            # Nothing.
            case OpCode.REPLACE:
                pass

    def render(self, left: Sequence[str], right: Sequence[str]) -> Generator[str]:
        left = self.source.slice(left)
        right = self.target.slice(right)

        match self.code:
            case OpCode.DELETE:
                yield self._hunk()
                yield from _render_delete(left)
            case OpCode.INSERT:
                yield self._hunk()
                yield from _render_insert(right)
            case OpCode.REPLACE:
                yield self._hunk()
                yield from _render_delete(left)
                yield from _render_insert(right)
            case OpCode.EQUAL:
                return
                yield

    def _hunk(self):
        return _gen_hunk(
            self.source.start,
            self.source.finish,
            self.target.start,
            self.target.finish,
        )


def difflib_diff(a: Sequence[str], b: Sequence[str]) -> Generator[Op]:
    """
    Yields a generator of `Op`.
    """

    matcher = SequenceMatcher(None, a, b)

    for code, i1, i2, j1, j2 in matcher.get_opcodes():
        yield Op(
            code=OpCode(code),
            source=LineRange(i1, i2),
            target=LineRange(j1, j2),
        )


def unified_diff(a: Sequence[str], b: Sequence[str], fromfile: str, tofile: str):
    diff = difflib_diff(a, b)

    yield f"--- {fromfile}"
    yield f"+++ {tofile}"

    for op in diff:
        rendered = list(op.render(a, b))

        if rendered:
            yield from rendered


def _render_delete(lines: Sequence[str]):
    for l in lines:
        yield "-" + l


def _render_insert(lines: Sequence[str]):
    for l in lines:
        yield "+" + l


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
