# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

import dataclasses as dcls
import difflib
import enum
import typing
from collections import abc as cabc

import numpy as np
from numpy import typing as npt

__all__ = [
    "unified_diff",
    "difflib_diff",
    "DiffOp",
    "DiffOpCode",
    "LineRange",
    "DnaDiffer",
    "Differ",
]


class Differ[T](typing.Protocol):
    def __call__(
        self, left: cabc.Sequence[T], right: cabc.Sequence[T], /
    ) -> cabc.Generator[DiffOp]: ...


class DiffOpCode(enum.StrEnum):
    "All the possibilities of output of `difflib`."

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

    def slice(self, text: cabc.Sequence[str]):
        assert self.finish <= len(text)
        idx = slice(self.start, self.finish)
        return text[idx]


@dcls.dataclass(frozen=True)
class DiffOp:
    """
    `DiffOp` is an easier to work with representation of `get_opcodes()` output.

    It replaces the 5-tuples with 3 objects:

    `OpCode` enum to show the op.
    Source lines and target lines for easier slicing and validation.
    """

    code: DiffOpCode
    source: LineRange
    target: LineRange

    def __post_init__(self) -> None:
        match self.code:
            # len(lhs) = len(rhs).
            case DiffOpCode.EQUAL:
                if len(self.source) != len(self.target):
                    raise ValueError(f"{self.code=} but {self.source=}, {self.target=}")

                if len(self.source) == 0:
                    raise ValueError(
                        f"{self.code=}, but chunk of size 0 at {self.source=}, {self.target=}"
                    )

            # len(rhs) = 0.
            case DiffOpCode.DELETE:
                if len(self.target):
                    raise ValueError(f"{self.code=} but {self.target=}.")

            # len(lhs) = 0.
            case DiffOpCode.INSERT:
                if len(self.source):
                    raise ValueError(f"{self.code=} but {self.source=}.")

            # Nothing.
            case DiffOpCode.REPLACE:
                pass

    def render(
        self, left: cabc.Sequence[str], right: cabc.Sequence[str]
    ) -> cabc.Generator[str]:
        left = self.source.slice(left)
        right = self.target.slice(right)

        match self.code:
            case DiffOpCode.DELETE:
                yield self._hunk()
                yield from _render_delete(left)
            case DiffOpCode.INSERT:
                yield self._hunk()
                yield from _render_insert(right)
            case DiffOpCode.REPLACE:
                yield self._hunk()
                yield from _render_delete(left)
                yield from _render_insert(right)
            case DiffOpCode.EQUAL:
                return
                yield

    def _hunk(self):
        return _gen_hunk(
            self.source.start,
            self.source.finish,
            self.target.start,
            self.target.finish,
        )


def difflib_diff(
    a: cabc.Sequence[str], b: cabc.Sequence[str]
) -> cabc.Generator[DiffOp]:
    """
    Yields a generator of `DiffOp`.
    """

    matcher = difflib.SequenceMatcher(None, a, b)

    for code, i1, i2, j1, j2 in matcher.get_opcodes():
        yield DiffOp(
            code=DiffOpCode(code),
            source=LineRange(i1, i2),
            target=LineRange(j1, j2),
        )


def unified_diff(
    a: cabc.Sequence[str], b: cabc.Sequence[str], fromfile: str, tofile: str
):
    """
    The function mimicking the `unified_diff` function from `difflib`.
    """

    diff = difflib_diff(a, b)

    yield f"--- {fromfile}"
    yield f"+++ {tofile}"

    for op in diff:
        rendered = list(op.render(a, b))

        if rendered:
            yield from rendered


def _render_delete(lines: cabc.Sequence[str]):
    for l in lines:
        yield "-" + l


def _render_insert(lines: cabc.Sequence[str]):
    for l in lines:
        yield "+" + l


def _gen_hunk(i1: int, i2: int, j1: int, j2: int) -> str:
    return f"@@ -{i1+1},{i2-i1} +{j1+1},{j2-j1} @@"


@dcls.dataclass(frozen=True)
class DnaDiffResult[T]:
    left: cabc.Sequence[T | None]
    right: cabc.Sequence[T | None]
    score_matrix: npt.NDArray

    def __post_init__(self):
        assert len(self.left) == len(self.right)
        assert self.score_matrix.ndim == 2

    @property
    def score(self):
        return float(self.score_matrix[-1, -1])

    def ops(self) -> cabc.Generator[DiffOp]:
        """
        Convert the diffs into `DiffOp`s.

        This is a very simplified implementation and does not do aggregation.

        Yields:
            `DiffOp` where each line corresponds to 1 operation.
        """

        li = ri = 0
        for l, r in zip(self.left, self.right):
            assert not (l is r is None), "Impossible!"

            # Insert.
            if l is None:
                yield DiffOp(
                    code=DiffOpCode.INSERT,
                    source=LineRange(li, li),
                    target=LineRange(ri, ri := (ri + 1)),
                )

            # Delete.
            elif r is None:
                yield DiffOp(
                    code=DiffOpCode.DELETE,
                    source=LineRange(li, li := (li + 1)),
                    target=LineRange(ri, ri),
                )

            # Equal.
            elif l == r:
                yield DiffOp(
                    code=DiffOpCode.EQUAL,
                    source=LineRange(li, li := (li + 1)),
                    target=LineRange(ri, ri := (ri + 1)),
                )

            # Replace.
            else:
                yield DiffOp(
                    code=DiffOpCode.REPLACE,
                    source=LineRange(li, li := (li + 1)),
                    target=LineRange(ri, ri := (ri + 1)),
                )


@dcls.dataclass(frozen=True)
class DnaDiffer:
    equal: float = 1
    diff: float = -1
    gap: float = -1

    def __call__[T](self, left: cabc.Sequence[T], right: cabc.Sequence[T]):
        return self.align(left, right).ops()

    def align[T](self, left: cabc.Sequence[T], right: cabc.Sequence[T]):
        score = self.lcs(left, right)
        result_left, result_right = self.traceback(left, right, score)
        return DnaDiffResult(result_left, result_right, score)

    def lcs[T](self, left: cabc.Sequence[T], right: cabc.Sequence[T]) -> npt.NDArray:
        "Do a variation of LCS used in DNA."

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

    def traceback[T](
        self, seq1: cabc.Sequence[T], seq2: cabc.Sequence[T], score: npt.NDArray
    ):
        "Outputs the tracebacks. `None` denote gaps."

        i, j = len(seq1), len(seq2)
        align_left: list[T | None] = []
        align_right: list[T | None] = []

        while i > 0 and j > 0:
            matching = self.equal if seq1[i - 1] == seq2[j - 1] else self.diff

            if score[i, j] == score[i - 1, j - 1] + matching:
                align_left.append(seq1[i - 1])
                align_right.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif score[i, j] == score[i - 1, j] + self.gap:
                align_left.append(seq1[i - 1])
                align_right.append(None)
                i -= 1
            else:
                align_right.append(seq2[j - 1])
                align_left.append(None)
                j -= 1

        return list(reversed(align_left)), list(reversed(align_right))
