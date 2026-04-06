"""Probe file to retrigger AI detection after fixing the workflow filter."""

from __future__ import annotations


def pairwise_total(values: list[int]) -> int:
    total = 0
    for left, right in zip(values, values[1:]):
        total += left + right
    return total


RESULT = pairwise_total([1, 3, 5, 8])
