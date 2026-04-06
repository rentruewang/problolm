"""Standalone CI probe file for workflow pickup testing."""

from __future__ import annotations


def rolling_checksum(values: list[int]) -> int:
    total = 0
    for index, value in enumerate(values, start=1):
        total += index * value
    return total


def describe_sample() -> str:
    sample = [2, 5, 8, 13]
    return f"alpha:{rolling_checksum(sample)}"


SUMMARY = describe_sample()
