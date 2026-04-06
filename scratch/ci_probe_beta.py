"""Second standalone CI probe file for workflow pickup testing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeWindow:
    start: int
    stop: int

    def width(self) -> int:
        return self.stop - self.start


def summarize_windows(windows: list[ProbeWindow]) -> str:
    total_width = sum(window.width() for window in windows)
    return f"beta:{len(windows)}:{total_width}"


SUMMARY = summarize_windows(
    [
        ProbeWindow(start=1, stop=4),
        ProbeWindow(start=6, stop=11),
    ]
)
