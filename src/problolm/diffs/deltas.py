# Copyright (c) ProBloLM Authors - All Rights Reserved

"The diff information."

from collections.abc import Sequence
from difflib import SequenceMatcher

__all__ = ["unified_diff_from_seq"]


def unified_diff_from_seq(
    a: Sequence[str], b: Sequence[str], fromfile: str, tofile: str
):
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
