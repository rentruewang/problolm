# Copyright (c) ProBloLM Authors - All Rights Reserved

"The SHA in git."

import contextlib as ctxl

from . import repos

__all__ = ["Sha", "ShaLike", "sha", "set_short_sha_size"]

_SHORT_SHA_LEN: int = 7
"The number of short SHA characters. Default to 7 (same as git)."

_SHA_LEN = 40
"The length of SHA."

type ShaLike = str | Sha
"Types that can be converted to a ``Sha``."


class Sha:
    __match_args__ = ("sha",)

    def __init__(self, sha: str):
        self._long_sha = repos.global_repo().commit(sha).hexsha
        "The sha of the commit."

        assert len(self._long_sha) == _SHA_LEN

    def __str__(self) -> str:
        return self.short_sha

    def __repr__(self):
        return f"Sha({self.short_sha})"

    def __eq__(self, sha: object) -> bool:
        match sha:
            # Check if it's prefix.
            case str():
                return self._long_sha.startswith(sha)

            case Sha():
                return self._long_sha == sha._long_sha

        return NotImplemented

    @property
    def short_sha(self) -> str:
        return self._long_sha[:_SHORT_SHA_LEN]

    @staticmethod
    def set_short_size(size: int):
        return set_short_sha_size(size)


def sha(sha: ShaLike) -> Sha:
    """
    Convert the ``sha`` argument to ``Sha``.
    """

    match sha:
        case str():
            return Sha(sha)
        case Sha():
            return sha
    raise TypeError(type(sha))


@ctxl.contextmanager
def set_short_sha_size(size: int):
    """
    Set the size of short sha.

    Args:
        size: The length of the short sha.
    """

    global _SHORT_SHA_LEN

    if size <= 0 or size > _SHA_LEN:
        raise ValueError(f"The inequality 0 < {size=} < {_SHA_LEN} should be upheld.")

    old_val = _SHORT_SHA_LEN

    try:
        _SHORT_SHA_LEN = size
        yield
    finally:
        _SHORT_SHA_LEN = old_val
