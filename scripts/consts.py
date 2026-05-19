# Copyright (c) ProBloLM Authors - All Rights Reserved

import os

import github3

__all__ = ["github", "AI_KWDS"]


def github(token: str | None = None):
    """
    Get the github interface.
    """

    token = token or os.environ["GITHUB_TOKEN"]
    gh = github3.login(token)

    if not gh:
        raise RuntimeError(f"Authentication failed for token: {token}")

    return gh


AI_KWDS = [
    "claude",
    "codex",
    "copilot",
    "aider",
    "cursor",
    "cody",
    "windsurf",
    "codeium",
    "cline",
    "gemini",
    "devin",
    "openhands",
    "swe-agent",
    "coderabbit",
]
