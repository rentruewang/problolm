# Copyright (c) ProBloLM Authors - All Rights Reserved

"Handle the environment and some github3 integration."

import enum
import functools
import os
import typing

import github3

__all__ = ["github_repo", "github_ref", "github_event"]


def github_repo() -> str:
    "Get the current repo in the `owner/repo` format."

    return os.environ["GITHUB_REPOSITORY"]


def github_token() -> str:
    "Get the github token for authentication."

    return os.environ["GITHUB_TOKEN"]


def github_ref():
    "Get info about the current ref."

    return os.environ["GITHUB_REF"]


class EventType(enum.StrEnum):
    "The event type that we are handling."

    ISSUE = enum.auto()
    PULL_REQUEST = enum.auto()
    OTHER = enum.auto()


def github_event():
    """
    Get the event name as a literal ('pull_request', 'issue').

    Raises:
        NotImplementedError: If it's not a PR or an issue.
    """

    event_name = os.environ["GITHUB_EVENT_NAME"]

    match event_name:
        case "pull_request" | "issue":
            return event_name
        case _:
            raise NotImplementedError(f"Unknown event type: '{event_name}'.")


@functools.cache
@typing.no_type_check
def login() -> github3.GitHub:
    token = github_token()

    if session := github3.login(token=token):
        return session

    raise ValueError("Authentication failed with your `github_token`.")
