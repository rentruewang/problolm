# Copyright (c) ProBloLM Authors - All Rights Reserved

"Interact with the environment in github actions."
import os
import re
import typing
from enum import StrEnum
from enum import auto as Auto

import github3
from github3 import GitHub
from github3.pulls import PullRequest
from github3.repos import Repository

__all__ = ["EventType", "github_repo", "github_token"]


class EventType(StrEnum):
    "The event type that we are handling."

    ISSUE = Auto()
    PULL_REQUEST = Auto()
    OTHER = Auto()


def github_repo() -> str:
    "Get the current repo in the `owner/repo` format."

    return os.environ["GITHUB_REPOSITORY"]


def github_token() -> str:
    "Get the github token for authentication."

    return os.environ["GITHUB_TOKEN"]


def github_sha() -> str:
    "Get the SHA of the latest commit."

    return os.environ["GITHUB_SHA"]


def github_ref():
    "Get info about the current ref."

    return os.environ["GITHUB_REF"]


def event_name() -> EventType:
    "Get the event name as an enum."

    event_name = os.environ["GITHUB_EVENT_NAME"]

    match event_name:
        case "pull_request":
            return EventType.PULL_REQUEST
        case "issue":
            return EventType.ISSUE
        case _:
            return EventType.OTHER


@typing.no_type_check
def login() -> GitHub:
    token = github_token()

    if session := github3.login(token=token):
        return session

    raise ValueError("Authentication failed with your `github_token`.")


@typing.no_type_check
def repo() -> Repository:
    owner, repo = github_repo().split("/")

    if repo := login().repository(owner=owner, repository=repo):
        return repo

    raise ValueError("Reponsitory is not found. Most likely an authentication issue.")


@typing.no_type_check
def pr(number: int, /) -> PullRequest:
    "Get a PR by number."

    return repo().pull_request(number)


def current_pr() -> PullRequest:
    "Get the current PR. Raise ``ValueError`` if this is not called in a PR."
    if not event_name() == EventType.PULL_REQUEST:
        raise ValueError("This is not a pull request!")

    ref = github_ref()

    if not (merge := re.fullmatch(r"refs/pull/(\d+)/merge", ref)):
        raise ValueError(f"Cannot parse PR {merge}.")

    pr_num = int(merge.group(1))
    return pr(pr_num)


def close_current_pr() -> None:
    current_pr().close()
