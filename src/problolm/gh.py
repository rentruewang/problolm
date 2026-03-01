# Copyright (c) ProBloLM Authors - All Rights Reserved

import functools
import os
from enum import StrEnum
from enum import auto as Auto

import github3


class EventType(StrEnum):
    "The event type that we are handling."

    COMMIT = Auto()
    ISSUE = Auto()
    PULL_REQUEST = Auto()


@functools.cache
def login():
    token = github_token()

    if (session := github3.login(token=token)) is None:
        raise ValueError("Authentication failed with your `github_token`.")

    return session


def github_token() -> str:
    try:
        token = os.environ["GITHUB_TOKEN"]
    except KeyError as ke:
        raise ValueError("You must provide an with `github_token` field.") from ke
    return token
