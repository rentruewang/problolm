# Copyright (c) ProBloLM Authors - All Rights Reserved

from enum import StrEnum, auto as Auto
import github3, functools, os


class EventType(StrEnum):
    "The event type that we are handling."

    COMMIT = Auto()
    ISSUE = Auto()
    PULL_REQUEST = Auto()


@functools.cache
def login():
    token = os.environ["GITHUB_TOKEN"]
    session = github3.login(token=token)
