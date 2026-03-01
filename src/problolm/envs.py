# Copyright (c) ProBloLM Authors - All Rights Reserved

"Interact with the environment in github actions."

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


def github_repo() -> str:
    "Get the current repo in the `owner/repo` format."

    return os.environ["GITHUB_REPOSITORY"]


def github_token() -> str:
    "Get the github token for authentication."

    return os.environ["GITHUB_TOKEN"]
