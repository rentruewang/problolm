# Copyright (c) ProBloLM Authors - All Rights Reserved

import abc
import dataclasses as dcls
import json
import os
import typing
from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, Self


class Event(ABC):
    @abc.abstractmethod
    def log(self) -> None: ...

    @classmethod
    @abc.abstractmethod
    def parse(cls, event, /) -> Self: ...


@dcls.dataclass(frozen=True)
class _NumberedEvent(Event):
    "Base class for PR and issue, both has a #number."

    TYPE: ClassVar[str]

    title: str
    body: str
    number: int

    @typing.override
    def log(self) -> None:
        print(f"{self.TYPE} number:", self.number)
        print(f"{self.TYPE} title:", self.title)
        print(f"{self.TYPE} body:", self.body)

    @classmethod
    @typing.override
    def parse(cls, event, /) -> Self:
        return cls(
            title=event["title"],
            body=event["body"],
            number=event["number"],
        )


@dcls.dataclass(frozen=True)
class PullRequestEvent(_NumberedEvent):
    TYPE: ClassVar[str] = "PR"


@dcls.dataclass(frozen=True)
class IssueEvent(_NumberedEvent):
    TYPE: ClassVar[str] = "ISSUE"


@dcls.dataclass(frozen=True)
class Commit:
    sha: str
    message: str
    author: str
    email: str


@dcls.dataclass(frozen=True)
class CommitEvent(Event):
    commits: Sequence[Commit]

    @typing.override
    def log(self) -> None:
        for commit in self.commits:
            print("Commit sha:", commit.sha)
            print("Commit message:", commit.message)
            print("Commit author:", commit.author)
            print("Commit email:", commit.email)

    @classmethod
    def parse(cls, event, /) -> Self:
        return cls(
            [
                Commit(
                    sha=entry["head"]["sha"],
                    message=entry["message"],
                    author=entry["author"]["name"],
                    email=entry["author"]["email"],
                )
                for entry in event
            ]
        )


def load_event() -> Event:
    event_path = os.environ["GITHUB_EVENT_PATH"]
    if not event_path:
        raise RuntimeError("Missing GITHUB_EVENT_PATH")

    with open(event_path, "r") as f:
        data = json.load(f)

    if event := data.get("pull_request"):
        return PullRequestEvent.parse(event)

    if event := data.get("issue"):
        return IssueEvent.parse(event)

    if event := data.get("commits"):
        return CommitEvent.parse(event)

    raise ValueError(f"Do not know how to handle event: {event}")


if __name__ == "__main__":
    event = load_event()
    event.log()
