# Copyright (c) ProBloLM Authors - All Rights Reserved

"Custom github objects for easier use of github API."

import dataclasses as dcls
import functools
import re
import typing
from typing import Self
from github3.exceptions import NotFoundError

from github3.pulls import PullRequest
from github3.repos import Repository

from . import envs

__all__ = ["GitHubRepo", "GitHubPR", "GitHubCommit"]


@dcls.dataclass(frozen=True)
class GitHubRepo:
    "Information about the repository name."

    owner: str
    "Owner of the repository."

    slug: str
    "The name of the repo."

    def __str__(self) -> str:
        return self.url

    def pr(self, number: int, /) -> "GitHubPR":
        return GitHubPR(repo=self, number=number)

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.slug}"

    @functools.cached_property
    def github3(self) -> Repository:
        "GitHub3 integration."

        if repo := envs.login().repository(owner=self.owner, repository=self.slug):
            return repo

        raise ValueError(
            "Reponsitory is not found. Most likely an authentication issue."
        )

    @classmethod
    def from_name(cls, repo: str, /) -> Self:
        owner, slug = repo.split("/")
        return cls(owner=owner, slug=slug)

    @classmethod
    def from_env(cls) -> Self:
        "Load the github repository from environment."

        return cls.from_name(envs.github_repo())


@dcls.dataclass(frozen=True)
class GitHubPR:
    "The github PR."

    repo: GitHubRepo
    "The repo information."

    number: int
    "The PR number."

    def __post_init__(self) -> None:
        try:
            _ = self.github3
        except NotFoundError as ne:
            raise ValueError(
                f"PR number {self.number} is not valid. "
                f"Check on url: {self.url}. Is this a PR?"
            ) from ne

    def __str__(self) -> str:
        return self.url

    @property
    def url(self):
        return f"{self.repo.url}/pull/{self.number}"

    @functools.cached_property
    @typing.no_type_check
    def github3(self) -> PullRequest:
        "GitHub3 integration."

        return self.repo.github3.pull_request(self.number)

    @classmethod
    def from_env(cls) -> Self:
        "Get the PR from environment variable."

        # If ``github_event()`` is not PR, or raises and error,
        # re-raise a ``ValueError``.
        try:
            if envs.github_event() != "pull_request":
                raise NotImplementedError
        except NotImplementedError:
            raise ValueError("Only works with PR.")

        ref = envs.github_ref()

        if not (merge := re.fullmatch(r"refs/pull/(\d+)/merge", ref)):
            raise ValueError(f"Cannot parse PR {merge}.")

        pr_num = int(merge.group(1))
        return cls(repo=GitHubRepo.from_env(), number=pr_num)


@dcls.dataclass(frozen=True)
class GitHubCommit:
    pass
