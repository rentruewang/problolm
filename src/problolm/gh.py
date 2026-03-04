# Copyright (c) ProBloLM Authors - All Rights Reserved

"Custom github objects for easier use of github API."

import dataclasses as dcls
import logging
import re
import typing
from collections.abc import Generator
from typing import Self

from github3.exceptions import NotFoundError
from github3.pulls import PullRequest
from github3.repos import Repository

from . import envs
from .commits import Commit

__all__ = ["GitHubRepo", "GitHubPr", "github_repo", "github_pull_request"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class GitHubRepo:
    "Information about the repository name."

    owner: str
    "Owner of the repository."

    slug: str
    "The name of the repo."

    github3: Repository
    "The github3 object."

    def __str__(self) -> str:
        return self.url

    @typing.no_type_check
    def pr(self, number: int, /) -> "GitHubPr":
        return GitHubPr(
            repo=self,
            number=number,
            github3=self.github3.pull_request(number),
        )

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.slug}"

    @classmethod
    def from_name(cls, repo: str, /) -> Self:
        owner, slug = repo.split("/")
        return cls.from_owner_slug(owner=owner, slug=slug)

    @classmethod
    def from_owner_slug(cls, owner: str, slug: str):
        if repo := envs.login().repository(owner=owner, repository=slug):
            return cls(owner=owner, slug=slug, github3=repo)

        raise ValueError(
            "Reponsitory is not found. Most likely an authentication issue."
        )


def github_repo(repo: str = "") -> GitHubRepo:
    """
    The public function for creating a repo object.

    Args:
        repo:
            If given, the repo is created with owner/slug format.
            If not given (or ""), environment's `GITHUB_REPOSITORY` is used.

    Returns:
        A repo object.
    """

    repo = repo or envs.github_repo()
    return GitHubRepo.from_name(repo)


@dcls.dataclass(frozen=True)
class GitHubPr:
    "The github PR."

    repo: GitHubRepo
    "The repo information."

    number: int
    "The PR number."

    github3: PullRequest
    "The pr object of github."

    def __post_init__(self) -> None:
        LOGGER.debug("Creating a PR object.")

        try:
            _ = self.github3
        except NotFoundError as ne:
            raise ValueError(
                f"PR number {self.number} is not valid. "
                f"Check on url: {self.url}. Is this a PR?"
            ) from ne

    def __str__(self) -> str:
        return self.url

    def __iter__(self) -> Generator[Commit]:
        yield from self.commits()

    @property
    def url(self):
        return f"{self.repo.url}/pull/{self.number}"

    def commits(self) -> Generator[Commit]:
        for commit in self.github3.commits():
            yield Commit(commit.sha)

    def patch(self) -> str:
        return self.github3.patch().decode()


def github_pull_request(number: int = 0, repo: str = "") -> GitHubPr:
    """
    Create a github pull request object.

    Args:
        number:
            The PR number. Must be found on github.
            If not given, this is parsed from `GITHUB_REF` in environment.
        repo: Passed to ``github_repo``. See documentation there for details.

    Returns:
        A PR object.
    """

    number = number or _parse_env_pr_num()
    return github_repo(repo).pr(number)


def _parse_env_pr_num():
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

    return int(merge.group(1))
