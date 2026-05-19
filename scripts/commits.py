# Copyright (c) ProBloLM Authors - All Rights Reserved

import itertools
import typing

import rich
from github3 import repos

from .consts import AI_KWDS, github

__all__ = ["check_commit_ai"]
print = rich.print


def check_commit_ai(owner: str, repo_name: str, commits: int = 1000):
    repo: repos.Repository | None = github().repository(owner, repo_name)
    if not repo:
        print("Repository not found.")
        raise ValueError
    assert isinstance(repo, repos.Repository)

    found_hits = []
    commit_count = 0

    # Scanning commits (the API handles trailers in the message and author fields)
    for commit in repo.commits(number=commits):
        commit_count += 1
        _scan_commit_metadata(found_hits, commit)

    if found_hits:
        print(
            f"{owner}/{repo_name}: Detected AI activity in {len(found_hits)} of the last {commit_count} commits:"
        )
    else:
        print(f"{owner}/{repo_name}: No AI keywords found in recent commit metadata.")
    return len(found_hits) / commit_count


def _scan_commit_metadata(found_hits: list[dict[str, typing.Any]], commit):
    # Check the message and author
    message = commit.message.lower()
    git_author = commit.commit.author["name"].lower()

    to_check: list[str] = [message, git_author]
    for kw, text in itertools.product(AI_KWDS, to_check):
        if kw not in text:
            continue

        found_hits.append(
            {
                "sha": commit.sha[:7],
                "author": commit.commit.author["name"],
                "keyword": kw,
            }
        )
        return
