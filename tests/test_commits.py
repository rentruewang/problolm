# Copyright (c) ProBloLM Authors - All Rights Reserved

import pathlib

import pytest

from problolm import Commit, CommitRange, set_git_repo
from problolm.git.commits import set_short_sha_size


@pytest.fixture(scope="module", autouse=True)
def set_repo_to_problolm(repo_root: pathlib.Path):
    with set_git_repo(path=str(repo_root)):
        yield


def _commits():
    yield "3cd26e765f27ab2a40ea322c1beb8feac3389d70"
    yield "HEAD"


@pytest.fixture(scope="module", params=_commits())
def commit(request: pytest.FixtureRequest) -> Commit:
    return Commit(request.param)


@pytest.fixture(scope="module")
def parent(commit) -> Commit:
    return commit.parent


@pytest.fixture(scope="module")
def commit_changes(commit: Commit, parent: Commit) -> CommitRange:
    return CommitRange(commit, parent)


def test_commits_eq(commit: Commit):
    with set_short_sha_size(9):
        short_commit = commit.short_sha
    assert commit == short_commit


def test_commit_ne(commit: Commit, parent: Commit):
    assert commit != parent
    assert commit.parent == parent


def test_commit_range_eq(
    commit: Commit,
    parent: Commit,
    commit_changes: CommitRange,
):
    assert commit_changes == commit - parent
    assert commit_changes == commit


def test_show(commit: Commit):
    commit.show()
