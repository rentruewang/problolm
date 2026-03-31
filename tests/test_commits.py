# Copyright (c) ProBloLM Authors - All Rights Reserved

from pathlib import Path

import pytest

import problolm
from problolm.git import commits


@pytest.fixture(scope="module", autouse=True)
def set_repo_to_problolm(repo_root: Path):
    with problolm.set_git_repo(path=str(repo_root)):
        yield


def _commits():
    yield "3cd26e765f27ab2a40ea322c1beb8feac3389d70"
    yield "HEAD"


@pytest.fixture(scope="module", params=_commits())
def commit(request: pytest.FixtureRequest) -> problolm.Commit:
    return problolm.Commit(request.param)


@pytest.fixture(scope="module")
def parent(commit) -> problolm.Commit:
    return commit.parent


@pytest.fixture(scope="module")
def commit_changes(
    commit: problolm.Commit, parent: problolm.Commit
) -> problolm.CommitRange:
    return problolm.CommitRange(commit, parent)


def test_commits_eq(commit: problolm.Commit):
    with commits.set_short_sha_size(9):
        short_commit = commit.short_sha
    assert commit == short_commit


def test_commit_ne(commit: problolm.Commit, parent: problolm.Commit):
    assert commit != parent
    assert commit.parent == parent


def test_commit_range_eq(
    commit: problolm.Commit,
    parent: problolm.Commit,
    commit_changes: problolm.CommitRange,
):
    assert commit_changes == commit - parent
    assert commit_changes == commit


def test_show(commit: problolm.Commit):
    commit.show()
