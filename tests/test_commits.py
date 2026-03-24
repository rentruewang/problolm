# Copyright (c) ProBloLM Authors - All Rights Reserved

from pathlib import Path

import pytest

import problolm
from problolm import Commit, CommitRange, RepoIsDirty


@pytest.fixture(scope="module", autouse=True)
def set_repo_to_problolm(repo_root: Path):
    try:
        with problolm.set_git_repo(path=str(repo_root)):
            yield
    except RepoIsDirty as e:
        pytest.xfail(str(e))


@pytest.fixture(scope="module")
def commit():
    return Commit("3cd26e765f27ab2a40ea322c1beb8feac3389d70")


@pytest.fixture(scope="module")
def short_commit():
    return Commit("3cd26e7")


@pytest.fixture(scope="module")
def parent(commit) -> Commit:
    return commit.parent


@pytest.fixture(scope="module")
def commit_parrent_diff(commit: Commit, parent: Commit) -> CommitRange:
    return CommitRange(commit, parent)


def test_commits_eq(commit: Commit, short_commit: Commit):
    assert commit == short_commit


def test_commit_ne(commit: Commit, parent: Commit):
    assert commit != parent
    assert commit.parent == parent


def test_commit_range_eq(
    commit: Commit,
    parent: Commit,
    commit_parrent_diff: CommitRange,
):
    assert commit_parrent_diff == commit - parent
    assert commit_parrent_diff == commit
