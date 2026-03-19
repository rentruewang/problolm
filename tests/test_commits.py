# Copyright (c) ProBloLM Authors - All Rights Reserved

import pytest

from problolm import Commit, RangeDiff


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
def commit_parrent_diff(commit, parent):
    return RangeDiff(commit, parent)


def test_commits_eq(commit, short_commit):
    assert commit == short_commit


def test_commit_ne(commit, parent):
    assert commit != parent
    assert commit.parent == parent


def test_commit_range_eq(commit, parent, commit_parrent_diff):
    assert commit_parrent_diff == commit - parent
    assert commit_parrent_diff == commit
