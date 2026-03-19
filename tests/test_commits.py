# Copyright (c) ProBloLM Authors - All Rights Reserved

import pytest

from problolm import Commit


@pytest.fixture(scope="module")
def commit():
    return Commit("3cd26e765f27ab2a40ea322c1beb8feac3389d70")


@pytest.fixture(scope="module")
def short_commit():
    return Commit("3cd26e7")


@pytest.fixture(scope="module")
def parent(commit):
    return commit.parent


def test_commits_eq(commit, short_commit):
    assert commit == short_commit


def test_commit_ne(commit, parent):
    assert commit != parent
    assert commit.parent == parent
