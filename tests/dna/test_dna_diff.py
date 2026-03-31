# Copyright (c) ProBloLM Authors - All Rights Reserved

import pathlib
from collections.abc import Generator

import pytest

import problolm


@pytest.fixture(scope="module")
def cases_dir():
    folder = pathlib.Path(__file__).parent / "cases"
    assert folder.exists() and folder.is_dir()
    return folder


@pytest.fixture(scope="module")
def case_1(cases_dir: pathlib.Path) -> pathlib.Path:
    return cases_dir / "case_1.cpp"


@pytest.fixture(scope="module")
def case_2(cases_dir: pathlib.Path) -> pathlib.Path:
    return cases_dir / "case_2.cpp"


@pytest.fixture(scope="module")
def case_1_grammar(case_1: pathlib.Path) -> list[str]:
    return [res.grammar for res in problolm.TreeSitterFileParser(case_1).parse()]


@pytest.fixture(scope="module")
def case_2_grammar(case_2: pathlib.Path) -> list[str]:
    return [res.grammar for res in problolm.TreeSitterFileParser(case_2).parse()]


def _differ() -> Generator[problolm.Differ[str]]:
    yield problolm.difflib_diff
    yield problolm.DnaDiffer()


@pytest.fixture(params=_differ())
def differ(request: pytest.FixtureRequest) -> problolm.Differ[str]:
    return request.param


def test_equal(case_1_grammar: list[str], differ: problolm.Differ[str]):
    diffs = list(differ(case_1_grammar, case_1_grammar))
    assert all(diff.code == problolm.DiffOpCode.EQUAL for diff in diffs)


def test_not_equal(
    case_1_grammar: list[str], case_2_grammar: list[str], differ: problolm.Differ[str]
):
    diffs = list(differ(case_1_grammar, case_2_grammar))

    assert len(diffs) > 0
    assert not all(diff.code == problolm.DiffOpCode.EQUAL for diff in diffs)
