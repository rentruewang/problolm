# Copyright (c) ProBloLM Authors - All Rights Reserved

from collections.abc import Generator
from pathlib import Path

import pytest
from pytest import FixtureRequest

import problolm
from problolm import Differ, DiffOpCode, DnaDiffer, TreeSitterFileParser


@pytest.fixture(scope="module")
def cases_dir():
    folder = Path(__file__).parent / "cases"
    assert folder.exists() and folder.is_dir()
    return folder


@pytest.fixture(scope="module")
def case_1(cases_dir: Path) -> Path:
    return cases_dir / "case_1.cpp"


@pytest.fixture(scope="module")
def case_2(cases_dir: Path) -> Path:
    return cases_dir / "case_2.cpp"


@pytest.fixture(scope="module")
def case_1_grammar(case_1: Path) -> list[str]:
    return [res.grammar for res in TreeSitterFileParser(case_1).parse()]


@pytest.fixture(scope="module")
def case_2_grammar(case_2: Path) -> list[str]:
    return [res.grammar for res in TreeSitterFileParser(case_2).parse()]


def _differ() -> Generator[Differ[str]]:
    yield problolm.difflib_diff
    yield DnaDiffer()


@pytest.fixture(params=_differ())
def differ(request: FixtureRequest) -> Differ[str]:
    return request.param


def test_equal(case_1_grammar: list[str], differ: Differ[str]):
    diffs = list(differ(case_1_grammar, case_1_grammar))
    assert all(diff.code == DiffOpCode.EQUAL for diff in diffs)


def test_not_equal(
    case_1_grammar: list[str], case_2_grammar: list[str], differ: Differ[str]
):
    diffs = list(differ(case_1_grammar, case_2_grammar))

    assert len(diffs) > 0
    assert not all(diff.code == DiffOpCode.EQUAL for diff in diffs)
