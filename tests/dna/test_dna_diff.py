# Copyright (c) ProBloLM Authors - All Rights Reserved

from pathlib import Path

import pytest

import problolm
from problolm import DiffOpCode, DnaDiffer, TreeSitterFileParser


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


@pytest.fixture(scope="module")
def differ():
    return DnaDiffer()


def test_equal(case_1_grammar: list[str], differ: DnaDiffer):
    result = differ.align(case_1_grammar, case_1_grammar)
    diffs = list(problolm.difflib_diff(result.left, result.right))
    assert all(diff.code == DiffOpCode.EQUAL for diff in diffs)


def test_not_equal(
    case_1_grammar: list[str], case_2_grammar: list[str], differ: DnaDiffer
):
    result = differ.align(case_1_grammar, case_2_grammar)
    diffs = list(problolm.difflib_diff(result.left, result.right))

    assert len(diffs) > 0
    assert not all(diff.code == DiffOpCode.EQUAL for diff in diffs)
