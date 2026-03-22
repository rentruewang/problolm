# Copyright (c) ProBloLM Authors - All Rights Reserved

from pathlib import Path

import pytest

import problolm


def _cases():
    cases_dir = Path(__file__).parent / "cases"
    assert cases_dir.exists()
    assert cases_dir.is_dir()

    return cases_dir.glob("*.*")


@pytest.fixture(scope="session", params=_cases())
def case(request) -> Path:
    result = request.param
    assert isinstance(result, Path)
    assert result.exists()
    assert result.is_file()
    return result


def test_parse_case(case) -> None:
    result = problolm.parse_file_syntax(case)
    assert isinstance(result, list) and all(isinstance(i.grammar, str) for i in result)
