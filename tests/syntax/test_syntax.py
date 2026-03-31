# Copyright (c) ProBloLM Authors - All Rights Reserved

import pathlib
from collections import abc as cabc

import pytest
import tree_sitter as ts

import problolm


def _cases():
    cases_dir = pathlib.Path(__file__).parent / "cases"
    assert cases_dir.exists()
    assert cases_dir.is_dir()

    return cases_dir.glob("*.*")


@pytest.fixture(scope="session", params=_cases())
def case(request) -> pathlib.Path:
    result = request.param
    assert isinstance(result, pathlib.Path)
    assert result.exists()
    assert result.is_file()
    return result


def test_parse_case(case) -> None:
    parser = problolm.TreeSitterFileParser(case)
    result = parser.parse()
    assert all(isinstance(i.grammar, str) for i in result)


def _flatten_children(tree: ts.Node) -> cabc.Generator[ts.Node]:
    yield tree

    for child in tree.children:
        yield from _flatten_children(child)


def test_tree_generator(case):
    parser = problolm.TreeSitterFileParser(case)
    result = parser.parse()
    assert len(result) == len(list(_flatten_children(result.tree.root_node)))
