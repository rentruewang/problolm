# Copyright (c) ProBloLM Authors - All Rights Reserved

from collections.abc import Generator
from pathlib import Path

import pytest
from tree_sitter import Node

from problolm import TreeSitterFileParser


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
    parser = TreeSitterFileParser(case)
    result = parser.parse()
    assert all(isinstance(i.grammar, str) for i in result)


def _flatten_children(tree: Node) -> Generator[Node]:
    yield tree

    for child in tree.children:
        yield from _flatten_children(child)


def test_tree_generator(case):
    parser = TreeSitterFileParser(case)
    result = parser.parse()
    assert len(result) == len(list(_flatten_children(result.tree.root_node)))
