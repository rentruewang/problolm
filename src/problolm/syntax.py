# Copyright (c) ProBloLM Authors - All Rights Reserved

"Parsing the code into syntaxes."

import dataclasses as dcls
import functools
import logging
from collections.abc import Generator
from os import PathLike
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Point, Tree, TreeCursor

__all__ = ["parse_syntax_types"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class ParsedSyntax:
    item: str
    grammar: str

    def __repr__(self):
        return f"[{self.grammar}] {self.item}"


def parse_syntax_types(file: PathLike) -> list[ParsedSyntax]:
    """
    Parse the syntax types of a source file in occuring order.
    """

    tree = parse_tree(file)
    return syntax_types_ordered(file, tree)


def parse_tree(file: PathLike) -> Tree:
    """
    Parse the tree from the source file.
    """

    file = Path(file)

    match file.suffix:
        case ".py":
            return python(file.read_bytes())
        case _:
            raise NotImplementedError(f"Don't know how to handle {file}.")


def syntax_types_ordered(file: PathLike, tree: Tree) -> list[ParsedSyntax]:
    """
    Parse the tree sitter `Tree` and convert each into their syntax nodes.
    The nodes are ordered by their occurence in the original source.
    """

    code = Path(file).read_bytes()
    flattened = list(_flatten(tree.walk()))

    # This is pre-order traversal, start points must be monotonically increasing.
    start_points = [node.start_point for node in flattened]
    assert all(_cmp_points(a, b) <= 0 for a, b in zip(start_points, start_points[1:]))

    return [
        ParsedSyntax(
            item=code[node.start_byte : node.end_byte].decode(),
            grammar=node.grammar_name,
        )
        for node in flattened
    ]


def _cmp_points(first: Point, second: Point) -> int:
    return (first.row - second.row) or (first.column - second.column)


def _flatten(cursor: TreeCursor) -> Generator[Node]:
    assert cursor.node
    yield cursor.node

    cursor.goto_first_child()

    while cursor.goto_next_sibling():
        yield from _flatten(cursor)

    cursor.goto_parent()


@functools.cache
def _parser():
    lang = Language(tspython.language())
    return Parser(lang)


def python(code: bytes) -> Tree:
    return _parser().parse(code)
