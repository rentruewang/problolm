# Copyright (c) ProBloLM Authors - All Rights Reserved

"Parsing the code into syntaxes."

import dataclasses as dcls
import logging
from collections.abc import Generator
from pathlib import Path

from tree_sitter import Language, Node, Parser, Point, Tree, TreeCursor

from . import langs

__all__ = ["parse_file_syntax", "parse_code_into_tree"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class ParsedSyntax:
    "The syntax in the output."

    item: str
    grammar: str

    def __repr__(self):
        return f"{{{self.grammar}}} {self.item!r}"


def parse_file_syntax(file: str | Path) -> list[ParsedSyntax]:
    """
    Parse the syntax types of a source file in occuring order.
    """

    file = Path(file)
    code = file.read_bytes()
    tree = parse_code_into_tree(code=code, filename=file)
    return parse_syntax_tree(code=code, tree=tree)


def parse_code_into_tree(code: bytes, filename: str | Path) -> Tree:
    """
    Parse the tree from the source file.
    """

    grammar_gen = langs.grammar_for(filename)
    grammar = grammar_gen()
    assert isinstance(grammar, Language), type(grammar)
    parser = Parser(language=grammar)
    return parser.parse(code)


def parse_syntax_tree(code: bytes, tree: Tree) -> list[ParsedSyntax]:
    """
    Parse the tree sitter `Tree` and convert each into their syntax nodes.
    The nodes are ordered by their occurence in the original source.
    """

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

    # End the traversal when there is no children.
    if not cursor.goto_first_child():
        return

    # Do-while loop rather than while because we already entered the first children.
    while True:
        yield from _flatten(cursor)

        if not cursor.goto_next_sibling():
            break

    # Exit. Go up.
    cursor.goto_parent()
