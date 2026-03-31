# Copyright (c) ProBloLM Authors - All Rights Reserved

"Parsing the code into syntaxes."

import dataclasses as dcls
import functools
import logging
import pathlib
from collections import abc as cabc

import tree_sitter as ts

from . import langs

__all__ = ["TreeSitterFileParser", "ParsedSyntax"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class ParsedSyntax:
    "The syntax in the output."

    item: str
    grammar: str

    def __repr__(self):
        return f"{{{self.grammar}}} {self.item!r}"


@dcls.dataclass(frozen=True)
class TreeSitterResult:
    tree: ts.Tree
    syntax: cabc.Sequence[ParsedSyntax]

    def __len__(self) -> int:
        return len(self.syntax)

    def __getitem__(self, idx: int) -> ParsedSyntax:
        return self.syntax[idx]

    def __iter__(self) -> cabc.Generator[ParsedSyntax]:
        yield from self.syntax


class TreeSitterFileParser:
    def __init__(self, file: str | pathlib.Path):
        self._file = pathlib.Path(file)
        self._result: TreeSitterResult | None = None

    def __repr__(self) -> str:
        return f"TreeSitterParser({self.file})"

    def parse(self) -> TreeSitterResult:
        if self._result is not None:
            return self._result

        # Do the work.
        tree = parse_code_into_tree(code=self.code, filename=self.file)
        result = parse_syntax_tree(code=self.code, tree=tree)
        self._result = TreeSitterResult(tree=tree, syntax=result)

        return self._result

    @property
    def file(self) -> pathlib.Path:
        return self._file

    @property
    def tree(self) -> ts.Tree:
        return self.result.tree

    @property
    def result(self) -> TreeSitterResult:
        assert self._result is not None, f"{self.parse!r} has not been called yet!"
        return self._result

    @functools.cached_property
    def code(self) -> bytes:
        return self.file.read_bytes()


def parse_file_syntax(file: str | pathlib.Path) -> list[ParsedSyntax]:
    """
    Parse the syntax types of a source file in occuring order.
    """

    file = pathlib.Path(file)
    code = file.read_bytes()
    tree = parse_code_into_tree(code=code, filename=file)
    return parse_syntax_tree(code=code, tree=tree)


def parse_code_into_tree(code: bytes, filename: str | pathlib.Path) -> ts.Tree:
    """
    Parse the tree from the source file.
    """

    # Get grammar. This requires external tree-sitter-* libraries.
    try:
        grammar_gen = langs.grammar_for_file(filename)
    except ImportError as ie:
        raise ImportError(
            "Import not found. Try installing with `problolm[langs]` extras."
        ) from ie

    language = ts.Language(grammar_gen())
    parser = ts.Parser(language=language)
    return parser.parse(code)


def parse_syntax_tree(code: bytes, tree: ts.Tree) -> list[ParsedSyntax]:
    """
    Parse the tree sitter `ts.Tree` and convert each into their syntax nodes.
    The nodes are ordered by their occurence in the original source.
    """

    flattened = list(flatten(tree))

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


def _cmp_points(first: ts.Point, second: ts.Point) -> int:
    return (first.row - second.row) or (first.column - second.column)


def flatten(tree: ts.Tree) -> cabc.Generator[ts.Node]:
    "Flatten the given tree into its internal nodes."

    yield from _flatten(tree.walk())


def _flatten(cursor: ts.TreeCursor) -> cabc.Generator[ts.Node]:
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
