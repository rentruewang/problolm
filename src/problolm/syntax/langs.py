# Copyright (c) ProBloLM Authors - All Rights Reserved

"Different languages support for tree sitter."

import typing
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pygments import lexers
from tree_sitter import Language, Tree

__all__ = ["guess_from_filename", "grammar_for", "SupportedFileType"]

type LangaugeParser = Callable[[bytes], Tree]


def grammar_for(filename: str | Path, /) -> Callable[[], Language]:
    """
    Register a new tree-sitter parser into the global language registry,
    that can be looked up with the specified `aliases`.
    """

    lang = guess_from_filename(filename)
    result: Any = _get_grammar_from_lang(lang)
    return result


class SupportedFileType(StrEnum):
    PYTHON = "Python"
    CPP = "C++"
    GO = "Go"
    HTML = "HTML"
    CSS = "CSS"
    JS = "JavaScript"
    TS = "TypeScript"
    TSX = "TSX"
    FORTRAN = "Fortran"
    RUST = "Rust"
    RUBY = "Ruby"
    PHP = "PHP"


@typing.no_type_check
def guess_from_filename(filename: str | Path, /) -> SupportedFileType:
    "Guess the given filename as the language."

    guess = _guess_from_filename(filename)

    if guess is None:
        raise ValueError(
            f"Even Pygments don't know what your file type for {filename=}!"
        )

    # Go to the above defined types.
    if guess in SupportedFileType:
        return SupportedFileType(guess)

    # Handle some known cases.
    match guess:
        # Since C++ is compatible with C, and often .h files are C++, we treat C as C++.
        case "C":
            return SupportedFileType.CPP

        # TSX is a superset of JSX, and we don't have a JSX parser.
        case "JSX":
            return SupportedFileType.TSX

        # I don't know why fortran has fixed vs normal.
        case "FortranFixed":
            return SupportedFileType.FORTRAN

    raise NotImplementedError(f"Don't know how to handle the file type: {guess}")


def _guess_from_filename(filename: str | Path, /) -> str | None:
    try:
        lexer = lexers.get_lexer_for_filename(str(filename).lower())
        return lexer.name
    except Exception:
        return None


def _get_grammar_from_lang(file_type: SupportedFileType, /) -> Callable[[], object]:

    match file_type:
        case SupportedFileType.PYTHON:
            import tree_sitter_python

            return tree_sitter_python.language

        case SupportedFileType.CPP:
            import tree_sitter_cpp

            return tree_sitter_cpp.language

        case SupportedFileType.GO:
            import tree_sitter_go

            return tree_sitter_go.language

        case SupportedFileType.JS:
            import tree_sitter_javascript

            return tree_sitter_javascript.language

        case SupportedFileType.TS:
            import tree_sitter_typescript

            return tree_sitter_typescript.language_typescript

        case SupportedFileType.TSX:
            import tree_sitter_typescript

            return tree_sitter_typescript.language_tsx

        case SupportedFileType.FORTRAN:
            import tree_sitter_fortran

            return tree_sitter_fortran.language

        case SupportedFileType.RUST:
            import tree_sitter_rust

            return tree_sitter_rust.language

        case SupportedFileType.RUBY:
            import tree_sitter_ruby

            return tree_sitter_ruby.language

        case SupportedFileType.HTML:
            import tree_sitter_html

            return tree_sitter_html.language

        case SupportedFileType.CSS:
            import tree_sitter_css

            return tree_sitter_css.language

        case SupportedFileType.PHP:
            import tree_sitter_php

            return tree_sitter_php.language_php

    raise NotImplementedError(f"Unreachable! Forgot to handle file type: {file_type}.")
