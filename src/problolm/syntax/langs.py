# Copyright (c) ProBloLM Authors - All Rights Reserved

"Different languages support for tree sitter."

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pygments import lexers
from tree_sitter import Language, Tree

__all__ = ["guess_from_filename", "grammar_for"]

type LangaugeParser = Callable[[bytes], Tree]


def grammar_for(filename: str | Path, /) -> Callable[[], Language]:
    """
    Register a new tree-sitter parser into the global language registry,
    that can be looked up with the specified `aliases`.
    """

    result: Any = _grammar_for(filename)
    return result


def guess_from_filename(filename: str | Path, /) -> str:
    "Guess the given filename as the language."
    try:
        lexer = lexers.get_lexer_for_filename(filename)
        return lexer.name
    except Exception:
        return "Unknown"


def _grammar_for(filename: str | Path, /) -> Callable[[], object]:
    lang = guess_from_filename(filename)

    match lang := guess_from_filename(filename):
        case "Python":
            import tree_sitter_python

            return tree_sitter_python.language

        # Since C++ is compatible with C, and often .h files are C++, we treat C as C++.
        case "C++" | "C":
            import tree_sitter_cpp

            return tree_sitter_cpp.language

        case "Go":
            import tree_sitter_go

            return tree_sitter_go.language

        case "JavaScript":
            import tree_sitter_javascript

            return tree_sitter_javascript.language

        case "TypeScript":
            import tree_sitter_typescript

            return tree_sitter_typescript.language_typescript

        # TSX is a superset of JSX.
        case "JSX" | "TSX":
            import tree_sitter_typescript

            return tree_sitter_typescript.language_tsx

        case "Fortran" | "FortranFixed":
            import tree_sitter_fortran

            return tree_sitter_fortran.language

        case "Rust":
            import tree_sitter_rust

            return tree_sitter_rust.language

        case "Unknown":
            raise ValueError(f"Even pygments don't know the file type for {filename=}.")

        case _:
            raise NotImplementedError(f"Language {lang} is not handled!")
