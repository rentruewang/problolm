# Copyright (c) ProBloLM Authors - All Rights Reserved

"Different languages support for tree sitter."

import enum
import pathlib
import typing
from collections import abc as cabc

import tree_sitter as ts
from pygments import lexers

__all__ = ["guess_from_filename", "grammar_for_file", "LangName", "grammar_for_lang"]

type LangaugeParser = cabc.Callable[[bytes], ts.Tree]


def grammar_for_file(filename: str | pathlib.Path, /) -> cabc.Callable[[], object]:
    """
    Register a new tree-sitter parser into the global language registry,
    that can be looked up with the specified `aliases`.
    """

    lang = guess_from_filename(filename)
    return grammar_for_lang(lang)


class LangName(enum.StrEnum):
    "Enum for the names of the languages."

    PYTHON = "Python"
    CPP = "C++"
    GO = "Go"
    HTML = "HTML"
    CSS = "CSS"
    JAVA = "Java"
    CS = "C#"
    JS = "JavaScript"
    TS = "TypeScript"
    TSX = "TSX"
    BASH = "Bash"
    FORTRAN = "Fortran"
    RUST = "Rust"
    RUBY = "Ruby"
    PHP = "PHP"
    ELIXIR = "Elixir"
    MAKE = "Makefile"
    JSON = "JSON"
    TOML = "TOML"
    YAML = "YAML"
    MD = "Markdown"


def guess_from_filename(filename: str | pathlib.Path, /) -> LangName:
    "Guess the given filename as the language."

    guess = _guess_from_filename(filename)

    if guess is None:
        raise ValueError(
            f"Even Pygments don't know what your file type for {filename=}!"
        )

    # Go to the above defined types.
    if guess in LangName:
        return LangName(guess)

    # Handle some known types we want to handle as another type..
    match guess:
        # Since C++ is compatible with C, and often .h files are C++, we treat C as C++.
        case "C":
            return LangName.CPP

        # TSX is a superset of JSX, and we don't have a JSX parser.
        case "JSX":
            return LangName.TSX

        # I don't know why fortran has fixed vs normal.
        case "FortranFixed":
            return LangName.FORTRAN

    raise NotImplementedError(f"Filetype is not known for '{filename}'.")


@typing.no_type_check
def _guess_from_filename(filename: str | pathlib.Path, /) -> str | None:
    try:
        lexer = lexers.get_lexer_for_filename(str(filename).lower())
        return lexer.name
    except Exception:
        return None


def grammar_for_lang(file_type: str | LangName, /) -> cabc.Callable[[], object]:
    try:
        file_type = _resolve_file_type(file_type)
    except ValueError as ve:
        raise ValueError(
            f"If using a `str` file type, must be keys or values of enum {LangName}"
        ) from ve

    match file_type:
        case LangName.PYTHON:
            import tree_sitter_python

            return tree_sitter_python.language

        case LangName.CPP:
            import tree_sitter_cpp

            return tree_sitter_cpp.language

        case LangName.GO:
            import tree_sitter_go

            return tree_sitter_go.language

        case LangName.JS:
            import tree_sitter_javascript

            return tree_sitter_javascript.language

        case LangName.TS:
            import tree_sitter_typescript

            return tree_sitter_typescript.language_typescript

        case LangName.TSX:
            import tree_sitter_typescript

            return tree_sitter_typescript.language_tsx

        case LangName.JAVA:
            import tree_sitter_java

            return tree_sitter_java.language

        case LangName.CS:
            import tree_sitter_c_sharp

            return tree_sitter_c_sharp.language

        case LangName.FORTRAN:
            import tree_sitter_fortran

            return tree_sitter_fortran.language

        case LangName.RUST:
            import tree_sitter_rust

            return tree_sitter_rust.language

        case LangName.RUBY:
            import tree_sitter_ruby

            return tree_sitter_ruby.language

        case LangName.HTML:
            import tree_sitter_html

            return tree_sitter_html.language

        case LangName.CSS:
            import tree_sitter_css

            return tree_sitter_css.language

        case LangName.PHP:
            import tree_sitter_php

            return tree_sitter_php.language_php

        case LangName.ELIXIR:
            import tree_sitter_elixir

            return tree_sitter_elixir.language

        case LangName.BASH:
            import tree_sitter_bash

            return tree_sitter_bash.language

        case LangName.MAKE:
            import tree_sitter_make

            return tree_sitter_make.language

        case LangName.JSON:
            import tree_sitter_json

            return tree_sitter_json.language

        case LangName.TOML:
            import tree_sitter_toml

            return tree_sitter_toml.language

        case LangName.YAML:
            import tree_sitter_yaml

            return tree_sitter_yaml.language

        case LangName.MD:
            import tree_sitter_markdown

            return tree_sitter_markdown.language

    # Hint: Hover on the `file_type` and see if it's `Never` type.
    raise NotImplementedError(f"Unreachable! Forgot to handle file type: {file_type}.")


def _resolve_file_type(file_type: str | LangName) -> LangName:
    if isinstance(file_type, LangName):
        return file_type

    try:
        file_type = LangName(file_type)
        return file_type
    except ValueError:
        pass

    try:
        file_type = LangName[file_type]
    except KeyError:
        pass

    raise ValueError(
        f"Must specify either valid keys or valid values in: {list(LangName)}. "
        "E.g. 'PYTHON' or 'Python'."
    )
