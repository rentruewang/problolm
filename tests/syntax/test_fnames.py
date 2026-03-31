# Copyright (c) ProBloLM Authors - All Rights Reserved

import typing

import pytest

import problolm


class FileLang(typing.NamedTuple):
    file: str
    lang: str


def file_names_and_types():
    # Normal.
    yield FileLang("main.py", "Python")
    yield FileLang("app.js", "JavaScript")
    yield FileLang("index.html", "HTML")
    yield FileLang("style.css", "CSS")
    yield FileLang("program.cpp", "C++")
    yield FileLang("script.sh", "Bash")
    yield FileLang("server.go", "Go")
    yield FileLang("docs.md", "Markdown")

    # Upper case.
    yield FileLang("MAIN.PY", "Python")
    yield FileLang("App.Js", "JavaScript")
    yield FileLang("Index.HTML", "HTML")

    # No extension.
    yield FileLang("Makefile", "Makefile")
    yield FileLang(".bashrc", "Bash")

    # Multiple dots
    yield FileLang("app.test.js", "JavaScript")
    yield FileLang("component.spec.ts", "TypeScript")

    # Unicodes.
    yield FileLang("你好.py", "Python")
    yield FileLang("файл.js", "JavaScript")


@pytest.mark.parametrize("fname, ftype", file_names_and_types())
def test_guess_file_type(fname: str, ftype: str):
    assert problolm.guess_from_filename(fname) == ftype


@pytest.mark.parametrize("file_lang", file_names_and_types())
def test_grammar_for_file(file_lang: FileLang):
    assert callable(problolm.grammar_for_file(file_lang.file))


@pytest.mark.parametrize("file_lang", file_names_and_types())
def test_grammar_for_lang(file_lang: FileLang):
    assert callable(problolm.grammar_for_lang(file_lang.lang))
