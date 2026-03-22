# Copyright (c) ProBloLM Authors - All Rights Reserved

import pytest

import problolm


def file_names_and_types():
    # Normal.
    yield "main.py", "Python"
    yield "app.js", "JavaScript"
    yield "index.html", "HTML"
    yield "style.css", "CSS"
    yield "program.cpp", "C++"
    yield "script.sh", "Bash"
    yield "server.go", "Go"
    yield "docs.md", "Markdown"

    # Upper case.
    yield "MAIN.PY", "Python"
    yield "App.Js", "JavaScript"
    yield "Index.HTML", "HTML"

    # No extension.
    yield "Makefile", "Makefile"
    yield ".bashrc", "Bash"

    # Multiple dots
    yield "app.test.js", "JavaScript"
    yield "component.spec.ts", "TypeScript"

    # Unicodes.
    yield "你好.py", "Python"
    yield "файл.js", "JavaScript"


@pytest.mark.parametrize("fname, ftype", file_names_and_types())
def test_guess_file_type(fname: str, ftype: str):
    assert problolm.guess_from_filename(fname) == ftype
