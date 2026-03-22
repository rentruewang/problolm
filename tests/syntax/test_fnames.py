# Copyright (c) ProBloLM Authors - All Rights Reserved

import pytest

import problolm


def file_names_and_types():
    # Basic extensions
    yield "main.py", "Python"
    yield "app.js", "JavaScript"
    yield "index.html", "HTML"
    yield "style.css", "CSS"
    yield "program.cpp", "C++"
    yield "script.sh", "Bash"
    yield "server.go", "Go"
    yield "docs.md", "Markdown"

    # Case sensitivity
    yield "MAIN.PY", "Python"
    yield "App.Js", "JavaScript"
    yield "Index.HTML", "HTML"

    # Unknown extensions
    yield "file.xyz", "Unknown"
    yield "data.abc", "Unknown"
    yield "weird.customext", "Unknown"

    # No extension (likely content-based)
    yield "Makefile", "Makefile"
    yield "Dockerfile", "Docker"
    yield "README", "Markdown"
    yield "LICENSE", "Unknown"

    # Hidden files
    yield ".bashrc", "Bash"
    yield ".gitignore", "Git"
    yield ".env", "Config"

    # Multiple dots
    yield "archive.tar.gz", "Unknown"
    yield "app.test.js", "JavaScript"
    yield "component.spec.ts", "TypeScript"

    # Unicode filenames
    yield "你好.py", "Python"
    yield "файл.js", "JavaScript"


@pytest.mark.parametrize("fname, ftype", file_names_and_types())
def test_guess_file_type(fname: str, ftype: str):
    assert problolm.guess_from_filename(fname) == ftype
