from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def load_changed_files_module():
    module_path = Path("scripts/changed_files.py")
    spec = importlib.util.spec_from_file_location("changed_files", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pull_request_uses_base_ref_triple_dot(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_changed_files_module()
    calls: list[list[str]] = []

    def fake_check_output(args: list[str], text: bool = True) -> str:
        calls.append(args)
        return "src/problolm/cli.py\nREADME.md\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    files = module.changed_python_files(
        event_name="pull_request",
        before="ignored",
        sha="ignored",
        base_ref="main",
        default_branch="main",
    )

    assert files == ["src/problolm/cli.py"]
    assert calls == [
        [
            "git",
            "diff",
            "--diff-filter=AM",
            "--name-only",
            "origin/main...HEAD",
        ]
    ]


def test_push_uses_before_and_sha_when_before_is_real(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_changed_files_module()
    calls: list[list[str]] = []

    def fake_check_output(args: list[str], text: bool = True) -> str:
        calls.append(args)
        return "src/problolm/cli.py\ntests/test_ok.py\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    files = module.changed_python_files(
        event_name="push",
        before="abc123",
        sha="def456",
        base_ref=None,
        default_branch="main",
    )

    assert files == ["src/problolm/cli.py", "tests/test_ok.py"]
    assert calls == [
        ["git", "diff", "--diff-filter=AM", "--name-only", "abc123", "def456"]
    ]


def test_push_with_zero_before_uses_merge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_changed_files_module()
    calls: list[list[str]] = []

    def fake_check_output(args: list[str], text: bool = True) -> str:
        calls.append(args)
        if args[:3] == ["git", "merge-base", "HEAD"]:
            return "base789\n"
        return "src/problolm/cli.py\nnotes.txt\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    files = module.changed_python_files(
        event_name="push",
        before=module.ZERO_SHA,
        sha="def456",
        base_ref=None,
        default_branch="main",
    )

    assert files == ["src/problolm/cli.py"]
    assert calls == [
        ["git", "merge-base", "HEAD", "origin/main"],
        [
            "git",
            "diff",
            "--diff-filter=AM",
            "--name-only",
            "base789...HEAD",
        ],
    ]
