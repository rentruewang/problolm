from __future__ import annotations

from pathlib import Path


def test_ai_check_workflow_filters_python_files_with_single_dollar() -> None:
    workflow = Path(".github/workflows/ai-check.yaml").read_text(encoding="utf-8")

    assert r"grep '\.py$'" in workflow


def test_ai_check_workflow_installs_project_on_supported_python() -> None:
    workflow = Path(".github/workflows/ai-check.yaml").read_text(encoding="utf-8")

    assert "python-version: '3.14'" in workflow
    assert "pip install -e ." in workflow
