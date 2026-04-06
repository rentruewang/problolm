from __future__ import annotations

from pathlib import Path


def test_package_has_module_entrypoint() -> None:
    entrypoint = Path("src/problolm/__main__.py")

    assert entrypoint.exists()
