# Copyright (c) ProBloLM Authors - All Rights Reserved

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).parent.absolute()
