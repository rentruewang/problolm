# Copyright (c) ProBloLM Authors - All Rights Reserved

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).parent.absolute()
