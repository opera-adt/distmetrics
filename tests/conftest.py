from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir() -> Path:
    """Return the absolute path to the test_data directory."""
    return (Path(__file__).parent / 'test_data').resolve()
