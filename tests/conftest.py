from pathlib import Path

import pytest


@pytest.fixture
def cropped_despeckled_data_dir() -> Path:
    """Return the absolute path to the cropped despeckled data directory."""
    return (Path(__file__).parent / 'test_data' / 'T009_019294_IW2_cropped_tv').resolve()


@pytest.fixture
def cropped_vh_data_dir() -> Path:
    """Return the absolute path to the cropped data directory."""
    return (Path(__file__).parent / 'test_data' / 'T009_019294_IW2_cropped').resolve()
