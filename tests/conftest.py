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


@pytest.fixture
def categorical_merge_input_data() -> Path:
    """Return the absolute paths to the test categorical merge data inputs."""
    parent = (Path(__file__).parent / 'test_data' / 'categorical_merge_test_data').resolve()
    inputs = parent.glob('disturb_*.tif')
    return sorted(list(inputs))


@pytest.fixture
def categorical_merge_output_data() -> Path:
    """Return the absolute paths to the test categorical merge data inputs."""
    merged_path = (Path(__file__).parent / 'test_data' / 'categorical_merge_test_data' / 'merged.tif').resolve()
    return merged_path
