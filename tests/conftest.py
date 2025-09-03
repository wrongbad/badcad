"""Shared pytest fixtures for the badcad test suite."""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for tests."""
    temp_file = temp_dir / "test_file.txt"
    temp_file.touch()
    return temp_file


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = MagicMock()
    config.debug = False
    config.verbose = False
    config.output_format = "stl"
    return config


@pytest.fixture
def sample_points():
    """Sample 3D points for geometry testing."""
    return [
        (0, 0, 0),
        (1, 0, 0), 
        (1, 1, 0),
        (0, 1, 0)
    ]


@pytest.fixture
def sample_mesh_data():
    """Sample mesh data for testing."""
    return {
        "vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        "faces": [(0, 1, 2)],
        "normals": [(0, 0, 1)]
    }


@pytest.fixture
def mock_display():
    """Mock display object for testing UI components."""
    display = MagicMock()
    display.show = MagicMock()
    display.clear = MagicMock()
    return display


@pytest.fixture(autouse=True)
def cleanup_test_files(temp_dir):
    """Automatically cleanup any test files after each test."""
    yield
    # Cleanup is handled automatically by temp_dir fixture