"""
Pytest configuration and fixtures for the lacuna_rnn test suite.
"""

import shutil
import tempfile

import pytest

# No need to modify Python path - pytest handles package imports from project root


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_greek_text():
    """Provide sample Greek text for testing."""
    return "αβγδεζηθικλμνξοπρστυφχψω"


@pytest.fixture
def sample_text_with_lacunae():
    """Provide sample text with lacunae markers for testing."""
    return "αβγ[δε]ζη[θι]κλμ"


@pytest.fixture
def sample_json_data():
    """Provide sample JSON data structure for testing."""
    return [
        {
            "language": "grc",
            "training_text": "αβγ[δε]ζη",
            "test_cases": [{"alternatives": ["δε"]}]
        },
        {
            "language": "grc",
            "training_text": "ικλ<gap/>μνξ",
            "test_cases": [{"alternatives": ["οπ"]}]
        }
    ]


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    import logging

    # Set logging level to INFO so we can capture log messages in tests
    logging.getLogger().setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)