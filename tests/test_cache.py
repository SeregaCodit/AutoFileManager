import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.cache import CacheIO
from const_utils.default_values import AppSettings


@pytest.fixture
def mock_settings():
    """Provides mock AppSettings for testing CacheIO."""
    settings = AppSettings(
        log_path=Path("./test_log"),
        cache_file_path=Path("./test_cache")
    )
    return settings


@pytest.fixture
def cache_io_instance(mock_settings):
    """Provides a CacheIO instance with mock settings."""
    return CacheIO(settings=mock_settings)


@pytest.fixture
def temp_cache_file(tmp_path):
    """Provides a temporary cache file path for testing."""
    return tmp_path / "test_cache.parquet"


@pytest.fixture
def sample_hash_map():
    """Provides a sample hash map for testing."""
    return {
        Path("/path/to/file1.jpg"): np.array([True, False, True], dtype=bool),
        Path("/path/to/file2.png"): np.array([False, True, False], dtype=bool),
    }


def test_cache_io_init(cache_io_instance):
    """Tests if CacheIO initializes correctly."""
    assert isinstance(cache_io_instance, CacheIO)
    assert cache_io_instance.settings is not None
    assert cache_io_instance.logger is not None


def test_save_empty_hash_map(cache_io_instance, temp_cache_file):
    """Tests saving an empty hash map, which should not create a file."""
    hash_map = {}
    cache_io_instance.save(hash_map, temp_cache_file)
    assert not temp_cache_file.exists()


def test_save_valid_hash_map(cache_io_instance, temp_cache_file, sample_hash_map):
    """Tests saving a valid hash map and checks if the file is created."""
    cache_io_instance.save(sample_hash_map, temp_cache_file)
    assert temp_cache_file.exists()


def test_load_non_existent_file(cache_io_instance, temp_cache_file):
    """Tests loading from a cache file that does not exist."""
    loaded_data = cache_io_instance.load(temp_cache_file)
    assert loaded_data == {}


def test_load_damaged_file(cache_io_instance, temp_cache_file):
    """Tests loading from a damaged cache file (simulated by EOFError)."""
    temp_cache_file.write_text("corrupted data")
    with patch('pandas.read_parquet', side_effect=EOFError):
        loaded_data = cache_io_instance.load(temp_cache_file)
        assert loaded_data == {}


def test_load_valid_file(cache_io_instance, temp_cache_file, sample_hash_map):
    """Tests loading from a valid cache file."""
    cache_io_instance.save(sample_hash_map, temp_cache_file)
    loaded_data = cache_io_instance.load(temp_cache_file)

    assert len(loaded_data) == len(sample_hash_map)
    for path, hash_array in sample_hash_map.items():
        assert path in loaded_data
        assert np.array_equal(loaded_data[path], hash_array)


def test_generate_cache_filename_no_custom_name():
    """Tests generating a cache filename without a custom name."""
    source_path = Path("/home/user/images")
    hash_type = "dhash"
    core_size = 16
    filename = CacheIO.generate_cache_filename(source_path, hash_type, core_size, None)
    assert filename.startswith("cache_")
    assert "_dimages" in filename
    assert "dhash_s16.parquet" in filename


def test_generate_cache_filename_with_custom_name():
    """Tests generating a cache filename with a custom name."""
    source_path = Path("/home/user/videos")
    hash_type = "phash"
    core_size = 32
    custom_name = Path("my_video_cache")
    filename = CacheIO.generate_cache_filename(source_path, hash_type, core_size, custom_name)
    assert filename == "my_video_cache_phash_s32.parquet"


def test_generate_cache_filename_with_custom_name_with_suffix():
    """Tests generating a cache filename with a custom name that already includes the suffix."""
    source_path = Path("/home/user/documents")
    hash_type = "ahash"
    core_size = 8
    custom_name = Path("doc_cache.parquet")
    filename = CacheIO.generate_cache_filename(source_path, hash_type, core_size, custom_name)
    assert filename == "doc_cache_ahash_s8.parquet"
