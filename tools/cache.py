import hashlib
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

from const_utils.default_values import AppSettings
from logger.logger import LoggerConfigurator
from logger.logger_protocol import LoggerProtocol


class CacheIO:
    SUFFIX = ".parquet"
    def __init__(self, settings: AppSettings):
        """Initializes the CacheIO class.

        This class helps to save and load cache files. It makes data loading faster.

        Args:
            settings (AppSettings): The application settings for logging and paths.
        """
        self.settings = settings
        self.logger = LoggerConfigurator.setup(
            name=self.__class__.__name__,
            log_path=Path(self.settings.log_path) / f"{self.__class__.__name__}.log",
            log_level=self.settings.log_level
        )


    def load(self: LoggerProtocol, cache_file: Path) -> Dict[Path, np.ndarray]:
        """Loads data from a cache file.

        It reads a parquet file and converts it into a dictionary of hashes.

        Args:
            cache_file (Path): The path to the cache file.

        Returns:
            Dict[Path, np.ndarray]: A dictionary where keys are file paths and values are hashes.
                Returns an empty dictionary if the file does not exist or is broken.
        """
        if not cache_file.exists():
            self.logger.warning(f"Cache file {cache_file} does not exist")
            return {}

        try:
            self.logger.info(f"Loading cache file {cache_file}")
            df = pd.read_parquet(cache_file)
            data = {
                Path(row['path']): np.array(row['hash'], dtype=bool)
                for _, row in df.iterrows()
            }
            return data
        except EOFError:
            self.logger.warning(f"Cache file {cache_file} is damaged. Deleting cache file")
            return {}


    def save(self: LoggerProtocol, hash_map: Dict[Path, np.ndarray], cache_file: Path) -> None:
        """Saves a hash map to a cache file.

        It converts the dictionary into a parquet file for later use.

        Args:
            hash_map (Dict[Path, np.ndarray]): The dictionary of hashes to save.
            cache_file (Path): The path where the cache file will be saved.
        """
        if not hash_map:
            self.logger.warning("Hash map is empty, skipping saving cache file")
            return

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving {len(hash_map)} hashes to {cache_file.name}")

        try:
            data = [
                {'path': str(p), 'hash': h.tolist()}
                for p, h in hash_map.items()
            ]
            df = pd.DataFrame(data)
            df.to_parquet(cache_file, engine="pyarrow", compression="snappy", index=False)
            self.logger.info(f"Cache saved successfully.")
        except Exception as e:
            self.logger.error(f"Critical error saving cache: {e}")


    @classmethod
    def generate_cache_filename(cls, source_path: Path, hash_type: str, core_size: int, cache_name: Optional[Path]) -> str:
        """Creates a unique name for the cache file.

        The name is based on the folder path, hash type, and size.

        Args:
            source_path (Path): The folder path being processed.
            hash_type (str): The type of hash used (e.g., 'dhash').
            core_size (int): The size of the hash.
            cache_name (Optional[Path]): A custom name for the cache file, if provided.

        Returns:
            str: The generated filename for the cache.
        """
        suffix = f"{hash_type}_s{core_size}{cls.SUFFIX}"
        if cache_name is None:
            abs_path = str(source_path.resolve())
            path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()
            folder_name = str(source_path.name.replace(' ', '_').strip("."))[:30]
            return f"cache_{path_hash}_d{folder_name}{suffix}"
        else:
            cache_name = str(cache_name).replace(" ", "_").strip(".")
            if cache_name.endswith(cls.SUFFIX):
                cache_name = cache_name[:-len(cls.SUFFIX)]

            cache_name = f"{cache_name}_{suffix}"
            return cache_name
