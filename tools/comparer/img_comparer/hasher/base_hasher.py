import multiprocessing
from abc import ABC, abstractmethod
from linecache import cache
from pathlib import Path
from typing import Union, Tuple, Dict, List, Set
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from logger.logger import LoggerConfigurator
from const_utils.default_values import DefaultValues
from tools.cache import CacheIO


class BaseHasher(ABC):
    """a base hasher class"""
    def __init__(
        self,
        hash_type: str = DefaultValues.dhash,
        core_size: Union[Tuple[int, int], int] = DefaultValues.core_size,
        threshold: int = DefaultValues.hash_threshold,
        log_path: Path = DefaultValues.log_path,
        cache_io: CacheIO = CacheIO(),
        n_jobs: int = DefaultValues.n_jobs
    ):
        """

        :param hash_type: type of hash algorithm to use
        :param core_size: size of resizing image in algorithm, hash_size will be square of this value.
            biggest core size makes details at image more important. So with equal threshold with different core size
            same images can be duplicates or not duplicates
        :param threshold: threshold in percentage for hemming distance. Files that have lower hemming distance will be
            considered duplicates.
        :param log_path: path to log file
        """
        self.logger = LoggerConfigurator.setup(
            name=self.__class__.__name__,
            log_path=Path(log_path) / f"{self.__class__.__name__}.log" if log_path else None,
            log_level=DefaultValues.log_level
        )

        self.hash_type = hash_type
        self.core_size = core_size
        self.threshold = threshold
        self.cache_io = cache_io
        self.n_jobs = n_jobs
    @staticmethod
    @abstractmethod
    def compute_hash(image_path: Path, core_size: int = DefaultValues.core_size) -> np.ndarray:
        pass

    def get_hashmap(self, image_paths: Tuple[Path]) -> Dict[Path, np.ndarray]:
        """
        :param image_paths: list of images paths
        creating a dict with image path's and their hashes
        """
        image_count = len(image_paths)
        filename = self.cache_io.generate_cache_filename(
            image_paths[0].parent.resolve(),
            hash_type=self.hash_type,
            core_size=self.core_size)

        cache_file_name = DefaultValues.cache_file_path / filename
        cache_file_name.parent.mkdir(parents=True, exist_ok=True)

        hash_map = self.cache_io.load(cache_file_name)
        if hash_map:
            return hash_map

        self.logger.info(f"Building hashmap in parallel using {self.n_jobs} workers for {image_count} images...")
        hash_func = partial(self.__class__.compute_hash, core_size=self.core_size)

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            hashes = list(executor.map(hash_func, image_paths))

        hash_map = {
            path: h for path, h in zip(image_paths, hashes)
            if h is not None
        }

        self.logger.info(f"Successfully hashed {len(hash_map)} out of {image_count} images")
        self.cache_io.save(hash_map, cache_file_name)
        return hash_map

    def find_duplicates(self, hashmap: Dict[Path, np.ndarray]) -> List[Path]:
        """
        :param hashmap: a has_map of all files in the source_directory
        comparing all files via each with each principe in hashmap by hemming distance
        """

        if not hashmap:
            return []

        self.logger.info(f"Vectorizing comparison for {len(hashmap)} images...")

        paths: List[Path] = list(hashmap.keys())
        try:
            matrix = np.array(list(hashmap.values()), dtype=bool)
        except ValueError as e:
            self.logger.error("Failed to create matrix. Some hashes have different lengths!")
            raise e

        duplicates_indices: Set[int] = set()

        for index in range(len(paths)):
            if index in duplicates_indices:
                continue

            current_hash = matrix[index]
            hamming_distances = np.count_nonzero(matrix != current_hash, axis=1)
            matches = np.where(hamming_distances <= self.threshold)[0]

            for match_idx in matches:
                if match_idx > index:
                    duplicates_indices.add(int(match_idx))

        result = [paths[idx] for idx in duplicates_indices]
        self.logger.info(f"Vectorized search finished. Found {len(result)} duplicates.")
        return result

    @property
    def core_size(self) -> int:
        return self._core_size

    @core_size.setter
    def core_size(self, value: Union[Tuple[int, int], int, float, str]) -> None:
        """
        stups hash size, converts into int type
        :param value: size of hash. Using for resizing an image
        """
        if isinstance(value, int):
            self._core_size = value
        elif isinstance(value, (float, str)):
            try:
                self._core_size = int(value)
            except TypeError:
                self.logger.error(f"hash size must be int, got {type(value)}")
                raise TypeError(f"hash size must be int, got {type(value)}")
        elif isinstance(value, tuple):
            self._core_size = value[0] if value[0] >= 0 else value[1]
        else:
            self.logger.error(f"hash size must be int or tuple, got {type(value)}")
            raise TypeError(f"hash size must be int or tuple, got {type(value)}")

    @property
    def threshold(self) -> int:
        return self._threshold

    @threshold.setter
    def threshold(self, value: Union[float, int]) -> None:
        """
        setting a threshold value in percentage. If value is not float type - trying to convert it to float.
        :param value: float or int minimal value in percents that means an image is not a duplicate of comparing image
        """

        try:
            value = int(value)
        except TypeError:
            self.logger.error(f"threshold must be float, got {type(value)}")
            raise TypeError(f"threshold must be float, got {type(value)}")
        except ValueError:
            self.logger.error(f"threshold must be real number, got {type(value)}")
            raise TypeError(f"threshold must be float, got {type(value)}")

        if not (0 <= value <= DefaultValues.max_percentage):
            self.logger.error(f"threshold must be between 0 and 100, got {value}")
            raise ValueError(f"threshold must be between 0 and 100, got {type(value)}")

        hash_sqr = self.core_size * self.core_size
        self._threshold = int(hash_sqr * (value / DefaultValues.max_percentage))

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: Union[int, float, str]) -> None:
        """
            safe setting n_jobs
        """
        # check types and try to int
        if not isinstance(value, int):
            try:
                value = int(float(value))
            except TypeError:
                self.logger.error(f"n_jobs must be int, got {type(value)}")
                raise TypeError(f"n_jobs must be int, got {type(value)}")
            except ValueError:
                self.logger.error(f"n_jobs must be real number, got {type(value)}")
                raise ValueError(f"n_jobs must be real number, got {type(value)}")

        # check cores and control that n_jobs can be set
        cores = multiprocessing.cpu_count()

        if value <= 1:
            self.logger.warning(f"n_jobs must be greater than 1, got {value}")
            value = 1
        elif 1 < cores <= value:
            self.logger.warning(f"n_jobs must be less than {cores}, got {value}")
            value = cores - 1
        elif cores == 1 and value != 1:
            value = 1

        self._n_jobs = value
        self.logger.info(f"n_jobs set to {value}")