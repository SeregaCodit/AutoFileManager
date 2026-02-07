from pathlib import Path
from typing import Tuple, List

from const_utils.copmarer import Constants
from const_utils.default_values import AppSettings
from logger.logger import LoggerConfigurator
from tools.comparer.img_comparer.hasher.dhash import DHash


class ImageComparer:
    """
    An orchestrator for comparing images using different hashing algorithms.

    This class acts as a manager that selects a specific hashing strategy
    (like dHash) based on the project settings. It coordinates the process
    of generating hashes and identifying duplicate files.

    Attributes:
        settings (AppSettings): Global configuration object containing
            hashing parameters and paths.
        method_mapping (Dict): A map that links algorithm names to their
            corresponding classes.
        method (BaseHasher): An instance of the selected hashing algorithm.
        logger (logging.Logger): Logger instance for tracking comparison tasks.
    """
    def __init__(self, settings: AppSettings):
        """
        Initializes the ImageComparer with settings and sets up the algorithm.

        Args:
            settings (AppSettings): Global configuration object that includes
                default and user-defined parameters.
        """
        super().__init__()
        self.settings = settings

        self.method_mapping = {
            Constants.dhash: DHash,
        }

        self.method = self.method_mapping[self.settings.method](
            settings=self.settings,
        )

        self.logger = LoggerConfigurator.setup(
            name=self.__class__.__name__,
            log_path=Path(self.settings.log_path) / f"{self.__class__.__name__}.log" if self.settings.log_path else None,
            log_level=self.settings.log_level
        )


    def compare(self, file_paths: Tuple[Path]) -> List[Path]:
        """
        Compares files using the each-with-each principle to find duplicates.

        The process consists of two steps:
        1. Building a hash map for all provided files.
        2. Analyzing the hash map to find matches that satisfy the
           Hamming distance threshold.

        Args:
            file_paths (Tuple[Path]): A collection of paths to the image files
                to be compared.

        Returns:
            List[Path]: A list of file paths that are identified as duplicates.
        """
        hash_map = self.method.get_hashmap(file_paths)
        matches = self.method.find_duplicates(hash_map)
        return matches
