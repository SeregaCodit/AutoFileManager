from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from logger.log_level_mapping import LevelMapping
from logger.logger import LoggerConfigurator
from tools.annotation_converter.reader.voc import XMLReader
from tools.annotation_converter.reader.yolo import TXTReader


class BaseStats(ABC):
    """
    Base stats class. Based on the source format, defines reader classes for processing data, defines default
        source annotation file suffixes
    """
    def __init__(
            self,
            source_format: str,
            log_level: str = LevelMapping.debug,
            log_path: Optional[Path] = None,
            **kwargs
    ):
        """
        Initialize the stats class.

        Args:
            source_format (str): The format of source annotation (e.g., 'yolo').
            log_level: (str): The lowest logging level print to (e.g., 'debug').
            **kwargs (dict): Additional parameters like 'img_path' or 'labels_path'.
        """
        self.reader_mapping = {
            ".xml": XMLReader,
            ".txt": TXTReader
        }

        self.suffix_mapping = {
            "voc": ".xml",
            "yolo": ".txt"
        }

        self.source_suffix = self.suffix_mapping.get(source_format)
        self.reader = self.reader_mapping[self.source_suffix]()

        self.logger = LoggerConfigurator.setup(
            name=self.__class__.__name__,
            log_level=log_level,
            log_path=Path(log_path) / f"{self.__class__.__name__}.log" if log_path else None
        )


    @abstractmethod
    def get_stats(self):
        pass