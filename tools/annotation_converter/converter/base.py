from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

from logger.log_level_mapping import LevelMapping
from logger.logger import LoggerConfigurator
from tools.annotation_converter.reader.base import BaseReader
from tools.annotation_converter.reader.voc import XMLReader
from tools.annotation_converter.reader.yolo import TXTReader
from tools.annotation_converter.writer.base import BaseWriter
from tools.annotation_converter.writer.voc import XMLWriter
from tools.annotation_converter.writer.yolo import YoloWriter


class BaseConverter(ABC):
    """
    Base converter class. Based on the source and destination formats, defines reader and writer classes for
        processing data, defines default source and destination annotation file suffixes

    """
    def __init__(
            self,
            source_format: str,
            dest_format: str,
            log_level: str = LevelMapping.debug,
            log_path: Optional[Path] = None,
            **kwargs
    ):
        """
        Initialize the converter class.

        Args:
            source_format (str): The format of source annotation (e.g., 'yolo').
            dest_format (str): The format of output annotations (e.g., 'voc').
            log_level: (str): The lowest logging level print to (e.g., 'debug').
            **kwargs (dict): Additional parameters like 'img_path' or 'labels_path'.
        """
        self.reader_mapping = {
            ".xml": XMLReader,
            ".txt": TXTReader
        }

        self.writer_mapping = {
            ".txt": YoloWriter,
            ".xml": XMLWriter
        }

        self.suffix_mapping = {
            "voc": ".xml",
            "yolo": ".txt"
        }

        self.source_suffix = self.suffix_mapping.get(source_format)
        self.dest_suffix = self.suffix_mapping.get(dest_format)
        self.reader = self.reader_mapping[self.source_suffix]()
        self.writer = self.writer_mapping[self.dest_suffix]()

        self.logger = LoggerConfigurator.setup(
            name=self.__class__.__name__,
            log_level=log_level,
            log_path=Path(log_path) / f"{self.__class__.__name__}.log" if log_path else None
        )


    @abstractmethod
    def convert(self, file_paths: Tuple[Path], target_path: Path, n_jobs: int = 1) -> None:
        """
        Abstract method to convert source annotation file to destination annotation file by running custom workers in
            subclasses

        Args:
            file_paths (Tuple[Path]): List of paths to the annotation files.
            target_path (Path): Directory where converted files will be stored.
            n_jobs (int): Number of parallel workers to use. Defaults to 1.
        """
        pass
    
    @property
    def reader(self) -> BaseReader:
        """BaseReader: returns a reader for the annotation file."""
        return self._reader
    
    @reader.setter
    def reader(self, reader: BaseReader) -> None:
        """
        Sets a reader for the annotation filetype.

        Args:
            reader (BaseReader): Reader for the annotation filetype.
        """
        self._reader = reader

    @property
    def writer(self) -> BaseWriter:
        """BaseWriter: returns a writer for the annotation filetype."""
        return self._writer

    @writer.setter
    def writer(self, writer: BaseWriter) -> None:
        """
        Sets a writer for the annotation filetype.

        Args:
            writer (BaseWriter): Writer for the annotation filetype.
        """
        self._writer = writer