from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any, Union

import numpy as np

from tools.annotation_converter.converter.base import BaseConverter
from tools.annotation_converter.reader.base import BaseReader
from tools.annotation_converter.writer.base import BaseWriter


class YoloVocConverter(BaseConverter):
    CLASSES_FILE = "classes.txt"
    DEFAULT_OBJ_NAME = "annotated_object"

    def __init__(self, source_format: str, dest_format: str, **kwargs):
        """
        Convert annotations from YOLO (.txt format) to VOC (.xml format). If classes.txt not found in source folder,
        all classes will be annotated as '{self.DEFAULT_OBJ_NAME}'.
            Params:

        """
        super().__init__(source_format, dest_format, **kwargs)
        self.labels_path: Path = kwargs.get("labels_path", None)
        self.img_path: Path = kwargs.get("img_path", None)
        self.objects: list = list()
        self.object_mapping: Dict[str, int] = dict()

    @staticmethod
    def _convert_worker(
            file_path: Path,
            destination_path: Path,
            reader: BaseReader,
            writer: BaseWriter,
            class_mapping: Dict[str, int],
            img_path: Path,
            suffix: str
    ) -> bool:
        """
        pipline for parsing annotations, recalculating annotated objects data to YOLO format and savin it in
            destination path

        :param file_path: path to annotation file
        :type file_path: Path
        :param destination_path: path to output annotation file
        :type destination_path: Path
        :param reader: reader object for parsing annotations
        :type reader: BaseReader
        :param writer: writer object for writing converted annotation files
        :type writer: BaseWriter
        :param class_mapping: mapping from class name to class id
        :type class_mapping: Dict[str, int]
        :param tolerance: an int value that determines to which decimal place to round a converted in YOLO
            format coordinates.
        :type tolerance: int
        :param suffix: suffix to add to filename
        :type suffix: str
        :return: True if a file was successfully converted, else returns False

        """

        data = reader.read(file_path)

        if not data:
            return False

        return True

    def convert(self, file_paths: Tuple[Path], target_path: Path, n_jobs: int = 1) -> None:
        classes_file = next((path for path in file_paths if path.name == self.CLASSES_FILE), self.DEFAULT_OBJ_NAME)
        if classes_file is None:
            self.logger.warning(
                f"No classes file found at {target_path}, all classes will be annotated as '{self.DEFAULT_OBJ_NAME}'"
            )

        self.object_mapping = self.reader.read(classes_file)

        convert_func = partial(
            self.__class__._convert_worker,
            destination_path=target_path,
            reader=self.reader,
            writer=self.writer,
            class_mapping=self.object_mapping,
            img_path = self.img_path,
            suffix=self.dest_suffix
        )

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            executor.map(convert_func, file_paths)

    @property
    def img_path(self) -> Path:
        return self._img_path

    @img_path.setter
    def img_path(self, img_path: Union[Path, str, None]) -> None:
        if isinstance(img_path, Path):
            self._img_path = img_path
        elif isinstance(img_path, str):
            self._img_path = Path(img_path)
        elif img_path is None :
            self._img_path = self.labels_path
            self.logger.warning(f"Dataset images path is not defined. Set same annotations path: {self.labels_path}")
        else:
            self.logger.error(f"img_path must be Path or str, not {type(img_path)}")
            raise TypeError(f"img_path must be Path or str, not {type(img_path)}")