from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Dict, Set, Tuple

import numpy as np

from tools.annotation_converter.converter.base import BaseConverter


class YoloVocConverter(BaseConverter):
    CLASSES_FILE = "classes.txt"
    DEFAULT_OBJ_NAME = "annotated_object"

    def __init__(self, source_format, dest_format):
        """
        Convert annotations from YOLO (.txt format) to VOC (.xml format). If classes.txt not found in source folder,
        all classes will be annotated as '{self.DEFAULT_OBJ_NAME}'.
            Params:

        """
        super().__init__(source_format, dest_format)
        self.objects: list = list()
        self.class_mapping: Dict[str, int] = dict()

    def convert(self, file_paths: Tuple[Path], target_path: Path, n_jobs: int = 1) -> None:
        classes_file = next((path for path in file_paths if path.name == self.CLASSES_FILE), self.DEFAULT_OBJ_NAME)
        if classes_file is None:
            self.logger.warning(
                f"No classes file found at {target_path}, all classes will be annotated as '{self.DEFAULT_OBJ_NAME}'"
            )


        print(file_paths)
        pass