from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Dict, Set, Tuple

import numpy as np

from tools.annotation_converter.converter.base import BaseConverter
from tools.annotation_converter.reader.base import BaseReader


class VocYOLOConverter(BaseConverter):
    TARGET_FORMAT = ".xml"
    DESTINATION_FORMAT = ".txt"
    def __init__(self, tolerance: int = 6):
        super().__init__()

        self.tolerance = tolerance
        self.reader = self.reader_mapping[self.TARGET_FORMAT]()
        self.writer = self.writer_mapping[self.DESTINATION_FORMAT]()
        self.objects: list = list()
        self.class_mapping: Dict[str, int] = dict()

    @staticmethod
    def _get_classes_worker(annotation_paths: Path, reader: BaseReader) -> Set[str]:
        try:
            data = reader.read(annotation_paths)
            annotation = data.get("annotation", {})
            objects = annotation.get("object", list())
            if not isinstance(objects, list):
                objects = [objects]
            return {obj["name"] for obj in objects}
        except Exception:
            return set()

    @staticmethod
    def _convert_worker(file_path: Path, reader: BaseReader, class_mapping: Dict[str, int], tolerance: int) -> List[str]:
        data = reader.read(file_path)

        if data.get("annotation") is None:
            return []

        annotation = data["annotation"]

        try:
            img_width = int(annotation["size"]["width"])
            img_height = int(annotation["size"]["height"])

            if img_width == 0 or img_height == 0:
                raise ValueError(f"Image size is zero in annotation {file_path}!")
        except (KeyError, ValueError, TypeError):
            return []

        annotated_objects = annotation.get("object", list())

        # reader using xmltodict that returns a dict if there is just one object, if more - returns a list
        if not isinstance(annotated_objects, list):
            annotated_objects = [annotated_objects]

        converted_objects = list()

        for obj in annotated_objects:
            try:
                # saving objectnames for classes.txt
                name = obj["name"]

                if name not in class_mapping:
                    continue


                class_id = class_mapping[name]

                # calculate yolo format cords
                bbox = obj["bndbox"]
                xmin, ymin, xmax, ymax = (
                    float(bbox["xmin"]), float(bbox["ymin"]),
                    float(bbox["xmax"]), float(bbox["ymax"])
                )

                width = ((xmax - xmin) / img_width)
                height = (ymax - ymin) / img_height
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height

                x_center, y_center, width, height = map(lambda x: np.clip(x, 0, 1),
                                                        [x_center, y_center, width, height])

                row = (f"{class_id} "
                       f"{x_center:.{tolerance}f} "
                       f"{y_center:.{tolerance}f} "
                       f"{width:.{tolerance}f} "
                       f"{height:.{tolerance}f}")

                converted_objects.append(row)
            except (KeyError, ValueError, TypeError):
                continue
        return converted_objects

    def convert(self, file_paths: Tuple[Path], target_path: Path, n_jobs: int = 1) -> None:
        # TODO: add file writing as multiprocessing
        self.logger.info(f"Start converting annotations with {n_jobs} workers...")

        classes_func = partial(self._get_classes_worker, reader=self.reader)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            classes = list(executor.map(classes_func, file_paths))

        self.objects = sorted(set().union(*classes))
        class_mapping = {name: i for i, name in enumerate(self.objects)}
        self.logger.info(f"Unified class mapping created: {len(self.objects)} classes")

        worker_func = partial(
            self._convert_worker,
            reader=self.reader,
            class_mapping=class_mapping,
            tolerance=self.tolerance
        )

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for source_path, yolo_data in zip(file_paths, executor.map(worker_func, file_paths)):
                if yolo_data:
                    dest_file = target_path / (source_path.stem + self.DESTINATION_FORMAT)
                    self.writer.write(yolo_data, dest_file)

                # Зберігаємо файл класів (специфіка YOLO)
            self.writer.write(self.objects, target_path / "classes.txt")

    # def convert(self, file_path: Path) -> List[str]:
    #     data = self.reader.read(file_path)
    #
    #     if data.get("annotation") is None:
    #         return []
    #
    #     converted_objects = list()
    #     annotation = data["annotation"]
    #
    #     try:
    #         img_width = int(annotation["size"]["width"])
    #         img_height = int(annotation["size"]["height"])
    #
    #         if img_width == 0 or img_height == 0:
    #             self.logger.warning(f"Image size is zero in annotation {file_path}!")
    #             raise ValueError(f"Image size is zero in annotation {file_path}!")
    #     except (KeyError, ValueError, TypeError) as e:
    #             self.logger.warning(f"Skipping {file_path.name}: Invalid image size metadata. Error: {e}")
    #             return []
    #
    #     annotated_objects  = annotation.get("object", list())
    #
    #     # reader using xmltodict that returns a dict if there is just one object, if more - returns a list
    #     if not isinstance(annotated_objects, list):
    #         annotated_objects = [annotated_objects]
    #
    #     for obj in annotated_objects:
    #         try:
    #             # saving objectnames for classes.txt
    #             name = obj["name"]
    #
    #             if name not in self.class_mapping:
    #                 self.objects.append(name)
    #                 self.class_mapping[name] = len(self.objects) - 1
    #
    #             class_id = self.class_mapping[name]
    #
    #             # calculate yolo format cords
    #             bbox = obj["bndbox"]
    #             xmin, ymin, xmax, ymax = (
    #                 float(bbox["xmin"]), float(bbox["ymin"]),
    #                 float(bbox["xmax"]), float(bbox["ymax"])
    #             )
    #
    #             width = ((xmax - xmin) / img_width)
    #             height = (ymax - ymin) / img_height
    #             x_center = (xmin + xmax) / 2 / img_width
    #             y_center = (ymin + ymax) / 2 / img_height
    #
    #             x_center, y_center, width, height = map(lambda x: np.clip(x, 0, 1),
    #                                                     [x_center, y_center, width, height])
    #
    #             row = (f"{class_id} "
    #                    f"{x_center:.{self.tolerance}f} "
    #                    f"{y_center:.{self.tolerance}f} "
    #                    f"{width:.{self.tolerance}f} "
    #                    f"{height:.{self.tolerance}f}")
    #
    #             converted_objects.append(row)
    #         except (KeyError, ValueError, TypeError) as e:
    #             self.logger.warning(f"Skipping object in {file_path.name}: Missing or invalid bndbox data. Error: {e}")
    #             continue
    #     return converted_objects

    @property
    def tolerance(self) -> int:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: int):
        if isinstance(value, int):
            self._tolerance = value
        else:
            try:
                self._tolerance = int(float(value))
            except TypeError as e:
                self.logger.warning(f"Can`t convert {value} to int from type {type(value)})\n{e}")
                raise TypeError(e)
