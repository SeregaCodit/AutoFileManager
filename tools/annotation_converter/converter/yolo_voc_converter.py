from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import xmltodict

from tools.annotation_converter.converter.base import BaseConverter
from tools.annotation_converter.reader.base import BaseReader
from tools.annotation_converter.writer.base import BaseWriter


class YoloVocConverter(BaseConverter):
    CLASSES_FILE = "classes.txt"
    DEFAULT_OBJ_NAME = "annotated_object"
    _worker_image_map = {}
    def __init__(
            self,
            source_format: str,
            dest_format: str,
            extensions: Tuple[str, ...],
            **kwargs
    ):
        """
        Convert annotations from YOLO (.txt format) to VOC (.xml format). If classes.txt not found in source folder,
        all classes will be annotated as '{self.DEFAULT_OBJ_NAME}'.
            Params:

        """
        super().__init__(source_format, dest_format, **kwargs)
        self.extensions: Tuple[str, ...] = extensions
        self.labels_path: Path = kwargs.get("labels_path", None)
        self.img_path: Path = kwargs.get("img_path", None)
        self.objects: list = list()
        self.object_mapping: Dict[str, str] = dict()

    @classmethod
    def _init_worker(cls, image_dict: Dict[str, str]):
        cls._worker_image_map = image_dict

    @staticmethod
    def _convert_worker(
            file_path: Path,
            destination_path: Path,
            reader: BaseReader,
            writer: BaseWriter,
            class_mapping: Dict[str, str],
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
        :param suffix: suffix to add to filename
        :type suffix: str
        :return: True if a file was successfully converted, else returns False

        """

        yolo_annotations = reader.read(file_path).keys()

        if not yolo_annotations:
            return False

        objects = list()
        # ----- correspond image information -----
        correspond_img_str = YoloVocConverter._worker_image_map.get(file_path.stem)

        if correspond_img_str is None:
            return False

        img_height, img_width, im_depth = cv2.imread(correspond_img_str).shape
        correspond_img = Path(correspond_img_str)
        im_path = correspond_img.resolve()
        im_dir = correspond_img.parent.stem
        im_name = correspond_img.name

        for annotation in yolo_annotations:
            class_id, *coords = annotation.split(" ")
            class_name = class_mapping.get(class_id, f"object_{class_id}")
            obj_xcenter, obj_ycenter, obj_width, obj_height = map(float, coords)
            # ----- bbox transformation -----
            xmin = round(((obj_xcenter - obj_width / 2) * img_width))
            ymin = round((obj_ycenter - obj_height / 2) * img_height)
            xmax = round((obj_xcenter + obj_width / 2) * img_width)
            ymax= round((obj_ycenter + obj_height / 2) * img_height)
            voc_object = {
                "name": class_name,
                "pose": "Unspecified",
                "truncated": int(
                    any([
                        xmin <= 0,
                        ymin <= 0,
                        xmax >= img_width,
                        ymax >= img_height,
                    ])
                ),
                "difficult": 0,
                "bndbox":{
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            }
            objects.append(voc_object)

        converted_dict = {
            "annotation": {
                "folder": im_dir,
                "filename": im_name,
                "path": im_path,
                "source": {
                    "database": "Unknown"
                },
                "size": {
                    "width": img_width,
                    "height": img_height,
                    "depth": im_depth
                },
                "segmented": 0,
                "object": objects if len(objects) > 1 else objects[0]
            }
        }
        try:
            xml = xmltodict.unparse(converted_dict, pretty=True)
            annotation_path = Path(destination_path / f"{file_path.stem}{suffix}")
            writer.write(data=xml, file_path=annotation_path)
        except Exception:
            return False
        return True

    def convert(self, file_paths: Tuple[Path], target_path: Path, n_jobs: int = 1) -> None:
        target_path.mkdir(exist_ok=True, parents=True)
        classes_file = next((path for path in file_paths if path.name == self.CLASSES_FILE), None)
        if classes_file is None:
            self.logger.error(
                f"No classes file found at {target_path}, all classes will be annotated as 'object_<id>'"
            )
        file_paths = tuple(f for f in file_paths if f.name != self.CLASSES_FILE)
        count_to_convert = len(file_paths)
        self.logger.info(
            f"Starting converting from YOLO format to VOC format for {count_to_convert} files, with {n_jobs} workers"
        )
        self.object_mapping = self.reader.read(classes_file)
        self.object_mapping = {value: key for key, value in self.object_mapping.items()}
        images = {img.stem: str(img.resolve()) for img in self.img_path.iterdir() if img.suffix.lower() in self.extensions}

        convert_func = partial(
            self.__class__._convert_worker,
            destination_path=target_path,
            reader=self.reader,
            writer=self.writer,
            class_mapping=self.object_mapping,
            suffix=self.dest_suffix
        )

        with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=self.__class__._init_worker,
                initargs=(images,)
        ) as executor:
            converted_results = executor.map(convert_func, file_paths)
            converted_count = sum(converted_results)

        self.logger.info(f"Converted {converted_count}/{count_to_convert} annotations from YOLO to VOC")


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
            msg = f"img_path must be Path or str, not {type(img_path)}"
            self.logger.error(msg)
            raise TypeError(msg)

    @property
    def extensions(self) -> Tuple[str, ...]:
        return self._extensions

    @extensions.setter
    def extensions(self, value: Tuple[str, ...]) -> None:
        if isinstance(value, tuple):
            self._extensions = value
        else:
            try:
                self._extensions = tuple(value)
            except TypeError as e:
                msg = f"extensions must be convertable into tuple, got {type(value)}"
                self.logger.error(msg)
                raise TypeError(msg)
