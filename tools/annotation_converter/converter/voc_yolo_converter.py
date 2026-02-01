from pathlib import Path
from typing import Optional, List
from tools.annotation_converter.converter.base import BaseConverter


class VocYOLOConverter(BaseConverter):
    TARGET_FORMAT = ".xml"
    DESTINATION_FORMAT = ".txt"
    def __init__(self, tolerance: int = 6):
        super().__init__()

        self.tolerance = tolerance
        self.reader = self.reader_mapping[self.TARGET_FORMAT]()
        self.writer = self.writer_mapping[self.DESTINATION_FORMAT]()
        self.classes: Optional[str] = None
        self.objects: list = list()


    def read(self, source: str) -> str:
        pass

    def convert(self, file_path: Path) -> List[str]:
        data = self.reader.read(file_path)
        converted_objects = list()

        if data is not None:
            annotated_objects  = data["annotation"]["object"]

            if not isinstance(annotated_objects, list):
                annotated_objects = [annotated_objects]

            for obj in annotated_objects:
                # saving objectnames for classes.txt
                if obj["name"] not in self.objects:
                    self.objects.append(obj["name"])

                # calculate yolo format cords
                xmin = int(obj["bndbox"]["xmin"])
                ymin = int(obj["bndbox"]["ymin"])
                xmax = int(obj["bndbox"]["xmax"])
                ymax = int(obj["bndbox"]["ymax"])

                img_width, img_height = map(int, list(data["annotation"]["size"].values())[:2])
                width = ((xmax - xmin) / img_width)
                height = (ymax - ymin) / img_height
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height

                cords = map(lambda x: round(x, self.tolerance),
                                                        [x_center, y_center, width, height])


                converted_data = map(lambda x: str(x), [self.objects.index(obj["name"])] + list(cords))
                data_string = " ".join(converted_data)
                converted_objects.append(data_string)

        return converted_objects

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
