from pathlib import Path
from typing import List, Dict, Any, Union


class FeatureExtractor:
    """
    Extracts geometric and spatial characteristics from raw annotation data.

    This class processes annotation dictionaries to calculate detailed object
    features, such as relative area, position in specific image sectors
    (quadrants or sides), and truncation status at the borders.
    """
    @staticmethod
    def extract_features(filepath: Union[Path, str], data: dict, margin_threshold: int = 5) -> List[Dict[str, Any]]:
        """
        Analyzes a single annotation dictionary and computes features for every object.

        Args:
            filepath: The path to the annotation file (used as a record key).
            data: Raw dictionary containing image size and object bounding boxes.
            margin_threshold: The pixel distance from the edge to mark
                an object as truncated.

        Returns:
            List[dict]: Features for each object. Returns empty list if image size is invalid.
        """
        if not isinstance(filepath, str):
            filepath = str(filepath)

        has_neighbors = 1
        image_data = data.get("size", {})

        im_width = int(image_data.get("width", 0))
        im_height = int(image_data.get("height", 0))
        im_depth = int(image_data.get("depth", 0))
        im_area = im_width * im_height
        img_center_x = im_width / 2
        img_center_y = im_height / 2
        annotated_objects = data.get("object", [])

        if isinstance(annotated_objects, dict):
            annotated_objects = [annotated_objects]
            has_neighbors = 0

        result = []
        objects_count = len(annotated_objects)

        for obj in annotated_objects:
            bbox = obj.get("bndbox")

            xmax = int(bbox.get("xmax", 0))
            ymax = int(bbox.get("ymax", 0))
            xmin = int(bbox.get("xmin", 0))
            ymin = int(bbox.get("ymin", 0))

            width = xmax - xmin
            height = ymax - ymin
            area = width * height
            relative_area = area / im_area

            # if bbox corners in all four image quarters
            in_center = 1 if all([
                xmin <= img_center_x <= xmax,
                ymin <= img_center_y <= ymax
            ]) else 0
            # if object on im_center_y coord but has right offset
            in_right_side = 1 if all([
                ymin < img_center_y < ymax,
                xmin > img_center_x
            ]) else 0
            # if object on im_center_y coord but has left offset
            in_left_side = 1 if all([
                ymin < img_center_y < ymax,
                xmax < img_center_x
            ]) else 0
            # if object on im_center_x coord but has top offset
            in_top_side = 1 if all([
                xmin < img_center_x < xmax,
                ymax < img_center_y
            ]) else 0
            # if object on im_center_x coord but has bottom offset
            in_bottom_side = 1 if all([
                xmin < img_center_x < xmax,
                ymin > img_center_y
            ]) else 0
            # object absolutely in top left quarter
            in_left_top = 1 if all([
                xmax < img_center_x,
                ymax < img_center_y
            ]) else 0
            # object absolutely in top right quarter
            in_right_top = 1 if all([
                xmin > img_center_x,
                ymax > img_center_y
            ]) else 0
            # object absolutely in left bottom quarter
            in_left_bottom = 1 if all([
                xmax < img_center_x,
                ymin > img_center_y
            ]) else 0
            # object absolutely in right bottom quarter
            in_right_bottom = 1 if all([
                xmin > img_center_x,
                ymin > img_center_y
            ]) else 0

            truncated_left = 1 if xmin < margin_threshold else 0
            truncated_right = 1 if xmax > (im_width - margin_threshold) else 0
            truncated_top = 1 if ymin < margin_threshold else 0
            truncated_bottom = 1 if ymax > (im_height - margin_threshold) else 0

            full_size = 1 if all([
                truncated_bottom,
                truncated_left,
                truncated_right,
                truncated_top,
                truncated_bottom]) else 0

            object_data = dict(
                path=filepath,
                class_name=obj.get("name", "unfilled"),
                objects_count=objects_count,
                im_width=im_width,
                im_height=im_height,
                im_depth=im_depth,
                has_neighbors=has_neighbors,
                object_width=width,
                object_height=height,
                object_area=area,
                object_relative_area=relative_area,
                object_in_center=in_center,
                object_in_right_side=in_right_side,
                object_in_left_side=in_left_side,
                object_in_top_side=in_top_side,
                object_in_bottom_side=in_bottom_side,
                object_in_left_top=in_left_top,
                object_in_right_top=in_right_top,
                object_in_left_bottom=in_left_bottom,
                object_in_right_bottom=in_right_bottom,
                full_size=full_size,
                truncated_left=truncated_left,
                truncated_right=truncated_right,
                truncated_top=truncated_top,
                truncated_bottom=truncated_bottom,
            )
            result.append(object_data)
        return result
