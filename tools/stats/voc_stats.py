from pathlib import Path
from typing import Optional, Dict, List

from tools.annotation_converter.reader.base import BaseReader
from tools.stats.base_stats import BaseStats
from tools.stats.extractor import FeatureExtractor


class VOCStats(BaseStats):
    """
    Statistics analyzer for Pascal VOC annotation format.

    This class implements the worker logic to process XML files. It coordinates
    the reading of files and the extraction of geometric features for
    dataset analysis.
    """
    @staticmethod
    def _analyze_worker(
            file_path: Path,
            reader: BaseReader,
            margin_threshold: int = 5,
            class_mapping: Optional[Dict[str, str]] = None

    ) -> List[Dict[str, str]]:
        """
        Processes a single VOC XML file to extract object features.

        Args:
            file_path (Path): Path to the .xml annotation file.
            reader (BaseReader): An instance of XMLReader to parse the file.
            margin_threshold (int): Distance from the image edge to detect truncation.
            class_mapping (Optional[Dict[str, str]]): Optional mapping to rename classes.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing features
                for each object found in the XML. Returns an empty list
                if the file is invalid.
        """
        try:
            annotation_data = reader.read(file_path).get("annotation")

            if not annotation_data:
                return []

            stat_data = FeatureExtractor.extract_features(file_path, annotation_data, margin_threshold)
            return stat_data
        except Exception as e:
            return []
