from pathlib import Path
from typing import Union

import cv2
import numpy as np

from tools.comparer.img_comparer.hasher.base_hasher import BaseHasher

class DHash(BaseHasher):
    """
    Implementation of the dHash (Difference Hashing) algorithm.

    This class provides a specific strategy for image hashing. It calculates
    the hash by comparing the brightness (intensity) difference between
    adjacent pixels in a resized version of the image. It is very effective
    at identifying visual similarities while ignoring minor changes in
    color or compression.
    """
    @staticmethod
    def compute_hash(image_path: Path, core_size: int) -> Union[np.ndarray, None]:
        """
        Calculates the dHash for a single image.

        The process includes:
        1. Loading the image in grayscale.
        2. Resizing it to (core_size + 1, core_size) to allow horizontal
           pixel comparison.
        3. Generating a boolean mask where each bit represents whether the
           left pixel is brighter than the right pixel.

        Args:
            image_path (Path): The file path to the image.
            core_size (int): The resolution used for resizing. The resulting
                hash length will be core_size squared (e.g., 8x8 = 64 bits).

        Returns:
            Union[np.ndarray, None]: A 1D NumPy array of boolean values
                representing the hash, or None if the image file is
                invalid or cannot be read.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            return None

        resized_image = cv2.resize(image, (core_size + 1, core_size), interpolation=cv2.INTER_AREA)
        gradient_difference = resized_image[:, 1:] > resized_image[:, :-1]

        return gradient_difference.flatten()
