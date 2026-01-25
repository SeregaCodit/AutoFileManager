from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.comparer.img_comparer.hasher.dhash import DHash


#-----FIXTURES-----
@pytest.fixture
def hasher():
    """create instance of DHash"""
    return DHash(core_size=8, threshold=10)

@pytest.fixture
def create_test_image(tmp_path):
    """
        a fixture for creating real test image in temp folder. After testing the image will be deleted
        The image will be just a simple gradient that changing pixel bright from left to right
    """
    def _generate(filename: str, width: int = 100, height: int = 100):
        img_path = tmp_path / filename
        img = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        cv2.imwrite(str(img_path), img)
        return img_path
    return _generate

@pytest.mark.parametrize("input_value, expected_vals", [
    (0.1, 0),    # int(0.1) * (16*16) = 0
    (10, 25),    # int((10 / 100 ) * (16*16)) = 25
    (1, 2),       # int(1 / 100) * (16*16) = 2
])

#-----TEST-----

def test_threshold(hasher, input_value, expected_vals):
    """test threshold setter"""

    hasher = DHash(
        hash_type="dhash",
        core_size=16,
        threshold=input_value
    )
    assert isinstance(hasher.threshold, int)
    assert hasher.threshold == expected_vals


@pytest.mark.parametrize("input_value, expected_val", [
    (16, 16),
    ((10, 25), 10),
    (17.5, 17),
    ("8", 8)
])
def test_core_size(input_value, expected_val):

    hasher = DHash(
        hash_type="dhash",
        core_size=input_value,
        threshold=10
    )
    assert isinstance(hasher.core_size, int)
    assert hasher.core_size == expected_val


@pytest.mark.parametrize("input_value, expected_val", [
    # equal hashes
    ((np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])), 0),
    # dist == 1
    ((np.array([0, 0, 0, 1]), np.array([0, 1, 0, 1])), 1),
    # dist == 2
    ((np.array([0, 0, 0, 0]), np.array([0, 1, 0, 1])), 2),
    # dist == 3
    ((np.array([1, 0, 1, 1]), np.array([0, 0, 0, 0])), 3),
    # dist == 4
    ((np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])), 4)
])
def test_calculate_distance(hasher, input_value, expected_val):
    hash1, hash2 = input_value
    dist = hasher.calculate_distance(hash1, hash2)

    assert isinstance(dist, int)
    assert dist == expected_val

@pytest.mark.parametrize("input_value, expected_val", [
    (16, 16 * 16),
    ((10, 25), 10 * 10),
    (17.5, 17 * 17),
    ("8", 8 * 8)
])
def test_compute_hash_returns_correct_shape(hasher, create_test_image, input_value, expected_val):
    # Arrange
    hasher.core_size = input_value
    image_path = create_test_image("valid.jpg")

    # Act
    result = hasher.compute_hash(image_path)

    # Assert
    assert result is not None
    assert isinstance(result, np.ndarray)
    # the shape must be core_size * core_size (16*16=256)
    assert result.shape == (expected_val,)
    assert result.dtype == bool

def test_compute_hash_with_invalid_file(hasher, tmp_path):
    # Arrange
    not_an_image = tmp_path / "text.txt"
    not_an_image.write_text("This is not a picture")

    # Act
    result = hasher.compute_hash(not_an_image)

    # Assert
    assert result is None

def test_compute_hash_consistency(hasher, create_test_image):
    # the same input must give the same result
    img_path = create_test_image("consistent.png")

    hash1 = hasher.compute_hash(img_path)
    hash2 = hasher.compute_hash(img_path)

    assert np.array_equal(hash1, hash2)


def test_threshold_conversion(hasher):
    """Перевірка, що відсоток правильно конвертувався в біти"""
    # 8*8 = 64 біти. 10% від 64 = 6.4, очікуємо 6 (int)
    assert hasher.threshold == 6

def test_find_duplicates(hasher):
    """
    Тестуємо схожість у межах 10% (поріг 6 бітів).
    Відрізняються у 4 бітах — це має бути дублікат.
    """
    h1 = np.zeros(64, dtype=bool)
    h2 = np.zeros(64, dtype=bool)
    h2[:4] = True  # відстань 4 (менше за поріг 6)

    hashmap = {Path("img1.jpg"): h1, Path("img2.jpg"): h2}

    result = hasher.find_duplicates(hashmap)
    assert Path("img2.jpg") in result

def test_find_duplicates_outside_percentage(hasher):
    """
    Тестуємо різницю понад 10%.
    Відрізняються у 10 бітах — це НЕ дублікат.
    """
    h1 = np.zeros(64, dtype=bool)
    h2 = np.zeros(64, dtype=bool)
    h2[:10] = True  # відстань 10 (більше за поріг 6)

    hashmap = {Path("img1.jpg"): h1, Path("img2.jpg"): h2}

    result = hasher.find_duplicates(hashmap)
    assert result == []

def test_find_duplicates_logic_flow(hasher):
    """
    Перевірка логічного ланцюжка.
    A і B схожі (відст 2) -> B дублікат.
    A і C схожі (відст 3) -> C дублікат.
    """
    h_a = np.zeros(64, dtype=bool)
    h_b = np.zeros(64, dtype=bool); h_b[:2] = True
    h_c = np.zeros(64, dtype=bool); h_c[:3] = True

    hashmap = {
        Path("A.jpg"): h_a,
        Path("B.jpg"): h_b,
        Path("C.jpg"): h_c
    }

    result = hasher.find_duplicates(hashmap)

    assert len(result) == 2
    assert Path("B.jpg") in result
    assert Path("C.jpg") in result
    assert Path("A.jpg") not in result