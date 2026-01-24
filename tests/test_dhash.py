import numpy as np
import pytest

from tools.comparer.img_comparer.hasher.dhash import DHash

@pytest.mark.parametrize("input_value, expected_vals", [
    (0.1, 0),    # int(0.1) * (16*16) = 0
    (10, 25),    # int((10 / 100 ) * (16*16)) = 25
    (1, 2),       # int(1 / 100) * (16*16) = 2
])
def test_threshold(input_value, expected_vals):
    """test threshold setter"""

    hasher = DHash(
        hash_type="dhash",
        hash_size=16,
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
def test_hash_size(input_value, expected_val):

    hasher = DHash(
        hash_type="dhash",
        hash_size=input_value,
        threshold=10
    )
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
def test_calculate_distance(input_value, expected_val):
    hasher = DHash()
    hash1, hash2 = input_value
    dist = hasher.calculate_distance(hash1, hash2)

    assert isinstance(dist, int)
    assert dist == expected_val

