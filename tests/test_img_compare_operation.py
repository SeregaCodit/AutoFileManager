import pytest
from pathlib import Path


from const_utils.default_values import DefaultValues
from file_operations.compare import CompareOperation


def test_compare_command():
    operation = CompareOperation(
        src="/mnt/84EAB1A8EAB196C0/!DATA_SHARED/PycharmProjects/AutoFileManager/media/imgs",
        dst="/mnt/84EAB1A8EAB196C0/!DATA_SHARED/PycharmProjects/AutoFileManager/media/imgs",
        threshold=10,
        method=DefaultValues.dhash
    )

    duplicates = operation.comparer.compare_files(
        image_dir=operation.src,
        threshold=operation.threshold,
    )

    assert isinstance(duplicates, list)
