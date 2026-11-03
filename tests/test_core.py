import numpy as np
import pytest

from brone.core import _iou, align, match, unit, TEMPLATE


def test_unit_and_match():
    gallery = {"a": unit([1, 0]), "b": unit([0, 1])}
    assert np.isclose(np.linalg.norm(unit([3, 4])), 1)
    assert match([1, 0], gallery, 0.8)["predicted_id"] == "a"
    assert match([-1, 0], gallery, 0.8)["predicted_id"] == "unknown"


def test_alignment_identity():
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[50, 38] = (255, 0, 0)
    result = align(image, TEMPLATE)
    assert result.shape == (112, 112, 3)
    assert result[50, 38, 0] > 200


def test_iou():
    assert np.isclose(_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])), 1)
    assert _iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0


def test_zero_embedding():
    with pytest.raises(ValueError):
        unit([0, 0])
