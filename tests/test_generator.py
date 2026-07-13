"""Tests for the galaxy texture generator."""

import pytest
import torch

from galaxy_textures import CLASSES, make_galaxy


def test_class_labels_are_stable():
    """The public class order is part of the dataset contract."""
    assert CLASSES == [
        "elliptical",
        "spiral",
        "barred spiral",
        "merger",
        "edge-on",
        "irregular",
    ]


def test_make_galaxy_returns_expected_shape_and_range():
    """Each morphology should render to a single-channel bounded tensor."""
    torch.manual_seed(0)
    for kind in range(len(CLASSES)):
        image = make_galaxy(kind=kind, size=16)
        assert tuple(image.shape) == (1, 16, 16)
        assert image.dtype == torch.float32
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 3.0


def test_make_galaxy_rejects_unknown_kind():
    """Invalid class ids should fail fast instead of producing junk output."""
    with pytest.raises(ValueError, match="unknown galaxy kind"):
        make_galaxy(kind=99, size=16)