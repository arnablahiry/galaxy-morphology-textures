"""Procedural synthetic galaxy image generator.

This module keeps the public ``make_galaxy`` entry point and delegates the
class-specific rendering to focused modules under :mod:`galaxy_zoo`.
"""

import numpy as np
import torch

from .barred_spiral import draw_barred_spiral
from .edge_on import draw_edge_on
from .elliptical import draw_elliptical
from .irregular import draw_irregular
from .merger import draw_merger
from .shared import add_background_and_noise, add_satellites
from .spiral import draw_spiral


CLASSES = ["elliptical", "spiral", "barred spiral", "merger",
           "edge-on", "irregular"]


def _make_meshgrid(size):
    """Create the base coordinate grid used by all renderers."""
    return torch.meshgrid(torch.linspace(-1, 1, size),
                          torch.linspace(-1, 1, size), indexing="ij")


def make_galaxy(kind, size=128, stage=None, merger_barred=None):
    """Render one synthetic galaxy image.

    Parameters
    ----------
    kind:
        Integer class index from :data:`CLASSES`.
    size:
        Output resolution in pixels.
    stage, merger_barred:
        Optional overrides used only for merger systems.
    """
    yy, xx = _make_meshgrid(size)

    # Keep each morphology framed slightly differently so the composition
    # reads naturally instead of feeling copy-pasted.
    if kind == 3:
        zoom = float(0.85 + 0.25 * torch.rand(1))
        offset_x = float(0.15 * (torch.rand(1) - 0.5) * 2)
        offset_y = float(0.15 * (torch.rand(1) - 0.5) * 2)
    elif kind == 4:
        zoom = float(0.80 + 0.30 * torch.rand(1))
        offset_x = float(0.25 * (torch.rand(1) - 0.5) * 2)
        offset_y = float(0.25 * (torch.rand(1) - 0.5) * 2)
    elif kind == 5:
        zoom = float(0.70 + 0.35 * torch.rand(1))
        offset_x = float(0.30 * (torch.rand(1) - 0.5) * 2)
        offset_y = float(0.30 * (torch.rand(1) - 0.5) * 2)
    else:
        zoom = float(0.6 + 0.4 * torch.rand(1))
        offset_x = float(0.6 * (torch.rand(1) - 0.5) * 2)
        offset_y = float(0.6 * (torch.rand(1) - 0.5) * 2)
    xx = (xx - offset_x) / zoom
    yy = (yy - offset_y) / zoom

    img = torch.zeros_like(xx)
    spin_dir = 1.0 if torch.rand(1).item() > 0.5 else -1.0

    if kind == 0:
        img = draw_elliptical(xx, yy)
    elif kind == 1:
        img = draw_spiral(xx, yy, spin_dir)
    elif kind == 2:
        img = draw_barred_spiral(xx, yy, spin_dir)
    elif kind == 3:
        img = draw_merger(xx, yy, spin_dir, stage=stage, merger_barred=merger_barred)
    elif kind == 4:
        img = draw_edge_on(xx, yy)
    elif kind == 5:
        img = draw_irregular(xx, yy)
    else:
        raise ValueError(f"unknown galaxy kind: {kind}")

    img = add_satellites(img, xx, yy)
    img = add_background_and_noise(img, size, xx, yy)
    return img.clamp(0, 3).unsqueeze(0)
