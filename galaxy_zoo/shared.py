"""Shared rendering helpers for the synthetic galaxy generators."""

import numpy as np
import torch


def add_satellites(img, xx, yy):
    """Add faint satellite blobs around the main galaxy image."""
    num_satellites = int(torch.randint(12, 21, (1,)))
    for _ in range(num_satellites):
        sat_r = float(0.9 + 2.0 * torch.rand(1))
        sat_theta = float(2 * np.pi * torch.rand(1))
        sat_x = sat_r * np.cos(sat_theta)
        sat_y = sat_r * np.sin(sat_theta)
        sat_w = float(0.07 + 0.08 * torch.rand(1))
        sat_amp = float(0.05 + 0.2 * torch.rand(1))
        img += sat_amp * torch.exp(-((xx - sat_x) ** 2 + (yy - sat_y) ** 2) / sat_w ** 2)
    return img


def add_background_and_noise(img, size, xx, yy):
    """Add a sparse background, sky glow, and Gaussian read noise."""
    num_stars = int(torch.randint(300, 501, (1,)))
    for _ in range(num_stars):
        star_x = float(8.0 * torch.rand(1) - 4.0)
        star_y = float(8.0 * torch.rand(1) - 4.0)
        star_w = float(0.01 + 0.015 * torch.rand(1))
        star_amp = float(0.1 + 0.3 * torch.rand(1))
        img += star_amp * torch.exp(-((xx - star_x) ** 2 + (yy - star_y) ** 2) / star_w ** 2)

    bg_x = float(6.0 * torch.rand(1) - 3.0)
    bg_y = float(6.0 * torch.rand(1) - 3.0)
    bg_w = float(2.0 + 2.0 * torch.rand(1))
    bg_amp = float(0.05 + 0.1 * torch.rand(1))
    img += bg_amp * torch.exp(-((xx - bg_x) ** 2 + (yy - bg_y) ** 2) / bg_w ** 2)
    img = img + 0.05
    img = img + float(0.04 + 0.08 * torch.rand(1)) * torch.randn(size, size)
    return img