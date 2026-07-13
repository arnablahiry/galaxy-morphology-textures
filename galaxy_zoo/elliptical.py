"""Elliptical galaxy renderer."""

import numpy as np
import torch


def draw_elliptical(xx, yy):
    """Render a smooth elliptical galaxy with a warped outer halo."""
    pa = float(2 * np.pi * torch.rand(1))
    q = float(0.55 + 0.35 * torch.rand(1))
    xr = xx * np.cos(pa) + yy * np.sin(pa)
    yr = (-xx * np.sin(pa) + yy * np.cos(pa)) / q
    r = (xr ** 2 + yr ** 2).sqrt() + 1e-6
    theta = torch.atan2(yr, xr)
    w = 0.12 + 0.06 * torch.rand(1)
    warp_phase1 = float(2 * np.pi * torch.rand(1))
    warp_phase2 = float(2 * np.pi * torch.rand(1))
    w_amp1 = float(0.04 + 0.06 * torch.rand(1))
    w_amp2 = float(0.02 + 0.05 * torch.rand(1))
    warp_factor = 1.0 + torch.tanh(r / 0.15) * (
        w_amp1 * torch.sin(theta + warp_phase1) + w_amp2 * torch.sin(2 * theta + warp_phase2)
    )
    r_warped = r * warp_factor
    brightness_asym = 1.0 + 0.25 * torch.sin(theta + warp_phase1)
    n_bulge = float(0.8 + 0.7 * torch.rand(1))
    n_halo = float(0.5 + 1.0 * torch.rand(1))
    bulge_amp = float(1.5 + 1.5 * torch.rand(1))
    return bulge_amp * torch.exp(-(r / w) ** (1.0 / n_bulge)) + \
        0.5 * brightness_asym * torch.exp(-(r_warped / 0.5) ** (1.0 / n_halo))