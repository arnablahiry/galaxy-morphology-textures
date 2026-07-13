"""Edge-on disk renderer."""

import numpy as np
import torch


def draw_edge_on(xx, yy):
    """Render an edge-on disk with a dust lane and star-forming knots."""
    pa = float(2 * np.pi * torch.rand(1))
    xr = xx * np.cos(pa) + yy * np.sin(pa)
    yr = -xx * np.sin(pa) + yy * np.cos(pa)
    h_R = float(0.38 + 0.14 * torch.rand(1))
    h_z = float(0.050 + 0.035 * torch.rand(1))
    warp = float(0.14 * (torch.rand(1) - 0.5))
    yr_w = yr - warp * xr ** 3
    disk_amp = float(1.8 + 0.8 * torch.rand(1))
    radial = torch.exp(-torch.abs(xr) / h_R) * torch.exp(-(xr / (2.4 * h_R)) ** 4)
    asym = 1.0 + float(0.4 * (torch.rand(1) - 0.5)) * torch.tanh(xr / h_R)
    hz_map = h_z * (1.0 + 0.5 * torch.abs(xr) / h_R)
    img = disk_amp * asym * radial / torch.cosh(yr_w / hz_map) ** 2
    img += 0.25 * disk_amp * radial / torch.cosh(yr_w / (2.8 * hz_map)) ** 2
    r_c = (xr ** 2 + yr ** 2).sqrt() + 1e-6
    w_b = float(0.09 + 0.06 * torch.rand(1))
    n_b = float(0.9 + 0.6 * torch.rand(1))
    img += float(2.0 + 1.0 * torch.rand(1)) * torch.exp(-(r_c / w_b) ** (1.0 / n_b))
    img += 1.0 * torch.exp(-(r_c / (0.35 * w_b)) ** 2)
    dust = float(0.35 + 0.25 * torch.rand(1))
    lane_off = h_z * float(0.5 + 1.0 * torch.rand(1)) * (1.0 if torch.rand(1).item() > 0.5 else -1.0)
    lane_w = h_z * float(0.8 + 0.8 * torch.rand(1))
    img = img * (1.0 - dust * torch.exp(-((yr_w - lane_off) / lane_w) ** 2) * torch.exp(-(xr / (1.6 * h_R)) ** 2))
    n_kn = int(torch.randint(5, 13, (1,)))
    for _ in range(n_kn):
        x_k = float(2 * torch.rand(1) - 1) * 1.6 * h_R
        y_k = warp * x_k ** 3 + float(torch.randn(1)) * h_z * 0.8
        sx = np.cos(pa) * x_k - np.sin(pa) * y_k
        sy = np.sin(pa) * x_k + np.cos(pa) * y_k
        lum_skew = float(torch.rand(1)) ** 3
        kw = float(0.010 + 0.018 * lum_skew)
        ka = float(0.15 + 0.8 * lum_skew) * float(np.exp(-np.abs(x_k) / h_R))
        img += ka * torch.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / kw ** 2)
    return img