"""Barred spiral galaxy renderer."""

import numpy as np
import torch


def draw_barred_spiral(xx, yy, spin_dir):
    """Render a barred spiral with two arms rooted at the bar tips."""
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
    n_bulge, n_disk, bulge_amp = 1.2, 0.5, 2.8
    img = bulge_amp * torch.exp(-(r / w) ** (1.0 / n_bulge)) + \
        0.4 * torch.exp(-(r_warped / 0.45) ** (1.0 / n_disk)) + \
        0.15 * brightness_asym * torch.exp(-(r_warped / 0.45) ** (1.0 / n_disk))
    bar_len = float(0.35 + 0.10 * torch.rand(1))
    bar_width = float(0.09 + 0.04 * torch.rand(1))
    bar_amp = float(2.4 + 0.6 * torch.rand(1))
    img += bar_amp * torch.exp(-(xr / bar_len) ** 4 - (yr / bar_width) ** 2)
    arm_start = bar_len
    base_pitch = float(5 + 4 * torch.rand(1))
    warp_blend = torch.sigmoid((r - arm_start) * 20.0)
    r_blend = r + warp_blend * (r_warped - r)
    dr = (r_blend - arm_start).clamp(min=0)
    arm_mask = torch.sigmoid((r_blend - arm_start) * 25.0)
    clean_radius = float(0.16 + 0.10 * torch.rand(1))
    n_splits = int(torch.multinomial(torch.tensor([0.35, 0.45, 0.20]), 1))
    split_arm_idx = torch.randperm(2)[:min(n_splits, 2)].tolist()
    sf_activity = float(torch.rand(1))
    for i in range(2):
        phase_i = i * np.pi + float(0.25 * (torch.rand(1) - 0.5))
        pitch_i = base_pitch * float(0.75 + 0.5 * torch.rand(1))
        fade_i = float(0.30 + 0.15 * torch.rand(1))
        width_i = float(0.30 + 0.20 * torch.rand(1))
        amp_i = float(0.8 + 0.6 * torch.rand(1))
        phi_i = theta - spin_dir * pitch_i * dr - phase_i
        arm_fade_i = torch.exp(-dr / fade_i)
        arm_width_i = width_i * torch.exp(-dr / (fade_i * 3)).clamp(min=0.08)
        img += (bulge_amp * 0.6) * amp_i * brightness_asym * arm_fade_i * torch.exp((torch.cos(phi_i) - 1) / arm_width_i) * arm_mask
        num_knots = int(torch.randint(0, 4, (1,))) + int(sf_activity * 18)
        for _ in range(num_knots):
            t = float(torch.rand(1))
            dr_pt = clean_radius * 0.6 + t * (fade_i * 2.2)
            r_pt = arm_start + dr_pt + float(0.02 * (torch.rand(1) - 0.5))
            theta_pt = phase_i + spin_dir * pitch_i * dr_pt + float(0.18 * (torch.rand(1) - 0.5))
            xr_pt = r_pt * np.cos(theta_pt)
            yr_pt = r_pt * np.sin(theta_pt)
            x_pt = np.cos(pa) * xr_pt - np.sin(pa) * q * yr_pt
            y_pt = np.sin(pa) * xr_pt + np.cos(pa) * q * yr_pt
            lum_skew = float(torch.rand(1)) ** 3
            knot_w = float(0.012 + 0.022 * lum_skew + 0.006 * torch.rand(1))
            knot_amp = (bulge_amp * 0.35) * float(0.25 + 1.2 * lum_skew) * float(np.exp(-dr_pt / fade_i))
            img += knot_amp * torch.exp(-((xx - x_pt) ** 2 + (yy - y_pt) ** 2) / knot_w ** 2)
        if i in split_arm_idx:
            split_dr = clean_radius + fade_i * float(0.5 + 0.4 * torch.rand(1))
            branch_side = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            branch_rate = float(2.0 + 2.0 * torch.rand(1))
            branch_width = width_i * float(0.8 + 0.3 * torch.rand(1))
            branch_amp = amp_i * float(0.5 + 0.3 * torch.rand(1))
            dr_from_split = (dr - split_dr).clamp(min=0)
            split_mask = torch.sigmoid((dr - split_dr) * 20.0) * arm_mask
            branch_fade = torch.exp(-dr_from_split / fade_i)
            phi_branch = phi_i - branch_side * branch_rate * dr_from_split
            branch_width_r = branch_width * torch.exp(-dr_from_split / (fade_i * 3)).clamp(min=0.08)
            img += (bulge_amp * 0.6) * branch_amp * brightness_asym * branch_fade * torch.exp((torch.cos(phi_branch) - 1) / branch_width_r) * split_mask
    return img