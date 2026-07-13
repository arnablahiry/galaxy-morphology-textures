"""Merger galaxy renderer."""

import numpy as np
import torch


def _interacting_disk(xx, yy, cx, cy, toward, scale, barred, spin, stage, lum=1.0, boost=1.0):
    """Render one disk in a tidally interacting pair."""
    x = (xx - cx) / scale
    y = (yy - cy) / scale
    r = (x ** 2 + y ** 2).sqrt() + 1e-6
    th = torch.atan2(y, x)
    img = torch.zeros_like(xx)
    s = min(stage * boost, 1.3)
    tid = s * float(0.18 + 0.18 * torch.rand(1))
    r_t = r * (1.0 - tid * torch.tanh(r / 0.2) * torch.cos(2 * (th - toward)))
    w = float(0.09 + 0.04 * torch.rand(1))
    img += float(1.3 + 0.6 * torch.rand(1)) * torch.exp(-(r / (w * 0.35)) ** 2)
    img += float(1.6 + 0.8 * torch.rand(1)) * torch.exp(-(r / w) ** (1.0 / 1.1))
    img += 0.30 * torch.exp(-(r_t / 0.35) ** (1.0 / 0.9))
    n_halo = float(0.6 + 0.6 * torch.rand(1))
    img += float(0.07 + 0.07 * torch.rand(1)) * torch.exp(-(r_t / 0.55) ** (1.0 / n_halo))
    if barred:
        bar_angle = toward + float(0.5 * (torch.rand(1) - 0.5))
        bar_len = float(0.26 + 0.10 * torch.rand(1))
        bar_w = float(0.07 + 0.03 * torch.rand(1))
        xb = x * np.cos(bar_angle) + y * np.sin(bar_angle)
        yb = -x * np.sin(bar_angle) + y * np.cos(bar_angle)
        img += float(1.1 + 0.4 * torch.rand(1)) * torch.exp(-(xb / bar_len) ** 4 - (yb / bar_w) ** 2)
        arm_start = bar_len
        phase_bridge = bar_angle
        phase_tail = bar_angle + np.pi
    else:
        arm_start = w
        phase_bridge = toward + float(0.35 * (torch.rand(1) - 0.5))
        phase_tail = toward + np.pi + float(0.35 * (torch.rand(1) - 0.5))
    dr = (r_t - arm_start).clamp(min=0)
    arm_mask = torch.sigmoid((r_t - arm_start) * 25.0)
    arms = [
        (phase_bridge, float(5.0 + 3.0 * torch.rand(1)), float(0.22 + 0.10 * torch.rand(1)) + 0.18 * s,
         float(0.20 + 0.10 * torch.rand(1)), float(1.10 + 0.45 * torch.rand(1)) * (1.0 + 0.3 * s), 0.9, 0.03),
        (phase_tail, float(4.5 + 3.0 * torch.rand(1)), float(0.35 + 0.15 * torch.rand(1)) + 0.90 * s,
         float(0.20 + 0.10 * torch.rand(1)), float(1.00 + 0.40 * torch.rand(1)) * (0.7 + 0.9 * s), 0.9, float(0.03 + 0.07 * s)),
    ]
    for (phase, pitch, fade, width0, amp, unwind, fan) in arms:
        wind = pitch * dr / (1.0 + unwind * dr)
        phi = th - spin * wind - phase
        fade_map = torch.exp(-dr / fade)
        width_r = width0 + fan * dr
        img += amp * fade_map * torch.exp((torch.cos(phi) - 1) / width_r) * arm_mask
        num_knots = int(torch.randint(4, 10, (1,))) + int(6 * s)
        for _ in range(num_knots):
            t = float(torch.rand(1)) ** 0.8
            dr_pt = 0.05 + t * fade * 2.0
            wind_pt = pitch * dr_pt / (1.0 + unwind * dr_pt)
            th_pt = phase + spin * wind_pt + float(0.10 * (torch.rand(1) - 0.5))
            r_pt = arm_start + dr_pt
            x_pt = r_pt * np.cos(th_pt)
            y_pt = r_pt * np.sin(th_pt)
            lum_skew = float(torch.rand(1)) ** 3
            knot_w = float(0.015 + 0.030 * lum_skew + 0.006 * torch.rand(1))
            knot_amp = float(0.20 + 1.0 * lum_skew) * float(np.exp(-dr_pt / fade))
            img += knot_amp * torch.exp(-((x - x_pt) ** 2 + (y - y_pt) ** 2) / knot_w ** 2)
    return img * lum


def draw_merger(xx, yy, spin_dir, stage=None, merger_barred=None):
    """Render a two-body interacting merger with a connecting tidal bridge."""
    del spin_dir
    barred_b = (torch.rand(1).item() > 0.5) if merger_barred is None else bool(merger_barred)
    st = float(torch.rand(1)) if stage is None else float(stage)
    sep = float(0.62 - 0.32 * st)
    angle = float(2 * np.pi * torch.rand(1))
    dx, dy = sep * np.cos(angle), sep * np.sin(angle)
    spin_a = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    spin_b = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    mass_ratio = float(0.30 + 0.70 * torch.rand(1))
    scale_a = float(0.48 + 0.10 * torch.rand(1))
    scale_b = scale_a * mass_ratio ** 0.6
    lum_b = mass_ratio ** 0.8
    boost_a = float(0.55 + 0.45 * mass_ratio)
    boost_b = 1.0 / float(0.45 + 0.55 * mass_ratio)
    img = _interacting_disk(xx, yy, dx, dy, angle + np.pi, scale_a, False, spin_a, st, boost=boost_a)
    img = img + _interacting_disk(xx, yy, -dx, -dy, angle, scale_b, barred_b, spin_b, st, lum=lum_b, boost=boost_b)
    ux, uy = np.cos(angle), np.sin(angle)
    pxa, pya = -uy, ux
    along = xx * ux + yy * uy
    perp = xx * pxa + yy * pya
    bend = float(0.05 + 0.09 * torch.rand(1)) * torch.sin(along * np.pi / sep) * spin_a
    bw = float(0.06 + 0.05 * torch.rand(1))
    img = img + (0.12 + 0.45 * st) * torch.exp(-((perp - bend) / bw) ** 2) * torch.exp(-(along / (sep * 1.1)) ** 2)
    n_bridge_knots = int(2 + 5 * st)
    for _ in range(n_bridge_knots):
        t = float(torch.rand(1)) * 2 - 1.0
        a_pt = t * sep * 0.8
        b_pt = float(0.05 + 0.09 * torch.rand(1)) * np.sin(a_pt * np.pi / sep) * spin_a + float(0.03 * (torch.rand(1) - 0.5))
        x_pt = a_pt * ux + b_pt * pxa
        y_pt = a_pt * uy + b_pt * pya
        knot_w = float(0.010 + 0.018 * torch.rand(1))
        knot_amp = float(0.08 + 0.25 * torch.rand(1)) * st
        img += knot_amp * torch.exp(-((xx - x_pt) ** 2 + (yy - y_pt) ** 2) / knot_w ** 2)
    return img