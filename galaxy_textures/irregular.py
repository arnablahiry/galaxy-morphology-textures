"""Irregular galaxy renderer."""

import numpy as np
import torch


def draw_irregular(xx, yy):
    """Render a clumpy irregular galaxy with optional tidal debris."""
    q_irr = float(0.45 + 0.35 * torch.rand(1))
    pa_irr = float(2 * np.pi * torch.rand(1))
    n_clumps = int(torch.randint(7, 13, (1,)))
    cxp = float(0.20 * (torch.rand(1) - 0.5))
    cyp = float(0.20 * (torch.rand(1) - 0.5))
    pts = []
    for _ in range(n_clumps):
        cw = float(0.045 + 0.075 * torch.rand(1))
        ca = float(0.35 + 0.50 * torch.rand(1))
        img = ca * torch.exp(-((xx - cxp) ** 2 + (yy - cyp) ** 2) / cw ** 2)
        img += float(0.35 + 0.20 * torch.rand(1)) * ca * torch.exp(-((xx - cxp) ** 2 + (yy - cyp) ** 2) / (3.2 * cw) ** 2)
        pts.append((cxp, cyp))
        stp = float(0.10 + 0.16 * torch.rand(1))
        ang = float(2 * np.pi * torch.rand(1))
        cxp += stp * (np.cos(ang) * abs(np.cos(ang - pa_irr)) + 0.4 * np.cos(ang))
        cyp += stp * (np.sin(ang) * abs(np.cos(ang - pa_irr)) + 0.4 * np.sin(ang))
        rr = (cxp ** 2 + cyp ** 2) ** 0.5
        if rr > 0.60:
            cxp *= 0.60 / rr
            cyp *= 0.60 / rr
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    ex = (xx - mx - float(0.10 * (torch.rand(1) - 0.5))) * np.cos(pa_irr) + (yy - my - float(0.10 * (torch.rand(1) - 0.5))) * np.sin(pa_irr)
    ey = (-(xx - mx) * np.sin(pa_irr) + (yy - my) * np.cos(pa_irr)) / q_irr
    img += float(0.18 + 0.12 * torch.rand(1)) * torch.exp(-(ex ** 2 + ey ** 2) / 0.40 ** 2)
    n_arms_irr = int(torch.rand(1).item() < 0.70) + int(torch.rand(1).item() < 0.30)
    r_s = ((xx - mx) ** 2 + (yy - my) ** 2).sqrt() + 1e-6
    th_s = torch.atan2(yy - my, xx - mx)
    spin_irr = 1.0 if torch.rand(1).item() > 0.5 else -1.0
    for arm_i in range(n_arms_irr):
        phase_s = float(2 * np.pi * torch.rand(1))
        arm_start_s = float(0.15 + 0.10 * torch.rand(1))
        pitch_s = float(1.2 + 1.8 * torch.rand(1))
        fade_s = float(0.50 + 0.30 * torch.rand(1))
        width_s = float(0.15 + 0.10 * torch.rand(1))
        fan_s = float(0.05 + 0.06 * torch.rand(1))
        wig_a = float(0.18 + 0.20 * torch.rand(1))
        wig_f = float(4.0 + 4.0 * torch.rand(1))
        wig_p = float(2 * np.pi * torch.rand(1))
        dim = 1.0 if arm_i == 0 else 0.55
        dr_s = (r_s - arm_start_s).clamp(min=0)
        wind_s = pitch_s * dr_s / (1.0 + 0.9 * dr_s)
        phi_s = th_s - spin_irr * wind_s - phase_s - wig_a * torch.sin(wig_f * dr_s + wig_p)
        width_r = width_s + fan_s * dr_s
        img += dim * float(0.48 + 0.20 * torch.rand(1)) * torch.exp(-dr_s / fade_s) * torch.exp((torch.cos(phi_s) - 1) / width_r) * torch.sigmoid((r_s - arm_start_s) * 20.0)
        n_stream = int(torch.randint(14, 30, (1,)))
        for _ in range(n_stream):
            t = float(torch.rand(1)) ** 0.7
            dr_pt = t * fade_s * 1.8
            wind_pt = pitch_s * dr_pt / (1.0 + 0.9 * dr_pt)
            th_pt = phase_s + spin_irr * wind_pt + wig_a * np.sin(wig_f * dr_pt + wig_p) + float(0.10 * (torch.rand(1) - 0.5))
            r_pt = arm_start_s + dr_pt
            kx = mx + r_pt * np.cos(th_pt)
            ky = my + r_pt * np.sin(th_pt)
            kw = float(0.008 + 0.012 * torch.rand(1))
            ka = dim * float(0.15 + 0.30 * torch.rand(1)) * (1.0 - 0.6 * t)
            img += ka * torch.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / kw ** 2)
    n_kn = int(torch.randint(15, 36, (1,)))
    for _ in range(n_kn):
        bx, by = pts[int(torch.randint(0, len(pts), (1,)))]
        kx = bx + float(torch.randn(1)) * 0.07
        ky = by + float(torch.randn(1)) * 0.07
        lum_skew = float(torch.rand(1)) ** 3
        kw = float(0.008 + 0.020 * lum_skew)
        ka = float(0.20 + 0.70 * lum_skew)
        img += ka * torch.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / kw ** 2)
    return img