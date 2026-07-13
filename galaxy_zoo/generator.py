"""Procedural synthetic galaxy image generator.

Every image is built from explicit, hand-tuned analytic profiles (Sersic
bulges/halos, exponential disks, logarithmic spiral arms, sech^2 edge-on
disks, tidally-distorted merger components, random-walk irregulars) rather
than any simulation or real imaging data. Nothing here is calibrated against
real photometry -- it is a stylized approximation intended for teaching /
interpretability work (e.g. training a small classifier and inspecting what
it attends to), not for astrophysical research.

Six classes are produced (see CLASSES below):
    0 = elliptical
    1 = spiral
    2 = barred spiral
    3 = merger (spiral-spiral or spiral-barred, continuous interaction stage)
    4 = edge-on disk with bulge (Sombrero-like, with a dust lane)
    5 = disturbed irregular (clumpy Magellanic-type, starbursting)
"""

import numpy as np
import torch


def _interacting_disk(xx, yy, cx, cy, toward, scale, barred, spin, stage, lum=1.0,
                      boost=1.0):
    """One disk galaxy of an interacting pair, centred at (cx, cy).

    toward : sky angle (radians) pointing at the companion.
    scale  : physical size of this galaxy (local coords are (sky - centre)/scale).
    barred : if True, draw a stellar bar and root the two arms at its tips.
    spin   : +1 / -1 winding direction of the arms.
    stage  : 0..1 interaction stage. Early (0) = mostly undisturbed spiral with a
             thin bridge arm; advanced (1) = strongly stretched bridge and a long,
             fanned-out tidal tail on the far side.
    lum    : overall brightness scaling (for unequal-mass companions).
    boost  : tidal susceptibility. Tides are differential, so the LESS massive
             member of a pair is far more disrupted than the primary -- pass
             boost > 1 for the small companion, < 1 for the primary, and all
             stage-driven distortions (elongation, tail length/brightness,
             triggered star formation) scale accordingly.

    The physics being cartooned: tidal forces are differential, so material on
    the NEAR side of each disk is pulled toward the companion (-> the bridge)
    while material on the FAR side lags behind and is flung outward (-> the
    tail). Hence one arm is aligned with `toward` and kept compact, and the
    opposite arm is given a much longer fade, a widening cross-section, and an
    unwinding pitch so it straightens into the classic long curved tail.
    """
    x = (xx - cx) / scale
    y = (yy - cy) / scale
    r = (x ** 2 + y ** 2).sqrt() + 1e-6
    th = torch.atan2(y, x)

    img = torch.zeros_like(xx)

    # Effective tidal forcing felt by THIS galaxy: encounter stage times its
    # susceptibility (small companions get shredded, primaries less so).
    s = min(stage * boost, 1.3)

    # m=2 tidal elongation of the whole disk along the companion axis, growing
    # outward (tanh keeps the nucleus round) and with interaction stage.
    tid = s * float(0.18 + 0.18 * torch.rand(1))
    r_t = r * (1.0 - tid * torch.tanh(r / 0.2) * torch.cos(2 * (th - toward)))

    w = float(0.09 + 0.04 * torch.rand(1))

    # Bright compact nucleus + bulge: the two nuclei must stay clearly the
    # brightest points in the frame so the pair reads as two distinct galaxies,
    # never one smeared band.
    img += float(1.3 + 0.6 * torch.rand(1)) * torch.exp(-(r / (w * 0.35)) ** 2)
    img += float(1.6 + 0.8 * torch.rand(1)) * torch.exp(-(r / w) ** (1.0 / 1.1))
    # Disturbed disk + soft extended envelope (tidally puffed-up material).
    # Kept deliberately faint: if these glow terms are too strong the two
    # galaxies wash into one bright haze and the arm/tail structure vanishes.
    img += 0.30 * torch.exp(-(r_t / 0.35) ** (1.0 / 0.9))
    n_halo = float(0.6 + 0.6 * torch.rand(1))
    img += float(0.07 + 0.07 * torch.rand(1)) * torch.exp(-(r_t / 0.55) ** (1.0 / n_halo))

    if barred:
        # Bar jittered around the companion axis so one tip roughly faces the
        # companion; the bridge/tail arms then leave from the two bar tips,
        # exactly like the isolated barred spiral.
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

    # (phase, pitch, fade, width0, amp, unwind, fan)
    #   unwind: how quickly the logarithmic winding saturates -> the arm
    #           straightens outward instead of wrapping into a ring.
    #   fan:    growth of the arm's cross-section with radius (tails fan out).
    arms = [
        # Bridge arm -- reaches toward the companion; stretches with the tides.
        (phase_bridge,
         float(5.0 + 3.0 * torch.rand(1)),
         float(0.22 + 0.10 * torch.rand(1)) + 0.18 * s,
         float(0.20 + 0.10 * torch.rand(1)),
         float(1.10 + 0.45 * torch.rand(1)) * (1.0 + 0.3 * s),
         0.9,
         0.03),
        # Tidal tail -- far side; long, thin, unwinding, amplified by the tides.
        # Winds like a normal arm near the disk (higher pitch, gentler unwind)
        # then straightens into the long smooth arc, Mice-style.
        (phase_tail,
         float(4.5 + 3.0 * torch.rand(1)),
         float(0.35 + 0.15 * torch.rand(1)) + 0.90 * s,
         float(0.20 + 0.10 * torch.rand(1)),
         float(1.00 + 0.40 * torch.rand(1)) * (0.7 + 0.9 * s),
         0.9,
         float(0.03 + 0.07 * s)),
    ]
    for (phase, pitch, fade, width0, amp, unwind, fan) in arms:
        wind = pitch * dr / (1.0 + unwind * dr)
        phi = th - spin * wind - phase
        fade_map = torch.exp(-dr / fade)
        width_r = width0 + fan * dr
        img += amp * fade_map * torch.exp((torch.cos(phi) - 1) / width_r) * arm_mask

        # HII-region knots strung along the arm's ridge line (interactions
        # trigger star formation, so knots are welcome here). Same skewed
        # luminosity function as the isolated spirals: mostly faint, the
        # occasional dominant complex.
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


def make_galaxy(kind, size=128, stage=None, merger_barred=None):
    """Toy galaxy image with explicit Sersic profiles.
    kind 0 = elliptical
    kind 1 = spiral
    kind 2 = barred spiral
    kind 3 = merger (spiral-spiral or spiral-barred, staged tidal interaction)
    kind 4 = edge-on disk with bulge (Sombrero-like, with a dust lane)
    kind 5 = disturbed irregular (clumpy Magellanic-type, starbursting)
    size : Controls pixel resolution
    stage / merger_barred : optional overrides for kind 3 (stage in [0,1];
        merger_barred True -> spiral-barred pair). Randomized when None.
    """
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size),
                            torch.linspace(-1, 1, size), indexing="ij")

    # --- ZOOM & OFFSET (PANNING) ---
    if kind == 3:
        # Mergers need both galaxies AND their tails in frame -> gentler crop.
        zoom = float(0.85 + 0.25 * torch.rand(1))
        offset_x = float(0.15 * (torch.rand(1) - 0.5) * 2)
        offset_y = float(0.15 * (torch.rand(1) - 0.5) * 2)
    elif kind == 4:
        # Edge-on disks are long and thin -- keep the full extent in frame.
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

    if kind in [0, 1, 2]:
        # --- ISOLATED GALAXIES ---
        pa = float(2 * np.pi * torch.rand(1))
        q = float(0.55 + 0.35 * torch.rand(1))
        xr = xx * np.cos(pa) + yy * np.sin(pa)
        yr = (-xx * np.sin(pa) + yy * np.cos(pa)) / q
        r = (xr ** 2 + yr ** 2).sqrt() + 1e-6
        theta = torch.atan2(yr, xr)
        w = 0.12 + 0.06 * torch.rand(1)
        # --- INTRINSIC ASYMMETRY / TIDAL WARPS ---
        # Generate random phases for the asymmetric distortions
        warp_phase1 = float(2 * np.pi * torch.rand(1))
        warp_phase2 = float(2 * np.pi * torch.rand(1))
        # Amplitude of distortion: m=1 (lopsidedness), m=2 (oval distortion)
        # We use tanh so the bright dense core remains stable while the outer disk warps
        w_amp1 = float(0.04 + 0.06 * torch.rand(1))
        w_amp2 = float(0.02 + 0.05 * torch.rand(1))
        warp_factor = 1.0 + torch.tanh(r / 0.15) * (w_amp1 * torch.sin(theta + warp_phase1) + w_amp2 * torch.sin(2 * theta + warp_phase2))
        r_warped = r * warp_factor
        # Brightness asymmetry so one side/arm is naturally a bit brighter (like real galaxies)
        brightness_asym = 1.0 + 0.25 * torch.sin(theta + warp_phase1)

        if kind == 0:
            n_bulge = float(0.8 + 0.7 * torch.rand(1))
            n_halo = float(0.5 + 1.0 * torch.rand(1))
            bulge_amp = float(1.5 + 1.5 * torch.rand(1))
            # The core remains symmetric (r), but the extended halo is warped (r_warped)
            img = bulge_amp * torch.exp(-(r / w) ** (1.0 / n_bulge)) + \
                  0.5 * brightness_asym * torch.exp(-(r_warped / 0.5) ** (1.0 / n_halo))

        elif kind == 1:
            n_bulge, n_disk, bulge_amp = 1, 1.0, 3.8
            # Soft diffuse glow around the core -- amplitude/radius bumped up (and
            # no longer scaled thin by brightness_asym) so it shows up reliably in
            # every spiral instead of only being visible in some crops.
            img = bulge_amp * torch.exp(-(r / w) ** (1.0 / n_bulge)) + \
                  0.4 * torch.exp(-(r_warped / 0.45) ** (1.0 / n_disk)) + \
                  0.15 * brightness_asym * torch.exp(-(r_warped / 0.45) ** (1.0 / n_disk))
            # --- SPIRAL ARMS THAT BIFURCATE, RATHER THAN N COPIES FROM THE CENTRE ---
            # Real disks have at most 2-3 arms actually rooted at the bulge; what
            # looks like "4+ arms" in nature is usually 2-3 primary arms that each
            # fork into two branches partway out (a well-known spiral structure).
            # So: at most 3 arms leave the centre, each built as its own randomized
            # curve, and each may bifurcate into two fainter branches at some radius.
            n_arms = int(torch.multinomial(torch.tensor([0.6, 0.4]), 1)) + 2  # always 2 or 3, from the centre
            base_phase = float(2 * np.pi * torch.rand(1))
            base_pitch = float(4 + 4 * torch.rand(1))
            arm_start = w
            dr = (r_warped - arm_start).clamp(min=0)
            arm_mask = torch.sigmoid((r_warped - arm_start) * 30.0)
            # Radius (in dr) out to which each arm is a perfectly clean, smooth
            # curve before any forking.
            clean_radius = float(0.16 + 0.10 * torch.rand(1))
            # Total number of forks across the WHOLE galaxy is capped at 0-2 (not
            # an independent coin flip per arm), so splitting stays rare and
            # deliberate rather than turning every arm into a fork.
            n_splits = int(torch.multinomial(torch.tensor([0.35, 0.45, 0.20]), 1))
            split_arm_idx = torch.randperm(n_arms)[:min(n_splits, n_arms)].tolist()
            # Star-formation activity: one draw per galaxy (not per arm), so
            # the whole disk is consistently quiescent or actively star-
            # forming, rather than every spiral getting the same knot count.
            # Real spirals range from almost no visible HII knots (early-type,
            # gas-poor) to dozens (starbursting "fireworks" galaxies).
            sf_activity = float(torch.rand(1))
            for i in range(n_arms):
                # Arms are roughly evenly spaced, but jittered so spacing isn't exact
                phase_i = base_phase + i * (2 * np.pi / n_arms) + float(0.35 * (torch.rand(1) - 0.5))
                pitch_i = base_pitch * float(0.75 + 0.5 * torch.rand(1))
                fade_i = float(0.30 + 0.20 * torch.rand(1))       # how far the arm reaches
                # Narrow angular width so each arm reads as a distinct curve, not
                # a ring -- this coefficient divides (cos(phi)-1), so it must be
                # small (<1) to fall off sharply away from the arm's ridge line.
                width_i = float(0.30 + 0.20 * torch.rand(1))
                amp_i = float(0.9 + 0.7 * torch.rand(1))          # some arms fainter than others
                phi_i = theta - spin_dir * pitch_i * dr - phase_i
                arm_fade_i = torch.exp(-dr / fade_i)
                # Gentle, purely smooth taper (no texture/noise term at all) so the
                # arm is a single clean curve from the bulge all the way out.
                arm_width_i = width_i * torch.exp(-dr / (fade_i * 3)).clamp(min=0.08)
                img += amp_i * brightness_asym * arm_fade_i * \
                       torch.exp((torch.cos(phi_i) - 1) / arm_width_i) * arm_mask
                # HII-region-like knots: real spiral arms are speckled with
                # bright, resolved young-star clusters/star-forming clumps,
                # not a perfectly smooth analytic curve. Placed only beyond
                # clean_radius (arm base near the bulge stays smooth) and
                # scattered along the arm's ridge line with the same radial
                # fade as the arm itself, so they vanish where the arm does.
                # Count is driven by the galaxy's overall sf_activity so some
                # galaxies read as quiescent (few/no knots) and others as
                # actively star-forming (many), rather than a fixed count.
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
                    # Skewed toward faint: real HII regions follow a steep
                    # luminosity function -- most are small/faint, with one
                    # or two dominant giant complexes per arm, not a dozen
                    # equally-bright clumps. Cubing a uniform draw pushes most
                    # values low while still occasionally landing near 1.
                    lum_skew = float(torch.rand(1)) ** 3
                    knot_w = float(0.012 + 0.022 * lum_skew + 0.006 * torch.rand(1))
                    knot_amp = float(0.25 + 1.2 * lum_skew) * float(np.exp(-dr_pt / fade_i))
                    img += knot_amp * torch.exp(-((xx - x_pt) ** 2 + (yy - y_pt) ** 2) / knot_w ** 2)
                # Bifurcation: only the pre-selected arm(s) fork, and only once
                # well past clean_radius, so near the bulge it's always just the
                # plain primary arms with nothing forking.
                if i in split_arm_idx:
                    split_dr = clean_radius + fade_i * float(0.5 + 0.4 * torch.rand(1))
                    branch_side = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                    branch_rate = float(2.0 + 2.0 * torch.rand(1))         # gentle peel, not a sharp jump
                    branch_width = width_i * float(0.8 + 0.3 * torch.rand(1))
                    branch_amp = amp_i * float(0.55 + 0.3 * torch.rand(1))  # fork is noticeably fainter
                    dr_from_split = (dr - split_dr).clamp(min=0)
                    split_mask = torch.sigmoid((dr - split_dr) * 20.0) * arm_mask
                    branch_fade = torch.exp(-dr_from_split / fade_i)
                    phi_branch = phi_i - branch_side * branch_rate * dr_from_split
                    branch_width_r = branch_width * torch.exp(-dr_from_split / (fade_i * 3)).clamp(min=0.08)
                    img += branch_amp * brightness_asym * branch_fade * \
                           torch.exp((torch.cos(phi_branch) - 1) / branch_width_r) * split_mask

        elif kind == 2:
            n_bulge, n_disk, bulge_amp = 1.2, 0.5, 2.8
            # Same reliable diffuse core glow as the plain spiral: a symmetric
            # baseline term plus a smaller asymmetric one, instead of a single
            # term fully scaled by brightness_asym (which could nearly vanish).
            img = bulge_amp * torch.exp(-(r / w) ** (1.0 / n_bulge)) + \
                  0.4 * torch.exp(-(r_warped / 0.45) ** (1.0 / n_disk)) + \
                  0.15 * brightness_asym * torch.exp(-(r_warped / 0.45) ** (1.0 / n_disk))
            n_arms = 2
            bar_len = float(0.35 + 0.10 * torch.rand(1))
            bar_width = float(0.09 + 0.04 * torch.rand(1))
            bar_amp = float(2.4 + 0.6 * torch.rand(1))
            # Higher power (quartic, not quadratic) along the bar's long axis
            # gives it a flatter, more solid-looking core with a sharp cutoff at
            # the tips -- reads as a proper bar, not a soft blob -- and that
            # sharp cutoff is also what makes bar_len a precise, well-defined
            # "edge" for the arms to latch onto.
            img += bar_amp * torch.exp(-(xr / bar_len) ** 4 - (yr / bar_width) ** 2)
            # Two arms trailing off the ends of the bar. To make them start
            # exactly at the bar's tip (not before, not with a gap after), the
            # start/transition radius is measured in the SAME unwarped frame the
            # bar itself is drawn in (r, not r_warped) -- otherwise the tidal
            # warp could shift the transition off from the bar's actual edge.
            # The warp is blended back in smoothly only after the arm has
            # cleared the bar, so tidal lopsidedness still shows up further out.
            arm_start = bar_len
            base_pitch = float(5 + 4 * torch.rand(1))
            warp_blend = torch.sigmoid((r - arm_start) * 20.0)
            r_blend = r + warp_blend * (r_warped - r)
            dr = (r_blend - arm_start).clamp(min=0)
            arm_mask = torch.sigmoid((r_blend - arm_start) * 25.0)
            clean_radius = float(0.16 + 0.10 * torch.rand(1))
            n_splits = int(torch.multinomial(torch.tensor([0.35, 0.45, 0.20]), 1))
            split_arm_idx = torch.randperm(n_arms)[:min(n_splits, n_arms)].tolist()
            sf_activity = float(torch.rand(1))
            for i in range(n_arms):
                # Anchored to the two ends of the bar (0 and pi), with a little
                # jitter so the two sides aren't perfectly identical.
                phase_i = i * np.pi + float(0.25 * (torch.rand(1) - 0.5))
                pitch_i = base_pitch * float(0.75 + 0.5 * torch.rand(1))
                fade_i = float(0.30 + 0.15 * torch.rand(1))
                width_i = float(0.30 + 0.20 * torch.rand(1))     # narrow -> a real arm, not a ring
                amp_i = float(0.8 + 0.6 * torch.rand(1))         # the two arms needn't match in brightness
                phi_i = theta - spin_dir * pitch_i * dr - phase_i
                arm_fade_i = torch.exp(-dr / fade_i)
                arm_width_i = width_i * torch.exp(-dr / (fade_i * 3)).clamp(min=0.08)
                img += (bulge_amp * 0.6) * amp_i * brightness_asym * arm_fade_i * \
                       torch.exp((torch.cos(phi_i) - 1) / arm_width_i) * arm_mask
                # Same HII-knot speckling as the plain spiral arms, anchored
                # to this arm's own ridge line (which starts at the bar's
                # tip here, not the bulge), with the same per-galaxy activity
                # level and skewed-brightness luminosity function.
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
                    img += (bulge_amp * 0.6) * branch_amp * brightness_asym * branch_fade * \
                           torch.exp((torch.cos(phi_branch) - 1) / branch_width_r) * split_mask

    elif kind == 3:
        # --- MERGER SYSTEMS: two REAL disk galaxies tidally interacting ---
        # Instead of generic blobs with decorations bolted on, each component
        # is built by the same recipe as the isolated spirals (nucleus, bulge,
        # arms rooted at the bulge or bar tips, HII knots) and the tidal
        # response is encoded directly in the arms: the near-side arm of each
        # galaxy stretches toward its companion (the bridge), the far-side arm
        # is flung outward as a long unwinding tail. A continuous `stage`
        # parameter moves the system from a wide early passage (two obviously
        # separate spirals, thin connecting bridge, short tails) to an
        # advanced encounter (close nuclei, strong bridge, dramatic tails).
        barred_b = (torch.rand(1).item() > 0.5) if merger_barred is None else bool(merger_barred)
        st = float(torch.rand(1)) if stage is None else float(stage)
        # Separation shrinks as the interaction advances.
        sep = float(0.62 - 0.32 * st)
        angle = float(2 * np.pi * torch.rand(1))
        dx, dy = sep * np.cos(angle), sep * np.sin(angle)
        spin_a = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        spin_b = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        # Mass asymmetry: A is the primary; B ranges from a near-equal twin
        # (mass_ratio ~ 1, e.g. the Mice) down to a clearly smaller minor-
        # merger companion (~1/3 the mass). Size scales ~ M^0.6 and
        # luminosity ~ M^0.8, so a low-mass companion is visibly BOTH smaller
        # and fainter, never a same-size twin.
        mass_ratio = float(0.30 + 0.70 * torch.rand(1))
        scale_a = float(0.48 + 0.10 * torch.rand(1))
        scale_b = scale_a * mass_ratio ** 0.6
        lum_b = mass_ratio ** 0.8
        # Differential tides: the small companion is disrupted much more
        # strongly than the primary (boost > 1), while a massive primary
        # facing a puny companion barely responds (boost < 1).
        boost_a = float(0.55 + 0.45 * mass_ratio)
        boost_b = 1.0 / float(0.45 + 0.55 * mass_ratio)
        # A sits at (dx,dy), so its companion lies along angle+pi; vice versa for B.
        img = img + _interacting_disk(xx, yy, dx, dy, angle + np.pi,
                                      scale_a, False, spin_a, st, boost=boost_a)
        img = img + _interacting_disk(xx, yy, -dx, -dy, angle,
                                      scale_b, barred_b, spin_b, st, lum=lum_b,
                                      boost=boost_b)
        # --- TIDAL BRIDGE ---
        # A gently S-curved low-surface-brightness band along the axis between
        # the two nuclei, strengthening as the encounter advances. Together
        # with the two bridge arms pointing at each other, this forms the
        # continuous stream of material connecting the pair.
        ux, uy = np.cos(angle), np.sin(angle)
        pxa, pya = -uy, ux
        along = xx * ux + yy * uy
        perp = xx * pxa + yy * pya
        bend = float(0.05 + 0.09 * torch.rand(1)) * torch.sin(along * np.pi / sep) * spin_a
        bw = float(0.06 + 0.05 * torch.rand(1))
        img = img + (0.12 + 0.45 * st) * torch.exp(-((perp - bend) / bw) ** 2) * \
              torch.exp(-(along / (sep * 1.1)) ** 2)
        # A few knots strewn along the bridge (tidal dwarf clumps / triggered
        # star formation in the stream).
        n_bridge_knots = int(2 + 5 * st)
        for _ in range(n_bridge_knots):
            t = float(torch.rand(1)) * 2 - 1.0          # -1..1 along the bridge
            a_pt = t * sep * 0.8
            b_pt = float(0.05 + 0.09 * torch.rand(1)) * np.sin(a_pt * np.pi / sep) * spin_a \
                   + float(0.03 * (torch.rand(1) - 0.5))
            x_pt = a_pt * ux + b_pt * pxa
            y_pt = a_pt * uy + b_pt * pya
            knot_w = float(0.010 + 0.018 * torch.rand(1))
            knot_amp = float(0.08 + 0.25 * torch.rand(1)) * st
            img += knot_amp * torch.exp(-((xx - x_pt) ** 2 + (yy - y_pt) ** 2) / knot_w ** 2)

    elif kind == 4:
        # --- EDGE-ON DISK WITH BULGE ---
        # Seen edge-on, a disk is exponential along its major axis and
        # sech^2 in height -- a thin streak (h_z << h_R) -- plus a round
        # Sersic bulge poking above the plane, crossed by a dark dust lane
        # (the classic Sombrero silhouette).
        pa = float(2 * np.pi * torch.rand(1))
        xr = xx * np.cos(pa) + yy * np.sin(pa)      # along the disk plane
        yr = -xx * np.sin(pa) + yy * np.cos(pa)     # height above the plane
        h_R = float(0.38 + 0.14 * torch.rand(1))    # radial scale length
        h_z = float(0.050 + 0.035 * torch.rand(1))  # scale height (thin, but
        # resolvable -- a few percent of the disk length, like real edge-ons
        # such as NGC 891, not a hairline)
        # Integral-sign warp: real edge-on disks are rarely perfectly flat;
        # the outer disk bends up on one side, down on the other (~ xr^3).
        # Kept SUBTLE -- a strong coefficient bends the streak into a
        # noodle, and it stops reading as a disk at all.
        warp = float(0.14 * (torch.rand(1) - 0.5))
        yr_w = yr - warp * xr ** 3
        disk_amp = float(1.8 + 0.8 * torch.rand(1))
        # Exponential along the plane with an outer truncation (real stellar
        # disks cut off at ~3-4 scale lengths, they don't fade forever).
        radial = torch.exp(-torch.abs(xr) / h_R) * torch.exp(-(xr / (2.4 * h_R)) ** 4)
        # One side of the disk slightly brighter (lopsidedness).
        asym = 1.0 + float(0.4 * (torch.rand(1) - 0.5)) * torch.tanh(xr / h_R)
        # Flaring: real disks get vertically thicker toward their outskirts.
        hz_map = h_z * (1.0 + 0.5 * torch.abs(xr) / h_R)
        img = disk_amp * asym * radial / torch.cosh(yr_w / hz_map) ** 2
        # Thicker, fainter old-star disk enveloping the thin one.
        img += 0.25 * disk_amp * radial / torch.cosh(yr_w / (2.8 * hz_map)) ** 2
        r_c = (xr ** 2 + yr ** 2).sqrt() + 1e-6
        w_b = float(0.09 + 0.06 * torch.rand(1))
        n_b = float(0.9 + 0.6 * torch.rand(1))
        img += float(2.0 + 1.0 * torch.rand(1)) * torch.exp(-(r_c / w_b) ** (1.0 / n_b))
        img += 1.0 * torch.exp(-(r_c / (0.35 * w_b)) ** 2)   # sharp nucleus
        dust = float(0.35 + 0.25 * torch.rand(1))            # dust lane depth
        # Dust lane: a dark absorption band lying just off the midplane
        # (offset because we never view a disk at exactly 90 degrees).
        lane_off = h_z * float(0.5 + 1.0 * torch.rand(1)) * \
                   (1.0 if torch.rand(1).item() > 0.5 else -1.0)
        lane_w = h_z * float(0.8 + 0.8 * torch.rand(1))
        img = img * (1.0 - dust * torch.exp(-((yr_w - lane_off) / lane_w) ** 2) *
                     torch.exp(-(xr / (1.6 * h_R)) ** 2))
        # Star-forming knots strung along the (warped) plane.
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

    elif kind == 5:
        # --- DISTURBED IRREGULAR ---
        # No symmetric profile at all: a connected chain of clumps laid down
        # by a random walk (so the body is lumpy but contiguous, not a
        # scatter of unrelated blobs), wrapped in a lopsided diffuse
        # envelope, heavily speckled with star-forming knots (irregulars
        # are starbursty), plus a faint one-sided plume -- the "disturbed"
        # signature, without a visible companion in frame.
        # Many SMALL, hard-edged clumps on a long meandering walk -- big soft
        # clumps blur together into a smooth elliptical-looking blob, which
        # is exactly what an irregular must NOT look like. Anisotropic
        # stretching elongates the whole body.
        q_irr = float(0.45 + 0.35 * torch.rand(1))
        pa_irr = float(2 * np.pi * torch.rand(1))
        n_clumps = int(torch.randint(7, 13, (1,)))
        cxp = float(0.20 * (torch.rand(1) - 0.5))
        cyp = float(0.20 * (torch.rand(1) - 0.5))
        pts = []
        for _ in range(n_clumps):
            # Amplitudes kept LOW: clumps + halos + knots all overlap and
            # add up, and once the interior sum passes the display range it
            # clips to a featureless white blob -- the clumpy texture only
            # survives if the total stays comfortably below saturation.
            cw = float(0.045 + 0.075 * torch.rand(1))
            ca = float(0.35 + 0.50 * torch.rand(1))
            img += ca * torch.exp(-((xx - cxp) ** 2 + (yy - cyp) ** 2) / cw ** 2)
            # Diffuse emission enveloping each clump: a broad halo (~3x the
            # clump size). Because it's attached per-clump, the diffuse
            # light follows the body's irregular outline instead of forming
            # a smooth ellipse around everything. Amplitude must be a
            # sizeable fraction of the clump's own -- the display range is
            # set by the bright clumps/stars, so anything much fainter than
            # ~15% of peak reads as pure black.
            img += float(0.35 + 0.20 * torch.rand(1)) * ca * \
                   torch.exp(-((xx - cxp) ** 2 + (yy - cyp) ** 2) / (3.2 * cw) ** 2)
            pts.append((cxp, cyp))
            stp = float(0.10 + 0.16 * torch.rand(1))
            ang = float(2 * np.pi * torch.rand(1))
            # Bias the walk along the body's long axis -> elongated chain.
            cxp += stp * (np.cos(ang) * abs(np.cos(ang - pa_irr)) + 0.4 * np.cos(ang))
            cyp += stp * (np.sin(ang) * abs(np.cos(ang - pa_irr)) + 0.4 * np.sin(ang))
            rr = (cxp ** 2 + cyp ** 2) ** 0.5
            if rr > 0.60:                      # keep the body in frame
                cxp *= 0.60 / rr
                cyp *= 0.60 / rr
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        # Weak, ELONGATED, off-centre envelope -- faint enough that the
        # clumpy skeleton stays the dominant feature.
        ex = (xx - mx - float(0.10 * (torch.rand(1) - 0.5))) * np.cos(pa_irr) + \
             (yy - my - float(0.10 * (torch.rand(1) - 0.5))) * np.sin(pa_irr)
        ey = (-(xx - mx) * np.sin(pa_irr) + (yy - my) * np.cos(pa_irr)) / q_irr
        img += float(0.18 + 0.12 * torch.rand(1)) * \
               torch.exp(-(ex ** 2 + ey ** 2) / 0.40 ** 2)
        # --- OPTIONAL CURVED TIDAL FEATURES ---
        # Only some irregulars are tidally disturbed: ~70% get one feature,
        # a subset a second fainter one. Each is a curved, DISTORTED debris
        # arc -- diffuse cross-section that fans and fades like the merger
        # tails, but with a deliberately LOW pitch so it bends gently
        # (a banana-shaped plume) rather than winding around the body like
        # a spiral arm, plus a strong meander for the distorted look.
        n_arms_irr = int(torch.rand(1).item() < 0.70) + int(torch.rand(1).item() < 0.30)
        r_s = ((xx - mx) ** 2 + (yy - my) ** 2).sqrt() + 1e-6
        th_s = torch.atan2(yy - my, xx - mx)
        spin_irr = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        for arm_i in range(n_arms_irr):
            phase_s = float(2 * np.pi * torch.rand(1))
            arm_start_s = float(0.15 + 0.10 * torch.rand(1))
            pitch_s = float(1.2 + 1.8 * torch.rand(1))       # LOW: gentle bend, no spiral wrap
            fade_s = float(0.50 + 0.30 * torch.rand(1))
            width_s = float(0.15 + 0.10 * torch.rand(1))
            fan_s = float(0.05 + 0.06 * torch.rand(1))
            wig_a = float(0.18 + 0.20 * torch.rand(1))       # strong meander -> distorted arc
            wig_f = float(4.0 + 4.0 * torch.rand(1))
            wig_p = float(2 * np.pi * torch.rand(1))
            dim = 1.0 if arm_i == 0 else 0.55                # second arm fainter
            dr_s = (r_s - arm_start_s).clamp(min=0)
            # Same bounded winding as the merger tails: curves tightly near
            # the body, straightens gently further out.
            wind_s = pitch_s * dr_s / (1.0 + 0.9 * dr_s)
            phi_s = th_s - spin_irr * wind_s - phase_s - wig_a * torch.sin(wig_f * dr_s + wig_p)
            width_r = width_s + fan_s * dr_s
            img += dim * float(0.48 + 0.20 * torch.rand(1)) * \
                   torch.exp(-dr_s / fade_s) * \
                   torch.exp((torch.cos(phi_s) - 1) / width_r) * \
                   torch.sigmoid((r_s - arm_start_s) * 20.0)
            # Knots strung along the arm's curved ridge line.
            n_stream = int(torch.randint(14, 30, (1,)))
            for _ in range(n_stream):
                t = float(torch.rand(1)) ** 0.7
                dr_pt = t * fade_s * 1.8
                wind_pt = pitch_s * dr_pt / (1.0 + 0.9 * dr_pt)
                th_pt = phase_s + spin_irr * wind_pt + \
                        wig_a * np.sin(wig_f * dr_pt + wig_p) + \
                        float(0.10 * (torch.rand(1) - 0.5))
                r_pt = arm_start_s + dr_pt
                kx = mx + r_pt * np.cos(th_pt)
                ky = my + r_pt * np.sin(th_pt)
                kw = float(0.008 + 0.012 * torch.rand(1))
                ka = dim * float(0.15 + 0.30 * torch.rand(1)) * (1.0 - 0.6 * t)
                img += ka * torch.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / kw ** 2)
        # Heavy HII speckling clustered on the clumps.
        n_kn = int(torch.randint(15, 36, (1,)))
        for _ in range(n_kn):
            bx, by = pts[int(torch.randint(0, len(pts), (1,)))]
            kx = bx + float(torch.randn(1)) * 0.07
            ky = by + float(torch.randn(1)) * 0.07
            lum_skew = float(torch.rand(1)) ** 3
            kw = float(0.008 + 0.020 * lum_skew)
            ka = float(0.20 + 0.70 * lum_skew)
            img += ka * torch.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / kw ** 2)

    # --- SATELLITES ---
    num_satellites = int(torch.randint(12, 21, (1,)))
    for _ in range(num_satellites):
        sat_r = float(0.9 + 2.0 * torch.rand(1))
        sat_theta = float(2 * np.pi * torch.rand(1))
        sat_x = sat_r * np.cos(sat_theta)
        sat_y = sat_r * np.sin(sat_theta)
        sat_w = float(0.07 + 0.08 * torch.rand(1))
        sat_amp = float(0.05 + 0.2 * torch.rand(1))
        img += sat_amp * torch.exp(-((xx - sat_x) ** 2 + (yy - sat_y) ** 2) / sat_w ** 2)

    # --- STARRY BACKGROUND (Point Sources) ---
    num_stars = int(torch.randint(300, 501, (1,)))
    for _ in range(num_stars):
        star_x = float(8.0 * torch.rand(1) - 4.0)
        star_y = float(8.0 * torch.rand(1) - 4.0)
        star_w = float(0.01 + 0.015 * torch.rand(1))
        star_amp = float(0.1 + 0.3 * torch.rand(1))
        img += star_amp * torch.exp(-((xx - star_x) ** 2 + (yy - star_y) ** 2) / star_w ** 2)

    # --- BACKGROUND FLUCTUATIONS & NOISE ---
    bg_x = float(6.0 * torch.rand(1) - 3.0)
    bg_y = float(6.0 * torch.rand(1) - 3.0)
    bg_w = float(2.0 + 2.0 * torch.rand(1))
    bg_amp = float(0.05 + 0.1 * torch.rand(1))
    img += bg_amp * torch.exp(-((xx - bg_x) ** 2 + (yy - bg_y) ** 2) / bg_w ** 2)
    sky_bg = 0.05
    img = img + sky_bg
    noise_level = float(0.04 + 0.08 * torch.rand(1))
    img = img + noise_level * torch.randn(size, size)
    return img.clamp(0, 3).unsqueeze(0)


CLASSES = ["elliptical", "spiral", "barred spiral", "merger",
           "edge-on", "irregular"]
