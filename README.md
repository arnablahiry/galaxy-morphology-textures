# toy-galaxy-morphology-textures

A procedural, PyTorch-based synthetic galaxy image generator built for
teaching neural network interpretability — not for astrophysical research.
Every image is assembled from explicit analytic profiles (Sersic bulges,
exponential disks, logarithmic spiral arms, sech² edge-on disks, tidally
distorted merger components, random-walk irregulars). There is no simulation
or real imaging data underneath it, and nothing here is photometrically
calibrated.

The point of building it this way is that every generative factor is known
and controllable: arm count, bar length, merger mass ratio, interaction
stage, star-formation activity, and so on are all explicit parameters (or
per-image draws) rather than hidden in a black-box simulator. That makes it
useful for correlating what a classifier attends to (saliency maps, Grad-CAM,
activation maps) against ground truth you actually control.

## Classes

| kind | name          | notes |
|------|---------------|-------|
| 0    | elliptical    | Sersic bulge + warped extended halo |
| 1    | spiral        | 2–3 arms rooted at the bulge, occasional bifurcation, HII knots |
| 2    | barred spiral | quartic bar, arms rooted at the bar tips |
| 3    | merger        | two tidally interacting disks (spiral-spiral or spiral-barred), continuous interaction `stage`, mass-ratio asymmetry, bridge + outward tails |
| 4    | edge-on       | Sombrero-like thin disk + bulge + dust lane |
| 5    | irregular     | clumpy random-walk body, optional tidal debris plume, heavy HII speckling |

## Install

```bash
pip install -r requirements.txt
# or, as an editable package:
pip install -e .
```

## Usage

```python
import torch
from galaxy_zoo import make_galaxy, CLASSES

torch.manual_seed(0)
img = make_galaxy(kind=1, size=128)   # -> torch.Tensor, shape (1, 128, 128), values in [0, 3]

# Merger-specific overrides (kind == 3 only):
img = make_galaxy(kind=3, size=128, stage=0.8, merger_barred=True)
```

Render a demo grid sampling all six classes:

```bash
python examples/render_grid.py --out galaxy_zoo_grid.png
```

## Known simplifications

This is a stylized teaching dataset, not simulation-grade. Known gaps worth
being aware of before drawing conclusions from anything trained on it:

- Single-band grayscale — no color gradients (real ellipticals are redder,
  star-forming arms bluer), which is a real classification cue in actual
  Galaxy Zoo data.
- Flat additive Gaussian noise rather than Poisson (source-dependent) noise,
  and no PSF/seeing blur.
- Merger tidal dynamics (bridge/tail geometry, arm stretching) are a
  physically-motivated cartoon of the Toomre & Toomre mechanism, not an
  N-body integration — the two galaxies' spins and orbital angular momentum
  aren't fully self-consistent.
- Classes are, by construction, fairly cleanly separable (e.g. mergers
  always have exactly two nuclei). Real Galaxy Zoo classification is
  hard specifically because of the ambiguous middle ground this generator
  doesn't fully reproduce — worth keeping in mind if you're using saliency
  maps to reason about what a trained classifier has "discovered."

## License

MIT — see `LICENSE`.
