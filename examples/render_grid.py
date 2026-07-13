"""Render a demo grid sampling all six galaxy classes.

Usage:
    python examples/render_grid.py
    python examples/render_grid.py --seed 42 --rows 6 --cols 6 --size 128
"""

import argparse

import matplotlib.pyplot as plt
import torch

from galaxy_zoo import CLASSES, make_galaxy

plt.rcParams.update({
    "font.family": "serif",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.dpi": 150,
    "savefig.dpi": 150,
})


def clean_axis(ax):
    """No ticks, but keep a thin black frame around the subplot."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")


def main(seed: int = 999, size: int = 128, rows: int = 5, cols: int = 6,
         out_path: str = "galaxy_textures_grid.png") -> None:
    torch.manual_seed(seed)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    for a in axes.flat:
        k = int(torch.randint(0, len(CLASSES), (1,)))
        img = make_galaxy(k, size=size)
        a.imshow(img.squeeze(), cmap="gray")
        a.set_title(CLASSES[k], fontsize=9)
        clean_axis(a)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--out", type=str, default="galaxy_textures_grid.png")
    args = parser.parse_args()
    main(seed=args.seed, size=args.size, rows=args.rows, cols=args.cols,
         out_path=args.out)
