"""Toy galaxy morphology texture generator package."""

from .barred_spiral import draw_barred_spiral
from .edge_on import draw_edge_on
from .elliptical import draw_elliptical
from .generator import CLASSES, make_galaxy
from .irregular import draw_irregular
from .merger import draw_merger
from .spiral import draw_spiral

__all__ = [
	"make_galaxy",
	"CLASSES",
	"draw_elliptical",
	"draw_spiral",
	"draw_barred_spiral",
	"draw_merger",
	"draw_edge_on",
	"draw_irregular",
]
