"""Sphinx configuration for the galaxy-textures documentation site."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "galaxy-textures"
copyright = "2026, Arnab Lahiry"
author = "Arnab Lahiry"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "galaxy-textures"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
}