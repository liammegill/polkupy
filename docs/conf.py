"""Sphinx configuration for polkupy's documentation."""

project = "polkupy"
copyright = "2026, Liam Megill"
author = "Liam Megill"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

exclude_patterns = ["_build"]

html_theme = "furo"

html_theme_options = {
    "navigation_with_keys": True,
}
