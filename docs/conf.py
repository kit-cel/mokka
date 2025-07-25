# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from sphinx_pyproject import SphinxConfig

module_dir = os.path.join(os.path.dirname(__file__), "../src/mokka")

config = SphinxConfig("../pyproject.toml", globalns=globals())

author=config.author
project=config.name

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
