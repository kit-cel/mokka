# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from sphinx_pyproject import SphinxConfig

module_dir = os.path.join(os.path.dirname(__file__), "../src/mokka")

config = SphinxConfig("../pyproject.toml", globalns=globals(), style="poetry")

author=config.author
project=config.name
