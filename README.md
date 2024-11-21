![tests](https://github.com/kit-cel/mokka/actions/workflows/python-package.yml/badge.svg) [![codecov](https://codecov.io/gh/kit-cel/mokka/graph/badge.svg?token=GXXZDNJ9W8)](https://codecov.io/gh/kit-cel/mokka) [![PyPI - Version](https://img.shields.io/pypi/v/mokka)](https://pypi.org/project/mokka) [![docs](https://img.shields.io/badge/docs-available-green)](https://kit-cel.github.io/mokka/#)

# Machine Learning and Optimization for Communications (MOKka) 
(german: **M**aschinelles Lernen und **O**ptimierung für **K**ommuni**ka**tionssysteme)

This Python package is intended to provide signal and information processing blocks in popular machine learning frameworks.
Most of the functionality currently is provided for the PyTorch machine learning framework. Some parts are already implemented to interface with the Sionna Python package.

## Prerequisites

This package leverages poetry for management of dependencies in a virtual environment. If you clone this repository you can install this package with all
optional extras with `poetry install --all-extras`. Alternatively check which additional functionalities suit you in [pyproject.toml](./pyproject.toml).

## Installation

Either install the package from PyPI with `pip install mokka` or install it from this source directory with the help of poetry `poetry install --all-extras`.
To install the development dependencies (formatters, documentation checkers, etc.) install mokka with `poetry install --with=dev`.
Please check the poetry documentation for other commands and usage of poetry [here](https://python-poetry.org/docs/).

## Usage

This package provides a range of modules, covering a range of functions within a communication system.
To access functionality within a specific module first `import mokka` and then access of modules is possible, e.g. `mokka.mapping.torch` provides
constellation mappers and demappers implemented with and compatible to the PyTorch framework.

## Development

In order to allow for consistent development we use pre-commit to check formatting and documentation of source code before creating commits. To install required tools for development specify option `[dev]` during installation of mokka. E.g. if you are installing in editable mode from the git root of this repository you can run `pip install ".[torch,dev]"` to install dependencies for running mokka with torch and development tools.

To install the required pre-commit hooks you run `pre-commit install` from the git root, this will use the `.pre-commit.yml` file in the root to install hooks required to check formatting and documentation of newly commited source code.

## Acknowledgment
This  work  has  received  funding  from  the  European  Research Council (ERC) under the European Union's Horizon2020 research and innovation programme (grant agreement No. 101001899).
