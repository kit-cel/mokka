![tests](https://github.com/kit-cel/mokka/actions/workflows/python-package.yml/badge.svg) [![codecov](https://codecov.io/gh/kit-cel/mokka/graph/badge.svg?token=GXXZDNJ9W8)](https://codecov.io/gh/kit-cel/mokka) [![PyPI - Version](https://img.shields.io/pypi/v/mokka)](https://pypi.org/project/mokka) [![docs](https://img.shields.io/badge/docs-available-green)](https://kit-cel.github.io/mokka/#)

# Machine Learning and Optimization for Communications (MOKka) 
(german: **M**aschinelles Lernen und **O**ptimierung für **K**ommuni**ka**tionssysteme)

This Python package is intended to provide signal and information processing blocks in popular machine learning frameworks.
Most of the functionality currently is provided for the PyTorch machine learning framework. Some parts are already implemented to interface with the Sionna Python package.

## Prerequisites

This package works best if it is installed in a Python virtualenvironment, either provided by `python -m venv` or `conda`.
To leverage GPU for training with PyTorch and/or latest features of the PyTorch nightly version follow https://pytorch.org/get-started/locally/ to
install PyTorch from the PyTorch repositories and avoid installing the PyPI provided general-purpose version of PyTorch.

## Installation

To install `mokka` with functionality to work with PyTorch clone this repository and install (in a virtual environment):
```
pip install '.[torch]'
```

Several more optional dependencies are provided to support Sionna (and tensorflow) `[tf]` or for formatting and more development tools `[dev]`.
The popular machine learning reporting and experiment tracking tool weights & biases can be installed with `[remotelog]`.
For all possible optional dependencies enabled e.g.:
```
pip install '.[torch,tf,dev,remotelog]'
```

## Usage

This package provides a range of modules, covering a range of functions within a communication system.
To access functionality within a specific module first `import mokka` and then access of modules is possible, e.g. `mokka.mapping.torch` provides
constellation mappers and demappers implemented with and compatible to the PyTorch framework.
