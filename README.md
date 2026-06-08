![tests](https://github.com/kit-cel/mokka/actions/workflows/python-package.yml/badge.svg) [![codecov](https://codecov.io/gh/kit-cel/mokka/graph/badge.svg?token=GXXZDNJ9W8)](https://codecov.io/gh/kit-cel/mokka) [![PyPI - Version](https://img.shields.io/pypi/v/mokka)](https://pypi.org/project/mokka) [![docs](https://img.shields.io/badge/docs-available-green)](https://kit-cel.github.io/mokka/#)

# Machine Learning and Optimization for Communications (MOKka) 
(german: **M**aschinelles Lernen und **O**ptimierung für **K**ommuni**ka**tionssysteme)

This Python package is intended to provide signal and information processing blocks in popular machine learning frameworks.
Most of the functionality currently is provided for the PyTorch machine learning framework. Some parts are already implemented to interface with the Sionna Python package.

## Prerequisites

This package uses [UV](https://github.com/astral-sh/uv) for fast Python package management and virtual environments. UV is a modern alternative to pip/poetry that provides significantly faster dependency resolution and installation.

## Installation

### From PyPI (Recommended)

```bash
# Install using UV (fastest method)
uv add mokka

# Or using pip (traditional method)
pip install mokka
```

### From Source

```bash
# Clone the repository
git clone https://github.com/kit-cel/mokka.git
cd mokka

# Install with all optional extras using UV
uv sync --all-extras

# Or install specific extras (e.g., torch + dev tools)
uv sync --extra torch --extra dev
```

### Development Setup

```bash
# Install development dependencies
uv sync --with=dev

# Install pre-commit hooks for code quality
pre-commit install
```

For more advanced usage, refer to the [UV documentation](https://github.com/astral-sh/uv).

## Usage

This package provides a range of modules, covering a range of functions within a communication system.
To access functionality within a specific module first `import mokka` and then access of modules is possible, e.g. `mokka.mapping.torch` provides
constellation mappers and demappers implemented with and compatible to the PyTorch framework.

## Development

For consistent development, we use pre-commit hooks to check code formatting and documentation before commits. The project has been updated to use UV for faster dependency management.

### Development Installation

```bash
# Install in development mode with UV
uv sync --editable --extra torch --extra dev

# Install pre-commit hooks
pre-commit install
```

### Common Development Commands

```bash
# Run tests with coverage
uv run pytest --cov=src --cov-report=xml

# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy src
```

The pre-commit hooks will automatically run formatting and linting checks on each commit. For more information about the development workflow, see the [Contributing Guide](CONTRIBUTING.md).

## Acknowledgment
This  work  has  received  funding  from  the  European  Research Council (ERC) under the European Union's Horizon2020 research and innovation programme (grant agreement No. 101001899).
