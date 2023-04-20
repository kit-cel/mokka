"""Module with data generators."""

from . import numpy  # noqa

try:
    from . import torch  # noqa
except ImportError:
    pass
