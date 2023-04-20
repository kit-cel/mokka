"""Operations for bit manipulation."""
from . import numpy  # noqa
from .numpy import gray  # noqa

try:
    from . import torch  # noqa
except ImportError:
    pass
