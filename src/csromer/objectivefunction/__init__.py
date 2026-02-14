from .fi import *
from .ofunction import *
from . import priors
from .priors import *

def __getattr__(name: str):
    """Re-export TSV/TV from priors (lazy so prox_tv not required for ChiSquared-only use)."""
    if name in ("TSV", "TV"):
        return getattr(priors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
