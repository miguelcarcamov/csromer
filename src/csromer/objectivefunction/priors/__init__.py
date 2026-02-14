from .chi_squared import ChiSquared
from .l1 import L1
from .l2 import L2


def __getattr__(name: str):
    """Lazy load TSV/TV so ChiSquared tests can run without prox_tv."""
    if name == "TSV":
        from .tsv import TSV
        return TSV
    if name == "TV":
        from .tv import TV
        return TV
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
