"""Line search for CG and FISTA (Pyralysis-style)."""
from .backtracking import BacktrackingArmijo
from .brent import Brent
from .fibonacci_search import Fibonacci, pure_fibonacci_search
from .fista_backtracking import FISTABacktracking
from .fixed import Fixed
from .gll_backtracking import GLLArmijo
from .golden_section_search import GoldenSection, pure_golden_section_search
from .goldstein import Goldstein
from .linesearcher import LineSearcher
from .seeders import (
    BarzilaiBorwein,
    BarzilaiBorweinAdaptiveMin1,
    BarzilaiBorweinAdaptiveMin2,
    BarzilaiBorweinAlternating,
    CubicInterpolationSeeder,
    QuadraticInterpolationSeeder,
    StepSizeSeeder,
)

__all__ = [
    "BacktrackingArmijo",
    "BarzilaiBorwein",
    "BarzilaiBorweinAdaptiveMin1",
    "BarzilaiBorweinAdaptiveMin2",
    "BarzilaiBorweinAlternating",
    "Brent",
    "CubicInterpolationSeeder",
    "Fibonacci",
    "FISTABacktracking",
    "Fixed",
    "GLLArmijo",
    "GoldenSection",
    "Goldstein",
    "LineSearcher",
    "QuadraticInterpolationSeeder",
    "StepSizeSeeder",
    "pure_fibonacci_search",
    "pure_golden_section_search",
]
