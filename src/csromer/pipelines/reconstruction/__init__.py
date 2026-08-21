"""
Reconstruction pipelines: optimization (CS-ROMER), pol-angle gradient, QU fitting.

- Optimization: CSROMERReconstructorWrapper + steps (L2Zero → … → RestoredStats).
- Pol-angle gradient: PolAngleGradientReconstructorWrapper (fit line to pol angle vs λ²).
- QU fitting: QUFittingReconstructorWrapper (placeholder).
"""
from csromer.pipelines.core import Step, run_pipeline

from .base import FaradayReconstructorWrapper
from .defaults import build_measurement_operator, build_parameter, default_objective_factory
from .optimization import CLEANReconstructorWrapper, CSROMERReconstructorWrapper
from .optimizer_factories import make_cg_optimizer, make_fista_optimizer
from .pol_angle_gradient import PolAngleGradientReconstructorWrapper
from .qufitting import QUFittingReconstructorWrapper

__all__ = [
    "Step",
    "run_pipeline",
    "FaradayReconstructorWrapper",
    "CLEANReconstructorWrapper",
    "CSROMERReconstructorWrapper",
    "PolAngleGradientReconstructorWrapper",
    "QUFittingReconstructorWrapper",
    "build_parameter",
    "build_measurement_operator",
    "default_objective_factory",
    "make_cg_optimizer",
    "make_fista_optimizer",
]
