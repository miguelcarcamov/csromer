"""
Pipelines: simulation and reconstruction.

- simulation: run_simulation(ctx, steps), SimulateStep, ApplyNoiseStep, ApplyRFIStep
- reconstruction: optimization (CS-ROMER), pol-angle gradient, QU fitting
"""
from csromer.pipelines.core import Step, run_pipeline
from csromer.pipelines.reconstruction import (
    CSROMERReconstructorWrapper,
    FaradayReconstructorWrapper,
    PolAngleGradientReconstructorWrapper,
    QUFittingReconstructorWrapper,
    build_measurement_operator,
    build_parameter,
    default_objective_factory,
    make_cg_optimizer,
    make_fista_optimizer,
)
from csromer.pipelines.simulation import (
    ApplyNoiseStep,
    ApplyRFIStep,
    SimulateStep,
    run_simulation,
)

__all__ = [
    "Step",
    "run_pipeline",
    "run_simulation",
    "SimulateStep",
    "ApplyNoiseStep",
    "ApplyRFIStep",
    "FaradayReconstructorWrapper",
    "CSROMERReconstructorWrapper",
    "PolAngleGradientReconstructorWrapper",
    "QUFittingReconstructorWrapper",
    "build_parameter",
    "build_measurement_operator",
    "default_objective_factory",
    "make_cg_optimizer",
    "make_fista_optimizer",
]
