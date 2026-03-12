"""
QU fitting reconstructor: fit Q(λ²), U(λ²).

Placeholder for future QU-fitting pipeline. Overrides reconstruct() (no pipeline).
"""
from dataclasses import dataclass

from .base import FaradayReconstructorWrapper


@dataclass
class QUFittingReconstructorWrapper(FaradayReconstructorWrapper):

    def config_fd_space(self):
        pass

    def reconstruct(self):
        pass

    def calculate_second_moment(self):
        pass
