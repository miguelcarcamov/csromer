from dataclasses import dataclass

from .faraday_reconstructor import FaradayReconstructorWrapper


@dataclass
class QUFittingReconstructorWrapper(FaradayReconstructorWrapper):

    def __post_init__(self):
        super().__init__()

    def config_fd_space(self):
        pass

    def reconstruct(self):
        pass

    def calculate_second_moment(self):
        pass
