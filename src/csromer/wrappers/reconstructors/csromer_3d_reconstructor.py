from dataclasses import dataclass, field
from typing import Union

import numpy as np
from astropy.stats import sigma_clipped_stats

from ...dictionaries import Wavelet
from ...objectivefunction import L1, TSV, TV, Chi2, OFunction
from ...optimization import FISTA
from ...reconstruction import Parameter
from ...transformers.dfts import NDFT1D, NUFFT1D
from ...transformers.flaggers.flagger import Flagger
from .faraday_reconstructor import FaradayReconstructorWrapper


@dataclass(init=True, repr=True)
class CSROMER3DReconstructorWrapper(FaradayReconstructorWrapper):
    sigma_threshold_p: float = None
    sigma_threshold_intensity: float = None
    spectral_index: Union[Union[str, float], np.ndarray] = None
