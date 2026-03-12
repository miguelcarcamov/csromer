import numpy as np

from csromer.simulation import ALL_BANDS, LOFAR_HIGH, LOFAR_LOW, SKA_LOW, SKA_MID_B2, get_band


def test_ska_low_freq_range():
    f = SKA_LOW.freq_array()
    assert f.min() >= 50e6 and f.max() <= 350e6
    assert len(f) > 1


def test_band_freq_array_dask():
    f = SKA_LOW.freq_array(use_dask=True)
    assert hasattr(f, "compute")
    np.testing.assert_allclose(f.compute(), SKA_LOW.freq_array())


def test_get_band():
    assert get_band("SKA-LOW") is not None
    assert get_band("ska-low") is not None
    assert get_band("unknown") is None


def test_all_bands():
    assert len(ALL_BANDS) >= 6
    names = [b.name for b in ALL_BANDS]
    assert "SKA-LOW" in names
    assert "LOFAR-High" in names


def test_lofar_high_low():
    low = LOFAR_LOW.freq_array()
    high = LOFAR_HIGH.freq_array()
    assert low.max() < high.min()
