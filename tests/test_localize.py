"""Test picasso.localize functions as well as the associated functions
in picasso.gausslq, picasso.gaussmle, picasso.zfit

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

# TODO: add identifying more data types? like .tif, .nd2? this can go to test_io.py though

import numpy as np
import pytest
from picasso import io, localize, gausslq, gaussmle, zfit

# parameters for localization
BOX = 7
MIN_NG = 5000
CAMERE_INFO = {
    "Baseline": 0,
    "Sensitivity": 1,
    "Gain": 1,
}
ROI = ((0, 0), (16, 32))
DRIFT_SEG = 100
CALIB_3D = {
    "X Coefficients": [
        -1.6680708772714857e-18,
        2.4038209829154137e-15,
        2.1771067332017187e-12,
        -3.0324788231238476e-09,
        3.5433326085494675e-06,
        0.0023039289366630425,
        1.2026032603707493,
    ],
    "Y Coefficients": [
        -1.7708672355491796e-18,
        9.808249540501714e-16,
        2.10653248543535e-12,
        2.228026137415219e-11,
        3.628007433361433e-06,
        -0.001646865504353452,
        1.2257249554338714,
    ],
}


@pytest.fixture(scope="module")
def movie_data():
    """Load movie data once per test module."""
    movie, info = io.load_movie("./tests/data/testdata.raw")
    return movie, info


@pytest.fixture
def movie(movie_data):
    """Provide movie data to test functions."""
    return movie_data[0]


@pytest.fixture
def info(movie_data):
    """Provide info data to test functions."""
    return movie_data[1]


@pytest.fixture
def identifications(movie):
    """Provide identifications to test functions."""
    ids = localize.identify(movie, MIN_NG, BOX)
    return ids


@pytest.fixture
def spots(movie, identifications):
    """Provide extracted spots to test functions."""
    spots = localize.get_spots(movie, identifications, BOX, CAMERE_INFO)
    return spots


@pytest.fixture
def theta_lq(spots):
    """Provide least-squares fitting results."""
    return gausslq.fit_spots(spots)


@pytest.fixture
def locs_lq(identifications, theta_lq):
    """Provide 2D localizations from least-squares fitting."""
    return gausslq.locs_from_fits(identifications, theta_lq, BOX, False)


def test_identification_with_roi(movie, identifications):
    """Test identification of spots in movie frames.

    Parameters
    ----------
    movie : np.memmap or np.ndarray
        Movie data as 3D array (frames, height, width)
    info : dict
        Movie metadata information

    Returns
    -------
    pd.DataFrame
        DataFrame containing identified spots with columns:
        'frame', 'x', 'y', 'net_gradient'
    """
    # run identification with the defined parameters
    identifications_roi = localize.identify(movie, MIN_NG, BOX, roi=ROI)

    # basic validation tests
    assert not identifications.empty, "No spots were identified"
    assert all(
        col in identifications.columns
        for col in ["frame", "x", "y", "net_gradient"]
    ), "Missing required columns in identification results"
    assert (
        identifications["net_gradient"].min() >= MIN_NG
    ), f"Found spots with net_gradient below minimum threshold {MIN_NG}"
    assert identifications["frame"].min() >= 0, "Invalid frame numbers found"
    assert identifications["frame"].max() < len(
        movie
    ), "Frame numbers exceed movie length"

    # compare the two identifications
    assert len(identifications_roi) <= len(
        identifications
    ), "ROI identification returned more spots than full frame identification"


def test_identification_threaded_vs_non_threaded(movie, identifications):
    """Test that threaded and non-threaded identification give
    consistent results."""
    # get results from both modes
    ids_threaded = localize.identify(movie, MIN_NG, BOX, threaded=True)

    # compare results (should be identical or very similar)
    assert len(ids_threaded) == len(
        identifications
    ), "Different number of spots identified in threaded vs non-threaded mode"


def test_localize_lq(identifications, spots, theta_lq):
    """Test localizing identified spots using least-squares fitting,
    including their preprocessing."""
    # test extracting spots
    assert len(spots) == len(
        identifications
    ), "Number of extracted spots does not match number of identifications"

    # test localization via least-squares fitting
    theta_lq_multi = gausslq.fit_spots_parallel(spots, asynch=False)
    assert len(theta_lq) == len(
        spots
    ), "Number of localized spots (LQ) does not match number of extracted spots"
    assert len(theta_lq_multi) == len(
        spots
    ), "Number of localized spots (LQ multi) does not match number of extracted spots"
    assert np.allclose(
        theta_lq, theta_lq_multi
    ), "LQ fitting results differ between single-threaded and multi-threaded implementations"


def test_localize_mle(identifications, spots):
    """Test localizing identified spots via maximum-likelihood
    fitting."""
    # test localization via maximum-likelihood fitting
    theta_mle_xy, CRLBs, lls, its = gaussmle.gaussmle(
        spots, eps=1e-3, max_it=1000, method="sigmaxy"
    )
    theta_mle, _, _, _ = gaussmle.gaussmle(
        spots, eps=1e-3, max_it=1000, method="sigma"
    )
    locs_mle = gaussmle.locs_from_fits(
        identifications, theta_mle_xy, CRLBs, lls, its, BOX
    )
    assert len(theta_mle_xy) == len(
        spots
    ), "Number of localized spots (MLE sigmaxy) does not match number of extracted spots"
    assert len(theta_mle) == len(
        spots
    ), "Number of localized spots (MLE sigma) does not match number of extracted spots"

    # TODO: the interface for mle fitting via parallel processing is a
    # little messy, needs to be made more analogous to the lq fitting,
    # then the tests can be added here as well.


def test_localize_3d(info, locs_lq):
    """Test 3D localization of identified spots."""
    locs_3d = zfit.fit_z(locs_lq, info, CALIB_3D, 0.79, 130)
    locs_3d_multi = zfit.fit_z_parallel(
        locs_lq, info, CALIB_3D, 0.79, 130, asynch=False
    )
    assert len(locs_3d) == len(
        locs_lq
    ), "Number of 3D localized spots does not match number of 2D localized spots"
    assert len(locs_3d_multi) == len(
        locs_lq
    ), "Number of 3D localized spots (multi) does not match number of 2D localized spots"
