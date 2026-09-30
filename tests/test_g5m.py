"""Test ``picasso.g5m``.

Covers the numba kernels directly, the end-to-end runs through the
public ``g5m.g5m``, and the CLI/GUI wiring. The published models
(spherical 2D, diagonal 3D) are pinned against golden values so that
adding a covariance type cannot silently perturb them.

Most fixtures (``locs``, ``info``) live in ``tests/conftest.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pytest

from picasso import clusterer, g5m, zfit
from tests.conftest import CALIB_3D

# ---------------------------------------------------------------------------
# Golden values, captured from the published models before the rotated
# covariance type was added. These pin the exact float32 output of
# ``g5m.g5m`` at its defaults; any deviation means a supposedly additive
# change perturbed the published code path.
# ---------------------------------------------------------------------------

GOLDEN_2D = {
    "n": 13,
    "x": (
        25.629241943359375,
        25.499746322631836,
        15.626664161682129,
        5.626276016235352,
        25.377365112304688,
        15.498655319213867,
        5.62256383895874,
        25.498985290527344,
        15.376860618591309,
        15.624945640563965,
        25.622488021850586,
        15.37427806854248,
        25.374717712402344,
    ),
    "y": (
        25.438974380493164,
        5.6814141273498535,
        5.433107376098633,
        5.561403274536133,
        5.439255237579346,
        25.437074661254883,
        15.435615539550781,
        25.311582565307617,
        15.562479972839355,
        15.444314956665039,
        25.31130599975586,
        5.435296535491943,
        15.69091796875,
    ),
    "fitted_sigma": (
        1.0386015176773071,
        1.5896075963974,
        1.1116163730621338,
        0.9552916288375854,
        0.91796875,
        1.038244366645813,
        0.9711275100708008,
        1.117280125617981,
        0.928167998790741,
        1.1781824827194214,
        1.1979488134384155,
        1.1343822479248047,
        1.0053337812423706,
    ),
    "p_val": (
        0.8738712072372437,
        0.7839063405990601,
        0.7222988605499268,
        0.656937837600708,
        0.7742874622344971,
        0.6707575917243958,
        0.5236965417861938,
        0.7757023572921753,
        0.824918270111084,
        0.5161045789718628,
        0.5175729393959045,
        0.5181434154510498,
        0.7041453719139099,
    ),
}

GOLDEN_3D = {
    "n": 13,
    "x": GOLDEN_2D["x"],
    "y": GOLDEN_2D["y"],
    "z": (
        0.0854271948337555,
        -0.16333098709583282,
        0.30976176261901855,
        -0.5918141007423401,
        -0.09213722497224808,
        -0.7355197072029114,
        -0.22255362570285797,
        -0.46149805188179016,
        1.039459228515625,
        0.4614470601081848,
        -0.7819962501525879,
        -0.3121948540210724,
        0.6350541114807129,
    ),
    "fitted_sigma_x": (
        1.028760313987732,
        1.6458141803741455,
        1.101087212562561,
        0.9944224953651428,
        0.9092680811882019,
        1.0283931493759155,
        0.9948172569274902,
        1.1066838502883911,
        0.91938716173172,
        1.1670256853103638,
        1.3318597078323364,
        1.167402982711792,
        0.9958165287971497,
    ),
    "fitted_sigma_y": (
        1.048536777496338,
        1.6774659156799316,
        1.1222461462020874,
        1.0135606527328491,
        0.9267526865005493,
        1.0481898784637451,
        1.0139511823654175,
        1.127977967262268,
        0.9370326995849609,
        1.1894458532333374,
        1.3575001955032349,
        1.1898596286773682,
        1.0149420499801636,
    ),
    "fitted_sigma_z": (
        1.741146206855774,
        2.463754653930664,
        1.680849313735962,
        1.6309601068496704,
        1.8074944019317627,
        1.560007929801941,
        1.4714486598968506,
        1.6870323419570923,
        1.382270097732544,
        1.3063181638717651,
        2.1090328693389893,
        1.513647437095642,
        2.3484909534454346,
    ),
    "p_val": (
        0.8294602632522583,
        0.9255040884017944,
        0.8304082155227661,
        0.71590656042099,
        0.7293727993965149,
        0.837804913520813,
        0.5986433029174805,
        0.9383277893066406,
        0.8152422308921814,
        0.5699061751365662,
        0.7578170299530029,
        0.8004807233810425,
        0.3054245114326477,
    ),
}


@pytest.fixture
def dbscan_locs(locs):
    """2D clustered localizations, no ``angle`` column."""
    out = clusterer.dbscan(locs, radius=2 / 130, min_samples=2)[0]
    assert len(out) > 0
    assert "angle" not in out.columns
    return out


@pytest.fixture
def dbscan_locs_3d(dbscan_locs, info):
    """3D clustered localizations, no ``angle`` column."""
    out = dbscan_locs.copy()
    rng = np.random.default_rng(42)
    out["z"] = rng.normal(0, 2, size=len(out))
    out["lpz"] = zfit.axial_localization_precision(
        out, info, calibration=CALIB_3D, fitting_method="gaussmle"
    )
    return out


def _rot_cov(cov_maj, cov_min, cov_z, theta):
    """Reference block covariance, built with plain numpy."""
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    xy = R @ np.diag([cov_maj, cov_min]) @ R.T
    out = np.zeros((3, 3))
    out[:2, :2] = xy
    out[2, 2] = cov_z
    return out


class TestCircularMeanAngle:
    """The PSF angle is axial (period pi), so it must not be averaged
    linearly."""

    def test_wrap_at_ninety_degrees(self):
        # +89 and -89 deg point in nearly the same direction; a linear
        # mean would return 0 deg, which is perpendicular to the truth
        angles = np.deg2rad(np.array([89.0, -89.0]))
        resp = np.ones((2, 1))
        theta = g5m._circular_weighted_mean_angle(resp, angles)
        assert abs(abs(np.rad2deg(theta[0])) - 90.0) < 1e-6

    def test_matches_linear_mean_away_from_wrap(self):
        angles = np.deg2rad(np.array([28.0, 30.0, 32.0]))
        resp = np.ones((3, 1))
        theta = g5m._circular_weighted_mean_angle(resp, angles)
        assert np.rad2deg(theta[0]) == pytest.approx(30.0, abs=1e-6)

    def test_respects_weights_and_components(self):
        angles = np.deg2rad(np.array([0.0, 60.0]))
        # component 0 sees only the first angle, component 1 only the
        # second
        resp = np.array([[1.0, 0.0], [0.0, 1.0]])
        theta = np.rad2deg(g5m._circular_weighted_mean_angle(resp, angles))
        assert theta[0] == pytest.approx(0.0, abs=1e-6)
        assert theta[1] == pytest.approx(60.0, abs=1e-6)


class TestPrecisionCholesky:
    @pytest.mark.parametrize("theta_deg", [0.0, 17.0, -33.0, 60.0, 89.0])
    @pytest.mark.parametrize(
        "cov_maj, cov_min, cov_z", [(4.0, 1.0, 9.0), (2.5, 2.5, 0.5)]
    )
    def test_matches_sklearn_definition(
        self, theta_deg, cov_maj, cov_min, cov_z
    ):
        """sklearn defines precisions_cholesky for full covariances as
        ``inv(cholesky(cov, lower=True)).T``."""
        from scipy.linalg import cholesky, solve_triangular

        cov = _rot_cov(cov_maj, cov_min, cov_z, np.deg2rad(theta_deg))
        covs = cov.reshape(1, 3, 3)

        got = g5m._precision_chol_3D_rot(covs)[0]
        expected = solve_triangular(
            cholesky(cov, lower=True), np.eye(3), lower=True
        ).T

        np.testing.assert_allclose(got, expected, atol=1e-12)
        # C @ C.T == inv(cov)
        np.testing.assert_allclose(got @ got.T, np.linalg.inv(cov), atol=1e-10)
        # upper triangular
        assert got[1, 0] == 0.0 and got[2, 0] == 0.0 and got[2, 1] == 0.0

    def test_log_det_identity(self):
        cov = _rot_cov(4.0, 1.0, 9.0, np.deg2rad(25.0))
        C = g5m._precision_chol_3D_rot(cov.reshape(1, 3, 3))[0]
        log_det = np.log(C[0, 0]) + np.log(C[1, 1]) + np.log(C[2, 2])
        assert log_det == pytest.approx(-0.5 * np.log(np.linalg.det(cov)))

    def test_reduces_to_diagonal_form_at_zero_angle(self):
        """At theta = 0 the block model must reproduce the plain
        ``1 / sqrt(cov)`` of the diagonal 3D model."""
        cov_diag = np.array([4.0, 1.0, 9.0])
        cov = _rot_cov(*cov_diag, 0.0)
        C = g5m._precision_chol_3D_rot(cov.reshape(1, 3, 3))[0]
        np.testing.assert_allclose(
            np.diag(C), 1.0 / np.sqrt(cov_diag), atol=1e-12
        )
        assert C[0, 1] == pytest.approx(0.0, abs=1e-12)


class TestAssembleCovs:
    @pytest.mark.parametrize("theta_deg", [0.0, 30.0, -45.0, 75.0])
    def test_matches_numpy_rotation(self, theta_deg):
        theta = np.deg2rad(theta_deg)
        got = g5m._assemble_covs_3D_rot(
            np.array([4.0]),
            np.array([1.0]),
            np.array([9.0]),
            np.array([theta]),
        )[0]
        np.testing.assert_allclose(
            got, _rot_cov(4.0, 1.0, 9.0, theta), atol=1e-12
        )

    @pytest.mark.parametrize("theta_deg", [0.0, 30.0, -45.0, 75.0])
    def test_eigendecomposition_round_trip(self, theta_deg):
        """The major axis of the assembled block must point along the
        angle it was built with."""
        theta = np.deg2rad(theta_deg)
        covs = g5m._assemble_covs_3D_rot(
            np.array([4.0]),
            np.array([1.0]),
            np.array([9.0]),
            np.array([theta]),
        )
        eigvals, eigvecs = np.linalg.eigh(covs[0][:2, :2])
        # eigh returns ascending eigenvalues; the major axis is last
        assert eigvals[1] == pytest.approx(4.0)
        assert eigvals[0] == pytest.approx(1.0)
        recovered = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
        # axial data: compare modulo 180 deg
        diff = (np.rad2deg(recovered - theta) + 90.0) % 180.0 - 90.0
        assert abs(diff) < 1e-8


class TestExponentialTerm:
    @pytest.mark.parametrize("theta_deg", [0.0, 30.0, -60.0])
    def test_matches_scipy_logpdf(self, theta_deg):
        from scipy.stats import multivariate_normal

        cov = _rot_cov(4.0, 1.0, 9.0, np.deg2rad(theta_deg))
        mean = np.array([3.0, -2.0, 0.5])
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3)) * 2.0 + mean

        pc = g5m._precision_chol_3D_rot(cov.reshape(1, 3, 3))
        sq = g5m._gauss_exponential_term_3D_rot(X, mean.reshape(1, 3), pc)[
            :, 0
        ]
        log_det = (
            np.log(pc[0, 0, 0]) + np.log(pc[0, 1, 1]) + np.log(pc[0, 2, 2])
        )
        got = -0.5 * (3 * np.log(2 * np.pi) + sq) + log_det

        expected = multivariate_normal(mean, cov).logpdf(X)
        np.testing.assert_allclose(got, expected, atol=1e-10)

    def test_stable_for_large_coordinates(self):
        """The kernel must take the difference before squaring. A naive
        ``x**2 - 2*x*mu + mu**2`` expansion loses precision badly when
        the coordinates are large and the spread is small, which is
        exactly the DNA-PAINT regime."""
        from scipy.stats import multivariate_normal

        cov = _rot_cov(0.01, 0.004, 0.02, np.deg2rad(35.0))
        mean = np.array([5000.5, 4000.5, 3000.5])
        rng = np.random.default_rng(1)
        X = mean + rng.normal(scale=0.1, size=(200, 3))

        pc = g5m._precision_chol_3D_rot(cov.reshape(1, 3, 3))
        sq = g5m._gauss_exponential_term_3D_rot(X, mean.reshape(1, 3), pc)[
            :, 0
        ]
        log_det = (
            np.log(pc[0, 0, 0]) + np.log(pc[0, 1, 1]) + np.log(pc[0, 2, 2])
        )
        got = -0.5 * (3 * np.log(2 * np.pi) + sq) + log_det

        expected = multivariate_normal(mean, cov).logpdf(X)
        np.testing.assert_allclose(got, expected, rtol=1e-9)


# A calibration with a strong, z-independent astigmatism: the polynomials
# are constant, so the fitted sigma ratio along the principal axes is
# 2.0 everywhere. The bundled CALIB_3D is almost isotropic (ratio ~0.99),
# which would make any component near-circular and its angle meaningless.
CALIB_ASTIG = {
    "X Coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    "Y Coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "Magnification factor": 0.79,
}
# sigma_u / sigma_v imposed by CALIB_ASTIG
CALIB_ASTIG_RATIO = 2.0


def _rotated_cluster(
    angle_deg, sigma_u_nm=14.0, sigma_v_nm=7.0, n=400, pixelsize=130.0, seed=0
):
    """One simulated molecule whose xy cloud is elongated at
    ``angle_deg``, in the convention of the ``angle`` column written by
    ``picasso.localize`` (degrees, wrapped to [-90, 90)).

    The u axis (width ``sigma_u_nm``) lies along ``angle_deg``.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    theta = np.deg2rad(angle_deg)
    su = sigma_u_nm / pixelsize
    sv = sigma_v_nm / pixelsize

    du = rng.normal(0.0, su, size=n)
    dv = rng.normal(0.0, sv, size=n)
    # rotate the principal-frame offsets into the camera frame
    dx = du * np.cos(theta) - dv * np.sin(theta)
    dy = du * np.sin(theta) + dv * np.cos(theta)

    lp = 8.0 / pixelsize
    return pd.DataFrame(
        {
            # frames spread across the movie and split into many binding
            # events so the postprocess filters would not remove it
            "frame": np.sort(
                rng.choice(np.arange(2000), size=n, replace=False)
            ).astype(np.float32),
            "x": 25.0 + dx,
            "y": 15.0 + dy,
            "z": rng.normal(0.0, 20.0, size=n),
            "lpx": np.full(n, lp),
            "lpy": np.full(n, lp),
            "lpz": np.full(n, 3.0 * lp * pixelsize),
            "angle": np.full(n, float(angle_deg)),
            "photons": np.full(n, 1000.0),
            "group": np.zeros(n, dtype=np.int32),
        }
    )


class TestRotatedAngleConvention:
    """The fitted xy block must be elongated along the direction the
    localizations report in their ``angle`` column. A sign or handedness
    error here is otherwise silent."""

    @pytest.mark.parametrize(
        "angle_deg", [0.0, 30.0, -30.0, 60.0, 89.0, -89.0]
    )
    def test_recovers_input_angle(self, angle_deg, info):
        locs = _rotated_cluster(angle_deg)
        mols, _, _ = g5m.g5m(
            locs,
            info,
            min_locs=20,
            covariance_type="rotated",
            calibration=CALIB_ASTIG,
            sigma_bounds=(0.3, 3.0),
            asynch=False,
            postprocess=False,
        )
        assert len(mols) == 1
        got = float(mols["angle"].iloc[0])
        # axial data: compare modulo 180 deg
        diff = (got - angle_deg + 90.0) % 180.0 - 90.0
        assert (
            abs(diff) < 8.0
        ), f"expected the major axis near {angle_deg} deg, got {got} deg"

    def test_recovers_elongation(self, info):
        locs = _rotated_cluster(30.0)
        mols, _, _ = g5m.g5m(
            locs,
            info,
            min_locs=20,
            covariance_type="rotated",
            calibration=CALIB_ASTIG,
            sigma_bounds=(0.3, 3.0),
            asynch=False,
            postprocess=False,
        )
        assert len(mols) == 1
        assert float(mols["axis_ratio"].iloc[0]) > 1.3
        assert (
            mols["fitted_sigma_major"].iloc[0]
            > mols["fitted_sigma_minor"].iloc[0]
        )


class TestAutoSelection:
    """ "auto" picks the rotated model exactly when the localizations are
    3D, astigmatism-fitted and carry an ``angle`` column."""

    def test_auto_selects_rotated_for_3d_astigmatism_with_angle(self, info):
        locs = _rotated_cluster(30.0)
        _, _, out_info = g5m.g5m(
            locs,
            info,
            min_locs=20,
            calibration=CALIB_ASTIG,
            asynch=False,
            postprocess=False,
        )
        assert out_info[-1]["Covariance type"] == "rotated"

    def test_auto_keeps_diagonal_without_angle(self, dbscan_locs_3d, info):
        _, _, out_info = g5m.g5m(
            dbscan_locs_3d,
            info,
            min_locs=5,
            calibration=CALIB_3D,
            asynch=False,
            postprocess=False,
        )
        assert out_info[-1]["Covariance type"] == "diagonal"

    def test_auto_keeps_diagonal_for_spline_even_with_angle(self, info):
        locs = _rotated_cluster(30.0)
        _, _, out_info = g5m.g5m(
            locs,
            info,
            min_locs=20,
            mode="spline",
            calibration=None,
            asynch=False,
            postprocess=False,
        )
        assert out_info[-1]["Covariance type"] == "diagonal"

    def test_auto_keeps_spherical_for_2d_even_with_angle(
        self, dbscan_locs, info
    ):
        # 2D rotated PSF fitting also writes an "angle" column, but the
        # rotated G5M model is 3D only
        locs = dbscan_locs.copy()
        locs["angle"] = 30.0
        _, _, out_info = g5m.g5m(
            locs, info, min_locs=5, asynch=False, postprocess=False
        )
        assert out_info[-1]["Covariance type"] == "spherical"

    def test_diagonal_forces_old_model_on_angle_carrying_data(self, info):
        """The explicit escape hatch for reproducing earlier results."""
        locs = _rotated_cluster(30.0)
        mols, _, out_info = g5m.g5m(
            locs,
            info,
            min_locs=20,
            covariance_type="diagonal",
            calibration=CALIB_ASTIG,
            asynch=False,
            postprocess=False,
        )
        assert out_info[-1]["Covariance type"] == "diagonal"
        assert "angle" not in mols.columns


class TestErrorPaths:
    def test_rotated_rejects_spline_mode(self, info):
        locs = _rotated_cluster(30.0)
        with pytest.raises(ValueError, match="astigmatism"):
            g5m.g5m(
                locs,
                info,
                min_locs=20,
                covariance_type="rotated",
                mode="spline",
                calibration=None,
                asynch=False,
            )

    def test_rotated_rejects_missing_angle_column(self, dbscan_locs_3d, info):
        with pytest.raises(ValueError, match="angle"):
            g5m.g5m(
                dbscan_locs_3d,
                info,
                min_locs=5,
                covariance_type="rotated",
                calibration=CALIB_3D,
                asynch=False,
            )

    def test_rotated_rejects_2d_data(self, dbscan_locs, info):
        with pytest.raises(ValueError, match="2D"):
            g5m.g5m(
                dbscan_locs,
                info,
                min_locs=5,
                covariance_type="rotated",
                asynch=False,
            )

    def test_spherical_rejects_3d_data(self, dbscan_locs_3d, info):
        with pytest.raises(ValueError, match="3D"):
            g5m.g5m(
                dbscan_locs_3d,
                info,
                min_locs=5,
                covariance_type="spherical",
                calibration=CALIB_3D,
                asynch=False,
            )

    def test_unknown_covariance_type(self, dbscan_locs, info):
        with pytest.raises(ValueError, match="covariance_type"):
            g5m.g5m(
                dbscan_locs,
                info,
                min_locs=5,
                covariance_type="garbage",
                asynch=False,
            )


class TestOutputContract:
    """Downstream code indexes these columns by name, so the rotated
    model must add to them rather than replace them."""

    def test_rotated_keeps_diagonal_columns_and_adds_shape_columns(self, info):
        locs = _rotated_cluster(30.0)
        mols, _, _ = g5m.g5m(
            locs,
            info,
            min_locs=20,
            covariance_type="rotated",
            calibration=CALIB_ASTIG,
            sigma_bounds=(0.3, 3.0),
            asynch=False,
            postprocess=False,
        )
        for col in [
            "x",
            "y",
            "z",
            "lpx",
            "lpy",
            "lpz",
            "fitted_sigma_x",
            "fitted_sigma_y",
            "fitted_sigma_z",
            "rel_sigma_x",
            "rel_sigma_y",
            "rel_sigma_z",
        ]:
            assert col in mols.columns, f"missing published column '{col}'"
        for col in [
            "fitted_sigma_major",
            "fitted_sigma_minor",
            "rel_sigma_major",
            "rel_sigma_minor",
            "axis_ratio",
            "angle",
        ]:
            assert col in mols.columns, f"missing new column '{col}'"
        # the molecule's own orientation replaces the input column
        # rather than being averaged into "angle_mean"
        assert "angle_mean" not in mols.columns

    def test_rel_sigma_plot_still_works(self, info, tmp_path):
        """``lib.plot_rel_sigma_check`` indexes rel_sigma_x/y/z directly
        and is called automatically by the Render GUI."""
        from picasso import lib

        locs = _rotated_cluster(30.0)
        mols, _, out_info = g5m.g5m(
            locs,
            info,
            min_locs=20,
            covariance_type="rotated",
            calibration=CALIB_ASTIG,
            asynch=False,
            postprocess=False,
        )
        path = tmp_path / "relsigma.png"
        lib.plot_rel_sigma_check(mols, out_info, str(path))
        assert path.exists() and path.stat().st_size > 0

    def test_no_nan_and_survives_save_round_trip(self, info, tmp_path):
        """``lib.ensure_sanity`` drops any row with a NaN in ANY column,
        so a NaN in a new column would silently delete molecules on
        save."""
        from picasso import io

        locs = _rotated_cluster(30.0)
        mols, _, out_info = g5m.g5m(
            locs,
            info,
            min_locs=20,
            covariance_type="rotated",
            calibration=CALIB_ASTIG,
            asynch=False,
            postprocess=False,
        )
        n_before = len(mols)
        assert n_before > 0
        bad = mols.replace([np.inf, -np.inf], np.nan).isna().sum().sum()
        assert bad == 0, f"{bad} NaN/inf values in the rotated output"

        path = tmp_path / "mols.hdf5"
        io.save_locs(str(path), mols, out_info)
        reloaded, _ = io.load_locs(str(path))
        assert len(reloaded) == n_before

    def test_molecule_map_reads_without_a_yaml(
        self, info, tmp_path, monkeypatch
    ):
        """A molecule map must open like any other localization file when
        no sidecar .yaml is written ("Save metadata in .yaml" off).

        The metadata a molecule map inherits is whatever the upstream
        steps put there, and the in-GUI ones (picking, clustering, ...)
        record numpy values. YAML writes those, JSON does not: embedding
        them used to raise inside ``io.save_locs`` *after* the locs
        dataset was written and *before* the .yaml, leaving a molecule map
        that could not be opened at all.
        """
        from picasso import io

        monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: False)
        info = list(info) + [
            {
                "Generated by": "Picasso Render : Pick",
                "Pick Diameter": np.float32(1.5),
                "Number of picks": np.int64(2),
                "Picks": np.zeros((2, 2)),
            }
        ]
        locs = _rotated_cluster(30.0)
        mols, _, out_info = g5m.g5m(
            locs,
            info,
            min_locs=20,
            covariance_type="rotated",
            calibration=CALIB_ASTIG,
            asynch=False,
            postprocess=True,
        )
        path = tmp_path / "mols.hdf5"
        io.save_locs(str(path), mols, out_info)
        assert not (tmp_path / "mols.yaml").exists()

        reloaded, reloaded_info = io.load_locs(str(path))
        assert len(reloaded) == len(mols)
        # the G5M block survives verbatim, the numpy values as JSON
        assert reloaded_info[-1] == out_info[-1]
        assert reloaded_info[-2]["Number of picks"] == 2
        assert reloaded_info[-2]["Picks"] == [[0.0, 0.0], [0.0, 0.0]]


class TestSumG5Ms:
    def test_sum_two_2d_models(self, dbscan_locs):
        """``sum_G5Ms`` used to pass calibration= unconditionally, which
        G5M_2D.__init__ does not accept."""
        X = dbscan_locs[["x", "y"]].to_numpy()
        lp = dbscan_locs[["lpx", "lpy"]].mean(axis=1).to_numpy()
        fits = []
        for _ in range(2):
            fit = g5m.G5M_2D(
                n_components=1, min_locs=2, sigma_bounds=(0.8, 1.5)
            ).fit(X, lp=lp)
            assert fit is not None
            fits.append(fit)
        summed = g5m.sum_G5Ms(fits)
        assert summed.covariance_type == "spherical"
        assert len(summed.weights) == 2
        assert summed.weights.sum() == pytest.approx(1.0)


class TestCLIWiring:
    def test_dispatch_order_matches_signature(self):
        """``main()`` calls ``_g5m`` positionally, so a parameter added
        in the wrong place would silently bind to the wrong argument."""
        import inspect

        from picasso.__main__ import _g5m

        assert list(inspect.signature(_g5m).parameters) == [
            "files",
            "min_locs",
            "loc_prec_handle",
            "min_sigma",
            "max_sigma",
            "max_rounds",
            "bootstrap_sem",
            "calibration",
            "mode",
            "covariance_type",
            "postprocess",
            "max_locs",
            "asynch",
            "group_column",
        ]

    def test_parser_accepts_covariance_type(self, capsys):
        import sys

        from picasso.__main__ import main

        argv = sys.argv
        try:
            sys.argv = ["picasso", "g5m", "--help"]
            with pytest.raises(SystemExit):
                main()
        finally:
            sys.argv = argv
        help_text = capsys.readouterr().out
        assert "--covariance-type" in help_text
        for choice in ["auto", "spherical", "diagonal", "rotated"]:
            assert choice in help_text


class TestRenderDialog:
    """The molecule shape is resolved from the data, not chosen by the
    user, so the Render dialog must expose no control for it."""

    @staticmethod
    def _dialog(has_angle, is_3d=True, extra_info=None):
        import pandas as pd
        from PyQt6 import QtWidgets

        from picasso.gui.render import G5MDialog

        n = 10
        cols = {
            "x": np.zeros(n),
            "y": np.zeros(n),
            "lpx": np.ones(n),
            "lpy": np.ones(n),
        }
        if is_3d:
            cols["z"] = np.zeros(n)
            cols["lpz"] = np.ones(n)
        if has_angle:
            cols["angle"] = np.zeros(n)

        info = {"Pixelsize": 130.0, "Frames": 100}
        info.update(extra_info or {})

        class _View:
            locs = [pd.DataFrame(cols)]
            infos = [[info]]
            pixelsize = 130.0

        class _Window(QtWidgets.QMainWindow):
            view = _View()

        window = _Window()
        dialog = G5MDialog(window, 0)
        # keep the parent alive: Qt deletes the child dialog with it
        dialog._test_window = window
        return dialog

    @pytest.mark.parametrize("has_angle", [True, False])
    @pytest.mark.parametrize("is_3d", [True, False])
    def test_no_molecule_shape_control(self, qt_offscreen, has_angle, is_3d):
        from PyQt6 import QtWidgets

        dialog = self._dialog(has_angle=has_angle, is_3d=is_3d)
        assert not hasattr(dialog, "covariance_type")
        labels = [w.text() for w in dialog.findChildren(QtWidgets.QLabel)]
        assert not any("shape" in text.lower() for text in labels)

    def test_no_fit_mode_control(self, qt_offscreen):
        """Astigmatism vs. spline follows from the localizations, so the
        dialog must not offer a combo box to pick between them."""
        from PyQt6 import QtWidgets

        dialog = self._dialog(has_angle=False, is_3d=True)
        assert not hasattr(dialog, "z_mode")
        for combo in dialog.findChildren(QtWidgets.QComboBox):
            items = [combo.itemText(i) for i in range(combo.count())]
            assert not any("spline" in text.lower() for text in items)

    @pytest.mark.parametrize(
        "extra_info, expected",
        [
            ({"Fit method": "spline-mle"}, "spline"),
            ({"Spline calibration model": "spline-3d"}, "spline"),
            ({"Fit method": "lq-3d"}, "astigmatism"),
            ({}, "astigmatism"),
        ],
    )
    def test_fit_mode_detected_from_metadata(
        self, qt_offscreen, extra_info, expected
    ):
        dialog = self._dialog(
            has_angle=False, is_3d=True, extra_info=extra_info
        )
        assert dialog.spline_mode == (expected == "spline")
        # the calibration button is only meaningful for astigmatism
        assert dialog.load_calib_button.isHidden() == (expected == "spline")

    def test_max_locs_keyed_by_absolute_channel(self, qt_offscreen):
        """``View._g5m`` looks max_locs_per_cluster up by the absolute
        channel index. Building it as a list over only the processed
        channels raised IndexError whenever a single channel other than
        the first was run on multichannel data."""
        from picasso.gui.render import View

        class _Stub:
            locs = [None] * 3  # three loaded channels
            check_group = staticmethod(lambda i, col: True)
            check_max_locs = staticmethod(lambda i, col: 100 * (i + 1))
            _g5m_max_locs_per_channel = View._g5m_max_locs_per_channel

        stub = _Stub()

        # single channel, not the first one
        out = stub._g5m_max_locs_per_channel([2], "group")
        assert out[2] == 300  # would IndexError on a 1-element list

        # all channels
        out_all = stub._g5m_max_locs_per_channel(range(3), "group")
        assert out_all == {0: 100, 1: 200, 2: 300}

    def test_max_locs_returns_none_when_not_clustered(self, qt_offscreen):
        from picasso.gui.render import View

        class _Stub:
            locs = [None] * 2
            check_group = staticmethod(lambda i, col: False)
            check_max_locs = staticmethod(lambda i, col: 1)
            _g5m_max_locs_per_channel = View._g5m_max_locs_per_channel

        assert _Stub()._g5m_max_locs_per_channel([0], "group") is None

    def test_getparams_leaves_covariance_type_to_g5m(self):
        """``getParams`` must not pin covariance_type, so that
        ``View._g5m_in_channel_job`` falls back to "auto" and the model
        is resolved from the data."""
        import inspect

        from picasso.gui.render import G5MDialog, View

        # getParams must never write the key into params
        assert 'params["covariance_type"]' not in inspect.getsource(
            G5MDialog.getParams
        )
        # so View._g5m_in_channel_job always takes the "auto" fallback
        src = inspect.getsource(View._g5m_in_channel_job)
        assert 'params.get("covariance_type", "auto")' in src


class TestBootstrap:
    """``_bootstrap_sem`` refits the model on resampled data, so it must
    rebuild it with the *same* settings. Getting this wrong is either a
    crash (mode) or, worse, a silently biased SEM (covariance_type)."""

    def test_bootstrap_sem_works_in_spline_mode(self, dbscan_locs_3d, info):
        """``_bootstrap_sem`` used to rebuild ``G5M_3D`` without passing
        ``mode``, so spline data (calibration is None) tripped the
        astigmatism assertion in ``G5M_3D.__init__``."""
        mols, _, _ = g5m.g5m(
            dbscan_locs_3d,
            info,
            min_locs=5,
            bootstrap_check=True,
            mode="spline",
            calibration=None,
            asynch=False,
            postprocess=False,
        )
        assert len(mols) > 0
        # SEM must be a real, positive uncertainty
        for col in ["lpx", "lpy", "lpz"]:
            assert np.isfinite(mols[col]).all()
            assert (mols[col] > 0).all()

    @staticmethod
    def _fit_rotated(locs, pixelsize=130.0):
        X = locs[["x", "y", "z"]].to_numpy().astype(float)
        X[:, 2] /= pixelsize
        lp = locs[["lpx", "lpy", "lpz"]].to_numpy().astype(float)
        lp[:, 2] /= pixelsize
        angles = np.deg2rad(locs["angle"].to_numpy())
        fit = g5m.G5M_3D(
            n_components=1,
            min_locs=20,
            sigma_bounds=(0.3, 3.0),
            calibration=CALIB_ASTIG,
            covariance_type="rotated",
        ).fit(X, lp=lp, loc_prec_handle="local", angles=angles)
        assert fit is not None
        return fit

    def test_bootstrap_sem_rebuilds_the_same_model(self, monkeypatch):
        """The resampling model must carry over mode, covariance_type
        and the angles. Dropping covariance_type does not raise - it just
        bootstraps a different (diagonal) model - so assert on how it is
        constructed."""
        locs = _rotated_cluster(30.0)
        fit = self._fit_rotated(locs)

        seen = {}
        real_cls = g5m.G5M_3D

        class _Recording(real_cls):
            def __init__(self, *args, **kwargs):
                seen.update(kwargs)
                super().__init__(*args, **kwargs)

            def fit(self, X, lp, loc_prec_handle="local", angles=None):
                seen["angles_passed"] = angles
                return super().fit(X, lp, loc_prec_handle, angles)

        monkeypatch.setattr(g5m, "G5M_3D", _Recording)
        sem = g5m._bootstrap_sem(fit, locs, n_bootstraps=2)

        assert seen["covariance_type"] == "rotated"
        assert seen["mode"] == "astigmatism"
        assert seen["calibration"] is not None
        angles = seen["angles_passed"]
        assert angles is not None and len(angles) == len(locs)
        # radians, matching the m-step's expectation
        np.testing.assert_allclose(angles, np.deg2rad(30.0), atol=1e-6)

        assert sem.shape == (1, 3)
        assert np.isfinite(sem).all() and (sem > 0).all()

    def test_bootstrap_end_to_end_for_rotated(self, info):
        mols, _, out_info = g5m.g5m(
            _rotated_cluster(30.0),
            info,
            min_locs=20,
            bootstrap_check=True,
            covariance_type="rotated",
            calibration=CALIB_ASTIG,
            sigma_bounds=(0.3, 3.0),
            asynch=False,
            postprocess=False,
        )
        assert out_info[-1]["Covariance type"] == "rotated"
        assert len(mols) > 0
        for col in ["lpx", "lpy", "lpz"]:
            assert np.isfinite(mols[col]).all()
            assert (mols[col] > 0).all()


class TestPublishedModelsUnchanged:
    """The spherical 2D and diagonal 3D models are published; they must
    stay bit-for-bit identical as new covariance types are added."""

    def test_2d_spherical_golden(self, dbscan_locs, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs, info, min_locs=5, asynch=False, postprocess=False
        )
        assert len(mols) == GOLDEN_2D["n"]
        for col in ["x", "y", "fitted_sigma", "p_val"]:
            np.testing.assert_array_equal(
                mols[col].to_numpy(),
                np.array(GOLDEN_2D[col], dtype=np.float32),
                err_msg=f"published 2D spherical model moved in '{col}'",
            )

    def test_3d_diagonal_golden(self, dbscan_locs_3d, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs_3d,
            info,
            min_locs=5,
            calibration=CALIB_3D,
            asynch=False,
            postprocess=False,
        )
        assert len(mols) == GOLDEN_3D["n"]
        for col in [
            "x",
            "y",
            "z",
            "fitted_sigma_x",
            "fitted_sigma_y",
            "fitted_sigma_z",
            "p_val",
        ]:
            np.testing.assert_array_equal(
                mols[col].to_numpy(),
                np.array(GOLDEN_3D[col], dtype=np.float32),
                err_msg=f"published 3D diagonal model moved in '{col}'",
            )


# ---------------------------------------------------------------------------
# End-to-end coverage through the public ``g5m.g5m``
# ---------------------------------------------------------------------------


class TestG5M:
    def test_g5m_2d_with_bootstrap(self, dbscan_locs, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs, info, min_locs=5, bootstrap_check=True, asynch=False
        )
        assert "p_val" in mols.columns
        # p-values must be in [0, 1]
        assert (mols["p_val"] >= 0).all() and (mols["p_val"] <= 1).all()

    def test_g5m_2d_global_loc_prec(self, dbscan_locs, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs,
            info,
            min_locs=5,
            bootstrap_check=False,
            loc_prec_handle="abs",
            sigma_bounds=(1 / 130, 3 / 130),
        )
        assert "p_val" in mols.columns
        assert len(mols) > 0


class TestG5M3D:
    def test_g5m_3d_with_bootstrap(self, dbscan_locs_3d, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs_3d,
            info,
            min_locs=5,
            bootstrap_check=True,
            calibration=CALIB_3D,
            asynch=False,
        )
        assert len(mols) > 0
        assert "p_val" in mols.columns
        assert (mols["p_val"] >= 0).all() and (mols["p_val"] <= 1).all()

    def test_g5m_3d_global_loc_prec(self, dbscan_locs_3d, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs_3d,
            info,
            min_locs=5,
            bootstrap_check=False,
            calibration=CALIB_3D,
            loc_prec_handle="abs",
            sigma_bounds=(1 / 130, 3 / 130),
        )
        assert len(mols) > 0
        assert "p_val" in mols.columns

    def test_g5m_3d_spline_no_calibration(self, dbscan_locs_3d, info):
        # spline mode uses the plain diagonal 3D model and reads lpz
        # directly from the locs, so no calibration is required
        mols, _, out_info = g5m.g5m(
            dbscan_locs_3d,
            info,
            min_locs=5,
            bootstrap_check=False,
            mode="spline",
            calibration=None,
            asynch=False,
        )
        assert len(mols) > 0
        assert "z" in mols.columns
        assert "p_val" in mols.columns
        assert (mols["p_val"] >= 0).all() and (mols["p_val"] <= 1).all()
        # info records the fit mode but no astigmatism coefficients
        assert out_info[-1]["Fit mode"] == "spline"
        assert "X Coefficients" not in out_info[-1]

    def test_g5m_3d_spline_requires_lpz(self, dbscan_locs_3d, info):
        locs_no_lpz = dbscan_locs_3d.drop(columns=["lpz"])
        with pytest.raises(ValueError):
            g5m.g5m(
                locs_no_lpz,
                info,
                min_locs=5,
                mode="spline",
                calibration=None,
                asynch=False,
            )

    def test_g5m_3d_astigmatism_requires_calibration(
        self, dbscan_locs_3d, info
    ):
        with pytest.raises(ValueError):
            g5m.g5m(
                dbscan_locs_3d,
                info,
                min_locs=5,
                mode="astigmatism",
                calibration=None,
                asynch=False,
            )


# ---------------------------------------------------------------------
# Progress reporting (uniform duck-typed interface, see
# lib.normalize_progress)
# ---------------------------------------------------------------------


class _RecordingProgress:
    """Duck-typed progress tracker recording the calls it receives."""

    def __init__(self):
        self.values = []
        self.maxima = []
        self.closed = False

    def set_value(self, value):
        self.values.append(value)

    def setMaximum(self, maximum, *args, **kwargs):
        self.maxima.append(maximum)

    def maximum(self):
        return self.maxima[-1] if self.maxima else 0

    def zero_progress(self, description=None, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        self.closed = True


class TestG5MProgress:
    @pytest.mark.parametrize("callback_parent", [None, "console"])
    def test_progress_modes(self, dbscan_locs, info, callback_parent):
        mols, _, _ = g5m.g5m(
            dbscan_locs,
            info,
            min_locs=5,
            asynch=False,
            callback_parent=callback_parent,
        )
        assert len(mols) > 0

    def test_reports_progress_to_duck_typed_tracker(self, dbscan_locs, info):
        tracker = _RecordingProgress()
        g5m.g5m(
            dbscan_locs,
            info,
            min_locs=5,
            asynch=False,
            callback_parent=tracker,
        )
        assert tracker.values, "no progress was reported"
        assert tracker.closed, "progress tracker was not closed"
        # the final update fills the bar
        assert tracker.values[-1] == max(tracker.values)
