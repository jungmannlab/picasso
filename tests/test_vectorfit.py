"""Tests for picasso.vectorfit, the vectorial PSF fit with VectorFit.

The tests that run the fit itself need the optional ``vectorfit_py``
bindings (``pip install picassosr[vectorfit]``) and are skipped without them;
the settings, the conversion to localizations and the wiring into
``localize``, the command line and the GUI are tested either way.

:author: Thomas Bonte, 2026
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from picasso import io, lib, localize, vectorfit

requires_vectorfit = pytest.mark.skipif(
    not vectorfit.VECTORFIT_AVAILABLE,
    reason="the optional vectorfit_py bindings are not installed",
)

PIXELSIZE = 110.0
BOX = 7
CAMERA_INFO = {
    "Baseline": 100,
    "Sensitivity": 0.45,
    "Gain": 1,
    "Pixelsize": 130,
}


def _gaussian_spots(
    n_spots: int, seed: int = 0, photons: float = 3000.0, bg: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poisson-noisy Gaussian spots with random sub-pixel offsets; returns
    the spots and the true x and y offsets from the box center (pixels)."""
    rng = np.random.default_rng(seed)
    dx = rng.uniform(-0.5, 0.5, n_spots)
    dy = rng.uniform(-0.5, 0.5, n_spots)
    yy, xx = np.mgrid[:BOX, :BOX] - BOX // 2
    sigma = 1.0
    spots = np.empty((n_spots, BOX, BOX))
    for i in range(n_spots):
        g = np.exp(-((xx - dx[i]) ** 2 + (yy - dy[i]) ** 2) / (2 * sigma**2))
        spots[i] = rng.poisson(g / g.sum() * photons + bg)
    return spots, dx, dy


def _identifications(n_spots: int, frames=None) -> pd.DataFrame:
    """Box centers spread over a 64 x 64 frame."""
    rng = np.random.default_rng(1)
    if frames is None:
        frames = np.arange(n_spots)
    return pd.DataFrame(
        {
            "frame": np.asarray(frames, dtype=np.uint32),
            "x": rng.integers(BOX, 64 - BOX, n_spots).astype(np.int32),
            "y": rng.integers(BOX, 64 - BOX, n_spots).astype(np.int32),
            "net_gradient": rng.uniform(1e3, 1e4, n_spots).astype(np.float32),
        }
    )


def _identifications_in_movie(n_spots: int) -> pd.DataFrame:
    """Box centers that fit inside the bundled 32 x 32 test movie."""
    return pd.DataFrame(
        {
            "frame": np.arange(n_spots, dtype=np.uint32),
            "x": np.full(n_spots, 15, dtype=np.int32),
            "y": np.full(n_spots, 15, dtype=np.int32),
            "net_gradient": np.ones(n_spots, dtype=np.float32),
        }
    )


def _identification_parameters() -> localize.IdentificationParameters:
    return localize.IdentificationParameters(
        method=localize.NetGradient(5000), box=BOX
    )


def _fake_result(n_spots: int, is_3d: bool = True) -> dict:
    """A ``fit_spots`` result with distinguishable values per parameter."""
    n_parameters = 5 if is_3d else 4
    theta = np.zeros((n_spots, n_parameters))
    theta[:, 0] = 11.0  # y, nm
    theta[:, 1] = -22.0  # x, nm
    if is_3d:
        theta[:, 2] = 50.0  # z, nm (VectorFit's sign)
    theta[:, -2] = 1000.0  # photons
    theta[:, -1] = 5.0  # bg
    crlb = np.zeros_like(theta)
    crlb[:, 0] = 2.2  # y
    crlb[:, 1] = 4.4  # x
    if is_3d:
        crlb[:, 2] = 9.0  # z
    crlb[:, -2] = 30.0
    crlb[:, -1] = 0.5
    return {
        "theta": theta,
        "crlb": crlb,
        "iterations": np.full(n_spots, 7),
        "log_likelihood": np.full(n_spots, -20.0),
        "outlier": np.zeros(n_spots, dtype=bool),
    }


class _RecordingProgress(lib.MockProgress):
    """Remembers every value and phase it was given."""

    def __init__(self):
        super().__init__()
        self.values = []
        self.phases = []

    def set_value(self, value, *args, **kwargs):
        self.values.append(value)

    def zero_progress(self, description=None, unit=None):
        self.phases.append(description)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestParameters:
    def test_defaults_fill_missing_keys(self):
        parameters = vectorfit.parameters_with_defaults({"NA": 1.3})
        assert parameters["NA"] == 1.3
        for key, value in vectorfit.DEFAULT_PARAMETERS.items():
            if key != "NA":
                assert parameters[key] == value

    def test_none_gives_the_defaults(self):
        assert (
            vectorfit.parameters_with_defaults(None)
            == vectorfit.DEFAULT_PARAMETERS
        )

    def test_does_not_mutate_its_input(self):
        given = {"Wavelength": 600.0}
        vectorfit.parameters_with_defaults(given)
        assert given == {"Wavelength": 600.0}

    def test_unknown_key_is_rejected(self):
        """A typo must not silently fall back to the default."""
        with pytest.raises(ValueError, match="Wavelenght"):
            vectorfit.parameters_with_defaults({"Wavelenght": 600})

    def test_primary_aberrations_default_to_zero(self):
        for key in vectorfit._PRIMARY_ABERRATIONS:
            assert vectorfit.DEFAULT_PARAMETERS[key] == 0

    @pytest.mark.parametrize("key", list(vectorfit._PRIMARY_ABERRATIONS))
    def test_aberrations_can_be_negative(self, key):
        parameters = vectorfit.parameters_with_defaults({key: -25.0})
        assert parameters[key] == -25.0

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="Fit model"):
            vectorfit.parameters_with_defaults({"Fit model": "4D"})

    @pytest.mark.parametrize(
        "key", ["Wavelength", "NA", "Refractive index medium"]
    )
    def test_non_positive_value_is_rejected(self, key):
        with pytest.raises(ValueError, match=key):
            vectorfit.parameters_with_defaults({key: 0})

    @pytest.mark.parametrize("box", list(vectorfit.N_PUPIL))
    def test_supported_boxes_pass(self, box):
        vectorfit.check_box(box)

    @pytest.mark.parametrize("box", [3, 6, 11, 19])
    def test_unsupported_boxes_are_rejected(self, box):
        with pytest.raises(ValueError, match="box size"):
            vectorfit.check_box(box)

    @pytest.mark.parametrize("box, n_pupil", vectorfit.N_PUPIL.items())
    def test_pupil_sampling_suits_the_chirp_z_transform(self, box, n_pupil):
        """VectorFit's FFT needs ``box + n_pupil - 1`` to be a power of 2
        or three times one."""
        size = box + n_pupil - 1
        if size % 3 == 0:
            size //= 3
        assert size & (size - 1) == 0

    def test_missing_library_says_how_to_install_it(self, monkeypatch):
        def fail(name):
            raise ImportError(name)

        monkeypatch.setattr(vectorfit.importlib, "import_module", fail)
        with pytest.raises(ImportError, match=r"picassosr\[vectorfit\]"):
            vectorfit._import_vectorfit()

    def test_outliers_discarded_reads_the_metadata(self):
        key = vectorfit.OUTLIERS_KEY
        assert vectorfit.outliers_discarded({key: 5}) == 5
        # 0 is a count, not a missing entry (lib.get_from_metadata skips
        # falsy values)
        assert vectorfit.outliers_discarded([{}, {key: 0}]) == 0
        assert vectorfit.outliers_discarded([{key: 1}, {key: 2}]) == 2
        assert vectorfit.outliers_discarded([{"Fit method": "avg"}]) is None
        assert vectorfit.outliers_discarded({}) is None


# ---------------------------------------------------------------------------
# From fit results to localizations
# ---------------------------------------------------------------------------


class TestLocsFromFits:
    def test_columns_and_units_3d(self):
        """Every length in nm, as everywhere in Picasso 1.0."""
        identifications = _identifications(4)
        locs = vectorfit.locs_from_fits(
            identifications, _fake_result(4), pixelsize=PIXELSIZE, em=False
        )
        np.testing.assert_allclose(
            locs["x"], identifications["x"] * PIXELSIZE - 22.0, rtol=1e-6
        )
        np.testing.assert_allclose(
            locs["y"], identifications["y"] * PIXELSIZE + 11.0, rtol=1e-6
        )
        np.testing.assert_allclose(locs["lpx"], 4.4, rtol=1e-6)
        np.testing.assert_allclose(locs["lpy"], 2.2, rtol=1e-6)
        # z pointing the other way than VectorFit's
        np.testing.assert_allclose(locs["z"], -50.0)
        np.testing.assert_allclose(locs["lpz"], 9.0)
        np.testing.assert_allclose(locs["photons"], 1000.0)
        np.testing.assert_allclose(locs["bg"], 5.0)
        np.testing.assert_allclose(locs["photons_unc"], 30.0)
        np.testing.assert_allclose(locs["bg_unc"], 0.5)
        assert (locs["iterations"] == 7).all()
        np.testing.assert_allclose(locs["log_likelihood"], -20.0)
        np.testing.assert_allclose(
            locs["net_gradient"], identifications["net_gradient"]
        )
        # no Gaussian widths to report
        assert "sx" not in locs and "sy" not in locs

    def test_2d_has_no_z(self):
        locs = vectorfit.locs_from_fits(
            _identifications(3),
            _fake_result(3, is_3d=False),
            pixelsize=PIXELSIZE,
            em=False,
        )
        assert "z" not in locs and "lpz" not in locs
        np.testing.assert_allclose(locs["photons"], 1000.0)

    @pytest.mark.parametrize("pixelsize", [0, -1, np.nan])
    def test_pixelsize_must_be_positive(self, pixelsize):
        with pytest.raises(ValueError, match="pixel size"):
            vectorfit.locs_from_fits(
                _identifications(2),
                _fake_result(2),
                pixelsize=pixelsize,
                em=False,
            )

    def test_emccd_doubles_the_variances(self):
        identifications = _identifications(3)
        result = _fake_result(3)
        plain = vectorfit.locs_from_fits(
            identifications, result, pixelsize=PIXELSIZE, em=False
        )
        em = vectorfit.locs_from_fits(
            identifications, result, pixelsize=PIXELSIZE, em=True
        )
        for column in ("lpx", "lpy", "lpz", "photons_unc", "bg_unc"):
            np.testing.assert_allclose(
                em[column], np.sqrt(2) * plain[column], rtol=1e-6
            )
        np.testing.assert_allclose(em["x"], plain["x"])

    def test_outliers_are_discarded_only_when_asked(self):
        identifications = _identifications(5)
        result = _fake_result(5)
        result["outlier"][[1, 3]] = True
        kept = vectorfit.locs_from_fits(
            identifications,
            result,
            pixelsize=PIXELSIZE,
            em=False,
            discard_outliers=False,
        )
        dropped = vectorfit.locs_from_fits(
            identifications, result, pixelsize=PIXELSIZE, em=False
        )
        assert len(kept) == 5
        assert len(dropped) == 3
        assert sorted(dropped["frame"]) == [0, 2, 4]

    def test_sorted_by_frame(self):
        frames = [4, 1, 3, 0, 2]
        locs = vectorfit.locs_from_fits(
            _identifications(5, frames),
            _fake_result(5),
            pixelsize=PIXELSIZE,
            em=False,
        )
        assert list(locs["frame"]) == sorted(frames)
        assert list(locs.index) == list(range(5))

    def test_reduced_chi_square_with_spots(self):
        spots = np.ones((3, BOX, BOX))
        locs = vectorfit.locs_from_fits(
            _identifications(3),
            _fake_result(3),
            pixelsize=PIXELSIZE,
            em=False,
            spots=spots,
        )
        # -2 * log_likelihood / (n_pixels - n_parameters)
        np.testing.assert_allclose(
            locs["reduced_chi_square"], 40.0 / (BOX * BOX - 5), rtol=1e-6
        )

    def test_log_likelihood_is_minus_half_the_deviance(self):
        rng = np.random.default_rng(3)
        mu = rng.uniform(1, 20, (4, BOX, BOX))
        spots = rng.poisson(mu).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(spots > 0, spots * np.log(spots / mu), 0.0)
        deviance = 2 * np.sum(mu - spots + term, axis=(1, 2))
        np.testing.assert_allclose(
            vectorfit._log_likelihood(spots, mu), -deviance / 2
        )
        # a perfect model scores 0, the maximum
        np.testing.assert_allclose(
            vectorfit._log_likelihood(spots + 1, spots + 1), 0, atol=1e-9
        )


# ---------------------------------------------------------------------------
# Wiring into picasso.localize
# ---------------------------------------------------------------------------


class TestLocalizeWiring:
    def test_is_a_fit_method(self):
        assert "vectorfit" in localize.FIT_METHODS
        assert localize.parse_gauss_code("vectorfit") is None

    def test_calibration_3d_is_ignored_with_a_warning(self):
        """The 3D model fits z itself, as the spline fit does."""
        with pytest.warns(UserWarning, match="VectorFit"):
            assert not localize._validate_calibration_3d({}, "vectorfit")

    def test_unsupported_box_is_rejected_before_cutting(self, movie):
        with pytest.raises(ValueError, match="box size"):
            localize.fit(
                movie,
                camera_info=dict(CAMERA_INFO),
                identifications=_identifications(2),
                box=11,
                fitting_method="vectorfit",
            )

    def test_unknown_setting_is_rejected(self, movie):
        with pytest.raises(ValueError, match="Unknown VectorFit"):
            localize.fit(
                movie,
                camera_info=dict(CAMERA_INFO),
                identifications=_identifications(2),
                box=BOX,
                fitting_method="vectorfit",
                vectorfit_parameters={"Lambda": 600},
            )

    def test_metadata_records_the_settings_and_schedule(self):
        info = localize._fit_localize_info(
            "vectorfit",
            None,
            None,
            12,
            None,
            None,
            {"Fit model": "2D", "NA": 1.4},
        )
        assert info["Fit method"] == "vectorfit"
        assert info["Convergence criterion"] == vectorfit.TOLERANCE
        assert info["Max iterations"] == 12
        assert info["VectorFit Fit model"] == "2D"
        assert info["VectorFit NA"] == 1.4
        assert (
            info["VectorFit Wavelength"]
            == vectorfit.DEFAULT_PARAMETERS["Wavelength"]
        )

    def test_other_methods_record_no_vectorfit_settings(self):
        info = localize._fit_localize_info(
            "avg", None, None, None, None, None, {"NA": 1.4}
        )
        assert not any(key.startswith("VectorFit") for key in info)

    @pytest.mark.parametrize("n_discarded", [0, 3])
    def test_metadata_records_the_outliers_discarded(
        self, movie, monkeypatch, n_discarded
    ):
        """As Vermeer reports them: the spots the fit dropped as outliers."""
        identifications = _identifications_in_movie(8)
        given = {}

        def fake_fit(*, identifications, **kwargs):
            given.update(kwargs)
            return identifications.iloc[n_discarded:].reset_index(drop=True)

        monkeypatch.setattr(localize, "_fit2d_vectorfit", fake_fit)
        locs, info = localize.fit(
            movie,
            camera_info=dict(CAMERA_INFO),
            identifications=identifications,
            box=BOX,
            fitting_method="vectorfit",
        )
        assert len(locs) == 8 - n_discarded
        assert vectorfit.outliers_discarded(info) == n_discarded
        # what the VectorFit fit needs from the movie and the camera
        assert given["image_shape"] == tuple(movie.shape[1:3])
        assert given["pixelsize"] == CAMERA_INFO["Pixelsize"]
        assert given["parallel"] is True

    def test_aborted_fit_records_no_outliers(self, movie, monkeypatch):
        monkeypatch.setattr(
            localize, "_fit2d_vectorfit", lambda **kwargs: None
        )
        locs, info = localize.fit(
            movie,
            camera_info=dict(CAMERA_INFO),
            identifications=_identifications(0),
            box=BOX,
            fitting_method="vectorfit",
        )
        assert locs is None
        assert vectorfit.outliers_discarded(info) is None

    def test_other_methods_record_no_outliers(self, movie):
        _, info = localize.fit(
            movie,
            camera_info=dict(CAMERA_INFO),
            identifications=_identifications_in_movie(3),
            box=BOX,
            fitting_method="avg",
            parallel=False,
        )
        assert vectorfit.outliers_discarded(info) is None

    def test_fit_documents_every_setting_with_its_default(self):
        """``picasso.vectorfit`` is not public, so ``localize.fit`` names
        the settings and their defaults itself; pin them."""
        doc = " ".join(localize.fit.__doc__.split())
        for key, value in vectorfit.DEFAULT_PARAMETERS.items():
            assert f'"{key}"' in doc
            stated = f"{value:g}" if isinstance(value, float) else str(value)
            assert stated in doc, key

    def test_settings_reach_every_entry_point(self):
        """``vectorfit_parameters`` must be accepted (and forwarded) by
        every function that ends in ``fit``."""
        import inspect

        for function in (
            localize.fit,
            localize.localize,
            localize.localize_frames,
            localize.fit_independent,
            localize.fit_split_fov_independent,
        ):
            parameters = inspect.signature(function).parameters
            assert "vectorfit_parameters" in parameters, function.__name__
            assert "vectorfit_parameters=" in inspect.getsource(function)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


class TestCommandLine:
    def _parser(self):
        from picasso import __main__ as cli

        parser = argparse.ArgumentParser()
        cli._add_vectorfit_args(parser)
        return parser

    def test_is_a_fit_method(self):
        from picasso import __main__ as cli

        assert cli._FIT_METHOD_MAP["vectorfit"] == "vectorfit"

    def test_options_left_out_take_the_defaults(self):
        from picasso import __main__ as cli

        args = self._parser().parse_args([])
        assert cli._vectorfit_parameters_from_args(args) == {}

    def test_options_set_the_settings(self):
        from picasso import __main__ as cli

        args = self._parser().parse_args(
            [
                "--vf-model",
                "2D",
                "--vf-na",
                "1.27",
                "--vf-wavelength",
                "600",
                "--vf-vertical-astigmatism",
                "-20",
                "--vf-spherical-aberration",
                "15",
                "--vf-keep-outliers",
            ]
        )
        parameters = cli._vectorfit_parameters_from_args(args)
        assert parameters == {
            "Fit model": "2D",
            "NA": 1.27,
            "Wavelength": 600.0,
            "Vertical astigmatism": -20.0,
            "Spherical aberration": 15.0,
            "Discard outliers": False,
        }
        # every option maps onto a real setting
        vectorfit.parameters_with_defaults(parameters)
        for key in cli._VECTORFIT_ARGS.values():
            assert key in vectorfit.DEFAULT_PARAMETERS

    def test_help_states_the_real_defaults(self):
        """The help text repeats the defaults (the parser does not import
        picasso.vectorfit), so pin them to ``DEFAULT_PARAMETERS``."""
        from picasso import __main__ as cli

        flags = {
            f"--{name.replace('_', '-')}": key
            for name, key in cli._VECTORFIT_ARGS.items()
        }
        for action in self._parser()._actions:
            flag = next((_ for _ in action.option_strings if _ in flags), None)
            if flag is None:
                continue
            stated = re.search(r"default (\S+)\)", action.help).group(1)
            default = vectorfit.DEFAULT_PARAMETERS[flags[flag]]
            if isinstance(default, str):
                assert stated == default
            else:
                assert float(stated) == default


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


@pytest.mark.gui
class TestGui:
    def _dialog(self):
        from PyQt6 import QtWidgets
        from picasso.gui import localize as localize_gui

        class _StubWindow(QtWidgets.QMainWindow):
            movie = None

            def draw_frame(self):
                pass

        return localize_gui.ParametersDialog(_StubWindow())

    def test_model_is_offered_without_an_optimizer(self):
        from picasso.gui import localize as localize_gui

        entry = localize_gui.FIT_MODELS["Vectorial PSF (VectorFit)"]
        assert entry["optimizers"] is None
        assert entry["code"] == "vectorfit"
        assert localize_gui._CONVERGENCE_DEFAULTS["vectorfit"] == (
            vectorfit.TOLERANCE,
            vectorfit.MAX_ITERATIONS,
        )

    def test_model_tooltip_lists_it_on_its_own(self):
        """Each model gets its own paragraph in the tooltip."""
        from picasso.gui import localize as localize_gui

        assert "\n\nVectorial PSF (VectorFit):\n" in localize_gui.MODEL_TOOLTIP

    def test_selecting_it_shows_its_settings(self, qt_offscreen):
        from picasso.gui import localize as localize_gui

        dialog = self._dialog()
        try:
            dialog.show()
            dialog.fit_model.setCurrentIndex(
                dialog.fit_model.findText("Vectorial PSF (VectorFit)")
            )
            assert dialog.current_fit_code() == "vectorfit"
            assert dialog.vectorfit_groupbox.isVisible()
            # z comes from the model itself, not from an astigmatism
            # calibration, and there is no spline to load
            assert not dialog.z_groupbox.isVisible()
            assert not dialog.spline_groupbox.isVisible()
            assert not dialog.fit_optimizer.isVisible()
            # it iterates, so its convergence schedule is shown
            assert dialog.fit_stack.isVisible()
            assert dialog.fit_stack.currentIndex() == 1
            assert dialog.convergence_criterion.value() == pytest.approx(
                vectorfit.TOLERANCE
            )
            assert dialog.max_it.value() == vectorfit.MAX_ITERATIONS
            assert dialog.convergence_criterion.text() == "5e-10"

            dialog.fit_model.setCurrentIndex(
                dialog.fit_model.findText("Average of ROI")
            )
            assert not dialog.vectorfit_groupbox.isVisible()
            assert not dialog.fit_stack.isVisible()
            dialog.fit_model.setCurrentIndex(0)
            assert not dialog.vectorfit_groupbox.isVisible()
            assert dialog.z_groupbox.isVisible()
            # the other methods keep their own minimum and schedule
            code = dialog.current_fit_code()
            assert dialog.convergence_criterion.minimum() == pytest.approx(
                localize_gui._CONVERGENCE_MINIMUM
            )
            assert dialog.convergence_criterion.value() == pytest.approx(
                localize_gui._CONVERGENCE_DEFAULTS[code][0]
            )
        finally:
            dialog.deleteLater()

    def test_tolerance_box_takes_small_values(self, qt_offscreen):
        """The convergence box shows, takes and steps tolerances over many
        orders of magnitude."""
        from PyQt6 import QtGui

        from picasso.gui import localize as localize_gui

        box = localize_gui.ToleranceSpinBox()
        try:
            box.setRange(1e-12, 1e6)
            box.setValue(5e-10)
            assert box.value() == pytest.approx(5e-10)
            assert box.text() == "5e-10"
            box.setValue(0.01)
            assert box.text() == "0.01"
            assert box.valueFromText("1e-9") == pytest.approx(1e-9)
            assert box.valueFromText("0,5") == pytest.approx(0.5)
            State = QtGui.QValidator.State
            assert box.validate("1e-9", 0)[0] == State.Acceptable
            assert box.validate("1e-", 0)[0] == State.Intermediate
            assert box.validate("1e-15", 0)[0] == State.Intermediate
            assert box.validate("abc", 0)[0] == State.Invalid
            box.setValue(1e-6)
            box.stepBy(-1)
            assert box.value() == pytest.approx(1e-7)
            box.stepBy(2)
            assert box.value() == pytest.approx(1e-5)
        finally:
            box.deleteLater()

    def test_worker_forwards_the_settings(self, monkeypatch):
        """The worker hands the dialog's settings to the fit and passes the
        fit's metadata, the VectorFit settings and outliers included, on to
        the window, which saves it as it is."""
        import types

        from picasso.gui import localize as localize_gui

        identifications = _identifications(5)
        settings = {"Fit model": "2D", "NA": 1.3}
        given = {}
        fit_info = [{"Fit method": "vectorfit", vectorfit.OUTLIERS_KEY: 2}]

        def fake_fit(*args, **kwargs):
            given.update(kwargs)
            return identifications.iloc[2:], fit_info

        monkeypatch.setattr(localize_gui.localize, "fit", fake_fit)
        worker = localize_gui.FitWorker(
            None,
            [],
            {},
            identifications,
            BOX,
            "vectorfit",
            None,
            None,
            False,
            False,
            False,
            vectorfit_parameters=settings,
        )
        out = {}
        worker.finished = types.SimpleNamespace(
            emit=lambda locs, info, *a: out.update(locs=locs, info=info)
        )
        worker.run()
        assert given["vectorfit_parameters"] == settings
        assert given["fitting_method"] == "vectorfit"
        assert len(out["locs"]) == 3
        assert out["info"] is fit_info

    def test_status_bar_notes_the_outliers_discarded(self):
        from picasso.gui import localize as localize_gui

        note = localize_gui._outliers_note([{vectorfit.OUTLIERS_KEY: 4}])
        assert "4 outliers" in note
        assert "0 outliers" in localize_gui._outliers_note(
            [{vectorfit.OUTLIERS_KEY: 0}]
        )
        # any other fit adds no note
        assert localize_gui._outliers_note([{"Fit method": "avg"}]) == ""

    def test_saving_uses_the_fit_metadata(self):
        """``save_locs`` writes the fit's own metadata, which already holds
        the VectorFit settings and outliers; it must not rebuild them."""
        import inspect

        from picasso.gui import localize as localize_gui

        source = inspect.getsource(localize_gui.Window.save_locs)
        assert "io.save_locs(path, self.locs, self.extra_info)" in source
        assert "vectorfit" not in source

    def test_dialog_starts_with_the_defaults(self, qt_offscreen):
        dialog = self._dialog()
        try:
            assert (
                dialog.vectorfit_parameters() == vectorfit.DEFAULT_PARAMETERS
            )
        finally:
            dialog.deleteLater()

    def test_settings_round_trip(self, qt_offscreen):
        settings = {
            "Fit model": "2D",
            "Wavelength": 600.0,
            "NA": 1.27,
            "Refractive index medium": 1.33,
            "Refractive index cover slip": 1.52,
            "Refractive index immersion": 1.33,
            "Vertical astigmatism": -60.5,
            "Oblique astigmatism": 30.0,
            "Horizontal coma": -12.0,
            "Vertical coma": 8.25,
            "Spherical aberration": -40.0,
            "Discard outliers": False,
        }
        dialog = self._dialog()
        try:
            dialog.set_vectorfit_parameters(settings)
            assert dialog.vectorfit_parameters() == settings
            # a snapshot taken before VectorFit existed restores defaults
            dialog.set_vectorfit_parameters(None)
            assert (
                dialog.vectorfit_parameters() == vectorfit.DEFAULT_PARAMETERS
            )
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# The fit itself (needs vectorfit_py)
# ---------------------------------------------------------------------------


def _fit_spots(spots, identifications, **kwargs):
    """``vectorfit.fit_spots`` with the test's pixel size and frame."""
    kwargs.setdefault("parameters", {"Fit model": "2D"})
    kwargs.setdefault("n_threads", 2)
    return vectorfit.fit_spots(
        spots,
        identifications,
        pixelsize=PIXELSIZE,
        image_shape=(64, 64),
        **kwargs,
    )


@requires_vectorfit
class TestFit:
    def test_default_schedule_is_vermeers(self):
        """Vermeer's schedule, so that both fit the same; the library's
        own (1e-6, 30) stops small boxes after one iteration."""
        assert vectorfit.TOLERANCE == 5e-10
        assert vectorfit.MAX_ITERATIONS == 20
        params = vectorfit.fit_params(
            {}, box=BOX, pixelsize=PIXELSIZE, image_shape=(32, 32)
        )
        assert params.Tollim == vectorfit.TOLERANCE
        assert params.NiterMax == vectorfit.MAX_ITERATIONS

    def test_primary_aberrations_set_their_coefficients(self):
        """All primary aberrations are given in nm."""
        import vectorfit_py

        parameters = vectorfit.parameters_with_defaults(
            {
                "Wavelength": 600.0,
                "Vertical astigmatism": 50.0,
                "Oblique astigmatism": -25.0,
                "Horizontal coma": 10.0,
                "Vertical coma": -20.0,
                "Spherical aberration": 30.0,
            }
        )
        params = vectorfit.fit_params(
            parameters, box=BOX, pixelsize=PIXELSIZE, image_shape=(64, 64)
        )
        gammas, orders = vectorfit._fixed_aberrations(
            vectorfit_py, params, parameters
        )
        expected = np.zeros(vectorfit._N_LEGENDRE)
        expected[[4, 5, 9, 10, 12]] = [50.0, -25.0, 10.0, -20.0, 30.0]
        np.testing.assert_allclose(gammas, expected)
        assert orders[:, 2].any()

    def test_fit_params_carry_the_settings(self):
        params = vectorfit.fit_params(
            {"NA": 1.3, "Wavelength": 600},
            box=9,
            pixelsize=PIXELSIZE,
            image_shape=(40, 50),
            tolerance=1e-4,
            max_iterations=12,
            n_threads=2,
        )
        assert params.NA == pytest.approx(1.3)
        assert params.Lambda == pytest.approx(600)
        assert params.Mx == params.My == 9
        assert params.NPupil == vectorfit.N_PUPIL[9]
        assert (params.imgSizeY, params.imgSizeX) == (40, 50)
        assert params.PixelSizeX == params.PixelSizeY == PIXELSIZE
        assert params.Tollim == pytest.approx(1e-4)
        assert params.NiterMax == 12
        assert params.NumThreads == 2

    @pytest.mark.parametrize("model", ["2D", "3D"])
    def test_result_layout(self, model):
        n_spots = 20
        spots, _, _ = _gaussian_spots(n_spots)
        result = _fit_spots(
            spots, _identifications(n_spots), parameters={"Fit model": model}
        )
        n_parameters = 5 if model == "3D" else 4
        assert result["theta"].shape == (n_spots, n_parameters)
        assert result["crlb"].shape == (n_spots, n_parameters)
        assert result["iterations"].shape == (n_spots,)
        assert result["log_likelihood"].shape == (n_spots,)
        assert result["outlier"].dtype == bool
        assert (result["crlb"] > 0).all()
        assert (result["log_likelihood"] <= 0).all()

    def test_recovers_its_own_model(self):
        """Spots drawn from VectorFit's own model must be fitted back to the
        positions they were drawn at, with errors consistent with the
        Cramer-Rao bound."""
        import vectorfit_py

        n_spots = 300
        seed_spots, _, _ = _gaussian_spots(n_spots, photons=2000)
        identifications = _identifications(n_spots)
        parameters = vectorfit.parameters_with_defaults({"Fit model": "2D"})
        # one fit to get model images at known parameters
        params = vectorfit.fit_params(
            parameters,
            box=BOX,
            pixelsize=PIXELSIZE,
            image_shape=(64, 64),
            n_threads=2,
        )
        gammas, orders = vectorfit._fixed_aberrations(
            vectorfit_py, params, parameters
        )
        roi = np.stack(
            [identifications["y"] + 1.0, identifications["x"] + 1.0]
        )
        fitter = vectorfit_py.Fitter(
            vectorfit_py.doubleArray4D(seed_spots[..., None].copy()),
            vectorfit_py.doubleArray2D(np.ascontiguousarray(roi)),
            params,
            "cpu",
        )
        fitter.update_zernike_coefficients(
            vectorfit_py.doubleArray1D(gammas),
            vectorfit._aberrations(vectorfit_py, orders),
        )
        fitter.init()
        fitter.update()
        truth = np.array(fitter.theta())[:, :4]
        mu = np.array(fitter.mu())[..., 0]

        rng = np.random.default_rng(5)
        spots = rng.poisson(mu).astype(float)
        result = _fit_spots(spots, identifications, parameters=parameters)
        ok = ~result["outlier"]
        assert ok.mean() > 0.9
        error = (result["theta"][ok, :2] - truth[ok, :2]) / result["crlb"][
            ok, :2
        ]
        # unbiased, and scattered as the CRLB says
        assert np.all(np.abs(np.mean(error, axis=0)) < 0.25)
        assert np.all(np.abs(np.std(error, axis=0) - 1) < 0.25)

    def test_tracks_the_spot_position(self):
        """On Gaussian spots, the fitted offsets follow the true ones: x
        and y are not swapped and are measured from the box center."""
        n_spots = 100
        spots, dx, dy = _gaussian_spots(n_spots)
        identifications = _identifications(n_spots)
        result = _fit_spots(spots, identifications)
        locs = vectorfit.locs_from_fits(
            identifications,
            result,
            pixelsize=PIXELSIZE,
            em=False,
            discard_outliers=False,
        )
        # one spot per frame, so the frame order is the spot order; the
        # offsets from the box centers, back in camera pixels
        fit_dx = (
            locs["x"].to_numpy() / PIXELSIZE - identifications["x"].to_numpy()
        )
        fit_dy = (
            locs["y"].to_numpy() / PIXELSIZE - identifications["y"].to_numpy()
        )
        assert np.corrcoef(fit_dx, dx)[0, 1] > 0.95
        assert np.corrcoef(fit_dy, dy)[0, 1] > 0.95
        assert abs(np.median(fit_dx - dx)) < 0.05
        assert abs(np.median(fit_dy - dy)) < 0.05

    def test_progress_and_abort(self, monkeypatch):
        monkeypatch.setattr(vectorfit, "SPOTS_PER_BATCH", 4)
        n_spots = 10
        spots, _, _ = _gaussian_spots(n_spots)
        identifications = _identifications(n_spots)
        progress = _RecordingProgress()
        result = _fit_spots(
            spots, identifications, n_threads=1, progress=progress
        )
        assert progress.phases == ["Fitting spots"]
        assert progress.maximum() == n_spots
        # once per batch, ending at the maximum
        assert progress.values[:3] == [4, 8, 10]
        assert progress.values[-1] == n_spots
        assert len(result["theta"]) == n_spots

        calls = []

        def abort():
            calls.append(None)
            return len(calls) > 1  # after the first batch

        assert (
            _fit_spots(
                spots, identifications, n_threads=1, abort_callback=abort
            )
            is None
        )

    def test_batches_give_the_same_result(self, monkeypatch):
        n_spots = 12
        spots, _, _ = _gaussian_spots(n_spots)
        identifications = _identifications(n_spots)
        whole = _fit_spots(spots, identifications, n_threads=1)
        monkeypatch.setattr(vectorfit, "SPOTS_PER_BATCH", 5)
        batched = _fit_spots(spots, identifications, n_threads=1)
        np.testing.assert_allclose(batched["theta"], whole["theta"])
        np.testing.assert_allclose(batched["crlb"], whole["crlb"])

    def test_no_spots(self):
        result = _fit_spots(
            np.zeros((0, BOX, BOX)),
            _identifications(0),
            parameters={"Fit model": "3D"},
        )
        assert result["theta"].shape == (0, 5)
        locs = vectorfit.locs_from_fits(
            _identifications(0), result, pixelsize=PIXELSIZE, em=False
        )
        assert len(locs) == 0 and "z" in locs

    def test_localize_agrees_with_the_gaussian_fit(self, movie):
        """On the bundled movie, the vectorial and the Gaussian fit of the
        same identifications must land on the same positions."""
        gauss, _ = localize.localize(
            movie,
            camera_info=dict(CAMERA_INFO),
            identification_parameters=_identification_parameters(),
            fitting_method="gausslq",
        )
        vector, info = localize.localize(
            movie,
            camera_info=dict(CAMERA_INFO),
            identification_parameters=_identification_parameters(),
            fitting_method="vectorfit",
            vectorfit_parameters={"Discard outliers": False},
        )
        assert len(vector) == len(gauss)
        assert lib.get_from_metadata(info, "Fit method") == "vectorfit"
        assert lib.get_from_metadata(info, "VectorFit Fit model") == "2D"
        # all were kept
        assert vectorfit.outliers_discarded(info) == 0
        assert "z" not in vector
        merged = gauss.merge(
            vector, on=["frame", "net_gradient"], suffixes=("_g", "_v")
        )
        assert len(merged) == len(gauss)
        # within 0.05 camera pixels, in nm
        atol = 0.05 * CAMERA_INFO["Pixelsize"]
        np.testing.assert_allclose(merged["x_v"], merged["x_g"], atol=atol)
        np.testing.assert_allclose(merged["y_v"], merged["y_g"], atol=atol)

    def test_calibration_3d_is_not_applied(self, movie):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            locs, info = localize.localize(
                movie,
                camera_info=dict(CAMERA_INFO),
                identification_parameters=_identification_parameters(),
                fitting_method="vectorfit",
                vectorfit_parameters={"Fit model": "2D"},
                calibration_3d={"X Coefficients": [0] * 7},
            )
        assert "z" not in locs
        assert lib.get_from_metadata(info, "Z Calibration") is None

    def test_command_line(self, tmp_path, monkeypatch, capsys):
        from picasso import __main__ as cli

        for extension in (".raw", ".yaml"):
            shutil.copy(
                f"./tests/data/testdata{extension}",
                tmp_path / f"movie{extension}",
            )
        argv = [
            "picasso",
            "localize",
            str(tmp_path / "movie.raw"),
            "-b",
            str(BOX),
            "-g",
            "5000",
            "-a",
            "vectorfit",
            "-d",
            "0",
            "--pixelsize",
            str(CAMERA_INFO["Pixelsize"]),
            "--vf-model",
            "2D",
            "--vf-na",
            "1.4",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
        locs, info = io.load_locs(str(tmp_path / "movie_locs.hdf5"))
        assert len(locs)
        assert "z" not in locs
        assert lib.get_from_metadata(info, "Fit method") == "vectorfit"
        assert lib.get_from_metadata(info, "VectorFit Fit model") == "2D"
        assert lib.get_from_metadata(info, "VectorFit NA") == 1.4
        n_outliers = vectorfit.outliers_discarded(info)
        assert n_outliers >= 0
        # and the console says so, as Vermeer does
        printed = capsys.readouterr().out
        assert f"VectorFit discarded {n_outliers:,} outliers" in printed
