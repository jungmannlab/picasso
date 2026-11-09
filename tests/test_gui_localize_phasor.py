"""Picasso: Localize with the "Phasor (pSMLM)" model, which runs
``localize.fit(fitting_method="phasor")``: the settings it shows and
hides, the fit and its precision note, z with an astigmatic phasor
calibration, the calibrations of the wrong kind it refuses, the phasor
calibration action and the settings it keeps.

The fit thread runs for real on a tiny memory-mapped movie and is waited
for, as in ``test_gui_localize_dialogs``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from picasso import io, lib, zfit
from picasso.gui import localize as gui_localize
from tests.test_gui_localize_calibration import _load, _Messages
from tests.test_gui_localize_dialogs import _memmap, _status, _wait
from tests import test_phasor

PIXELSIZE = test_phasor.PIXELSIZE


@pytest.fixture
def window(qt_offscreen, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    window = gui_localize.Window()
    yield window
    _wait(window)
    window.close()


def _loaded(window, tmp_path, **kwargs):
    frames, _ = test_phasor._frames(np.random.default_rng(0), **kwargs)
    _load(window, [_memmap(frames, tmp_path / "movie.raw")], tmp_path)
    dialog = window.parameters_dialog
    dialog.baseline.setValue(0.0)
    dialog.sensitivity.setValue(1.0)
    dialog.gain.setValue(1)
    dialog.pixelsize.setValue(PIXELSIZE)
    dialog.mng_slider.setValue(1500)
    return window


@pytest.fixture
def loaded(window, tmp_path):
    return _loaded(window, tmp_path)


def _select(window, box=7) -> None:
    dialog = window.parameters_dialog
    dialog.fit_model.setCurrentText(gui_localize.PHASOR_MODEL)
    dialog.box_spinbox.setValue(box)


def _phasor_calibration(tmp_path) -> str:
    locs, info = test_phasor._calibration_locs(np.random.default_rng(1))
    path = str(tmp_path / "phasor_3d_calib.yaml")
    zfit.calibrate_z_phasor(
        locs, info, d=test_phasor.CAL_STEP, magnification_factor=1.0, path=path
    )
    return path


def _locs(tmp_path):
    return io.load_locs(str(tmp_path / "Channel 0_locs.hdf5"))


def test_the_model_shows_only_what_it_uses(window):
    dialog = window.parameters_dialog
    _select(window)
    # no optimizer, nothing to converge, CPU only
    assert dialog.fit_optimizer.isHidden()
    assert dialog.fit_stack.isHidden()
    assert not dialog.gpu_checkbox.isEnabled()
    # it fits identified spots, in 3D with its own calibration
    assert not dialog.identification_settings.isHidden()
    assert not dialog.z_groupbox.isHidden()
    assert dialog.spline_groupbox.isHidden()
    assert dialog.multi_emitter_groupbox.isHidden()
    assert dialog.current_fit_code() == "phasor"
    assert dialog.fit_model_docs_url().endswith(
        "localize/phasor.html#localize-phasor"
    )
    dialog.fit_model.setCurrentIndex(0)
    assert not dialog.fit_optimizer.isHidden()
    assert not dialog.fit_stack.isHidden()


def test_the_tooltip_says_where_the_precision_comes_from():
    tooltip = gui_localize.model_tooltip()
    assert gui_localize.PHASOR_MODEL in tooltip
    assert "error-propagation estimate" in tooltip


def test_localize_saves_phasor_localizations(loaded, tmp_path, monkeypatch):
    messages = _Messages(monkeypatch)
    _select(loaded)
    loaded.localize()
    _wait(loaded)
    assert messages.shown == []
    locs, info = _locs(tmp_path)
    assert len(locs) == test_phasor.FRAMES * len(test_phasor.SPOTS)
    assert {"phasor_magnitude_x", "phasor_magnitude_y"} <= set(locs.columns)
    assert "sx" not in locs.columns
    entry = next(e for e in info if e.get("Step") == "Fit 2D")
    assert entry["Fit method"] == "phasor"
    # labeled where the user looks: the status bar and the metadata
    assert "error-propagation estimate" in _status(loaded)
    assert "Localization precision" in entry


def test_3d_with_a_phasor_calibration(window, tmp_path, monkeypatch):
    messages = _Messages(monkeypatch)
    _loaded(window, tmp_path, z=100.0, photons=6000)
    _select(window, box=test_phasor.CAL_BOX)
    dialog = window.parameters_dialog
    dialog.update_z_calib(_phasor_calibration(tmp_path))
    assert dialog.fit_z_checkbox.isChecked()
    assert dialog.z_calib_label.text() == "phasor_3d_calib.yaml (phasor)"
    dialog.magnification_factor.setValue(0.8)
    window.localize()
    _wait(window)
    assert messages.shown == []
    locs, info = _locs(tmp_path)
    assert {"z", "lpz"} <= set(locs.columns)
    assert abs(locs["z"].mean() - 0.8 * 100) < 0.8 * 30
    entry = next(e for e in info if e.get("Step") == "Fit 3D phasor")
    assert entry["Magnification factor"] == 0.8
    # no Gaussian z fit ran on top
    assert not any(e.get("Step") == "Fit 3D" for e in info)


def test_a_gaussian_calibration_is_refused(loaded, tmp_path, monkeypatch):
    from tests.conftest import CALIB_3D

    path = str(tmp_path / "gauss_3d_calib.yaml")
    io.save_calibration(path, dict(CALIB_3D))
    _select(loaded)
    loaded.parameters_dialog.update_z_calib(path)
    messages = _Messages(monkeypatch)
    loaded.localize()
    _wait(loaded)
    assert messages.kinds == ["warning"]
    assert "Gaussian astigmatism calibration" in messages.text("warning")
    assert not os.path.exists(tmp_path / "Channel 0_locs.hdf5")


def test_the_gaussian_models_refuse_a_phasor_calibration(
    loaded, tmp_path, monkeypatch
):
    loaded.parameters_dialog.fit_model.setCurrentText(
        gui_localize.ASTIGMATISM_FIT_MODEL
    )
    loaded.parameters_dialog.box_spinbox.setValue(test_phasor.CAL_BOX)
    loaded.parameters_dialog.update_z_calib(_phasor_calibration(tmp_path))
    messages = _Messages(monkeypatch)
    loaded.localize()
    _wait(loaded)
    assert messages.kinds == ["warning"]
    assert gui_localize.PHASOR_MODEL in messages.text("warning")


def test_the_phasor_calibration_needs_the_phasor_model(loaded, monkeypatch):
    messages = _Messages(monkeypatch)
    loaded.parameters_dialog.fit_model.setCurrentIndex(0)
    loaded.calibrate_z_phasor()
    assert messages.kinds == ["warning"]
    assert gui_localize.PHASOR_MODEL in messages.text("warning")


def test_the_calibration_is_built_from_the_phasor_fit(
    window, tmp_path, monkeypatch
):
    locs, info = test_phasor._calibration_locs(np.random.default_rng(2))
    # where the window draws them
    for column in ("x", "y"):
        locs[column] = np.float32(16 * PIXELSIZE)
    for column in ("lpx", "lpy"):
        locs[column] = np.float32(1.0)
    _load(
        window,
        [np.zeros((test_phasor.CAL_STEPS, 32, 32), np.uint16)],
        tmp_path,
        names=["beads"],
    )
    path = str(tmp_path / "beads_phasor_3d_calib.yaml")
    offered = []
    monkeypatch.setattr(
        gui_localize.Calibrate3DDialog,
        "getCalibrationSpecs",
        staticmethod(
            lambda parent=None: (test_phasor.CAL_STEP, 1, "fov", 1, True)
        ),
    )

    def save_dialog(parent, title, default, **kwargs):
        offered.append(default)
        return path, "*.yaml"

    monkeypatch.setattr(
        gui_localize.lib, "get_save_filename_ext_dialog", save_dialog
    )
    info = lib.append_metadata(window.info, [{**info[0], **info[1]}])
    window.on_fit_finished(locs, info, 1.0, False, True)
    assert offered[0].endswith("_phasor_3d_calib.yaml")
    calibration = io.load_calibration(path)
    assert zfit._is_phasor_calibration(calibration)
    assert calibration["Fit box size"] == test_phasor.CAL_BOX
    assert "3D calibrated" in _status(window)


def test_the_model_is_kept(qt_offscreen):
    first = gui_localize.Window()
    first.parameters_dialog.fit_model.setCurrentText(gui_localize.PHASOR_MODEL)
    first.close()
    second = gui_localize.Window()
    try:
        assert (
            second.parameters_dialog.fit_model.currentText()
            == gui_localize.PHASOR_MODEL
        )
    finally:
        second.close()
