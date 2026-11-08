"""Picasso: Localize with the "Overlapping emitters (3D-DAOSTORM)" model,
which runs ``localize.localize_multi_emitter``: the settings it shows and
hides, the fit it runs without an identification, the live preview and
the settings it keeps.

The fit thread runs for real on a tiny memory-mapped movie and is waited
for, as in ``test_gui_localize_dialogs``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from picasso import io
from picasso.gui import localize as gui_localize
from tests.test_gui_localize_calibration import _load, _Messages
from tests.test_gui_localize_dialogs import _memmap, _status, _wait
from tests.test_multi_emitter import _emitters, _render, _spline_calibration

PIXELSIZE = 130.0
FRAMES = 12
SIZE = 48


def _movie() -> np.ndarray:
    """Four single emitters and two pairs (2.5 PSF sigma apart) per
    frame, on a baseline of 100 counts."""
    rows = []
    for frame in range(FRAMES):
        for x, y in ((8, 8), (8, 38), (38, 8), (38, 38)):
            rows.append((frame, x, y, 3000))
        for x, y in ((20, 24), (28, 24)):
            rows += [(frame, x - 1.25, y, 3000), (frame, x + 1.25, y, 3000)]
    return _render(
        _emitters(rows), FRAMES, size=SIZE, background=20.0, baseline=100.0
    )


@pytest.fixture
def window(qt_offscreen):
    window = gui_localize.Window()
    yield window
    _wait(window)
    window.close()


@pytest.fixture
def loaded(window, tmp_path):
    _load(window, [_memmap(_movie(), tmp_path / "movie.raw")], tmp_path)
    dialog = window.parameters_dialog
    dialog.baseline.setValue(100.0)
    dialog.sensitivity.setValue(1.0)
    dialog.gain.setValue(1)
    dialog.pixelsize.setValue(PIXELSIZE)
    return window


def _select(window) -> None:
    dialog = window.parameters_dialog
    dialog.fit_model.setCurrentText(gui_localize.MULTI_EMITTER_MODEL)


def test_the_model_shows_only_what_it_uses(window):
    dialog = window.parameters_dialog
    assert dialog.multi_emitter_groupbox.isHidden()
    _select(window)
    assert not dialog.multi_emitter_groupbox.isHidden()
    # it finds the spots itself, in 2D, by maximum likelihood on the CPU
    assert dialog.identification_settings.isHidden()
    assert dialog.z_groupbox.isHidden()
    assert [
        dialog.fit_optimizer.itemText(i)
        for i in range(dialog.fit_optimizer.count())
    ] == ["MLE"]
    assert not dialog.gpu_checkbox.isEnabled()
    # the preview, the ROIs and the frame range still apply
    assert not dialog.preview_checkbox.isHidden()
    assert not dialog.frames_edit.isHidden()
    dialog.fit_model.setCurrentIndex(0)
    assert dialog.multi_emitter_groupbox.isHidden()
    assert not dialog.identification_settings.isHidden()


def test_use_gpu_is_hidden(window, monkeypatch):
    """The multi-emitter fit runs on the CPU only."""
    monkeypatch.setattr(gui_localize, "CUDA_AVAILABLE", True)
    dialog = window.parameters_dialog
    dialog.fit_model.setCurrentIndex(0)
    _select(window)
    assert dialog.gpu_checkbox.isHidden()
    dialog.fit_model.setCurrentIndex(0)
    assert not dialog.gpu_checkbox.isHidden()


def test_the_preview_waits_for_the_typed_value(window):
    """Typing 10 does not fit the preview with 1 first."""
    dialog = window.parameters_dialog
    for spinbox in (
        dialog.snr_threshold_spinbox,
        dialog.psf_sigma_spinbox,
        dialog.peak_radius_spinbox,
        dialog.background_sigma_spinbox,
        dialog.fit_region_spinbox,
        dialog.max_cycles_spinbox,
    ):
        assert not spinbox.keyboardTracking()


def test_the_psf_model_selects_the_fit_and_its_z(window):
    dialog = window.parameters_dialog
    _select(window)
    assert dialog.psf_model_combo.currentText() == "Fixed sigma"
    expected = {
        "Fixed sigma": ("fixed", True),
        "Fitted sigma": ("spherical", True),
        # z is read from the sigmas in x and y
        "Fitted sx and sy (astigmatic 3D)": ("elliptical", False),
        # gives z itself
        "Experimental PSF (cubic spline)": ("spline", True),
    }
    for name, (model, z_hidden) in expected.items():
        dialog.psf_model_combo.setCurrentText(name)
        assert dialog.multi_emitter_parameters().psf_model == model
        assert dialog.z_groupbox.isHidden() == z_hidden
    # the other models keep the box
    dialog.fit_model.setCurrentIndex(0)
    assert not dialog.z_groupbox.isHidden()


def test_the_spline_takes_its_psf_from_the_calibration(window):
    """The spline shows the calibration loader and its fit region instead
    of the PSF sigma, and the peak search in z for a 3D calibration."""
    dialog = window.parameters_dialog
    _select(window)
    assert dialog.spline_groupbox.isHidden()
    assert not dialog.psf_sigma_spinbox.isHidden()
    assert dialog.fit_region_spinbox.isHidden()
    dialog.psf_model_combo.setCurrentText("Experimental PSF (cubic spline)")
    assert not dialog.spline_groupbox.isHidden()
    for widget in (
        dialog.psf_sigma_label,
        dialog.psf_sigma_spinbox,
        dialog.psf_sigma_estimate_checkbox,
    ):
        assert widget.isHidden()
    assert not dialog.fit_region_spinbox.isHidden()
    # "Auto": from the calibration
    assert dialog.fit_region_spinbox.text() == "Auto"
    assert dialog.multi_emitter_parameters().fit_region_size is None
    dialog.fit_region_spinbox.setValue(11)
    assert dialog.multi_emitter_parameters().fit_region_size == 11
    # no calibration, or a 2D one: no z to search
    assert dialog.peak_search_z_edit.isHidden()
    dialog.spline_calibration = _spline_calibration("spline-3d")
    dialog._update_calib_group_visibility()
    assert not dialog.peak_search_z_edit.isHidden()
    dialog.peak_search_z_edit.setText("-200, 0, 200")
    parameters = dialog.multi_emitter_parameters()
    assert parameters.psf_model == "spline"
    assert parameters.psf_sigma is None
    assert parameters.peak_search_z == (-200.0, 0.0, 200.0)
    dialog.psf_model_combo.setCurrentText("Fixed sigma")
    assert dialog.spline_groupbox.isHidden()
    assert dialog.peak_search_z_edit.isHidden()
    assert dialog.fit_region_spinbox.isHidden()
    assert not dialog.psf_sigma_spinbox.isHidden()
    # the Gaussians take their region from the sigma
    assert dialog.multi_emitter_parameters().fit_region_size is None


def test_the_spline_fit_gives_z(loaded, tmp_path, monkeypatch):
    messages = _Messages(monkeypatch)
    _select(loaded)
    dialog = loaded.parameters_dialog
    dialog.psf_model_combo.setCurrentText("Experimental PSF (cubic spline)")
    # without a calibration, the fit asks for one
    loaded.localize()
    assert messages.kinds == ["warning"]
    assert "spline PSF calibration" in messages.text("warning")
    dialog.spline_calibration = _spline_calibration("spline-3d")
    messages.shown.clear()
    loaded.localize()
    _wait(loaded)
    assert messages.shown == []
    locs, info = io.load_locs(str(tmp_path / "Channel 0_locs.hdf5"))
    assert {"z", "lpz"} <= set(locs.columns)
    assert "sx" not in locs.columns
    assert len(locs) >= 4 * FRAMES
    entry = next(e for e in info if e.get("Step") == "Fit multi-emitter")
    assert entry["PSF Model"] == "spline"
    assert entry["Spline calibration model"] == "spline-3d"


def test_the_spline_preview(loaded):
    _select(loaded)
    dialog = loaded.parameters_dialog
    dialog.psf_model_combo.setCurrentText("Experimental PSF (cubic spline)")
    dialog.preview_checkbox.setChecked(True)
    loaded.draw_frame()
    assert "load a spline calibration" in _status(loaded)
    dialog.spline_calibration = _spline_calibration("spline-2d")
    loaded.draw_frame()
    markers = [
        item
        for item in loaded.scene.items()
        if isinstance(item, gui_localize.FitMarker)
    ]
    assert len(markers) == 8
    assert _status(loaded) == "Preview: 8 emitters in this frame."


def test_the_dialog_leaves_no_gap(window):
    """The optimizer stack and the multichannel rows keep no space for
    the multi-emitter fit, and the dialog shrinks with its contents."""
    app = QtWidgets.QApplication.instance()

    def settle():
        for _ in range(5):
            app.processEvents()
            QtCore.QThread.msleep(10)

    dialog = window.parameters_dialog
    dialog.fit_model.setCurrentIndex(0)
    dialog.show()
    settle()
    contents = dialog.scroll_area.widget()
    gaussian = contents.sizeHint().height()
    dialog.resize(dialog.width(), 2000)
    settle()
    height = dialog.height()
    _select(window)
    settle()
    multi_emitter = contents.sizeHint().height()
    assert dialog.fit_stack.isHidden()
    for widget in (dialog.fit_mode_combo, dialog.identify_mode_combo):
        assert not widget.sizePolicy().retainSizeWhenHidden()
    # the identification settings, the convergence settings and the
    # reserved rows go; the overlapping-emitter settings come
    removed = (
        dialog.identification_settings.sizeHint().height()
        + dialog.fit_stack.sizeHint().height()
        - dialog.multi_emitter_groupbox.sizeHint().height()
    )
    assert gaussian - multi_emitter > removed
    assert dialog.height() == height - (gaussian - multi_emitter)
    dialog.fit_model.setCurrentIndex(0)
    settle()
    assert contents.sizeHint().height() == gaussian
    assert not dialog.fit_stack.isHidden()
    assert dialog.fit_mode_combo.sizePolicy().retainSizeWhenHidden()


def test_the_psf_sigma_is_typed_in_only_when_not_estimated(window):
    dialog = window.parameters_dialog
    assert dialog.psf_sigma_estimate_checkbox.isChecked()
    assert not dialog.psf_sigma_spinbox.isEnabled()
    assert dialog.multi_emitter_parameters().psf_sigma is None
    dialog.psf_sigma_estimate_checkbox.setChecked(False)
    dialog.psf_sigma_spinbox.setValue(150.0)
    assert dialog.multi_emitter_parameters().psf_sigma == 150.0
    # the preview uses the sigma shown, estimated or not
    dialog.psf_sigma_estimate_checkbox.setChecked(True)
    assert dialog.multi_emitter_parameters(estimate=False).psf_sigma == 150.0


def test_localize_fits_without_identifying(loaded, tmp_path, monkeypatch):
    messages = _Messages(monkeypatch)
    _select(loaded)
    loaded.parameters_dialog.snr_threshold_spinbox.setValue(8.0)
    loaded.localize()
    _wait(loaded)
    assert messages.shown == []
    locs, info = io.load_locs(str(tmp_path / "Channel 0_locs.hdf5"))
    # every emitter, the pairs resolved
    assert len(locs) == 8 * FRAMES
    assert {"significance", "n_emitters"} <= set(locs.columns)
    assert (locs["n_emitters"] == 2).sum() == 4 * FRAMES
    entry = next(e for e in info if e.get("Step") == "Fit multi-emitter")
    assert entry["SNR Threshold"] == 8.0
    assert entry["PSF Sigma Estimated"] is True
    assert entry["PSF Sigma (nm)"] == pytest.approx(PIXELSIZE, rel=0.05)
    # the estimate is shown, for the preview
    assert loaded.parameters_dialog.psf_sigma_spinbox.value() == (
        pytest.approx(entry["PSF Sigma (nm)"], abs=0.1)
    )
    assert loaded.identifications is None or not loaded.ready_for_fit


def test_the_estimated_psf_sigma_is_shown_during_the_fit(loaded, monkeypatch):
    _Messages(monkeypatch)
    shown = []
    _select(loaded)
    dialog = loaded.parameters_dialog
    dialog.psf_sigma_spinbox.setValue(200.0)
    original = loaded.status_bar.showMessage

    def show(message, *args):
        # the spin box as it stands when each message is shown
        shown.append((message, dialog.psf_sigma_spinbox.value()))
        original(message, *args)

    monkeypatch.setattr(loaded.status_bar, "showMessage", show)
    loaded.localize()
    _wait(loaded)
    fitting = [
        (message, value)
        for message, value in shown
        if message.startswith("Estimated PSF sigma: ")
    ]
    assert fitting
    # the spin box is set as soon as the estimate is known
    for message, value in fitting:
        assert value != 200.0
        assert message.startswith(f"Estimated PSF sigma: {value:.1f} nm.")
        assert "Fitting frames" in message


def test_the_elliptical_fit_gives_z(loaded, tmp_path, monkeypatch):
    from tests.conftest import CALIB_3D

    messages = _Messages(monkeypatch)
    _select(loaded)
    dialog = loaded.parameters_dialog
    dialog.psf_model_combo.setCurrentText("Fitted sx and sy (astigmatic 3D)")
    dialog.psf_sigma_estimate_checkbox.setChecked(False)
    dialog.psf_sigma_spinbox.setValue(PIXELSIZE)
    dialog.z_calibration = dict(CALIB_3D)
    dialog.fit_z_checkbox.setEnabled(True)
    dialog.fit_z_checkbox.setChecked(True)
    dialog.magnification_factor.setValue(0.5)
    loaded.localize()
    _wait(loaded)
    assert messages.shown == []
    locs, info = io.load_locs(str(tmp_path / "Channel 0_locs.hdf5"))
    assert {"z", "d_zcalib", "lpz"} <= set(locs.columns)
    # the singles at least; a fitted sigma takes the pairs, 2.5 sigma
    # apart, for single wide emitters
    assert len(locs) >= 4 * FRAMES
    entry = next(e for e in info if e.get("Step") == "Fit multi-emitter")
    assert entry["PSF Model"] == "elliptical"
    z_entry = next(e for e in info if e.get("Step") == "Fit 3D")
    # the magnification factor on the dialog
    assert z_entry["Magnification factor"] == 0.5


def test_identify_points_to_the_fit(loaded):
    _select(loaded)
    loaded.identify()
    assert "finds the spots itself" in _status(loaded)
    assert getattr(loaded, "identification_worker", None) is None


def test_the_preview_fits_the_current_frame(loaded):
    _select(loaded)
    dialog = loaded.parameters_dialog
    dialog.psf_sigma_spinbox.setValue(PIXELSIZE)
    dialog.preview_checkbox.setChecked(True)
    loaded.draw_frame()
    markers = [
        item
        for item in loaded.scene.items()
        if isinstance(item, gui_localize.FitMarker)
    ]
    assert len(markers) == 8
    colors = [item.childItems()[0].pen().color() for item in markers]
    overlapping = sum(
        color == gui_localize.OVERLAPPING_MARKER_COLOR for color in colors
    )
    assert overlapping == 4
    assert _status(loaded) == "Preview: 8 emitters in this frame."


def _markers(window) -> list:
    return [
        item
        for item in window.scene.items()
        if isinstance(item, gui_localize.FitMarker)
    ]


def test_the_preview_shows_only_its_own_fit(loaded):
    """Identified with another model, the identifications stay in memory
    but give way to the multi-emitter preview, and come back with the
    other model."""
    dialog = loaded.parameters_dialog
    loaded.identify()
    _wait(loaded)
    assert loaded.ready_for_fit
    n_boxes = len(loaded.scene.items())
    dialog.psf_sigma_spinbox.setValue(PIXELSIZE)
    dialog.preview_checkbox.setChecked(True)
    _select(loaded)
    assert loaded.ready_for_fit
    assert len(_markers(loaded)) == 8
    # no identification boxes beside the preview's markers
    assert not [
        item
        for item in loaded.scene.items()
        if isinstance(item, QtWidgets.QGraphicsRectItem)
        and item.parentItem() is None
        and item.rect().width() < SIZE
    ]
    dialog.fit_model.setCurrentIndex(0)
    assert loaded.ready_for_fit
    assert not _markers(loaded)
    assert len(loaded.scene.items()) == n_boxes


def test_the_preview_hides_the_localizations(loaded, monkeypatch):
    """A fit's localizations show, except under the preview."""
    _Messages(monkeypatch)
    _select(loaded)
    dialog = loaded.parameters_dialog
    dialog.snr_threshold_spinbox.setValue(8.0)
    loaded.localize()
    _wait(loaded)
    assert loaded.locs is not None
    assert len(_markers(loaded)) == 8
    dialog.preview_checkbox.setChecked(True)
    assert loaded.locs is not None
    # the preview's 8, not 8 more on top of the localizations
    assert len(_markers(loaded)) == 8
    assert _status(loaded).startswith("Preview:")
    dialog.preview_checkbox.setChecked(False)
    assert len(_markers(loaded)) == 8


def test_two_channels_are_refused(window, tmp_path, monkeypatch):
    messages = _Messages(monkeypatch)
    movie = _movie()
    _load(
        window,
        [
            _memmap(movie, tmp_path / "a.raw"),
            _memmap(movie, tmp_path / "b.raw"),
        ],
        tmp_path,
    )
    _select(window)
    window.fit()
    assert messages.kinds == ["warning"]
    assert "one channel" in messages.text("warning")


def test_the_settings_are_kept(qt_offscreen):
    first = gui_localize.Window()
    dialog = first.parameters_dialog
    dialog.psf_model_combo.setCurrentText("Fitted sigma")
    dialog.snr_threshold_spinbox.setValue(7.5)
    dialog.psf_sigma_estimate_checkbox.setChecked(False)
    dialog.psf_sigma_spinbox.setValue(142.0)
    dialog.max_cycles_spinbox.setValue(12)
    dialog.peak_radius_spinbox.setValue(4.0)
    dialog.background_sigma_spinbox.setValue(10.0)
    dialog.peak_search_z_edit.setText("-300, 0")
    dialog.fit_region_spinbox.setValue(9)
    first.close()
    second = gui_localize.Window()
    try:
        dialog = second.parameters_dialog
        assert dialog.multi_emitter_psf_model() == "spherical"
        assert dialog.snr_threshold_spinbox.value() == 7.5
        assert not dialog.psf_sigma_estimate_checkbox.isChecked()
        assert dialog.psf_sigma_spinbox.value() == 142.0
        assert dialog.max_cycles_spinbox.value() == 12
        assert dialog.peak_radius_spinbox.value() == 4.0
        assert dialog.background_sigma_spinbox.value() == 10.0
        assert dialog.peak_search_z() == (-300.0, 0.0)
        assert dialog.fit_region_spinbox.value() == 9
    finally:
        second.close()
