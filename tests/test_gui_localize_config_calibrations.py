"""Localize's auto-loading of the calibrations stored in the camera
configuration.

The configuration can hold a z (astigmatism), a spline PSF and an sCMOS
camera calibration per camera and emission wavelength, and Localize picks
them up whenever the camera or the wavelength changes. The switch has to
work in both directions: moving to a camera or wavelength the config knows
nothing about must *clear* the calibration, not leave the previous
camera's one in place, or the fit would silently keep using a PSF or a
sensor map that no longer describes the data.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import types

import pytest

from picasso.gui import localize as gui_localize


CONFIG = {
    "z-calibrations": {
        "CamA": {525: "/calib/z_a_525.yaml", 595: "/calib/z_a_595.yaml"},
        "CamB": {525: "/calib/z_b_525.yaml"},
    },
    "spline-calibrations": {
        "CamA": {525: "/calib/psf_a_525.hdf5", 595: "/calib/psf_a_595.hdf5"},
        "CamB": {525: "/calib/psf_b_525.hdf5"},
    },
    "camera-calibrations": {
        "CamA": {525: "/calib/scmos_a_525.hdf5"},
        # a single path in place of the wavelength mapping serves every
        # wavelength
        "CamB": "/calib/scmos_b.hdf5",
    },
}

#: The three sections, with the loader that reads each one and the method
#: it hands the path to.
SECTIONS = [
    ("z-calibrations", "update_z_calib_with_config_path", "update_z_calib"),
    (
        "spline-calibrations",
        "update_spline_calib_with_config_path",
        "update_spline_calib",
    ),
    (
        "camera-calibrations",
        "update_camera_calib_with_config_path",
        "update_camera_calib",
    ),
]


class _ComboStub:
    """A ``QComboBox`` as the loaders use it: only its current text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def currentText(self) -> str:
        return self.text


class _CheckboxStub:
    def __init__(self) -> None:
        self.checked = False

    def isChecked(self) -> bool:
        return self.checked

    def setChecked(self, checked: bool) -> None:
        self.checked = bool(checked)


def _dialog(camera: str, wavelength: str = "525"):
    """The parts of ``ParametersDialog`` the config auto-load touches.

    The three ``update_*_calib`` methods are recorded rather than run: what
    is under test is which path (or ``None``, meaning clear) each one is
    handed when the camera or wavelength changes.
    """
    dialog = types.SimpleNamespace(
        camera=_ComboStub(camera),
        emission_combos={
            "CamA": _ComboStub(wavelength),
            "CamB": _ComboStub(wavelength),
        },
        spline_groupbox=object(),  # the fit UI is built
        fit_z_checkbox=_CheckboxStub(),
        calls={},
    )
    for _, _, setter in SECTIONS:
        dialog.calls[setter] = []
        setattr(
            dialog,
            setter,
            lambda path, setter=setter: dialog.calls[setter].append(path),
        )
    for _, loader, _ in SECTIONS:
        setattr(
            dialog,
            loader,
            getattr(gui_localize.ParametersDialog, loader).__get__(dialog),
        )
    setattr(
        dialog,
        "config_calib_path",
        gui_localize.ParametersDialog.config_calib_path.__get__(dialog),
    )
    return dialog


@pytest.fixture
def config(monkeypatch):
    """Localize's module-level CONFIG, holding the test configuration."""
    monkeypatch.setattr(gui_localize, "CONFIG", CONFIG)
    return CONFIG


class TestConfigCalibPath:
    def test_the_configured_path_is_found(self, config):
        dialog = _dialog("CamA", "595")

        path = dialog.config_calib_path("spline-calibrations")

        assert path == "/calib/psf_a_595.hdf5"

    def test_an_unconfigured_camera_has_no_path(self, config):
        dialog = _dialog("CamC")

        assert dialog.config_calib_path("spline-calibrations") is None

    def test_an_unconfigured_wavelength_has_no_path(self, config):
        dialog = _dialog("CamB", "595")

        assert dialog.config_calib_path("spline-calibrations") is None

    def test_a_single_path_serves_every_wavelength(self, config):
        for wavelength in ("525", "595"):
            dialog = _dialog("CamB", wavelength)

            path = dialog.config_calib_path("camera-calibrations")

            assert path == "/calib/scmos_b.hdf5"

    def test_a_missing_section_has_no_path(self, monkeypatch):
        monkeypatch.setattr(gui_localize, "CONFIG", {})
        dialog = _dialog("CamA")

        assert dialog.config_calib_path("spline-calibrations") is None


@pytest.mark.parametrize("section,loader,setter", SECTIONS)
class TestAutoLoad:
    def test_the_configured_calibration_is_loaded(
        self, config, section, loader, setter
    ):
        dialog = _dialog("CamA")

        getattr(dialog, loader)()

        assert dialog.calls[setter] == [config[section]["CamA"][525]]

    def test_an_unconfigured_camera_clears_the_calibration(
        self, config, section, loader, setter
    ):
        """The reported bug: with a calibration loaded for one camera,
        switching to a camera the config has no entry for used to leave it
        in place."""
        dialog = _dialog("CamC")

        getattr(dialog, loader)()

        assert dialog.calls[setter] == [None]

    def test_an_unconfigured_wavelength_clears_the_calibration(
        self, config, section, loader, setter
    ):
        dialog = _dialog("CamB", "595")

        getattr(dialog, loader)()

        expected = [None]
        if section == "camera-calibrations":
            # CamB's entry is a single path, which serves 595 nm too
            expected = ["/calib/scmos_b.hdf5"]
        assert dialog.calls[setter] == expected

    def test_a_missing_section_leaves_the_calibration_alone(
        self, monkeypatch, section, loader, setter
    ):
        """With nothing configured for any camera nothing was auto-loaded,
        so a manually loaded calibration must survive a camera change."""
        monkeypatch.setattr(gui_localize, "CONFIG", {})
        dialog = _dialog("CamA")

        getattr(dialog, loader)()

        assert dialog.calls[setter] == []


class TestFitZCheckbox:
    def test_loading_a_z_calibration_does_not_turn_on_the_z_fit(self, config):
        """``update_z_calib`` checks 'Fit Z' when it loads a calibration; a
        config auto-load must not start fitting z behind the user's back."""
        dialog = _dialog("CamA")
        dialog.fit_z_checkbox.setChecked(True)

        dialog.update_z_calib_with_config_path()

        assert not dialog.fit_z_checkbox.isChecked()

    def test_clearing_leaves_the_checkbox_to_update_z_calib(self, config):
        """Nothing is loaded, so the state of the checkbox is
        ``update_z_calib``'s business (it disables it) - the loader must not
        report a state it did not set."""
        dialog = _dialog("CamC")
        dialog.fit_z_checkbox.setChecked(True)

        dialog.update_z_calib_with_config_path()

        assert dialog.fit_z_checkbox.isChecked()
        assert dialog.calls["update_z_calib"] == [None]
