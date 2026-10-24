"""The camera quantum efficiency (QE) is ignored everywhere.

QE has not been used in the photon conversion since Picasso 0.6.0. Localize
no longer shows it, but a ``Quantum Efficiency`` entry in an existing camera
config, ``--qe`` on the command line and a ``Qe`` key in ``camera_info`` must
keep working (be ignored) instead of raising.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from types import SimpleNamespace

from picasso import __main__ as cli
from picasso import io, localize
from picasso.gui import localize as gui_localize


#: QE given in every shape older configs used, and in shapes that would
#: break a lookup by wavelength.
CONFIG = {
    "Cameras": {
        "Scalar QE": {
            "Baseline": 100,
            "Sensitivity": 0.5,
            "Quantum Efficiency": 0.8,
        },
        "Mapped QE": {
            "Baseline": 100,
            "Sensitivity": 0.5,
            # does not cover the channel device's 700 nm
            "Quantum Efficiency": {525: 0.7, 595: 0.72},
            "Channel Device": {
                "Name": "Turret-Label",
                "Emission Wavelengths": {"1-Blue": 525, "3-Red": 700},
            },
        },
        "No wavelengths": {"Baseline": 100, "Sensitivity": 0.5},
    },
    "spline-calibrations": {
        "Scalar QE": {488: "/calib/psf_488.hdf5"},
    },
}


class TestCameraWavelengths:
    def test_qe_keys_are_not_wavelengths(self, monkeypatch):
        monkeypatch.setattr(gui_localize, "CONFIG", CONFIG)
        cam_config = CONFIG["Cameras"]["Mapped QE"]

        wavelengths = gui_localize._camera_wavelengths("Mapped QE", cam_config)

        assert wavelengths == ["525", "700"]

    def test_calibration_sections_give_wavelengths(self, monkeypatch):
        monkeypatch.setattr(gui_localize, "CONFIG", CONFIG)
        cam_config = CONFIG["Cameras"]["Scalar QE"]

        wavelengths = gui_localize._camera_wavelengths("Scalar QE", cam_config)

        assert wavelengths == ["488"]

    def test_sorted_numerically(self, monkeypatch):
        monkeypatch.setattr(gui_localize, "CONFIG", {})
        cam_config = {
            "Channel Device": {
                "Name": "Turret-Label",
                "Emission Wavelengths": {"a": 1000, "b": 525, "c": 700},
            },
        }

        wavelengths = gui_localize._camera_wavelengths("Cam", cam_config)

        assert wavelengths == ["525", "700", "1000"]


class TestLocalizeWindow:
    def test_qe_in_the_config_is_ignored(self, qt_offscreen, monkeypatch):
        monkeypatch.setattr(gui_localize, "CONFIG", CONFIG)
        window = gui_localize.Window()
        dialog = window.parameters_dialog

        assert not hasattr(dialog, "qe")
        assert set(dialog.emission_combos) == {"Mapped QE", "Scalar QE"}
        for index in range(dialog.camera.count()):
            dialog.camera.setCurrentIndex(index)
            assert "Qe" not in window.camera_info


class TestMovieCameraParameters:
    def test_qe_is_not_looked_up(self):
        movie = SimpleNamespace(
            meta={
                "Camera": "Mapped QE",
                "Micro-Manager Metadata": {"Turret-Label": "3-Red"},
            },
            path="m.ome.tif",
        )

        parameters = io.TiffMultiMap.camera_parameters(movie, CONFIG)

        assert parameters["wavelength"] == ["700"]
        assert "qe" not in parameters


class TestCommandLine:
    def test_qe_is_accepted_but_hidden(self):
        parser, _ = cli._build_parser()

        args = parser.parse_args(["localize", "movie.ome.tif", "--qe", "0.5"])

        assert args.qe == 0.5
        subparsers = next(
            action
            for action in parser._actions
            if hasattr(action, "choices") and action.choices
        )
        help_text = subparsers.choices["localize"].format_help()
        assert "--qe" not in help_text


class TestApi:
    def test_qe_is_not_read_from_the_info(self):
        info = [{"Baseline": 100, "Sensitivity": 0.5, "Gain": 1, "Qe": 0.8}]

        camera_info = localize._camera_info_from_info(info)

        assert camera_info == {"Baseline": 100, "Sensitivity": 0.5, "Gain": 1}

    def test_qe_does_not_change_the_photons(self):
        spots = [[[200.0, 300.0], [400.0, 500.0]]]
        camera_info = {"Baseline": 100, "Sensitivity": 0.5, "Gain": 1}

        without_qe = localize._to_photons(spots, camera_info)
        with_qe = localize._to_photons(spots, camera_info | {"Qe": 0.2})

        assert (without_qe == with_qe).all()
