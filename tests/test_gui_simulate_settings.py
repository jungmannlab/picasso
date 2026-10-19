"""Loading the settings of a simulation in Picasso: Simulate.

``File > Load settings...`` reads the .yaml file written next to a
simulated movie. The files here are written by Simulate itself (a short
simulation), so the test follows what Simulate saves.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from PyQt6 import QtWidgets

from picasso import io, lib
from picasso.gui import simulate as gui_simulate


@pytest.fixture
def make_window(qt_offscreen, restore_theme):
    windows = []

    def make():
        window = gui_simulate.Window()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.close()


@pytest.fixture
def warnings(monkeypatch):
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, text: shown.append(text),
    )
    return shown


def _simulate(window, path, monkeypatch) -> None:
    monkeypatch.setattr(
        lib, "get_save_filename_ext_dialog", lambda *a, **k: (str(path), "")
    )
    window.simulate()


def _load(window, path, monkeypatch) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(path), ""),
    )
    window.loadSettings()


def _settings(window) -> dict:
    """The settings ``loadSettings`` restores, including the photon rate
    and its std that the simulation uses. The photon slope std is left
    out: outside the advanced mode, it is not used (the photon rate std
    follows from the photon slope)."""
    return {
        "frames": window.framesEdit.value(),
        "structure frame": window.structureframeEdit.value(),
        "structures": window.structurenoEdit.value(),
        "structure x": window.structurexxEdit.text(),
        "structure y": window.structureyyEdit.text(),
        "structure ex": window.structureexEdit.text(),
        "structure z": window.structure3DEdit.text(),
        "incorporation": window.structureIncorporationEdit.value(),
        "random arrangement": window.structurerandomEdit.isChecked(),
        "random orientation": (
            window.structurerandomOrientationEdit.isChecked()
        ),
        "k_on": window.konEdit.value(),
        "imager": window.imagerconcentrationEdit.value(),
        "taub": window.taubEdit.value(),
        "psf": window.psfEdit.value(),
        "photon rate": window.photonrateEdit.value(),
        "photon rate std": window.photonratestdEdit.value(),
        "constant photon rate std": window.photonslopemodeEdit.isChecked(),
        "photon budget": window.photonbudgetEdit.value(),
        "laser power": window.laserpowerEdit.value(),
        "photon slope": window.photonslopeEdit.value(),
        "background": window.backgroundlevelEdit.value(),
        "image size": window.camerasizeEdit.value(),
        "integration time": window.integrationtimeEdit.value(),
        "pixel size": window.pixelsizeEdit.value(),
    }


@pytest.fixture
def simulated(make_window, tmp_path, monkeypatch):
    """Settings and positions of a short simulation with non-default
    settings, and the path of its .yaml file."""
    window = make_window()
    window.framesEdit.setValue(20)
    window.structurenoEdit.setValue(2)
    window.structureframeEdit.setValue(5)
    window.structurexxEdit.setText("0,30,60")
    window.structureyyEdit.setText("0,10,0")
    window.structureexEdit.setText("1,1,2")
    window.structure3DEdit.setText("0,5,10")
    window.structureIncorporationEdit.setValue(90)
    window.structurerandomEdit.setChecked(True)
    window.photonslopemodeEdit.setChecked(
        not window.photonslopemodeEdit.isChecked()
    )
    window.backgroundlevelEdit.setValue(window.backgroundlevelEdit.value() + 5)
    window.camerasizeEdit.setValue(40)
    window.pixelsizeEdit.setValue(120)
    window.generatePositions()
    _simulate(window, tmp_path / "sim.raw", monkeypatch)
    return _settings(window), window.newstruct.copy(), tmp_path / "sim.yaml"


def test_settings_round_trip(make_window, simulated, monkeypatch, warnings):
    settings, positions, path = simulated
    window = make_window()
    assert _settings(window) != settings

    _load(window, path, monkeypatch)

    assert warnings == []
    assert _settings(window) == settings
    assert window.structurecombo.currentIndex() == 2  # Custom
    np.testing.assert_allclose(window.newstruct, positions)


def test_older_file_without_optional_settings(
    make_window, simulated, monkeypatch, warnings
):
    settings, positions, path = simulated
    info = io.load_info(str(path))
    for key in (
        "Structure.Structure3D",
        "Structure.3D",
        "Structure.CX (nm)",
        "Structure.CY (nm)",
        "Imager.Constant Photonrate Std",
        "Imager.BackgroundLevel",
    ):
        del info[0][key]
    with open(path, "w") as file:
        yaml.dump_all(info, file)
    window = make_window()
    before = _settings(window)

    _load(window, path, monkeypatch)

    assert warnings == []
    loaded = _settings(window)
    for name in ("constant photon rate std", "background"):
        assert loaded.pop(name) == before[name]
        settings.pop(name)
    # Without its z, the structure gets z = 0 when its x is set.
    loaded.pop("structure z")
    settings.pop("structure z")
    assert loaded == settings
    np.testing.assert_allclose(window.newstruct, positions)


@pytest.mark.parametrize(
    "content, message",
    [
        (
            {"Frames": 100, "Height": 32, "Width": 32},
            "does not contain the settings",
        ),
        ("", "does not contain the settings"),
        ("{not: valid", "Could not read"),
    ],
    ids=["movie metadata", "empty", "invalid yaml"],
)
def test_other_files_are_reported(
    make_window, tmp_path, monkeypatch, warnings, content, message
):
    path = tmp_path / "other.yaml"
    with open(path, "w") as file:
        if isinstance(content, dict):
            yaml.dump(content, file)
        else:
            file.write(content)
    window = make_window()
    before = _settings(window)
    positions = window.newstruct.copy()

    _load(window, path, monkeypatch)

    (warning,) = warnings
    assert message in warning
    assert _settings(window) == before
    np.testing.assert_array_equal(window.newstruct, positions)


def test_missing_settings_are_listed(
    make_window, simulated, monkeypatch, warnings
):
    _, _, path = simulated
    info = io.load_info(str(path))
    del info[0]["PAINT.k_on"]
    with open(path, "w") as file:
        yaml.dump_all(info, file)

    _load(make_window(), path, monkeypatch)

    (warning,) = warnings
    assert warning.endswith("Missing: PAINT.k_on.")


def test_unreadable_positions_change_nothing(
    make_window, simulated, monkeypatch, warnings
):
    _, _, path = simulated
    info = io.load_info(str(path))
    info[0]["Structure.HandleX"] = "1,2,a"
    with open(path, "w") as file:
        yaml.dump_all(info, file)
    window = make_window()
    before = _settings(window)

    _load(window, path, monkeypatch)

    (warning,) = warnings
    assert "docking strand positions" in warning
    assert _settings(window) == before


def test_calibration_of_older_file_is_converted_to_nm(
    make_window, simulated, monkeypatch, warnings
):
    """Before Picasso 1.0 the 3D calibration ("Structure.CX/CY") gave
    the spot widths in camera pixels; the file's camera pixel size
    converts it. New files record the widths in nm and the camera pixel
    size as ``Pixelsize``, where Localize looks for it."""
    _, _, path = simulated
    info = io.load_info(str(path))
    pixelsize = info[0]["Camera.Pixelsize"]
    assert info[0]["Pixelsize"] == pixelsize
    del info[0]["Structure.CX (nm)"]
    del info[0]["Structure.CY (nm)"]
    info[0]["Structure.CX"] = [1.0] * 7
    info[0]["Structure.CY"] = [2.0] * 7
    with open(path, "w") as file:
        yaml.dump_all(info, file)
    window = make_window()

    _load(window, path, monkeypatch)

    assert warnings == []
    assert window.cx == [pixelsize] * 7
    assert window.cy == [2 * pixelsize] * 7
