"""Structure types, multi-round movies and the noise model calibration
of Picasso: Simulate.

The structure type (grid, circle or custom) fills in the docking strands
of the structure. A movie can be simulated over several rounds, which
are added into one movie saved after the last round. The noise model
calibration (offered in the advanced mode) evaluates the background of
.tif movies whose names give the imager concentration, integration time
and laser power, and fits the noise model to them.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import tifffile
from PyQt6 import QtWidgets

from picasso import io, lib, simulate
from picasso.gui import simulate as gui_simulate

Accepted = QtWidgets.QDialog.DialogCode.Accepted
Rejected = QtWidgets.QDialog.DialogCode.Rejected


@pytest.fixture
def messages(monkeypatch):
    """Message boxes shown, as ``(title, text)``."""
    shown = []

    def box(parent, title, text, *args, **kwargs):
        shown.append((title, text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind, staticmethod(box))
    return shown


@pytest.fixture
def window(qt_offscreen, restore_theme, messages):
    window = gui_simulate.Window()
    yield window
    window.close()


@pytest.fixture
def advanced_window(qt_offscreen, restore_theme, messages, monkeypatch):
    """A window in the advanced mode, which has the noise model."""
    monkeypatch.setattr(gui_simulate, "ADVANCEDMODE", 1)
    window = gui_simulate.Window()
    yield window
    window.close()


def _values(edit) -> list[float]:
    return [float(v) for v in edit.text().split(",")]


# -- structure types --------------------------------------------------


def test_circle_places_the_labels_on_a_circle(window):
    window.structurecombo.setCurrentIndex(1)

    assert window.structure1Edit.isHidden()
    assert window.structure2.text() == "Number of labels"
    assert window.structure3.text() == "Diameter"
    x = np.array(_values(window.structurexxEdit))
    y = np.array(_values(window.structureyyEdit))
    assert len(x) == len(y) == 12
    np.testing.assert_allclose(np.hypot(x, y), 50, atol=0.01)
    assert _values(window.structureexEdit) == [1] * 12
    assert _values(window.structure3DEdit) == [0] * 12
    assert window.exchangeroundsEdit.text() == "1"


@pytest.mark.parametrize("labels", [5, 93, 121])
def test_circle_has_as_many_labels_as_set(window, labels):
    window.structurecombo.setCurrentIndex(1)
    window.structure2Edit.setValue(labels)

    x = np.array(_values(window.structurexxEdit))
    y = np.array(_values(window.structureyyEdit))
    assert len(x) == labels
    # evenly spaced, none twice
    angles = np.sort(np.arctan2(y, x))
    np.testing.assert_allclose(np.diff(angles), 2 * np.pi / labels, atol=1e-3)


def test_circle_diameter_that_is_not_a_number_is_100_nm(window):
    window.structurecombo.setCurrentIndex(1)
    window.structure3Edit.setText("wide")
    window.changeStructDefinition()

    x = np.array(_values(window.structurexxEdit))
    y = np.array(_values(window.structureyyEdit))
    np.testing.assert_allclose(np.hypot(x, y), 50, atol=0.01)


def test_grid_has_the_columns_and_rows_set(window):
    window.structurecombo.setCurrentIndex(1)
    window.structurecombo.setCurrentIndex(0)

    assert window.structure1.text() == "Columns"
    assert window.structure2.text() == "Rows"
    assert not window.structure1Edit.isHidden()
    x = _values(window.structurexxEdit)
    y = _values(window.structureyyEdit)
    assert sorted(set(x)) == [0, 20, 40]
    assert sorted(set(y)) == [0, 20, 40, 60]
    assert len(set(zip(x, y))) == len(x) == 12
    # every docking strand has an exchange round and a z
    assert len(_values(window.structureexEdit)) == 12
    assert _values(window.structure3DEdit) == [0] * 12


def test_grid_spacing_that_is_not_a_number_is_1_nm(window):
    window.structure1Edit.setValue(2)
    window.structure2Edit.setValue(2)
    window.structure3Edit.setText("a,b")
    window.changeStructDefinition()

    assert _values(window.structurexxEdit) == [0, 0, 1, 1]
    assert _values(window.structureyyEdit) == [0, 1, 0, 1]


def test_custom_keeps_the_docking_strands(window):
    window.structurexxEdit.setText("0,15")
    window.structureyyEdit.setText("5,5")
    window.structureexEdit.setText("1,2")
    window.structure3DEdit.setText("0,0")

    window.structurecombo.setCurrentIndex(2)

    for widget in (
        window.structure1Edit,
        window.structure2Edit,
        window.structure3Edit,
    ):
        assert widget.isHidden()
    assert window.structurexxEdit.text() == "0,15"
    assert window.exchangeroundsEdit.text() == "1,2"


def test_first_structure_without_docking_strands_is_previewed_empty(
    window,
):
    """With a low incorporation, the first structure may have no
    docking strand incorporated."""
    np.random.seed(0)
    window.structurenoEdit.setValue(1)
    window.structurecombo.setCurrentIndex(2)  # Custom
    window.structurexxEdit.setText("0")
    window.structureyyEdit.setText("0")
    window.structureexEdit.setText("1")
    window.structure3DEdit.setText("0")

    window.structureIncorporationEdit.setValue(1)  # generates positions

    assert window.newstruct.shape[1] == 0
    xx, yy, exchange = window._structure_preview
    assert len(xx) == len(yy) == len(exchange) == 0


# -- multi-round movies -----------------------------------------------


@pytest.fixture
def short_movie(window, monkeypatch):
    """A one-structure, 20-frame simulation without the final noise, so
    that the saved movie is the sum of the rendered rounds; the
    rendered movies are recorded."""
    np.random.seed(0)
    window.framesEdit.setValue(20)
    window.structurenoEdit.setValue(1)
    window.structureIncorporationEdit.setValue(100)
    monkeypatch.setattr(simulate, "noisy_p", lambda movie, mu: movie)
    rendered = []
    render = window._render_movie_frames

    def record(*args, **kwargs):
        movie = render(*args, **kwargs)
        rendered.append(movie.copy())
        return movie

    monkeypatch.setattr(window, "_render_movie_frames", record)
    return rendered


def _save_to(monkeypatch, *paths) -> None:
    """Answer the save dialog with ``paths``, one per call; a call more
    fails the test."""
    answers = iter(paths)
    monkeypatch.setattr(
        lib,
        "get_save_filename_ext_dialog",
        lambda *args, **kwargs: (next(answers), ""),
    )


def _files(folder) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def test_rounds_are_added_into_one_movie(
    window, short_movie, monkeypatch, tmp_path
):
    folder = tmp_path / "out"
    folder.mkdir()
    window.conroundsEdit.setValue(2)
    window.exportkinetics.setChecked(True)
    _save_to(monkeypatch, str(folder / "sim.raw"))

    window.simulate()

    assert window.currentround == 1
    # the movie is saved after the last round
    assert _files(folder) == ["sim_1.yaml", "sim_1_kinetics.yaml"]
    status = window.statusBar().currentMessage()
    assert status.endswith(
        "Current round: 1 of 2. Please set and start next round."
    )

    window.simulate()  # asks for no file name

    assert window.currentround == 0
    assert _files(folder) == [
        "sim.raw",
        "sim.yaml",
        "sim_1.yaml",
        "sim_1_kinetics.yaml",
        "sim_2_kinetics.yaml",
    ]
    movie, info = io.load_movie(str(folder / "sim.raw"))
    np.testing.assert_array_equal(
        np.asarray(movie), simulate.check_type(short_movie[0] + short_movie[1])
    )
    assert (
        window.statusBar()
        .currentMessage()
        .startswith("All computations finished. Last file saved to: ")
    )


def _one_and_two_strands(window) -> None:
    """A custom structure of one docking strand in exchange round 1 and
    two in exchange round 2."""
    window.structurecombo.setCurrentIndex(2)  # Custom
    window.structurexxEdit.setText("0,20,40")
    window.structureyyEdit.setText("0,0,0")
    window.structureexEdit.setText("1,2,2")
    window.structure3DEdit.setText("0,0,0")
    window.generatePositions()


def test_concatenated_exchange_rounds_simulate_one_round_each(
    window, short_movie, monkeypatch, tmp_path
):
    folder = tmp_path / "out"
    folder.mkdir()
    _one_and_two_strands(window)
    window.concatExchangeEdit.setChecked(True)
    _save_to(monkeypatch, str(folder / "sim.raw"))

    window.simulate()
    window.simulate()

    assert window.currentround == 0
    assert _files(folder) == ["sim.raw", "sim.yaml", "sim_1.yaml"]
    # round 1 has the docking strand of exchange round 1, round 2 the
    # two of exchange round 2
    for name, n_sites in (("sim_1.yaml", 1), ("sim.yaml", 2)):
        (info,) = io.load_info(str(folder / name))
        events = info["Spotkinetics.ON_Events"].split(",")
        assert len(events) == n_sites


def test_each_exchange_round_gets_its_own_movie(
    window, short_movie, monkeypatch, tmp_path
):
    folder = tmp_path / "out"
    folder.mkdir()
    _one_and_two_strands(window)
    _save_to(monkeypatch, str(folder / "sim.raw"))

    window.simulate()

    assert window.currentround == 0
    assert _files(folder) == [
        "sim_1.raw",
        "sim_1.yaml",
        "sim_2.raw",
        "sim_2.yaml",
    ]
    for name, n_sites in (("sim_1.yaml", 1), ("sim_2.yaml", 2)):
        (info,) = io.load_info(str(folder / name))
        events = info["Spotkinetics.ON_Events"].split(",")
        assert len(events) == n_sites


def test_canceling_the_first_round_starts_nothing(
    window, short_movie, monkeypatch, tmp_path
):
    folder = tmp_path / "out"
    folder.mkdir()
    window.conroundsEdit.setValue(2)
    _save_to(monkeypatch, "")

    window.simulate()

    assert window.currentround == 0
    assert short_movie == []
    assert _files(folder) == []


# -- noise model calibration ------------------------------------------

#: name (imager concentration, integration time, laser power), and the
#: pixel values of its frames
TIFS = {
    "5nM_100ms_20mW_.tif": (190, 230),
    "10nM_100ms_40mW_.tif": (210, 250),
    "2nM_50ms_10mW_.tif": (200, 220),
}


@pytest.fixture
def tif_folder(tmp_path):
    folder = tmp_path / "tifs"
    folder.mkdir()
    for name, (low, high) in TIFS.items():
        movie = np.full((4, 6, 6), low, dtype=np.uint16)
        movie[:, :, 3:] = high
        tifffile.imwrite(str(folder / name), movie, photometric="minisblack")
    (folder / "notes.txt").write_text("not a movie")
    return folder


@pytest.fixture
def dialog(qt_offscreen, messages):
    dialog = gui_simulate.CalibrationDialog()
    yield dialog
    dialog.close()


def _choose_folder(monkeypatch, folder) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        staticmethod(lambda *args, **kwargs: str(folder)),
    )


def _answer(monkeypatch, *answers) -> None:
    """Answer the text input dialogs with ``(text, ok)`` pairs."""
    answers = iter(answers)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        staticmethod(lambda *args, **kwargs: next(answers)),
    )


def _rows(dialog) -> dict[str, list[str]]:
    """The table by file name: the texts of the other columns."""
    table = dialog.table
    rows = {}
    for row in range(table.rowCount()):
        cells = [table.item(row, c) for c in range(table.columnCount())]
        texts = ["" if c is None else c.text() for c in cells]
        rows[texts[0].split("/")[-1]] = texts[1:]
    return rows


def _evaluate(dialog, monkeypatch, folder, *answers) -> None:
    _choose_folder(monkeypatch, folder)
    dialog.loadTif()
    _answer(monkeypatch, *answers)
    dialog.evalTif()


def test_calibration_dialog_has_ok_and_cancel(dialog):
    Button = QtWidgets.QDialogButtonBox.StandardButton
    assert dialog.buttons.standardButtons() == Button.Ok | Button.Cancel
    assert dialog.table.rowCount() == 0


def test_loading_a_folder_lists_its_tifs(dialog, tif_folder, monkeypatch):
    _choose_folder(monkeypatch, tif_folder)
    dialog.loadTif()

    assert dialog.tifCounter == 3
    assert sorted(_rows(dialog)) == sorted(TIFS)
    assert dialog.table.columnCount() == 6


def test_canceling_the_folder_dialog_lists_nothing(dialog, monkeypatch):
    _choose_folder(monkeypatch, "")
    dialog.loadTif()
    assert dialog.table.rowCount() == 0


def test_evaluating_tifs_fills_in_the_table(dialog, tif_folder, monkeypatch):
    _evaluate(dialog, monkeypatch, tif_folder, ("200", True), ("2", True))

    rows = _rows(dialog)
    # pixels below the baseline count as negative photons
    conc, itime, power, mean, std = rows["5nM_100ms_20mW_.tif"]
    assert (conc, itime, power) == ("5", "100", "20")
    assert float(mean) == pytest.approx(20)  # (-10 * 2 + 30 * 2) / 2
    assert float(std) == pytest.approx(40)
    assert rows["2nM_50ms_10mW_.tif"][:3] == ["2", "50", "10"]
    assert dialog.pbar.value() == 100


def test_canceled_camera_settings_are_the_defaults(
    dialog, tif_folder, monkeypatch
):
    _evaluate(dialog, monkeypatch, tif_folder, ("", False), ("", False))

    mean = float(_rows(dialog)["10nM_100ms_40mW_.tif"][3])
    # baseline 200, sensitivity 1.47
    assert mean == pytest.approx(30 * 1.47)


def test_camera_settings_that_are_not_numbers_are_refused(
    dialog, messages, tif_folder, monkeypatch
):
    _evaluate(dialog, monkeypatch, tif_folder, ("high", True))

    ((title, text),) = messages
    assert "high" in text
    assert all(cells[3] == "" for cells in _rows(dialog).values())


def test_tifs_named_without_the_settings_are_left_blank(
    dialog, tif_folder, monkeypatch
):
    (tif_folder / "5nM_100ms_20mW_.tif").rename(tif_folder / "noise.tif")

    _evaluate(dialog, monkeypatch, tif_folder, ("200", True), ("1", True))

    conc, itime, power, mean, std = _rows(dialog)["noise.tif"]
    assert (conc, itime, power) == ("", "", "")
    assert float(mean) == pytest.approx(10)


def test_table_values(dialog, tif_folder, monkeypatch):
    _evaluate(dialog, monkeypatch, tif_folder, ("200", True), ("1", True))

    bg, bgstd, las, time, conc = dialog.evalTable()
    rows = _rows(dialog)
    names = [
        dialog.table.item(row, 0).text().split("/")[-1]
        for row in range(dialog.table.rowCount())
    ]
    assert conc == [float(rows[name][0]) for name in names]
    assert time == [float(rows[name][1]) for name in names]
    assert las == [float(rows[name][2]) for name in names]
    assert bg == pytest.approx([float(rows[name][3]) for name in names])
    assert bgstd == pytest.approx([float(rows[name][4]) for name in names])

    long, short = dialog.readoutTable()
    assert [short[row] for row in range(3)] == [
        rows[name][1] for name in names
    ]
    assert [long[row] for row in range(3)] == [rows[name][2] for name in names]
    assert short[3] == long[13] == "None"


def test_export_table(dialog, tif_folder, monkeypatch, tmp_path):
    _evaluate(dialog, monkeypatch, tif_folder, ("200", True), ("1", True))
    path = tmp_path / "calibration.csv"
    for answer in ("", str(path)):
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *args, a=answer, **kwargs: (a, "")),
        )
        dialog.exportTable()
        assert path.exists() == bool(answer)

    with open(path, newline="") as file:
        header, *rows = list(csv.reader(file))
    assert header[0] == "FileName"
    assert len(header) == 6
    assert {row[0].split("/")[-1]: row[1:] for row in rows} == _rows(dialog)


def test_ok_needs_three_evaluated_tifs(
    dialog, messages, tif_folder, monkeypatch
):
    _choose_folder(monkeypatch, tif_folder)
    dialog.loadTif()

    dialog.accept()  # not evaluated

    assert dialog.result() != Accepted
    assert len(messages) == 1
    (tif_folder / "2nM_50ms_10mW_.tif").unlink()
    _evaluate(dialog, monkeypatch, tif_folder, ("200", True), ("1", True))

    dialog.accept()  # two tifs

    assert dialog.result() != Accepted
    assert len(messages) == 2


def test_canceled_calibration_dialog_returns_no_values(
    qt_offscreen, monkeypatch
):
    monkeypatch.setattr(
        gui_simulate.CalibrationDialog, "exec", lambda self: Rejected
    )
    assert gui_simulate.CalibrationDialog.setExt() == (
        [],
        [],
        [],
        [],
        [],
        False,
    )


def test_accepted_calibration_dialog_returns_the_table(
    qt_offscreen, tif_folder, monkeypatch
):
    _choose_folder(monkeypatch, tif_folder)
    _answer(monkeypatch, ("200", True), ("1", True))

    def evaluate(self):
        self.loadTif()
        self.evalTif()
        return Accepted

    monkeypatch.setattr(gui_simulate.CalibrationDialog, "exec", evaluate)
    *values, ok = gui_simulate.CalibrationDialog.setExt()

    assert ok
    bg, bgstd, las, time, conc = values
    assert sorted(conc) == [2, 5, 10]
    assert sorted(bg) == pytest.approx([10, 10, 30])


def _noise_model(window) -> list[float]:
    return [
        window.lasercEdit.value(),
        window.imagercEdit.value(),
        window.EquationAEdit.value(),
        window.EquationBEdit.value(),
        window.EquationCEdit.value(),
    ]


def test_noise_model_is_fitted_to_the_calibration(
    advanced_window, monkeypatch
):
    window = advanced_window
    conc = np.array([1.0, 2.0, 5.0, 10.0, 2.0, 5.0])
    las = np.array([10.0, 20.0, 20.0, 40.0, 40.0, 10.0])
    time = np.array([100.0, 50.0, 100.0, 100.0, 200.0, 50.0])
    bg = gui_simulate.fitFuncBg(np.array([conc, las, time]), 0.02, 0.004)
    bgstd = gui_simulate.fitFuncStd(
        np.array([las, time, bg]), 0.001, 0.3, 10.0
    )
    monkeypatch.setattr(
        gui_simulate.CalibrationDialog,
        "setExt",
        staticmethod(
            lambda parent=None: (
                list(bg),
                list(bgstd),
                list(las),
                list(time),
                list(conc),
                True,
            )
        ),
    )

    window.calibrateNoise()

    np.testing.assert_allclose(
        _noise_model(window), [0.02, 0.004, 0.001, 0.3, 10.0], atol=2e-6
    )
    plots = window._noise_model_window
    assert [a.get_title() for a in plots.figure.axes] == [
        "Background Model:",
        "Background Model Std:",
    ]
    plots.close()


def test_canceled_calibration_changes_nothing(advanced_window, monkeypatch):
    window = advanced_window
    monkeypatch.setattr(
        gui_simulate.CalibrationDialog,
        "setExt",
        staticmethod(lambda parent=None: ([], [], [], [], [], False)),
    )
    before = _noise_model(window)

    window.calibrateNoise()

    assert _noise_model(window) == before
    assert window._noise_model_window is None


@pytest.mark.parametrize("canceled", range(4))
def test_canceled_experiment_import_changes_nothing(
    advanced_window, monkeypatch, canceled
):
    """The import asks for the integration time, imager concentration,
    laser power and camera baseline, in this order."""
    window = advanced_window
    monkeypatch.setattr(io, "load_locs", lambda path: (None, [{}]))
    answers = [("100", True), ("5", True), ("50", True), ("100", True)]
    answers[canceled] = ("", False)
    _answer(monkeypatch, *answers[: canceled + 1])
    before = window.integrationtimeEdit.value()

    window.readhdf5("experiment.hdf5")

    assert window.integrationtimeEdit.value() == before
    assert window._experiment_window is None
