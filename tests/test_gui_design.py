"""The user flows of Picasso: Design.

The design canvas follows the windows' theme: the background and the
labels of the design are drawn in the colors of the windows, light or
dark (``picasso.gui.theme``), and change with them.

The flows are run as a user would, by clicking the hexagons of the
origami and the buttons of the window: coloring and clearing binding
sites, choosing the extensions per color, saving and loading designs,
exporting the plates for ordering, the pipetting scheme and the folding
table, and taking screenshots. Modal dialogs are answered by patches;
the refusals and canceled dialogs leave the design and the disk as they
were.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pytest
import yaml
from PyQt6 import QtCore, QtTest, QtWidgets

from picasso import design
from picasso.gui import design as gui_design
from picasso.gui import theme
from picasso.gui.theme import NEUTRALS

N_SITES = len(gui_design.ORIGAMI_SITES)  # 176 extendable staples
# the palette hexagons: colors 1 to 7, the gray "eraser" and the swatch
# showing the selected color
PALETTE = {i: gui_design.COLOR_SITES[8 - i] for i in range(1, 8)}
ERASER = gui_design.COLOR_SITES[0]
SWATCH = gui_design.COLOR_SITES[8]
# the first two paint handles offered in the extension table
HANDLES = gui_design.PAINT_SEQUENCES[1:3]
NONE_7 = ["None"] * 7
ACCEPTED = QtWidgets.QDialog.DialogCode.Accepted
REJECTED = QtWidgets.QDialog.DialogCode.Rejected


def _colors(window):
    scene = window.window.mainscene
    background = scene.backgroundBrush().color().name()
    labels = {
        item.defaultTextColor().name()
        for item in scene.items()
        if isinstance(item, QtWidgets.QGraphicsTextItem)
    }
    return background, labels


def test_design_canvas_follows_the_windows_theme(qt_offscreen, restore_theme):
    theme.apply(restore_theme, theme.Appearance(mode="Dark"))
    window = gui_design.MainWindow()
    assert _colors(window) == (
        NEUTRALS["Dark"]["base"],
        {NEUTRALS["Dark"]["text"]},
    )
    # an open window changes with the theme
    theme.apply(restore_theme, theme.Appearance(mode="Light"))
    assert _colors(window) == (
        NEUTRALS["Light"]["base"],
        {NEUTRALS["Light"]["text"]},
    )
    window.close()


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class Dialogs:
    """Answers to the modal dialogs and the messages they showed."""

    def __init__(self):
        self.messages = []  # (kind, text) of every message box
        self.open_path = ""
        self.save_paths = []  # answers of the save dialogs, in order
        self.save_filter = ""
        self.folder = ""
        self.question = QtWidgets.QMessageBox.StandardButton.Cancel
        self.directories = []  # start directories of the file dialogs

    def texts(self, kind=None):
        return [text for k, text in self.messages if kind in (None, k)]


@pytest.fixture
def dialogs(monkeypatch):
    """Patch every modal dialog Design opens; nothing waits for a user."""
    answers = Dialogs()
    box = QtWidgets.QMessageBox
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            box,
            kind,
            lambda *args, kind=kind, **kwargs: answers.messages.append(
                (kind, args[2])
            ),
        )

    def question(*args, **kwargs):
        answers.messages.append(("question", args[2]))
        return answers.question

    monkeypatch.setattr(box, "question", question)

    def get_open(parent, caption="", directory="", *args, **kwargs):
        answers.directories.append(directory)
        return answers.open_path, ""

    def get_save(parent, caption="", directory="", *args, **kwargs):
        answers.directories.append(directory)
        path = answers.save_paths.pop(0) if answers.save_paths else ""
        return path, answers.save_filter

    def get_folder(parent, caption="", directory="", *args, **kwargs):
        answers.directories.append(directory)
        return answers.folder

    file_dialog = QtWidgets.QFileDialog
    monkeypatch.setattr(file_dialog, "getOpenFileName", get_open)
    monkeypatch.setattr(file_dialog, "getSaveFileName", get_save)
    monkeypatch.setattr(file_dialog, "getExistingDirectory", get_folder)
    return answers


@pytest.fixture
def app(qt_offscreen, dialogs):
    """Design's main window, shown so that it can be clicked."""
    main = gui_design.MainWindow()
    main.show()
    qt_offscreen.processEvents()
    main.dialogs = dialogs
    # every message shown in the status bar, also those replaced since
    main.status = []
    main.window.statusBar().messageChanged.connect(main.status.append)
    yield main
    main.close()
    plt.close("all")


def _click(app, site):
    """Click the hexagon at ``site`` (row, column) like a user."""
    view = app.window.view
    point = view.mapFromScene(QtCore.QPointF(*gui_design.indextoHex(*site)))
    QtTest.QTest.mouseClick(
        view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point
    )


def _paint(app, color, sites):
    """Pick ``color`` (1 to 7) in the palette and click ``sites``."""
    _click(app, PALETTE[color])
    for site in sites:
        _click(app, site)


def _canvas(app):
    """Color index of every origami site, in ``ORIGAMI_SITES`` order."""
    return app.window.mainscene.evaluateCanvas()[::-1]


def _site_colors(app):
    """``{site: color}`` of the colored origami sites."""
    return {
        site: color
        for site, color in zip(gui_design.ORIGAMI_SITES, _canvas(app))
        if color
    }


def _fill(app, color):
    """Color every origami site, faster than clicking 176 hexagons."""
    scene = app.window.mainscene
    for item in scene.items()[-N_SITES:]:
        item.setBrush(gui_design.allcolors[color])
    scene.evaluateCanvas()


def _status(app):
    return app.window.statusBar().currentMessage()


def _answer_extensions(monkeypatch, choices, result=ACCEPTED, edits=None):
    """Answer the extension table: pick the paint handle ``choices[row]``
    (index in the table's combo box, or several in a row) per row, then
    type ``edits`` (``{(row, column): text}``) and press OK or Cancel."""
    seen = {}

    def exec_(dialog):
        seen["dialog"] = dialog
        for row, choice in enumerate(choices):
            for index in [choice] if isinstance(choice, int) else choice:
                dialog.table.cellWidget(row, 2).setCurrentIndex(index)
        for (row, column), text in (edits or {}).items():
            dialog.table.item(row, column).setText(text)
        return result

    monkeypatch.setattr(gui_design.SeqDialog, "exec", exec_)
    return seen


def _designed(app, monkeypatch):
    """Two sites of color 1 (5xR1) and one of color 3 (5xR2)."""
    _paint(app, 1, [(0, 0), (0, 1)])
    _paint(app, 3, [(5, 7)])
    _answer_extensions(monkeypatch, [1, 2])
    app.window.setSeq()


def _answer_plate_dialog(monkeypatch, selection, result=ACCEPTED):
    def exec_(dialog):
        if selection == 1:
            dialog.radio1.setChecked(True)
        elif selection == 2:
            dialog.radio2.setChecked(True)
        return result

    monkeypatch.setattr(gui_design.PlateDialog, "exec", exec_)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


def test_a_new_design_has_no_colored_sites(app):
    assert _site_colors(app) == {}
    assert app.window.mainscene.colorcounts == [0] * 7 + [N_SITES]
    assert app.window.mainscene.tableshort == NONE_7


def test_clicking_a_hexagon_colors_it_with_the_selected_color(app):
    _paint(app, 2, [(0, 0), (3, 4)])
    scene = app.window.mainscene
    assert _site_colors(app) == {(0, 0): 2, (3, 4): 2}
    assert scene.colorcounts == [0, 2, 0, 0, 0, 0, 0, N_SITES - 2]
    # the palette shows the selected color and the count per color
    swatch = scene.itemAt(
        QtCore.QPointF(*gui_design.indextoHex(*SWATCH)),
        app.window.view.transform(),
    )
    assert swatch.brush().color() == gui_design.allcolors[2]
    assert scene.alllbl[1].toPlainText() == "2"
    assert scene.alllbl[0].toPlainText().strip() == ""


def test_clicking_a_hexagon_again_with_its_color_uncolors_it(app):
    _paint(app, 1, [(0, 0), (0, 0)])
    assert _site_colors(app) == {}


def test_clicking_a_colored_hexagon_with_another_color_recolors_it(app):
    _paint(app, 1, [(0, 0)])
    _paint(app, 4, [(0, 0)])
    assert _site_colors(app) == {(0, 0): 4}


def test_the_gray_palette_hexagon_uncolors_sites(app):
    _paint(app, 1, [(0, 0), (1, 1)])
    _click(app, ERASER)
    _click(app, (0, 0))
    assert _site_colors(app) == {(1, 1): 1}


def test_clicking_the_swatch_or_the_background_changes_nothing(app):
    _paint(app, 5, [(0, 0)])
    _click(app, SWATCH)
    _click(app, (11, 30))  # empty canvas
    _click(app, (2, 2))  # a removed site, no hexagon
    assert _site_colors(app) == {(0, 0): 5}
    _click(app, (1, 1))  # still painting with color 5
    assert _site_colors(app) == {(0, 0): 5, (1, 1): 5}


def test_clear_canceled_keeps_the_design(app, monkeypatch):
    _designed(app, monkeypatch)
    app.dialogs.question = QtWidgets.QMessageBox.StandardButton.Cancel
    app.window.clearDialog()
    assert len(app.dialogs.texts("question")) == 1
    assert len(_site_colors(app)) == 3
    assert app.window.mainscene.tableshort[0] == HANDLES[0][0]


def test_clear_confirmed_removes_colors_and_extensions(app, monkeypatch):
    _designed(app, monkeypatch)
    app.dialogs.question = QtWidgets.QMessageBox.StandardButton.Discard
    app.window.clearDialog()
    scene = app.window.mainscene
    assert _site_colors(app) == {}
    assert scene.tableshort == NONE_7
    assert scene.tablelong == NONE_7
    # the counts of colors 1 to 7 are gone, all sites are uncolored
    assert all(scene.alllbl[i].toPlainText().strip() == "" for i in range(7))
    assert scene.alllbl[7].toPlainText() == str(N_SITES)
    assert all(
        lbl.toPlainText().strip() == "" for lbl in scene.alllblseq.values()
    )
    assert _status(app) == "Cleared."


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------


def test_extensions_without_colored_sites_are_refused(app, monkeypatch):
    seen = _answer_extensions(monkeypatch, [])
    app.window.setSeq()
    assert "dialog" not in seen
    assert "No hexagons marked" in _status(app)


def test_extensions_dialog_lists_the_used_colors(app, monkeypatch):
    _paint(app, 1, [(0, 0)])
    _paint(app, 3, [(5, 7)])
    seen = _answer_extensions(monkeypatch, [], result=REJECTED)
    app.window.setSeq()
    table = seen["dialog"].table
    assert table.rowCount() == 2
    assert [table.item(row, 0).text() for row in range(2)] == ["1", "3"]
    assert [table.item(row, 1).text() for row in range(2)] == [
        "Ext 1",
        "Ext 3",
    ]


def test_extensions_dialog_sets_the_chosen_handles(app, monkeypatch):
    _designed(app, monkeypatch)
    scene = app.window.mainscene
    assert scene.tableshort == [
        HANDLES[0][0],
        "None",
        HANDLES[1][0],
        "None",
        "None",
        "None",
        "None",
    ]
    assert scene.tablelong[0] == HANDLES[0][1]
    assert scene.tablelong[2] == HANDLES[1][1]
    assert scene.alllblseq[0].toPlainText() == HANDLES[0][0]
    assert scene.alllblseq[2].toPlainText() == HANDLES[1][0]
    assert _status(app) == "Extensions set."


def test_extensions_dialog_shows_the_extensions_already_set(app, monkeypatch):
    _designed(app, monkeypatch)
    seen = _answer_extensions(monkeypatch, [], result=REJECTED)
    app.window.setSeq()
    table = seen["dialog"].table
    assert table.item(0, 3).text() == HANDLES[0][0]
    assert table.item(1, 4).text() == HANDLES[1][1]


def test_extensions_dialog_canceled_keeps_the_extensions(app, monkeypatch):
    _designed(app, monkeypatch)
    _answer_extensions(monkeypatch, [0, 0], result=REJECTED)
    app.window.setSeq()
    assert app.window.mainscene.tableshort[0] == HANDLES[0][0]
    assert app.window.mainscene.tableshort[2] == HANDLES[1][0]


def test_handle_none_unsets_the_extension_of_a_color(app, monkeypatch):
    _designed(app, monkeypatch)
    _answer_extensions(monkeypatch, [(1, 0), 2])
    app.window.setSeq()
    assert app.window.mainscene.tableshort[:3] == [
        "None",
        "None",
        HANDLES[1][0],
    ]


def test_an_extension_typed_by_hand_is_set(app, monkeypatch):
    _paint(app, 2, [(0, 0)])
    _answer_extensions(monkeypatch, [], edits={(0, 3): "P1", (0, 4): "TTAT"})
    app.window.setSeq()
    assert app.window.mainscene.tableshort[1] == "P1"
    assert app.window.mainscene.tablelong[1] == "TTAT"


def test_an_extension_name_erased_by_hand_is_not_set(app, monkeypatch):
    """An empty cell counts as no extension for the color of its row,
    not for the color numbered like the row."""
    _paint(app, 1, [(0, 0)])
    _paint(app, 3, [(5, 7)])
    _answer_extensions(monkeypatch, [1, 2], edits={(1, 3): "", (1, 4): ""})
    app.window.setSeq()
    scene = app.window.mainscene
    assert scene.tableshort[:3] == [HANDLES[0][0], "None", "None"]
    assert scene.tablelong[:3] == [HANDLES[0][1], "None", "None"]
    assert app.window.checkSeq() == 1


def test_extensions_dialog_lists_a_color_on_every_site(app, monkeypatch):
    _fill(app, 1)
    seen = _answer_extensions(monkeypatch, [1])
    app.window.setSeq()
    assert seen["dialog"].table.rowCount() == 1
    assert app.window.mainscene.tableshort[0] == HANDLES[0][0]


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def test_design_round_trips_through_save_and_load(app, monkeypatch, tmp_path):
    _designed(app, monkeypatch)
    path = str(tmp_path / "origami.yaml")
    app.dialogs.save_paths = [path]
    app.window.saveDialog()
    assert _status(app) == "File saved as: " + path

    with open(path) as file:
        info = yaml.safe_load(file)
    structure = {(s[0], s[1]): s[2] for s in info["Structure"]}
    assert len(structure) == N_SITES
    assert structure[("A", 1)] == 1 and structure[("B", 1)] == 1
    assert sum(color != 0 for color in structure.values()) == 3
    assert info["Extensions Short"][0] == HANDLES[0][0]
    # the coordinates of the colored sites, for Picasso: Simulate
    assert info["Structure.StructureEx"] == "1, 1, 3"
    assert len(info["Structure.StructureX"].split(", ")) == 3
    assert len(info["Structure.StructureY"].split(", ")) == 3

    colors = _site_colors(app)
    scene = app.window.mainscene
    tableshort, tablelong = list(scene.tableshort), list(scene.tablelong)
    app.dialogs.question = QtWidgets.QMessageBox.StandardButton.Discard
    app.window.clearDialog()

    app.dialogs.open_path = path
    app.window.openDialog()
    assert _site_colors(app) == colors
    assert scene.tableshort == tableshort
    assert scene.tablelong == tablelong
    assert scene.alllblseq[2].toPlainText() == HANDLES[1][0]
    assert scene.alllbl[0].toPlainText() == "2"
    assert _status(app) == "File loaded from: " + path
    # the next dialog opens in the folder of the design
    assert app.dialogs.directories[-1] == str(tmp_path)
    app.window.saveDialog()
    assert app.dialogs.directories[-1] == str(tmp_path)


def test_design_saved_as_yml_is_saved_as_yaml(app, tmp_path):
    _paint(app, 1, [(0, 0)])
    app.dialogs.save_paths = [str(tmp_path / "origami.yml")]
    app.window.saveDialog()
    assert os.listdir(tmp_path).count("origami.yaml") == 1
    assert not (tmp_path / "origami.yml").exists()


def test_save_canceled_writes_nothing(app, tmp_path):
    _paint(app, 1, [(0, 0)])
    before = set(os.listdir(tmp_path))
    app.window.saveDialog()
    assert set(os.listdir(tmp_path)) == before
    assert _status(app) == "Filename not specified. Design not saved."


def test_load_canceled_keeps_the_design(app):
    _paint(app, 1, [(0, 0)])
    app.window.openDialog()
    assert _site_colors(app) == {(0, 0): 1}
    assert _status(app) == "Filename not specified. File not loaded."


def test_loading_a_file_that_is_no_design_is_refused(app, tmp_path):
    path = tmp_path / "other.yaml"
    path.write_text("Width: 32\nHeight: 32\n")
    _paint(app, 1, [(0, 0)])
    app.dialogs.open_path = str(path)
    app.window.openDialog()
    assert _site_colors(app) == {(0, 0): 1}
    assert "not recognized" in _status(app)


# ---------------------------------------------------------------------------
# Plates for ordering
# ---------------------------------------------------------------------------


def test_plates_without_extensions_are_refused(app, monkeypatch, tmp_path):
    _paint(app, 1, [(0, 0)])
    _answer_plate_dialog(monkeypatch, 1)
    app.dialogs.save_paths = [str(tmp_path / "plates.csv")]
    app.window.generatePlates()
    assert "No extensions have been set" in _status(app)
    assert not (tmp_path / "plates.csv").exists()


def test_plates_with_a_color_without_extension_are_refused(
    app, monkeypatch, tmp_path
):
    _designed(app, monkeypatch)
    _paint(app, 6, [(4, 4)])
    _answer_plate_dialog(monkeypatch, 1)
    app.dialogs.save_paths = [str(tmp_path / "plates.csv")]
    app.window.generatePlates()
    assert _status(app).startswith("Error: 1 color(s) do not have")
    assert not (tmp_path / "plates.csv").exists()


@pytest.mark.parametrize("column", [3, 4], ids=["name", "sequence"])
def test_plates_with_an_extension_name_or_sequence_only_are_refused(
    app, monkeypatch, tmp_path, column
):
    _paint(app, 1, [(0, 0)])
    _answer_extensions(monkeypatch, [], edits={(0, column): "P1"})
    app.window.setSeq()
    _answer_plate_dialog(monkeypatch, 1)
    app.dialogs.save_paths = [str(tmp_path / "plates.csv")]
    app.window.generatePlates()
    assert not (tmp_path / "plates.csv").exists()
    assert _status(app).startswith("Error")


@pytest.mark.parametrize(
    "selection, result",
    [(1, REJECTED), (0, ACCEPTED)],
    ids=["canceled", "nothing selected"],
)
def test_plate_dialog_canceled_writes_nothing(
    app, monkeypatch, tmp_path, selection, result
):
    _designed(app, monkeypatch)
    _answer_plate_dialog(monkeypatch, selection, result)
    app.dialogs.save_paths = [str(tmp_path / "plates.csv")]
    app.window.generatePlates()
    assert not (tmp_path / "plates.csv").exists()
    assert app.dialogs.save_paths  # no save dialog was opened


def test_plates_save_canceled_writes_nothing(app, monkeypatch, tmp_path):
    _designed(app, monkeypatch)
    _answer_plate_dialog(monkeypatch, 1)
    before = set(os.listdir(tmp_path))
    app.window.generatePlates()
    assert set(os.listdir(tmp_path)) == before
    assert _status(app) == "Filename not specified. Plates not saved."


def _export_plates(app, monkeypatch, path, selection=1):
    _answer_plate_dialog(monkeypatch, selection)
    app.dialogs.save_paths = [str(path)]
    app.window.generatePlates()
    return design.read_plate(str(path))


def test_plates_of_the_design_carry_the_extensions(app, monkeypatch, tmp_path):
    _designed(app, monkeypatch)
    rows = _export_plates(app, monkeypatch, tmp_path / "plates.csv")
    assert rows[0] == [
        "PLATE NAME",
        "PLATE POSITION",
        "OLIGO NAME",
        "SEQUENCE",
    ]
    assert len(rows) == 1 + 2 * 96
    assert {row[0] for row in rows[1:]} == {"CUSTOM_1", "CUSTOM_2"}
    staples = {row[2]: row[3] for row in rows[1:] if row[2].strip()}
    assert len(staples) == N_SITES
    base = {row[0]: row for row in gui_design.BASE_SEQUENCES[1:]}
    # A1 (site (0, 0)) carries 5xR1, its unmodified neighbor A2 does not
    a1 = base["A1"]
    assert staples[a1[1][:-3] + HANDLES[0][0]] == a1[2] + " " + HANDLES[0][1]
    assert base["A2"][1] in staples
    extended = [name for name in staples if not name.endswith("BLK")]
    assert len(extended) == 3
    assert _status(app) == "Plates saved to : " + str(tmp_path / "plates.csv")


def test_full_plates_are_exported_per_extension(app, monkeypatch, tmp_path):
    _designed(app, monkeypatch)
    rows = _export_plates(app, monkeypatch, tmp_path / "plates.csv", 2)
    names = {row[0] for row in rows if row[0] != "PLATE NAME"}
    assert names == {
        f"{plate}_{i}"
        for plate in ("BLK", HANDLES[0][0], HANDLES[1][0])
        for i in (1, 2)
    }
    r1 = [row for row in rows if row[0].startswith(HANDLES[0][0])]
    assert sum(row[2].endswith(HANDLES[0][0]) for row in r1) == N_SITES


# ---------------------------------------------------------------------------
# Pipetting scheme
# ---------------------------------------------------------------------------


def _answer_pipetting(monkeypatch, folder, result=ACCEPTED):
    """Select ``folder`` in the pipetting dialog and press OK/Cancel."""
    seen = {}

    def exec_(dialog):
        seen["dialog"] = dialog
        if folder is not None:
            dialog.loadButton.click()
        return result

    monkeypatch.setattr(gui_design.PipettingDialog, "exec", exec_)
    return seen


@pytest.fixture
def plate_folder(app, monkeypatch, tmp_path):
    """A folder with the plates of the design."""
    _designed(app, monkeypatch)
    folder = tmp_path / "plates"
    folder.mkdir()
    _export_plates(app, monkeypatch, folder / "plates.csv")
    app.dialogs.folder = str(folder)
    return folder


def test_pipetting_scheme_without_extensions_is_refused(app, monkeypatch):
    seen = _answer_pipetting(monkeypatch, None)
    _paint(app, 1, [(0, 0)])
    app.window.pipettingScheme()
    assert "dialog" not in seen
    assert "No extensions have been set" in _status(app)


def test_pipetting_scheme_is_saved_for_the_ordered_plates(
    app, monkeypatch, tmp_path, plate_folder
):
    seen = _answer_pipetting(monkeypatch, plate_folder)
    cwd = os.getcwd()
    pdf = tmp_path / "scheme.pdf"
    app.dialogs.save_paths = [str(pdf)]
    app.window.pipettingScheme()
    dialog = seen["dialog"]
    assert dialog.folderEdit.text() == str(plate_folder)
    assert dialog.csvCounter.text() == "A total of 1 *.csv files detected."
    assert "2  plates" in dialog.plateCounter.text()
    assert any("All sequences found in 2 Plates" in m for m in app.status)
    assert _status(app).endswith("scheme saved to: " + str(pdf))
    assert pdf.read_bytes().startswith(b"%PDF")
    # choosing the folder does not change the working directory
    assert os.getcwd() == cwd


def test_pipetting_scheme_reports_missing_sequences(
    app, monkeypatch, tmp_path, plate_folder
):
    path = plate_folder / "plates.csv"
    rows = design.read_plate(str(path))
    design.save_plate(str(path), [rows[:50]])
    _answer_pipetting(monkeypatch, plate_folder)
    pdf = tmp_path / "scheme.pdf"
    app.dialogs.save_paths = [str(pdf)]
    app.window.pipettingScheme()
    # a warning before saving, which the saved message does not replace
    (warning,) = app.dialogs.texts("warning")
    assert "sequences of the design were not found" in warning
    assert pdf.exists()
    assert _status(app) == (
        f"Pipetting scheme saved to: {pdf} "
        f"({warning.split(' of ')[0]} sequences missing)."
    )


def test_pipetting_scheme_folder_without_csv_is_refused(
    app, monkeypatch, tmp_path, plate_folder
):
    empty = tmp_path / "empty"
    empty.mkdir()
    app.dialogs.folder = str(empty)
    _answer_pipetting(monkeypatch, empty)
    app.dialogs.save_paths = [str(tmp_path / "scheme.pdf")]
    app.window.pipettingScheme()
    assert _status(app) == "No *.csv found. Scheme not created."
    assert not (tmp_path / "scheme.pdf").exists()


def test_pipetting_dialog_canceled_creates_no_scheme(
    app, monkeypatch, tmp_path, plate_folder
):
    _answer_pipetting(monkeypatch, plate_folder, result=REJECTED)
    app.dialogs.save_paths = [str(tmp_path / "scheme.pdf")]
    app.window.pipettingScheme()
    assert not (tmp_path / "scheme.pdf").exists()
    assert app.dialogs.save_paths  # no save dialog was opened
    assert _status(app) == "Canceled. Scheme not created."


def test_pipetting_dialog_lists_only_the_last_folder(
    app, monkeypatch, tmp_path, plate_folder
):
    seen = _answer_pipetting(monkeypatch, plate_folder)
    app.window.pipettingScheme()
    dialog = seen["dialog"]
    n_rows = len(dialog.getfulllist())
    other = tmp_path / "other"
    other.mkdir()
    rows = design.read_plate(str(plate_folder / "plates.csv"))
    design.save_plate(str(other / "few.csv"), [rows[:10]])
    app.dialogs.folder = str(other)
    dialog.loadButton.click()
    assert len(dialog.getfulllist()) == 10
    assert n_rows == len(rows)


def test_pipetting_scheme_save_canceled_writes_nothing(
    app, monkeypatch, tmp_path, plate_folder
):
    _answer_pipetting(monkeypatch, plate_folder)
    before = set(os.listdir(tmp_path))
    app.window.pipettingScheme()
    assert set(os.listdir(tmp_path)) == before


def test_pipetting_scheme_leaves_no_figures_open(
    app, monkeypatch, tmp_path, plate_folder
):
    _answer_pipetting(monkeypatch, plate_folder)
    figures = plt.get_fignums()
    app.dialogs.save_paths = [str(tmp_path / "scheme.pdf")]
    app.window.pipettingScheme()
    assert plt.get_fignums() == figures


# ---------------------------------------------------------------------------
# Folding table
# ---------------------------------------------------------------------------


@pytest.fixture
def folding(monkeypatch):
    """The folding dialogs opened, instead of running them."""
    opened = []

    def exec_(dialog):
        opened.append(dialog)
        return REJECTED

    monkeypatch.setattr(gui_design.FoldingDialog, "exec", exec_)
    return opened


def _table(dialog):
    table = dialog.table
    return [
        [
            table.item(row, col).text() if table.item(row, col) else ""
            for col in range(table.columnCount())
        ]
        for row in range(table.rowCount())
    ]


def test_folding_table_lists_one_mix_per_color(app, monkeypatch, folding):
    _designed(app, monkeypatch)
    app.window.foldingScheme()
    rows = _table(folding[0])
    assert [row[0] for row in rows] == [
        "Scaffold",
        "Core Mix",
        HANDLES[0][0] + " Mix",
        HANDLES[1][0] + " Mix",
        "Biotin 1:10",
        "H2O",
        "10x Folding Buffer",
        "Total Volume",
    ]
    # parts: the staples in each mix
    assert [row[2] for row in rows[1:4]] == [str(N_SITES - 3), "2", "1"]


def test_folding_table_volumes_add_up_to_the_total(app, monkeypatch, folding):
    _paint(app, 1, [(0, 0)])
    _answer_extensions(monkeypatch, [1])
    app.window.setSeq()
    app.window.foldingScheme()
    rows = _table(folding[0])
    volumes = {row[0]: float(row[5]) for row in rows}
    assert volumes["Scaffold"] == pytest.approx(4.0)
    assert volumes["Core Mix"] == pytest.approx(7.0)
    assert volumes[HANDLES[0][0] + " Mix"] == pytest.approx(0.4)
    assert volumes["Biotin 1:10"] == pytest.approx(8.0)
    assert volumes["10x Folding Buffer"] == pytest.approx(4.0)
    assert volumes["H2O"] == pytest.approx(16.6)
    total = volumes.pop("Total Volume")
    assert total == 40
    assert sum(volumes.values()) == pytest.approx(total)


def test_folding_table_recalculates_edited_values(app, monkeypatch, folding):
    _designed(app, monkeypatch)
    app.window.foldingScheme()
    dialog = folding[0]
    total = dialog.table.rowCount() - 1
    dialog.table.item(total, 5).setText("80")
    dialog.clcButton.click()
    rows = _table(dialog)
    assert float(rows[0][5]) == pytest.approx(8.0)  # scaffold
    assert float(rows[total - 1][5]) == pytest.approx(8.0)  # buffer


def test_folding_table_marks_a_negative_water_volume(
    app, monkeypatch, folding
):
    _designed(app, monkeypatch)
    app.window.foldingScheme()
    dialog = folding[0]
    dialog.table.item(0, 4).setText("100")  # 10x more scaffold
    dialog.clcButton.click()
    water = dialog.table.item(dialog.table.rowCount() - 3, 5)
    assert float(water.text()) < 0
    assert water.background().color().name() == "#ff0000"


def test_folding_table_with_invalid_input_warns(app, monkeypatch, folding):
    _designed(app, monkeypatch)
    app.window.foldingScheme()
    dialog = folding[0]
    volume = dialog.table.item(0, 5).text()
    dialog.table.item(0, 1).setText("a lot")
    dialog.clcButton.click()
    assert len(app.dialogs.texts("warning")) == 1
    assert dialog.table.item(0, 5).text() == volume


def test_folding_table_of_a_design_without_core_staples(
    app, monkeypatch, folding
):
    _fill(app, 1)
    _answer_extensions(monkeypatch, [1])
    app.window.setSeq()
    app.window.foldingScheme()
    rows = _table(folding[0])
    assert rows[-1][0] == "Total Volume"
    assert rows[2][0] == HANDLES[0][0] + " Mix"
    assert float(rows[1][5]) == 0  # no core staples to add


def test_folding_table_is_exported(app, monkeypatch, folding, tmp_path):
    _designed(app, monkeypatch)
    app.window.foldingScheme()
    dialog = folding[0]
    path = tmp_path / "folding.csv"
    app.dialogs.save_paths = [str(path)]
    dialog.exportButton.click()
    rows = design.read_plate(str(path))
    assert rows[0][0] == "Component" and len(rows[0]) == 7
    assert [row[:-1] for row in rows[1:]] == [
        row[:6] for row in _table(dialog)
    ]
    assert all(len(row) == 7 for row in rows)


def test_folding_table_export_canceled_writes_nothing(
    app, monkeypatch, folding, tmp_path
):
    _designed(app, monkeypatch)
    app.window.foldingScheme()
    before = set(os.listdir(tmp_path))
    folding[0].exportButton.click()
    assert set(os.listdir(tmp_path)) == before


# ---------------------------------------------------------------------------
# Screenshots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, header", [("design.png", b"\x89PNG"), ("design.pdf", b"%PDF")]
)
def test_screenshot_is_saved(app, tmp_path, name, header):
    path = tmp_path / name
    app.dialogs.save_paths = [str(path)]
    app.dialogs.save_filter = "*" + os.path.splitext(name)[1]
    app.window.takeScreenshot()
    assert path.read_bytes().startswith(header)
    assert _status(app) == f"Screenshot saved to {path}"


def test_screenshot_canceled_writes_nothing(app, tmp_path):
    before = set(os.listdir(tmp_path))
    app.window.takeScreenshot()
    assert set(os.listdir(tmp_path)) == before
    assert _status(app) == "Filename not specified. Screenshot not saved."
