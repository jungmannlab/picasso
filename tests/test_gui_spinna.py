"""User flows in SPINNA's three tabs and their dialogs.

Designing, saving and loading structures; loading structures,
experimental data and masks into the Simulations tab, running single
simulations, generating and loading search spaces, fitting, comparing
models and fitting labeling efficiencies; plotting and saving nearest
neighbor distances; and generating, thresholding and saving masks. All
modal dialogs are answered by ``Dialogs``, which records the messages
shown, and canceled file dialogs, rejected dialogs and invalid files
must leave the loaded state as it was.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtWidgets

from picasso import io, lib, spinna
from picasso.gui import spinna as gui_spinna

FOV = 2_000.0  # nm, side of the field of view of the molecules
N_MOL = 150  # molecules per target in the experimental data
PIXELSIZE = 130.0  # nm
DEPTH = 400  # nm

Button = QtWidgets.QMessageBox.StandardButton
Accepted = QtWidgets.QDialog.DialogCode.Accepted
Rejected = QtWidgets.QDialog.DialogCode.Rejected


class Dialogs:
    """Answers to the patched modal dialogs, first in first out; a
    dialog without a queued answer is canceled. Records the messages
    shown as ``(kind, text)``."""

    def __init__(self) -> None:
        self.messages = []
        self.open = []  # paths, or lists of paths for getOpenFileNames
        self.save = []
        self.dirs = []
        self.text = []
        self.int = []
        self.double = []
        self.question = Button.No
        self.asked = []  # names of the input dialogs opened

    def texts(self, kind: str | None = None) -> list[str]:
        return [t for k, t in self.messages if kind in (None, k)]

    def _pop(self, queue: list, name: str, canceled):
        self.asked.append(name)
        return queue.pop(0) if queue else canceled


@pytest.fixture
def dialogs(monkeypatch):
    d = Dialogs()
    box = QtWidgets.QMessageBox
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(
            box,
            kind,
            lambda *args, kind=kind, **kwargs: d.messages.append(
                (kind, args[2])
            ),
        )

    def question(*args, **kwargs):
        d.messages.append(("question", args[2]))
        return d.question

    monkeypatch.setattr(box, "question", question)

    files = QtWidgets.QFileDialog
    monkeypatch.setattr(
        files,
        "getOpenFileName",
        lambda *a, **k: (d._pop(d.open, "open", ""), ""),
    )
    monkeypatch.setattr(
        files,
        "getOpenFileNames",
        lambda *a, **k: (d._pop(d.open, "open", []), ""),
    )

    def get_save(*args, **kwargs):
        path = d._pop(d.save, "save", "")
        return path, "*" + os.path.splitext(path)[1] if path else ""

    monkeypatch.setattr(files, "getSaveFileName", get_save)
    monkeypatch.setattr(
        files,
        "getExistingDirectory",
        lambda *a, **k: d._pop(d.dirs, "dir", ""),
    )

    inputs = QtWidgets.QInputDialog
    monkeypatch.setattr(
        inputs, "getText", lambda *a, **k: d._pop(d.text, "text", ("", False))
    )
    monkeypatch.setattr(
        inputs, "getInt", lambda *a, **k: d._pop(d.int, "int", (0, False))
    )
    monkeypatch.setattr(
        inputs,
        "getDouble",
        lambda *a, **k: d._pop(d.double, "double", (0.0, False)),
    )
    return d


def answer(monkeypatch, dialog_class, result, setup=None) -> None:
    """Make ``dialog_class.exec`` return ``result`` after calling
    ``setup(dialog)``, instead of showing the dialog."""

    def exec_(self):
        if setup is not None:
            setup(self)
        return result

    monkeypatch.setattr(dialog_class, "exec", exec_)


@pytest.fixture
def window(qt_offscreen, dialogs, tmp_path):
    window = gui_spinna.Window()
    window.pwd = str(tmp_path)
    yield window
    window.close()
    plt.close(window.simulations_tab.nnd_fig)


@pytest.fixture
def out(tmp_path):
    """Folder for the files a test saves (``tmp_path`` also holds the
    user settings)."""
    path = tmp_path / "out"
    path.mkdir()
    return path


def save_structures(path, structures: list[spinna.Structure]) -> str:
    io.save_info(str(path), [s.get_info() for s in structures])
    return str(path)


def monomer_dimer(target: str = "A") -> list[spinna.Structure]:
    monomer = spinna.Structure("Monomer")
    monomer.define_coordinates(target, [0.0], [0.0], [0.0])
    dimer = spinna.Structure("Dimer")
    dimer.define_coordinates(target, [-5.0, 5.0], [0.0, 0.0], [0.0, 0.0])
    return [monomer, dimer]


def heterodimer() -> list[spinna.Structure]:
    """Monomers of A and B and an A-B heterodimer, 10 nm apart."""
    mono_a = spinna.Structure("MonoA")
    mono_a.define_coordinates("A", [0.0], [0.0], [0.0])
    mono_b = spinna.Structure("MonoB")
    mono_b.define_coordinates("B", [0.0], [0.0], [0.0])
    dimer = spinna.Structure("AB")
    dimer.define_coordinates("A", [-5.0], [0.0], [0.0])
    dimer.define_coordinates("B", [5.0], [0.0], [0.0])
    return [mono_a, mono_b, dimer]


def save_molecules(path, seed: int = 0, z: bool = False) -> str:
    """Uniformly distributed molecules in a ``FOV`` x ``FOV`` nm field
    of view, with the pick area recorded like Render does."""
    rng = np.random.default_rng(seed)
    n = N_MOL
    locs = pd.DataFrame(
        {
            "frame": np.zeros(n, dtype="u4"),
            "x": rng.uniform(0, FOV, n).astype("f4"),
            "y": rng.uniform(0, FOV, n).astype("f4"),
            "lpx": np.ones(n, dtype="f4"),
            "lpy": np.ones(n, dtype="f4"),
        }
    )
    if z:
        locs["z"] = rng.uniform(-DEPTH / 2, DEPTH / 2, n).astype("f4")
    side = int(FOV / PIXELSIZE) + 1
    info = [
        {"Width": side, "Height": side, "Pixelsize": PIXELSIZE, "Frames": 1},
        {"Area (um^2)": (FOV / 1e3) ** 2},
    ]
    io.save_locs(str(path), locs, info)
    return str(path)


def save_mask(locs_path: str, mask_path) -> str:
    """A 2D density mask generated from ``locs_path``."""
    generator = spinna.MaskGenerator(locs_path, binsize=100, sigma=100)
    generator.generate_mask()
    generator.save_mask(str(mask_path))
    return str(mask_path)


# ---------------------------------------------------------------------------
# Structures tab
# ---------------------------------------------------------------------------


@pytest.fixture
def structures_tab(window):
    return window.structures_tab


def add_structure(tab, dialogs, title: str) -> None:
    dialogs.text.append((title, True))
    tab.add_structure()


def target_rows(tab) -> list[list[QtWidgets.QWidget]]:
    """Name, x, y, z and delete widgets of each molecular target row in
    the layout of the molecular targets box."""
    layout = tab.mol_tar_box.content_layout
    widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
    widgets = [w for w in widgets if w.objectName()]
    return [list(row) for row in zip(*[iter(widgets)] * 5)]


def set_row(row, name: str, x: float, y: float, z: float) -> None:
    row[0].setText(name)
    row[0].editingFinished.emit()
    for spin, value in zip(row[1:4], (x, y, z)):
        spin.setValue(value)


def test_add_structure_creates_and_selects_it(structures_tab, dialogs):
    add_structure(structures_tab, dialogs, "Trimer")

    assert [s.title for s in structures_tab.structures] == ["Trimer"]
    assert structures_tab.current_structure == "Trimer"
    assert structures_tab.preview.structure.title == "Trimer"
    assert dialogs.messages == []


def test_add_structure_canceled_adds_nothing(structures_tab, dialogs):
    structures_tab.add_structure()

    assert dialogs.asked == ["text"]
    assert structures_tab.structures == []
    assert structures_tab.current_structure is None


def test_add_structure_with_a_taken_title_warns(structures_tab, dialogs):
    add_structure(structures_tab, dialogs, "Dimer")
    add_structure(structures_tab, dialogs, "Dimer")

    assert len(structures_tab.structures) == 1
    assert dialogs.texts("warning") == ["Structure title already taken."]


def test_add_molecular_target_without_a_structure_informs(
    structures_tab, dialogs
):
    structures_tab.add_molecular_target()

    assert target_rows(structures_tab) == []
    assert "No structure has been selected" in dialogs.texts()[0]


def test_molecular_target_rows_define_the_structure(structures_tab, dialogs):
    add_structure(structures_tab, dialogs, "AB")
    structures_tab.add_molecular_target()
    structures_tab.add_molecular_target()
    rows = target_rows(structures_tab)
    # a new row takes the name of the row above
    assert [row[0].text() for row in rows] == ["target", "target"]

    set_row(rows[0], "A", -5.0, 1.0, 2.0)
    set_row(rows[1], "B", 5.0, -1.0, 0.0)

    structure = structures_tab.structures[0]
    assert structure.targets == ["A", "B"]
    assert structure.x == {"A": [-5.0], "B": [5.0]}
    assert structure.y == {"A": [1.0], "B": [-1.0]}
    assert structure.z == {"A": [2.0], "B": [0.0]}
    assert structures_tab.preview.coords is not None


def test_deleting_a_molecular_target_removes_its_row(structures_tab, dialogs):
    add_structure(structures_tab, dialogs, "Trimer")
    for _ in range(3):
        structures_tab.add_molecular_target()
    rows = target_rows(structures_tab)
    for row, x in zip(rows, (-10.0, 0.0, 10.0)):
        set_row(row, "A", x, 0.0, 0.0)
    deleted = rows[1]

    rows[1][4].released.emit()

    structure = structures_tab.structures[0]
    assert structure.x == {"A": [-10.0, 10.0]}
    rows = target_rows(structures_tab)
    assert len(rows) == 2 and structures_tab.n_mol_tar == 2
    # the row below took over the number of the deleted one
    assert [w.objectName() for w in rows[1]] == [
        "target1",
        "x1",
        "y1",
        "z1",
        "del1",
    ]
    # the deleted row does not stay behind in the box
    box = structures_tab.mol_tar_box
    assert not any(widget.isVisibleTo(box) for widget in deleted)
    # and the renumbered delete button deletes its own row
    rows[1][4].released.emit()
    assert structure.x == {"A": [-10.0]}


def test_switching_structures_keeps_their_targets(structures_tab, dialogs):
    add_structure(structures_tab, dialogs, "First")
    structures_tab.add_molecular_target()
    set_row(target_rows(structures_tab)[0], "A", 1.0, 2.0, 3.0)
    add_structure(structures_tab, dialogs, "Second")
    assert target_rows(structures_tab) == []
    structures_tab.add_molecular_target()
    set_row(target_rows(structures_tab)[0], "B", 4.0, 5.0, 6.0)

    structures_tab.on_structure_clicked("First")

    rows = target_rows(structures_tab)
    assert len(rows) == 1
    assert rows[0][0].text() == "A"
    assert [spin.value() for spin in rows[0][1:4]] == [1.0, 2.0, 3.0]
    first, second = structures_tab.structures
    assert first.x == {"A": [1.0]} and second.x == {"B": [4.0]}


def test_deleting_structures_selects_the_first_remaining_one(
    structures_tab, dialogs
):
    for title in ("First", "Second", "Third"):
        add_structure(structures_tab, dialogs, title)

    structures_tab.on_structure_deleted("Third")
    assert [s.title for s in structures_tab.structures] == ["First", "Second"]
    assert structures_tab.current_structure == "First"

    structures_tab.on_structure_deleted("Second")
    structures_tab.on_structure_deleted("First")
    assert structures_tab.structures == []
    assert structures_tab.current_structure is None
    assert structures_tab.preview.structure is None


def test_saved_structures_load_back(structures_tab, dialogs, out, window):
    add_structure(structures_tab, dialogs, "AB")
    structures_tab.add_molecular_target()
    structures_tab.add_molecular_target()
    rows = target_rows(structures_tab)
    set_row(rows[0], "A", -5.0, 0.0, 1.0)
    set_row(rows[1], "B", 5.0, 0.0, -1.0)
    add_structure(structures_tab, dialogs, "Empty")
    path = str(out / "structures.yaml")
    dialogs.save.append(path)

    structures_tab.save_structures()

    loaded, targets = io.load_structures(path)
    assert [s.title for s in loaded] == ["AB", "Empty"]
    assert targets == ["A", "B"]
    assert loaded[0].x == {"A": [-5.0], "B": [5.0]}
    assert loaded[0].z == {"A": [1.0], "B": [-1.0]}
    assert window.pwd == str(out)

    # loading appends them to the structures of a new window
    other = gui_spinna.Window()
    try:
        dialogs.open.append(path)
        other.structures_tab.load_structures()
        tab = other.structures_tab
        assert [s.title for s in tab.structures] == ["AB", "Empty"]
        assert tab.current_structure == "AB"
        assert [row[0].text() for row in target_rows(tab)] == ["A", "B"]
    finally:
        other.close()


def test_save_structures_canceled_writes_nothing(structures_tab, dialogs, out):
    add_structure(structures_tab, dialogs, "Monomer")

    structures_tab.save_structures()

    assert dialogs.asked[-1] == "save"
    assert os.listdir(out) == []


def test_load_structures_canceled_keeps_the_structures(
    structures_tab, dialogs
):
    add_structure(structures_tab, dialogs, "Monomer")

    structures_tab.load_structures()

    assert [s.title for s in structures_tab.structures] == ["Monomer"]
    assert structures_tab.current_structure == "Monomer"


def test_loading_structures_twice_warns_about_the_titles(
    structures_tab, dialogs, out
):
    """Structures are selected and deleted by title, so only the first
    of two with one title could be edited."""
    path = save_structures(out / "md.yaml", monomer_dimer())
    dialogs.open.append(path)
    structures_tab.load_structures()
    assert dialogs.messages == []

    dialogs.open.append(path)
    structures_tab.load_structures()

    titles = [s.title for s in structures_tab.structures]
    assert titles == ["Monomer", "Dimer", "Monomer", "Dimer"]
    (text,) = dialogs.texts("warning")
    assert "titled Monomer, Dimer" in text


def test_load_structures_from_an_invalid_file_shows_an_error(
    structures_tab, dialogs, out
):
    add_structure(structures_tab, dialogs, "Monomer")
    path = out / "not_structures.yaml"
    path.write_text("a: 1\n")
    dialogs.open.append(str(path))

    structures_tab.load_structures()

    assert [s.title for s in structures_tab.structures] == ["Monomer"]
    assert len(dialogs.texts("critical")) == 1
    assert "not_structures.yaml" in dialogs.texts("critical")[0]


def test_save_preview_writes_the_shown_image(structures_tab, dialogs, out):
    add_structure(structures_tab, dialogs, "Monomer")
    structures_tab.add_molecular_target()
    structures_tab.save_preview()  # canceled
    assert os.listdir(out) == []

    dialogs.save.append(str(out / "preview.png"))
    structures_tab.save_preview()

    assert os.path.getsize(out / "preview.png") > 0


def test_right_click_adds_a_target_where_clicked(structures_tab, dialogs):
    add_structure(structures_tab, dialogs, "Dimer")
    structures_tab.add_molecular_target()
    preview = structures_tab.preview
    origin = preview.ORIGIN
    # needs two targets to know the scale
    preview.add_molecular_target_mouse(origin[0] + 50, origin[1])
    assert len(target_rows(structures_tab)) == 1

    structures_tab.add_molecular_target()
    set_row(target_rows(structures_tab)[1], "target", 10.0, 0.0, 0.0)
    factor = preview.factor

    preview.add_molecular_target_mouse(origin[0] + 50, origin[1] - 20)

    structure = structures_tab.structures[0]
    assert len(structure.x["target"]) == 3
    assert structure.x["target"][2] == pytest.approx(50 / factor, abs=0.01)
    assert structure.y["target"][2] == pytest.approx(20 / factor, abs=0.01)


def test_reset_rotation_zeros_the_preview_angles(structures_tab, dialogs):
    add_structure(structures_tab, dialogs, "Monomer")
    preview = structures_tab.preview
    preview.angx, preview.angy, preview.angz = 0.1, 0.2, 0.3

    structures_tab.update_preview(reset_angles=True)

    assert (preview.angx, preview.angy, preview.angz) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Simulations tab: loading
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_tab(window):
    tab = window.simulations_tab
    # single process, single simulation per candidate: fast fits
    tab.settings_dialog.parallel_check.setChecked(False)
    tab.settings_dialog.fitting_mode.setCurrentText("Brute force")
    return tab


def load_structures(tab, dialogs, path) -> None:
    dialogs.open.append(str(path))
    tab.load_structures()


def load_exp(tab, dialogs, target: str, path) -> None:
    dialogs.open.append(str(path))
    tab.load_exp_data(f"exp{target}")


@pytest.fixture
def loaded(sim_tab, dialogs, out):
    """Simulations tab with monomers and dimers of A and molecules of A
    loaded."""
    path = save_structures(out / "md.yaml", monomer_dimer())
    load_structures(sim_tab, dialogs, path)
    load_exp(sim_tab, dialogs, "A", save_molecules(out / "A.hdf5"))
    assert dialogs.messages == []
    return sim_tab


@pytest.fixture
def loaded_ab(sim_tab, dialogs, out):
    """Simulations tab with the A-B heterodimer model and molecules of A
    and B loaded."""
    path = save_structures(out / "het.yaml", heterodimer())
    load_structures(sim_tab, dialogs, path)
    load_exp(sim_tab, dialogs, "A", save_molecules(out / "A.hdf5", seed=1))
    load_exp(sim_tab, dialogs, "B", save_molecules(out / "B.hdf5", seed=2))
    assert dialogs.messages == []
    return sim_tab


def test_loading_structures_builds_the_per_target_widgets(
    sim_tab, dialogs, out
):
    path = save_structures(out / "het.yaml", heterodimer())
    load_structures(sim_tab, dialogs, path)

    assert sim_tab.targets == ["A", "B"]
    assert [s.title for s in sim_tab.structures] == ["MonoA", "MonoB", "AB"]
    for widgets in (
        sim_tab.densities_spins,
        sim_tab.le_spins,
        sim_tab.label_unc_spins,
        sim_tab.load_exp_data_buttons,
        sim_tab.load_mask_buttons,
    ):
        assert len(widgets) == 2
    assert [s.value() for s in sim_tab.prop_str_input_spins] == [100, 0, 0]
    assert set(sim_tab.settings_dialog.nn_counts) == {"A-A", "A-B", "B-B"}
    assert set(sim_tab.nn_plot_settings_dialog.nn_counts) == {
        "A-A",
        "A-B",
        "B-A",
        "B-B",
    }
    # Fit LE needs exactly two targets
    assert not sim_tab.fit_le_button.isHidden()
    assert sim_tab.get_heterodimer_distance() == pytest.approx(10.0)


def test_loading_structures_canceled_keeps_the_loaded_ones(loaded, dialogs):
    loaded.load_structures()

    assert [s.title for s in loaded.structures] == ["Monomer", "Dimer"]
    assert "A" in loaded.exp_data


def test_loading_structures_from_an_invalid_file_shows_an_error(
    loaded, dialogs, out
):
    path = out / "bad.yaml"
    path.write_text("- 1\n- 2\n")

    load_structures(loaded, dialogs, path)

    assert len(dialogs.texts("critical")) == 1
    assert [s.title for s in loaded.structures] == ["Monomer", "Dimer"]
    assert loaded.structures_path == str(out / "md.yaml")


def test_structures_with_the_same_targets_keep_the_experimental_data(
    loaded, dialogs, out
):
    path = save_structures(out / "monomer.yaml", monomer_dimer()[:1])

    load_structures(loaded, dialogs, path)

    assert [s.title for s in loaded.structures] == ["Monomer"]
    assert "A" in loaded.exp_data and loaded.nnd_hist_data_exp
    assert len(loaded.prop_str_input_spins) == 1


def test_structures_with_other_targets_reset_the_experimental_data(
    loaded, dialogs, out
):
    path = save_structures(out / "b.yaml", monomer_dimer("B"))

    load_structures(loaded, dialogs, path)

    assert loaded.targets == ["B"]
    assert loaded.exp_data == {} and loaded.nnd_hist_data_exp == []
    assert loaded.load_exp_data_buttons[0].text() == "Load B"
    assert loaded.fit_le_button.isHidden()


def test_loading_experimental_data_histograms_their_distances(loaded):
    assert loaded.load_exp_data_buttons[0].text() == "A loaded"
    assert loaded.exp_data["A"].shape == (N_MOL, 2)
    # the observed density follows from the pick area
    density = N_MOL / (FOV / 1e3) ** 2
    assert loaded.densities_spins[0].value() == pytest.approx(density, 0.01)
    assert len(loaded.nnd_hist_data_exp) == 1
    assert len(loaded.nnd_hist_data_exp[0]["counts"]) == 2  # 1st, 2nd NN
    assert loaded.nnd_ax.has_data()


def test_loading_experimental_data_canceled_keeps_the_plots(loaded, dialogs):
    loaded.run_single_sim()
    exp, sim = loaded.nnd_hist_data_exp, loaded.nnd_hist_data_sim
    assert exp and sim

    loaded.load_exp_data("expA")

    assert dialogs.asked[-1] == "open"
    assert loaded.nnd_hist_data_exp is exp
    assert loaded.nnd_hist_data_sim is sim


def test_loading_invalid_experimental_data_shows_an_error(
    loaded, dialogs, out
):
    path = out / "broken.hdf5"
    path.write_bytes(b"not hdf5")

    load_exp(loaded, dialogs, "A", path)

    assert len(dialogs.texts("critical")) == 1
    assert loaded.exp_data_paths["A"] == str(out / "A.hdf5")


def test_loading_a_mask_marks_its_target(loaded, dialogs, out):
    path = save_mask(str(out / "A.hdf5"), out / "A_mask.npy")
    loaded.load_mask("maskA")  # canceled
    assert loaded.masks == {}

    dialogs.open.append(path)
    loaded.load_mask("maskA")

    assert loaded.masks["A"].sum() == pytest.approx(1.0)
    assert loaded.mask_paths["A"] == path
    assert loaded.load_mask_buttons[0].text() == "A loaded"


def test_loading_an_invalid_mask_shows_an_error(loaded, dialogs, out):
    path = out / "mask.npy"
    np.save(path, np.ones((4, 4)))  # without the metadata of SPINNA
    (out / "mask.yaml").write_text("Generated by: Picasso Render\n")
    dialogs.open.append(str(path))

    loaded.load_mask("maskA")

    assert loaded.masks == {}
    assert len(dialogs.texts("critical")) == 1
    assert loaded.load_mask_buttons[0].text() == "Load A"


# ---------------------------------------------------------------------------
# Simulations tab: single simulations and dimensionality
# ---------------------------------------------------------------------------


def test_single_simulation_with_experimental_data_scores_the_fit(loaded):
    loaded.run_single_sim()

    assert loaded.mixer is not None
    assert len(loaded.nnd_hist_data_sim) == 1
    # Monomers only: as many molecules as observed, divided by the LE
    assert loaded.single_sim_n_total() == int(N_MOL / 0.5)
    assert 0 < loaded.current_score <= 1
    assert "KS2" in loaded.nnd_ax.get_title()


def test_proportions_are_capped_at_100_percent(loaded):
    monomer, dimer = loaded.prop_str_input_spins
    assert monomer.value() == 100

    dimer.setValue(30)

    assert monomer.value() == 100 and dimer.value() == 0
    monomer.setValue(60)
    dimer.setValue(70)
    assert (monomer.value(), dimer.value()) == (60, 40)


def test_single_simulation_with_proportions_below_100_percent_warns(
    loaded, dialogs
):
    loaded.prop_str_input_spins[0].setValue(40)

    loaded.run_single_sim()

    assert loaded.mixer is None
    assert "do not sum to 100%" in dialogs.texts("warning")[0]


def test_saving_a_single_simulation_asks_for_the_pixel_size_once(
    loaded, dialogs, out
):
    loaded.save_sim_result_check.setChecked(True)
    loaded.run_single_sim()  # save dialog canceled: nothing simulated
    assert loaded.mixer is None

    dialogs.save.append(str(out / "sim.hdf5"))
    loaded.run_single_sim()  # pixel size prompt canceled: nothing saved
    assert not os.path.exists(out / "sim_A.hdf5")

    dialogs.save += [str(out / "sim.hdf5"), str(out / "sim2.hdf5")]
    dialogs.double.append((100.0, True))
    loaded.run_single_sim()
    loaded.run_single_sim()

    assert dialogs.asked.count("double") == 2
    locs, info = io.load_locs(str(out / "sim2_A.hdf5"))
    assert len(locs) > 0
    assert lib.get_from_metadata(info, "Pixelsize") == 100.0


def test_saving_a_single_simulation_in_masks_takes_their_pixel_size(
    loaded, dialogs, out
):
    dialogs.open.append(save_mask(str(out / "A.hdf5"), out / "A_mask.npy"))
    loaded.load_mask("maskA")
    loaded.set_mask_den_stack("Masks")
    loaded.save_sim_result_check.setChecked(True)
    dialogs.save.append(str(out / "sim.hdf5"))

    loaded.run_single_sim()

    assert "double" not in dialogs.asked
    _, info = io.load_locs(str(out / "sim_A.hdf5"))
    assert lib.get_from_metadata(info, "Pixelsize") == PIXELSIZE


def test_a_refused_single_simulation_does_not_ask_where_to_save(
    loaded, dialogs
):
    loaded.set_mask_den_stack("Masks")
    loaded.save_sim_result_check.setChecked(True)

    loaded.run_single_sim()

    assert dialogs.texts() == ["Please load all masks."]
    assert "save" not in dialogs.asked


def test_simulating_in_masks_needs_all_masks(loaded, dialogs):
    loaded.set_mask_den_stack("Masks")

    loaded.run_single_sim()

    assert loaded.mixer is None
    assert dialogs.texts() == ["Please load all masks."]


def test_3d_simulation_needs_a_z_range(loaded, dialogs):
    loaded.dim_widget.setCurrentIndex(1)
    assert loaded.depth_stack.currentIndex() == 1  # the z range button

    loaded.run_single_sim()

    assert loaded.mixer is None
    assert "Please enter depth" in dialogs.texts()[0]


def test_z_range_button_sets_the_depth(sim_tab, dialogs, out):
    path = save_structures(out / "md.yaml", monomer_dimer())
    load_structures(sim_tab, dialogs, path)
    sim_tab.dim_widget.setCurrentIndex(1)
    sim_tab.on_depth_button_clicked()  # canceled
    assert sim_tab.depth is None

    dialogs.int.append((DEPTH, True))
    sim_tab.on_depth_button_clicked()

    assert sim_tab.depth == DEPTH
    assert sim_tab.depth_button.text() == f"Z range: {DEPTH} nm"
    sim_tab.run_single_sim()
    assert sim_tab.mixer.roi[2] == DEPTH


def test_z_range_converts_the_observed_densities(loaded, dialogs):
    density_2d = loaded.densities_spins[0].value()
    loaded.dim_widget.setCurrentIndex(1)
    dialogs.int.append((DEPTH, True))

    loaded.on_depth_button_clicked()

    density_3d = loaded.densities_spins[0].value()
    assert density_3d == pytest.approx(density_2d / (DEPTH / 1e3), 0.01)
    # and back to 2D
    loaded.dim_widget.setCurrentIndex(0)
    assert loaded.densities_spins[0].value() == pytest.approx(density_2d, 0.01)


def test_3d_simulation_of_2d_experimental_data_is_refused(loaded, dialogs):
    loaded.dim_widget.setCurrentIndex(1)
    dialogs.int.append((DEPTH, True))
    loaded.on_depth_button_clicked()

    loaded.run_single_sim()

    assert loaded.mixer is None
    assert "2D experimental data was loaded for A" in dialogs.texts()[0]


def test_2d_simulation_of_3d_experimental_data_uses_xy(
    sim_tab, dialogs, out, monkeypatch
):
    path = save_structures(out / "md.yaml", monomer_dimer())
    load_structures(sim_tab, dialogs, path)
    load_exp(sim_tab, dialogs, "A", save_molecules(out / "A.hdf5", z=True))
    xyz = sim_tab.exp_data["A"]
    assert xyz.shape == (N_MOL, 3)
    scored = []
    score = spinna.NND_score
    monkeypatch.setattr(
        spinna,
        "NND_score",
        lambda exp, sim: scored.append(exp) or score(exp, sim),
    )

    sim_tab.run_single_sim()

    assert sim_tab.mixer.roi[2] is None
    assert dialogs.messages == []
    assert len(sim_tab.nnd_hist_data_sim) == 1
    # scored against the lateral distances, like the plotted histograms
    n = sim_tab.mixer.get_neighbor_counts("A", "A")
    xy = xyz[:, :2]
    np.testing.assert_allclose(scored[0][0], spinna.get_NN_dist(xy, xy, n))


# ---------------------------------------------------------------------------
# Simulations tab: search space and fitting
# ---------------------------------------------------------------------------

GRANULARITY = 3


def generate(monkeypatch, tab, granularity=GRANULARITY, save=False):
    def setup(dialog):
        dialog.n_sim_spin.setValue(1)
        dialog.granularity_spin.setValue(granularity)
        dialog.save_check.setChecked(save)

    answer(monkeypatch, gui_spinna.GenerateSearchSpaceDialog, Accepted, setup)
    tab.generate_search_space()


@pytest.mark.parametrize(
    "method", ["generate_search_space", "fit_n_str", "compare_models"]
)
def test_fitting_without_structures_is_refused(sim_tab, dialogs, method):
    getattr(sim_tab, method)()

    assert dialogs.texts() == ["Please load structures first."]
    assert dialogs.asked == []


@pytest.mark.parametrize(
    "method", ["generate_search_space", "fit_n_str", "compare_models"]
)
def test_fitting_without_experimental_data_is_refused(
    sim_tab, dialogs, out, method
):
    path = save_structures(out / "md.yaml", monomer_dimer())
    load_structures(sim_tab, dialogs, path)

    getattr(sim_tab, method)()

    assert dialogs.texts() == ["Please load experimental data first."]


@pytest.mark.parametrize("method", ["fit_n_str", "compare_models", "fit_le"])
def test_fitting_without_a_search_space_is_refused(loaded, dialogs, method):
    getattr(loaded, method)()

    assert dialogs.texts() == ["Please generate/load search space."]


def test_generate_search_space_rejected_changes_nothing(
    loaded, dialogs, monkeypatch
):
    answer(monkeypatch, gui_spinna.GenerateSearchSpaceDialog, Rejected)

    loaded.generate_search_space()

    assert loaded.N_structures_fit == {} and loaded.granularity is None


def test_generate_search_space_fills_the_fit_button(
    loaded, dialogs, monkeypatch
):
    generate(monkeypatch, loaded)

    assert loaded.granularity == GRANULARITY and loaded.n_sim_fit == 1
    assert loaded.n_total == {"A": int(N_MOL / 0.5)}
    n = len(loaded.N_structures_fit["Monomer"])
    assert n == len(loaded.N_structures_fit["Dimer"]) == GRANULARITY
    assert f"# tested combinations: {n}" in loaded.fit_button.text()
    assert "Estimated time" in loaded.fit_button.text()
    # other densities invalidate the search space
    loaded.densities_spins[0].setValue(10.0)
    assert loaded.granularity is None


def test_saved_search_space_loads_back(loaded, dialogs, monkeypatch, out):
    path = str(out / "search_space.csv")
    generate(monkeypatch, loaded, save=False)
    generate(monkeypatch, loaded, save=True)  # save dialog canceled
    assert not os.path.exists(path)
    dialogs.save.append(path)
    generate(monkeypatch, loaded, granularity=5, save=True)
    saved = {k: v.copy() for k, v in loaded.N_structures_fit.items()}
    generate(monkeypatch, loaded)  # back to granularity 3

    dialogs.open.append(path)
    dialogs.int.append((2, True))
    loaded.load_search_space()

    assert loaded.n_sim_fit == 2
    assert loaded.granularity == "loaded from file"
    for title, counts in saved.items():
        np.testing.assert_array_equal(loaded.N_structures_fit[title], counts)
    assert "# tested combinations: 5" in loaded.fit_button.text()


def test_load_search_space_canceled_keeps_the_search_space(
    loaded, dialogs, monkeypatch, out
):
    path = str(out / "search_space.csv")
    dialogs.save.append(path)
    generate(monkeypatch, loaded, granularity=5, save=True)
    generate(monkeypatch, loaded)
    before = {k: v.copy() for k, v in loaded.N_structures_fit.items()}
    loaded.load_search_space()  # file dialog canceled

    dialogs.open.append(path)  # number of simulations canceled
    loaded.load_search_space()

    assert loaded.granularity == GRANULARITY and loaded.n_sim_fit == 1
    for title, counts in before.items():
        np.testing.assert_array_equal(loaded.N_structures_fit[title], counts)


def test_load_search_space_of_other_structures_warns(loaded, dialogs, out):
    path = out / "other.csv"
    pd.DataFrame({"N_X": [1], "Prop_X": [100.0]}).to_csv(path, index=False)
    dialogs.open.append(str(path))

    loaded.load_search_space()

    assert "do not correspond" in dialogs.texts("warning")[0]
    assert loaded.N_structures_fit == {}


def test_load_search_space_from_an_invalid_file_shows_an_error(
    loaded, dialogs, out
):
    path = out / "empty.csv"
    path.write_text("")
    dialogs.open.append(str(path))

    loaded.load_search_space()

    assert len(dialogs.texts("critical")) == 1
    assert loaded.N_structures_fit == {}


def test_fit_shows_the_best_proportions_and_saves_a_summary(
    loaded, dialogs, monkeypatch, out
):
    generate(monkeypatch, loaded)
    loaded.save_fit_results_check.setChecked(True)
    loaded.fit_n_str()  # scores save dialog canceled: no fit
    assert loaded.fit_results_display.text().strip() == ""

    dialogs.save += [str(out / "scores.csv"), str(out / "summary.txt")]
    loaded.fit_n_str()

    text = loaded.fit_results_display.text()
    assert text.startswith("Monomer - ") and "Dimer - " in text
    props = [spin.value() for spin in loaded.prop_str_input_spins]
    assert sum(props) == pytest.approx(100, abs=0.01)
    assert len(pd.read_csv(out / "scores.csv")) == GRANULARITY
    summary = (out / "summary.txt").read_text()
    assert f"Best fitting proportions (%): {text}" in summary
    assert "Number of neighbors considered (A-A)" in summary
    assert str(out / "A.hdf5") in summary
    # the best fit is simulated and plotted
    assert len(loaded.nnd_hist_data_sim) == 1
    # and can be restored after editing the proportions
    loaded.prop_str_input_spins[0].setValue(0)
    best = [
        w
        for w in loaded.prop_str_input.findChildren(QtWidgets.QPushButton)
        if w.text() == "Best fitting combination"
    ]
    best[0].released.emit()
    assert [s.value() for s in loaded.prop_str_input_spins] == props


def add_models(paths: list, dialogs):
    """``setup`` of the Compare models dialog adding ``paths``."""

    def setup(dialog):
        dialogs.open.append([str(p) for p in paths])
        dialog.on_add_model()

    return setup


def test_compare_models_needs_a_generated_search_space(
    loaded, dialogs, monkeypatch, out
):
    dialogs.save.append(str(out / "search_space.csv"))
    generate(monkeypatch, loaded, save=True)
    dialogs.open.append(str(out / "search_space.csv"))
    dialogs.int.append((1, True))
    loaded.load_search_space()

    loaded.compare_models()

    assert "Loading a search space is not allowed" in dialogs.texts()[0]


def test_compare_models_rejected_or_without_models_changes_nothing(
    loaded, dialogs, monkeypatch, out
):
    generate(monkeypatch, loaded)
    model = save_structures(out / "monomer.yaml", monomer_dimer()[:1])
    answer(
        monkeypatch,
        gui_spinna.CompareModelsDialog,
        Rejected,
        add_models([model], dialogs),
    )
    loaded.compare_models()
    answer(monkeypatch, gui_spinna.CompareModelsDialog, Accepted)
    loaded.compare_models()

    assert [s.title for s in loaded.structures] == ["Monomer", "Dimer"]
    assert loaded.fit_results_display.text().strip() == ""


def test_compare_models_refuses_models_of_other_targets(
    loaded, dialogs, monkeypatch, out
):
    generate(monkeypatch, loaded)
    paths = [
        save_structures(out / "b.yaml", monomer_dimer("B")),
        out / "bad.yaml",
    ]
    paths[1].write_text("a: 1\n")
    models = []

    def setup(dialog):
        for path in paths:
            dialogs.open.append([str(path)])
            dialog.on_add_model()
        models.extend(dialog.models)

    answer(monkeypatch, gui_spinna.CompareModelsDialog, Rejected, setup)
    loaded.compare_models()

    assert models == []
    assert "does not contain all molecular targets" in dialogs.texts()[0]
    assert len(dialogs.texts("critical")) == 1


def test_compare_models_loads_the_best_model(
    loaded, dialogs, monkeypatch, out
):
    generate(monkeypatch, loaded)
    paths = [
        save_structures(out / "monomer.yaml", monomer_dimer()[:1]),
        save_structures(out / "monomer_dimer.yaml", monomer_dimer()),
        save_structures(out / "removed.yaml", monomer_dimer()[1:]),
    ]

    def setup(dialog):
        add_models(paths, dialogs)(dialog)
        assert len(dialog.models) == 3
        dialog.on_model_clicked(paths[2])  # removes it again
        dialog.save_fit_scores.setChecked(True)

    answer(monkeypatch, gui_spinna.CompareModelsDialog, Accepted, setup)
    loaded.compare_models()  # folder dialog canceled
    assert loaded.fit_results_display.text().strip() == ""

    dialogs.dirs.append(str(out))
    loaded.compare_models()

    text = loaded.fit_results_display.text()
    assert text.startswith("Best fitting model: monomer")
    best = text.split(": ")[1].split(",")[0]
    titles = {
        "monomer.yaml": ["Monomer"],
        "monomer_dimer.yaml": ["Monomer", "Dimer"],
    }[best]
    assert [s.title for s in loaded.structures] == titles
    assert len(loaded.prop_str_input_spins) == len(titles)
    assert len(loaded.nnd_hist_data_sim) == 1
    assert any(name.endswith(".csv") for name in os.listdir(out))


def test_fit_le_needs_two_targets(loaded, dialogs, monkeypatch):
    generate(monkeypatch, loaded)

    loaded.fit_le()

    assert dialogs.texts() == [
        "Fit LE requires exactly two molecular targets."
    ]


def test_fit_le_rejected_changes_nothing(loaded_ab, dialogs, monkeypatch):
    generate(monkeypatch, loaded_ab)
    answer(monkeypatch, gui_spinna.FitLEDialog, Rejected)

    loaded_ab.fit_le()

    assert loaded_ab.fit_results_display.text().strip() == ""
    assert [s.value() for s in loaded_ab.le_spins] == [50, 50]


def test_fit_le_without_a_folder_for_the_scores_does_not_fit(
    loaded_ab, dialogs, monkeypatch
):
    """As in Compare models: the scores to save need a folder."""
    generate(monkeypatch, loaded_ab)
    answer(
        monkeypatch,
        gui_spinna.FitLEDialog,
        Accepted,
        setup=lambda dialog: dialog.save_fit_scores.setChecked(True),
    )

    loaded_ab.fit_le()  # the folder dialog is canceled

    assert dialogs.asked[-1] == "dir"
    assert loaded_ab.fit_results_display.text().strip() == ""
    assert [s.value() for s in loaded_ab.le_spins] == [50, 50]


def test_fit_le_dialog_counts_the_fitting_rounds(loaded_ab):
    dialog = gui_spinna.FitLEDialog(loaded_ab)
    # one fixed distance and label uncertainty: the final comparison only
    assert dialog.rounds_label.text() == "Estimated SPINNA rounds: 1"
    assert dialog.distance_fixed_spin.value() == pytest.approx(10.0)

    dialog.distance_checkbox.setChecked(True)  # 5, 10, ..., 25 nm
    assert not dialog.distance_fixed_spin.isEnabled()
    assert dialog.rounds_label.text() == "Estimated SPINNA rounds: 5"
    dialog.label_unc_checkbox.setChecked(True)  # 3, 4, ..., 8 nm each
    assert dialog.rounds_label.text() == "Estimated SPINNA rounds: 65"


def test_fit_le_shows_the_labeling_efficiencies(
    loaded_ab, dialogs, monkeypatch, out
):
    generate(monkeypatch, loaded_ab)
    answer(monkeypatch, gui_spinna.FitLEDialog, Accepted)
    dialogs.save.append(str(out / "fit_le_summary.txt"))

    loaded_ab.fit_le()

    text = loaded_ab.fit_results_display.text()
    assert text.startswith("LE A: ") and "distance: 10.00 nm" in text
    # the fitted structures keep the titles of the loaded ones
    titles = [s.title for s in loaded_ab.structures]
    assert titles == ["MonoA", "MonoB", "AB"]
    assert [s.value() for s in loaded_ab.le_spins] == [100, 100]
    summary = (out / "fit_le_summary.txt").read_text()
    assert "Fitted labeling efficiency (%) (A)" in summary
    assert len(loaded_ab.nnd_hist_data_sim) == 4  # A-A, A-B, B-A, B-B


@pytest.mark.parametrize("mode", ["Coarse to fine", "Bayesian"])
def test_other_fitting_modes_find_proportions(
    loaded, dialogs, monkeypatch, mode
):
    loaded.settings_dialog.fitting_mode.setCurrentText(mode)
    generate(monkeypatch, loaded, granularity=5)

    loaded.fit_n_str()

    assert loaded.fit_results_display.text().startswith("Monomer - ")
    assert dialogs.messages == []


def test_bootstrap_fit_shows_the_uncertainties(loaded, dialogs, monkeypatch):
    loaded.bootstrap_check.setChecked(True)
    generate(monkeypatch, loaded)

    loaded.fit_n_str()

    assert "+/-" in loaded.fit_results_display.text()
    props = [spin.value() for spin in loaded.prop_str_input_spins]
    assert sum(props) == pytest.approx(100, abs=0.01)


def test_2d_fit_of_3d_experimental_data_keeps_their_z(
    sim_tab, dialogs, monkeypatch, out
):
    path = save_structures(out / "md.yaml", monomer_dimer())
    load_structures(sim_tab, dialogs, path)
    load_exp(sim_tab, dialogs, "A", save_molecules(out / "A.hdf5", z=True))
    generate(monkeypatch, sim_tab)

    sim_tab.fit_n_str()

    assert sim_tab.exp_data["A"].shape == (N_MOL, 3)
    sim_tab.dim_widget.setCurrentIndex(1)
    dialogs.int.append((DEPTH, True))
    sim_tab.on_depth_button_clicked()
    sim_tab.run_single_sim()
    assert dialogs.messages == []
    assert sim_tab.mixer.roi[2] == DEPTH


def test_fit_in_masks_of_another_dimensionality_is_refused(
    loaded, dialogs, monkeypatch, out
):
    dialogs.open.append(save_mask(str(out / "A.hdf5"), out / "A_mask.npy"))
    loaded.load_mask("maskA")
    loaded.set_mask_den_stack("Masks")
    generate(monkeypatch, loaded)
    loaded.dim_widget.setCurrentIndex(1)

    loaded.fit_n_str()

    assert "3D simulation was requested but a 2D" in dialogs.texts()[-1]
    assert loaded.fit_results_display.text().strip() == ""


# ---------------------------------------------------------------------------
# Simulations tab: optional settings
# ---------------------------------------------------------------------------


def test_manual_neighbor_counts_are_used_for_fitting(
    loaded, dialogs, monkeypatch, out
):
    settings = loaded.settings_dialog
    assert not settings.nn_counts["A-A"].isEnabled()
    settings.auto_nn_check.setChecked(False)
    assert settings.nn_counts["A-A"].isEnabled()
    settings.nn_counts["A-A"].setValue(3)
    generate(monkeypatch, loaded)
    dialogs.save.append(str(out / "summary.txt"))

    loaded.fit_n_str()

    summary = (out / "summary.txt").read_text()
    assert "Number of neighbors considered (A-A): 3" in summary


def test_fitting_mode_is_remembered(sim_tab, qt_offscreen):
    sim_tab.settings_dialog.fitting_mode.setCurrentText("Coarse to fine")

    other = gui_spinna.Window()
    try:
        mode = other.simulations_tab.settings_dialog.fitting_mode
        assert mode.currentText() == "Coarse to fine"
    finally:
        other.close()
        plt.close(other.simulations_tab.nnd_fig)


# ---------------------------------------------------------------------------
# Simulations tab: nearest neighbor distance plots
# ---------------------------------------------------------------------------


@pytest.fixture
def plot_settings(loaded):
    loaded.run_single_sim()
    return loaded.nn_plot_settings_dialog


@pytest.mark.parametrize(
    "min_dist, max_dist, binsize, message",
    [
        (100, 50, 4.0, "Max. distance is lower than min. distance."),
        (0, 10, 20.0, "Bin size is larger than the distance range."),
    ],
)
def test_invalid_plot_ranges_warn(
    plot_settings, dialogs, min_dist, max_dist, binsize, message
):
    plot_settings.binsize_sim.setValue(binsize)
    plot_settings.min_dist.setValue(min_dist)
    plot_settings.max_dist.setValue(max_dist)

    plot_settings.update_plots()

    assert dialogs.texts("warning") == [message]
    # the new bin size applies once the range is corrected
    plot_settings.min_dist.setValue(0)
    plot_settings.max_dist.setValue(200)
    plot_settings.update_plots()
    bins = plot_settings.sim_tab.nnd_hist_data_sim[0]["bins"][0]
    assert np.diff(bins)[0] == binsize
    assert len(dialogs.messages) == 1


def test_invalid_colors_warn(plot_settings, dialogs):
    plot_settings.colors[0].setText("not a color")
    plot_settings.check_color_labels()
    assert "Incorrect color" in dialogs.texts("warning")[0]

    plot_settings.colors[0].setText("black")
    plot_settings.check_color_labels()
    assert len(dialogs.messages) == 1
    assert plot_settings.extract_params()["colors"][0] == "black"


def test_update_plots_rehistograms_after_bin_size_changes(plot_settings):
    tab = plot_settings.sim_tab
    plot_settings.update_plots()  # nothing changed: plotted again only
    sim = tab.nnd_hist_data_sim

    plot_settings.binsize_exp.setValue(10.0)
    plot_settings.update_plots()
    assert np.diff(tab.nnd_hist_data_exp[0]["bins"][0])[0] == 10.0
    assert tab.nnd_hist_data_sim is sim

    plot_settings.binsize_sim.setValue(20.0)
    plot_settings.update_plots()
    assert np.diff(tab.nnd_hist_data_sim[0]["bins"][0])[0] == 20.0
    assert not (plot_settings.rehist_exp or plot_settings.rehist_sim)


def test_neighbor_counts_set_the_plotted_neighbors(plot_settings):
    tab = plot_settings.sim_tab
    plot_settings.nn_counts["A-A"].setValue(4)

    plot_settings.update_plots()

    assert len(tab.nnd_hist_data_exp[0]["counts"]) == 4
    assert len(tab.nnd_hist_data_sim[0]["counts"]) == 4


def test_plot_labels_and_limits_reach_the_plot(plot_settings):
    tab = plot_settings.sim_tab
    plot_settings.title.setText("Distances")
    plot_settings.xlabel.setText("d (nm)")
    plot_settings.max_dist.setValue(80)
    plot_settings.ylimmax.setValue(0.5)
    plot_settings.nnd_legend_check.setChecked(True)

    plot_settings.update_plots()

    assert tab.nnd_ax.get_title().startswith("Distances A → A")
    assert tab.nnd_ax.get_xlabel() == "d (nm)"
    assert tab.nnd_ax.get_xlim() == (0, 80)
    assert tab.nnd_ax.get_ylim() == (0, 0.5)
    assert tab.nnd_ax.get_legend() is not None


def test_plot_fonts_are_remembered(plot_settings, qt_offscreen):
    plot_settings.fontsize_title.setValue(13)
    plot_settings.fontsize_ticks.setValue(5)

    other = gui_spinna.Window()
    try:
        dialog = other.simulations_tab.nn_plot_settings_dialog
        assert dialog.fontsize_title.value() == 13
        assert dialog.fontsize_ticks.value() == 5
        assert dialog.font_title.currentText() == (
            plot_settings.font_title.currentText()
        )
    finally:
        other.close()
        plt.close(other.simulations_tab.nnd_fig)


def test_arrows_cycle_through_the_target_pairs(loaded_ab):
    loaded_ab.run_single_sim()
    titles = []
    for _ in range(4):
        titles.append(loaded_ab.nnd_ax.get_title().split("\n")[0])
        loaded_ab.on_right_nnd_clicked()

    assert titles == [
        "NNDs: A → A",
        "NNDs: A → B",
        "NNDs: B → A",
        "NNDs: B → B",
    ]
    assert loaded_ab.current_nnd_idx == 0
    loaded_ab.on_left_nnd_clicked()
    assert loaded_ab.current_nnd_idx == 3


def test_experimental_plots_without_simulation_follow_the_pairs(loaded_ab):
    assert loaded_ab.mixer is None
    assert len(loaded_ab.nnd_hist_data_exp) == 4

    loaded_ab.on_right_nnd_clicked()

    assert loaded_ab.nnd_ax.get_title() == "NNDs: A → B"


def test_save_nnd_plots_writes_one_image_per_pair(loaded_ab, dialogs, out):
    loaded_ab.run_single_sim()
    loaded_ab.save_nnd_plots()  # canceled
    before = set(os.listdir(out))

    dialogs.save.append(str(out / "nnd.png"))
    loaded_ab.save_nnd_plots()

    assert set(os.listdir(out)) - before == {
        f"nnd_{pair}.png" for pair in ("A_A", "A_B", "B_A", "B_B")
    }


def test_save_nnd_values_writes_the_histograms(loaded, dialogs, out):
    loaded.run_single_sim()
    loaded.save_nnd_values()  # canceled
    before = set(os.listdir(out))

    dialogs.save.append(str(out / "values.csv"))
    loaded.save_nnd_values()

    assert set(os.listdir(out)) - before == {
        "values_A_A_sim.csv",
        "values_A_A_exp.csv",
    }
    sim = pd.read_csv(out / "values_A_A_sim.csv")
    assert list(sim.columns) == [
        "bins_sim",
        "NN1_values_sim",
        "NN2_values_sim",
    ]
    counts = loaded.nnd_hist_data_sim[0]["counts"][0]
    np.testing.assert_allclose(sim["NN1_values_sim"], counts, rtol=1e-6)


def test_save_nnd_values_of_monomers_only_writes_the_histograms(
    sim_tab, dialogs, out
):
    path = save_structures(out / "monomer.yaml", monomer_dimer()[:1])
    load_structures(sim_tab, dialogs, path)
    load_exp(sim_tab, dialogs, "A", save_molecules(out / "A.hdf5"))
    dialogs.save.append(str(out / "values.csv"))

    sim_tab.save_nnd_values()

    assert os.path.exists(out / "values_A_A_exp.csv")


def test_saving_without_plots_does_nothing(sim_tab, dialogs):
    sim_tab.save_nnd_plots()
    sim_tab.save_nnd_values()

    assert dialogs.asked == []


# ---------------------------------------------------------------------------
# Mask generation tab
# ---------------------------------------------------------------------------


@pytest.fixture
def mask_tab(window):
    tab = window.mask_generator_tab
    tab.mask_binsize_xy.setValue(100)
    tab.mask_blur_xy.setValue(100)
    return tab


def load_mask_locs(tab, dialogs, path) -> None:
    dialogs.open.append(str(path))
    tab.load_locs()


def test_load_molecules_for_a_mask(mask_tab, dialogs, out):
    mask_tab.load_locs()  # canceled
    assert mask_tab.mask_generator is None
    assert not mask_tab.generate_mask_button.isEnabled()

    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5"))

    assert mask_tab.generate_mask_button.isEnabled()
    # 2D molecules give 2D masks only
    ndims = [mask_tab.mask_ndim.itemText(i) for i in range(2)]
    assert ndims == ["2D", ""]

    load_mask_locs(mask_tab, dialogs, save_molecules(out / "B.hdf5", z=True))
    ndims = [mask_tab.mask_ndim.itemText(i) for i in range(2)]
    assert ndims == ["2D", "3D"]


def test_load_invalid_molecules_for_a_mask_shows_an_error(
    mask_tab, dialogs, out
):
    path = out / "broken.hdf5"
    path.write_bytes(b"not hdf5")

    load_mask_locs(mask_tab, dialogs, path)

    assert len(dialogs.texts("critical")) == 1
    assert mask_tab.mask_generator is None


def test_generated_mask_is_shown_and_described(mask_tab, dialogs, out):
    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5"))

    mask_tab.generate_mask()

    mask = mask_tab.mask
    assert mask.sum() == pytest.approx(1.0)
    assert mask_tab.mask_dims_value.text() == "x".join(map(str, mask.shape))
    assert float(mask_tab.mask_area_value.text()) > 0
    assert mask_tab.save_mask_button.isEnabled()
    assert mask_tab.preview.qimage is not None
    assert mask_tab.thresholding_value.value() == pytest.approx(
        mask_tab.mask_generator.thresh, abs=1e-8
    )
    # loading other molecules discards the mask
    load_mask_locs(mask_tab, dialogs, out / "A.hdf5")
    assert mask_tab.mask is None and mask_tab.preview.qimage is None
    assert not mask_tab.save_mask_button.isEnabled()


@pytest.mark.parametrize("mask_type", [0, 1])  # density map, binary
def test_threshold_removes_pixels_below_it(mask_tab, dialogs, out, mask_type):
    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5"))
    mask_tab.mask_type.setCurrentIndex(mask_type)
    mask_tab.generate_mask()
    unthresholded = mask_tab.mask.copy()
    image = mask_tab.mask_generator.image
    thresh = np.quantile(image, 0.7)
    mask_tab.thresholding_value.setValue(thresh)

    mask_tab.thresholding_check.setChecked(True)

    mask = mask_tab.mask
    assert mask.sum() == pytest.approx(1.0)
    assert (mask[image < thresh] == 0).all()
    assert (mask > 0).sum() < (unthresholded > 0).sum()
    mask_tab.thresholding_check.setChecked(False)
    np.testing.assert_array_equal(mask_tab.mask, unthresholded)


@pytest.mark.parametrize("mask_type", [0, 1])  # density map, binary
def test_the_threshold_cannot_empty_the_mask(
    mask_tab, dialogs, out, mask_type
):
    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5"))
    mask_tab.mask_type.setCurrentIndex(mask_type)
    mask_tab.generate_mask()
    peak = mask_tab.mask_generator.image.max()

    mask_tab.thresholding_value.setValue(1.0)  # far above every pixel
    mask_tab.thresholding_check.setChecked(True)

    assert mask_tab.thresholding_value.value() < peak
    assert np.isfinite(mask_tab.mask).all()
    assert mask_tab.mask.sum() == pytest.approx(1.0)
    assert (mask_tab.mask > 0).sum() >= 1


def test_saved_mask_loads_back(mask_tab, dialogs, out):
    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5"))
    mask_tab.generate_mask()
    mask_tab.thresholding_value.setValue(
        np.quantile(mask_tab.mask_generator.image, 0.5)
    )
    mask_tab.thresholding_check.setChecked(True)
    mask_tab.save_mask()  # canceled
    assert not os.path.exists(out / "A_mask.npy")

    dialogs.save.append(str(out / "A_mask.npy"))
    mask_tab.save_mask()

    mask, info = io.load_mask(str(out / "A_mask.npy"))
    np.testing.assert_allclose(mask, mask_tab.mask)
    assert info["Threshold value"] == pytest.approx(
        mask_tab.thresholding_value.value()
    )
    assert info["Binsize (nm)"] == [100.0, 100.0]


def test_3d_mask_saves_images_of_its_z_slices(mask_tab, dialogs, out):
    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5", z=True))
    mask_tab.mask_ndim.setCurrentIndex(1)
    assert not mask_tab.zslice_check.isHidden()
    mask_tab.generate_mask()
    mask = mask_tab.mask.copy()
    assert mask.ndim == 3
    assert mask_tab.mask_area_label.text() == "Volume (μm³):"
    mask_tab.zslice_check.setChecked(True)
    mask_tab.zslice_slider.setValue(0)
    np.testing.assert_allclose(mask_tab.preview.image, mask[:, :, 0])
    dialogs.save.append(str(out / "mask.npy"))
    dialogs.question = Button.Yes

    mask_tab.save_mask()

    n_z = mask.shape[2]
    assert {f"mask_z{z}.png" for z in range(n_z)} <= set(os.listdir(out))
    # the images do not change the mask
    np.testing.assert_array_equal(mask_tab.mask, mask)
    saved, _ = io.load_mask(str(out / "mask.npy"))
    assert saved.shape == mask.shape


def test_mask_preview_zooms_pans_and_fits(mask_tab, dialogs, out):
    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5"))
    mask_tab.generate_mask()
    preview = mask_tab.preview
    preview.resize(300, 300)
    preview.fit_view()
    fit_scale = preview.scale

    preview.zoom(4.0)
    assert preview.scale == pytest.approx(4 * fit_scale)
    assert not preview.fit
    center = preview.center
    preview.pan(10, 0)
    assert preview.center[0] < center[0]
    preview.update_cursor_info(QtCore.QPointF(150, 150))
    assert "p = " in mask_tab.cursor_info.text()

    preview.fit_view()
    assert preview.fit and preview.scale == pytest.approx(fit_scale)


def test_mask_preview_saves_the_current_view(mask_tab, dialogs, out):
    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5"))
    mask_tab.generate_mask()
    mask_tab.preview.resize(300, 300)
    mask_tab.preview.fit_view()
    mask_tab.scalebar_check.setChecked(True)
    mask_tab.preview.save_current_view()  # canceled
    assert not os.path.exists(out / "view.png")

    dialogs.save.append(str(out / "view.png"))
    mask_tab.preview.save_current_view()

    assert os.path.getsize(out / "view.png") > 0


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


def test_closing_the_window_remembers_the_folder(qt_offscreen, dialogs, out):
    window = gui_spinna.Window()
    window.pwd = str(out)

    window.close()
    plt.close(window.simulations_tab.nnd_fig)

    assert io.load_user_settings()["SPINNA"]["PWD"] == str(out)
    other = gui_spinna.Window()
    try:
        assert other.pwd == str(out)
    finally:
        other.close()
        plt.close(other.simulations_tab.nnd_fig)
