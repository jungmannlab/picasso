"""Single simulations in SPINNA's Simulations tab without experimental
data.

The numbers of molecules used to come from the loaded experimental data
only, so running a single simulation right after loading structures
failed with ``KeyError`` on the target name. Without experimental data,
the ROI now holds ``SINGLE_SIM_N_MOL`` molecules of the first target and
the numbers of molecules follow from the observed densities.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest

from picasso import io, spinna
from picasso.gui import spinna as gui_spinna

TARGET = "EGFR"
DENSITY = 100.0  # um^-2 (2D) or um^-3 (3D)
LE = 50.0  # %
DEPTH = 500.0  # nm


@pytest.fixture
def sim_tab(qt_offscreen, tmp_path, monkeypatch):
    """Simulations tab with a monomer/dimer mixture loaded and no
    experimental data."""
    monomer = spinna.Structure("Monomer")
    monomer.define_coordinates(TARGET, [0.0], [0.0], [0.0])
    dimer = spinna.Structure("Dimer")
    dimer.define_coordinates(TARGET, [-5.0, 5.0], [0.0, 0.0], [0.0, 0.0])
    path = str(tmp_path / "structures.yaml")
    io.save_info(path, [monomer.get_info(), dimer.get_info()])

    # no modal dialogs in tests; record the messages instead
    messages = []
    for name in ("information", "warning"):
        monkeypatch.setattr(
            gui_spinna.QtWidgets.QMessageBox,
            name,
            lambda *args: messages.append(args[2]),
        )
    monkeypatch.setattr(
        gui_spinna.QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (path, ""),
    )

    window = gui_spinna.Window()
    tab = window.findChild(gui_spinna.SimulationsTab)
    tab.load_structures()
    for spin in tab.prop_str_input_spins:
        spin.setValue(50.0)
    tab.densities_spins[0].setValue(DENSITY)
    tab.le_spins[0].setValue(LE)
    tab.save_sim_result_check.setChecked(False)
    tab.messages = messages
    yield tab
    window.close()


def _roi_size(roi: list) -> float:
    """ROI area (um^2) or volume (um^3) from width, height, depth (nm)."""
    width, height, depth = roi
    if depth is None:
        return width * height * 1e-6
    return width * height * depth * 1e-9


@pytest.mark.parametrize("dim", ["2D", "3D"])
def test_single_sim_without_exp_data(sim_tab, dim):
    assert not sim_tab.check_exp_loaded()
    if dim == "3D":
        sim_tab.dim_widget.setCurrentIndex(1)
        sim_tab.depth = DEPTH

    sim_tab.run_single_sim()

    assert sim_tab.messages == []
    mixer = sim_tab.mixer
    assert mixer is not None
    assert mixer.roi[2] == (DEPTH if dim == "3D" else None)
    # the ROI holds SINGLE_SIM_N_MOL molecules at the observed density
    n_total = sim_tab.single_sim_n_total()
    assert n_total == pytest.approx(gui_spinna.SINGLE_SIM_N_MOL, abs=1)
    assert DENSITY * _roi_size(mixer.roi) / (LE / 100) == pytest.approx(
        gui_spinna.SINGLE_SIM_N_MOL
    )
    # the simulated NNDs are histogrammed, the experimental ones are not
    assert len(sim_tab.nnd_hist_data_sim) == 1
    assert sim_tab.nnd_hist_data_exp == []
    counts = sim_tab.nnd_hist_data_sim[0]["counts"]
    assert len(counts) > 0 and all(np.isfinite(c).all() for c in counts)


def test_single_sim_masks_without_exp_data_warns(sim_tab):
    sim_tab.mask_den_stack.setCurrentIndex(0)

    sim_tab.run_single_sim()

    assert sim_tab.mixer is None
    assert len(sim_tab.messages) == 1
    assert "requires experimental data" in sim_tab.messages[0]


def test_single_sim_zero_density_warns(sim_tab):
    sim_tab.densities_spins[0].setValue(0.0)

    sim_tab.run_single_sim()

    assert sim_tab.mixer is None
    assert len(sim_tab.messages) == 1
    assert "non-zero observed density" in sim_tab.messages[0]
