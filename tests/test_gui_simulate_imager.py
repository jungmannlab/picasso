"""The photon rate that Picasso: Simulate simulates.

Outside the advanced mode, the photon rate and its std are hidden and
follow from the power density and the photon detection rate; the
simulation must use the photons per frame that the window shows.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest

from picasso.gui import simulate as gui_simulate


@pytest.fixture
def window(qt_offscreen, restore_theme):
    window = gui_simulate.Window()
    yield window
    window.close()


def _simulated_photons(window) -> tuple[float, float]:
    """Mean and std of the photons per frame the simulation uses."""
    params = window._read_simulate_params()
    return (
        params["photonrate"] * params["itime"],
        params["photonratestd"] * params["itime"],
    )


def _shown_photons(window) -> tuple[float, float]:
    return (
        float(window.photonsframeEdit.text()),
        float(window.photonstdframeEdit.text()),
    )


def test_defaults_simulate_the_photons_shown(window):
    assert window.photonrateEdit.value() == pytest.approx(
        gui_simulate.PHOTONSLOPE_DEFAULT * gui_simulate.LASERPOWER_DEFAULT
    )
    assert _simulated_photons(window) == pytest.approx(
        _shown_photons(window), abs=0.5
    )


@pytest.mark.parametrize("power_density", [0.05, 2.0, 10.0])
def test_power_density_sets_the_photon_rate(window, power_density):
    # 10 kW cm^-2 at the default detection rate is 13,000 photons per ms.
    window.laserpowerEdit.setValue(power_density)
    assert window.photonrateEdit.value() == pytest.approx(
        gui_simulate.PHOTONSLOPE_DEFAULT * power_density
    )
    assert _simulated_photons(window) == pytest.approx(
        _shown_photons(window), abs=0.5
    )
