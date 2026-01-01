"""Test picasso functions related to undrifting.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pytest
from picasso import io, spinna

# parameters for spinna
PIXELSIZE = 130  # camera pixel size, nm
LABEL_UNC = 6.0  # label position uncertainty, nm
LE = 0.375  # labeling efficiency, 37.5%
GRANULARITY = 5
N_SIM = 1
ROI = 10_000


@pytest.fixture(scope="module")
def mols():
    """Load molecules data once per test module."""
    mols, info = io.load_locs("./tests/data/testdata_mols.hdf5")
    return mols


def test_spinna(mols):
    """Test SPINNA."""
    coords = mols[["x", "y"]].to_numpy()
    n = int(len(coords) / LE)

    # define spinna structures
    monomer = spinna.Structure(title="Monomer")
    monomer.define_coordinates(target="target", x=[0], y=[0], z=[0])
    dimer = spinna.Structure(title="Dimer")
    dimer.define_coordinates(
        target="target", x=[-10.5, 10.5], y=[0, 0], z=[0, 0]
    )
    structures = [monomer, dimer]

    # set up the mixer object which is used for simulating the structures
    mixer = spinna.StructureMixer(
        structures=structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )

    # generate parameter search space - tested stoichiometries
    search_space = spinna.generate_N_structures(
        structures=structures,
        N_total={"target": n},
        granularity=GRANULARITY,
    )

    # run SPINNA
    spinner = spinna.SPINNA(
        mixer=mixer,
        gt_coords={"target": coords},
        N_sim=N_SIM,
    )
    np.random.seed(0)  # for reproducibility

    for asynch in [False, True]:
        for bootstrap in [False, True]:
            best_proportions, best_score = spinner.fit_stoichiometry(
                N_structures=search_space,
                asynch=asynch,  # multiprocessing
                bootstrap=bootstrap,  # bootstrap resampling
            )
            assert isinstance(best_score, float) or isinstance(
                best_score, tuple
            ), f"Best score is not float or tuple for asynch={asynch} bootstrap={bootstrap}."


def test_spinna_masked(mols):
    """Test SPINNA with masked simulations."""
    coords = mols[["x", "y"]].to_numpy()
    n = int(len(coords) / LE)

    # define spinna structures
    monomer = spinna.Structure(title="Monomer")
    monomer.define_coordinates(target="target", x=[0], y=[0], z=[0])
    dimer = spinna.Structure(title="Dimer")
    dimer.define_coordinates(
        target="target", x=[-10.5, 10.5], y=[0, 0], z=[0, 0]
    )
    structures = [monomer, dimer]

    # set up the mixer object which is used for simulating the structures
    mask, mask_info = io.load_mask("./tests/data/testdata_mask_spinna.npy")
    mask = {"target": mask}
    mask_info = {"target": mask_info}
    mixer = spinna.StructureMixer(
        structures=structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        mask_dict={"mask": mask, "info": mask_info},
    )

    # generate parameter search space - tested stoichiometries
    search_space = spinna.generate_N_structures(
        structures=structures,
        N_total={"target": n},
        granularity=GRANULARITY,
    )

    # run SPINNA
    spinner = spinna.SPINNA(
        mixer=mixer,
        gt_coords={"target": coords},
        N_sim=N_SIM,
    )
    np.random.seed(0)  # for reproducibility
    best_proportions, best_score = spinner.fit_stoichiometry(
        N_structures=search_space,
        asynch=False,
    )
    assert isinstance(
        best_score, float
    ), f"Best score is not float for masked SPINNA."
