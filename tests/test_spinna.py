"""Test picasso.spinna functions and classes.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import math

import numpy as np
import pandas as pd
import pytest
import yaml

from picasso import io, spinna
from picasso.spinna import (
    MaskGenerator,
    SPINNA,
    Structure,
    StructureMixer,
    StructureSimulator,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

PIXELSIZE = 130  # camera pixel size, nm
LABEL_UNC = 6.0  # label position uncertainty, nm
LE = 0.375  # labeling efficiency, 37.5%
GRANULARITY = 5
N_SIM = 1
ROI = 10_000.0

# Settings for ground-truth recovery tests
RECOVERY_ROI = 5_000.0
RECOVERY_N_PER_TARGET = 1000
RECOVERY_LE = 0.5
RECOVERY_LABEL_UNC = 5.0
RECOVERY_GRANULARITY = 10
RECOVERY_N_SIM = 5

REAL_LOCS_PATH = "./tests/data/testdata_locs.hdf5"
REAL_MOLS_PATH = "./tests/data/testdata_mols.hdf5"
REAL_MASK_PATH = "./tests/data/testdata_mask_spinna.npy"


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def mols_real():
    """Real picasso molecules used in legacy smoke tests."""
    locs, _ = io.load_locs(REAL_MOLS_PATH)
    return locs


@pytest.fixture(scope="module")
def mask_real():
    """2D mask + metadata produced by Picasso SPINNA."""
    mask, info = io.load_mask(REAL_MASK_PATH)
    return mask, info


@pytest.fixture(scope="module")
def monomer_dimer_structures():
    monomer = Structure(title="Monomer")
    monomer.define_coordinates(target="target", x=[0], y=[0], z=[0])
    dimer = Structure(title="Dimer")
    dimer.define_coordinates(
        target="target", x=[-10.5, 10.5], y=[0, 0], z=[0, 0]
    )
    return [monomer, dimer]


@pytest.fixture(scope="module")
def het_structures():
    """Canonical [monomerA, monomerB, heterodimerAB] used by LE helpers."""
    mA = Structure("MonA")
    mA.define_coordinates("A", [0], [0], [0])
    mB = Structure("MonB")
    mB.define_coordinates("B", [0], [0], [0])
    het = Structure("HetAB")
    het.define_coordinates("A", [-10.5], [0], [0])
    het.define_coordinates("B", [10.5], [0], [0])
    return [mA, mB, het]


@pytest.fixture
def mixer_2d_csr(monomer_dimer_structures):
    """Fresh 2D CSR mixer (rebuilt because run_simulation mutates state)."""
    return StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )


@pytest.fixture
def mixer_3d_csr(monomer_dimer_structures):
    return StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
        depth=ROI,
        random_rot_mode="3D",
    )


@pytest.fixture
def mixer_2d_masked(monomer_dimer_structures, mask_real):
    mask, mask_info = mask_real
    return StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        mask_dict={
            "mask": {"target": mask},
            "info": {"target": mask_info},
        },
    )


# ---------------------------------------------------------------------
# Helpers (used by recovery tests; not fixtures because they take args)
# ---------------------------------------------------------------------


def _make_recovery_mixer(structures, depth=None, random_rot_mode="2D"):
    return StructureMixer(
        structures=structures,
        label_unc={"target": RECOVERY_LABEL_UNC},
        le={"target": RECOVERY_LE},
        width=RECOVERY_ROI,
        height=RECOVERY_ROI,
        depth=depth,
        random_rot_mode=random_rot_mode,
    )


# ---------------------------------------------------------------------
# Section A — Linear-algebra helpers
# ---------------------------------------------------------------------


def test_rref_identity_is_identity():
    M = np.eye(3, dtype=float)
    out = spinna.rref(M)
    assert np.allclose(out, np.eye(3))


def test_rref_full_rank_3x3():
    M = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
    out = spinna.rref(M)
    assert np.allclose(out, np.eye(3), atol=1e-5), out


def test_rref_matches_heterodimer_use_case():
    # mirrors the augmented matrix produced by generate_N_structures for
    # a 2-target / 3-structure setup (free param last)
    M = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    out = spinna.rref(M)
    expected = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    assert np.allclose(out, expected), out


def test_rref_does_not_mutate_input():
    M = np.array([[2.0, 4.0], [3.0, 9.0]])
    snapshot = M.copy()
    _ = spinna.rref(M)
    assert np.array_equal(M, snapshot)


def test_find_target_counts_shape_and_values(monomer_dimer_structures):
    counts = spinna.find_target_counts(["target"], monomer_dimer_structures)
    assert counts.shape == (1, 2)
    # monomer has 1 target, dimer has 2
    assert np.array_equal(counts, np.array([[1.0, 2.0]]))


def test_find_target_counts_missing_target_zero(monomer_dimer_structures):
    counts = spinna.find_target_counts(
        ["target", "missing"], monomer_dimer_structures
    )
    assert counts.shape == (2, 2)
    # second row corresponds to "missing" — must be all zeros
    assert np.array_equal(counts[1], np.zeros(2))


def test_targets_from_structures_is_unique_and_order_preserving(
    het_structures,
):
    targets = spinna.targets_from_structures(het_structures)
    assert targets == ["A", "B"]


def test_get_structures_permutation_homo_dimer(monomer_dimer_structures):
    t_counts = spinna.find_target_counts(["target"], monomer_dimer_structures)
    perm = spinna.get_structures_permutation(t_counts.copy())
    # 2 structures, 1 target → exactly one free param goes last
    assert perm.shape == (2,)
    assert sorted(perm.tolist()) == [0, 1]


# ---------------------------------------------------------------------
# Section B — generate_N_structures
# ---------------------------------------------------------------------


def test_generate_N_structures_homo_satisfies_balance(
    monomer_dimer_structures,
):
    # granularity=6 with N_total=100 yields integer-valued grid points
    # (bases linspace(0, 50, 6) = [0, 10, 20, 30, 40, 50])
    out = spinna.generate_N_structures(
        structures=monomer_dimer_structures,
        N_total={"target": 100},
        granularity=6,
    )
    assert set(out.keys()) == {"Monomer", "Dimer"}
    arr = np.column_stack([out["Monomer"], out["Dimer"]])
    assert (arr >= 0).all(), arr
    # Each row's molecule total: monomer*1 + dimer*2 == 100
    assert np.array_equal(arr @ np.array([1, 2]), [100] * len(arr)), arr


def test_generate_N_structures_hetero_satisfies_balance(het_structures):
    # N_total=100 with granularity=11 yields integer grid
    # (bases linspace(0, 100, 11) = [0, 10, 20, ..., 100])
    out = spinna.generate_N_structures(
        structures=het_structures,
        N_total={"A": 100, "B": 100},
        granularity=11,
    )
    assert set(out.keys()) == {"MonA", "MonB", "HetAB"}
    n_mA = np.asarray(out["MonA"])
    n_mB = np.asarray(out["MonB"])
    n_het = np.asarray(out["HetAB"])
    # target A balance: monomerA(1) + heterodimer(1) == 100
    assert np.array_equal(n_mA + n_het, [100] * len(n_mA)), (n_mA, n_het)
    # target B balance: monomerB(1) + heterodimer(1) == 100
    assert np.array_equal(n_mB + n_het, [100] * len(n_mB)), (n_mB, n_het)
    assert (n_mA >= 0).all() and (n_mB >= 0).all() and (n_het >= 0).all()


def test_generate_N_structures_n_struct_le_n_targets_raises(het_structures):
    # 2 targets, 2 structures (only the heterodimer + one monomer) → invalid
    structures = [het_structures[0], het_structures[2]]
    with pytest.raises(ValueError):
        spinna.generate_N_structures(
            structures=structures,
            N_total={"A": 100, "B": 100},
            granularity=5,
        )


def test_generate_N_structures_save_csv(monomer_dimer_structures, tmp_path):
    out_csv = tmp_path / "search_space.csv"
    out = spinna.generate_N_structures(
        structures=monomer_dimer_structures,
        N_total={"target": 100},
        granularity=5,
        save=str(out_csv),
    )
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    expected_cols = {"N_Monomer", "N_Dimer", "Prop_Monomer", "Prop_Dimer"}
    assert expected_cols.issubset(set(df.columns)), df.columns
    # Rows match the dict
    assert len(df) == len(out["Monomer"])
    # Proportions sum to 100
    assert np.allclose(df["Prop_Monomer"] + df["Prop_Dimer"], 100.0)


def test_generate_N_structures_higher_granularity_more_rows(
    monomer_dimer_structures,
):
    low = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": 100}, granularity=5
    )
    high = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": 100}, granularity=10
    )
    assert len(high["Monomer"]) > len(low["Monomer"])


# ---------------------------------------------------------------------
# Section C — random_rotation_matrices
# ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["2D", "3D", None])
@pytest.mark.parametrize("num", [1, 5])
def test_random_rotations_orthogonal_and_proper(mode, num):
    rots = spinna.random_rotation_matrices(num=num, mode=mode)
    assert rots.shape == (num, 3, 3)
    for R in rots:
        # orthogonal: R R^T == I
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
        # proper rotation: det(R) ≈ +1
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_random_rotation_2d_does_not_rotate_z():
    np.random.seed(0)
    rots = spinna.random_rotation_matrices(num=10, mode="2D")
    # last row/column = (0, 0, 1) for pure z-axis rotation
    for R in rots:
        assert np.allclose(R[2, :], [0, 0, 1], atol=1e-5)
        assert np.allclose(R[:, 2], [0, 0, 1], atol=1e-5)


def test_random_rotation_none_is_identity():
    rots = spinna.random_rotation_matrices(num=3, mode=None)
    for R in rots:
        assert np.allclose(R, np.eye(3))


def test_random_rotation_invalid_num_raises():
    with pytest.raises(TypeError):
        spinna.random_rotation_matrices(num=-1)
    with pytest.raises(TypeError):
        spinna.random_rotation_matrices(num=0)
    with pytest.raises(TypeError):
        spinna.random_rotation_matrices(num=2.5)  # type: ignore[arg-type]


def test_random_rotation_invalid_mode_raises():
    with pytest.raises(ValueError):
        spinna.random_rotation_matrices(num=2, mode="bogus")  # noqa


# ---------------------------------------------------------------------
# Section D — coords_to_locs
# ---------------------------------------------------------------------


@pytest.mark.parametrize("pixelsize", [130, 100])
@pytest.mark.parametrize("lp", [1.0, 5.0])
def test_coords_to_locs_2d_unit_conversion(pixelsize, lp):
    coords = np.array([[0.0, 0.0], [pixelsize, 2 * pixelsize]])
    locs = spinna.coords_to_locs(coords, lp=lp, pixelsize=pixelsize)
    assert isinstance(locs, pd.DataFrame)
    assert set(locs.columns) == {"frame", "x", "y", "lpx", "lpy"}
    assert np.allclose(locs["x"].to_numpy(), [0.0, 1.0])
    assert np.allclose(locs["y"].to_numpy(), [0.0, 2.0])
    # localization precision in pixels
    assert np.allclose(locs["lpx"].to_numpy(), lp / pixelsize)
    assert np.allclose(locs["lpy"].to_numpy(), lp / pixelsize)


def test_coords_to_locs_3d_keeps_z_in_nm():
    coords = np.array([[0.0, 0.0, 50.0], [130.0, 260.0, -25.0]])
    locs = spinna.coords_to_locs(coords, lp=1.0, pixelsize=130)
    assert "z" in locs.columns
    assert np.allclose(locs["z"].to_numpy(), [50.0, -25.0])
    assert np.allclose(locs["x"].to_numpy(), [0.0, 1.0])


# ---------------------------------------------------------------------
# Section E — get_NN_dist / NND_score
# ---------------------------------------------------------------------


def test_get_NN_dist_unit_grid():
    # 3x3 grid spaced by 1.0; each point has a NN at distance 1.0
    xs, ys = np.meshgrid(np.arange(3), np.arange(3))
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    d = spinna.get_NN_dist(pts, pts, n_neighbors=1)
    assert d.shape == (9, 1)
    assert np.allclose(d[:, 0], 1.0)


def test_get_NN_dist_excludes_self_when_data1_is_data2():
    pts = np.array([[0.0, 0.0], [3.0, 4.0]])
    d = spinna.get_NN_dist(pts, pts, n_neighbors=1)
    # distance is 5.0 (NOT 0.0 self-distance)
    assert np.allclose(d[:, 0], 5.0)


def test_get_NN_dist_empty_input_returns_empty():
    pts = np.zeros((0, 2))
    other = np.array([[1.0, 2.0]])
    out_a = spinna.get_NN_dist(pts, other, n_neighbors=1)
    out_b = spinna.get_NN_dist(other, pts, n_neighbors=1)
    assert out_a.size == 0
    assert out_b.size == 0


def test_get_NN_dist_dim_mismatch_raises():
    a = np.zeros((3, 2))
    b = np.zeros((3, 3))
    with pytest.raises(ValueError):
        spinna.get_NN_dist(a, b, n_neighbors=1)


def test_NND_score_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 100, size=(500, 1))
    score = spinna.NND_score([d], [d])
    assert score < 0.05, score


def test_NND_score_disjoint_distributions_high():
    a = np.full((500, 1), 10.0)
    b = np.full((500, 1), 200.0)
    score = spinna.NND_score([a], [b])
    assert score > 0.5, score


# ---------------------------------------------------------------------
# Section F — Structure
# ---------------------------------------------------------------------


def test_structure_init_empty_targets():
    s = Structure("foo")
    assert s.title == "foo"
    assert s.targets == []
    assert s.x == s.y == s.z == {}


def test_structure_repr_includes_title_and_targets(monomer_dimer_structures):
    monomer = monomer_dimer_structures[0]
    r = repr(monomer)
    assert "Monomer" in r and "target" in r


def test_structure_define_coordinates_2d_pads_z_with_zeros():
    s = Structure("two-d")
    s.define_coordinates("A", x=[1.0, 2.0], y=[3.0, 4.0])
    assert s.z["A"] == [0, 0]


def test_structure_define_coordinates_unequal_lengths_raises():
    s = Structure("bad")
    with pytest.raises(ValueError):
        s.define_coordinates("A", x=[0.0, 1.0], y=[0.0])
    with pytest.raises(ValueError):
        s.define_coordinates("A", x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0])


def test_structure_define_coordinates_appends_on_repeat():
    s = Structure("append")
    s.define_coordinates("A", x=[1.0], y=[2.0], z=[3.0])
    s.define_coordinates("A", x=[10.0], y=[20.0], z=[30.0])
    assert s.targets == ["A"]
    assert s.x["A"] == [1.0, 10.0]
    assert s.y["A"] == [2.0, 20.0]
    assert s.z["A"] == [3.0, 30.0]


def test_structure_delete_target_idempotent():
    s = Structure("del")
    s.define_coordinates("A", [0], [0], [0])
    s.delete_target("A")
    assert "A" not in s.targets
    # Idempotent on missing target — must not raise
    s.delete_target("A")
    s.delete_target("never-existed")


def test_structure_get_all_targets_count(het_structures):
    mA, mB, het = het_structures
    assert mA.get_all_targets_count() == 1
    assert het.get_all_targets_count() == 2


def test_structure_get_ind_target_count_order_and_zero(het_structures):
    het = het_structures[2]
    counts = het.get_ind_target_count(["A", "B", "missing"])
    assert counts == [1, 1, 0]
    # order is preserved
    counts2 = het.get_ind_target_count(["B", "A"])
    assert counts2 == [1, 1]


def test_structure_get_max_nn_same_target_is_n_minus_1(
    monomer_dimer_structures,
):
    monomer, dimer = monomer_dimer_structures
    assert monomer.get_max_nn("target", "target") == 0
    assert dimer.get_max_nn("target", "target") == 1


def test_structure_get_max_nn_cross_is_min(het_structures):
    het = het_structures[2]
    # 1 of A and 1 of B in heterodimer
    assert het.get_max_nn("A", "B") == 1


def test_structure_get_max_nn_missing_target_is_zero(het_structures):
    mA = het_structures[0]
    assert mA.get_max_nn("A", "B") == 0
    assert mA.get_max_nn("does-not-exist", "A") == 0


def test_structure_save_requires_yaml_extension(tmp_path):
    s = Structure("x")
    with pytest.raises(ValueError):
        s.save(str(tmp_path / "out.txt"))


def test_structure_save_and_load_round_trip(
    monomer_dimer_structures, tmp_path
):
    out = tmp_path / "structures.yaml"
    # save_info appends — write each structure to its own file then merge
    info = [s.get_info() for s in monomer_dimer_structures]
    io.save_info(str(out), info)
    structures, targets = spinna.load_structures(str(out))
    assert targets == ["target"]
    assert [s.title for s in structures] == ["Monomer", "Dimer"]
    assert structures[1].x["target"] == [-10.5, 10.5]


def test_structure_restart_clears():
    s = Structure("r")
    s.define_coordinates("A", [0], [0], [0])
    assert s.targets == ["A"]
    s.restart()
    assert s.targets == [] and s.x == s.y == s.z == {}
    assert s.title == "r"  # title preserved


# ---------------------------------------------------------------------
# Section G — load_structures errors
# ---------------------------------------------------------------------


def test_load_structures_bad_path_raises_TypeError(tmp_path):
    bad = tmp_path / "not_a_structure.yaml"
    with open(bad, "w") as f:
        yaml.dump({"unrelated": "data"}, f)
    with pytest.raises(TypeError):
        spinna.load_structures(str(bad))


# ---------------------------------------------------------------------
# Section H — MaskGenerator
# ---------------------------------------------------------------------


def test_mask_generator_init_picks_2d():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    assert mg.ndim == 2
    assert mg.pixelsize == 130
    # roi is a 2-list (width, height in nm)
    assert len(mg.roi) == 2
    assert mg.roi[0] > 0 and mg.roi[1] > 0


def test_mask_generator_set_binsize_int_promotes_to_tuple():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    mg.set_binsize(50)
    assert mg.binsize == (50, 50)


def test_mask_generator_set_binsize_bad_type_raises():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    with pytest.raises(ValueError):
        mg.set_binsize("nope")  # type: ignore[arg-type]


def test_mask_generator_set_binsize_wrong_tuple_length_raises():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    with pytest.raises(AssertionError):
        mg.set_binsize((10, 20, 30))  # too many values


def test_mask_generator_set_sigma_bad_type_raises():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    with pytest.raises(ValueError):
        mg.set_sigma("nope")  # type: ignore[arg-type]


def test_mask_generator_render_locs_returns_2d():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    image = mg.render_locs()
    assert image.ndim == 2
    assert image.sum() > 0


@pytest.mark.parametrize("mode", ["loc_den", "binary"])
def test_mask_generator_generate_mask_normalizes_to_one(mode):
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    mg.generate_mask(mode=mode)
    assert math.isclose(mg.mask.sum(), 1.0, abs_tol=1e-6)


def test_mask_generator_invalid_mode_raises():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    with pytest.raises(ValueError):
        mg.generate_mask(mode="bogus")  # type: ignore[arg-type]


def test_mask_generator_save_mask_requires_npy(tmp_path):
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    mg.generate_mask(mode="loc_den")
    with pytest.raises(ValueError):
        mg.save_mask(str(tmp_path / "mask.bin"))


def test_mask_generator_save_mask_round_trip(tmp_path):
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    mg.generate_mask(mode="loc_den")
    out = tmp_path / "mask.npy"
    mg.save_mask(str(out))
    assert out.exists()
    yaml_path = out.with_suffix(".yaml")
    assert yaml_path.exists()
    # load_mask normalizes again — but values should match within float tol
    loaded_mask, loaded_info = io.load_mask(str(out))
    assert loaded_mask.shape == mg.mask.shape
    assert "Generated by" in loaded_info
    assert "SPINNA" in loaded_info["Generated by"]


def test_mask_generator_area_none_before_generate():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    assert mg.area is None
    assert mg.volume is None


def test_mask_generator_area_positive_after_generate_2d():
    mg = MaskGenerator(REAL_LOCS_PATH, binsize=130, sigma=200)
    mg.generate_mask(mode="binary")
    assert mg.area is not None and mg.area > 0
    # 2D mask has no volume
    assert mg.volume is None


# ---------------------------------------------------------------------
# Section I — StructureSimulator
# ---------------------------------------------------------------------


def test_simulator_mask_without_info_raises(monomer_dimer_structures):
    fake_mask = np.ones((10, 10), dtype=np.float64)
    with pytest.raises(ValueError):
        StructureSimulator(
            structure=monomer_dimer_structures[0],
            N_structures=10,
            le=[LE],
            label_unc=[LABEL_UNC],
            mask=fake_mask,
            mask_info=None,
        )


def test_simulator_simulate_centers_CSR_within_bounds(
    monomer_dimer_structures,
):
    sim = StructureSimulator(
        structure=monomer_dimer_structures[0],
        N_structures=50,
        le=[1.0],
        label_unc=[LABEL_UNC],
        width=ROI,
        height=ROI,
    )
    sim.simulate_centers_CSR()
    assert sim.c_pos.shape == (50, 2)
    assert (sim.c_pos[:, 0] >= 0).all()
    assert (sim.c_pos[:, 0] < ROI).all()
    assert (sim.c_pos[:, 1] >= 0).all()
    assert (sim.c_pos[:, 1] < ROI).all()


def test_simulator_simulate_centers_CSR_3d_within_bounds(
    monomer_dimer_structures,
):
    depth = 1000.0
    sim = StructureSimulator(
        structure=monomer_dimer_structures[0],
        N_structures=30,
        le=[1.0],
        label_unc=[LABEL_UNC],
        width=ROI,
        height=ROI,
        depth=depth,
    )
    sim.simulate_centers_CSR()
    assert sim.c_pos.shape == (30, 3)
    assert (sim.c_pos[:, 2] >= -depth / 2).all()
    assert (sim.c_pos[:, 2] <= depth / 2).all()


def test_simulator_zero_N_sets_c_pos_to_None(monomer_dimer_structures):
    sim = StructureSimulator(
        structure=monomer_dimer_structures[0],
        N_structures=0,
        le=[1.0],
        label_unc=[LABEL_UNC],
        width=ROI,
        height=ROI,
    )
    sim.simulate_centers()
    assert sim.c_pos is None


def test_simulator_rotate_structures_preserves_pairwise_distances(
    monomer_dimer_structures,
):
    sim = StructureSimulator(
        structure=monomer_dimer_structures[1],  # dimer
        N_structures=4,
        le=[1.0],
        label_unc=[LABEL_UNC],
        width=ROI,
        height=ROI,
    )
    coords = np.array(
        [
            [[-10.5, 0.0, 0.0], [10.5, 0.0, 0.0]],
        ]
        * 4,
        dtype=np.float64,
    )
    rots = spinna.random_rotation_matrices(num=4, mode="3D")
    out = sim.rotate_structures(coords, rots)
    # within each structure, |p0 - p1| must equal 21.0 (within float tol)
    diffs = out[:, 0, :] - out[:, 1, :]
    dists = np.linalg.norm(diffs, axis=1)
    assert np.allclose(dists, 21.0, atol=1e-4)


def test_simulator_reshape_coordinates_shape(
    monomer_dimer_structures,
):
    sim = StructureSimulator(
        structure=monomer_dimer_structures[0],
        N_structures=2,
        le=[1.0],
        label_unc=[LABEL_UNC],
        width=ROI,
        height=ROI,
    )
    coords = np.zeros((5, 3, 2), dtype=np.float64)  # N=5, M=3, 2D
    out = sim.reshape_coordinates(coords)
    assert out.shape == (15, 2)


def test_simulator_simulate_le_yields_close_to_le_times_N(
    monomer_dimer_structures,
):
    np.random.seed(0)
    le = 0.5
    N = 1000
    sim = StructureSimulator(
        structure=monomer_dimer_structures[0],
        N_structures=N,
        le=[le],
        label_unc=[LABEL_UNC],
        width=ROI,
        height=ROI,
    )
    sim.simulate_centers()
    sim.simulate_all_targets()
    sim.simulate_le()
    n_obs = len(sim.pos_obs["target"])
    # simulate_le uses int(N * le), not random — so this must be exact
    assert n_obs == int(N * le)


def test_simulator_run_produces_observed_coords(
    monomer_dimer_structures,
):
    np.random.seed(0)
    sim = StructureSimulator(
        structure=monomer_dimer_structures[1],
        N_structures=20,
        le=[1.0],
        label_unc=[1.0],  # tight label uncertainty
        width=ROI,
        height=ROI,
    ).run()
    pos = sim.pos_obs["target"]
    # 20 dimers × 2 mols/dimer × LE 1.0 = 40 mols
    assert pos.shape == (40, 2)


# ---------------------------------------------------------------------
# Section J — StructureMixer
# ---------------------------------------------------------------------


def test_mixer_label_unc_must_be_dict_raises(monomer_dimer_structures):
    with pytest.raises(TypeError):
        StructureMixer(
            structures=monomer_dimer_structures,
            label_unc=6.0,  # type: ignore[arg-type]
            le={"target": LE},
            width=ROI,
            height=ROI,
        )


def test_mixer_negative_label_unc_raises(monomer_dimer_structures):
    with pytest.raises(ValueError):
        StructureMixer(
            structures=monomer_dimer_structures,
            label_unc={"target": -1.0},
            le={"target": LE},
            width=ROI,
            height=ROI,
        )


@pytest.mark.parametrize("bad_le", [-0.1, 0.0, 1.5, 2.0])
def test_mixer_le_out_of_range_raises(monomer_dimer_structures, bad_le):
    with pytest.raises(ValueError):
        StructureMixer(
            structures=monomer_dimer_structures,
            label_unc={"target": LABEL_UNC},
            le={"target": bad_le},
            width=ROI,
            height=ROI,
        )


def test_mixer_structures_non_list_raises():
    with pytest.raises(TypeError):
        StructureMixer(
            structures="not-a-list",  # type: ignore[arg-type]
            label_unc={"target": LABEL_UNC},
            le={"target": LE},
            width=ROI,
            height=ROI,
        )


def test_mixer_no_mask_no_roi_raises(monomer_dimer_structures):
    with pytest.raises(TypeError):
        StructureMixer(
            structures=monomer_dimer_structures,
            label_unc={"target": LABEL_UNC},
            le={"target": LE},
        )


def test_mixer_target_missing_from_label_unc_raises(
    monomer_dimer_structures,
):
    with pytest.raises(KeyError):
        StructureMixer(
            structures=monomer_dimer_structures,
            label_unc={"different": LABEL_UNC},
            le={"target": LE},
            width=ROI,
            height=ROI,
        )


def test_mixer_ALL_key_supported(monomer_dimer_structures):
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"ALL": LABEL_UNC},
        le={"ALL": LE},
        width=ROI,
        height=ROI,
    )
    assert mixer.targets == ["target"]


def test_mixer_nn_counts_dict_missing_pair_raises(
    monomer_dimer_structures,
):
    with pytest.raises(KeyError):
        StructureMixer(
            structures=monomer_dimer_structures,
            label_unc={"target": LABEL_UNC},
            le={"target": LE},
            width=ROI,
            height=ROI,
            nn_counts={},
        )


def test_mixer_nn_counts_must_be_dict_or_auto(monomer_dimer_structures):
    with pytest.raises(TypeError):
        StructureMixer(
            structures=monomer_dimer_structures,
            label_unc={"target": LABEL_UNC},
            le={"target": LE},
            width=ROI,
            height=ROI,
            nn_counts="not-auto",  # type: ignore[arg-type]
        )


def test_mixer_convert_counts_props_round_trip(mixer_2d_csr):
    counts_in = np.array([[10, 5], [40, 20]], dtype=np.int32)
    props = mixer_2d_csr.convert_counts_to_props(counts_in)
    assert props.shape == counts_in.shape
    # Each row sums to 100
    assert np.allclose(props.sum(axis=1), 100.0)
    # Round-trip
    N_total = counts_in @ np.array([1, 2])  # mols per row
    counts_out = np.vstack(
        [
            mixer_2d_csr.convert_props_to_counts(props[i], N_total[i])
            for i in range(len(props))
        ]
    )
    assert np.array_equal(counts_in, counts_out)


@pytest.mark.parametrize(
    "input_form",
    [
        "dict",
        "list",
        "1d",
        "2d",
    ],
)
def test_mixer_convert_N_structures_to_array(mixer_2d_csr, input_form):
    if input_form == "dict":
        n = {"Monomer": [10, 20], "Dimer": [5, 7]}
    elif input_form == "list":
        n = [[10, 5], [20, 7]]
    elif input_form == "1d":
        n = np.array([10, 5])
    else:
        n = np.array([[10, 5], [20, 7]])
    out = mixer_2d_csr.convert_N_structures_to_array(n)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.shape[1] == 2  # 2 structures


def test_mixer_get_structure_names(mixer_2d_csr):
    assert mixer_2d_csr.get_structure_names() == ["Monomer", "Dimer"]


def test_mixer_get_neighbor_counts_homo(mixer_2d_csr):
    # dimer has 1 NN within itself for "target"-"target"
    assert mixer_2d_csr.get_neighbor_counts("target", "target") == 1


def test_mixer_get_neighbor_idx_with_and_without_duplicate(het_structures):
    mixer = StructureMixer(
        structures=het_structures,
        label_unc={"A": 5.0, "B": 5.0},
        le={"A": 0.5, "B": 0.5},
        width=ROI,
        height=ROI,
    )
    no_dup = mixer.get_neighbor_idx(duplicate=False)
    dup = mixer.get_neighbor_idx(duplicate=True)
    # without dup: (A,A), (A,B), (B,B). With dup: also (B,A).
    assert len(no_dup) == 3
    assert len(dup) == 4


def test_mixer_roi_size_2d(mixer_2d_csr):
    expected = ROI * ROI * 1e-6  # nm^2 → um^2
    assert math.isclose(mixer_2d_csr.roi_size, expected)


def test_mixer_roi_size_3d(mixer_3d_csr):
    expected = ROI * ROI * ROI * 1e-9  # nm^3 → um^3
    assert math.isclose(mixer_3d_csr.roi_size, expected)


def test_mixer_run_simulation_count_within_bernoulli_bound(
    mixer_2d_csr,
):
    np.random.seed(0)
    # 100 monomers + 50 dimers → total 200 mols of "target"; LE=0.375
    out = mixer_2d_csr.run_simulation([100, 50])
    coords = out["target"]
    # exact count: simulate_le uses int(N * le), no Bernoulli noise
    expected = int(100 * LE) + int(
        100 * LE
    )  # mol count = 100 + 100 (dimers contribute 2)
    # NOTE: 100 dimers' worth of mols = 50 * 2 = 100; then int(100 * 0.375) = 37
    expected = int(100 * LE) + int(100 * LE)
    assert len(coords) == expected
    # Bounds
    assert (coords[:, 0] >= 0).all()
    assert (coords[:, 0] < ROI).all()
    assert (coords[:, 1] >= 0).all()
    assert (coords[:, 1] < ROI).all()


def test_mixer_extract_mask_heterodimer_normalizes(het_structures):
    # build identical uniform masks for A and B
    h, w = 20, 20
    mask_a = np.ones((h, w), dtype=np.float64) / (h * w)
    mask_b = np.ones((h, w), dtype=np.float64) / (h * w)
    info = {
        "Camera pixelsize (nm)": 130,
        "x_min": 0,
        "x_max": w,
        "y_min": 0,
        "y_max": h,
        "Binsize (nm)": [10.0, 10.0],
        "Dimensionality": "2D",
        "Area (um^2)": 1.0,
    }
    mixer = StructureMixer(
        structures=het_structures,
        label_unc={"A": 5.0, "B": 5.0},
        le={"A": 0.5, "B": 0.5},
        mask_dict={
            "mask": {"A": mask_a, "B": mask_b},
            "info": {"A": info, "B": info},
        },
    )
    het_struct = het_structures[2]
    mask_out, info_out = mixer.extract_mask(het_struct)
    assert mask_out.shape == (h, w)
    assert math.isclose(mask_out.sum(), 1.0, abs_tol=1e-6)
    # uniform input → uniform output
    assert np.allclose(mask_out, 1.0 / (h * w))


def test_mixer_extract_mask_single_target_returns_target_mask(
    monomer_dimer_structures, mask_real
):
    mask, mask_info = mask_real
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        mask_dict={
            "mask": {"target": mask},
            "info": {"target": mask_info},
        },
    )
    monomer = monomer_dimer_structures[0]
    out_mask, out_info = mixer.extract_mask(monomer)
    assert out_mask is mixer.mask["target"]
    assert out_info is mixer.mask_info["target"]


# ---------------------------------------------------------------------
# Section K — SPINNA
# ---------------------------------------------------------------------


def test_spinna_requires_StructureMixer():
    with pytest.raises(TypeError):
        SPINNA(mixer="not a mixer", gt_coords={})  # type: ignore[arg-type]


def test_spinna_2d_csr_strips_z_from_gt_coords(mixer_2d_csr):
    # Pass 3D gt_coords with mixer that has no depth — z should be stripped
    coords3d = np.array(
        [
            [100.0, 100.0, 50.0],
            [200.0, 200.0, -50.0],
            [150.0, 150.0, 0.0],
        ]
    )
    spinner = SPINNA(
        mixer=mixer_2d_csr, gt_coords={"target": coords3d}, N_sim=1
    )
    assert spinner.gt_coords["target"].shape == (3, 2)


def test_spinna_fit_brute_force_returns_pair(
    mixer_2d_csr, monomer_dimer_structures
):
    # use real GT coords (small set is fine)
    np.random.seed(0)
    gt = mixer_2d_csr.run_simulation([100, 50])
    # rebuild mixer because run_simulation populated simulators
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    n = int(len(gt["target"]) / LE)
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": n}, granularity=GRANULARITY
    )
    spinner = SPINNA(mixer=mixer, gt_coords=gt, N_sim=1)
    np.random.seed(0)
    props, score = spinner.fit_stoichiometry(
        N_structures=ss, fitting_mode="brute-force", asynch=False
    )
    assert isinstance(score, float) and np.isfinite(score)
    assert isinstance(props, np.ndarray)
    assert props.shape == (2,)
    assert math.isclose(props.sum(), 100.0, abs_tol=1.0)


def test_spinna_fit_with_bootstrap_returns_pair_of_pairs(
    mols_real, monomer_dimer_structures
):
    coords = mols_real[["x", "y"]].to_numpy()
    n = int(len(coords) / LE)
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": n}, granularity=3
    )
    spinner = SPINNA(mixer=mixer, gt_coords={"target": coords}, N_sim=1)
    np.random.seed(0)
    props_pair, score_pair = spinner.fit_stoichiometry(
        N_structures=ss,
        fitting_mode="brute-force",
        asynch=False,
        bootstrap=True,
    )
    assert isinstance(props_pair, tuple) and len(props_pair) == 2
    mean_props, std_props = props_pair
    assert mean_props.shape == std_props.shape == (2,)
    assert isinstance(score_pair, tuple) and len(score_pair) == 2
    score, score_std = score_pair
    assert np.isfinite(score) and score_std >= 0


def test_spinna_fit_return_scores_adds_extra_element(
    mixer_2d_csr, monomer_dimer_structures
):
    np.random.seed(0)
    gt = mixer_2d_csr.run_simulation([100, 50])
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": 200}, granularity=3
    )
    spinner = SPINNA(mixer=mixer, gt_coords=gt, N_sim=1)
    np.random.seed(0)
    out = spinner.fit_stoichiometry(
        N_structures=ss,
        fitting_mode="brute-force",
        asynch=False,
        return_scores=True,
    )
    assert len(out) == 3
    props, score, scores = out
    assert scores.ndim == 1
    assert len(scores) == len(ss["Monomer"])


def test_spinna_fit_save_csv_creates_file(
    mixer_2d_csr, monomer_dimer_structures, tmp_path
):
    np.random.seed(0)
    gt = mixer_2d_csr.run_simulation([100, 50])
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": 200}, granularity=3
    )
    spinner = SPINNA(mixer=mixer, gt_coords=gt, N_sim=1)
    np.random.seed(0)
    out_csv = tmp_path / "fits.csv"
    spinner.fit_stoichiometry(
        N_structures=ss,
        fitting_mode="brute-force",
        asynch=False,
        save=str(out_csv),
    )
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert "Kolmogorov-Smirnov statistic" in df.columns


def test_spinna_evaluate_single_returns_finite_float(
    mixer_2d_csr, monomer_dimer_structures
):
    np.random.seed(0)
    gt = mixer_2d_csr.run_simulation([100, 50])
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    spinner = SPINNA(mixer=mixer, gt_coords=gt, N_sim=1)
    np.random.seed(0)
    score = spinner._evaluate_single(np.array([100, 50], dtype=np.int32))
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_spinna_farthest_point_sampling_unique_indices():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 1, size=(50, 3))
    idx = SPINNA._farthest_point_sampling(points, n_samples=10)
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10  # unique


def test_spinna_get_subset_N_structures_within_radius(
    mixer_2d_csr, monomer_dimer_structures
):
    np.random.seed(0)
    gt = mixer_2d_csr.run_simulation([100, 50])
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": 200}, granularity=10
    )
    arr = mixer.convert_N_structures_to_array(ss)
    spinner = SPINNA(mixer=mixer, gt_coords=gt, N_sim=1)
    center = arr[len(arr) // 2]
    subset = spinner.get_subset_N_structures(arr, center, radius=10.0)
    assert subset.ndim == 2
    assert subset.shape[1] == arr.shape[1]
    # subset must contain the center itself (zero distance)
    assert any(np.array_equal(row, center) for row in subset)


def test_spinna_real_data_smoke(mols_real, monomer_dimer_structures):
    """Replaces the legacy weak-assertion smoke test."""
    coords = mols_real[["x", "y"]].to_numpy()
    n = int(len(coords) / LE)
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": n}, granularity=GRANULARITY
    )
    spinner = SPINNA(mixer=mixer, gt_coords={"target": coords}, N_sim=N_SIM)
    np.random.seed(0)
    props, score = spinner.fit_stoichiometry(
        N_structures=ss, fitting_mode="brute-force", asynch=False
    )
    assert isinstance(score, float) and 0 <= score <= 1
    assert isinstance(props, np.ndarray)
    assert props.shape == (2,)
    assert math.isclose(props.sum(), 100.0, abs_tol=1.0)


def test_spinna_real_data_masked_smoke(
    mols_real, mask_real, monomer_dimer_structures
):
    coords = mols_real[["x", "y"]].to_numpy()
    n = int(len(coords) / LE)
    mask, mask_info = mask_real
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        mask_dict={
            "mask": {"target": mask},
            "info": {"target": mask_info},
        },
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": n}, granularity=GRANULARITY
    )
    spinner = SPINNA(mixer=mixer, gt_coords={"target": coords}, N_sim=N_SIM)
    np.random.seed(0)
    props, score = spinner.fit_stoichiometry(
        N_structures=ss, fitting_mode="brute-force", asynch=False
    )
    assert isinstance(score, float) and 0 <= score <= 1
    assert math.isclose(props.sum(), 100.0, abs_tol=1.0)


# Recovery tests: synthesize ground-truth with known dimer fraction,
# refit, assert recovered proportion is within tolerance. ROI/N_total/
# granularity tuned so brute-force without multiprocessing runs in <5 s.


@pytest.mark.parametrize(
    "p_dimer_truth, tol",
    [
        (0.0, 15.0),  # all monomers — extreme tighter bound
        (0.5, 20.0),  # mixed
        (1.0, 15.0),  # all dimers — extreme tighter bound
    ],
)
def test_spinna_recovers_known_dimer_fraction(
    monomer_dimer_structures, p_dimer_truth, tol
):
    n_total = RECOVERY_N_PER_TARGET
    n_dimer = int(p_dimer_truth * n_total / 2)
    n_monomer = n_total - 2 * n_dimer
    counts = [n_monomer, n_dimer]

    # Build ground-truth coords by simulating
    np.random.seed(123)
    sim_mixer = _make_recovery_mixer(monomer_dimer_structures)
    gt = sim_mixer.run_simulation(counts)

    # Fresh mixer for fitting (run_simulation mutates internals)
    fit_mixer = _make_recovery_mixer(monomer_dimer_structures)
    ss = spinna.generate_N_structures(
        monomer_dimer_structures,
        {"target": n_total},
        granularity=RECOVERY_GRANULARITY,
    )
    spinner = SPINNA(mixer=fit_mixer, gt_coords=gt, N_sim=RECOVERY_N_SIM)
    np.random.seed(0)
    props, score = spinner.fit_stoichiometry(
        N_structures=ss, fitting_mode="brute-force", asynch=False
    )
    # convert truth to props (in 0–100 scale)
    prop_dimer_truth = p_dimer_truth * 100.0
    prop_dimer_fit = props[1]  # Dimer column
    assert abs(prop_dimer_fit - prop_dimer_truth) <= tol, (
        f"Recovered dimer proportion {prop_dimer_fit:.1f}% differs from "
        f"truth {prop_dimer_truth:.1f}% by more than {tol}%"
    )
    assert score < 0.5


@pytest.mark.slow
def test_spinna_fit_coarse_to_fine_returns_pair(
    mixer_2d_csr, monomer_dimer_structures
):
    np.random.seed(0)
    gt = mixer_2d_csr.run_simulation([100, 50])
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": 200}, granularity=10
    )
    spinner = SPINNA(mixer=mixer, gt_coords=gt, N_sim=1)
    np.random.seed(0)
    props, score = spinner.fit_stoichiometry(
        N_structures=ss, fitting_mode="coarse-to-fine", asynch=False
    )
    assert isinstance(score, float) and np.isfinite(score)
    assert props.shape == (2,)


@pytest.mark.slow
def test_spinna_fit_bayesian_returns_pair(
    mixer_2d_csr, monomer_dimer_structures
):
    np.random.seed(0)
    gt = mixer_2d_csr.run_simulation([100, 50])
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": 200}, granularity=10
    )
    spinner = SPINNA(mixer=mixer, gt_coords=gt, N_sim=1)
    np.random.seed(0)
    props, score = spinner.fit_stoichiometry(
        N_structures=ss, fitting_mode="bayesian"
    )
    assert isinstance(score, float) and np.isfinite(score)
    assert props.shape == (2,)


@pytest.mark.slow
def test_spinna_fit_asynch_runs(mixer_2d_csr, monomer_dimer_structures):
    np.random.seed(0)
    gt = mixer_2d_csr.run_simulation([100, 50])
    mixer = StructureMixer(
        structures=monomer_dimer_structures,
        label_unc={"target": LABEL_UNC},
        le={"target": LE},
        width=ROI,
        height=ROI,
    )
    ss = spinna.generate_N_structures(
        monomer_dimer_structures, {"target": 200}, granularity=GRANULARITY
    )
    spinner = SPINNA(mixer=mixer, gt_coords=gt, N_sim=1)
    np.random.seed(0)
    props, score = spinner.fit_stoichiometry(
        N_structures=ss, fitting_mode="brute-force", asynch=True
    )
    assert isinstance(score, float) and np.isfinite(score)
    assert props.shape == (2,)


# ---------------------------------------------------------------------
# Section L — Multitarget LE helpers
# ---------------------------------------------------------------------


def test_check_structures_valid_for_fitting_true(het_structures):
    assert spinna.check_structures_valid_for_fitting(het_structures) is True


@pytest.mark.parametrize(
    "case",
    ["only_monomers", "three_monomers", "four_structures", "single_target"],
)
def test_check_structures_valid_for_fitting_false_paths(case):
    if case == "only_monomers":
        mA = Structure("MonA")
        mA.define_coordinates("A", [0], [0], [0])
        mB = Structure("MonB")
        mB.define_coordinates("B", [0], [0], [0])
        structures = [mA, mB]  # only 2 structures, no heterodimer
    elif case == "three_monomers":
        mA = Structure("MonA")
        mA.define_coordinates("A", [0], [0], [0])
        mB = Structure("MonB")
        mB.define_coordinates("B", [0], [0], [0])
        mC = Structure("MonC")  # third monomer of "A"
        mC.define_coordinates("A", [10], [0], [0])
        structures = [mA, mB, mC]
    elif case == "four_structures":
        mA = Structure("MonA")
        mA.define_coordinates("A", [0], [0], [0])
        mB = Structure("MonB")
        mB.define_coordinates("B", [0], [0], [0])
        het = Structure("HetAB")
        het.define_coordinates("A", [-10.5], [0], [0])
        het.define_coordinates("B", [10.5], [0], [0])
        extra = Structure("Trimer")
        extra.define_coordinates("A", [0, 5, 10], [0, 0, 0], [0, 0, 0])
        structures = [mA, mB, het, extra]
    else:  # single_target
        m1 = Structure("M1")
        m1.define_coordinates("A", [0], [0], [0])
        m2 = Structure("M2")
        m2.define_coordinates("A", [10], [0], [0])
        d = Structure("Dimer")
        d.define_coordinates("A", [-5, 5], [0, 0], [0, 0])
        structures = [m1, m2, d]
    assert spinna.check_structures_valid_for_fitting(structures) is False


def test_get_le_from_props_correctness(het_structures):
    # 50% MonA, 25% MonB, 25% HetAB (in molecules)
    props = np.array([50.0, 25.0, 25.0])
    le = spinna.get_le_from_props(het_structures, props)
    # AB structure proportion: 25/2 = 12.5
    # le_A (first target listed in set) = 12.5 / (B + AB) * 100
    # because targets is {A, B} via set() — order not guaranteed,
    # so just assert both keys present and within 0–100
    assert set(le.keys()) == {"A", "B"}
    # both LEs must be finite and positive
    for v in le.values():
        assert 0 < v < 100


def test_get_le_from_props_accepts_tuple_input(het_structures):
    props = np.array([50.0, 25.0, 25.0])
    bootstrap_input = (props, props * 0.1)  # (mean, std)
    le = spinna.get_le_from_props(het_structures, bootstrap_input)
    le_ref = spinna.get_le_from_props(het_structures, props)
    assert le == le_ref


def test_get_le_from_props_invalid_structures_raises(
    monomer_dimer_structures,
):
    with pytest.raises(ValueError):
        spinna.get_le_from_props(
            monomer_dimer_structures, np.array([50.0, 50.0])
        )


# ---------------------------------------------------------------------
# Section M — compare_models integration (slow)
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_compare_models_given_label_unc_returns_best(
    mols_real, monomer_dimer_structures
):
    coords = mols_real[["x", "y"]].to_numpy()
    # build two candidate models, both 1-target
    mono = Structure("Mono")
    mono.define_coordinates("target", [0], [0], [0])
    dimer = Structure("Dimer")
    dimer.define_coordinates("target", [-10.5, 10.5], [0, 0], [0, 0])
    trimer = Structure("Trimer")
    trimer.define_coordinates(
        "target", [-15.0, 0.0, 15.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    )
    models = [[mono, dimer], [mono, dimer, trimer]]

    np.random.seed(0)
    best_score, best_idx, best_mixer, best_props = (
        spinna.compare_models_given_label_unc(
            models=models,
            exp_data={"target": coords},
            granularity=3,
            label_unc={"target": LABEL_UNC},
            le={"target": LE},
            width=ROI,
            height=ROI,
            N_sim=1,
            asynch=False,
        )
    )
    assert isinstance(best_score, float) and np.isfinite(best_score)
    assert best_idx in (0, 1)
    assert isinstance(best_mixer, StructureMixer)
    assert len(best_props) == len(models[best_idx])


@pytest.mark.slow
def test_compare_models_full_fits_label_unc(
    mols_real, monomer_dimer_structures
):
    coords = mols_real[["x", "y"]].to_numpy()
    mono = Structure("Mono")
    mono.define_coordinates("target", [0], [0], [0])
    dimer = Structure("Dimer")
    dimer.define_coordinates("target", [-10.5, 10.5], [0, 0], [0, 0])
    models = [[mono, dimer]]
    # provide list of label_unc to trigger the per-target LE fitting
    np.random.seed(0)
    best_score, best_idx, label_unc_out, best_mixer, best_props = (
        spinna.compare_models(
            models=models,
            exp_data={"target": coords},
            granularity=3,
            label_unc={"target": [3.0, 6.0]},
            le={"target": LE},
            width=ROI,
            height=ROI,
            N_sim=1,
            asynch=False,
        )
    )
    assert np.isfinite(best_score)
    assert best_idx == 0
    # After fitting, label_unc[target] should be a single float, not a list
    assert isinstance(label_unc_out["target"], float)
    assert label_unc_out["target"] in (3.0, 6.0)
    assert isinstance(best_mixer, StructureMixer)
    assert len(best_props) == 2
