"""Picks of different shapes and sizes in one set.

Every function that takes picks accepts one shape and size for all
picks or one per pick (``lib.pick_shapes_and_sizes``). These tests check
that a mixed set gives each pick the same result as picking it on its
own, and that mixed sets survive the picks .yaml files.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import io, lib, postprocess

FOV = 5000.0  # nm


def _locs(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, n).astype(np.int32),
            "x": rng.uniform(0, FOV, n).astype(np.float32),
            "y": rng.uniform(0, FOV, n).astype(np.float32),
            "lpx": np.full(n, 5.0, dtype=np.float32),
            "lpy": np.full(n, 5.0, dtype=np.float32),
            "photons": np.full(n, 1000.0, dtype=np.float32),
        }
    )


INFO = [{"Width": FOV, "Height": FOV, "Frames": 1000, "Pixelsize": 100.0}]

PICKS = [
    (1000.0, 1000.0),
    (2000.0, 2000.0),
    ((3000.0, 3000.0), (3500.0, 3200.0)),
    (4000.0, 1000.0),
    [(100.0, 4000.0), (600.0, 4000.0), (600.0, 4500.0), (100.0, 4000.0)],
    [(80.0, [(2500.0, 500.0), (2800.0, 600.0)])],
    ((4000.0, 4000.0), (4500.0, 4300.0)),
]
SHAPES = ["Circle", "Circle", "Rectangle", "Square", "Polygon", "Brush", "Box"]
SIZES = [100.0, 300.0, 80.0, 200.0, None, None, None]


@pytest.fixture(scope="module")
def locs():
    return _locs()


class TestPickShapesAndSizes:
    def test_one_shape_and_size_for_all_picks(self):
        shapes, sizes = lib.pick_shapes_and_sizes(PICKS[:2], "Circle", 100)
        assert shapes == ["Circle", "Circle"]
        assert sizes == [100.0, 100.0]

    def test_one_shape_and_size_per_pick(self):
        shapes, sizes = lib.pick_shapes_and_sizes(PICKS, SHAPES, SIZES)
        assert shapes == SHAPES
        assert sizes == SIZES

    def test_sizes_of_shapes_without_size_are_none(self):
        _, sizes = lib.pick_shapes_and_sizes(
            PICKS[4:], SHAPES[4:], [1.0, 2.0, 3.0]
        )
        assert sizes == [None, None, None]

    def test_no_picks(self):
        assert lib.pick_shapes_and_sizes([], "Circle") == ([], [])
        assert lib.pick_shapes_and_sizes([], None) == ([], [])

    @pytest.mark.parametrize(
        "shape, size, message",
        [
            ("Hexagon", 1.0, "Unrecognized pick shape"),
            (["Circle", "Hexagon"], 1.0, "Unrecognized pick shape"),
            (["Circle"], 1.0, "1 shapes for 2 picks"),
            ("Circle", [1.0], "1 sizes for 2 picks"),
            ("Circle", None, "need a size"),
            ("Circle", [100.0, 0.0], "need a size"),
            ("Square", np.nan, "need a size"),
        ],
    )
    def test_invalid_arguments_raise(self, shape, size, message):
        with pytest.raises(ValueError, match=message):
            lib.pick_shapes_and_sizes(PICKS[:2], shape, size)

    def test_compact_is_the_inverse(self):
        shapes, sizes = lib.pick_shapes_and_sizes(PICKS, SHAPES, SIZES)
        assert lib._compact_pick_shapes_and_sizes(shapes, sizes) == (
            SHAPES,
            SIZES,
        )
        assert lib._compact_pick_shapes_and_sizes(
            ["Circle", "Circle"], [100.0, 100.0]
        ) == ("Circle", 100.0)
        assert lib._compact_pick_shapes_and_sizes([], []) == (None, None)


class TestMixedPickAreas:
    def test_each_pick_has_its_own_area(self):
        areas = lib.pick_areas(PICKS, SHAPES, SIZES)
        expected = [
            lib.pick_areas([pick], shape, size)[0]
            for pick, shape, size in zip(PICKS, SHAPES, SIZES)
        ]
        np.testing.assert_allclose(areas, expected)
        assert areas[0] == pytest.approx(np.pi * 50.0**2)
        assert areas[1] == pytest.approx(np.pi * 150.0**2)

    def test_open_polygons_are_left_out(self):
        open_polygon = [(0.0, 0.0), (10.0, 0.0)]
        areas = lib.pick_areas(
            [(0.0, 0.0), open_polygon, (5.0, 5.0)],
            ["Circle", "Polygon", "Square"],
            [2.0, None, 3.0],
        )
        np.testing.assert_allclose(areas, [np.pi, 9.0])

    def test_translate_per_shape(self):
        moved = lib.translate_picks(PICKS[:3], SHAPES[:3], 1.0, 2.0)
        assert moved[0] == (1001.0, 1002.0)
        assert moved[2] == ((3001.0, 3002.0), (3501.0, 3202.0))


class TestMixedPickedLocs:
    def test_each_pick_matches_picking_it_alone(self, locs):
        picked = postprocess.picked_locs(
            locs, INFO, picks=PICKS, pick_shape=SHAPES, pick_size=SIZES
        )
        assert len(picked) == len(PICKS)
        for i, (pick, shape, size) in enumerate(zip(PICKS, SHAPES, SIZES)):
            alone = postprocess.picked_locs(
                locs, INFO, picks=[pick], pick_shape=shape, pick_size=size
            )[0]
            assert len(alone) > 0
            assert sorted(picked[i].index) == sorted(alone.index)
            # the group is the index of the pick in the whole set
            assert (picked[i]["group"] == i).all()

    def test_rotated_coordinates_only_for_rectangles_alone(self, locs):
        # with other shapes, the columns would be NaN for their picks
        picked = postprocess.picked_locs(
            locs, INFO, picks=PICKS, pick_shape=SHAPES, pick_size=SIZES
        )
        assert not any("x_pick_rot" in p.columns for p in picked)
        rectangles = postprocess.picked_locs(
            locs,
            INFO,
            picks=[PICKS[2]],
            pick_shape=["Rectangle"],
            pick_size=80,
        )
        assert {"x_pick_rot", "y_pick_rot"} <= set(rectangles[0].columns)

    def test_saving_keeps_every_pick(self, locs, tmp_path):
        picked = postprocess.picked_locs(
            locs, INFO, picks=PICKS, pick_shape=SHAPES, pick_size=SIZES
        )
        path = str(tmp_path / "picked.hdf5")
        io.save_locs(path, pd.concat(picked, ignore_index=True), INFO)
        saved, _ = io.load_locs(path)
        counts = saved.groupby("group").size()
        assert counts.to_dict() == {i: len(p) for i, p in enumerate(picked)}

    def test_open_polygons_are_skipped(self, locs):
        picks = [PICKS[0], [(0.0, 0.0), (10.0, 0.0)], PICKS[1]]
        picked = postprocess.picked_locs(
            locs,
            INFO,
            picks=picks,
            pick_shape=["Circle", "Polygon", "Circle"],
            pick_size=[100.0, None, 300.0],
        )
        assert len(picked) == 2
        assert (picked[1]["group"] == 2).all()

    def test_index_blocks_smaller_than_a_pick_are_rebuilt(self, locs):
        # the blocks are for a radius of 10 nm, the second pick has 150
        small = postprocess.get_index_blocks(locs, INFO, 10.0)
        kwargs = dict(
            picks=PICKS[:2], pick_shape="Circle", pick_size=[100.0, 300.0]
        )
        with_blocks = postprocess.picked_locs(
            locs, INFO, index_blocks=small, **kwargs
        )
        without = postprocess.picked_locs(locs, INFO, **kwargs)
        assert [len(_) for _ in with_blocks] == [len(_) for _ in without]

    def test_larger_index_blocks_are_used(self, locs):
        # blocks larger than the radius hold every localization in the
        # 3x3 neighborhood of a pick, so they give the same result
        large = postprocess.get_index_blocks(locs, INFO, 400.0)
        kwargs = dict(picks=PICKS[:2], pick_shape="Circle", pick_size=200.0)
        with_blocks = postprocess.picked_locs(
            locs, INFO, index_blocks=large, **kwargs
        )
        without = postprocess.picked_locs(locs, INFO, **kwargs)
        assert [len(_) for _ in with_blocks] == [len(_) for _ in without]

    def test_remove_locs_in_mixed_picks(self, locs):
        picked = postprocess.picked_locs(
            locs, INFO, picks=PICKS, pick_shape=SHAPES, pick_size=SIZES
        )
        kept, info = postprocess.remove_locs_in_picks(
            locs, INFO, picks=PICKS, pick_shape=SHAPES, pick_size=SIZES
        )
        assert len(locs) - len(kept) == sum(len(_) for _ in picked)
        assert info[-1]["Pick shape"] == SHAPES
        assert info[-1]["Pick size (nm)"] == SIZES

    def test_metadata_of_circles_of_different_sizes(self, locs):
        _, info = postprocess.remove_locs_in_picks(
            locs,
            INFO,
            picks=PICKS[:2],
            pick_shape="Circle",
            pick_size=[100.0, 300.0],
        )
        assert info[-1]["Pick shape"] == "Circle"
        assert info[-1]["Pick diameter (nm)"] == [100.0, 300.0]

    def test_ripley_h_in_mixed_picks(self, locs):
        result = postprocess.ripley_h(
            locs,
            INFO,
            picks=PICKS,
            pick_shape=SHAPES,
            pick_size=SIZES,
            r_max=50.0,
            steps=5,
            n_simulations=2,
        )
        assert np.all(np.isfinite(result["h"]))


class TestPickSimilarSizes:
    def test_circles_of_different_sizes_use_the_median(self, locs):
        picks = [PICKS[0], PICKS[1], (4000.0, 4000.0)]
        kwargs = dict(picks=picks, pick_shape="Circle", std_range=5.0)
        mixed = postprocess.pick_similar(
            locs, INFO, pick_size=[100.0, 300.0, 200.0], **kwargs
        )
        median = postprocess.pick_similar(
            locs, INFO, pick_size=200.0, **kwargs
        )
        assert len(mixed) == len(median)
        np.testing.assert_allclose(mixed, median)

    @pytest.mark.parametrize(
        "shape, picks, sizes",
        [
            (
                "Square",
                [(1000.0, 1000.0), (2500.0, 2500.0), (4000.0, 1200.0)],
                [100.0, 300.0, 200.0],
            ),
            (
                "Rectangle",
                [
                    ((900.0, 1000.0), (1400.0, 1000.0)),
                    ((2500.0, 2300.0), (2500.0, 2800.0)),
                    ((3500.0, 3500.0), (3900.0, 3800.0)),
                ],
                [60.0, 120.0, 80.0],
            ),
        ],
    )
    def test_squares_and_rectangles_use_the_median(
        self, locs, shape, picks, sizes
    ):
        kwargs = dict(picks=picks, pick_shape=shape, std_range=5.0)
        mixed = postprocess.pick_similar(locs, INFO, pick_size=sizes, **kwargs)
        median = postprocess.pick_similar(
            locs, INFO, pick_size=float(np.median(sizes)), **kwargs
        )
        assert len(mixed) == len(median) > len(picks)
        np.testing.assert_allclose(
            np.asarray(mixed, dtype=float), np.asarray(median, dtype=float)
        )

    def test_different_shapes_raise(self, locs):
        with pytest.raises(ValueError, match="one shape"):
            postprocess.pick_similar(
                locs,
                INFO,
                picks=PICKS[:4],
                pick_shape=SHAPES[:4],
                pick_size=SIZES[:4],
            )


class TestMixedPicksFiles:
    def test_mixed_shapes_round_trip(self, tmp_path):
        path = str(tmp_path / "picks.yaml")
        io.save_picks(path, PICKS, SHAPES, SIZES)
        regions = io.load_yaml(path)
        assert regions["Shape"] == "Mixed"
        assert [_["Shape"] for _ in regions["Picks"]] == SHAPES
        assert regions["Picks"][0]["Diameter (nm)"] == 100.0
        picks, shape, size = io.load_picks(path)
        assert shape == SHAPES
        assert size == SIZES
        assert np.allclose(picks[0], PICKS[0])
        assert np.allclose(np.array(picks[2]), np.array(PICKS[2]))
        assert np.allclose(np.array(picks[4]), np.array(PICKS[4]))
        assert picks[5][0][0] == 80.0
        assert np.allclose(picks[5][0][1], PICKS[5][0][1])

    def test_mixed_file_picks_the_same_localizations(self, tmp_path, locs):
        path = str(tmp_path / "picks.yaml")
        io.save_picks(path, PICKS, SHAPES, SIZES)
        picks, shape, size = io.load_picks(path)
        loaded = postprocess.picked_locs(
            locs, INFO, picks=picks, pick_shape=shape, pick_size=size
        )
        original = postprocess.picked_locs(
            locs, INFO, picks=PICKS, pick_shape=SHAPES, pick_size=SIZES
        )
        assert [len(_) for _ in loaded] == [len(_) for _ in original]

    def test_one_shape_different_sizes(self, tmp_path):
        path = str(tmp_path / "picks.yaml")
        io.save_picks(path, PICKS[:2], "Circle", [100.0, 300.0])
        regions = io.load_yaml(path)
        assert regions["Shape"] == "Circle"
        assert regions["Diameter (nm)"] == [100.0, 300.0]
        _, shape, size = io.load_picks(path)
        assert (shape, size) == ("Circle", [100.0, 300.0])

    def test_shared_shape_and_size_keep_the_plain_format(self, tmp_path):
        path = str(tmp_path / "picks.yaml")
        io.save_picks(path, PICKS[:2], ["Circle", "Circle"], [50.0, 50.0])
        regions = io.load_yaml(path)
        assert regions["Shape"] == "Circle"
        assert regions["Diameter (nm)"] == 50.0

    def test_a_size_list_must_match_the_picks(self, tmp_path):
        path = tmp_path / "picks.yaml"
        io.save_yaml(
            str(path),
            {
                "Shape": "Square",
                "Centers": [[1.0, 1.0], [2.0, 2.0]],
                "Side Length (nm)": [10.0],
                io.FORMAT_VERSION_ATTR: io.FORMAT_VERSION,
            },
        )
        with pytest.raises(ValueError, match="1 sizes .* for 2 picks"):
            io.load_picks(str(path))

    @pytest.mark.parametrize(
        "regions",
        [
            {"Shape": "Circle", "Centers": [[1.0, 1.0]], "Diameter (nm)": 0.0},
            {
                "Shape": "Mixed",
                "Picks": [
                    {
                        "Shape": "Square",
                        "Center": [1.0, 1.0],
                        "Side Length (nm)": -5.0,
                    }
                ],
            },
        ],
    )
    def test_sizes_must_be_positive(self, tmp_path, regions):
        path = str(tmp_path / "picks.yaml")
        io.save_yaml(
            path, {**regions, io.FORMAT_VERSION_ATTR: io.FORMAT_VERSION}
        )
        with pytest.raises(ValueError, match="need a size"):
            io.load_picks(path)

    @pytest.mark.parametrize(
        "entry, message",
        [
            (
                {"Shape": "Hexagon", "Center": [1.0, 1.0]},
                "Unrecognized pick shape",
            ),
            (
                {"Shape": "Circle", "Center": [1.0, 1.0]},
                "missing the pick size",
            ),
        ],
    )
    def test_invalid_mixed_entries_raise(self, tmp_path, entry, message):
        path = str(tmp_path / "picks.yaml")
        io.save_yaml(
            path,
            {
                "Shape": "Mixed",
                "Picks": [entry],
                io.FORMAT_VERSION_ATTR: io.FORMAT_VERSION,
            },
        )
        with pytest.raises(ValueError, match=message):
            io.load_picks(path)

    def test_mixed_file_without_picks(self, tmp_path):
        path = str(tmp_path / "picks.yaml")
        io.save_picks(path, [], [], [])
        assert io.load_picks(path) == ([], [], None)
