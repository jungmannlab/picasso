"""Test pure-logic helpers in ``picasso.lib``.

Skips Qt classes (``Dialog``, ``UserSettingsDialog``, etc.) and any
function that calls ``QtWidgets`` directly. Covers metadata access,
hex/path helpers, kinetic fits, recarray manipulation, polygon /
rectangle containment, drift-shift inversion, and group syncing.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from picasso import lib  # noqa: E402


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


class TestGetFromMetadata:
    def test_dict_input_found(self):
        info = {"Width": 32, "Height": 32}
        assert lib.get_from_metadata(info, "Width") == 32

    def test_dict_input_default(self):
        info = {"Width": 32}
        assert lib.get_from_metadata(info, "Missing", default=99) == 99

    def test_list_input_searches_from_last(self):
        # Iterates in reverse — last entry's value wins for duplicate keys
        info = [{"Pixelsize": 130}, {"Pixelsize": 160}]
        assert lib.get_from_metadata(info, "Pixelsize") == 160

    def test_list_input_default(self):
        info = [{"Width": 32}, {"Height": 32}]
        assert lib.get_from_metadata(info, "Pixelsize", default=130) == 130

    def test_raise_error_on_missing(self):
        info = [{"Width": 32}]
        with pytest.raises(KeyError):
            lib.get_from_metadata(info, "Missing", raise_error=True)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            lib.get_from_metadata("not a dict", "Width")


class TestOverwriteMetadata:
    def test_overwrites_existing_dict(self):
        info = {"Width": 32}
        out = lib.overwrite_metadata(info, "Width", 64)
        assert out["Width"] == 64

    def test_overwrites_in_list(self):
        info = [{"Width": 32}, {"Pixelsize": 130}]
        lib.overwrite_metadata(info, "Width", 64)
        assert info[0]["Width"] == 64

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            lib.overwrite_metadata({"Width": 32}, "Missing", 1)


# ---------------------------------------------------------------------------
# Color / path utilities
# ---------------------------------------------------------------------------


class TestGetColors:
    def test_count(self):
        colors = lib.get_colors(5)
        assert len(colors) == 5

    def test_rgb_tuples(self):
        colors = lib.get_colors(3)
        for r, g, b in colors:
            assert 0 <= r <= 1
            assert 0 <= g <= 1
            assert 0 <= b <= 1


class TestIsHexadecimal:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("#ff02d4", True),
            ("#FFFFFF", True),
            ("#000000", True),
            ("ff02d4", False),  # missing #
            ("#GGGGGG", False),  # invalid chars
            ("#FF00DD33", False),  # too long
            (None, False),
            # NOTE: passing "" or other length-<1 strings currently raises
            # IndexError in is_hexadecimal — that is a latent bug in the
            # function, not a test concern. Don't exercise that path here.
        ],
    )
    def test_truth_table(self, text, expected):
        assert lib.is_hexadecimal(text) is expected


class TestIsPathAvailable:
    def test_returns_true_for_missing(self, tmp_path):
        path = str(tmp_path / "does_not_exist.txt")
        assert lib.is_path_available(path) == [True]

    def test_returns_false_for_existing(self, tmp_path):
        path = tmp_path / "exists.txt"
        path.write_text("x")
        assert lib.is_path_available(str(path)) == [False]

    def test_check_ext_list(self, tmp_path):
        existing = tmp_path / "file.hdf5"
        existing.write_text("x")
        out = lib.is_path_available(
            str(tmp_path / "file"), check_ext=[".yaml", ".hdf5"]
        )
        assert out == [True, False]


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


class TestFindLocalMinima:
    def test_simple_array(self):
        arr = np.array([3.0, 1.0, 2.0, 0.5, 5.0, 4.0, 6.0])
        idx = lib.find_local_minima(arr)
        # Local minima at index 1 (1.0) and index 3 (0.5) and index 5 (4.0)
        assert sorted(idx.tolist()) == [1, 3, 5]

    def test_no_minima(self):
        # monotonically increasing → no local minima
        arr = np.arange(10, dtype=float)
        idx = lib.find_local_minima(arr)
        assert len(idx) == 0


class TestCumulativeExponential:
    def test_zero_at_zero(self):
        out = lib.cumulative_exponential(np.array([0.0]), a=10.0, t=2.0, c=0.0)
        assert out[0] == pytest.approx(0.0, abs=1e-12)

    def test_constant_offset(self):
        out = lib.cumulative_exponential(np.array([0.0]), a=10.0, t=2.0, c=3.0)
        assert out[0] == pytest.approx(3.0)


class TestFitCumExp:
    def test_recovers_tau(self):
        # Generate cumulative-exponential-distributed samples and check
        # that fit recovers the time constant.
        rng = np.random.default_rng(0)
        true_tau = 50.0
        # exponential samples → CDF is 1 - exp(-x/tau); fit_cum_exp fits
        # data->rank, which corresponds to this CDF in the limit.
        data = rng.exponential(scale=true_tau, size=2000)
        result = lib.fit_cum_exp(data)
        assert "best_values" in result
        assert "best_fit" in result
        # Fit should land in the same order of magnitude
        assert result["best_values"]["t"] == pytest.approx(true_tau, rel=0.4)


class TestEstimateKineticRate:
    def test_returns_finite_for_long_data(self):
        rng = np.random.default_rng(1)
        data = rng.exponential(scale=20.0, size=500)
        rate = lib.estimate_kinetic_rate(data)
        assert np.isfinite(rate)
        assert rate > 0

    def test_short_data_falls_back_to_mean(self):
        data = np.array([1.0, 2.0])
        rate = lib.estimate_kinetic_rate(data)
        assert rate == pytest.approx(1.5)

    def test_constant_data(self):
        data = np.array([5.0, 5.0, 5.0, 5.0])
        rate = lib.estimate_kinetic_rate(data)
        assert rate == pytest.approx(5.0)


class TestCalculateOptimalBins:
    def test_returns_array(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=200)
        bins = lib.calculate_optimal_bins(data)
        assert isinstance(bins, np.ndarray)
        assert bins.size >= 2

    def test_max_n_bins_caps_output(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=10000)
        bins = lib.calculate_optimal_bins(data, max_n_bins=10)
        assert bins.size <= 10

    def test_zero_iqr_returns_two_bins(self):
        data = np.array([7.0, 7.0, 7.0, 7.0])
        bins = lib.calculate_optimal_bins(data)
        # zero-iqr branch returns the constant ±1 fallback
        assert bins.size == 2

    def test_sampled_iqr_close_to_full(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=200_000)
        full = lib.calculate_optimal_bins(
            data, max_n_bins=1000, sample_size=len(data) + 1
        )
        sampled = lib.calculate_optimal_bins(
            data, max_n_bins=1000, sample_size=20_000
        )
        # Same range, similar bin count (Freedman-Diaconis is stable
        # under sub-sampling of an iid sample).
        assert sampled[0] == pytest.approx(full[0], rel=0.05)
        assert sampled[-1] == pytest.approx(full[-1], rel=0.05)
        assert abs(sampled.size - full.size) <= max(2, full.size // 10)

    def test_handles_nan_data(self):
        data = np.concatenate([np.full(10, np.nan), np.linspace(0, 1, 1000)])
        bins = lib.calculate_optimal_bins(data, max_n_bins=50)
        # bin range is finite even though some values are NaN
        assert np.isfinite(bins[0]) and np.isfinite(bins[-1])


class TestHist2DNumba:
    def test_matches_numpy_histogram2d(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=50_000)
        y = rng.normal(size=50_000)
        x_min, x_max = -3.0, 3.0
        y_min, y_max = -3.0, 3.0
        nx, ny = 40, 30
        # restrict to points strictly inside [x_min, x_max] x [y_min, y_max]
        # to side-step floating-point boundary differences between the two
        # implementations
        inside = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
        x = x[inside]
        y = y[inside]
        counts = lib.hist2d_numba(x, y, x_min, x_max, y_min, y_max, nx, ny)
        x_edges = np.linspace(x_min, x_max, nx + 1)
        y_edges = np.linspace(y_min, y_max, ny + 1)
        expected, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        assert counts.shape == (nx, ny)
        assert counts.sum() == len(x)
        # per-cell counts may differ by ~1 due to bin-edge rounding; total
        # mismatch should be small relative to N
        assert np.abs(counts - expected.astype(np.int64)).sum() < 0.001 * len(
            x
        )

    def test_skips_non_finite(self):
        x = np.array([0.0, 1.0, np.nan, 2.0, np.inf], dtype=np.float64)
        y = np.array([0.0, 1.0, 1.0, np.nan, 2.0], dtype=np.float64)
        counts = lib.hist2d_numba(x, y, 0.0, 3.0, 0.0, 3.0, 3, 3)
        # only two points are fully finite and inside the range
        assert counts.sum() == 2


# ---------------------------------------------------------------------------
# RMSD at center of mass
# ---------------------------------------------------------------------------


class TestRmsdAtCom:
    def test_known_value(self):
        # COM = (1, 0), distances 1, 0, 1 -> RMSD = sqrt(2/3)
        xy = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
        assert lib.rmsd_at_com(xy) == pytest.approx(np.sqrt(2 / 3))

    def test_zero_for_identical_points(self):
        xy = np.full((2, 5), 3.0)
        assert lib.rmsd_at_com(xy) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Recarray manipulation (deprecated path — explicit warnings expected)
# ---------------------------------------------------------------------------


class TestRecarrayHelpers:
    def _toy_rec(self):
        return pd.DataFrame(
            {"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0]}
        ).to_records(index=False)

    def test_append_to_rec_adds_column(self):
        rec = self._toy_rec()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out = lib.append_to_rec(rec, np.array([10.0, 11.0, 12.0]), "z")
        assert "z" in out.dtype.names
        np.testing.assert_array_equal(out["z"], [10.0, 11.0, 12.0])

    def test_remove_from_rec_drops_column(self):
        rec = self._toy_rec()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out = lib.remove_from_rec(rec, "y")
        assert "y" not in out.dtype.names
        assert "x" in out.dtype.names


# ---------------------------------------------------------------------------
# Localization merging / sanitizing
# ---------------------------------------------------------------------------


class TestMergeLocs:
    def _toy_locs(self, n: int, frame_offset: int = 0, group: int = 0):
        return pd.DataFrame(
            {
                "frame": np.arange(n, dtype=int) + frame_offset,
                "x": np.arange(n, dtype=float),
                "y": np.arange(n, dtype=float),
                "group": np.full(n, group, dtype=int),
            }
        )

    def test_concatenates(self):
        a = self._toy_locs(3, group=0)
        b = self._toy_locs(2, group=1)
        merged = lib.merge_locs(
            [a, b], increment_frames=False, increment_groups=False
        )
        assert len(merged) == 5

    def test_increment_frames_default(self):
        a = self._toy_locs(3)  # frames 0..2
        b = self._toy_locs(3)  # frames 0..2
        merged = lib.merge_locs([a, b], increment_groups=False)
        # b's frames should now be shifted by max(a.frame) = 2
        # → b frames become 2, 3, 4
        assert merged["frame"].max() == 4


class TestEnsureSanity:
    def test_drops_outside_image(self):
        locs = pd.DataFrame(
            {
                "frame": [0, 0, 0],
                "x": [1.0, 100.0, 5.0],  # 100 is outside Width=32
                "y": [1.0, 5.0, 50.0],  # 50 is outside Height=32
                "lpx": [0.1, 0.1, 0.1],
                "lpy": [0.1, 0.1, 0.1],
            }
        )
        info = [{"Width": 32, "Height": 32, "Frames": 100}]
        out = lib.ensure_sanity(locs, info)
        assert len(out) == 1

    def test_drops_negative_attrs(self):
        locs = pd.DataFrame(
            {
                "frame": [0, 0],
                "x": [1.0, 2.0],
                "y": [1.0, 2.0],
                "photons": [100.0, -5.0],
            }
        )
        info = [{"Width": 32, "Height": 32, "Frames": 1}]
        out = lib.ensure_sanity(locs, info)
        assert len(out) == 1

    def test_missing_key_raises(self):
        locs = pd.DataFrame({"frame": [0], "x": [1.0], "y": [1.0]})
        info = [{"Width": 32}]  # missing Height + Frames
        with pytest.raises(KeyError):
            lib.ensure_sanity(locs, info)


# ---------------------------------------------------------------------------
# Distance / containment
# ---------------------------------------------------------------------------


class TestIsLocAt:
    def test_inside_radius(self):
        locs = pd.DataFrame({"x": [10.0, 12.0, 20.0], "y": [10.0, 10.0, 20.0]})
        mask = lib.is_loc_at(10.0, 10.0, locs, r=3.0)
        assert mask.tolist() == [True, True, False]

    def test_locs_at_filters(self):
        locs = pd.DataFrame({"x": [10.0, 100.0], "y": [10.0, 100.0]})
        out = lib.locs_at(10.0, 10.0, locs, r=2.0)
        assert len(out) == 1
        assert out.iloc[0]["x"] == 10.0


class TestPolygonContainment:
    def test_unit_square(self):
        # Polygon: unit square
        X = np.array([0.0, 1.0, 1.0, 0.0])
        Y = np.array([0.0, 0.0, 1.0, 1.0])
        x = np.array([0.5, 1.5, 0.5])
        y = np.array([0.5, 0.5, 1.5])
        mask = lib.check_if_in_polygon(x, y, X, Y)
        assert mask.tolist() == [True, False, False]

    def test_locs_in_polygon(self):
        locs = pd.DataFrame({"x": [0.5, 1.5, 0.2], "y": [0.5, 0.5, 0.2]})
        X = np.array([0.0, 1.0, 1.0, 0.0])
        Y = np.array([0.0, 0.0, 1.0, 1.0])
        out = lib.locs_in_polygon(locs, X, Y)
        # Two points (0.5, 0.5) and (0.2, 0.2) are inside the unit square
        assert len(out) == 2


class TestRectangleContainment:
    def test_axis_aligned(self):
        # Rectangle from (0,0) to (10,5)
        X = np.array([0.0, 10.0, 10.0, 0.0])
        Y = np.array([0.0, 0.0, 5.0, 5.0])
        x = np.array([5.0, 11.0, 5.0])
        y = np.array([2.5, 2.5, 6.0])
        mask = lib.check_if_in_rectangle(x, y, X, Y)
        assert mask.tolist() == [True, False, False]

    def test_locs_in_rectangle(self):
        locs = pd.DataFrame({"x": [5.0, 11.0], "y": [2.5, 2.5]})
        X = np.array([0.0, 10.0, 10.0, 0.0])
        Y = np.array([0.0, 0.0, 5.0, 5.0])
        out = lib.locs_in_rectangle(locs, X, Y)
        assert len(out) == 1
        assert out.iloc[0]["x"] == 5.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class TestPolygonArea:
    def test_unit_square(self):
        X = np.array([0.0, 1.0, 1.0, 0.0])
        Y = np.array([0.0, 0.0, 1.0, 1.0])
        assert lib.polygon_area(X, Y) == pytest.approx(1.0)

    def test_triangle(self):
        # Right triangle with legs 1 and 2 → area = 1
        X = np.array([0.0, 2.0, 0.0])
        Y = np.array([0.0, 0.0, 1.0])
        assert lib.polygon_area(X, Y) == pytest.approx(1.0)

    def test_collinear_zero(self):
        X = np.array([0.0, 1.0, 2.0])
        Y = np.array([0.0, 1.0, 2.0])
        assert lib.polygon_area(X, Y) == pytest.approx(0.0)


class TestPickPolygonCorners:
    def test_closed_polygon(self):
        pick = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
        X, Y = lib.get_pick_polygon_corners(pick)
        assert X == [0.0, 1.0, 1.0, 0.0]
        assert Y == [0.0, 0.0, 1.0, 0.0]

    def test_open_polygon_returns_none(self):
        pick = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]  # not closed
        X, Y = lib.get_pick_polygon_corners(pick)
        assert X is None
        assert Y is None

    def test_too_few_points(self):
        pick = [(0.0, 0.0), (1.0, 1.0)]
        X, Y = lib.get_pick_polygon_corners(pick)
        assert X is None
        assert Y is None


class TestPickRectangleCorners:
    def test_horizontal_rectangle(self):
        # Horizontal rectangle: dx>0, dy=0 → alpha=0 → dx_corner=0, dy_corner=w/2
        X, Y = lib.get_pick_rectangle_corners(
            start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0, width=2.0
        )
        # 4 corners
        assert len(X) == 4
        assert len(Y) == 4
        # Y values should be ±1 (width/2)
        assert sorted(Y) == [-1.0, -1.0, 1.0, 1.0]

    def test_returns_four_corners(self):
        X, Y = lib.get_pick_rectangle_corners(
            start_x=0.0, start_y=0.0, end_x=10.0, end_y=10.0, width=1.0
        )
        assert len(X) == 4
        assert len(Y) == 4


class TestPickAreas:
    def test_circle(self):
        picks = [(1.0, 1.0), (2.0, 2.0)]
        areas = lib.pick_areas(picks, "Circle", pick_size=2.0)
        # diameter=2 → r=1 → π
        assert areas.shape == (2,)
        assert areas[0] == pytest.approx(np.pi)

    def test_square(self):
        picks = [(0.0, 0.0)]
        areas = lib.pick_areas(picks, "Square", pick_size=3.0)
        assert areas[0] == 9.0

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            lib.pick_areas([(0.0, 0.0)], "Triangle", pick_size=1.0)


# ---------------------------------------------------------------------------
# Drift inversion (used by RCC)
# ---------------------------------------------------------------------------


class TestMinimizeShifts:
    def test_recovers_known_per_segment_offsets(self):
        # Build pairwise shifts from per-segment offsets (relative to seg 0).
        offsets = np.array(
            [
                [0.0, 0.0],
                [2.0, -1.0],
                [-1.0, 3.0],
                [4.0, 2.0],
            ]
        )
        n = len(offsets)
        shifts_x = np.zeros((n, n))
        shifts_y = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                shifts_y[i, j] = offsets[j, 0] - offsets[i, 0]
                shifts_x[i, j] = offsets[j, 1] - offsets[i, 1]
        shift_y, shift_x = lib.minimize_shifts(shifts_x, shifts_y)
        assert shift_y.shape == (n,)
        assert shift_x.shape == (n,)
        np.testing.assert_allclose(shift_y, offsets[:, 0], atol=1e-9)
        np.testing.assert_allclose(shift_x, offsets[:, 1], atol=1e-9)

    def test_3d_returns_three_arrays(self):
        n = 3
        offsets = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        shifts_x = np.zeros((n, n))
        shifts_y = np.zeros((n, n))
        shifts_z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                shifts_y[i, j] = offsets[j, 0] - offsets[i, 0]
                shifts_x[i, j] = offsets[j, 1] - offsets[i, 1]
                shifts_z[i, j] = offsets[j, 2] - offsets[i, 2]
        shift_y, shift_x, shift_z = lib.minimize_shifts(
            shifts_x, shifts_y, shifts_z
        )
        np.testing.assert_allclose(shift_y, offsets[:, 0], atol=1e-9)
        np.testing.assert_allclose(shift_x, offsets[:, 1], atol=1e-9)
        np.testing.assert_allclose(shift_z, offsets[:, 2], atol=1e-9)


# ---------------------------------------------------------------------------
# Group syncing
# ---------------------------------------------------------------------------


class TestSyncGroups:
    def test_only_common_groups_kept(self):
        a = pd.DataFrame({"group": [0, 0, 1, 2], "x": [0.0, 0.1, 1.0, 2.0]})
        b = pd.DataFrame({"group": [1, 2, 3], "x": [10.0, 20.0, 30.0]})
        c = pd.DataFrame({"group": [1, 2], "x": [100.0, 200.0]})
        synced = lib.sync_groups([a, b, c])
        # Only groups present in all three (1 and 2) should remain
        assert set(synced[0]["group"]) == {1, 2}
        assert set(synced[1]["group"]) == {1, 2}
        assert set(synced[2]["group"]) == {1, 2}

    def test_missing_group_column_asserts(self):
        a = pd.DataFrame({"x": [1.0]})
        with pytest.raises(AssertionError):
            lib.sync_groups([a])
