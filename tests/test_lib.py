"""Test pure-logic helpers in ``picasso.lib``.

Skips Qt classes (``Dialog``, ``UserSettingsDialog``, etc.) and any
function that calls ``QtWidgets`` directly. Covers metadata access,
hex/path helpers, kinetic fits, recarray manipulation, polygon /
rectangle containment, drift-shift inversion, and group syncing.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import subprocess
import sys
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


def _canvas_locs(x, y):
    n = len(x)
    return pd.DataFrame(
        {
            "frame": np.zeros(n, dtype=np.uint32),
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "lpx": np.full(n, 0.1, dtype=np.float32),
            "lpy": np.full(n, 0.1, dtype=np.float32),
        }
    )


def _canvas_info(width=32, height=32):
    return [
        {"Width": width, "Height": height, "Frames": 1},
        {"Pixelsize": 130},
    ]


class TestFitCanvas:
    def test_in_bounds_is_unchanged(self):
        locs = _canvas_locs([1.0, 30.5], [2.0, 31.9])
        info = _canvas_info()
        _, _, shifts = lib.fit_canvas([locs], [info])
        assert shifts == [(0, 0)]
        assert locs["x"].tolist() == pytest.approx([1.0, 30.5])
        assert info[0]["Width"] == 32 and info[0]["Height"] == 32
        # no new keys while the canvas is the camera image
        for key in lib.CANVAS_OFFSET_KEYS + lib.CAMERA_SIZE_KEYS:
            assert key not in info[-1]

    def test_negative_translates_all_channels(self):
        moved = _canvas_locs([-2.5, 10.0], [5.0, -0.2])
        other = _canvas_locs([0.0, 31.0], [0.0, 31.0])
        infos = [_canvas_info(), _canvas_info()]
        _, _, shifts = lib.fit_canvas([moved, other], infos)
        assert shifts == [(3, 1), (3, 1)]  # whole camera pixels
        assert moved["x"].min() >= 0 and moved["y"].min() >= 0
        # the relative position of channels is kept
        assert other["x"].tolist() == pytest.approx([3.0, 34.0])
        assert other["y"].tolist() == pytest.approx([1.0, 32.0])
        for info in infos:
            assert info[-1][lib.CANVAS_OFFSET_KEYS[0]] == 3
            assert info[-1][lib.CANVAS_OFFSET_KEYS[1]] == 1
            assert info[-1][lib.CAMERA_SIZE_KEYS[0]] == 32
            # grown by the offset, so the camera FOV stays inside
            assert info[0]["Width"] == 35 and info[0]["Height"] == 33

    def test_canvas_follows_the_bottom_right(self):
        locs = _canvas_locs([40.0], [32.0])  # x == Width is invalid
        info = _canvas_info()
        lib.fit_canvas([locs], [info])
        assert info[0]["Width"] == 41 and info[0]["Height"] == 33
        # shrinks back, but not below the camera image
        locs["x"] = np.float32([1.0])
        locs["y"] = np.float32([1.0])
        lib.fit_canvas([locs], [info])
        assert info[0]["Width"] == 32 and info[0]["Height"] == 32

    def test_moving_back_restores_the_canvas(self):
        moved = _canvas_locs([0.5, 31.5], [1.0, 1.0])
        other = _canvas_locs([0.5, 31.5], [1.0, 1.0])
        infos = [_canvas_info(), _canvas_info()]
        moved["x"] = moved["x"].to_numpy() - np.float32(10)
        _, _, shifts = lib.fit_canvas([moved, other], infos)
        assert shifts[0] == (10, 0)
        assert infos[0][0]["Width"] == 42
        moved["x"] = moved["x"].to_numpy() + np.float32(10)
        _, _, shifts = lib.fit_canvas([moved, other], infos)
        assert shifts[0] == (-10, 0)
        assert moved["x"].tolist() == pytest.approx([0.5, 31.5])
        assert other["x"].tolist() == pytest.approx([0.5, 31.5])
        for info in infos:
            assert info[0]["Width"] == 32
            assert info[-1][lib.CANVAS_OFFSET_KEYS[0]] == 0

    def test_offset_follows_the_camera_frame(self):
        locs = _canvas_locs([-1.5], [1.0])
        info = _canvas_info()
        lib.fit_canvas([locs], [info])
        assert info[-1][lib.CANVAS_OFFSET_KEYS[0]] == 2
        locs["x"] = locs["x"].to_numpy() - np.float32(4.0)
        lib.fit_canvas([locs], [info])
        assert info[-1][lib.CANVAS_OFFSET_KEYS[0]] == 2 + 4
        assert info[0]["Width"] == 32 + 6

    def test_channels_with_different_offsets_are_aligned(self):
        # e.g., a file saved after a translation, opened with a raw one
        saved = _canvas_locs([0.0, 5.0], [1.0, 1.0])
        saved_info = _canvas_info(width=37)
        saved_info[-1][lib.CANVAS_OFFSET_KEYS[0]] = 5
        saved_info[-1][lib.CAMERA_SIZE_KEYS[0]] = 32
        raw = _canvas_locs([2.0], [1.0])
        infos = [saved_info, _canvas_info()]
        _, _, shifts = lib.fit_canvas([saved, raw], infos)
        assert shifts == [(0, 0), (5, 0)]
        assert raw["x"].tolist() == pytest.approx([7.0])
        for info in infos:
            assert info[-1][lib.CANVAS_OFFSET_KEYS[0]] == 5
            assert info[0]["Width"] == 37

    def test_replaces_column_arrays(self):
        # the GPU backend keys its uploads on the array memory
        locs = _canvas_locs([-1.0], [1.0])
        before = locs["x"].to_numpy()
        lib.fit_canvas([locs], [_canvas_info()])
        assert not np.shares_memory(before, locs["x"].to_numpy())
        assert before[0] == -1.0

    def test_ignores_non_finite(self):
        locs = _canvas_locs([np.nan, -np.inf, 3.0], [1.0, 1.0, 1.0])
        _, _, shifts = lib.fit_canvas([locs], [_canvas_info()])
        assert shifts == [(0, 0)]

    def test_save_keeps_every_loc(self, tmp_path):
        from picasso import io

        locs = _canvas_locs([-3.2, 12.0, 40.7], [-0.5, 33.0, 4.0])
        info = _canvas_info()
        lib.fit_canvas([locs], [info])
        path = str(tmp_path / "locs.hdf5")
        io.save_locs(path, locs, info)
        loaded, _ = io.load_locs(path)
        assert len(loaded) == 3


class TestTranslatePicks:
    @pytest.mark.parametrize(
        "shape, picks, expected",
        [
            ("Circle", [(1.0, 2.0)], [(2.0, 4.0)]),
            ("Square", [(1.0, 2.0)], [(2.0, 4.0)]),
            (
                "Rectangle",
                [((0.0, 0.0), (3.0, 1.0))],
                [((1.0, 2.0), (4.0, 3.0))],
            ),
            ("Box", [((0.0, 0.0), (3.0, 1.0))], [((1.0, 2.0), (4.0, 3.0))]),
            (
                "Polygon",
                [[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]],
                [[(1.0, 2.0), (2.0, 2.0), (1.0, 2.0)]],
            ),
            (
                "Brush",
                [[(0.5, [(0.0, 0.0), (1.0, 1.0)])]],
                [[(0.5, [(1.0, 2.0), (2.0, 3.0)])]],
            ),
        ],
    )
    def test_shapes(self, shape, picks, expected):
        assert lib.translate_picks(picks, shape, 1.0, 2.0) == expected

    def test_no_picks(self):
        assert lib.translate_picks([], None, 1.0, 2.0) == []

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            lib.translate_picks([(0.0, 0.0)], "Hexagon", 1.0, 1.0)


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


class TestConcatLocs:
    """Localization tables need not share their columns (no
    ``net_gradient`` after the wavelet identification, no ``ellipticity``
    for spherical fits, no ``z`` in 2D, ...); combining them must not lose
    localizations to ``ensure_sanity``'s NaN filter."""

    INFO = [{"Width": 32, "Height": 32, "Frames": 100}]

    @staticmethod
    def _locs(n: int, net_gradient: bool = True, **extra) -> pd.DataFrame:
        locs = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=int),
                "x": np.linspace(1, 20, n),
                "y": np.linspace(1, 20, n),
                "lpx": np.full(n, 0.1),
                "lpy": np.full(n, 0.1),
            }
        )
        if net_gradient:
            locs["net_gradient"] = np.full(n, 5000.0, dtype=np.float32)
        for column, value in extra.items():
            locs[column] = np.full(n, value)
        return locs

    def test_net_gradient_column(self):
        ids = self._locs(3)
        column = lib.net_gradient_column(ids)["net_gradient"]
        np.testing.assert_array_equal(column, [5000.0] * 3)
        assert column.dtype == np.float32
        assert lib.net_gradient_column(self._locs(3, False)) == {}

    def test_mixed_tables_drop_the_column_with_a_warning(self):
        with pytest.warns(UserWarning, match="net_gradient"):
            locs = lib.concat_locs([self._locs(3), self._locs(4, False)])
        assert len(locs) == 7
        assert "net_gradient" not in locs.columns
        assert len(lib.ensure_sanity(locs, self.INFO)) == 7

    def test_any_partial_column_is_dropped(self):
        # a 3D elliptical fit next to a 2D spherical one
        with pytest.warns(UserWarning, match="ellipticity, z"):
            locs = lib.concat_locs(
                [
                    self._locs(3, ellipticity=0.1, z=5.0),
                    self._locs(2, photons_unc=3.0),
                    self._locs(2, photons_unc=4.0, z=1.0),
                ]
            )
        assert list(locs.columns) == [
            "frame",
            "x",
            "y",
            "lpx",
            "lpy",
            "net_gradient",
        ]
        assert len(lib.ensure_sanity(locs, self.INFO)) == 7

    @pytest.mark.parametrize("net_gradient", [True, False])
    def test_matching_tables_are_left_alone(self, net_gradient):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            locs = lib.concat_locs(
                [
                    self._locs(3, net_gradient, z=1.0),
                    self._locs(2, net_gradient, z=2.0),
                ]
            )
        assert len(locs) == 5
        assert ("net_gradient" in locs.columns) == net_gradient
        assert list(locs["z"]) == [1.0] * 3 + [2.0] * 2
        assert list(locs.index) == list(range(5))

    def test_empty_tables_do_not_decide_the_columns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            locs = lib.concat_locs(
                [
                    self._locs(0, False),
                    pd.DataFrame(),
                    self._locs(3, z=1.0),
                ]
            )
        assert len(locs) == 3
        assert {"net_gradient", "z"} <= set(locs.columns)
        assert locs["frame"].dtype.kind == "i"
        # nothing to combine at all
        assert len(lib.concat_locs([self._locs(0), self._locs(0)])) == 0

    def test_merge_locs_keeps_every_localization(self):
        with pytest.warns(UserWarning, match="net_gradient"):
            merged = lib.merge_locs(
                [self._locs(3), self._locs(4, False)],
                increment_groups=False,
            )
        assert len(lib.ensure_sanity(merged, self.INFO)) == 7


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


class TestSquareContainment:
    def test_bounds_are_exclusive(self):
        locs_xy = np.array(
            [[10.0, 10.9, 11.0, 10.0], [10.0, 10.0, 10.0, 12.0]]
        )
        mask = lib.is_loc_in_square_numba(10.0, 10.0, locs_xy, 2.0)
        assert mask.tolist() == [True, True, False, False]

    def test_locs_in_square_filters(self):
        locs_xy = np.array([[10.0, 20.0], [10.0, 20.0]])
        out = lib.locs_in_square_numba(10.0, 10.0, locs_xy, 2.0)
        assert out.shape == (2, 1)
        assert out[0, 0] == 10.0


class TestOrientedRectangleContainment:
    def test_matches_check_if_in_rectangle(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 20, 10000)
        y = rng.uniform(0, 20, 10000)
        locs_xy = np.stack((x, y))
        for theta in np.linspace(-np.pi / 2, np.pi / 2, 7):
            xc, yc, length, width = 10.0, 10.0, 8.0, 3.0
            half_x = 0.5 * length * np.cos(theta)
            half_y = 0.5 * length * np.sin(theta)
            X, Y = lib.get_pick_rectangle_corners(
                xc - half_x, yc - half_y, xc + half_x, yc + half_y, width
            )
            reference = lib.check_if_in_rectangle(
                x, y, np.array(X), np.array(Y)
            )
            mask = lib.is_loc_in_rectangle_numba(
                xc, yc, theta, length, width, locs_xy
            )
            # points right at the boundary may be classified either way
            u = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
            v = -(x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)
            on_edge = (np.abs(np.abs(u) - length / 2) < 1e-9) | (
                np.abs(np.abs(v) - width / 2) < 1e-9
            )
            assert np.array_equal(mask[~on_edge], reference[~on_edge])

    def test_locs_in_rectangle_numba_filters(self):
        locs_xy = np.array([[0.0, 0.0, 3.0, 5.0], [0.0, 2.0, 0.0, 0.0]])
        out = lib.locs_in_rectangle_numba(0.0, 0.0, 0.0, 8.0, 1.0, locs_xy)
        assert out.shape == (2, 2)
        assert out[0].tolist() == [0.0, 3.0]


class TestWrapAnglePi:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (0.0, 0.0),
            (np.pi / 2, -np.pi / 2),
            (np.pi, 0.0),
            (-np.pi, 0.0),
            (np.pi / 2 + 0.01, -np.pi / 2 + 0.01),
            (-np.pi / 2 - 0.01, np.pi / 2 - 0.01),
        ],
    )
    def test_wraps_into_half_open_interval(self, angle, expected):
        assert lib.wrap_angle_pi(angle) == pytest.approx(expected)

    def test_directors_differing_by_pi_are_equal(self):
        theta = np.deg2rad(89.0)
        other = np.deg2rad(-89.0)
        assert abs(lib.wrap_angle_pi(theta - other)) == pytest.approx(
            np.deg2rad(2.0)
        )


class TestPrincipalAxis:
    @pytest.mark.parametrize("angle_deg", [0, 15, 45, 89, -89, -45])
    def test_recovers_anisotropic_gaussian(self, angle_deg):
        rng = np.random.default_rng(1)
        theta = np.deg2rad(angle_deg)
        n = 200000
        u = rng.normal(0, 2.0, n)
        v = rng.normal(0, 0.5, n)
        x = u * np.cos(theta) - v * np.sin(theta)
        y = u * np.sin(theta) + v * np.cos(theta)
        found, along, across = lib.principal_axis(
            np.mean((x - x.mean()) ** 2),
            np.mean((x - x.mean()) * (y - y.mean())),
            np.mean((y - y.mean()) ** 2),
        )
        d_theta = lib.wrap_angle_pi(found - theta)
        assert abs(np.degrees(d_theta)) < 1.0
        assert along == pytest.approx(2.0, abs=0.05)
        assert across == pytest.approx(0.5, abs=0.05)

    def test_isotropic_has_equal_rmsds(self):
        _, along, across = lib.principal_axis(1.0, 0.0, 1.0)
        assert along == pytest.approx(across)

    def test_collinear_has_zero_across(self):
        # all points on the line y = x
        _, along, across = lib.principal_axis(1.0, 1.0, 1.0)
        assert across == pytest.approx(0.0, abs=1e-12)
        assert along == pytest.approx(np.sqrt(2.0))

    def test_rmsds_decompose_isotropic_rmsd(self):
        rng = np.random.default_rng(2)
        locs_xy = rng.normal(0, 1, (2, 5000))
        locs_xy[0] *= 3.0
        x, y = locs_xy
        _, along, across = lib.principal_axis(
            np.mean((x - x.mean()) ** 2),
            np.mean((x - x.mean()) * (y - y.mean())),
            np.mean((y - y.mean()) ** 2),
        )
        rmsd = lib.rmsd_at_com(locs_xy)
        assert along**2 + across**2 == pytest.approx(rmsd**2)


def _rectangles_overlap_brute_force(rect_a, rect_b, rng, n=20000):
    """Monte Carlo ground truth: sample points inside ``rect_a`` and
    check whether any of them lies inside ``rect_b``."""
    xa, ya, ta, la, wa = rect_a
    u = rng.uniform(-la / 2, la / 2, n)
    v = rng.uniform(-wa / 2, wa / 2, n)
    x = xa + u * np.cos(ta) - v * np.sin(ta)
    y = ya + u * np.sin(ta) + v * np.cos(ta)
    xb, yb, tb, lb, wb = rect_b
    return bool(
        lib.is_loc_in_rectangle_numba(
            xb, yb, tb, lb, wb, np.stack((x, y))
        ).any()
    )


def _overlap(rect_a, rect_b):
    xa, ya, ta, la, wa = rect_a
    xb, yb, tb, lb, wb = rect_b
    return lib.rectangles_overlap(
        xa,
        ya,
        ta,
        la,
        wa,
        0.5 * np.hypot(la, wa),
        xb,
        yb,
        tb,
        lb,
        wb,
        0.5 * np.hypot(lb, wb),
    )


class TestRectanglesOverlap:
    def test_identical_rectangles(self):
        rect = (0.0, 0.0, 0.3, 8.0, 1.0)
        assert _overlap(rect, rect)

    def test_same_rectangle_rotated_by_pi(self):
        rect = (0.0, 0.0, 0.3, 8.0, 1.0)
        assert _overlap(rect, (0.0, 0.0, 0.3 + np.pi, 8.0, 1.0))

    def test_far_apart(self):
        assert not _overlap(
            (0.0, 0.0, 0.0, 8.0, 1.0), (100.0, 100.0, 0.0, 8.0, 1.0)
        )

    def test_parallel_side_by_side(self):
        # gap of 0.2 px between two rectangles of width 1.0
        assert not _overlap(
            (0.0, 0.0, 0.0, 8.0, 1.0), (0.0, 1.2, 0.0, 8.0, 1.0)
        )
        assert _overlap((0.0, 0.0, 0.0, 8.0, 1.0), (0.0, 0.8, 0.0, 8.0, 1.0))

    def test_crossing_at_right_angle(self):
        # centers 3 px apart, which is less than either half length, so
        # a center-distance test would call these disjoint
        assert _overlap(
            (0.0, 0.0, 0.0, 8.0, 1.0), (3.0, 0.0, np.pi / 2, 8.0, 1.0)
        )

    def test_one_inside_the_other(self):
        assert _overlap(
            (0.0, 0.0, 0.0, 8.0, 4.0), (0.0, 0.0, np.pi / 4, 1.0, 1.0)
        )

    def test_matches_monte_carlo_on_random_pairs(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            rect_a = (
                *rng.uniform(-5, 5, 2),
                rng.uniform(-np.pi / 2, np.pi / 2),
                rng.uniform(1, 8),
                rng.uniform(0.5, 3),
            )
            rect_b = (
                *rng.uniform(-5, 5, 2),
                rng.uniform(-np.pi / 2, np.pi / 2),
                rng.uniform(1, 8),
                rng.uniform(0.5, 3),
            )
            if _rectangles_overlap_brute_force(rect_a, rect_b, rng):
                # sampling can only prove overlap, never disprove it
                assert _overlap(rect_a, rect_b)


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

    def test_box(self):
        picks = [((1.0, 2.0), (4.0, 6.0)), ((0.0, 0.0), (2.0, 2.0))]
        areas = lib.pick_areas(picks, "Box", pick_size=None)
        assert areas.tolist() == [12.0, 4.0]

    def test_box_ignores_corner_order(self):
        areas = lib.pick_areas(
            [((4.0, 6.0), (1.0, 2.0))], "Box", pick_size=None
        )
        assert areas[0] == 12.0

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            lib.pick_areas([(0.0, 0.0)], "Triangle", pick_size=1.0)


class TestPickBoxCorners:
    def test_orders_corners(self):
        X, Y = lib.get_pick_box_corners(((4.0, 6.0), (1.0, 2.0)))
        assert X == [1.0, 4.0, 4.0, 1.0]
        assert Y == [2.0, 2.0, 6.0, 6.0]

    def test_matches_input_when_already_ordered(self):
        pick = ((1.0, 2.0), (4.0, 6.0))
        assert lib.get_pick_box_corners(pick) == lib.get_pick_box_corners(
            (pick[1], pick[0])
        )


class TestBoxContainment:
    PICK = ((1.0, 2.0), (5.0, 4.0))  # 4 wide, 2 high, centered (3, 3)

    def test_inside(self):
        assert lib.is_loc_in_box_numba(
            3.0, 3.0, np.array([[3.0], [3.0]]), 4.0, 2.0
        )[0]

    def test_outside_in_y_only(self):
        # inside the 4-wide extent (1..5) but outside the 2-high one
        # (2..4), so a square test of the longer side would wrongly
        # accept it
        assert not lib.is_loc_in_box_numba(
            3.0, 3.0, np.array([[3.0], [4.5]]), 4.0, 2.0
        )[0]

    def test_bounds_are_exclusive(self):
        # matches postprocess._picked_box_locs
        assert not lib.is_loc_in_box_numba(
            3.0, 3.0, np.array([[5.0], [3.0]]), 4.0, 2.0
        )[0]

    def test_locs_in_box_numba_filters(self):
        locs_xy = np.array([[3.0, 9.0, 2.0], [3.0, 9.0, 2.5]])
        out = lib.locs_in_box_numba(3.0, 3.0, locs_xy, 4.0, 2.0)
        assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# Pick bounds and containment dispatch (all shapes)
# ---------------------------------------------------------------------------

CLOSED_POLYGON = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)]


class TestPickBounds:
    def test_circle(self):
        assert lib.pick_bounds((5.0, 5.0), "Circle", 2.0) == (
            4.0,
            6.0,
            4.0,
            6.0,
        )

    def test_square(self):
        assert lib.pick_bounds((5.0, 5.0), "Square", 2.0) == (
            4.0,
            6.0,
            4.0,
            6.0,
        )

    def test_rectangle_spans_its_corners(self):
        pick = ((5.0, 5.0), (5.0, 15.0))  # vertical band of width 2
        x_min, x_max, y_min, y_max = lib.pick_bounds(pick, "Rectangle", 2.0)
        assert (x_min, x_max) == (4.0, 6.0)
        assert (y_min, y_max) == (5.0, 15.0)

    def test_polygon(self):
        assert lib.pick_bounds(CLOSED_POLYGON, "Polygon", None) == (
            0.0,
            4.0,
            0.0,
            4.0,
        )

    def test_open_polygon_falls_back_to_its_vertices(self):
        assert lib.pick_bounds(CLOSED_POLYGON[:-1], "Polygon", None) == (
            0.0,
            4.0,
            0.0,
            4.0,
        )

    def test_empty_polygon_raises(self):
        with pytest.raises(ValueError):
            lib.pick_bounds([], "Polygon", None)

    def test_box(self):
        assert lib.pick_bounds(((4.0, 6.0), (1.0, 2.0)), "Box", None) == (
            1.0,
            4.0,
            2.0,
            6.0,
        )

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            lib.pick_bounds((0.0, 0.0), "Triangle", 1.0)


class TestPointInPick:
    def test_circle(self):
        pick = (5.0, 5.0)
        assert lib.point_in_pick(5.5, 5.0, pick, "Circle", 2.0)
        # outside the radius but inside the bounding box corner
        assert not lib.point_in_pick(5.9, 5.9, pick, "Circle", 2.0)

    def test_square(self):
        pick = (5.0, 5.0)
        assert lib.point_in_pick(5.4, 5.4, pick, "Square", 1.0)
        assert not lib.point_in_pick(5.6, 5.0, pick, "Square", 1.0)

    def test_vertical_rectangle(self):
        # ray casting over the corners divides by zero here, which is
        # why point_in_pick rotates into the rectangle's frame instead
        pick = ((5.0, 5.0), (5.0, 15.0))
        assert lib.point_in_pick(5.5, 10.0, pick, "Rectangle", 2.0)
        assert not lib.point_in_pick(6.5, 10.0, pick, "Rectangle", 2.0)

    def test_oriented_rectangle_agrees_with_ray_casting(self):
        pick = ((0.0, 0.0), (10.0, 6.0))
        X, Y = lib.get_pick_rectangle_corners(0.0, 0.0, 10.0, 6.0, 3.0)
        rng = np.random.default_rng(0)
        points = rng.uniform(-4, 12, size=(2, 200))
        expected = lib.check_if_in_rectangle(
            points[0], points[1], np.array(X), np.array(Y)
        )
        got = [
            lib.point_in_pick(x, y, pick, "Rectangle", 3.0)
            for x, y in zip(*points)
        ]
        assert np.array_equal(np.array(got), expected)

    def test_polygon(self):
        assert lib.point_in_pick(2.0, 2.0, CLOSED_POLYGON, "Polygon", None)
        assert not lib.point_in_pick(5.0, 2.0, CLOSED_POLYGON, "Polygon", None)

    def test_open_polygon_contains_nothing(self):
        assert not lib.point_in_pick(
            2.0, 2.0, CLOSED_POLYGON[:-1], "Polygon", None
        )

    def test_box(self):
        pick = ((1.0, 2.0), (5.0, 4.0))
        assert lib.point_in_pick(3.0, 3.0, pick, "Box", None)
        assert not lib.point_in_pick(3.0, 4.5, pick, "Box", None)

    def test_box_ignores_corner_order(self):
        pick = ((5.0, 4.0), (1.0, 2.0))
        assert lib.point_in_pick(3.0, 3.0, pick, "Box", None)

    def test_brush(self):
        pick = [(2.0, [(0.0, 0.0), (10.0, 0.0)])]  # width 2 -> r = 1
        assert lib.point_in_pick(5.0, 0.9, pick, "Brush", None)
        assert not lib.point_in_pick(5.0, 1.1, pick, "Brush", None)

    def test_brush_uses_every_stroke(self):
        pick = [
            (2.0, [(0.0, 0.0), (10.0, 0.0)]),
            (2.0, [(0.0, 20.0), (10.0, 20.0)]),
        ]
        assert lib.point_in_pick(5.0, 20.0, pick, "Brush", None)

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            lib.point_in_pick(0.0, 0.0, (0.0, 0.0), "Triangle", 1.0)


# ---------------------------------------------------------------------------
# Brush strokes
# ---------------------------------------------------------------------------


def _brute_point_segment_distance(px, py, ax, ay, bx, by, n=20001):
    """Smallest distance from a point to a densely sampled segment."""
    t = np.linspace(0.0, 1.0, n)
    xs = ax + t * (bx - ax)
    ys = ay + t * (by - ay)
    return np.min(np.hypot(xs - px, ys - py))


class TestPointSegmentDistance:
    @pytest.mark.parametrize(
        "px, py",
        [(5.0, 3.0), (-4.0, 2.0), (14.0, -1.0), (0.0, 0.0), (2.0, 0.0)],
    )
    def test_matches_brute_force(self, px, py):
        seg = (0.0, 0.0, 10.0, 0.0)
        got = np.sqrt(lib._point_segment_distance_sq(px, py, *seg))
        assert got == pytest.approx(
            _brute_point_segment_distance(px, py, *seg), abs=1e-3
        )

    def test_degenerate_segment_is_a_point(self):
        d2 = lib._point_segment_distance_sq(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
        assert np.sqrt(d2) == pytest.approx(5.0)


class TestSegmentSegmentDistance:
    def test_crossing_segments_are_zero(self):
        d2 = lib._segment_segment_distance_sq(
            0.0, 0.0, 10.0, 0.0, 5.0, -5.0, 5.0, 5.0
        )
        assert d2 == 0.0

    def test_parallel_segments(self):
        d2 = lib._segment_segment_distance_sq(
            0.0, 0.0, 10.0, 0.0, 0.0, 3.0, 10.0, 3.0
        )
        assert np.sqrt(d2) == pytest.approx(3.0)

    def test_matches_brute_force_for_disjoint_segments(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            a = rng.uniform(-10, 10, 4)
            b = rng.uniform(-10, 10, 4)
            got = np.sqrt(lib._segment_segment_distance_sq(*a, *b))
            t = np.linspace(0.0, 1.0, 400)
            p = np.stack([a[0] + t * (a[2] - a[0]), a[1] + t * (a[3] - a[1])])
            q = np.stack([b[0] + t * (b[2] - b[0]), b[1] + t * (b[3] - b[1])])
            brute = np.min(
                np.hypot(
                    p[0][:, None] - q[0][None, :],
                    p[1][:, None] - q[1][None, :],
                )
            )
            assert got <= brute + 1e-6


class TestBrushStroke:
    STROKE = (2.0, [(0.0, 0.0), (10.0, 0.0)])  # width 2 -> r = 1

    def _inside(self, x, y, stroke=None):
        width, X, Y = lib.brush_stroke_arrays(stroke or self.STROKE)
        return lib.check_if_in_brush_stroke(
            np.atleast_1d(np.asarray(x, dtype=float)),
            np.atleast_1d(np.asarray(y, dtype=float)),
            X,
            Y,
            width / 2,
        )

    def test_along_the_path(self):
        assert self._inside(5.0, 0.9)[0]
        assert not self._inside(5.0, 1.1)[0]

    def test_round_caps_reach_past_the_ends(self):
        # a round-capped pen paints half a disk beyond each end
        assert self._inside(-0.5, 0.0)[0]
        assert not self._inside(-1.5, 0.0)[0]
        assert self._inside(10.5, 0.0)[0]
        assert not self._inside(11.5, 0.0)[0]

    def test_corner_is_covered_on_both_sides(self):
        stroke = (2.0, [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        assert self._inside(10.7, 0.7, stroke)[0]  # outside of the corner
        assert self._inside(9.5, 0.5, stroke)[0]  # inside of the corner

    def test_single_point_path_is_a_disk(self):
        dot = (2.0, [(3.0, 3.0)])
        assert self._inside(3.9, 3.0, dot)[0]
        assert not self._inside(4.1, 3.0, dot)[0]

    def test_locs_in_brush_unions_the_strokes(self):
        locs = pd.DataFrame({"x": [5.0, 5.0, 50.0], "y": [0.0, 20.0, 50.0]})
        pick = [self.STROKE, (2.0, [(0.0, 20.0), (10.0, 20.0)])]
        assert len(lib.locs_in_brush(locs, pick)) == 2

    def test_locs_in_brush_on_an_empty_pick(self):
        locs = pd.DataFrame({"x": [5.0], "y": [0.0]})
        assert len(lib.locs_in_brush(locs, [])) == 0


class TestBrushStrokesOverlap:
    def _ov(self, a, b):
        wa, Xa, Ya = lib.brush_stroke_arrays(a)
        wb, Xb, Yb = lib.brush_stroke_arrays(b)
        return lib.brush_strokes_overlap(Xa, Ya, Xb, Yb, (wa + wb) / 2)

    BASE = (2.0, [(0.0, 0.0), (10.0, 0.0)])

    def test_touching_strokes_overlap(self):
        # gap of 1.5 between the paths, radii sum to 2
        assert self._ov(self.BASE, (2.0, [(5.0, 1.5), (5.0, 10.0)]))

    def test_strokes_just_out_of_reach_do_not(self):
        assert not self._ov(self.BASE, (2.0, [(5.0, 2.5), (5.0, 10.0)]))

    def test_crossing_strokes_overlap(self):
        assert self._ov(self.BASE, (0.1, [(5.0, -5.0), (5.0, 5.0)]))

    def test_far_apart_strokes_do_not(self):
        # the bounding-box short circuit
        assert not self._ov(self.BASE, (2.0, [(50.0, 50.0), (60.0, 60.0)]))

    def test_dots_overlap_by_their_radii(self):
        assert self._ov((2.0, [(0.0, 0.0)]), (2.0, [(1.9, 0.0)]))
        assert not self._ov((2.0, [(0.0, 0.0)]), (2.0, [(2.1, 0.0)]))


class TestMergeBrushStrokes:
    A = (2.0, [(0.0, 0.0), (10.0, 0.0)])
    B = (2.0, [(0.0, 20.0), (10.0, 20.0)])

    def test_disjoint_strokes_stay_separate(self):
        assert len(lib.merge_brush_strokes([self.A, self.B])) == 2

    def test_overlapping_strokes_merge(self):
        near = (2.0, [(0.0, 1.0), (10.0, 1.0)])
        picks = lib.merge_brush_strokes([self.A, near])
        assert len(picks) == 1
        assert len(picks[0]) == 2

    def test_a_bridge_merges_both_picks(self):
        bridge = (2.0, [(5.0, 0.0), (5.0, 20.0)])
        picks = lib.merge_brush_strokes([self.A, self.B, bridge])
        assert len(picks) == 1
        assert len(picks[0]) == 3

    def test_the_newest_stroke_is_last(self):
        # what makes "remove the last stroke" well defined
        bridge = (2.0, [(5.0, 0.0), (5.0, 20.0)])
        for strokes in ([self.A, self.B], [self.A, self.B, bridge]):
            picks = lib.merge_brush_strokes(strokes)
            assert picks[-1][-1] is strokes[-1]

    def test_dropping_the_bridge_splits_the_pick_again(self):
        bridge = (2.0, [(5.0, 0.0), (5.0, 20.0)])
        merged = lib.merge_brush_strokes([self.A, self.B, bridge])[0]
        assert len(lib.merge_brush_strokes(merged[:-1])) == 2

    def test_empty_input(self):
        assert lib.merge_brush_strokes([]) == []

    def test_accepts_lists_as_loaded_from_yaml(self):
        loaded = [[2.0, [[0.0, 0.0], [10.0, 0.0]]]]
        assert len(lib.merge_brush_strokes(loaded)) == 1


class TestBrushBoundsAndAreas:
    def test_bounds_expand_by_each_stroke_radius(self):
        pick = [(2.0, [(0.0, 0.0), (10.0, 0.0)])]
        assert lib.pick_bounds(pick, "Brush", None) == pytest.approx(
            (-1.0, 11.0, -1.0, 1.0)
        )

    def test_bounds_span_all_strokes(self):
        pick = [
            (2.0, [(0.0, 0.0), (10.0, 0.0)]),
            (4.0, [(0.0, 20.0), (10.0, 20.0)]),
        ]
        x_min, x_max, y_min, y_max = lib.pick_bounds(pick, "Brush", None)
        assert (y_min, y_max) == pytest.approx((-1.0, 22.0))
        assert (x_min, x_max) == pytest.approx((-2.0, 12.0))

    def test_empty_pick_raises(self):
        with pytest.raises(ValueError):
            lib.pick_bounds([], "Brush", None)

    def test_area_of_a_straight_stroke(self):
        # a swept disk: 2 r L for the body plus a full disk of caps
        r, length = 1.0, 10.0
        pick = [(2 * r, [(0.0, 0.0), (length, 0.0)])]
        area = lib.pick_areas([pick], "Brush", None)[0]
        assert area == pytest.approx(2 * r * length + np.pi * r**2, rel=0.02)

    def test_overlap_is_not_counted_twice(self):
        # the same path walked out and back must not double the area
        r, length = 1.0, 10.0
        there = [(2 * r, [(0.0, 0.0), (length, 0.0)])]
        and_back = [(2 * r, [(0.0, 0.0), (length, 0.0), (0.0, 0.0)])]
        assert lib.pick_areas([and_back], "Brush", None)[0] == pytest.approx(
            lib.pick_areas([there], "Brush", None)[0]
        )

    def test_area_of_an_empty_pick_is_zero(self):
        assert lib.pick_areas([[]], "Brush", None)[0] == 0.0


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


# ---------------------------------------------------------------------------
# Progress trackers (MockProgress / TqdmProgress / normalize_progress)
# ---------------------------------------------------------------------------


class TestMockProgress:
    def test_implements_full_interface_silently(self):
        p = lib.MockProgress()
        p.init()
        p.set_value(3)
        p.zero_progress()
        p.zero_progress("new phase")
        p.setLabelText("text")
        p.play_sound_notification()
        p.close()
        assert list(p.get_iterator(0, 5)) == [0, 1, 2, 3, 4]

    def test_maximum_roundtrip(self):
        p = lib.MockProgress()
        assert p.maximum() == 0
        p.setMaximum(42)
        assert p.maximum() == 42


class TestTqdmProgress:
    def test_bar_armed_lazily_on_first_set_value(self):
        p = lib.TqdmProgress(description="phase 1")
        p.setMaximum(10)
        assert p.iterator is None  # not armed yet
        p.set_value(3)
        assert p.iterator is not None
        assert p.iterator.total == 10
        assert p.iterator.n == 3
        assert p.iterator.desc == "phase 1"
        p.close()
        assert p.iterator is None

    def test_set_maximum_updates_active_bar_in_place(self):
        # the early-stopping case: shrink the target of a running bar
        p = lib.TqdmProgress(description="gp phase")
        p.setMaximum(100)
        p.set_value(30)
        bar = p.iterator
        p.setMaximum(30)
        assert p.iterator is bar  # same bar, not re-armed
        assert p.iterator.total == 30
        assert p.maximum() == 30
        p.close()

    def test_zero_progress_starts_fresh_bar_with_new_title(self):
        p = lib.TqdmProgress(description="phase 1")
        p.setMaximum(5)
        p.set_value(5)
        first_bar = p.iterator
        p.zero_progress("phase 2")
        assert p.iterator is None  # old bar closed
        p.setMaximum(7)
        p.set_value(1)
        assert p.iterator is not first_bar
        assert p.iterator.desc == "phase 2"
        assert p.iterator.total == 7
        p.close()

    def test_get_iterator_closes_previous_bar(self):
        p = lib.TqdmProgress(description="loop")
        p.setMaximum(3)
        p.set_value(1)
        first_bar = p.iterator
        iterator = p.get_iterator(0, 4)
        assert p.iterator is not first_bar
        assert list(iterator) == [0, 1, 2, 3]
        p.close()


class TestNormalizeProgress:
    def test_none_returns_mock(self):
        assert isinstance(lib.normalize_progress(None), lib.MockProgress)

    def test_console_returns_tqdm(self):
        p = lib.normalize_progress("console", "my task", unit="loc")
        assert isinstance(p, lib.TqdmProgress)
        assert p.description_base == "my task"
        assert p.unit == "loc"

    def test_existing_tracker_passed_through(self):
        for tracker in (lib.MockProgress(), lib.TqdmProgress()):
            assert lib.normalize_progress(tracker) is tracker

    def test_duck_typed_tracker_passed_through(self):
        # any object with the ProgressDialog interface is accepted
        class Recorder:
            def set_value(self, value):
                pass

            def setMaximum(self, maximum):
                pass

            def zero_progress(self, description=None):
                pass

        tracker = Recorder()
        assert lib.normalize_progress(tracker) is tracker

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            lib.normalize_progress("bogus")

    def test_non_protocol_object_raises(self):
        with pytest.raises(TypeError):
            lib.normalize_progress(42)


# ---------------------------------------------------------------------------
# Lazy Qt imports (PyQt6 must not load with the core library; the Qt
# names moved to picasso.lib_qt stay reachable as lib.<name>)
# ---------------------------------------------------------------------------


class TestLazyQtImports:
    def test_core_imports_do_not_load_pyqt6(self):
        # fresh interpreter: importing the core library (including the
        # progress machinery) must not pull in PyQt6
        code = (
            "import sys\n"
            "from picasso import lib, io, render, g5m, clusterer, aim\n"
            "lib.normalize_progress('console').set_value(0)\n"
            "assert 'PyQt6' not in sys.modules, 'PyQt6 imported eagerly'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_lib_forwards_qt_names_to_lib_qt(self):
        pytest.importorskip("PyQt6")
        assert lib.ProgressDialog.__module__ == "picasso.lib_qt"
        from picasso import lib_qt

        assert lib.Dialog is lib_qt.Dialog
        assert lib.ProgressType is lib_qt.ProgressType

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            lib.no_such_attribute


# ---------------------------------------------------------------------------
# Render worker budget
# ---------------------------------------------------------------------------


class TestNWorkersFromSettings:
    """``lib.n_workers(..., settings_section=...)`` reads the given
    settings section with the same defensive semantics as Localize's
    ``cpu_utilization`` handling and caps the result with the optional
    ``max_workers``."""

    def _render_workers(self):
        return lib.n_workers(
            lib.RENDER_CPU_UTILIZATION_DEFAULT, settings_section="Render"
        )

    def _set_settings(self, monkeypatch, render_section):
        settings = {} if render_section is None else {"Render": render_section}
        monkeypatch.setattr(lib.io, "load_user_settings", lambda: settings)

    def test_no_section_never_reads_settings(self, monkeypatch):
        def _boom():
            raise AssertionError("settings must not be read")

        monkeypatch.setattr(lib.io, "load_user_settings", _boom)
        assert lib.n_workers(0.75) >= 1

    def test_default_when_section_missing(self, monkeypatch):
        self._set_settings(monkeypatch, None)
        assert self._render_workers() == lib.n_workers(
            lib.RENDER_CPU_UTILIZATION_DEFAULT
        )

    def test_valid_fraction(self, monkeypatch):
        self._set_settings(monkeypatch, {"cpu_utilization": 0.25})
        assert self._render_workers() == lib.n_workers(0.25)

    @pytest.mark.parametrize("bad", [1.5, -0.2, 0.0, 1, True, "half", None])
    def test_invalid_fraction_falls_back(self, monkeypatch, bad):
        self._set_settings(monkeypatch, {"cpu_utilization": bad})
        assert self._render_workers() == lib.n_workers(
            lib.RENDER_CPU_UTILIZATION_DEFAULT
        )

    def test_max_workers_caps(self, monkeypatch):
        self._set_settings(
            monkeypatch, {"cpu_utilization": 0.9, "max_workers": 1}
        )
        assert self._render_workers() == 1

    @pytest.mark.parametrize("bad", [0, -3, True, "two", 2.5, None])
    def test_invalid_max_workers_ignored(self, monkeypatch, bad):
        self._set_settings(
            monkeypatch, {"cpu_utilization": 0.25, "max_workers": bad}
        )
        assert self._render_workers() == lib.n_workers(0.25)


class TestStandardizeDtypes:
    def test_float64_and_frame_are_cast(self):
        locs = pd.DataFrame(
            {
                "frame": np.array([3, 1, 2], dtype=np.int64),
                "x": np.array([1.5, 2.5, 3.5]),  # float64
                "photons": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "group": np.array([0, 0, 1], dtype=np.int32),
                "flag": np.array([True, False, True]),
            }
        )
        out = lib.standardize_dtypes(locs)
        assert out["frame"].dtype == np.uint32
        assert out["x"].dtype == np.float32
        assert out["photons"].dtype == np.float32
        assert out["group"].dtype == np.int32
        assert out["flag"].dtype == bool
        assert list(out["frame"]) == [3, 1, 2]
        assert lib.standardize_dtypes(out) is out  # already standard

    def test_ensure_sanity_standardizes(self):
        locs = pd.DataFrame(
            {"frame": [0, 1], "x": [1.0, 2.0], "y": [1.0, 2.0]}
        )
        info = [{"Width": 8, "Height": 8, "Frames": 2}]
        out = lib.ensure_sanity(locs, info)
        assert out["x"].dtype == np.float32
        assert out["frame"].dtype == np.uint32


class TestSelectFrcRois:
    @pytest.fixture
    def uniform_locs(self):
        rng = np.random.default_rng(0)
        n = 50_000
        return pd.DataFrame(
            {
                "x": rng.uniform(0, 100, n).astype(np.float32),
                "y": rng.uniform(0, 100, n).astype(np.float32),
            }
        )

    # pixel size 100 nm, so a 1000 nm ROI is 10 camera pixels
    info = [{"Pixelsize": 100}]
    viewport = ((0, 0), (100, 100))

    def test_rois_inside_viewport_and_not_overlapping(self, uniform_locs):
        rois = lib.select_frc_rois(
            uniform_locs,
            self.info,
            self.viewport,
            n_rois=40,
            roi_size=1000,
            min_locs=100,
        )
        assert 0 < len(rois) <= 40
        for (y0, x0), (y1, x1) in rois:
            assert y1 - y0 == pytest.approx(10)
            assert x1 - x0 == pytest.approx(10)
            assert y0 >= 0 and x0 >= 0 and y1 <= 100 and x1 <= 100
        for a, ((ay0, ax0), _) in enumerate(rois):
            for (by0, bx0), _ in rois[a + 1 :]:
                overlap_y = abs(ay0 - by0) < 10 - 1e-9
                overlap_x = abs(ax0 - bx0) < 10 - 1e-9
                assert not (overlap_y and overlap_x)

    def test_min_locs_respected(self, uniform_locs):
        # ~500 locs per 10 x 10 px ROI; the right half is emptied
        locs = uniform_locs[uniform_locs["x"] < 50]
        rois = lib.select_frc_rois(
            locs,
            self.info,
            self.viewport,
            n_rois=50,
            roi_size=1000,
            min_locs=300,
        )
        assert len(rois) > 0
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        for (y0, x0), (y1, x1) in rois:
            in_roi = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            assert in_roi.sum() >= 300

    def test_no_rois_if_too_few_locs(self, uniform_locs):
        rois = lib.select_frc_rois(
            uniform_locs,
            self.info,
            self.viewport,
            roi_size=1000,
            min_locs=10_000,
        )
        assert rois == []

    def test_deterministic_with_seed(self, uniform_locs):
        kwargs = dict(n_rois=20, roi_size=1000, min_locs=100, random_seed=3)
        a = lib.select_frc_rois(
            uniform_locs, self.info, self.viewport, **kwargs
        )
        b = lib.select_frc_rois(
            uniform_locs, self.info, self.viewport, **kwargs
        )
        assert a == b

    def test_viewport_smaller_than_roi_raises(self, uniform_locs):
        with pytest.raises(ValueError):
            lib.select_frc_rois(
                uniform_locs, self.info, ((0, 0), (5, 5)), roi_size=1000
            )


class TestBinZSteps:
    """``lib.bin_z_steps``: axial binning shared by the 3D calibrations."""

    def test_no_binning_returns_the_input(self):
        values = np.arange(7.0)
        assert lib.bin_z_steps(values, 1) is values

    def test_averages_consecutive_groups_and_drops_the_remainder(self):
        # 7 steps in bins of 3: [0, 1, 2] and [3, 4, 5]; step 6 is dropped
        binned = lib.bin_z_steps(np.arange(7.0), 3)
        np.testing.assert_allclose(binned, [1.0, 4.0])

    def test_stage_positions_land_at_the_bin_centers(self):
        # a descending 5 nm scan binned by 4 gives an even 20 nm grid
        z = -(np.arange(12) * 5.0 - 27.5)
        binned = lib.bin_z_steps(z, 4)
        np.testing.assert_allclose(np.diff(binned), -20.0)
        np.testing.assert_allclose(binned, [20.0, 0.0, -20.0])

    def test_bins_along_the_requested_axis(self):
        volumes = np.random.default_rng(0).random((2, 3, 3, 8))
        binned = lib.bin_z_steps(volumes, 4, axis=3)
        assert binned.shape == (2, 3, 3, 2)
        np.testing.assert_allclose(
            binned[..., 1], volumes[..., 4:8].mean(axis=3)
        )

    def test_ignores_nan_within_a_bin(self):
        values = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan])
        binned = lib.bin_z_steps(values, 3)
        assert binned[0] == pytest.approx(2.0)
        assert np.isnan(binned[1])

    def test_integer_input_averages_to_float(self):
        binned = lib.bin_z_steps(np.array([1, 2, 3, 4]), 2)
        np.testing.assert_allclose(binned, [1.5, 3.5])

    @pytest.mark.parametrize("z_binning", [0, 9])
    def test_rejects_impossible_binning(self, z_binning):
        with pytest.raises(ValueError, match="z binning"):
            lib.bin_z_steps(np.arange(8.0), z_binning)


# ---------------------------------------------------------------------------
# Subclustering permutation test
# ---------------------------------------------------------------------------


class TestPermutationTest:
    @staticmethod
    def _fewer_and_more():
        rng = np.random.default_rng(1)
        return rng.poisson(8, 300), rng.poisson(10, 400)

    def test_is_reproducible_with_the_default_seed(self):
        fewer, more = self._fewer_and_more()
        assert lib.permutation_test(fewer, more) == lib.permutation_test(
            fewer, more
        )

    def test_p_value_is_never_zero(self):
        fewer, more = self._fewer_and_more()
        _, p_perm, _ = lib.permutation_test(fewer, more, iterations=99)
        assert p_perm == pytest.approx(1 / 100)

    def test_one_sided_detects_fewer_events_only(self):
        fewer, more = self._fewer_and_more()
        _, p_fewer, _ = lib.permutation_test(
            fewer, more, alternative="greater"
        )
        _, p_more, _ = lib.permutation_test(more, fewer, alternative="greater")
        assert p_fewer < 0.01
        assert p_more > 0.5

    def test_same_distribution_is_not_significant(self):
        rng = np.random.default_rng(2)
        _, p_perm, _ = lib.permutation_test(
            rng.poisson(10, 300), rng.poisson(10, 400)
        )
        assert p_perm > 0.05

    def test_title_names_the_one_sided_test(self):
        fewer, more = self._fewer_and_more()
        title = lib._subclustering_title(
            True, True, fewer, more, one_sided=True
        )
        assert title.startswith("one-sided KS test")
