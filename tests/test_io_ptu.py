"""Test the PicoQuant ``.ptu`` reader of ``picasso.io`` (``PTUMovie``).

The adapter logic (detector channels, frame summing without overlap and
as a sliding window, metadata, microtime histograms, refusals) runs
against a lightweight fake of the slice of ``ptufile``'s API the reader
uses, so it is tested without the optional library (which needs Python
>= 3.12). When ``ptufile`` is installed, the same is checked on small
scans written with ``ptufile.imwrite``, and a summed movie is localized.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import datetime
import types

import numpy as np
import pytest
import yaml

from picasso import io, localize

# ---------------------------------------------------------------------------
# Fake ptufile
# ---------------------------------------------------------------------------

TAGS = {
    "File_GUID": "{0000}",
    "File_CreatingTime": datetime.datetime(2026, 9, 7, 14, 25, 52),
    "HW_Type": "MultiHarp",
    "ImgHdr_PixResol": 0.13,  # micrometer
    "LaserWL": [530.0],
    "Detector": ["Detector 1 (SPAD)", "Detector 2 (SPAD)"],
    "LSDDevice": "<?xml version='1.0'?><Device/>",  # left out
}


def _binned(axis_values: np.ndarray, sel: slice, axis: int) -> np.ndarray:
    """Select ``sel`` along ``axis``, summing ``sel.step`` neighbors into
    one bin (a trailing incomplete bin included), as ptufile does."""
    start, stop, step = sel.indices(axis_values.shape[axis])
    chunk = np.take(axis_values, np.arange(start, stop), axis=axis)
    n = -(-(stop - start) // step)
    out = [
        np.take(
            chunk,
            np.arange(i * step, min((i + 1) * step, stop - start)),
            axis=axis,
        ).sum(axis=axis, keepdims=True)
        for i in range(n)
    ]
    return np.concatenate(out, axis=axis)


class _FakePtuFile:
    """Mimics ``ptufile.PtuFile`` for a ``(T, Y, X, C, H)`` histogram."""

    def __init__(self, data, *, image=True, tags=None):
        self._data = data.astype(np.int64)
        self.is_t3 = True
        self.is_image = image
        self.dims = ("T", "Y", "X", "C", "H")
        self.shape = data.shape
        self.tags = dict(TAGS if tags is None else tags)
        self.tcspc_resolution = 4e-11
        self.closed = False
        per_channel = self._data.sum(axis=(0, 1, 2, 4))
        self.active_channels = tuple(np.flatnonzero(per_channel).tolist())

    def decode_image(
        self,
        selection=None,
        *,
        channel=None,
        dtime=None,
        dtype=None,
        out=None,
    ):
        sel = list(selection) if selection is not None else [Ellipsis]
        if Ellipsis in sel:
            i = sel.index(Ellipsis)
            sel[i : i + 1] = [slice(None)] * (5 - len(sel) + 1)
        sel += [slice(None)] * (5 - len(sel))
        if channel is not None:
            sel[3] = channel
        data = self._data
        for axis in (0, 1, 2, 4):
            if axis == 4 and dtime == -1:
                data = data.sum(axis=4, keepdims=True)
                continue
            if isinstance(sel[axis], slice):
                if (
                    sel[axis].stop is not None
                    and sel[axis].stop > data.shape[axis]
                ):
                    raise IndexError("stop out of range")
                data = _binned(data, sel[axis], axis)
        if isinstance(sel[3], int):
            data = data[:, :, :, sel[3] : sel[3] + 1, :]
        return data.astype(dtype or np.uint16)

    def close(self):
        self.closed = True


def _install_fake_ptufile(monkeypatch, data, **kwargs):
    files = []

    def open_file(path, *args, **kw):
        files.append(_FakePtuFile(data, **kwargs))
        return files[-1]

    monkeypatch.setattr(
        io, "ptufile", types.SimpleNamespace(PtuFile=open_file)
    )
    return files


def _scan(t=10, y=6, x=8, channels=(0,), c=2, h=16, seed=0):
    """Random photon histograms with photons in ``channels`` only."""
    rng = np.random.default_rng(seed)
    data = np.zeros((t, y, x, c, h), np.uint16)
    for channel in channels:
        data[..., channel, :] = rng.poisson(0.3, (t, y, x, h))
    return data


def _spot_scan(t=8, size=24, h=8, seed=3):
    """Scan frames of one detector with a still spot at (x, y) = (12.6,
    11.3) pixels on a dim background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    spot = np.exp(-((yy - 11.3) ** 2 + (xx - 12.6) ** 2) / (2 * 1.2**2))
    rate = (0.02 + 6.0 * spot)[None, :, :, None] / h
    data = np.zeros((t, size, size, 1, h), np.uint16)
    data[..., 0, :] = rng.poisson(np.broadcast_to(rate, (t, size, size, h)))
    return data


def _frames(data, channel=0):
    """Scan frames of ``channel`` integrated over the microtime."""
    return data[..., channel, :].astype(np.int64).sum(-1)


# ---------------------------------------------------------------------------
# Reader logic (fake ptufile)
# ---------------------------------------------------------------------------


class TestPTUFrames:
    def test_scan_frames_without_summing(self, tmp_path, monkeypatch):
        data = _scan()
        _install_fake_ptufile(monkeypatch, data)
        movie, info = io.load_movie(str(tmp_path / "scan.ptu"))
        assert isinstance(movie, io.PTUMovie)
        assert isinstance(movie, io.AbstractPicassoMovie)
        assert movie.shape == (10, 6, 8)
        assert len(movie) == info[0]["Frames"] == 10
        assert movie.dtype == np.uint16
        np.testing.assert_array_equal(np.stack(list(movie)), _frames(data))
        np.testing.assert_array_equal(movie[-1], _frames(data)[-1])

    def test_sums_that_do_not_overlap(self, tmp_path, monkeypatch):
        data = _scan(t=11)
        _install_fake_ptufile(monkeypatch, data)
        movie, info = io.load_movie(
            str(tmp_path / "scan.ptu"), frame_sum_window=3
        )
        # 11 scan frames make 3 sums of 3; the 2 left over are dropped
        assert len(movie) == 3
        assert movie.dtype == np.uint32
        expected = _frames(data)[:9].reshape(3, 3, 6, 8).sum(1)
        np.testing.assert_array_equal(movie[0:3], expected)
        assert info[0]["Frame Sum Window"] == 3
        assert info[0]["Frame Sum Stride"] == 3
        assert info[0]["Scan Frames"] == 11

    @pytest.mark.parametrize("window, stride", [(3, 1), (4, 2)])
    def test_overlapping_sums(self, tmp_path, monkeypatch, window, stride):
        data = _scan(t=11)
        _install_fake_ptufile(monkeypatch, data)
        movie, info = io.load_movie(
            str(tmp_path / "scan.ptu"),
            frame_sum_window=window,
            frame_sum_stride=stride,
        )
        frames = _frames(data)
        n = (11 - window) // stride + 1
        assert len(movie) == n == info[0]["Frames"]
        for k in range(n):
            start = k * stride
            np.testing.assert_array_equal(
                movie[k], frames[start : start + window].sum(0)
            )
        with pytest.raises(IndexError):
            movie[n]

    def test_frames_are_read_only(self, tmp_path, monkeypatch):
        _install_fake_ptufile(monkeypatch, _scan())
        movie, _ = io.load_movie(str(tmp_path / "scan.ptu"))
        with pytest.raises(ValueError):
            movie[0][0, 0] = 1

    def test_fewer_scan_frames_than_the_window(self, tmp_path, monkeypatch):
        _install_fake_ptufile(monkeypatch, _scan(t=4))
        with pytest.raises(ValueError, match="fewer than the 5"):
            io.load_movie(str(tmp_path / "scan.ptu"), frame_sum_window=5)

    @pytest.mark.parametrize(
        "window, stride",
        [(0, None), (2, 3), (2.0, None), (True, None), (3, 0)],
    )
    def test_invalid_frame_sums(self, tmp_path, monkeypatch, window, stride):
        _install_fake_ptufile(monkeypatch, _scan())
        with pytest.raises(ValueError, match="frame_sum"):
            io.load_movie(
                str(tmp_path / "scan.ptu"),
                frame_sum_window=window,
                frame_sum_stride=stride,
            )


class TestPTUChannels:
    def test_each_detector_with_photons_is_a_channel(
        self, tmp_path, monkeypatch
    ):
        data = _scan(channels=(1, 3), c=4)
        _install_fake_ptufile(monkeypatch, data)
        asked = []

        def prompt(channels):
            asked.append(channels)
            return "Detector 4"

        movie, info = io.load_movie(str(tmp_path / "scan.ptu"), prompt)
        assert asked == [["Detector 2", "Detector 4"]]
        assert info[0]["Channel"] == "Detector 4"
        np.testing.assert_array_equal(movie[2], _frames(data, 3)[2])

    def test_first_detector_without_a_prompt(self, tmp_path, monkeypatch):
        data = _scan(channels=(1, 3), c=4)
        _install_fake_ptufile(monkeypatch, data)
        movie, info = io.load_movie(str(tmp_path / "scan.ptu"))
        assert info[0]["Channel"] == "Detector 2"
        np.testing.assert_array_equal(movie[0], _frames(data, 1)[0])

    def test_load_movie_all_gives_one_movie_per_detector(
        self, tmp_path, monkeypatch
    ):
        data = _scan(channels=(0, 1))
        _install_fake_ptufile(monkeypatch, data)
        movies, infos = io.load_movie_all(
            str(tmp_path / "scan.ptu"), frame_sum_window=2
        )
        assert [i[0]["Channel"] for i in infos] == ["Detector 1", "Detector 2"]
        for channel, movie in enumerate(movies):
            np.testing.assert_array_equal(
                movie[1], _frames(data, channel)[2:4].sum(0)
            )

    def test_files_are_closed(self, tmp_path, monkeypatch):
        files = _install_fake_ptufile(monkeypatch, _scan())
        movie, _ = io.load_movie(str(tmp_path / "scan.ptu"))
        movie.close()
        assert all(f.closed for f in files)


class TestPTUMetadata:
    def test_photon_counting_pixel_size_and_header(
        self, tmp_path, monkeypatch
    ):
        _install_fake_ptufile(monkeypatch, _scan())
        _, info = io.load_movie(str(tmp_path / "scan.ptu"))
        info = info[0]
        assert info["Photon Counting"] is True
        assert info["Pixelsize"] == pytest.approx(130.0)
        assert info["Data Type"] == "uint16"
        header = info["PTU Metadata"]
        assert header["HW_Type"] == "MultiHarp"
        assert header["File_CreatingTime"] == "2026-09-07 14:25:52"
        assert "LSDDevice" not in header

    def test_metadata_round_trips_through_yaml(self, tmp_path, monkeypatch):
        _install_fake_ptufile(monkeypatch, _scan())
        _, info = io.load_movie(str(tmp_path / "scan.ptu"), frame_sum_window=2)
        path = tmp_path / "info.yaml"
        io.save_info(str(path), info)
        assert io.load_info(str(path)) == info
        # plain YAML: no Python-specific tags
        assert "!!python" not in path.read_text()

    def test_not_an_image_scan(self, tmp_path, monkeypatch):
        _install_fake_ptufile(monkeypatch, _scan(), image=False)
        with pytest.raises(ValueError, match="not a PicoQuant T3 image"):
            io.load_movie(str(tmp_path / "scan.ptu"))

    def test_no_photons(self, tmp_path, monkeypatch):
        _install_fake_ptufile(monkeypatch, _scan(channels=()))
        with pytest.raises(ValueError, match="no photons"):
            io.load_movie(str(tmp_path / "scan.ptu"))


class TestPTUMicrotimes:
    @pytest.mark.parametrize("window, stride", [(1, 1), (3, 3), (3, 1)])
    def test_histograms_sum_to_the_frames(
        self, tmp_path, monkeypatch, window, stride
    ):
        _install_fake_ptufile(monkeypatch, _scan(t=11))
        movie, _ = io.load_movie(
            str(tmp_path / "scan.ptu"),
            frame_sum_window=window,
            frame_sum_stride=stride,
        )
        hist = movie._microtime_histograms(1, len(movie))
        assert hist.shape == (len(movie) - 1, 6, 8, 16)
        assert hist.dtype == np.uint32
        np.testing.assert_array_equal(hist.sum(-1), movie[1 : len(movie)])

    def test_region_binning_and_gate(self, tmp_path, monkeypatch):
        data = _scan(t=9)
        _install_fake_ptufile(monkeypatch, data)
        movie, _ = io.load_movie(
            str(tmp_path / "scan.ptu"), frame_sum_window=3
        )
        hist = movie._microtime_histograms(
            1, 3, roi=((1, 2), (4, 7)), dtime_bin=4, dtime_range=(2, 14)
        )
        assert hist.shape == (2, 3, 5, 3)
        scan = data[3:9, 1:4, 2:7, 0, 2:14].astype(np.int64)
        expected = scan.reshape(2, 3, 3, 5, 3, 4).sum(axis=(1, 5))
        np.testing.assert_array_equal(hist, expected)
        assert movie._number_bins == 16
        assert movie._tcspc_resolution == pytest.approx(4e-11)

    def test_out_of_range(self, tmp_path, monkeypatch):
        _install_fake_ptufile(monkeypatch, _scan())
        movie, _ = io.load_movie(str(tmp_path / "scan.ptu"))
        with pytest.raises(IndexError):
            movie._microtime_histograms(5, 11)
        with pytest.raises(ValueError):
            movie._microtime_histograms(0, 2, dtime_range=(8, 8))
        with pytest.raises(ValueError):
            movie._microtime_histograms(0, 2, dtime_bin=0)


class TestPTUAvailability:
    def test_extension_advertised_when_library_present(self):
        assert (".ptu" in io.MOVIE_EXTENSIONS) == (io.ptufile is not None)

    def test_without_library_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io, "ptufile", None)
        with pytest.raises(ImportError, match=r"picassosr\[ptu\]"):
            io.load_movie(str(tmp_path / "scan.ptu"))
        with pytest.raises(ImportError, match=r"picassosr\[ptu\]"):
            io.load_movie_all(str(tmp_path / "scan.ptu"))

    @pytest.mark.parametrize("loader", [io.load_movie, io.load_movie_all])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"frame_sum_window": 2},
            {"frame_sum_stride": 2},
        ],
    )
    def test_camera_movies_cannot_be_summed(self, tmp_path, loader, kwargs):
        # refused before the file is even looked for
        with pytest.raises(ValueError, match="only available for photon"):
            loader(str(tmp_path / "movie.tif"), **kwargs)


class TestPTULocalize:
    def test_summed_movie_is_localized(self, tmp_path, monkeypatch):
        """A uint32 summed movie runs through identification and the fit
        with photon-counting camera parameters."""
        _install_fake_ptufile(monkeypatch, _spot_scan())
        movie, info = io.load_movie(
            str(tmp_path / "scan.ptu"), frame_sum_window=4
        )
        assert movie.dtype == np.uint32
        camera_info = {
            "Baseline": 0,
            "Sensitivity": 1,
            "Gain": 1,
            "Pixelsize": info[0]["Pixelsize"],
        }
        locs, _ = localize.localize(
            movie,
            camera_info=camera_info,
            identification_parameters=localize.IdentificationParameters(
                method=localize.NetGradient(20), box=7
            ),
            movie_info=info,
            fitting_method="gausslq",
            parallel=False,
        )
        assert len(locs) == 2
        np.testing.assert_allclose(locs["x"] / 130.0, 12.6, atol=0.3)
        np.testing.assert_allclose(locs["y"] / 130.0, 11.3, atol=0.3)


# ---------------------------------------------------------------------------
# Real ptufile (Python >= 3.12)
# ---------------------------------------------------------------------------


def _written_scan(tmp_path, t=7, y=12, x=16, h=64):
    """A scan with photons in detectors 1 and 3 written by ptufile."""
    ptufile = pytest.importorskip("ptufile")
    rng = np.random.default_rng(1)
    data = np.zeros((t, y, x, 4, h), np.uint8)
    data[..., 1, :] = rng.poisson(0.05, (t, y, x, h))
    data[..., 3, :] = rng.poisson(0.02, (t, y, x, h))
    data[..., 1, h - 1] = 1  # every microtime bin of detector 1 occupied
    path = str(tmp_path / "scan.ptu")
    ptufile.imwrite(path, data, 5e-8, 5e-8 / h, None, pixel_resolution=0.13)
    return path, data.astype(np.int64)


class TestPTUFile:
    def test_frames_and_channels(self, tmp_path):
        path, data = _written_scan(tmp_path)
        movies, infos = io.load_movie_all(path)
        assert [i[0]["Channel"] for i in infos] == ["Detector 2", "Detector 4"]
        for movie, channel in zip(movies, (1, 3)):
            np.testing.assert_array_equal(
                np.stack(list(movie)), data[..., channel, :].sum(-1)
            )
            movie.close()
        assert infos[0][0]["Pixelsize"] == pytest.approx(130.0)
        assert infos[0][0]["Scan Frames"] == 7

    @pytest.mark.parametrize("window, stride", [(3, None), (3, 1), (4, 2)])
    def test_frame_sums(self, tmp_path, window, stride):
        path, data = _written_scan(tmp_path)
        movie, _ = io.load_movie(
            path, frame_sum_window=window, frame_sum_stride=stride
        )
        stride = stride or window
        frames = data[..., 1, :].sum(-1)
        n = (7 - window) // stride + 1
        assert len(movie) == n
        for k in range(n):
            np.testing.assert_array_equal(
                movie[k], frames[k * stride : k * stride + window].sum(0)
            )
        movie.close()

    def test_microtime_histograms(self, tmp_path):
        path, data = _written_scan(tmp_path)
        movie, _ = io.load_movie(path, frame_sum_window=2)
        hist = movie._microtime_histograms(
            0, 3, roi=((2, 3), (10, 12)), dtime_bin=8
        )
        expected = (
            data[:6, 2:10, 3:12, 1, :]
            .reshape(3, 2, 8, 9, 8, 8)
            .sum(axis=(1, 5))
        )
        np.testing.assert_array_equal(hist, expected)
        assert movie._tcspc_resolution == pytest.approx(5e-8 / 64)
        movie.close()


def test_ptu_metadata_is_safe_yaml(tmp_path, monkeypatch):
    """``PTU Metadata`` holds builtins only, so the safe loader reads it."""
    _install_fake_ptufile(monkeypatch, _scan())
    _, info = io.load_movie(str(tmp_path / "scan.ptu"))
    text = yaml.safe_dump(info[0])
    assert yaml.safe_load(text) == info[0]
