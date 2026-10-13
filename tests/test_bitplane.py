"""Test reading Bitplane Imaris ``.ims`` movies (``picasso.ext.bitplane``
and ``picasso.io.load_ims``).

The reader is only defined when PyImarisWriter (Windows only) can be
imported, although reading needs nothing but ``h5py``. The tests load
``bitplane.py`` with stand-ins for PyImarisWriter and hdf5plugin so they
run on every platform, and read small ``.ims`` files written with
``h5py`` in the Imaris layout. The stand-in writer records the image
extents that the export hands to it.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib.util
import sys
import types

import h5py
import numpy as np
import pytest

from picasso import io
from picasso.ext import bitplane


class _ImageSize:
    def __init__(self, x=0, y=0, z=0, c=0, t=0):
        self.x, self.y, self.z, self.c, self.t = x, y, z, c, t

    def __truediv__(self, other):
        return _ImageSize(
            *(getattr(self, axis) // getattr(other, axis) for axis in "xyzct")
        )


class _Recorder:
    """Accepts any call and attribute; does nothing."""

    def __init__(self, *args, **kwargs):
        self.args = args

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _ImageConverter(_Recorder):
    finished = []

    def NeedCopyBlock(self, block_index):
        return False

    def Finish(self, image_extents, *args):
        self.finished.append(image_extents.args)


def _fake_writer() -> types.SimpleNamespace:
    """The parts of ``PyImarisWriter.PyImarisWriter`` that Picasso
    uses."""
    _ImageConverter.finished = []
    return types.SimpleNamespace(
        CallbackClass=object,
        ImageSize=_ImageSize,
        DimensionSequence=_Recorder,
        Options=_Recorder,
        eCompressionAlgorithmGzipLevel2=0,
        ImageConverter=_ImageConverter,
        ImageExtents=_Recorder,
        Parameters=_Recorder,
        ColorInfo=_Recorder,
    )


@pytest.fixture
def ims(monkeypatch):
    """``picasso.ext.bitplane`` loaded as if PyImarisWriter were
    installed."""
    writer = types.ModuleType("PyImarisWriter")
    writer.PyImarisWriter = _fake_writer()
    monkeypatch.setitem(sys.modules, "PyImarisWriter", writer)
    monkeypatch.setitem(
        sys.modules, "hdf5plugin", types.ModuleType("hdf5plugin")
    )
    spec = importlib.util.spec_from_file_location(
        "_bitplane_with_writer", bitplane.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.IMSWRITER
    return module


def _attr(value) -> np.ndarray:
    """Imaris stores attribute values as arrays of single characters."""
    return np.array(list(str(value)), dtype="S1")


def _write_ims(path, movies: dict[str, np.ndarray]) -> None:
    """Write ``movies`` (channel name -> ``(frames, y, x)``) as an
    ``.ims`` time series."""
    n_frames, height, width = next(iter(movies.values())).shape
    with h5py.File(path, "w") as f:
        level = f.create_group("DataSet/ResolutionLevel 0")
        for t in range(n_frames):
            for channel, movie in movies.items():
                level.create_dataset(
                    f"TimePoint {t}/{channel}/Data", data=movie[t][None]
                )
        image = f.create_group("DataSetInfo/Image")
        for key, value in {
            "X": width,
            "Y": height,
            "Z": 1,
            "ExtMin0": 0,
            "ExtMin1": 0,
            "ExtMin2": 0,
            "ExtMax0": width * 0.13,
            "ExtMax1": height * 0.13,
            "ExtMax2": 1,
        }.items():
            image.attrs[key] = _attr(value)


@pytest.fixture
def two_channels(tmp_path):
    rng = np.random.default_rng(0)
    movies = {
        "Channel 0": rng.integers(0, 1000, (3, 6, 8), dtype=np.uint16),
        "Channel 1": rng.integers(0, 1000, (3, 6, 8), dtype=np.uint16),
    }
    path = tmp_path / "movie.ims"
    _write_ims(path, movies)
    return str(path), movies


class TestIMSFile:
    def test_set_channel_selects_the_channel(self, ims, two_channels):
        path, movies = two_channels
        file = ims.IMSFile(path)
        assert file.channels == ["Channel 0", "Channel 1"]
        file.set_channel("Channel 1")
        file.read_movie()
        np.testing.assert_array_equal(
            np.array(list(file.movie)), movies["Channel 1"]
        )
        assert file.pixelsize == pytest.approx(130)

    @pytest.mark.parametrize("channel", ["Channel 2", None])
    def test_missing_channel_is_reported(self, ims, two_channels, channel):
        path, _ = two_channels
        file = ims.IMSFile(path)
        with pytest.raises(ValueError, match="Available channels: Channel 0"):
            file.set_channel(channel)
        # The previous channel stays selected.
        assert file.channel == "Channel 0"


class TestLoadIMS:
    @pytest.fixture(autouse=True)
    def reader(self, ims, monkeypatch):
        monkeypatch.setattr(io.bitplane, "IMSWRITER", True)
        monkeypatch.setattr(io, "IMSFile", ims.IMSFile, raising=False)

    def test_prompted_channel_is_loaded(self, two_channels):
        path, movies = two_channels
        movie, info = io.load_ims(path, prompt_info=lambda names: names[1])
        assert info[0]["Channel"] == "Channel 1"
        np.testing.assert_array_equal(movie[2], movies["Channel 1"][2])

    @pytest.mark.parametrize(
        "prompt_info", [None, lambda names: None], ids=["no prompt", "cancel"]
    )
    def test_first_channel_by_default(self, two_channels, prompt_info):
        path, movies = two_channels
        movie, info = io.load_ims(path, prompt_info=prompt_info)
        assert info[0]["Channel"] == "Channel 0"
        np.testing.assert_array_equal(movie[0], movies["Channel 0"][0])

    def test_unknown_prompted_channel_raises(self, two_channels):
        path, _ = two_channels
        with pytest.raises(ValueError, match="No channel 'GFP'"):
            io.load_ims(path, prompt_info=lambda names: "GFP")

    def test_single_channel_takes_the_files_name(self, tmp_path):
        movie = np.ones((2, 4, 4), dtype=np.uint16)
        path = tmp_path / "single.ims"
        _write_ims(path, {"Channel 3": movie})
        _, info = io.load_ims(str(path))
        assert info[0]["Channel"] == "Channel 3"


class TestImageExtents:
    def test_defaults_without_ims_metadata(self):
        info = [{"Frames": 10}, {"Generated by": "Picasso Localize"}]
        assert bitplane.image_extents(info) == bitplane.DEFAULT_EXTENTS

    def test_extents_of_load_ims_all(self):
        info = [
            {"ExtMin0": 10.0, "ExtMin1": 20.0},
            {"ExtMin2": 3.0, "ExtMax2": 5.0},
        ]
        assert bitplane.image_extents(info) == {
            "ExtMin0": 10.0,
            "ExtMin1": 20.0,
            "ExtMin2": 3.0,
            "ExtMax2": 5.0,
        }

    def test_global_extents_of_load_ims(self):
        info = [
            {
                "GlobalExtMin0": 10.0,
                "GlobalExtMin1": 20.0,
                "GlobalExtMin2": 3.0,
                "GlobalExtMax0": 50.0,
                "GlobalExtMax1": 60.0,
                "GlobalExtMax2": 5.0,
            }
        ]
        assert bitplane.image_extents(info) == {
            "ExtMin0": 10.0,
            "ExtMin1": 20.0,
            "ExtMin2": 3.0,
            "ExtMax2": 5.0,
        }

    def test_extmin_takes_precedence(self):
        # e.g. Render's export adds the resolved ExtMin fields.
        info = [{"GlobalExtMin0": 10.0}, {"ExtMin0": 12.0}]
        assert bitplane.image_extents(info)["ExtMin0"] == 12.0


class TestExportExtents:
    # 10 x 20 camera pixels from (x, y) = (4, 2), 100 nm pixels.
    VIEWPORT = ((2, 4), (12, 24))

    def test_offset_by_the_movie_corner_and_focal_plane(self):
        info = [
            {
                "GlobalExtMin0": 10.0,
                "GlobalExtMin1": 20.0,
                "GlobalExtMin2": 3.0,
                "GlobalExtMax2": 5.0,
            }
        ]
        extents = bitplane.export_extents(
            self.VIEWPORT, info, -200, 300, 100, 2, 5
        )
        np.testing.assert_allclose(extents, (10.4, 20.2, 3.8, 12.4, 21.2, 4.3))

    def test_2d_image_is_one_display_pixel_deep(self):
        extents = bitplane.export_extents(self.VIEWPORT, [{}], 0, 0, 100, 2, 1)
        # 50 nm display pixels around the focal plane at z = 0.
        np.testing.assert_allclose(
            extents, (0.4, 0.2, -0.025, 2.4, 1.2, 0.025)
        )


class TestNumpyToImaris:
    def test_movie_extents_reach_the_writer(self, ims, tmp_path):
        info = [
            {
                "GlobalExtMin0": 10.0,
                "GlobalExtMin1": 20.0,
                "GlobalExtMin2": 3.0,
                "GlobalExtMax2": 5.0,
            }
        ]
        image = np.zeros((2, 3, 40, 20), dtype=np.uint16)  # c, z, y, x
        ims.numpy_to_imaris(
            image,
            str(tmp_path / "export.ims"),
            ["red", "green"],
            2,
            ((2, 4), (12, 24)),
            info,
            -200,
            300,
            100,
        )
        (extents,) = ims.PW.ImageConverter.finished
        np.testing.assert_allclose(extents, (10.4, 20.2, 3.8, 12.4, 21.2, 4.3))
