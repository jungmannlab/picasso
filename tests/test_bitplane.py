"""Test reading Bitplane Imaris ``.ims`` movies (``picasso.ext.bitplane``
and ``picasso.io.load_ims``).

The reader is only defined when PyImarisWriter (Windows only) can be
imported, although reading needs nothing but ``h5py``. The tests load
``bitplane.py`` with stand-ins for PyImarisWriter and hdf5plugin so they
run on every platform, and read small ``.ims`` files written with
``h5py`` in the Imaris layout.

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


@pytest.fixture
def ims(monkeypatch):
    """``picasso.ext.bitplane`` loaded as if PyImarisWriter were
    installed."""
    writer = types.ModuleType("PyImarisWriter")
    writer.PyImarisWriter = types.SimpleNamespace(CallbackClass=object)
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
