"""Test the less traveled branches of ``picasso.io``: error paths users
can hit (truncated or corrupt files, missing metadata, wrong shapes),
multi-file movies, index edge cases, metadata parsing and the import and
export of other software's formats.

Complements ``tests/test_io.py``, whose writers and fake readers
(``_write_tif_stack``, ``_write_stk``, ``_FakeND2File``, the CZI and LIF
fakes) are reused here. Everything is written to ``tmp_path``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import logging
import os
import sys
import types

import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile

from picasso import io, lib

from tests.conftest import PIXELSIZE
from tests.test_io import (
    ND2_CONFIG,
    _FakeCziFile,
    _FakeCziImage,
    _FakeLifImage,
    _FakeND2File,
    _install_fake_czifile,
    _install_fake_liffile,
    _install_fake_nd2,
    _write_pyme_file,
    _write_single_frame,
    _write_stk,
    _write_tif_stack,
)


@pytest.fixture(autouse=True)
def _yaml_sidecar_on(monkeypatch):
    """Write the sidecar .yaml whatever the user settings say, as in
    ``tests/test_io.py``."""
    monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: True)


def _movie(n=4, h=6, w=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60000, size=(n, h, w), dtype="<u2")


# ---------------------------------------------------------------------------
# Windows long paths, user folders
# ---------------------------------------------------------------------------


class _FakeRegistryKey:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_winreg(value=None, error=None):
    def open_key(root, name):
        if error is not None:
            raise error
        return _FakeRegistryKey()

    return types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=object(),
        OpenKey=open_key,
        QueryValueEx=lambda key, name: (value, 4),
    )


class TestWindowsLongPathsSetting:
    """``_windows_long_paths_enabled`` reads the registry value
    ``LongPathsEnabled``; ``winreg`` is faked so this runs anywhere."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        io._windows_long_paths_enabled.cache_clear()
        yield
        io._windows_long_paths_enabled.cache_clear()

    @pytest.mark.parametrize("value, enabled", [(1, True), (0, False)])
    def test_reads_the_registry_value(self, value, enabled, monkeypatch):
        monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(value))
        assert io._windows_long_paths_enabled() is enabled

    def test_an_unreadable_registry_means_disabled(self, monkeypatch):
        fake = _fake_winreg(error=OSError("no such key"))
        monkeypatch.setitem(sys.modules, "winreg", fake)
        assert io._windows_long_paths_enabled() is False

    def test_a_long_path_passes_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(1))
        monkeypatch.setattr(io.os, "name", "nt")
        path = os.path.join(str(tmp_path), *["folder"] * 50, "locs.hdf5")
        assert len(path) > io._WINDOWS_MAX_PATH
        io._check_path_length(path)


def test_spill_directory_is_created_in_the_home_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    directory = io.spill_directory()
    assert directory == str(tmp_path / ".picasso" / "spill")
    assert os.path.isdir(directory)


# ---------------------------------------------------------------------------
# SpilledFrame failures
# ---------------------------------------------------------------------------


class TestSpillFailures:
    """A table that cannot be moved to disk raises ``SpillError`` and
    leaves nothing behind; one that cannot be read back is reported."""

    @staticmethod
    def _frame():
        return pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)})

    def test_a_short_write_is_detected(self, tmp_path, monkeypatch):
        save = np.save
        monkeypatch.setattr(
            io.np,
            "save",
            lambda file, array, **kwargs: save(file, array[:-1], **kwargs),
        )
        with pytest.raises(io.SpillError, match="was not written"):
            io.SpilledFrame(self._frame(), directory=str(tmp_path))
        assert os.listdir(tmp_path) == []

    def test_a_failed_write_is_a_spill_error(self, tmp_path, monkeypatch):
        def save(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(io.np, "save", save)
        with pytest.raises(io.SpillError, match="disk full") as error:
            io.SpilledFrame(self._frame(), directory=str(tmp_path))
        assert isinstance(error.value.__cause__, OSError)
        assert os.listdir(tmp_path) == []

    def test_a_shortened_column_is_reported_as_incomplete(self, tmp_path):
        spilled = io.SpilledFrame(self._frame(), directory=str(tmp_path))
        np.save(spilled._file(1), np.arange(3.0))
        with pytest.raises(io.SpillError, match="incomplete"):
            spilled.load()
        spilled.discard()


# ---------------------------------------------------------------------------
# Raw movies
# ---------------------------------------------------------------------------


class TestLoadRawFallbacks:
    @staticmethod
    def _info(byte_order="<"):
        return {
            "Byte Order": byte_order,
            "Data Type": "uint16",
            "Frames": 3,
            "Height": 4,
            "Width": 5,
        }

    def test_missing_metadata_is_asked_for_and_saved(self, tmp_path):
        movie = _movie(3, 4, 5)
        path = tmp_path / "movie.raw"
        movie.tofile(path)
        loaded, info = io.load_raw(
            str(path), prompt_info=lambda: (self._info(), True)
        )
        np.testing.assert_array_equal(loaded, movie)
        assert info[0]["Frames"] == 3
        # saved, so the next load needs no prompt
        loaded, _ = io.load_raw(str(path))
        np.testing.assert_array_equal(loaded, movie)

    def test_metadata_entered_without_saving(self, tmp_path):
        path = tmp_path / "movie.raw"
        _movie(3, 4, 5).tofile(path)
        io.load_raw(str(path), prompt_info=lambda: (self._info(), False))
        assert not (tmp_path / "movie.yaml").exists()

    def test_canceling_the_prompt_returns_none(self, tmp_path):
        path = tmp_path / "movie.raw"
        _movie(3, 4, 5).tofile(path)
        assert io.load_raw(str(path), prompt_info=lambda: None) is None

    def test_without_a_prompt_missing_metadata_raises(self, tmp_path):
        path = tmp_path / "movie.raw"
        _movie(3, 4, 5).tofile(path)
        with pytest.raises(FileNotFoundError):
            io.load_raw(str(path))

    def test_big_endian_frames_are_read_little_endian(self, tmp_path):
        movie = _movie(3, 4, 5)
        path = tmp_path / "movie.raw"
        movie.astype(">u2").tofile(path)
        io.save_info(str(tmp_path / "movie.yaml"), [self._info(">")])
        loaded, info = io.load_raw(str(path))
        np.testing.assert_array_equal(loaded, movie)
        assert info[0]["Byte Order"] == "<"


# ---------------------------------------------------------------------------
# Calibrations
# ---------------------------------------------------------------------------


class TestCalibrationFilesThatAreNotCalibrations:
    def test_a_multi_document_yaml_is_rejected(self, tmp_path):
        path = tmp_path / "calib.yaml"
        path.write_text("a: 1\n---\nb: 2\n")
        with pytest.raises(ValueError, match="single-document"):
            io.load_calibration(str(path))
        with pytest.raises(ValueError, match="single-document"):
            io.load_any_calibration(str(path))

    def test_a_3d_calibration_needs_x_coefficients(self, tmp_path):
        path = tmp_path / "calib.yaml"
        path.write_text("Y Coefficients: [1, 2]\n")
        with pytest.raises(ValueError, match="X Coefficients"):
            io.load_calibration(str(path))

    def test_any_calibration_must_be_a_mapping(self, tmp_path):
        path = tmp_path / "calib.yaml"
        path.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            io.load_any_calibration(str(path))

    def test_spline_metadata_sets_are_saved_as_lists(self, tmp_path):
        path = str(tmp_path / "spline_calib.hdf5")
        io.save_any_calibration(
            path, {"coefficients": np.zeros((2, 64)), "Tags": {"a"}}
        )
        assert io.load_any_calibration(path)["Tags"] == ["a"]


# ---------------------------------------------------------------------------
# Movie metadata fallback and loader dispatch
# ---------------------------------------------------------------------------


class _UnreadableMovie:
    """A movie whose frame count and metadata cannot be read."""

    height = 4
    width = 5

    def __len__(self):
        raise OSError("frames unknown")

    def info(self):
        raise ValueError("metadata broken")


class TestMetadataFallbackOfUnreadableMovies:
    def test_the_prompt_gets_the_dimensions_that_can_be_read(self, tmp_path):
        asked = {}

        def prompt(dims):
            asked.update(dims)
            return None

        path = str(tmp_path / "movie.tif")
        result = io._movie_info_or_prompt(_UnreadableMovie(), path, prompt)
        assert result is None
        assert asked == {"Height": 4, "Width": 5}

    def test_metadata_entered_by_hand_can_be_saved(self, tmp_path):
        path = str(tmp_path / "movie.tif")
        entered = {"Frames": 3, "Height": 4, "Width": 5}
        info = io._movie_info_or_prompt(
            _UnreadableMovie(), path, lambda dims: (entered, True)
        )
        assert info == entered
        assert io.load_info(path) == [entered]

    def test_without_a_prompt_the_error_names_its_cause(self, tmp_path):
        path = str(tmp_path / "movie.tif")
        with pytest.raises(io.NoMetadataFileError) as error:
            io._movie_info_or_prompt(_UnreadableMovie(), path, None)
        assert isinstance(error.value.__cause__, ValueError)

    def test_canceling_the_prompt_of_an_stk_returns_none(
        self, tmp_path, monkeypatch
    ):
        path = _write_stk(tmp_path / "m.stk", _movie())
        monkeypatch.setattr(io.STKMultiMovie, "info", lambda self: None)
        assert io.load_stk(path, prompt_info=lambda dims: None) is None

    def test_canceling_the_prompt_of_an_nd2_channel_returns_none(
        self, tmp_path, monkeypatch
    ):
        _install_fake_nd2(
            monkeypatch, np.zeros((2, 2, 4, 4), "<u2"), ("T", "C", "Y", "X")
        )
        monkeypatch.setattr(io.ND2Movie, "info", lambda self: None)
        path = str(tmp_path / "m.nd2")
        assert io.load_nd2_all(path, prompt_info=lambda dims: None) is None

    def test_canceling_the_prompt_of_a_concatenated_movie(
        self, tmp_path, monkeypatch
    ):
        paths = [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]
        for path in paths:
            _write_tif_stack(path, _movie())
        monkeypatch.setattr(io.ConcatenatedTiffMovie, "info", lambda s: {})
        assert io.load_tif_concatenated(paths, lambda dims: None) is None

    def test_load_movie_all_returns_none_when_canceled(
        self, tmp_path, monkeypatch
    ):
        path = str(tmp_path / "m.tif")
        _write_tif_stack(path, _movie())
        monkeypatch.setattr(io.TiffMultiMap, "info", lambda self: {})
        assert io.load_movie_all(path, prompt_info=lambda d: None) is None


class TestLoaderDispatch:
    def test_load_movie_opens_nd2(self, tmp_path, monkeypatch):
        _install_fake_nd2(monkeypatch, _movie(), ("T", "Y", "X"))
        movie, info = io.load_movie(str(tmp_path / "m.nd2"))
        assert isinstance(movie, io.ND2Movie)
        assert info[0]["Frames"] == 4

    def test_load_movie_opens_ims(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            io, "load_ims", lambda path, prompt_info=None: ("ims", path)
        )
        path = str(tmp_path / "m.ims")
        assert io.load_movie(path) == ("ims", path)

    def test_load_movie_all_opens_every_channel(self, tmp_path, monkeypatch):
        data = np.zeros((3, 2, 4, 4), "<u2")
        _install_fake_czifile(monkeypatch, data, ("T", "C", "Y", "X"))
        _install_fake_liffile(monkeypatch, [_FakeLifImage(data)])
        _install_fake_nd2(monkeypatch, data, ("T", "C", "Y", "X"))
        for ext in (".czi", ".lif", ".nd2"):
            movies, infos = io.load_movie_all(str(tmp_path / f"m{ext}"))
            assert len(movies) == len(infos) == 2
            for movie in movies:
                movie.close()

    def test_load_movie_all_opens_ims(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io, "load_ims_all", lambda path: ([1], [[{}]]))
        assert io.load_movie_all(str(tmp_path / "m.ims")) == ([1], [[{}]])


# ---------------------------------------------------------------------------
# YAML values with Python tags (files written before Picasso 1.0)
# ---------------------------------------------------------------------------


class TestYamlPythonTags:
    """Python-tagged values that Picasso does not rebuild are read as
    their plain data, and the tags are reported."""

    TEXT = "\n".join(
        [
            "scalar: !!python/object/apply:numpy.core.multiarray.scalar x",
            "objects: !!python/object/apply:numpy.core.multiarray."
            "_reconstruct",
            "  state: [1, [2], !!python/object/apply:numpy.dtype {args: [O]},"
            " false, [a, b]]",
            "short: !!python/object/apply:numpy._core.multiarray._reconstruct",
            "  state: [1]",
            "ordered: !!python/object/new:collections.OrderedDict",
            "  dictitems: {k: 1}",
            "items: !!python/object/new:mymodule.MyList",
            "  listitems: [1, 2]",
            "state: !!python/object/new:mymodule.Thing",
            "  state: {a: 1}",
            "text: !!python/object:mymodule.Text plain",
            "pair: !!python/object/new:mymodule.Pair {args: [1, 2]}",
            "",
        ]
    )

    def test_values_are_read_as_plain_data(self, tmp_path, caplog):
        path = tmp_path / "old.yaml"
        path.write_text(self.TEXT)
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            data = io.load_yaml(str(path))
        assert data == {
            "scalar": [],
            "objects": ["a", "b"],
            "short": [1],
            "ordered": {"k": 1},
            "items": [1, 2],
            "state": {"a": 1},
            "text": "plain",
            "pair": [1, 2],
        }
        assert "mymodule.MyList" in caplog.text
        # dict subclasses Picasso wrote itself are not reported
        assert "OrderedDict" not in caplog.text


# ---------------------------------------------------------------------------
# Picks, masks
# ---------------------------------------------------------------------------


class TestPicksFiles:
    @pytest.mark.parametrize(
        "shape, picks, size",
        [
            ("Rectangle", [[(0.0, 0.0), (100.0, 50.0)]], 20.0),
            ("Polygon", [[(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]], None),
            ("Square", [(30.0, 40.0), (50.0, 60.0)], 15.0),
        ],
    )
    def test_shapes_round_trip(self, shape, picks, size, tmp_path):
        path = str(tmp_path / "picks.yaml")
        io.save_picks(path, picks, shape, size)
        loaded, loaded_shape, loaded_size = io.load_picks(path)
        assert loaded_shape == shape
        assert loaded_size == size
        assert np.allclose(np.array(loaded, dtype=float), picks)

    def test_a_file_that_is_not_a_mapping_is_rejected(self, tmp_path):
        path = tmp_path / "picks.yaml"
        path.write_text("- 1\n- 2\n")
        with pytest.raises(ValueError, match="Unrecognized picks file"):
            io.load_picks(str(path))


@pytest.mark.parametrize(
    "info, message",
    [({"Width": 4}, "Incorrect file"), ({"Generated by": "Other"}, "SPINNA")],
)
def test_a_mask_not_made_by_spinna_is_rejected(info, message, tmp_path):
    path = tmp_path / "mask.npy"
    np.save(path, np.ones((4, 4)))
    io.save_info(str(tmp_path / "mask.yaml"), [info])
    with pytest.raises(TypeError, match=message):
        io.load_mask(str(path))


# ---------------------------------------------------------------------------
# User settings that cannot be backed up or kept
# ---------------------------------------------------------------------------


class TestUserSettingsCopiesFailing:
    """The conftest fixture ``isolated_user_settings`` points the
    settings file into ``tmp_path``."""

    def test_a_broken_file_that_cannot_be_kept_is_still_reported(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "settings.yaml").write_text("a: [unclosed\n")

        def copy2(*args, **kwargs):
            raise OSError("read-only")

        monkeypatch.setattr(io.shutil, "copy2", copy2)
        assert io.load_user_settings() == {}
        message, kept = io.settings_load_error()
        assert message
        assert kept is None

    def test_settings_are_saved_when_the_backup_fails(
        self, tmp_path, monkeypatch, caplog
    ):
        io.save_user_settings({"a": 1})

        def copy2(*args, **kwargs):
            raise OSError("read-only")

        monkeypatch.setattr(io.shutil, "copy2", copy2)
        with caplog.at_level(logging.WARNING):
            io.save_user_settings({"a": 2})
        assert "Could not back up" in caplog.text
        assert io.load_user_settings() == {"a": 2}

    def test_a_missing_file_has_no_signature(self, tmp_path):
        assert io._file_signature(str(tmp_path / "missing.yaml")) is None


def test_embedded_metadata_that_is_not_json_is_ignored(tmp_path):
    path = str(tmp_path / "locs.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("metadata", data="{not json")
    with pytest.warns(UserWarning, match="could not be read"):
        assert io._load_info_from_hdf5(path) is None


# ---------------------------------------------------------------------------
# ND2
# ---------------------------------------------------------------------------


class _ND2FileWithoutChannels(_FakeND2File):
    """An .nd2 file whose channel metadata cannot be read."""

    def __init__(self, data, dims):
        super().__init__(data, dims)
        self.metadata = types.SimpleNamespace(contents=None)


class TestND2Metadata:
    def test_missing_text_info_leaves_the_camera_unknown(
        self, tmp_path, monkeypatch
    ):
        _install_fake_nd2(monkeypatch, _movie(), ("T", "Y", "X"), text_info={})
        _, info = io.load_nd2(str(tmp_path / "m.nd2"))
        assert info[0]["Camera"] == "None"
        nd2_meta = info[0]["nd2 Metadata"]
        assert not {"capturing", "description", "optics"} & set(nd2_meta)

    def test_unreadable_channel_names_are_numbered(
        self, tmp_path, monkeypatch
    ):
        data = np.zeros((2, 2, 4, 4), "<u2")
        monkeypatch.setattr(
            io.nd2,
            "ND2File",
            lambda path: _ND2FileWithoutChannels(data, ("T", "C", "Y", "X")),
        )
        movie = io.ND2Movie(str(tmp_path / "m.nd2"), channel=1)
        assert movie.channels == ["Channel 0", "Channel 1"]
        # the metadata cannot be read either, so it is asked for
        assert movie.info() is None

    def test_nested_entries_create_missing_levels(self):
        out = {}
        io.ND2Movie.set_nested_dict_entry(out, ["a", "b", "c"], 1)
        assert out == {"a": {"b": {"c": 1}}}


class TestND2CameraParameters:
    @pytest.fixture
    def movie(self, tmp_path, monkeypatch):
        _install_fake_nd2(monkeypatch, _movie(), ("T", "Y", "X"))
        movie, _ = io.load_nd2(str(tmp_path / "m.nd2"))
        return movie

    def test_a_configuration_without_cameras_raises(self, movie):
        with pytest.raises(KeyError, match="'camera' key not found"):
            movie.camera_parameters({})

    def test_without_camera_settings_the_gain_is_one(self, movie, caplog):
        del movie.meta["Picasso Metadata"]
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            parameters = movie.camera_parameters(ND2_CONFIG)
        assert parameters["gain"] == [1]
        assert "gain 1 is used" in caplog.text

    def test_a_gain_property_is_not_supported(self, movie):
        config = {"Cameras": {"Zyla 4.2": {"Gain Property Name": "Gain"}}}
        with pytest.raises(NotImplementedError):
            movie.camera_parameters(config)

    def test_without_filter_wavelengths_the_wavelength_is_zero(self, movie):
        parameters = movie.camera_parameters({"Cameras": {"Zyla 4.2": {}}})
        assert parameters["wavelength"] == [0]
        assert parameters["gain"] == [1]

    def test_unreadable_metadata_raise_a_key_error(
        self, tmp_path, monkeypatch
    ):
        """The metadata of the movie could not be read (and were entered
        by hand), so its camera is unknown."""
        _install_fake_nd2(monkeypatch, _movie(), ("T", "Y", "X"))
        monkeypatch.setattr(
            io.ND2Movie,
            "get_metadata",
            lambda self, nd2file: (_ for _ in ()).throw(KeyError("broken")),
        )
        movie = io.ND2Movie(str(tmp_path / "m.nd2"))
        with pytest.raises(KeyError, match="'camera' key not found"):
            movie.camera_parameters(ND2_CONFIG)


# ---------------------------------------------------------------------------
# CZI and LIF adapters
# ---------------------------------------------------------------------------


class _RaisingProperty:
    """Descriptor raising on access, for metadata a library fails on."""

    def __get__(self, obj, owner=None):
        raise RuntimeError("unreadable")


class _CziImageWithoutMetadata(_FakeCziImage):
    """A CZI image whose channel names and metadata cannot be read."""

    channels = _RaisingProperty()
    attrs = _RaisingProperty()

    def __init__(self, data, dims):
        super().__init__(data, dims, mpp=None)
        del self.attrs  # back to the class attribute, which raises


class _FailingClose:
    def close(self):
        raise OSError("already gone")


class TestCZIMovieEdgeCases:
    def test_a_file_without_scenes_reads_all_subblocks(
        self, tmp_path, monkeypatch
    ):
        data = _movie(3, 4, 4)[:, None]
        image = _FakeCziImage(data, ("T", "C", "Y", "X"))
        czi = _FakeCziFile(image)
        czi.scenes = {}
        monkeypatch.setattr(
            io,
            "czifile",
            types.SimpleNamespace(
                CziFile=lambda path: czi,
                CziImage=lambda parent, entries: image,
            ),
        )
        with io.CZIMovie(str(tmp_path / "m.czi")) as movie:
            np.testing.assert_array_equal(movie[-1], data[-1, 0])
            np.testing.assert_array_equal(movie[0:2], data[0:2, 0])
        assert czi.closed

    def test_a_file_without_image_axes_is_rejected(
        self, tmp_path, monkeypatch
    ):
        _install_fake_czifile(monkeypatch, np.zeros((2, 2)), ("T", "C"))
        with pytest.raises(ValueError, match="no Y/X image axes"):
            io.CZIMovie(str(tmp_path / "m.czi"))

    def test_unreadable_metadata_is_left_out(
        self, tmp_path, monkeypatch, caplog
    ):
        data = np.zeros((2, 2, 4, 4), "<u2")
        image = _CziImageWithoutMetadata(data, ("T", "C", "Y", "X"))
        monkeypatch.setattr(
            io,
            "czifile",
            types.SimpleNamespace(CziFile=lambda path: _FakeCziFile(image)),
        )
        movie = io.CZIMovie(str(tmp_path / "m.czi"))
        assert movie.channels == ["Channel 0", "Channel 1"]
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            info = movie.info()
        assert "CZI Metadata" not in info
        assert "Pixelsize" not in info
        assert "could not be read" in caplog.text
        movie._czi = _FailingClose()
        movie.close()  # a failing close is ignored

    def test_an_unknown_channel_choice_selects_the_first(
        self, tmp_path, monkeypatch
    ):
        data = _movie(2, 4, 4).reshape(1, 2, 4, 4)
        _install_fake_czifile(monkeypatch, data, ("T", "C", "Y", "X"))
        movie, info = io.load_czi(
            str(tmp_path / "m.czi"), prompt_info=lambda channels: "other"
        )
        assert info[0]["Channel"] == "Channel 0"
        np.testing.assert_array_equal(movie[0], data[0, 0])

    def test_neutral_camera_parameters_and_no_raw_export(
        self, tmp_path, monkeypatch
    ):
        _install_fake_czifile(
            monkeypatch, np.zeros((2, 1, 4, 4), "<u2"), ("T", "C", "Y", "X")
        )
        movie = io.CZIMovie(str(tmp_path / "m.czi"))
        assert movie.camera_parameters({})["gain"] == [1]
        with pytest.raises(NotImplementedError, match="CZIMovie"):
            movie.tofile(None)


class _LifImageWithoutMetadata(_FakeLifImage):
    """A LIF image whose coordinates and metadata cannot be read."""

    coords = _RaisingProperty()
    attrs = _RaisingProperty()

    def __init__(self, data):
        super().__init__(data)
        del self.coords, self.attrs  # back to the raising class attributes


class TestLIFMovieEdgeCases:
    def test_a_file_without_images_is_rejected(self, tmp_path, monkeypatch):
        _install_fake_liffile(monkeypatch, [])
        with pytest.raises(ValueError, match="contains no images"):
            io.LIFMovie(str(tmp_path / "m.lif"))

    def test_an_image_without_image_axes_is_rejected(
        self, tmp_path, monkeypatch
    ):
        image = _FakeLifImage(np.zeros((2, 1, 4, 4), "<u2"))
        image.sizes = {"T": 2, "C": 1, "Y": 4}
        _install_fake_liffile(monkeypatch, [image])
        with pytest.raises(ValueError, match="no Y/X"):
            io.LIFMovie(str(tmp_path / "m.lif"))

    def test_unreadable_metadata_is_left_out(
        self, tmp_path, monkeypatch, caplog
    ):
        data = np.zeros((2, 2, 4, 4), "<u2")
        _install_fake_liffile(monkeypatch, [_LifImageWithoutMetadata(data)])
        movie = io.LIFMovie(str(tmp_path / "m.lif"))
        assert movie.channels == ["Channel 0", "Channel 1"]
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            info = movie.info()
        assert "LIF Metadata" not in info
        assert "Pixelsize" not in info
        movie._lif = _FailingClose()
        movie.close()  # a failing close is ignored

    def test_other_axes_are_read_at_their_first_index(
        self, tmp_path, monkeypatch
    ):
        image = _FakeLifImage(_movie(3, 4, 4)[:, None])
        image.sizes = {"T": 3, "C": 1, "Z": 2, "Y": 4, "X": 4}
        read = []
        frame = image.frame

        def record(**indices):
            read.append(indices)
            return frame(**indices)

        image.frame = record
        _install_fake_liffile(monkeypatch, [image])
        movie = io.LIFMovie(str(tmp_path / "m.lif"))
        movie[1]
        assert read == [{"T": 1, "C": 0, "Z": 0}]

    def test_rgb_frames_are_rejected(self, tmp_path, monkeypatch):
        image = _FakeLifImage(np.zeros((2, 1, 4, 4), "<u2"))
        image._data = np.zeros((2, 1, 4, 4, 3), "<u2")
        _install_fake_liffile(monkeypatch, [image])
        movie = io.LIFMovie(str(tmp_path / "m.lif"))
        with pytest.raises(ValueError, match="RGB frames are not supported"):
            movie[0]


def test_metadata_that_json_cannot_hold_is_kept_as_text():
    circular = []
    circular.append(circular)
    assert io._yaml_safe(circular) == "[[...]]"


# ---------------------------------------------------------------------------
# TIFF: per-thread file handles, page indexing, layouts
# ---------------------------------------------------------------------------


class _BrokenHandle:
    """A file handle that fails to seek and to close, like one of a
    network share that dropped its session."""

    def seek(self, offset):
        raise OSError(22, "Invalid argument")

    def close(self):
        raise OSError("cannot close")


class TestPerThreadFileHandles:
    @pytest.fixture
    def tif(self, tmp_path):
        data = _movie(3, 8, 8)
        path = tmp_path / "m.tif"
        _write_tif_stack(path, data)
        movie = io.TiffMap(str(path))
        yield movie, data, path
        movie.close()

    def test_a_closed_handle_is_replaced(self, tif):
        movie, data, _ = tif
        movie[0]
        stale = movie._local.file
        stale.close()
        np.testing.assert_array_equal(movie[1], data[1])
        assert movie._local.file is not stale
        assert stale not in movie._open_handles

    def test_a_handle_that_fails_to_close_is_dropped(self, tif):
        movie, data, _ = tif
        movie._local.file = _BrokenHandle()  # not in _open_handles
        np.testing.assert_array_equal(movie[2], data[2])
        assert not isinstance(movie._local.file, _BrokenHandle)

    def test_a_read_failing_twice_raises(self, tif, monkeypatch):
        movie, _, _ = tif
        monkeypatch.setattr(movie, "_handle", lambda: _BrokenHandle())
        with pytest.raises(OSError, match="Invalid argument"):
            movie[0]

    def test_a_truncated_file_is_reported(self, tif):
        movie, _, path = tif
        with open(path, "r+b") as f:
            f.truncate(movie._offsets[2] + 10)
        with pytest.raises(OSError, match="Truncated read"):
            movie[2]


class _Pages(tifffile.TiffPages):
    """A chain of ``n`` IFDs that has not been walked yet."""

    def __init__(self, n):
        self.n = n

    def __getitem__(self, index):
        if index >= self.n:
            raise IndexError(index)

    def __len__(self):
        return self.n


class _Recorder(lib.MockProgress):
    def __init__(self):
        super().__init__()
        self.values = []

    def set_value(self, value, *args, **kwargs):
        self.values.append(value)


class TestPageIndexing:
    def test_a_long_chain_is_reported_in_chunks(self):
        progress = _Recorder()
        assert io._index_pages(_Pages(4500), progress) == 4500
        assert progress.values == [2000, 4000]

    def test_a_list_of_pages_is_counted(self):
        assert io._index_pages([1, 2, 3], _Recorder()) == 3

    def test_counters_of_nested_multi_file_movies(self):
        assert io._join_counters("2/3", 0, 2) == "2/3, part 1/2"


class TestTiffLayouts:
    def test_compressed_stack_with_a_stray_last_page(self, tmp_path):
        data = _movie(3, 8, 8)
        path = tmp_path / "m.tif"
        with tifffile.TiffWriter(path) as tw:
            for frame in data:
                tw.write(frame, photometric="minisblack", compression="zlib")
            tw.write(
                np.zeros((4, 4), "<u2"),
                photometric="minisblack",
                compression="zlib",
            )
        with io.TiffMap(str(path)) as movie:
            assert movie.n_frames == 3
            np.testing.assert_array_equal(movie[...], data)

    def test_multi_strip_pages_are_decoded(self, tmp_path):
        data = _movie(3, 8, 8)
        path = tmp_path / "m.tif"
        with tifffile.TiffWriter(path) as tw:
            for frame in data:
                tw.write(frame, photometric="minisblack", rowsperstrip=2)
        with io.TiffMap(str(path)) as movie:
            assert movie._offsets is None
            np.testing.assert_array_equal(movie[...], data)

    def test_an_imagej_stack_with_a_page_per_frame(self, tmp_path):
        data = _movie(3, 8, 8)
        path = tmp_path / "m.tif"
        tifffile.imwrite(path, data, imagej=True)
        with io.TiffMap(str(path)) as movie:
            assert movie._imagej_planes is None
            np.testing.assert_array_equal(movie[...], data)

    def test_a_multi_strip_imagej_image_is_one_frame(self, tmp_path):
        path = tmp_path / "m.tif"
        tifffile.imwrite(path, _movie(1, 8, 8)[0], imagej=True, rowsperstrip=2)
        with io.TiffMap(str(path)) as movie:
            assert movie._imagej_planes is None
            assert movie.n_frames == 1

    def test_verbose_prints_the_path(self, tmp_path, capsys):
        path = tmp_path / "m.tif"
        _write_tif_stack(path, _movie())
        io.TiffMap(str(path), verbose=True).close()
        assert str(path) in capsys.readouterr().out

    def test_acquisition_comments_are_kept(self, tmp_path, monkeypatch):
        path = tmp_path / "m.tif"
        _write_tif_stack(path, _movie())
        comments = {"Micro-Manager Acquisition Comments": ["note"]}
        monkeypatch.setattr(
            io, "_mm_metadata_from_tifffile", lambda tif: comments
        )
        with io.TiffMap(str(path)) as movie:
            assert movie.info()["Micro-Manager Acquisition Comments"] == [
                "note"
            ]

    def test_unreadable_acquisition_comments_are_left_out(self, caplog):
        class Tif:
            pages = []
            micromanager_metadata = _RaisingProperty()

        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            assert io._mm_metadata_from_tifffile(Tif()) == {}
        assert "acquisition comments" in caplog.text


# ---------------------------------------------------------------------------
# Array-like access of every movie class
# ---------------------------------------------------------------------------


def _tiff_map(tmp_path, data):
    path = tmp_path / "m.tif"
    _write_tif_stack(path, data)
    return io.TiffMap(str(path))


def _tiff_multi_map(tmp_path, data):
    _write_tif_stack(tmp_path / "m.ome.tif", data[:2])
    _write_tif_stack(tmp_path / "m_1.ome.tif", data[2:])
    return io.TiffMultiMap(str(tmp_path / "m.ome.tif"))


def _stk_movie(tmp_path, data):
    return io.STKMovie(_write_stk(tmp_path / "m.stk", data))


def _stk_multi_movie(tmp_path, data):
    _write_stk(tmp_path / "m_001.stk", data[:2])
    _write_stk(tmp_path / "m_002.stk", data[2:])
    return io.STKMultiMovie(str(tmp_path / "m_001.stk"))


@pytest.fixture(
    params=[_tiff_map, _tiff_multi_map, _stk_movie, _stk_multi_movie],
    ids=["TiffMap", "TiffMultiMap", "STKMovie", "STKMultiMovie"],
)
def any_movie(request, tmp_path):
    data = _movie(5, 6, 7, seed=3)
    with request.param(tmp_path, data) as movie:
        yield movie, data


class TestArrayLikeAccess:
    def test_frames_slices_and_regions(self, any_movie):
        movie, data = any_movie
        assert len(movie) == 5
        np.testing.assert_array_equal(movie[np.int64(1)], data[1])
        np.testing.assert_array_equal(movie[1:4], data[1:4])
        np.testing.assert_array_equal(movie[::2], data[::2])
        np.testing.assert_array_equal(movie[...], data)
        np.testing.assert_array_equal(np.array(list(movie)), data)
        np.testing.assert_array_equal(movie[2, 1:3], data[2, 1:3])
        np.testing.assert_array_equal(movie[1:4, 2], data[1:4, 2])
        np.testing.assert_array_equal(movie[1:4, 2, 3:5], data[1:4, 2, 3:5])
        np.testing.assert_array_equal(movie[..., 3], data[..., 3])

    def test_an_empty_slice_is_an_empty_stack(self, any_movie):
        movie, _ = any_movie
        assert len(movie[3:3, 1]) == 0

    def test_too_many_indices_raise(self, any_movie):
        movie, _ = any_movie
        with pytest.raises(IndexError):
            movie[1:3, 1, 1, 1]

    @pytest.mark.parametrize("index", [1.5, np.float32(1.0)], ids=repr)
    def test_a_non_integer_index_raises(self, index, any_movie):
        movie, _ = any_movie
        with pytest.raises(TypeError):
            movie[index]

    def test_the_dtype_is_little_endian_uint16(self, any_movie):
        movie, _ = any_movie
        assert movie.dtype == np.dtype("<u2")


@pytest.mark.parametrize(
    "make",
    [_tiff_multi_map, _stk_movie, _stk_multi_movie],
    ids=["TiffMultiMap", "STKMovie", "STKMultiMovie"],
)
def test_camera_parameters_without_a_camera_are_neutral(make, tmp_path):
    with make(tmp_path, _movie(5, 6, 7)) as movie:
        parameters = movie.camera_parameters({})
    assert parameters["gain"] == [1]
    assert parameters["wavelength"] == [0]


# ---------------------------------------------------------------------------
# Multi-file TIFF movies
# ---------------------------------------------------------------------------


class TestTiffMultiMapFiles:
    def test_ndtiff_stacks_are_joined(self, tmp_path):
        data = _movie(6)
        _write_tif_stack(tmp_path / "acq_NDTiffStack.tif", data[:4])
        _write_tif_stack(tmp_path / "acq_NDTiffStack_1.tif", data[4:])
        movie, info = io.load_tif(str(tmp_path / "acq_NDTiffStack.tif"))
        with movie:
            assert info[0]["Frames"] == 6
            np.testing.assert_array_equal(movie[...], data)

    def test_a_failing_close_does_not_hide_the_open_error(self):
        class Component:
            def close(self):
                raise OSError("close failed")

        def open_one(index, item, progress):
            if index == 1:
                raise ValueError("second file unreadable")
            return Component()

        with pytest.raises(ValueError, match="second file"):
            io._open_components(open_one, ["a", "b"], None)


MM_CONFIG = {
    "Cameras": {
        "Andor": {
            "Gain Property Name": "Gain",
            "EM Switch Property": {"Name": "EMSwitch", True: "On"},
            "Sensitivity Categories": ["ReadoutMode"],
            "Channel Device": {
                "Name": "Filter",
                "Emission Wavelengths": {"Cy3": 570},
            },
        },
        "Zyla": {},
    }
}


class TestTiffMultiMapCameraParameters:
    @pytest.fixture
    def movie(self, tmp_path):
        mm = {
            "Camera": "Andor",
            "Andor-Gain": "300",
            "Andor-EMSwitch": "On",
            "Andor-ReadoutMode": "17 MHz",
            "Filter": "Cy3",
        }
        path = tmp_path / "m.tif"
        _write_tif_stack(path, _movie(), mm_metadata=mm)
        movie = io.TiffMultiMap(str(path))
        yield movie
        movie.close()

    def test_parameters_from_the_micromanager_metadata(self, movie):
        movie.info()
        parameters = movie.camera_parameters(MM_CONFIG)
        assert parameters["camera"] == "Andor"
        assert parameters["cam_index"] == 0
        assert parameters["gain"] == 300
        assert parameters["Sensitivity"] == {"ReadoutMode": "17 MHz"}
        assert parameters["wavelength"] == ["570"]

    def test_the_metadata_is_read_if_info_was_not_called(self, movie):
        assert movie.camera_parameters(MM_CONFIG)["camera"] == "Andor"

    def test_without_em_gain_or_a_known_filter(self, movie):
        movie.info()
        mm_info = movie.meta["Micro-Manager Metadata"]
        mm_info["Andor-EMSwitch"] = "Off"
        mm_info["Filter"] = "GFP"
        parameters = movie.camera_parameters(MM_CONFIG)
        assert parameters["gain"] == [1]
        assert parameters["wavelength"] == [0]

    def test_a_camera_missing_from_the_configuration(self, movie, caplog):
        config = {"Cameras": {"Zyla": {}}}
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            parameters = movie.camera_parameters(config)
        assert parameters == {"gain": [1], "wavelength": [0], "cam_index": 0}
        assert "'Andor'" in caplog.text

    def test_without_micromanager_metadata_the_gain_is_one(
        self, movie, caplog
    ):
        movie.info()
        del movie.meta["Micro-Manager Metadata"]
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            parameters = movie.camera_parameters(MM_CONFIG)
        assert parameters == {"gain": [1], "wavelength": [0], "cam_index": 0}
        assert "gain 1 is used" in caplog.text


class TestMMSeparateFilesEdgeCases:
    def _write(self, folder, data):
        names = []
        for t, frame in enumerate(data):
            name = f"img_channel000_position000_time{t:09d}_z000.tif"
            _write_single_frame(folder / name, frame)
            names.append(str(folder / name))
        return names

    def test_folders_next_to_the_frames_are_ignored(self, tmp_path):
        paths = self._write(tmp_path, _movie(3))
        # named like a frame, but a folder
        name = "img_channel000_position000_time000000009_z000.tif"
        (tmp_path / name).mkdir()
        assert io._mm_separate_files(paths[0]) == paths

    def test_no_files_are_rejected(self):
        with pytest.raises(ValueError, match="No TIFF files"):
            io.MMSeparateTiffMovie([])

    def test_a_frame_out_of_range_raises(self, tmp_path):
        paths = self._write(tmp_path, _movie(3))
        with io.MMSeparateTiffMovie(paths) as movie:
            with pytest.raises(IndexError, match="out of range"):
                movie[3]

    def test_tofile_writes_every_frame(self, tmp_path):
        data = _movie(3)
        paths = self._write(tmp_path, data)
        out = tmp_path / "m.raw"
        with io.MMSeparateTiffMovie(paths) as movie, open(out, "wb") as f:
            movie.tofile(f, byte_order=">")
        written = np.fromfile(out, dtype=">u2").reshape(data.shape)
        np.testing.assert_array_equal(written, data)

    def test_a_folder_that_cannot_be_listed_has_no_acquisition(self, tmp_path):
        assert io.find_mm_separate_first(str(tmp_path / "missing")) is None


def test_find_tif_movies_sorts_with_a_custom_key(tmp_path):
    for name in ("a.tif", "b.tif", "c.tif"):
        _write_tif_stack(tmp_path / name, _movie(1))
    (tmp_path / "notes.txt").write_text("not a movie")
    found = io.find_tif_movies(str(tmp_path), key=lambda p: -ord(p[-5]))
    assert [os.path.basename(_) for _ in found] == ["c.tif", "b.tif", "a.tif"]


# ---------------------------------------------------------------------------
# Localization files: format versions, dark times, truncation
# ---------------------------------------------------------------------------


class TestLocsFileChecks:
    def _write_locs(self, path, attrs):
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "locs", data=np.zeros(1, dtype=[("frame", "u4"), ("x", "f4")])
            )
            for key, value in attrs.items():
                f.attrs[key] = value

    def test_a_version_stored_as_bytes(self, tmp_path):
        path = str(tmp_path / "locs.hdf5")
        self._write_locs(path, {io.FORMAT_VERSION_ATTR: np.bytes_(b"1.0")})
        assert io.get_format_version(path) == (1, 0)

    def test_a_newer_format_names_the_picasso_that_wrote_it(self, tmp_path):
        path = str(tmp_path / "locs.hdf5")
        self._write_locs(
            path,
            {
                io.FORMAT_VERSION_ATTR: "99.0",
                io.PICASSO_VERSION_ATTR: np.bytes_(b"9.9.0"),
            },
        )
        with pytest.raises(io.FileFormatError, match="Picasso 9.9.0"):
            io.load_locs(path)

    def test_dark_times_of_an_unknown_version_are_flagged(self, caplog):
        """A file written before Picasso 1.0 whose dark times name no
        version."""
        locs = pd.DataFrame({"dark": [1.0]})
        info = [{"Step": "Dark", "Generated by": "Picasso Dark"}, "a note"]
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            io._check_dark_times(locs, info, None, "locs.hdf5")
        assert "dark times" in caplog.text

    def test_a_file_as_long_as_recorded_is_not_truncated(self, tmp_path):
        path = tmp_path / "locs.hdf5"
        path.write_bytes(b"x" * 100)
        error = OSError("truncated file: eof = 100, stored_eof = 50")
        io._raise_if_truncated(str(path), error)  # does not raise

    def test_reading_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            io._read_locs_dataset(str(tmp_path / "x.hdf5"), lib.MockProgress())


# ---------------------------------------------------------------------------
# ThunderSTORM
# ---------------------------------------------------------------------------


class TestThunderstormFiles:
    def _locs_3d(self):
        return pd.DataFrame(
            {
                "frame": np.array([2, 3], dtype=np.uint32),
                "x": np.array([650.0, 1300.0], dtype=np.float32),
                "y": np.array([780.0, 1430.0], dtype=np.float32),
                "z": np.array([-50.0, 80.0], dtype=np.float32),
                "sx": np.array([140.0, 150.0], dtype=np.float32),
                "sy": np.array([160.0, 170.0], dtype=np.float32),
                "photons": np.array([1000.0, 2000.0], dtype=np.float32),
                "bg": np.array([10.0, 20.0], dtype=np.float32),
                "lpx": np.array([6.0, 8.0], dtype=np.float32),
                "lpy": np.array([8.0, 10.0], dtype=np.float32),
                "len": np.array([3, 4], dtype=np.int32),
            }
        )

    def test_3d_locs_with_lengths_round_trip(self, tmp_path):
        locs = self._locs_3d()
        path = tmp_path / "ts.csv"
        io.export_thunderstorm(str(path), locs, [{}])
        columns = pd.read_csv(path).columns
        assert "z [nm]" in columns and "detections" in columns

        out, info = io.import_ts(str(path), pixelsize=PIXELSIZE)
        np.testing.assert_allclose(out["z"], locs["z"])
        np.testing.assert_allclose(out["sx"], locs["sx"])
        np.testing.assert_allclose(out["sy"], locs["sy"])
        np.testing.assert_array_equal(out["frame"], [0, 1])
        np.testing.assert_array_equal(out["detections"], [3, 4])
        assert info[0]["Frames"] == 2

    @pytest.mark.parametrize("z", [True, False], ids=["3D", "2D"])
    def test_missing_columns_are_named(self, z, tmp_path):
        columns = {"frame": [1], "x [nm]": [1.0], "y [nm]": [1.0]}
        if z:
            columns["z [nm]"] = [0.0]
        path = tmp_path / "ts.csv"
        pd.DataFrame(columns).to_csv(path, index=False)
        dimension = "3D" if z else "2D"
        with pytest.raises(ValueError, match=f"for {dimension} Thunder"):
            io.import_ts(str(path), pixelsize=PIXELSIZE)

    def test_extra_column_names_become_identifiers(self, tmp_path):
        data = pd.DataFrame(
            {
                "frame": [1, 2],
                "x [nm]": [650.0, 1300.0],
                "y [nm]": [650.0, 1300.0],
                "intensity [photon]": [1000.0, 2000.0],
                "offset [photon]": [10.0, 10.0],
                "uncertainty_xy [nm]": [6.5, 6.5],
                "sigma [nm]": [130.0, 130.0],
                "1st pass": [1.0, 2.0],
                "%": [3, 4],
                "x": [5.0, 6.0],  # taken by the converted x [nm]
                "label": ["a", "b"],  # not numeric
            }
        )
        path = tmp_path / "ts.csv"
        data.to_csv(path, index=False)
        out, _ = io.import_ts(str(path), pixelsize=PIXELSIZE)
        np.testing.assert_array_equal(out["_1st_pass"], [1.0, 2.0])
        np.testing.assert_array_equal(out["column"], [3, 4])
        np.testing.assert_allclose(out["x"], [650.0, 1300.0])
        assert "label" not in out.columns

    def test_a_canvas_of_no_finite_coordinates_is_one_pixel(self):
        assert io._canvas_pixels([np.nan, np.inf], PIXELSIZE) == 1


# ---------------------------------------------------------------------------
# SMAP
# ---------------------------------------------------------------------------


def _smap_loc(n=3, **fields):
    loc = {
        "xnm": np.linspace(100.0, 300.0, n),
        "ynm": np.linspace(200.0, 400.0, n),
        "frame": np.arange(1.0, n + 1),
    }
    loc.update(fields)
    return loc


class TestSMAPFiles:
    def _write_mat(self, path, variables):
        from scipy.io import savemat

        savemat(str(path), variables)
        return str(path)

    def _write_v73(self, path, loc, where=("saveloc", "loc")):
        with h5py.File(path, "w") as f:
            group = f
            for name in where:
                group = group.create_group(name)
            for field, values in loc.items():
                group.create_dataset(field, data=np.atleast_2d(values))
        return str(path)

    def test_v73_localizations_at_the_root(self, tmp_path):
        path = self._write_v73(tmp_path / "a_sml.mat", _smap_loc(), ["loc"])
        with h5py.File(path, "a") as f:
            f["loc"].create_group("nested")
            f["loc"].create_dataset("comment", data=np.array([b"text"]))
        locs, _ = io.import_smap(path, pixelsize=PIXELSIZE)
        assert len(locs) == 3
        assert "nested" not in locs.columns
        assert "comment" not in locs.columns

    def test_v73_split_parts_are_rejected(self, tmp_path):
        path = self._write_v73(tmp_path / "a_sml.mat", _smap_loc())
        with h5py.File(path, "a") as f:
            f.create_dataset("partnames", data=np.zeros(1))
        with pytest.raises(ValueError, match="split"):
            io.import_smap(path, pixelsize=PIXELSIZE)

    def test_v73_without_localizations_is_rejected(self, tmp_path):
        path = str(tmp_path / "a_sml.mat")
        with h5py.File(path, "w") as f:
            f.create_dataset("other", data=np.zeros(1))
        with pytest.raises(ValueError, match="Could not find SMAP"):
            io.import_smap(path, pixelsize=PIXELSIZE)

    def test_localizations_at_the_root(self, tmp_path):
        loc = _smap_loc(label=np.array(["a", "b", "c"]))
        path = self._write_mat(tmp_path / "a_sml.mat", {"loc": loc})
        locs, info = io.import_smap(path, pixelsize=PIXELSIZE)
        np.testing.assert_allclose(locs["x"], loc["xnm"])
        assert "label" not in locs.columns
        assert info[0]["Frames"] == 3

    def test_a_file_without_localizations_is_rejected(self, tmp_path):
        path = self._write_mat(tmp_path / "a_sml.mat", {"other": 1.0})
        with pytest.raises(ValueError, match="Could not find SMAP"):
            io.import_smap(path, pixelsize=PIXELSIZE)

    def test_missing_fields_are_named(self, tmp_path):
        loc = {"xnm": np.zeros(3), "ynm": np.zeros(3)}
        path = self._write_mat(tmp_path / "a_sml.mat", {"loc": loc})
        with pytest.raises(ValueError, match="frame"):
            io.import_smap(path, pixelsize=PIXELSIZE)

    def test_one_psf_width_and_separate_precisions(self, tmp_path):
        loc = _smap_loc(
            PSFxnm=np.array([120.0, 130.0, 140.0]),
            locprecxnm=np.array([5.0, 6.0, 7.0]),
            locprecynm=np.array([8.0, 9.0, 10.0]),
            x=np.array([1.0, 2.0, 3.0]),  # taken by the converted xnm
        )
        path = self._write_mat(tmp_path / "a_sml.mat", {"loc": loc})
        locs, _ = io.import_smap(path, pixelsize=PIXELSIZE)
        np.testing.assert_allclose(locs["sy"], loc["PSFxnm"])
        np.testing.assert_allclose(locs["lpx"], loc["locprecxnm"])
        np.testing.assert_allclose(locs["lpy"], loc["locprecynm"])
        np.testing.assert_allclose(locs["x"], loc["xnm"])

    def test_without_precisions_they_are_zero(self, tmp_path):
        path = self._write_mat(tmp_path / "a_sml.mat", {"loc": _smap_loc()})
        locs, _ = io.import_smap(path, pixelsize=PIXELSIZE)
        np.testing.assert_array_equal(locs["lpx"], 0)
        np.testing.assert_array_equal(locs["lpy"], 0)
        np.testing.assert_array_equal(locs["sx"], PIXELSIZE)

    def test_export_without_background_or_canvas(self, tmp_path):
        from scipy.io import loadmat

        locs = pd.DataFrame(
            {
                "frame": np.array([0, 1], dtype=np.uint32),
                "x": np.array([650.0, 1300.0], dtype=np.float32),
                "y": np.array([300.0, 900.0], dtype=np.float32),
                "photons": np.array([100.0, 200.0], dtype=np.float32),
                "lpx": np.array([5.0, 5.0], dtype=np.float32),
                "lpy": np.array([5.0, 5.0], dtype=np.float32),
            }
        )
        path = tmp_path / "a_sml.mat"
        io.export_smap(str(path), locs, [{"Pixelsize": PIXELSIZE}])
        mat = loadmat(str(path), struct_as_record=False, squeeze_me=True)
        np.testing.assert_array_equal(mat["saveloc"].loc.bg, [0, 0])
        assert mat["saveloc"].info.Width == 1300 // PIXELSIZE + 1
        assert mat["saveloc"].info.Height == 900 // PIXELSIZE + 1


# ---------------------------------------------------------------------------
# PyME
# ---------------------------------------------------------------------------


def _pyme_records(**columns):
    n = len(next(iter(columns.values())))
    dtype = [(name, np.asarray(v).dtype) for name, v in columns.items()]
    records = np.zeros(n, dtype=dtype)
    for name, values in columns.items():
        records[name] = values
    return records


class TestPyMEMetadataValues:
    def test_values_are_made_json_serializable(self):
        assert io._pyme_metadata_value(b"text") == "text"
        assert io._pyme_metadata_value(np.arange(2)) == [0, 1]
        assert io._pyme_metadata_value(object()) is None

    def test_photons_per_count_need_a_valid_gain(self):
        metadata = {"Camera.ElectronsPerCount": 2.0, "Camera.TrueEMGain": 0}
        assert io._pyme_photons_per_count(metadata) is None

    @pytest.mark.parametrize("raw", ["{not json", '{"a": 1}', "[1]"])
    def test_picasso_metadata_must_be_a_list_of_infos(self, raw):
        assert io._pyme_picasso_infos({"Picasso.Info": raw}) is None

    def test_the_pixel_size_of_a_non_hdf5_file_is_unknown(self, tmp_path):
        path = tmp_path / "a.hdf"
        path.write_text("not hdf5")
        assert io.pyme_pixelsize(str(path)) is None


class TestPyMEImport:
    def test_a_file_without_metadata_or_frames(self, tmp_path):
        import tables

        path = str(tmp_path / "a.hdf")
        records = _pyme_records(x=[100.0, 200.0], y=[300.0, 400.0])
        with tables.open_file(path, "w") as h5file:
            h5file.create_table("/", "Localizations", obj=records)
        locs, info = io.import_pyme(path, pixelsize=PIXELSIZE)
        np.testing.assert_array_equal(locs["frame"], [0, 0])
        assert info[0]["Frames"] == 1
        assert info[0]["PyME metadata"] == {}

    def test_negative_frames_start_at_zero(self, tmp_path):
        path = tmp_path / "a.hdf"
        records = _pyme_records(
            x=[100.0, 200.0], y=[300.0, 400.0], t=[-2.0, 3.0], frame=[7, 8]
        )  # "frame" is an extra column, not kept: the frames come from t
        _write_pyme_file(path, "Localizations", records)
        locs, _ = io.import_pyme(str(path), pixelsize=PIXELSIZE)
        np.testing.assert_array_equal(locs["frame"], [0, 5])


class TestPyMEChannels:
    def _locs(self, probes):
        n = len(probes)
        return pd.DataFrame(
            {
                "frame": np.zeros(n, dtype=np.uint32),
                "x": np.arange(n, dtype=np.float32),
                "y": np.arange(n, dtype=np.float32),
                "probe": np.array(probes, dtype=np.int32),
            }
        )

    def test_locs_without_probes_are_one_channel(self):
        locs = self._locs([0, 0]).drop(columns="probe")
        assert io.split_pyme_channels("a.hdf", locs, [{}]) == [
            ("a.hdf", locs, [{}])
        ]

    def test_without_picasso_metadata_each_channel_gets_a_copy(self, tmp_path):
        path = str(tmp_path / "missing.hdf")  # not readable
        info = [{"Pixelsize": PIXELSIZE}]
        channels = io.split_pyme_channels(path, self._locs([0, 2, 2]), info)
        assert [os.path.basename(p) for p, _, _ in channels] == [
            "missing_probe0.hdf",
            "missing_probe2.hdf",
        ]
        assert [len(locs) for _, locs, _ in channels] == [1, 2]
        assert [i[-1]["PyME probe"] for _, _, i in channels] == [0, 2]
        assert "PyME probe" not in info[-1]

    def test_picasso_metadata_of_other_channels_is_not_used(self, tmp_path):
        """The file stores the metadata of probes 0 and 1, the
        localizations have probes 0 and 2."""
        locs = self._locs([0, 1])
        infos = [
            [{"Pixelsize": PIXELSIZE, "Width": 8, "Height": 8, "Frames": 1}]
        ] * 2
        path = str(tmp_path / "a.hdf")
        io.export_pyme_multi(path, [locs, locs], infos)
        channels = io.split_pyme_channels(
            path, self._locs([0, 2]), [{"Pixelsize": PIXELSIZE}]
        )
        assert [i[-1]["PyME probe"] for _, _, i in channels] == [0, 2]


class TestPyMEExport:
    def _locs(self):
        return pd.DataFrame(
            {
                "frame": np.array([0, 1], dtype=np.uint32),
                "x": np.array([100.0, 200.0], dtype=np.float32),
                "y": np.array([100.0, 200.0], dtype=np.float32),
                "lpx": np.array([5.0, 5.0], dtype=np.float32),
                "lpy": np.array([5.0, 5.0], dtype=np.float32),
                "label": ["a", "b"],
            }
        )

    def test_text_columns_are_left_out(self, tmp_path):
        path = str(tmp_path / "a.hdf")
        io.export_pyme(path, self._locs(), [{"Pixelsize": PIXELSIZE}])
        locs, _ = io.import_pyme(path)
        assert "label" not in locs.columns

    def test_no_channels_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="each channel"):
            io.export_pyme_multi(str(tmp_path / "a.hdf"), [], [])

    def test_metadata_that_json_cannot_hold_is_left_out(self, tmp_path):
        path = str(tmp_path / "a.hdf")
        info = [{"Pixelsize": PIXELSIZE, "Object": object()}]
        with pytest.warns(UserWarning, match="Could not store"):
            io.export_pyme(path, self._locs(), info)
        assert len(io.import_pyme(path)[0]) == 2
