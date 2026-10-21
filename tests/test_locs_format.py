"""Test the localization file format: its version, the checks of
``picasso.io.load_locs``, the precedence and safe reading of the .yaml
metadata file and the metadata entries of processing steps
(``picasso.lib.step_info``).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from picasso import io, lib, __version__

INFO = [{"Width": 64, "Height": 64, "Frames": 100, "Pixelsize": 130}]


@pytest.fixture(autouse=True)
def _yaml_sidecar_off(monkeypatch):
    """No .yaml sidecar unless a test asks for one, whatever the user
    settings of the machine running the tests say."""
    monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: False)


def make_locs(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Localizations inside the field of view of ``INFO``."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 100, n).astype(np.uint32),
            "x": rng.uniform(0, 64, n).astype(np.float32),
            "y": rng.uniform(0, 64, n).astype(np.float32),
            "lpx": np.full(n, 0.1, np.float32),
            "lpy": np.full(n, 0.1, np.float32),
        }
    )


def write_raw(path, locs: pd.DataFrame, info=INFO, attrs=None) -> str:
    """Write a localization file the way other software (or Picasso
    before 1.0) does: no format version unless ``attrs`` give one.
    Without one, the file counts as written in camera pixels and is
    converted to nm when loaded (the ``make_locs`` coordinates fit the
    field of view either way)."""
    with h5py.File(path, "w") as f:
        f.create_dataset("locs", data=locs.to_records(index=False))
        f.create_dataset("metadata", data=json.dumps(info))
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
    return str(path)


class TestFormatVersion:
    def test_save_locs_records_the_versions(self, tmp_path):
        path = str(tmp_path / "locs.hdf5")
        io.save_locs(path, make_locs(), INFO)
        assert io.get_format_version(path) == (1, 0)
        with h5py.File(path, "r") as f:
            assert f.attrs[io.FORMAT_VERSION_ATTR] == io.LOCS_FORMAT_VERSION
            assert f.attrs[io.PICASSO_VERSION_ATTR] == __version__

    def test_files_before_1_0_have_no_version(self, tmp_path):
        path = write_raw(tmp_path / "old.hdf5", make_locs())
        assert io.get_format_version(path) is None
        locs, _ = io.load_locs(path)
        assert len(locs) == 100

    def test_newer_minor_version_loads(self, tmp_path):
        path = write_raw(
            tmp_path / "new.hdf5", make_locs(), attrs={"picasso_format": "1.7"}
        )
        assert io.get_format_version(path) == (1, 7)
        locs, _ = io.load_locs(path)
        assert len(locs) == 100

    def test_newer_major_version_is_refused(self, tmp_path):
        path = write_raw(
            tmp_path / "new.hdf5",
            make_locs(),
            attrs={"picasso_format": "2.0", "picasso_version": "2.3.1"},
        )
        with pytest.raises(io.LocsFormatError) as error:
            io.load_locs(path)
        message = str(error.value)
        assert "format 2.0" in message
        assert "Picasso 2.3.1" in message
        assert "update Picasso" in message
        # Filter refuses it, too
        with pytest.raises(io.LocsFormatError):
            io.load_filter(path)

    def test_malformed_version_is_refused(self, tmp_path):
        path = write_raw(
            tmp_path / "bad.hdf5", make_locs(), attrs={"picasso_format": "x"}
        )
        with pytest.raises(io.LocsFormatError, match="MAJOR.MINOR"):
            io.get_format_version(path)

    def test_legacy_likelihood_becomes_log_likelihood(self, tmp_path):
        locs = make_locs().assign(likelihood=np.float32(-3.5))
        path = write_raw(tmp_path / "old.hdf5", locs)
        loaded, _ = io.load_locs(path)
        assert "likelihood" not in loaded.columns
        assert np.all(loaded["log_likelihood"] == -3.5)
        filtered, _ = io.load_filter(path)
        assert "log_likelihood" in filtered.columns

    def test_likelihood_kept_in_files_of_the_current_format(self, tmp_path):
        # "likelihood" in a 1.0 file is someone's own column
        locs = make_locs().assign(likelihood=np.float32(0.5))
        path = write_raw(
            tmp_path / "new.hdf5", locs, attrs={"picasso_format": "1.0"}
        )
        loaded, _ = io.load_locs(path)
        assert "likelihood" in loaded.columns
        assert "log_likelihood" not in loaded.columns


class TestValidation:
    @pytest.mark.parametrize("column", io.MINIMUM_COLUMNS)
    def test_missing_column_raises(self, tmp_path, column):
        path = write_raw(tmp_path / "f.hdf5", make_locs().drop(columns=column))
        with pytest.raises(io.LocsFormatError, match=f"column '{column}'"):
            io.load_locs(path)

    @pytest.mark.parametrize("key", io.MINIMUM_METADATA_KEYS)
    def test_missing_metadata_key_raises(self, tmp_path, key):
        info = [{k: v for k, v in INFO[0].items() if k != key}]
        path = write_raw(tmp_path / "f.hdf5", make_locs(), info=info)
        with pytest.raises(io.LocsFormatError, match=f"key '{key}'"):
            io.load_locs(path)

    def test_message_lists_everything_missing(self, tmp_path):
        info = [{"Frames": 100}]
        locs = make_locs().drop(columns=["lpx", "lpy"])
        path = write_raw(tmp_path / "f.hdf5", locs, info=info)
        with pytest.raises(io.LocsFormatError) as error:
            io.load_locs(path)
        message = str(error.value)
        assert path in message
        assert "the columns 'lpx', 'lpy'" in message
        assert "the metadata keys 'Width', 'Height', 'Pixelsize'" in message
        assert "files.html#files-minimum-requirements" in message

    def test_error_is_a_value_error_not_a_key_error(self):
        # Render reports a KeyError as "no localizations in the file"
        assert issubclass(io.LocsFormatError, ValueError)
        assert not issubclass(io.LocsFormatError, KeyError)

    def test_pixelsize_is_required(self, tmp_path):
        # the field of view is Width x Height camera pixels of Pixelsize
        info = [{k: v for k, v in INFO[0].items() if k != "Pixelsize"}]
        path = write_raw(tmp_path / "f.hdf5", make_locs(), info=info)
        with pytest.raises(io.MissingPixelsizeError, match="Pixelsize"):
            io.load_locs(path)
        # a file of the current format without it asks for it as well
        path = write_raw(
            tmp_path / "new.hdf5",
            make_locs(),
            info=info,
            attrs={"picasso_format": "1.0"},
        )
        with pytest.raises(io.MissingPixelsizeError):
            io.load_locs(path)
        # MissingPixelsizeError is a LocsFormatError, so one handler fits
        assert issubclass(io.MissingPixelsizeError, io.LocsFormatError)
        # the pixel size given on the spot converts the legacy file
        locs, loaded_info = io.load_locs(
            str(tmp_path / "f.hdf5"), pixelsize=130
        )
        assert len(locs) == 100
        assert lib.get_from_metadata(loaded_info, "Pixelsize") == 130

    def test_legacy_files_are_converted_to_nm(self, tmp_path):
        locs = make_locs()  # in camera pixels, see write_raw
        path = write_raw(tmp_path / "old.hdf5", locs)
        loaded, info = io.load_locs(path)
        np.testing.assert_allclose(loaded["x"], locs["x"] * 130, rtol=1e-6)
        np.testing.assert_allclose(loaded["lpx"], 0.1 * 130, rtol=1e-6)
        assert info[-1]["Step"] == "Convert units"
        assert info[-1]["Pixelsize"] == 130
        # saved again, the file is in the current format: no conversion
        out = str(tmp_path / "new.hdf5")
        io.save_locs(out, loaded, info)
        again, _ = io.load_locs(out)
        np.testing.assert_allclose(again["x"], loaded["x"])

    def test_negative_frames_raise(self, tmp_path):
        locs = make_locs().assign(frame=lambda df: df["frame"].astype(int) - 5)
        path = write_raw(tmp_path / "f.hdf5", locs)
        with pytest.raises(io.LocsFormatError, match="negative frame"):
            io.load_locs(path)

    def test_fractional_frames_load(self, tmp_path):
        # cluster centers store the mean frame of their localizations
        locs = make_locs().assign(frame=np.float32(12.25))
        path = write_raw(tmp_path / "centers.hdf5", locs)
        loaded, _ = io.load_locs(path)
        assert np.all(loaded["frame"] == 12.25)

    def test_validate_locs_accepts_a_dict(self):
        io.validate_locs(make_locs(), INFO[0])
        with pytest.raises(io.LocsFormatError, match="The localizations"):
            io.validate_locs(make_locs().drop(columns="x"), INFO[0])


class TestDroppedLocalizations:
    def test_coordinates_beyond_the_field_of_view_are_reported(
        self, tmp_path, caplog
    ):
        # a file of the current format whose coordinates are far too
        # large for its canvas, e.g., written in nm but marked with a
        # too small Pixelsize
        locs = make_locs()
        locs["x"] *= 130**2
        locs["y"] *= 130**2
        path = write_raw(
            tmp_path / "nm.hdf5", locs, attrs={"picasso_format": "1.0"}
        )
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            loaded, _ = io.load_locs(path)
        assert len(loaded) < 10
        message = caplog.text
        assert "of 100 localizations" in message
        assert "Most of them lie beyond the field of view" in message
        assert "nm" in message

    def test_a_few_dropped_rows_are_reported_without_hint(
        self, tmp_path, caplog
    ):
        locs = make_locs()
        locs.loc[:2, "lpx"] = np.nan
        path = write_raw(tmp_path / "nan.hdf5", locs)
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            loaded, _ = io.load_locs(path)
        assert len(loaded) == 97
        assert "3 of 100 localizations" in caplog.text
        assert "beyond the field of view" not in caplog.text

    def test_nothing_reported_when_nothing_is_dropped(self, tmp_path, caplog):
        path = write_raw(tmp_path / "ok.hdf5", make_locs())
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            io.load_locs(path)
        assert caplog.text == ""


class TestDarkTimes:
    def _load(self, tmp_path, caplog, info, attrs=None):
        locs = make_locs().assign(dark=np.int32(3))
        path = write_raw(tmp_path / "dark.hdf5", locs, info=info, attrs=attrs)
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            io.load_locs(path)
        return "dark times" in caplog.text

    def test_computed_before_0_11_3(self, tmp_path, caplog):
        info = INFO + [{"Generated by": "Picasso v0.10.2 Dark"}]
        assert self._load(tmp_path, caplog, info)

    def test_computed_since_0_11_3(self, tmp_path, caplog):
        info = INFO + [{"Generated by": "Picasso v0.11.3 Dark"}]
        assert not self._load(tmp_path, caplog, info)

    def test_unknown_origin_in_an_old_file(self, tmp_path, caplog):
        assert self._load(tmp_path, caplog, INFO)

    def test_unknown_origin_in_a_current_file(self, tmp_path, caplog):
        attrs = {"picasso_format": "1.0"}
        assert not self._load(tmp_path, caplog, INFO, attrs)

    def test_old_dark_times_carried_into_a_current_file(
        self, tmp_path, caplog
    ):
        info = INFO + [
            {"Generated by": "Picasso v0.9.6 Dark"},
            lib.step_info("Filter"),
        ]
        attrs = {"picasso_format": "1.0"}
        assert self._load(tmp_path, caplog, info, attrs)

    def test_recorded_by_step_info(self, tmp_path, caplog):
        info = INFO + [lib.step_info("Dark")]
        assert not self._load(tmp_path, caplog, info)


class TestSidecar:
    def test_yaml_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: True)
        path = str(tmp_path / "locs.hdf5")
        io.save_locs(path, make_locs(), INFO)
        # edited by hand
        edited = [dict(INFO[0], Pixelsize=108)]
        with open(tmp_path / "locs.yaml", "w") as f:
            yaml.safe_dump_all(edited, f)
        _, info = io.load_locs(path)
        assert info[0]["Pixelsize"] == 108

    def test_stale_yaml_is_overwritten(self, tmp_path, monkeypatch):
        # saved first with the .yaml sidecar on, then again with it off
        path = str(tmp_path / "locs.hdf5")
        monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: True)
        io.save_locs(path, make_locs(), INFO)
        monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: False)
        new_info = [dict(INFO[0], Pixelsize=100), lib.step_info("Test")]
        io.save_locs(path, make_locs(), new_info)
        _, info = io.load_locs(path)
        assert info[0]["Pixelsize"] == 100
        assert info[-1]["Step"] == "Test"

    def test_no_yaml_written_when_off_and_absent(self, tmp_path):
        path = str(tmp_path / "locs.hdf5")
        io.save_locs(path, make_locs(), INFO)
        assert not (tmp_path / "locs.yaml").exists()

    def test_stale_yaml_of_identifications_is_overwritten(
        self, tmp_path, monkeypatch
    ):
        path = str(tmp_path / "ids.hdf5")
        ids = pd.DataFrame(
            {
                "frame": np.zeros(3, np.uint32),
                "x": np.arange(3, dtype=np.int32),
                "y": np.arange(3, dtype=np.int32),
            }
        )
        monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: True)
        io.save_identifications(path, ids, INFO)
        monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: False)
        io.save_identifications(path, ids, [dict(INFO[0], Frames=7)])
        assert io.load_info(path)[0]["Frames"] == 7


class TestYaml:
    def test_save_info_writes_plain_yaml(self, tmp_path):
        info = [
            {
                "Width": np.int64(64),
                "Pixelsize": np.float32(130.5),
                "Flag": np.bool_(True),
                "Pair": (1, 2),
                "Array": np.arange(4).reshape(2, 2),
                "Nested": {"Values": np.array([0.5, 1.5])},
                "Set": {3},
            },
            {"Ordered": collections.OrderedDict(a=np.int32(1))},
        ]
        path = tmp_path / "info.yaml"
        io.save_info(str(path), info)
        text = path.read_text()
        assert "!!" not in text
        loaded = list(yaml.safe_load_all(text))
        assert loaded == [
            {
                "Width": 64,
                "Pixelsize": 130.5,
                "Flag": True,
                "Pair": [1, 2],
                "Array": [[0, 1], [2, 3]],
                "Nested": {"Values": [0.5, 1.5]},
                "Set": [3],
            },
            {"Ordered": {"a": 1}},
        ]

    def test_reads_python_tags_written_before_1_0(self, tmp_path):
        # what PyYAML's default dumper wrote into Picasso's .yaml files
        info = [
            {
                "Width": np.int64(64),
                "Height": np.uint32(64),
                "Frames": np.int32(100),
                "Pixelsize": np.float64(130.5),
                "Precision": np.float32(0.25),
                "Flag": np.bool_(True),
                "Label": np.str_("abc"),
                "Pair": (1, 2),
                "Array": np.arange(6, dtype=np.float32).reshape(2, 3),
                "Fortran": np.asfortranarray(np.arange(6).reshape(2, 3)),
                "Ordered": collections.OrderedDict(a=1, b=2),
                "Settings": lib.AutoDict(),
            }
        ]
        info[0]["Settings"]["Render"]["Colormap"] = "magma"
        text = yaml.dump_all(info)
        assert "!!python/object/apply:numpy" in text
        path = tmp_path / "old.yaml"
        path.write_text(text)
        loaded = io.load_info(str(path))
        assert loaded == [
            {
                "Width": 64,
                "Height": 64,
                "Frames": 100,
                "Pixelsize": 130.5,
                "Precision": 0.25,
                "Flag": True,
                "Label": "abc",
                "Pair": (1, 2),
                "Array": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                "Fortran": [[0, 1, 2], [3, 4, 5]],
                "Ordered": {"a": 1, "b": 2},
                "Settings": {"Render": {"Colormap": "magma"}},
            }
        ]

    def test_reads_numpy_1_module_paths(self):
        text = yaml.dump({"Value": np.float64(2.5)}).replace(
            "numpy._core.", "numpy.core."
        )
        assert io._parse_yaml(text) == {"Value": 2.5}

    def test_never_runs_code_from_the_file(self, tmp_path, caplog):
        marker = tmp_path / "ran"
        text = (
            "Width: 64\n"
            f"Run: !!python/object/apply:os.system ['touch {marker}']\n"
            f"New: !!python/object/new:os.system ['touch {marker}']\n"
            "Name: !!python/name:os.system ''\n"
            "Object: !!python/object:collections.Counter {a: 1}\n"
        )
        path = tmp_path / "evil.yaml"
        path.write_text(text)
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            info = io.load_info(str(path))
        assert not marker.exists()
        assert info[0]["Width"] == 64
        assert info[0]["Run"] == [f"touch {marker}"]
        assert info[0]["Name"] == "os.system"
        assert info[0]["Object"] == {"a": 1}
        assert "os.system" in caplog.text

    def test_calibrations_and_picks_read_safely(self, tmp_path):
        marker = tmp_path / "ran"
        path = tmp_path / "picks.yaml"
        path.write_text(
            "Shape: Circle\nDiameter: 2.0\nCenters: [[1.0, 2.0]]\n"
            f"Run: !!python/object/apply:os.system ['touch {marker}']\n"
        )
        picks, shape, size = io.load_picks(str(path), pixelsize=130)
        assert shape == "Circle"
        assert not marker.exists()

    def test_camera_config_read_safely(self, tmp_path, monkeypatch):
        import picasso

        path = tmp_path / "config.yaml"
        path.write_text(
            "Cameras: !!python/object/new:collections.Counter []\n"
        )
        monkeypatch.setattr(picasso, "resolve_config_path", lambda: str(path))
        # refused, and Picasso starts without cameras (see test_config.py)
        with pytest.warns(UserWarning, match="not valid YAML"):
            assert picasso.load_config() == {}

    def test_no_unsafe_yaml_loader_in_picasso(self):
        # every .yaml Picasso opens may come from someone else; read it
        # with yaml.safe_load or io.load_yaml
        unsafe = re.compile(
            r"yaml\.(load|load_all|full_load|full_load_all|unsafe_load"
            r"|unsafe_load_all)\(|FullLoader|UnsafeLoader|yaml\.Loader\b"
        )
        package = os.path.dirname(io.__file__)
        found = []
        for folder, _, files in os.walk(package):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as file:
                    for number, line in enumerate(file, 1):
                        if unsafe.search(line):
                            found.append(
                                f"{os.path.relpath(path, package)}:{number}"
                            )
        assert not found, "Unsafe YAML loading in:\n  " + "\n  ".join(found)


class TestStepInfo:
    def test_keys(self):
        entry = lib.step_info("Undrift", **{"Segmentation": 1000})
        assert list(entry) == [
            "Generated by",
            "Picasso version",
            "Step",
            "Segmentation",
        ]
        assert entry["Generated by"] == f"Picasso v{__version__} Undrift"
        assert entry["Picasso version"] == __version__
        assert entry["Step"] == "Undrift"
        assert entry["Segmentation"] == 1000

    def test_filter_steps_found_by_step(self):
        info = INFO + [
            lib.step_info("Filter", photons=[100.0, 1000.0]),
            # a step whose name merely contains "Filter" is no filter
            lib.step_info("Render : Filter picks", photons=[0.0, 1.0]),
        ]
        ranges, _, _ = lib.extract_filter_steps(info, ["photons"])
        assert ranges == {"photons": [100.0, 1000.0]}

    def test_filter_steps_saved_before_1_0(self):
        info = INFO + [{"Generated by": "Picasso v0.11 Filter", "sx": [1, 2]}]
        ranges, _, _ = lib.extract_filter_steps(info, ["sx"])
        assert ranges == {"sx": [1.0, 2.0]}

    def test_saved_steps_load_back(self, tmp_path):
        path = str(tmp_path / "locs.hdf5")
        io.save_locs(path, make_locs(), INFO + [lib.step_info("Link")])
        _, info = io.load_locs(path)
        assert info[-1]["Step"] == "Link"
        assert info[-1]["Picasso version"] == __version__


class TestRenderLoader:
    """The Render loader shows the problems found while reading a file."""

    def _run(self, path):
        from picasso.gui.render import LocsLoadWorker

        worker = LocsLoadWorker([(path, None)])
        loaded, failed, warned = [], [], []
        worker.loaded.connect(lambda p, locs, info, index: loaded.append(p))
        worker.failed.connect(lambda p, message: failed.append(message))
        worker.warned.connect(lambda p, message: warned.append(message))
        worker.run()  # on this thread, so the signals arrive directly
        return loaded, failed, warned

    def test_dropped_localizations_are_shown(self, qapp, tmp_path):
        locs = make_locs()
        locs["x"] *= 130**2  # beyond the field of view
        path = write_raw(
            tmp_path / "nm.hdf5", locs, attrs={"picasso_format": "1.0"}
        )
        loaded, failed, warned = self._run(path)
        assert loaded == [path]
        assert failed == []
        assert len(warned) == 1
        assert "were not loaded" in warned[0]
        # the field of view with the values of the metadata
        assert "(8,320 x 8,320 nm) or have" in (warned[0])

    def test_format_errors_are_shown(self, qapp, tmp_path):
        path = write_raw(tmp_path / "f.hdf5", make_locs().drop(columns="lpy"))
        loaded, failed, warned = self._run(path)
        assert loaded == []
        assert len(failed) == 1
        assert "column 'lpy'" in failed[0]

    def test_clean_files_show_nothing(self, qapp, tmp_path):
        path = write_raw(tmp_path / "ok.hdf5", make_locs())
        loaded, failed, warned = self._run(path)
        assert loaded == [path]
        assert failed == [] and warned == []

    def test_records_of_other_threads_are_not_collected(self, qapp):
        import threading

        from picasso.gui.render import _ThreadLogRecords

        records = _ThreadLogRecords()
        caught = []

        def log_elsewhere():
            record = logging.LogRecord(
                "picasso.io", logging.WARNING, "", 0, "elsewhere", None, None
            )
            caught.append(record.thread != records.thread)
            records.emit(record)

        thread = threading.Thread(target=log_elsewhere)
        thread.start()
        thread.join()
        assert caught == [True]
        assert records.messages == []
