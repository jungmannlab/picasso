"""Picasso's library modules report through ``logging`` instead of
printing, say when they leave metadata out or fall back to defaults,
and leave the warning filters of the caller alone; the command line
shows their messages in the console.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from picasso import io, localize, postprocess
from picasso import __main__ as cli


def _no_logging_setup(monkeypatch):
    """Logging as the command line finds it: no handlers. In the test
    itself, as pytest adds its handlers to the root logger only then."""
    monkeypatch.setattr(logging.getLogger(), "handlers", [])


class TestCommandLine:
    @pytest.fixture
    def picasso_logger(self, monkeypatch):
        """The ``picasso`` logger, restored after the test."""
        logger = logging.getLogger("picasso")
        monkeypatch.setattr(logger, "handlers", [])
        level = logger.level
        yield logger
        logger.setLevel(level)

    def test_shows_information_and_warnings(
        self, picasso_logger, monkeypatch, capsys
    ):
        _no_logging_setup(monkeypatch)
        cli._log_to_console()
        cli._log_to_console()  # installed once
        logging.getLogger("picasso.io").info("Reading %s", "movie.raw")
        logging.getLogger("picasso.plugins").warning("%r failed", "x.py")
        logging.getLogger("picasso.io").debug("not shown")
        assert capsys.readouterr().err == (
            "Reading movie.raw\nWarning: 'x.py' failed\n"
        )

    def test_keeps_a_logging_setup(self, picasso_logger, monkeypatch):
        monkeypatch.setattr(
            logging.getLogger(), "handlers", [logging.NullHandler()]
        )
        cli._log_to_console()
        assert picasso_logger.handlers == []

    def test_a_command_shows_the_messages(
        self, picasso_logger, monkeypatch, capsys
    ):
        def pair_correlation(*args):
            logging.getLogger("picasso.postprocess").info("calculated")

        monkeypatch.setattr(cli, "_pair_correlation", pair_correlation)
        monkeypatch.setattr("sys.argv", ["picasso", "pair-correlation", "f"])
        _no_logging_setup(monkeypatch)
        cli.main()
        assert "calculated" in capsys.readouterr().err


def test_evaluate_picks_keeps_the_warning_filters(locs, info):
    picks = [locs.iloc[:200], locs.iloc[200:201]]  # one with one loc
    before = list(warnings.filters)
    postprocess.evaluate_picks(picks, info)
    assert warnings.filters == before


class TestMetadata:
    def test_unreadable_micro_manager_metadata(self, caplog):
        tag = SimpleNamespace(value=b"{not json")
        tif = SimpleNamespace(
            filename="movie.ome.tif",
            pages=[SimpleNamespace(tags={51123: tag})],
        )
        assert io._mm_per_image_metadata_from_tifffile(tif) == {}
        assert "Micro-Manager metadata of movie.ome.tif" in caplog.text

    def test_camera_not_in_the_configuration(self, caplog):
        movie = SimpleNamespace(meta={"Camera": "Andor"}, path="m.ome.tif")
        parameters = io.TiffMultiMap.camera_parameters(
            movie, {"Cameras": {"Hamamatsu": {}}}
        )
        assert parameters["gain"] == [1]
        (record,) = caplog.records
        assert record.levelname == "WARNING"
        assert "'Andor'" in record.getMessage()
        assert "'Hamamatsu'" in record.getMessage()

    def test_no_camera_configuration_is_no_warning(self, caplog):
        movie = SimpleNamespace(meta={"Camera": "Andor"}, path="m.ome.tif")
        with caplog.at_level(logging.WARNING):
            io.TiffMultiMap.camera_parameters(movie, {})
        assert caplog.records == []


class TestLocalizeChecks:
    def test_nena(self, locs, info, monkeypatch, caplog):
        monkeypatch.setattr(
            localize.postprocess, "nena", lambda *args, **kwargs: (None, 0.5)
        )
        with caplog.at_level(logging.INFO, logger="picasso.localize"):
            assert localize.check_nena(locs, info) == 0.5
        assert "NeNA: 0.50 px." in caplog.text

    def test_nena_failure_is_a_warning(self, locs, info, monkeypatch, caplog):
        def fail(*args, **kwargs):
            raise ValueError("too few localizations")

        monkeypatch.setattr(localize.postprocess, "nena", fail)
        assert np.isnan(localize.check_nena(locs, info))
        (record,) = caplog.records
        assert record.levelname == "WARNING"
        assert "too few localizations" in record.getMessage()
