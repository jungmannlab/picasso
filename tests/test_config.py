"""A camera configuration (``~/.picasso/config.yaml``) that cannot be
read does not stop Picasso: it is warned about, no camera is configured
and Localize says so once.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import picasso
from picasso import lib_qt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(picasso.__file__)))


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(picasso, "resolve_config_path", lambda: str(path))
    monkeypatch.setattr(picasso, "CONFIG_LOAD_ERROR", None)
    return path


def test_valid_config(config_file):
    config_file.write_text("Cameras:\n  Andor:\n    Baseline: 100\n")
    assert picasso.load_config() == {"Cameras": {"Andor": {"Baseline": 100}}}
    assert picasso.CONFIG_LOAD_ERROR is None


def test_empty_config(config_file, recwarn):
    config_file.write_text("")
    assert picasso.load_config() == {}
    assert picasso.CONFIG_LOAD_ERROR is None
    assert not recwarn.list


@pytest.mark.parametrize(
    "text, reason",
    [
        ("Cameras: [Andor\n", "not valid YAML"),
        ("- Andor\n- Hamamatsu\n", "does not hold settings"),
    ],
)
def test_broken_config_configures_no_camera(config_file, text, reason):
    config_file.write_text(text)
    with pytest.warns(UserWarning, match=reason):
        assert picasso.load_config() == {}
    assert str(config_file) in picasso.CONFIG_LOAD_ERROR


def test_import_survives_a_broken_config(tmp_path):
    # ~/.picasso/config.yaml of a fresh interpreter
    os.makedirs(tmp_path / ".picasso")
    (tmp_path / ".picasso" / "config.yaml").write_text("Cameras: [Andor\n")
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import picasso; print(picasso.CONFIG, "
            "picasso.CONFIG_LOAD_ERROR is not None)",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "{} True"
    assert "could not be read" in result.stderr


def test_localize_tells_once(qt_offscreen, monkeypatch):
    from picasso.gui import localize as gui_localize

    monkeypatch.setattr(
        picasso, "CONFIG_LOAD_ERROR", "The camera configuration x is broken"
    )
    monkeypatch.setattr(lib_qt, "_config_load_error_notified", False)
    shown = []
    monkeypatch.setattr(
        lib_qt.QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, text: shown.append(text),
    )
    gui_localize.Window()
    gui_localize.Window()
    assert len(shown) == 1
    assert "The camera configuration x is broken" in shown[0]
    assert "localize-camera-config" in shown[0]
