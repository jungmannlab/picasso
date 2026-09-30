"""Picasso: Render and the user settings file: the ``Render`` defaults
are written on start, a section that carries only some of the keys
Render knows is tolerated, and an unreadable file is reported instead of
silently replaced.

``~/.picasso/settings.yaml`` is a plain YAML file that users edit by
hand (File > Picasso settings), and other parts of Picasso write single
keys into it. Render must therefore cope with a ``Render`` section that
is incomplete, rather than assuming that a section which exists holds
every key.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
import yaml
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import render as gui_render


@pytest.fixture
def settings_path(tmp_path):
    return tmp_path / "settings.yaml"  # see ``isolated_user_settings``


def test_render_writes_missing_defaults_on_start(qt_offscreen, settings_path):
    io.save_user_settings({"Render": {"Colormap": "hot"}})
    window = gui_render.Window(plugins_loaded=True)
    saved = yaml.safe_load(settings_path.read_text())["Render"]
    assert saved["Colormap"] == "hot"
    assert saved["gpu"]["enabled"] == "auto"
    assert saved["max_blur_width"] == 100.0
    assert saved["interaction_subsample"] == "auto"
    window.close()


def test_info_dialog_renderer_row_has_help_and_wraps(qt_offscreen):
    window = gui_render.Window(plugins_loaded=True)
    dialog = window.info_dialog
    assert dialog.renderer_label.wordWrap()
    # a long GPU name wraps inside the dialog instead of widening it
    policy = dialog.renderer_label.sizePolicy().horizontalPolicy()
    assert policy == QtWidgets.QSizePolicy.Policy.Ignored
    width = dialog.sizeHint().width()
    dialog.renderer_label.setText(
        "GPU (NVIDIA GeForce RTX 4090 Laptop GPU with Max-Q Design via Vulkan)"
    )
    assert dialog.sizeHint().width() == width
    assert dialog.renderer_help.help_url.endswith("#gpu-rendering")
    assert "GPU" in dialog.renderer_help.toolTip()
    window.view.stop_render_worker()


def test_render_reports_an_unreadable_settings_file(
    qt_offscreen, settings_path, monkeypatch
):
    settings_path.write_text("Render: [")
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )
    window = gui_render.Window(plugins_loaded=True)
    assert len(shown) == 1
    assert "could not be read" in shown[0][2]
    kept = settings_path.with_name("settings.yaml.broken")
    assert kept.read_text() == "Render: ["
    assert io.settings_load_error() is None  # reported, hence dismissed
    window.close()


class TestLoadUserSettings:
    """What Render makes of the ``Render`` section it finds on start."""

    def test_no_settings_file(self, qt_offscreen):
        window = gui_render.Window(plugins_loaded=True)
        assert window.display_settings_dlg.colormap.currentText() == "magma"
        window.close()

    def test_partial_render_section(self, qt_offscreen):
        """A ``Render`` section without a colormap must not stop Render
        from starting; the default colormap is used."""
        io.save_user_settings({"Render": {"PWD": "/tmp"}})
        window = gui_render.Window(plugins_loaded=True)
        assert window.display_settings_dlg.colormap.currentText() == "magma"
        window.close()

    def test_starts_after_a_setting_created_the_render_section(
        self, qt_offscreen
    ):
        """Settings that persist their default (here the color bar
        format) write into the ``Render`` section, leaving it without a
        colormap - which must still start."""
        io.colorbar_export_format()
        window = gui_render.Window(plugins_loaded=True)
        assert window.display_settings_dlg.colormap.currentText() == "magma"
        window.close()

    def test_saved_colormap_is_applied(self, qt_offscreen):
        io.save_user_settings({"Render": {"Colormap": "hot"}})
        window = gui_render.Window(plugins_loaded=True)
        assert window.display_settings_dlg.colormap.currentText() == "hot"
        window.close()
