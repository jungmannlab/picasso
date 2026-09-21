"""How Render reads the user settings file at startup.

``~/.picasso/settings.yaml`` is a plain YAML file that users edit by
hand (File > Picasso settings), and other parts of Picasso write single
keys into it. Render must therefore cope with a ``Render`` section that
carries only some of the keys it knows, rather than assuming that a
section which exists is complete.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest

from picasso import io
from picasso.gui import render as gui_render


@pytest.fixture(autouse=True)
def settings_file(tmp_path, monkeypatch):
    """Never touch the developer's ~/.picasso/settings.yaml."""
    path = tmp_path / "settings.yaml"
    monkeypatch.setattr(io, "_user_settings_filename", lambda: str(path))
    return path


class TestLoadUserSettings:
    def test_no_settings_file(self, qt_offscreen):
        window = gui_render.Window(plugins_loaded=True)
        assert window.display_settings_dlg.colormap.currentText() == "magma"

    def test_partial_render_section(self, qt_offscreen):
        """A ``Render`` section without a colormap must not stop Render
        from starting; the default colormap is used."""
        io.save_user_settings({"Render": {"PWD": "/tmp"}})
        window = gui_render.Window(plugins_loaded=True)
        assert window.display_settings_dlg.colormap.currentText() == "magma"

    def test_starts_after_a_setting_created_the_render_section(
        self, qt_offscreen
    ):
        """Settings that persist their default (here the color bar
        format) write into the ``Render`` section, leaving it without a
        colormap - which must still start."""
        io.colorbar_export_format()
        window = gui_render.Window(plugins_loaded=True)
        assert window.display_settings_dlg.colormap.currentText() == "magma"

    def test_saved_colormap_is_applied(self, qt_offscreen):
        io.save_user_settings({"Render": {"Colormap": "hot"}})
        window = gui_render.Window(plugins_loaded=True)
        assert window.display_settings_dlg.colormap.currentText() == "hot"
