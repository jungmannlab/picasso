"""The user settings file (``~/.picasso/settings.yaml``) must never be
lost.

Every Picasso tool loads the settings, changes its own keys and writes
the whole file back. A file that could not be parsed used to be
silently replaced by default settings on the next save, taking every
other section with it. Now an unreadable file is kept as a copy and
reported, every save keeps the previous file as a backup, and the
``Render`` keys are written with their defaults when absent, as the
other settings are.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import logging

import pytest
import yaml

from picasso import io, lib
from picasso.render import backend


@pytest.fixture
def settings_path(tmp_path):
    # ``isolated_user_settings`` (conftest) already points the settings
    # file into ``tmp_path``; this names it for the tests.
    return tmp_path / "settings.yaml"


def _saved(settings_path):
    return yaml.safe_load(settings_path.read_text())


class TestUnreadableSettingsFile:
    BROKEN = "Render:\n  Colormap: [unclosed\n"

    def test_broken_file_is_kept_and_reported(self, settings_path, caplog):
        settings_path.write_text(self.BROKEN)
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            settings = io.load_user_settings()
        assert settings == {}
        kept = settings_path.with_name("settings.yaml.broken")
        assert kept.read_text() == self.BROKEN
        assert "could not be read" in caplog.text
        message, copy = io.settings_load_error()
        assert message
        assert copy == str(kept)
        assert io.settings_file_is_broken()

    def test_reported_once_per_file(self, settings_path, caplog):
        settings_path.write_text(self.BROKEN)
        with caplog.at_level(logging.WARNING, logger="picasso.io"):
            io.load_user_settings()
            io.load_user_settings()
        assert caplog.text.count("could not be read") == 1

    def test_fixed_file_loads_and_error_stays_until_dismissed(
        self, settings_path
    ):
        settings_path.write_text(self.BROKEN)
        io.load_user_settings()
        settings_path.write_text("Render:\n  Colormap: hot\n")
        assert io.load_user_settings()["Render"]["Colormap"] == "hot"
        assert not io.settings_file_is_broken()
        # sticky so a GUI started later in the process can still report it
        assert io.settings_load_error() is not None
        io.dismiss_settings_load_error()
        assert io.settings_load_error() is None

    def test_non_mapping_file_counts_as_broken(self, settings_path):
        settings_path.write_text("- just\n- a list\n")
        assert io.load_user_settings() == {}
        assert io.settings_load_error() is not None

    def test_missing_file_is_not_an_error(self, settings_path):
        assert io.load_user_settings() == {}
        assert io.settings_load_error() is None


class TestSaveKeepsABackup:
    def test_previous_file_becomes_bak(self, settings_path):
        io.save_user_settings({"Render": {"Colormap": "hot"}})
        backup = settings_path.with_name("settings.yaml.bak")
        assert not backup.exists()  # nothing to back up on the first save
        io.save_user_settings({"Render": {"Colormap": "magma"}})
        assert (
            yaml.safe_load(backup.read_text())["Render"]["Colormap"] == "hot"
        )
        assert _saved(settings_path)["Render"]["Colormap"] == "magma"

    def test_broken_file_never_replaces_the_backup(self, settings_path):
        io.save_user_settings({"Render": {"Colormap": "hot"}})
        io.save_user_settings({"Render": {"Colormap": "magma"}})
        settings_path.write_text("Render: [")
        io.load_user_settings()  # recorded as unreadable
        io.save_user_settings({"Render": {"Colormap": "viridis"}})
        backup = settings_path.with_name("settings.yaml.bak")
        assert (
            yaml.safe_load(backup.read_text())["Render"]["Colormap"] == "hot"
        )
        assert _saved(settings_path)["Render"]["Colormap"] == "viridis"
        assert not io.settings_file_is_broken()

    def test_save_leaves_no_temporary_file(self, settings_path):
        io.save_user_settings({"Render": {"Colormap": "hot"}})
        names = sorted(p.name for p in settings_path.parent.iterdir())
        assert names == ["settings.yaml"]


class TestPersistRenderDefaults:
    def test_missing_keys_are_written_and_others_kept(self, settings_path):
        io.save_user_settings(
            {
                "Render": {"Colormap": "hot"},
                "Localize": {"cpu_utilization": 0.8},
            }
        )
        assert backend.persist_render_defaults() is True
        saved = _saved(settings_path)
        assert saved["Localize"] == {"cpu_utilization": 0.8}
        render_section = saved["Render"]
        assert render_section["Colormap"] == "hot"
        assert render_section["cpu_utilization"] == (
            lib.RENDER_CPU_UTILIZATION_DEFAULT
        )
        assert render_section["interaction_subsample"] == "auto"
        assert render_section["max_blur_width"] == (
            lib.RENDER_MAX_BLUR_WIDTH_DEFAULT
        )
        assert render_section["gpu"] == {
            "enabled": "auto",
            "adapter": "high-performance",
            "vram_budget_mb": lib.RENDER_VRAM_BUDGET_MB_DEFAULT,
        }
        assert "max_workers" not in render_section  # optional, not persisted

    def test_existing_values_win_and_second_call_is_a_noop(
        self, settings_path
    ):
        io.save_user_settings(
            {"Render": {"gpu": {"enabled": "off"}, "max_blur_width": 0}}
        )
        assert backend.persist_render_defaults() is True
        saved = _saved(settings_path)["Render"]
        assert saved["gpu"]["enabled"] == "off"
        assert saved["gpu"]["adapter"] == "high-performance"
        assert saved["max_blur_width"] == 0
        before = settings_path.read_text()
        assert backend.persist_render_defaults() is False
        assert settings_path.read_text() == before

    def test_creates_the_file_when_absent(self, settings_path):
        assert backend.persist_render_defaults() is True
        assert _saved(settings_path)["Render"]["gpu"]["enabled"] == "auto"

    def test_unreadable_file_is_left_alone(self, settings_path):
        settings_path.write_text("Render: [")
        assert backend.persist_render_defaults() is False
        assert settings_path.read_text() == "Render: ["

    def test_persisted_defaults_are_what_the_readers_use(self, settings_path):
        backend.persist_render_defaults()
        assert backend.gpu_settings() == {
            "enabled": "auto",
            "adapter": "high-performance",
            "vram_budget_bytes": lib.RENDER_VRAM_BUDGET_MB_DEFAULT * 2**20,
        }
