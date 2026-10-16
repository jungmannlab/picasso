"""Test ``picasso.gui.plugins_loader`` — discovery and loading of user
plugins from ``~/.picasso/plugins``, plus the safeguards around sharing
them: enable-before-load, manifest hash verification and validation of
ids and paths coming from the registry.

Plugins are written as temporary ``.py`` files into a ``tmp_path`` and the
loader is pointed there by monkeypatching ``picasso.io.plugins_directory``.
A tiny fake window stands in for the GUI so the loader is exercised without
a ``QApplication`` (the loader keeps Qt imports inside the menu helper).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import hashlib
import json
import os
import types

import pytest

from picasso import io
from picasso import plugins as plugins_api
from picasso.gui import plugins_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMenu:
    """Minimal stand-in for ``window.plugin_menu`` used by plugins."""

    def __init__(self):
        self.actions = []

    def addAction(self, label):
        self.actions.append(label)
        return types.SimpleNamespace(
            triggered=types.SimpleNamespace(connect=lambda *a, **k: None)
        )

    def addSeparator(self):
        pass

    def clear(self):
        self.actions = []


def _make_window():
    return types.SimpleNamespace(plugin_menu=FakeMenu())


def _write_plugin(directory, filename, app_name, body=""):
    """Write a plugin ``.py`` file whose ``execute`` records the menu label
    'loaded-<name>' so the test can assert it ran."""
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        f.write(
            "class Plugin:\n"
            "    def __init__(self, window):\n"
            f"        self.name = {app_name!r}\n"
            "        self.window = window\n"
            "    def execute(self):\n"
            "        label = 'loaded-' + self.name\n"
            "        self.window.plugin_menu.addAction(label)\n"
            f"{body}"
        )
    return path


def _point_loader_at(monkeypatch, directory):
    monkeypatch.setattr(io, "plugins_directory", lambda: directory)


def _enable(filename, value=True):
    """Mark ``filename`` enabled in the sidecar, as the store dialog does."""
    state = plugins_loader.load_state()
    plugins_loader.set_enabled(state, filename, value)


def _write_enabled_plugin(directory, filename, app_name, body=""):
    path = _write_plugin(directory, filename, app_name, body=body)
    _enable(filename)
    return path


# ---------------------------------------------------------------------------
# plugins_directory
# ---------------------------------------------------------------------------


class TestPluginsDirectory:
    def test_created_under_dot_picasso(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
        directory = io.plugins_directory()
        assert directory == os.path.join(str(tmp_path), ".picasso", "plugins")
        assert os.path.isdir(directory)


# ---------------------------------------------------------------------------
# load_plugins
# ---------------------------------------------------------------------------


class TestLoadPlugins:
    def test_loads_and_executes_matching_plugin(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_enabled_plugin(str(tmp_path), "myplugin.py", "render")

        window = _make_window()
        result = plugins_loader.load_plugins(window, "render")

        assert len(result) == 1
        assert window.plugins is result
        assert "loaded-render" in window.plugin_menu.actions

    def test_filters_by_app_name(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_enabled_plugin(str(tmp_path), "render_plugin.py", "render")

        window = _make_window()
        plugins_loader.load_plugins(window, "localize")

        assert window.plugins == []
        assert "loaded-render" not in window.plugin_menu.actions

    def test_skips_underscore_files(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_enabled_plugin(str(tmp_path), "_helper.py", "render")

        window = _make_window()
        plugins_loader.load_plugins(window, "render")

        assert window.plugins == []

    def test_broken_plugin_is_skipped(self, tmp_path, monkeypatch, capsys):
        _point_loader_at(monkeypatch, str(tmp_path))
        # execute raises -> must be caught, logged, and skipped
        _write_enabled_plugin(
            str(tmp_path),
            "broken.py",
            "render",
            body="        raise RuntimeError('boom')\n",
        )

        window = _make_window()
        result = plugins_loader.load_plugins(window, "render")

        assert result == []
        assert "Failed to load plugin" in capsys.readouterr().out

    def test_empty_directory_returns_empty_list(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))

        window = _make_window()
        result = plugins_loader.load_plugins(window, "render")

        assert result == []
        assert window.plugins == []


# ---------------------------------------------------------------------------
# Enable-before-load
# ---------------------------------------------------------------------------


class TestEnabledGate:
    def test_file_dropped_into_folder_is_not_executed(
        self, tmp_path, monkeypatch
    ):
        """A plugin nobody enabled must never run, even if it is in place.

        This is the "a colleague sent me this file" case: discovery is fine,
        execution requires consent.
        """
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_plugin(str(tmp_path), "dropped.py", "render")

        window = _make_window()
        result = plugins_loader.load_plugins(window, "render")

        assert result == []
        assert "loaded-render" not in window.plugin_menu.actions

    def test_disabling_stops_loading(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_enabled_plugin(str(tmp_path), "toggle.py", "render")
        _enable("toggle.py", False)

        window = _make_window()
        assert plugins_loader.load_plugins(window, "render") == []

    def test_disabled_files_are_reported(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_plugin(str(tmp_path), "pending.py", "render")
        _write_enabled_plugin(str(tmp_path), "active.py", "render")

        assert plugins_loader.disabled_plugin_files() == ["pending.py"]

    def test_enabled_flag_survives_reload_from_disk(
        self, tmp_path, monkeypatch
    ):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_enabled_plugin(str(tmp_path), "sticky.py", "render")

        assert plugins_loader.is_enabled(
            plugins_loader.load_state(), "sticky.py"
        )

    def test_corrupt_sidecar_disables_everything(self, tmp_path, monkeypatch):
        """A corrupt sidecar must fail closed, not load unvetted code."""
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_plugin(str(tmp_path), "unvetted.py", "render")
        with open(tmp_path / ".installed.json", "w") as f:
            f.write("{not json")

        window = _make_window()
        assert plugins_loader.load_plugins(window, "render") == []


# ---------------------------------------------------------------------------
# Validation of ids and repository paths
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "value",
        ["render_tools", "abc-123", "A1"],
    )
    def test_accepts_plain_ids(self, value):
        assert plugins_loader.is_safe_id(value)

    @pytest.mark.parametrize(
        "value",
        [
            "../../../.bashrc",
            "a/b",
            "a\\b",
            ".hidden",
            "-leading",
            "with space",
            "",
            None,
            42,
        ],
    )
    def test_rejects_unsafe_ids(self, value):
        assert not plugins_loader.is_safe_id(value)

    @pytest.mark.parametrize(
        "value",
        ["plugins/render/tool.py", "tool.py"],
    )
    def test_accepts_relative_repo_paths(self, value):
        assert plugins_loader.is_safe_repo_path(value)

    @pytest.mark.parametrize(
        "value",
        [
            "../other_repo/evil.py",
            "/etc/passwd.py",
            "a\\b.py",
            "https://example.com/evil.py",
            "tool.txt",
            "",
            None,
        ],
    )
    def test_rejects_unsafe_repo_paths(self, value):
        assert not plugins_loader.is_safe_repo_path(value)

    def test_plugin_path_stays_inside_folder(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        assert plugins_loader.plugin_path("ok.py") == os.path.join(
            str(tmp_path), "ok.py"
        )
        for name in ("../escape.py", "sub/ok.py", "ok.txt"):
            with pytest.raises(ValueError):
                plugins_loader.plugin_path(name)

    def test_manifest_drops_unsafe_entries(
        self, tmp_path, monkeypatch, caplog
    ):
        _point_loader_at(monkeypatch, str(tmp_path))
        good = {"id": "good", "file": "good.py", "sha256": "a" * 64}
        manifest = {
            "plugins": [
                good,
                {"id": "../evil", "file": "evil.py", "sha256": "b" * 64},
                {"id": "evil2", "file": "../../evil.py", "sha256": "c" * 64},
                {"file": "no_id.py"},
                "not a dict",
            ]
        }
        monkeypatch.setattr(
            plugins_api,
            "_get",
            lambda url: json.dumps(manifest).encode(),
        )

        assert plugins_loader.fetch_manifest() == [good]
        assert "unsafe id or file path" in caplog.text

    def test_uninstall_ignores_unsafe_recorded_filename(
        self, tmp_path, monkeypatch
    ):
        """A tampered sidecar must not delete files outside the folder."""
        _point_loader_at(monkeypatch, str(tmp_path))
        outside = tmp_path.parent / "victim.py"
        outside.write_text("keep me")
        state = {
            "plugins": {"x": {"file": "../victim.py"}},
            "enabled": {},
            "trust_acknowledged": True,
        }

        plugins_loader.uninstall("x", state)

        assert outside.exists()
        assert "x" not in state["plugins"]


# ---------------------------------------------------------------------------
# Integrity-verified install
# ---------------------------------------------------------------------------


def _entry(content: bytes, **overrides):
    entry = {
        "id": "demo",
        "display_name": "Demo",
        "app": "render",
        "file": "plugins/demo.py",
        "version": "1.0",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    entry.update(overrides)
    return entry


class TestInstallIntegrity:
    def test_install_writes_verified_file_and_enables_it(
        self, tmp_path, monkeypatch
    ):
        _point_loader_at(monkeypatch, str(tmp_path))
        content = b"# demo plugin\n"
        monkeypatch.setattr(plugins_api, "_get", lambda url: content)
        state = plugins_loader.load_state()

        plugins_loader.install(_entry(content), state)

        target = tmp_path / "demo.py"
        assert target.read_bytes() == content
        record = state["plugins"]["demo"]
        assert record["sha256"] == hashlib.sha256(content).hexdigest()
        assert plugins_loader.is_enabled(state, "demo.py")
        # persisted, not just in memory
        assert plugins_loader.is_enabled(
            plugins_loader.load_state(), "demo.py"
        )

    def test_hash_mismatch_refuses_and_writes_nothing(
        self, tmp_path, monkeypatch
    ):
        _point_loader_at(monkeypatch, str(tmp_path))
        entry = _entry(b"# what was published\n")
        monkeypatch.setattr(
            plugins_api, "_get", lambda url: b"# what was served\n"
        )
        state = plugins_loader.load_state()

        with pytest.raises(ValueError, match="Integrity check failed"):
            plugins_loader.install(entry, state)

        assert not (tmp_path / "demo.py").exists()
        assert state["plugins"] == {}

    @pytest.mark.parametrize("digest", [None, "", "abc", "z" * 64])
    def test_missing_or_malformed_hash_refuses(
        self, tmp_path, monkeypatch, digest
    ):
        _point_loader_at(monkeypatch, str(tmp_path))
        content = b"# demo\n"
        monkeypatch.setattr(plugins_api, "_get", lambda url: content)
        entry = _entry(content, sha256=digest)
        if digest is None:
            del entry["sha256"]

        with pytest.raises(ValueError, match="integrity hash"):
            plugins_loader.install(entry, plugins_loader.load_state())

        assert not (tmp_path / "demo.py").exists()

    def test_unverified_entry_gets_its_own_status(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        entry = _entry(b"x", sha256="nope")
        status = plugins_loader.status_for(entry, plugins_loader.load_state())
        assert status == plugins_loader.UNVERIFIED

    def test_status_flags_hand_copied_registry_file(
        self, tmp_path, monkeypatch
    ):
        _point_loader_at(monkeypatch, str(tmp_path))
        content = b"# demo\n"
        (tmp_path / "demo.py").write_bytes(content)

        status = plugins_loader.status_for(
            _entry(content), plugins_loader.load_state()
        )

        assert status == plugins_loader.UNMANAGED

    def test_local_file_is_surfaced_in_merged_entries(
        self, tmp_path, monkeypatch
    ):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_plugin(str(tmp_path), "from_a_colleague.py", "render")

        state = plugins_loader.load_state()
        entries = plugins_loader.merged_entries([], state)

        assert [e["file"] for e in entries] == ["from_a_colleague.py"]
        assert entries[0]["_local"] is True
        assert (
            plugins_loader.status_for(entries[0], plugins_loader.load_state())
            == plugins_loader.LOCAL_ONLY
        )

    def test_manifest_file_is_not_duplicated_as_local(
        self, tmp_path, monkeypatch
    ):
        _point_loader_at(monkeypatch, str(tmp_path))
        content = b"# demo\n"
        (tmp_path / "demo.py").write_bytes(content)

        entries = plugins_loader.merged_entries(
            [_entry(content)], plugins_loader.load_state()
        )

        assert len(entries) == 1
        assert not entries[0].get("_local")


# ---------------------------------------------------------------------------
# Diff helper
# ---------------------------------------------------------------------------


class TestDiffSources:
    def test_reports_identical_sources(self):
        text = plugins_loader.diff_sources("a\n", "a\n", "p.py")
        assert "identical" in text

    def test_shows_added_and_removed_lines(self):
        text = plugins_loader.diff_sources("a\nb\n", "a\nc\n", "p.py")
        assert "-b" in text
        assert "+c" in text
