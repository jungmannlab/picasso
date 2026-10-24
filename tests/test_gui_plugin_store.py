"""The online plugin store dialog (``Plugins > Browse online plugins...``).

The registry is faked by patching ``picasso.plugins._get``, through which
every download goes (the manifest, the plugin sources), so the dialog runs
its real install, verify, enable and uninstall code against a temporary
plugins folder. The host window is a bare main window with a Plugins menu,
so installing or enabling a plugin really loads it into that menu.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import json
import os
import urllib.error

import pytest
from PyQt6 import QtGui, QtWidgets

from picasso import io, plugins
from picasso.gui import plugins_loader
from picasso.gui.plugins_loader import PluginStoreDialog

YES = QtWidgets.QMessageBox.StandardButton.Yes
NO = QtWidgets.QMessageBox.StandardButton.No


def _source(app: str, version: str = "1.0") -> bytes:
    """A GUI plugin that adds a menu entry naming itself when it runs."""
    return (
        "import os\n"
        "class Plugin:\n"
        "    def __init__(self, window):\n"
        f"        self.name = {app!r}\n"
        "        self.window = window\n"
        "    def execute(self):\n"
        "        name = os.path.splitext(os.path.basename(__file__))[0]\n"
        f"        self.window.plugin_menu.addAction(f'{{name}} v{version}')\n"
    ).encode()


class _Registry:
    """The online registry: a manifest and the files it lists."""

    def __init__(self):
        self.entries = []
        self.files = {}
        self.online = True

    def publish(self, plugin_id, app="render", version="1.0", **extra):
        content = extra.pop("content", None) or _source(app, version)
        path = f"plugins/{plugin_id}.py"
        self.files[f"{plugins.RAW_BASE}/{path}"] = content
        self.entries = [e for e in self.entries if e["id"] != plugin_id]
        entry = {
            "id": plugin_id,
            "display_name": plugin_id.title(),
            "app": app,
            "description": f"The {plugin_id} plugin",
            "version": version,
            "file": path,
            "sha256": plugins.sha256_bytes(content),
            "author": "Jane Doe",
        }
        entry.update(extra)
        self.entries.append(entry)
        return entry

    def serve(self, plugin_id, content):
        """Serve other bytes than the ones the manifest pins."""
        self.files[f"{plugins.RAW_BASE}/plugins/{plugin_id}.py"] = content

    def get(self, url):
        if not self.online:
            raise urllib.error.URLError("no network")
        if url == plugins.MANIFEST_URL:
            return json.dumps({"plugins": self.entries}).encode()
        return self.files[url]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    directory = tmp_path / "plugins"
    directory.mkdir()
    monkeypatch.setattr(io, "plugins_directory", lambda: str(directory))
    plugins.clear_module_cache()
    yield directory
    plugins.clear_module_cache()


@pytest.fixture
def registry(monkeypatch):
    registry = _Registry()
    monkeypatch.setattr(plugins, "_get", registry.get)
    return registry


@pytest.fixture
def host(qt_offscreen):
    window = QtWidgets.QMainWindow()
    window.plugin_menu = window.menuBar().addMenu("Plugins")
    yield window
    window.close()


@pytest.fixture
def store(host, folder, registry):
    """Open the store for Render; call it after publishing."""

    def open_store(app="render"):
        return PluginStoreDialog(host, app)

    return open_store


class _Answers:
    """Answer the dialog's message boxes and record what was asked."""

    def __init__(self, monkeypatch, reply=YES):
        # one reply for all, or {kind: reply}
        self.reply = reply
        self.asked = []
        for kind in ("warning", "question", "critical", "information"):
            monkeypatch.setattr(
                QtWidgets.QMessageBox, kind, self._recorder(kind)
            )

    def _recorder(self, kind):
        def answer(parent, title, text, *args, **kwargs):
            self.asked.append((kind, title, text))
            if isinstance(self.reply, dict):
                return self.reply.get(kind, YES)
            return self.reply

        return answer

    @property
    def titles(self):
        return [title for _, title, _ in self.asked]


@pytest.fixture
def answers(monkeypatch):
    return _Answers(monkeypatch)


@pytest.fixture
def shown(monkeypatch):
    """What the source viewer would show: (title, header, text)."""
    views = []

    def fake_exec(dialog):
        label = dialog.findChild(QtWidgets.QLabel)
        text = dialog.findChild(QtWidgets.QPlainTextEdit).toPlainText()
        views.append((dialog.windowTitle(), label.text(), text))
        return 0

    monkeypatch.setattr(plugins_loader.SourceViewDialog, "exec", fake_exec)
    return views


# -- reading the table -------------------------------------------------------


def _names(dialog):
    table = dialog.table
    return [table.item(row, 0).text() for row in range(table.rowCount())]


def _row(dialog, name):
    return _names(dialog).index(name)


def _status(dialog, name):
    return dialog.table.item(_row(dialog, name), 3).text()


def _buttons(dialog, name):
    widget = dialog.table.cellWidget(_row(dialog, name), 5)
    return {
        button.text(): button
        for button in widget.findChildren(QtWidgets.QPushButton)
    }


def _click(dialog, name, label):
    _buttons(dialog, name)[label].click()


def _checkbox(dialog, name):
    widget = dialog.table.cellWidget(_row(dialog, name), 4)
    return widget.findChild(QtWidgets.QCheckBox)


def _menu(host):
    return [a.text() for a in host.plugin_menu.actions() if a.text()]


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing:
    def test_shows_this_apps_plugins_and_api_plugins(self, store, registry):
        registry.publish("drift", app="render")
        registry.publish("tracker", app="localize")
        registry.publish("helpers", app=plugins.API_APP)
        dialog = store("render")
        assert _names(dialog) == ["Drift", "Helpers"]

    def test_show_all_lists_every_app(self, store, registry):
        registry.publish("drift", app="render")
        registry.publish("tracker", app="localize")
        dialog = store("render")
        dialog.show_all.setChecked(True)
        assert _names(dialog) == ["Drift", "Tracker"]
        dialog.show_all.setChecked(False)
        assert _names(dialog) == ["Drift"]

    def test_the_name_tooltip_says_who_and_which_version(
        self, store, registry
    ):
        entry = registry.publish("drift", version="2.1")
        dialog = store()
        tip = dialog.table.item(0, 0).toolTip()
        assert "Author: Jane Doe" in tip
        assert "Latest version: 2.1" in tip
        assert entry["sha256"] in tip

    def test_an_unreachable_registry_is_explained(self, store, registry):
        registry.publish("drift")
        registry.online = False
        dialog = store()
        assert "Could not reach" in dialog.status_label.text()
        assert _names(dialog) == []
        # Refresh tries again
        registry.online = True
        dialog._refresh()
        assert "Could not reach" not in dialog.status_label.text()
        assert _names(dialog) == ["Drift"]

    def test_unsafe_entries_are_not_listed(self, store, registry):
        registry.publish("drift")
        registry.publish("evil", file="../../evil.py")
        assert _names(store()) == ["Drift"]

    def test_open_repository(self, store, registry, monkeypatch):
        opened = []
        monkeypatch.setattr(
            QtGui.QDesktopServices,
            "openUrl",
            lambda url: opened.append(url.toString()),
        )
        dialog = store()
        (button,) = [
            b
            for b in dialog.findChildren(QtWidgets.QPushButton)
            if b.text() == "Open repository..."
        ]
        button.click()
        assert opened == [plugins.REPO_URL]


class TestStatuses:
    def test_not_installed(self, store, registry):
        registry.publish("drift")
        dialog = store()
        assert _status(dialog, "Drift") == "Not installed"
        assert set(_buttons(dialog, "Drift")) == {"Install", "View source"}
        # nothing on disk to enable
        assert _checkbox(dialog, "Drift") is None

    def test_without_a_hash_it_cannot_be_installed(self, store, registry):
        registry.publish("drift", sha256=None)
        dialog = store()
        assert _status(dialog, "Drift") == "No integrity hash"
        assert _buttons(dialog, "Drift") == {}

    def test_too_new_for_this_picasso(self, store, registry):
        registry.publish("drift", min_picasso_version="99.0")
        dialog = store()
        assert _status(dialog, "Drift") == "Requires newer Picasso"
        assert _buttons(dialog, "Drift") == {}


# ---------------------------------------------------------------------------
# Install, update, uninstall
# ---------------------------------------------------------------------------


class TestInstall:
    def test_installs_enables_and_loads(
        self, store, registry, folder, host, answers
    ):
        registry.publish("drift")
        dialog = store()
        _click(dialog, "Drift", "Install")

        assert answers.titles == ["Install plugin?"]
        assert (folder / "drift.py").read_bytes() == _source("render")
        assert _status(dialog, "Drift") == "Installed"
        assert _checkbox(dialog, "Drift").isChecked()
        assert "drift v1.0" in _menu(host)

    def test_the_trust_warning_is_asked_once(self, store, registry, answers):
        registry.publish("drift")
        registry.publish("grid")
        dialog = store()
        _click(dialog, "Drift", "Install")
        _click(dialog, "Grid", "Install")
        assert answers.titles == ["Install plugin?"]
        # remembered across dialogs
        _click(store(), "Drift", "Uninstall")
        _click(store(), "Drift", "Install")
        assert answers.titles == ["Install plugin?", "Uninstall plugin?"]

    def test_declining_the_trust_warning_installs_nothing(
        self, store, registry, folder, monkeypatch
    ):
        answers = _Answers(monkeypatch, reply=NO)
        registry.publish("drift")
        dialog = store()
        _click(dialog, "Drift", "Install")
        assert answers.titles == ["Install plugin?"]
        assert not (folder / "drift.py").exists()
        assert _status(dialog, "Drift") == "Not installed"
        assert not plugins.load_state()["trust_acknowledged"]

    def test_a_tampered_download_is_refused(
        self, store, registry, folder, answers
    ):
        registry.publish("drift")
        registry.serve("drift", b"import os  # not what was published\n")
        dialog = store()
        _click(dialog, "Drift", "Install")
        assert answers.titles == ["Install plugin?", "Installation failed"]
        assert "Integrity check failed" in answers.asked[-1][2]
        assert not (folder / "drift.py").exists()
        assert _status(dialog, "Drift") == "Not installed"

    def test_a_hand_copied_file_is_replaced_only_when_confirmed(
        self, store, registry, folder, monkeypatch
    ):
        registry.publish("drift")
        (folder / "drift.py").write_bytes(b"# my own edits\n")
        dialog = store()
        assert _status(dialog, "Drift") == "On disk, not installed by Picasso"
        assert set(_buttons(dialog, "Drift")) == {"Install", "View source"}

        answers = _Answers(monkeypatch, reply={"question": NO})
        _click(dialog, "Drift", "Install")
        assert answers.titles == ["Install plugin?", "Replace existing file?"]
        assert (folder / "drift.py").read_bytes() == b"# my own edits\n"

        answers.reply = YES
        _click(dialog, "Drift", "Install")
        assert (folder / "drift.py").read_bytes() == _source("render")
        assert _status(dialog, "Drift") == "Installed"


class TestUpdate:
    @pytest.fixture
    def outdated(self, store, registry, answers):
        registry.publish("drift", version="1.0")
        _click(store(), "Drift", "Install")
        registry.publish("drift", version="2.0")
        return store()

    def test_offers_the_update(self, outdated):
        assert _status(outdated, "Drift") == "Update available (1.0 → 2.0)"
        assert set(_buttons(outdated, "Drift")) == {
            "Update",
            "View changes",
            "Uninstall",
        }

    def test_shows_what_changes(self, outdated, shown):
        _click(outdated, "Drift", "View changes")
        ((title, header, diff),) = shown
        assert title == "Changes to drift.py"
        assert header == "1.0 → 2.0"
        line = "        self.window.plugin_menu.addAction(f'{name} v%s')"
        assert "-" + line % "1.0" in diff
        assert "+" + line % "2.0" in diff

    def test_updates_and_reloads(self, outdated, folder, host):
        _click(outdated, "Drift", "Update")
        assert (folder / "drift.py").read_bytes() == _source("render", "2.0")
        assert _status(outdated, "Drift") == "Installed"
        assert "drift v2.0" in _menu(host)
        assert "drift v1.0" not in _menu(host)


class TestUninstall:
    def test_removes_the_file_and_unloads_it(
        self, store, registry, folder, host, answers
    ):
        registry.publish("drift")
        dialog = store()
        _click(dialog, "Drift", "Install")
        _click(dialog, "Drift", "Uninstall")
        assert answers.titles == ["Install plugin?", "Uninstall plugin?"]
        assert "including any changes" in answers.asked[-1][2]
        assert not (folder / "drift.py").exists()
        assert _status(dialog, "Drift") == "Not installed"
        assert "drift v1.0" not in _menu(host)
        assert plugins.load_state()["plugins"] == {}

    def test_declining_keeps_the_plugin(
        self, store, registry, folder, host, answers
    ):
        registry.publish("drift")
        dialog = store()
        _click(dialog, "Drift", "Install")
        answers.reply = {"question": NO}
        _click(dialog, "Drift", "Uninstall")
        assert answers.titles == ["Install plugin?", "Uninstall plugin?"]
        assert (folder / "drift.py").exists()
        assert _status(dialog, "Drift") == "Installed"
        assert "drift v1.0" in _menu(host)
        assert "drift" in plugins.load_state()["plugins"]

    def test_a_plugin_dropped_from_the_registry_can_still_be_removed(
        self, store, registry, folder, answers
    ):
        registry.publish("drift")
        _click(store(), "Drift", "Install")
        registry.entries = []
        dialog = store()
        assert _status(dialog, "Drift") == "Installed (unlisted)"
        _click(dialog, "Drift", "Uninstall")
        # it cannot be downloaded again, and the question says so
        assert "cannot be installed again" in answers.asked[-1][2]
        assert not (folder / "drift.py").exists()
        assert _names(dialog) == []

    def test_offline_it_is_not_called_unlisted(self, store, registry, answers):
        """Not known to be gone from the registry, see TestOffline."""
        registry.publish("drift")
        _click(store(), "Drift", "Install")
        registry.online = False
        _click(store(), "Drift", "Uninstall")
        assert "cannot be installed again" not in answers.asked[-1][2]


# ---------------------------------------------------------------------------
# Enabling, and files that did not come from the registry
# ---------------------------------------------------------------------------


class TestLocalFiles:
    @pytest.fixture
    def local(self, folder, store):
        (folder / "mine.py").write_bytes(_source("render"))
        return store()

    def test_listed_but_not_enabled(self, local, host):
        assert _status(local, "mine.py") == "Local file"
        assert set(_buttons(local, "mine.py")) == {"View source"}
        assert not _checkbox(local, "mine.py").isChecked()
        assert "mine v1.0" not in _menu(host)

    def test_enabling_warns_and_can_be_declined(
        self, local, host, monkeypatch
    ):
        answers = _Answers(monkeypatch, reply=NO)
        _checkbox(local, "mine.py").click()
        assert answers.titles == ["Enable plugin?"]
        assert not _checkbox(local, "mine.py").isChecked()
        assert not plugins.is_enabled(plugins.load_state(), "mine.py")
        assert "mine v1.0" not in _menu(host)

    def test_enabling_loads_it_and_disabling_unloads_it(
        self, local, host, answers
    ):
        _checkbox(local, "mine.py").click()
        assert answers.titles == ["Enable plugin?"]
        assert plugins.is_enabled(plugins.load_state(), "mine.py")
        assert "mine v1.0" in _menu(host)

        _checkbox(local, "mine.py").click()  # no question to disable
        assert answers.titles == ["Enable plugin?"]
        assert not plugins.is_enabled(plugins.load_state(), "mine.py")
        assert "mine v1.0" not in _menu(host)

    def test_view_source(self, local, folder, shown):
        _click(local, "mine.py", "View source")
        ((title, header, text),) = shown
        assert title == "Source of mine.py"
        assert header == str(folder / "mine.py")
        assert text == _source("render").decode()

    def test_a_vanished_file_is_reported(self, local, folder, shown, answers):
        os.remove(folder / "mine.py")
        _click(local, "mine.py", "View source")
        assert shown == []
        assert answers.titles == ["Could not show source"]

    def test_re_enabling_a_registry_install_does_not_ask(
        self, store, registry, answers
    ):
        registry.publish("drift")
        dialog = store()
        _click(dialog, "Drift", "Install")
        _checkbox(dialog, "Drift").click()
        _checkbox(dialog, "Drift").click()
        assert answers.titles == ["Install plugin?"]
        assert _checkbox(dialog, "Drift").isChecked()


class TestOffline:
    def test_installed_plugins_are_not_called_unlisted(
        self, store, registry, answers
    ):
        """Without the registry nothing is known about what it lists, so an
        installed plugin must not be shown as dropped from it."""
        registry.publish("drift")
        _click(store(), "Drift", "Install")
        registry.online = False
        dialog = store()
        row = _row(dialog, "Drift")
        assert _status(dialog, "Drift") == "Installed"
        assert "no longer" not in dialog.table.item(row, 2).text()
        # it can still be looked at, disabled and uninstalled
        assert set(_buttons(dialog, "Drift")) == {"View source", "Uninstall"}
        assert _checkbox(dialog, "Drift").isChecked()


class TestFromThePluginsMenu:
    def test_installing_rebuilds_the_menu_it_was_opened_from(
        self, host, registry, folder, answers, monkeypatch
    ):
        """The store is opened by an action of the Plugins menu, and an
        install rebuilds that menu while the action is still emitting."""
        registry.publish("drift")
        plugins_loader.add_plugins_menu_actions(host, "render")

        def exec_and_install(dialog):
            _click(dialog, "Drift", "Install")
            return 0

        monkeypatch.setattr(PluginStoreDialog, "exec", exec_and_install)
        (browse,) = [
            a
            for a in host.plugin_menu.actions()
            if a.text() == "Browse online plugins..."
        ]
        browse.trigger()
        QtWidgets.QApplication.processEvents()
        assert _menu(host)[:2] == ["drift v1.0", "Browse online plugins..."]


class TestViewRemoteSource:
    def test_shows_the_verified_source(self, store, registry, shown):
        entry = registry.publish("drift")
        _click(store(), "Drift", "View source")
        ((title, header, text),) = shown
        assert title == "Source of Drift"
        assert entry["sha256"] in header
        assert "matches the registry" in header
        assert text == _source("render").decode()

    def test_a_tampered_source_is_not_shown(
        self, store, registry, shown, answers
    ):
        registry.publish("drift")
        registry.serve("drift", b"# something else\n")
        _click(store(), "Drift", "View source")
        assert shown == []
        assert answers.titles == ["Could not show source"]
        assert "Integrity check failed" in answers.asked[0][2]
