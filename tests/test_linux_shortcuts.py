"""``release/create_linux_shortcuts.py`` adds Picasso's modules to the
application menu of a Linux desktop: it writes a menu entry per module
and a ``picasso`` command, replaces them when run again and removes
them with ``--remove``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "release"
    / "create_linux_shortcuts.py"
)


@pytest.fixture(scope="module")
def shortcuts():
    """The release script, imported as a module."""
    spec = importlib.util.spec_from_file_location("shortcuts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def folders(tmp_path):
    # not created yet, as ~/.local/bin on a new account
    return str(tmp_path / "bin"), str(tmp_path / "applications")


def _entries(desktop_dir: str) -> dict[str, str]:
    return {
        path.name: path.read_text()
        for path in sorted(Path(desktop_dir).glob("*.desktop"))
    }


def test_writes_an_entry_per_module_and_the_command(shortcuts, folders):
    script_dir, desktop_dir = folders
    written = shortcuts.create(script_dir, desktop_dir, "/env/bin/python")
    script = Path(script_dir, "picasso")
    assert str(script) in written
    assert script.read_text().startswith("#!/env/bin/python\n")
    assert os.access(script, os.X_OK) or sys.platform == "win32"
    entries = _entries(desktop_dir)
    for module in ("design", "localize", "render", "spinna", "nanotron"):
        assert f"picasso_{module}.desktop" in entries
    spinna = entries["picasso_spinna.desktop"]
    assert "Name=Picasso SPINNA\n" in spinna
    assert 'Exec="/env/bin/python" -m picasso spinna\n' in spinna
    assert "Terminal=false\n" in spinna
    assert spinna.splitlines()[0] == "[Desktop Entry]"
    icon = spinna.split("Icon=")[1].splitlines()[0]
    assert os.path.isfile(icon), icon


def test_server_only_with_its_extra(shortcuts, folders, monkeypatch):
    script_dir, desktop_dir = folders
    real_find_spec = importlib.util.find_spec
    for installed in (False, True):
        monkeypatch.setattr(
            shortcuts.importlib.util,
            "find_spec",
            lambda name, *args, installed=installed: (
                (object() if installed else None)
                if name == "streamlit"
                else real_find_spec(name, *args)
            ),
        )
        shortcuts.remove(script_dir, desktop_dir)
        shortcuts.create(script_dir, desktop_dir)
        entries = _entries(desktop_dir)
        assert ("picasso_server.desktop" in entries) is installed
    # the server shows its address in a terminal
    assert "Terminal=true\n" in entries["picasso_server.desktop"]


def test_running_again_replaces_the_files(shortcuts, folders):
    script_dir, desktop_dir = folders
    shortcuts.create(script_dir, desktop_dir, "/old/bin/python")
    shortcuts.create(script_dir, desktop_dir, "/new/bin/python")
    assert "/new/bin/python" in Path(script_dir, "picasso").read_text()
    render = _entries(desktop_dir)["picasso_render.desktop"]
    assert '"/new/bin/python"' in render
    assert "/old/" not in render


def test_remove_deletes_only_its_files(shortcuts, folders):
    script_dir, desktop_dir = folders
    shortcuts.create(script_dir, desktop_dir)
    other = Path(desktop_dir, "other.desktop")
    other.write_text("[Desktop Entry]\n")
    removed = shortcuts.remove(script_dir, desktop_dir)
    assert str(Path(script_dir, "picasso")) in removed
    assert list(Path(desktop_dir).iterdir()) == [other]
    assert shortcuts.remove(script_dir, desktop_dir) == []
