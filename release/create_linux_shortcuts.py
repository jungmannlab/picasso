#!/usr/bin/env python3
"""Add Picasso's modules to the application menu of a Linux desktop
(GNOME, KDE, Xfce, ...), for Picasso installed with pip, e.g., in a
conda environment. Run it with the Python of that environment::

    conda activate picasso
    python create_linux_shortcuts.py

It writes, for the current user (run as root: for all users):

- a menu entry for each module (Design, Simulate, Localize, Filter,
  Render, Average, SPINNA, Nanotron and, with ``picassosr[server]``,
  Server) to ``~/.local/share/applications``
  (``/usr/share/applications``). Each starts the module with the Python
  of this environment, which therefore need not be activated;
- a ``picasso`` command to ``~/.local/bin`` (``/usr/local/bin``), which
  runs this environment's Picasso from any terminal.

Running it again, e.g., for a new environment, replaces these files;
``--remove`` deletes them.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

#: the modules with a menu entry, and their names in the menu
MODULES = {
    "design": "Design",
    "simulate": "Simulate",
    "localize": "Localize",
    "filter": "Filter",
    "render": "Render",
    "average": "Average",
    "spinna": "SPINNA",
    "nanotron": "Nanotron",
    "server": "Server",
}
#: the modules that run in a terminal window: the server shows its
#: address there and runs until the window is closed
TERMINAL = {"server"}

DESKTOP_TEMPLATE = """[Desktop Entry]
Type=Application
Name=Picasso {name}
Exec="{exec_path}" -m picasso {module}
Icon={icon}
Terminal={terminal}
Categories=Science;Education;
"""

SCRIPT_TEMPLATE = """#!{exec_path}
if __name__ == "__main__":
    from picasso.__main__ import main

    main()
"""


def default_paths() -> tuple[str, str]:
    """The folders of the ``picasso`` command and of the menu entries:
    the user's, or, for root, the system's."""
    if os.geteuid() == 0:
        return "/usr/local/bin", "/usr/share/applications"
    return (
        os.path.expanduser("~/.local/bin"),
        os.path.expanduser("~/.local/share/applications"),
    )


def modules() -> list[str]:
    """The modules to add: Server only if its extra is installed."""
    server = importlib.util.find_spec("streamlit") is not None
    return [name for name in MODULES if name != "server" or server]


def desktop_file(desktop_dir: str, module: str) -> str:
    return os.path.join(desktop_dir, f"picasso_{module}.desktop")


def create(
    script_dir: str,
    desktop_dir: str,
    exec_path: str = sys.executable,
) -> list[str]:
    """Write the ``picasso`` command and the menu entries, replacing
    earlier ones; return the paths written."""
    import picasso.gui

    icons = os.path.join(os.path.dirname(picasso.gui.__file__), "icons")
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(desktop_dir, exist_ok=True)
    script = os.path.join(script_dir, "picasso")
    with open(script, "w") as file:
        file.write(SCRIPT_TEMPLATE.format(exec_path=exec_path))
    os.chmod(script, 0o755)
    written = [script]
    for module in modules():
        path = desktop_file(desktop_dir, module)
        with open(path, "w") as file:
            file.write(
                DESKTOP_TEMPLATE.format(
                    name=MODULES[module],
                    exec_path=exec_path,
                    module=module,
                    icon=os.path.join(icons, f"{module}.ico"),
                    terminal="true" if module in TERMINAL else "false",
                )
            )
        os.chmod(path, 0o755)
        written.append(path)
    return written


def remove(script_dir: str, desktop_dir: str) -> list[str]:
    """Delete the files ``create`` writes; return the paths deleted."""
    paths = [os.path.join(script_dir, "picasso")]
    paths += [desktop_file(desktop_dir, module) for module in MODULES]
    removed = []
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--remove",
        action="store_true",
        help="delete the menu entries and the picasso command",
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        sys.exit("This script is for Linux only.")
    if importlib.util.find_spec("picasso") is None:
        sys.exit(
            "Picasso is not installed for this Python. Activate the "
            "environment with Picasso and run the script again."
        )
    script_dir, desktop_dir = default_paths()
    if args.remove:
        paths = remove(script_dir, desktop_dir)
        print("Deleted:" if paths else "Nothing to delete.", *paths, sep="\n")
        return
    print("Wrote:", *create(script_dir, desktop_dir), sep="\n")
    if script_dir not in os.environ.get("PATH", "").split(os.pathsep):
        print(
            f"\n{script_dir} is not on PATH: many distributions add it "
            "when you log in again; otherwise, add it to PATH to use the "
            "picasso command."
        )


if __name__ == "__main__":
    main()
