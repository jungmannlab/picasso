#!/usr/bin/env python3
import sys
import os
import os.path as op

try:
    import picasso
except ImportError:
    print(
        "This script must be run within an environment "
        "in which picasso is installed!",
        file=sys.stderr,
    )
    raise

SUBCMD = ("average", "design", "filter", "localize", "render", "simulate", "spinna")
SCRIPT_PATH_ROOT = (os.sep, "usr", "bin", "picasso")
DESKTOP_PATH_ROOT = (os.sep, "usr", "share", "applications", "picasso_{subcmd}.desktop")
SCRIPT_PATH_USER = ("~", "bin", "picasso")
DESKTOP_PATH_USER = ("~", ".local", "share", "applications", "picasso_{subcmd}.desktop")

DESKTOP_TEMPLATE = """[Desktop Entry]
Name=Picasso {subcmd_cap}
Exec={exec_path} -m picasso {subcmd}
Terminal=false
Type=Application
Icon={icon_path}
Categories=Education;
"""

SCRIPT_TEMPLATE = """#!{exec_path}
if __name__ == "__main__":
    from picasso.__main__ import main
    main()
"""


def main(exec_path=None, icon_path=None, script_path=None, desktop_path=None):
    if exec_path is None:
        exec_path = sys.executable
    if icon_path is None:
        import picasso.gui

        icon_path = op.join(op.dirname(picasso.gui.__file__), "icons")
    if os.geteuid() == 0:
        if script_path is None:
            script_path = op.join(*SCRIPT_PATH_ROOT)
        if desktop_path is None:
            desktop_path = op.join(*DESKTOP_PATH_ROOT)
    else:
        if script_path is None:
            script_path = op.expanduser(op.join(*SCRIPT_PATH_USER))
        if desktop_path is None:
            desktop_path = op.expanduser(op.join(*DESKTOP_PATH_USER))

    print("Writing files:")
    with open(script_path, "xt") as f:
        f.write(SCRIPT_TEMPLATE.format(exec_path=exec_path))
    print(script_path)
    os.chmod(script_path, 0o755)

    for subcmd in SUBCMD:
        icon_file = op.join(icon_path, f"{subcmd}.ico")
        desktop_file = op.join(desktop_path.format(subcmd=subcmd))
        with open(desktop_file, "xt") as f:
            f.write(
                DESKTOP_TEMPLATE.format(
                    subcmd=subcmd,
                    subcmd_cap=subcmd.capitalize(),
                    exec_path=exec_path,
                    icon_path=icon_file,
                )
            )
        print(desktop_file)
        os.chmod(desktop_file, 0o755)


if __name__ == "__main__":
    if sys.platform != "linux":
        raise RuntimeError("Other operating system than Linux detected.")
    main()