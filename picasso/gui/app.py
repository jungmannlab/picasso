"""
picasso.gui.app
~~~~~~~~~~~~~~~

The shared startup sequence of every Picasso GUI.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from PyQt6 import QtGui, QtWidgets

from picasso import diagnostics, lib


def _load_plugins(window: QtWidgets.QWidget, name: str) -> None:
    """Load the user plugins of the GUI ``name`` and add its plugin menu
    actions.

    A failure here is reported and swallowed: a broken plugin must not
    stop Picasso from starting.

    Parameters
    ----------
    window : QtWidgets.QWidget
        The GUI main window, passed on to the plugins.
    name : str
        The GUI the plugins are filtered by, e.g. ``"render"``.
    """
    # imported here (as in the GUIs before) to keep module import light
    from .plugins_loader import add_plugins_menu_actions, load_plugins

    try:
        load_plugins(window, name)
        add_plugins_menu_actions(window, name)
    except Exception:  # noqa: BLE001 - plugins must not block startup
        sys.excepthook(*sys.exc_info())


def _apply_theme(app: QtWidgets.QApplication) -> None:
    """Apply the appearance saved in the user settings, before the
    main window is built so that it is laid out in the theme's fonts
    and spacing.

    A failure here is reported and swallowed: Picasso then starts in the
    platform's own look.

    Parameters
    ----------
    app : QtWidgets.QApplication
        The application.
    """
    from . import theme

    try:
        theme.apply(app, theme.current())
    except Exception:  # noqa: BLE001 - the look must not block startup
        sys.excepthook(*sys.exc_info())


def run_gui(
    window_factory: Callable[[], QtWidgets.QWidget],
    name: str | None = None,
    icon: str | None = None,
) -> int:
    """Start a Picasso GUI and run its event loop until it quits.

    Parameters
    ----------
    window_factory : callable
        Called with no arguments to build the main window, e.g. the
        module's ``Window`` class.
    name : str, optional
        Name the user plugins are filtered by, e.g. ``"render"``. None
        (default) loads no plugins, for a GUI that does not support them.
    icon : str, optional
        Path to an application icon (``.ico``). None (default) leaves the
        icon to the window itself.

    Returns
    -------
    exit_code : int
        Exit code of the Qt event loop, or 1 if the window could not be
        built. Meant to be passed straight to ``sys.exit``.
    """
    # before Qt: in the one-click builds there is no console, so give the
    # process usable streams and start logging to ~/.picasso/logs
    diagnostics.ensure_std_streams()
    diagnostics.install_excepthooks()

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    if icon is not None:
        app.setWindowIcon(QtGui.QIcon(icon))

    # report errors in a message box from here on, i.e. already while the
    # window is being built; it is reinstalled below to parent the box on
    # the window once there is one
    lib.install_excepthook()

    _apply_theme(app)

    try:
        window = window_factory()
    except Exception:  # noqa: BLE001 - reported, then Picasso gives up
        sys.excepthook(*sys.exc_info())
        return 1

    # from here on the error box hangs on the main window
    lib.install_excepthook(window)

    if name is not None:
        _load_plugins(window, name)

    window.show()

    from ..updater import setup_gui_update_check

    setup_gui_update_check(window)

    return app.exec()
