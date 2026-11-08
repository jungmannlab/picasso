"""
picasso.gui.app
~~~~~~~~~~~~~~~

The shared startup sequence of every Picasso GUI.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from PyQt6 import QtCore, QtGui, QtWidgets

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


#: Widgets whose value the mouse wheel changes; scroll bars excepted.
_WHEEL_INPUTS = (
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSlider,
)


def _takes_wheel(widget: QtWidgets.QWidget) -> bool:
    """True if the mouse wheel changes the value of ``widget``."""
    return isinstance(widget, _WHEEL_INPUTS) and not isinstance(
        widget, QtWidgets.QScrollBar
    )


def _in_scroll_area(widget: QtWidgets.QWidget) -> bool:
    """True if ``widget`` lies inside a scroll area of its window, e.g.,
    a scrollable dialog or a table."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QtWidgets.QAbstractScrollArea):
            return True
        if parent.isWindow():
            return False
        parent = parent.parentWidget()
    return False


def _no_wheel_focus(widget: QtWidgets.QWidget) -> None:
    """Let ``widget`` take the focus by click and tab, not by the
    wheel."""
    if widget.focusPolicy() == QtCore.Qt.FocusPolicy.WheelFocus:
        widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)


class _ScrollGuard(QtCore.QObject):
    """Application event filter that passes the wheel over an input
    without the focus on to the scroll area around it, see
    ``install_scroll_guard``. It reacts only to polish and wheel
    events."""

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QtCore.QEvent.Type.Polish:
            if _takes_wheel(obj):
                _no_wheel_focus(obj)
        elif kind == QtCore.QEvent.Type.Wheel:
            if (
                _takes_wheel(obj)
                and not obj.hasFocus()
                and _in_scroll_area(obj)
            ):
                # an ignored event goes on to the parents, up to the
                # scroll area
                event.ignore()
                return True
        return False


def install_scroll_guard(app: QtWidgets.QApplication) -> None:
    """Keep scrolling through a dialog from changing its inputs.

    A spin box, combo box or slider inside a scroll area (e.g., a
    scrollable dialog or a table) takes the mouse wheel only while it
    has the keyboard focus, i.e., after it was clicked or tabbed into;
    without the focus, the wheel scrolls the area around it. Inputs
    outside scroll areas keep reacting to the wheel.

    Qt gives the focus to a spin box when the wheel turns over it
    (``WheelFocus``), before any event filter sees the wheel, so the
    inputs take the focus by click and tab only (``StrongFocus``).
    Installing twice has no further effect.

    Parameters
    ----------
    app : QtWidgets.QApplication
        The application.
    """
    if app.findChild(_ScrollGuard) is not None:
        return
    # widgets polished before the filter was installed
    for widget in app.allWidgets():
        if _takes_wheel(widget):
            _no_wheel_focus(widget)
    app.installEventFilter(_ScrollGuard(app))


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
    # name inputs and icon-only buttons for screen readers
    from . import accessibility

    accessibility.install(app)
    install_scroll_guard(app)

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
    # the smoke test of the packages and installers (release/smoke_test.py)
    # starts each GUI with this set: a clean exit means the window opened
    if os.environ.get("PICASSO_QUIT_AFTER_START"):
        QtCore.QTimer.singleShot(0, app.quit)

    from ..updater import setup_gui_update_check

    setup_gui_update_check(window)

    # release memory while the GUI is left open but unused
    from . import idle

    watcher = idle.install(app)
    try:
        return app.exec()
    finally:
        if watcher is not None:
            watcher.stop()
        _delete_windows(app)


def _delete_windows(app: QtWidgets.QApplication) -> None:
    """Delete the windows once the event loop is over, while Qt and
    Python are intact. Left to PyQt's clean-up at interpreter exit, they
    are deleted in no fixed order, which crashed Picasso on quitting
    (seen on macOS)."""
    from . import toolbars

    for widget in app.topLevelWidgets():
        toolbars.clear_toolbars(widget)
        widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete.value
    )
