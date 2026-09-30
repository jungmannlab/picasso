"""
picasso.diagnostics
~~~~~~~~~~~~~~~~~~~

Keep errors visible in every install type, in particular in the one-click
(PyInstaller) builds.

The one-click GUIs are started from ``picassow.exe``, which is built with
PyInstaller's ``--windowed`` option and therefore has no console attached:
PyInstaller sets ``sys.stdout`` and ``sys.stderr`` to None. Everything that
writes to them then either vanishes silently (``print``) or raises
``AttributeError: 'NoneType' object has no attribute 'write'`` (``tqdm``,
which writes to ``sys.stderr``). Worse, uncaught exceptions in worker
threads disappear without a trace, because ``threading.excepthook``
returns early when ``sys.stderr`` is None - which is why the installer
build appears to "ignore" errors that are plainly reported when the same
code runs from a terminal.

This module fixes both halves of that problem:

* ``ensure_std_streams`` replaces None std streams with a file-backed
  stream (``~/.picasso/logs/picasso.log``) that never raises, so
  ``print``, ``tqdm`` and the interpreter's own traceback printing all
  keep working in windowed builds. It is called by the child processes of
  the process pools as well (see ``release/pyinstaller``).
* ``install_excepthooks`` routes every uncaught exception - main thread,
  worker threads and unraisable ones - to that log file and, optionally,
  to a callback. ``picasso.lib_qt.install_excepthook`` passes a callback
  that shows the traceback in a message box.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import datetime
import functools
import os
import sys
import threading
import traceback
from collections.abc import Callable

from picasso import user_config_dir
from picasso.version import __version__

#: Size (bytes) above which the log file is rotated to ``picasso.log.1``.
MAX_LOG_BYTES = 5 * 1024 * 1024

_lock = threading.Lock()
_log_file = None  # shared, lazily opened file object of the Picasso log
_streams_replaced = False  # True once None std streams were replaced


def log_dir() -> str:
    """Return the directory of the Picasso log (``~/.picasso/logs``).

    Kept next to the other per-user Picasso files so that it survives
    uninstalling the one-click build and is writable without admin
    rights.

    Returns
    -------
    path : str
        Directory of the Picasso log.
    """
    return os.path.join(user_config_dir(), "logs")


def log_path() -> str:
    """Return the path of the Picasso log file.

    Returns
    -------
    path : str
        ``~/.picasso/logs/picasso.log``.
    """
    return os.path.join(log_dir(), "picasso.log")


def _rotate(path: str) -> None:
    """Move ``path`` to ``path + '.1'`` if it grew past
    ``MAX_LOG_BYTES``."""
    try:
        if os.path.getsize(path) > MAX_LOG_BYTES:
            os.replace(path, path + ".1")
    except OSError:  # missing file, or another process rotating it
        pass


def _open_log_file():
    """Open (once) the shared Picasso log file in append mode.

    Returns
    -------
    file : file object or None
        Line-buffered text file, or None if the log location is not
        writable.
    """
    global _log_file
    with _lock:
        if _log_file is not None:
            return _log_file
        try:
            os.makedirs(log_dir(), exist_ok=True)
            path = log_path()
            _rotate(path)
            _log_file = open(
                path, "a", buffering=1, encoding="utf-8", errors="replace"
            )
        except OSError:
            return None
        try:
            now = datetime.datetime.now().isoformat(timespec="seconds")
            _log_file.write(
                f"\n--- Picasso {__version__}, pid {os.getpid()},"
                f" started {now} ---\n"
            )
        except (OSError, ValueError):
            pass
        return _log_file


class _LogStream:
    """Minimal text stream writing to the Picasso log file.

    Stands in for ``sys.stdout``/``sys.stderr`` in windowed (no console)
    builds, where PyInstaller sets both to None. Writing never raises:
    diagnostics must not be able to break the operation they document.

    Parameters
    ----------
    file : file object
        Open text file the output is appended to.
    """

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, file) -> None:
        self._file = file

    def write(self, text: str) -> int:
        try:
            self._file.write(text)
        except (OSError, ValueError):  # full disk, closed file, ...
            pass
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        try:
            self._file.flush()
        except (OSError, ValueError):
            pass

    def fileno(self) -> int:
        return self._file.fileno()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def close(self) -> None:
        """No-op: the log file is shared and stays open."""

    @property
    def closed(self) -> bool:
        return getattr(self._file, "closed", False)


def ensure_std_streams() -> bool:
    """Give the process usable ``sys.stdout``/``sys.stderr``.

    No-op unless one of them is None, which happens in PyInstaller's
    windowed builds (``picassow.exe``) and in the worker processes they
    spawn. The missing streams are replaced by a stream backed by
    ``~/.picasso/logs/picasso.log`` (or by the null device if that file
    cannot be opened), so that ``tqdm`` and friends stop raising and
    tracebacks end up somewhere a user can find them.

    Returns
    -------
    replaced : bool
        Whether any stream was replaced.
    """
    global _streams_replaced

    missing = [
        name for name in ("stdout", "stderr") if getattr(sys, name) is None
    ]
    if not missing:
        return False

    file = _open_log_file()
    if file is None:  # no writable log location: at least do not raise
        try:
            file = open(os.devnull, "w")
        except OSError:
            return False
    stream = _LogStream(file)
    for name in missing:
        setattr(sys, name, stream)
        if getattr(sys, f"__{name}__") is None:
            setattr(sys, f"__{name}__", stream)
    _streams_replaced = True
    return True


def log_message(message: str) -> None:
    """Append ``message`` to the Picasso log file, timestamped.

    Never raises.

    Parameters
    ----------
    message : str
        Text to log, e.g. a formatted traceback.
    """
    file = _open_log_file()
    if file is None:
        return
    now = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        file.write(f"[{now}] {message.rstrip()}\n")
        file.flush()
    except (OSError, ValueError):
        pass


def _handle_exception(
    message: str, report: Callable[[str], None] | None
) -> None:
    """Log ``message`` and, if given, pass it to ``report``.

    Exceptions raised by ``report`` are ignored.
    """
    log_message(message)
    if report is not None:
        try:
            report(message)
        except Exception:  # noqa: BLE001 - reporting must not raise
            pass


def _excepthook(
    type_,
    value,
    tback,
    report: Callable[[str], None] | None,
    default_excepthook: Callable,
) -> None:
    _handle_exception(
        "".join(traceback.format_exception(type_, value, tback)), report
    )
    if not _streams_replaced:  # console builds: keep printing there
        default_excepthook(type_, value, tback)


def _thread_excepthook(
    args,
    report: Callable[[str], None] | None,
    default_thread_excepthook: Callable,
) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "unknown"
    message = "".join(
        traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback
        )
    )
    _handle_exception(f"Exception in thread {name}:\n{message}", report)
    if not _streams_replaced:
        default_thread_excepthook(args)


def _unraisablehook(args, default_unraisablehook: Callable) -> None:
    message = "".join(
        traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback
        )
    )
    log_message(f"{args.err_msg or 'Unraisable exception'}:\n{message}")
    if not _streams_replaced:
        default_unraisablehook(args)


def install_excepthooks(report: Callable[[str], None] | None = None) -> None:
    """Log (and optionally report) every uncaught exception.

    Installs ``sys.excepthook``, ``threading.excepthook`` and
    ``sys.unraisablehook``. Without the threading hook, an exception in a
    worker thread is dropped without any output whenever ``sys.stderr``
    is None, as it is in the one-click windowed builds.

    Parameters
    ----------
    report : callable, optional
        Called with the formatted traceback, from whichever thread
        raised. ``picasso.lib_qt.install_excepthook`` passes a Qt signal
        emitter here, which marshals the traceback to the GUI thread.
        Exceptions raised by ``report`` are ignored.
    """
    # the interpreter's own hooks, not the currently installed ones, so
    # that installing twice (CLI entry point, then a GUI) does not chain
    # the handlers and log everything twice
    sys.excepthook = functools.partial(
        _excepthook, report=report, default_excepthook=sys.__excepthook__
    )
    threading.excepthook = functools.partial(
        _thread_excepthook,
        report=report,
        default_thread_excepthook=threading.__excepthook__,
    )
    sys.unraisablehook = functools.partial(
        _unraisablehook, default_unraisablehook=sys.__unraisablehook__
    )
