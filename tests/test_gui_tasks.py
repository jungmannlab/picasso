"""Cancelable GUI tasks: ``picasso.lib_qt.run_task``.

A task runs a function on a worker thread behind a progress dialog with a
Cancel button. The function's progress updates are the cancellation
points, results are applied on the GUI thread and user input is blocked
while the task runs.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
import time

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets, sip

from picasso import lib, lib_qt


def _wait_until(qapp, condition, timeout: float = 10.0) -> None:
    start = time.monotonic()
    while not condition():
        qapp.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("task did not end")


@pytest.fixture
def parent(qt_offscreen):
    widget = QtWidgets.QWidget()
    widget.show()
    yield widget
    widget.close()


def test_result_is_delivered_on_gui_thread(qt_offscreen, parent):
    results = []
    threads = []

    def fn(progress):
        threads.append(threading.current_thread())
        for i in progress.get_iterator(0, 10):
            progress.set_value(i + 1)
        return 42

    def on_finished(result):
        results.append((result, threading.current_thread()))

    task = lib.run_task(fn, "Working", parent, on_finished, maximum=10)
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "finished"
    assert results == [(42, threading.main_thread())]
    assert threads[0] is not threading.main_thread()
    assert lib_qt._running_tasks == []


def test_cancel_stops_at_next_progress_update(qt_offscreen, parent):
    started = threading.Event()
    steps = []
    calls = []

    def fn(progress):
        started.set()
        for i in range(1000):
            steps.append(i)
            progress.set_value(i)
            time.sleep(0.002)
        return "done"

    task = lib.run_task(
        fn,
        "Working",
        parent,
        on_finished=lambda r: calls.append("finished"),
        on_canceled=lambda: calls.append("canceled"),
        maximum=1000,
    )
    started.wait(5)
    # as if the Cancel button was clicked
    task.dialog.canceled.emit()
    assert task.dialog.cancel_was_requested
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "canceled"
    assert calls == ["canceled"]
    assert len(steps) < 1000


def test_late_result_after_cancel_is_discarded(qt_offscreen, parent):
    """A function without further progress updates still completes; its
    result must not be applied once the user canceled."""
    release = threading.Event()
    calls = []

    def fn(progress):
        release.wait(5)
        return "late"

    task = lib.run_task(
        fn,
        "Working",
        parent,
        on_finished=lambda r: calls.append(r),
        on_canceled=lambda: calls.append("canceled"),
    )
    task.cancel()
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert calls == ["canceled"]


def test_error_goes_to_on_failed(qt_offscreen, parent):
    errors = []

    def fn(progress):
        raise ValueError("bad input")

    task = lib.run_task(fn, "Working", parent, on_failed=errors.append)
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "failed"
    assert isinstance(errors[0], ValueError)


def test_escape_and_close_cancel_instead_of_hiding(qt_offscreen, parent):
    release = threading.Event()

    def fn(progress):
        while not release.is_set():
            progress.set_value(0)
            time.sleep(0.005)

    task = lib.run_task(fn, "Working", parent, maximum=10)
    task.dialog.show()
    task.dialog.reject()  # Escape
    assert task.progress.canceled
    task.dialog.close()
    assert task.dialog.isVisible()  # only the task closes the dialog
    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert task.outcome == "canceled"
    assert sip.isdeleted(task.dialog) or not task.dialog.isVisible()


def _click(button: QtWidgets.QPushButton) -> None:
    """Click through the event path user input takes (unlike
    ``QPushButton.click``, which the input filter cannot see)."""
    center = button.rect().center()
    for kind in (
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QEvent.Type.MouseButtonRelease,
    ):
        event = QtGui.QMouseEvent(
            kind,
            QtCore.QPointF(center),
            QtCore.QPointF(button.mapToGlobal(center)),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(button, event)


def test_input_is_blocked_while_running(qt_offscreen, parent):
    button = QtWidgets.QPushButton("Other action", parent)
    button.show()
    clicks = []
    button.clicked.connect(lambda: clicks.append(1))
    release = threading.Event()

    task = lib.run_task(lambda progress: release.wait(5), "Working", parent)
    _click(button)
    assert clicks == []
    # nor can the window be closed under the task
    parent.close()
    assert parent.isVisible()

    release.set()
    _wait_until(qt_offscreen, lambda: not task.is_running())
    _click(button)
    assert clicks == [1]


def test_phases_and_callbacks_drive_one_dialog(qt_offscreen, parent):
    seen = []

    def fn(progress):
        first = progress.callback("First phase", 4)
        second = progress.callback("Second phase", 2)
        for i in range(4):
            first(i + 1)
        for i in range(2):
            second(i + 1)
        progress.zero_progress("Third phase")
        progress.setMaximum(0)

    task = lib.run_task(fn, "Start", parent, maximum=0)
    task.progress.phase_changed.connect(seen.append)
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert seen == ["First phase", "Second phase", "Third phase"]


def test_task_progress_works_with_progress_consumers(qt_offscreen, parent):
    """Analysis functions normalize their progress argument; a
    TaskProgress must pass through unchanged."""
    results = []

    def fn(progress):
        assert lib.normalize_progress(progress) is progress
        progress.setMaximum(5)
        for i in progress.get_iterator(0, 5):
            progress.set_value(i + 1)
        progress.close()
        return progress.value()

    task = lib.run_task(fn, "Working", parent, results.append)
    _wait_until(qt_offscreen, lambda: not task.is_running())
    assert results == [5]


def test_callback_can_start_the_next_task(qt_offscreen, parent):
    order = []

    def second(result):
        order.append(result)

    def first(result):
        order.append(result)
        lib.run_task(lambda p: "second", "Second", parent, second)

    lib.run_task(lambda p: "first", "First", parent, first)
    _wait_until(qt_offscreen, lambda: len(order) == 2)
    _wait_until(qt_offscreen, lambda: not lib_qt._running_tasks)
    assert order == ["first", "second"]
