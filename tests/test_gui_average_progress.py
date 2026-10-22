"""How Picasso: Average shows the progress of ``average.average``.

The averaging reports one progress phase per iteration (over the groups)
and hands the localizations of each finished iteration to
``iteration_callback``; the worker turns the former into its status
signal and the latter into the preview.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pandas as pd
import pytest

from picasso import average, lib
from picasso.gui import average as gui_average


@pytest.fixture(autouse=True)
def _qt_app(qapp):
    """The worker is a QThread, which needs a QApplication."""
    return qapp


def _locs():
    return pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "group": [0, 1]})


def test_the_worker_reports_iterations_and_previews(monkeypatch):
    def fake_average(
        locs,
        info,
        *,
        iterations,
        progress=None,
        iteration_callback=None,
        abort_callback=None,
        **kwargs,
    ):
        assert isinstance(progress, lib.Progress)
        for it in range(iterations):
            lib.start_phase(
                progress,
                f"Averaging (iteration {it + 1}/{iterations})",
                2,
                unit="group",
            )
            progress.set_value(0)
            progress.set_value(2)
            locs["x"] += 1  # the working DataFrame keeps changing
            iteration_callback(it + 1, locs)
        return locs, info

    monkeypatch.setattr(average, "average", fake_average)
    worker = gui_average.Worker(_locs(), [{}], 5.0, 2)
    status, previews = [], []
    worker.progressMade.connect(lambda *a: status.append(a))
    worker.iterationFinished.connect(
        lambda it, n, locs: previews.append((it, n, locs))
    )
    worker.run()

    assert status == [(1, 2, 0, 2), (1, 2, 2, 2), (2, 2, 0, 2), (2, 2, 2, 2)]
    assert [(it, n) for it, n, _ in previews] == [(1, 2), (2, 2)]
    # each preview is a copy, not the DataFrame the averaging goes on with
    assert previews[0][2]["x"].tolist() == [1.0, 2.0]
    assert previews[1][2]["x"].tolist() == [2.0, 3.0]
    assert not worker.was_aborted
