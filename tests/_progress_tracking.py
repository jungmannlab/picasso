"""Progress trackers for testing functions that take a ``progress``
argument (see ``picasso.lib.Progress``).

- ``RecordingProgress`` records every phase a function starts, with its
  label, maximum and the values reported in it.
- ``assert_phases_complete`` checks that every recorded phase reported
  monotone values ending at its maximum.
- ``CancelingProgress`` raises ``lib.OperationCanceled`` from
  ``set_value`` after a given number of calls, like a GUI task whose
  Cancel button was pressed.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from picasso import lib


class RecordingProgress(lib.MockProgress):
    """Record the phases and values a function reports.

    ``phases`` is a list of dicts with the keys ``"label"``,
    ``"maximum"``, ``"unit"`` and ``"values"`` (the values reported
    while the phase was current), in the order the phases started.
    """

    def __init__(self):
        super().__init__()
        self.phases: list[dict] = []
        self.n_closed = 0

    def zero_progress(self, description=None, *args, unit=None, **kwargs):
        super().zero_progress(description)
        self.phases.append(
            {
                "label": self.description_base,
                "maximum": self._maximum,
                "unit": unit,
                "values": [],
            }
        )

    def setMaximum(self, maximum, *args, **kwargs):
        super().setMaximum(maximum)
        if self.phases:
            self.phases[-1]["maximum"] = maximum

    def set_value(self, value, *args, **kwargs):
        assert self.phases, "set_value before any phase was started"
        self.phases[-1]["values"].append(value)

    def close(self, *args, **kwargs):
        self.n_closed += 1

    @property
    def labels(self) -> list[str]:
        """The labels of the recorded phases, in order."""
        return [phase["label"] for phase in self.phases]


def assert_phases_complete(
    tracker: RecordingProgress, labels: list[str] | None = None
) -> None:
    """Assert that every phase reported non-negative, monotone values
    ending at its maximum, and, if given, that the phases carry
    ``labels``."""
    assert tracker.phases, "no phase was started"
    for phase in tracker.phases:
        values = phase["values"]
        assert values, f"phase {phase['label']!r} reported no value"
        assert values[0] >= 0, phase
        assert all(b >= a for a, b in zip(values, values[1:])), phase
        assert values[-1] == phase["maximum"], phase
    if labels is not None:
        assert tracker.labels == labels
    # the function must not close a tracker it was given
    assert tracker.n_closed == 0


class CancelingProgress(RecordingProgress):
    """Raise ``lib.OperationCanceled`` from the ``after``-th call of
    ``set_value`` on (counting from 1)."""

    def __init__(self, after: int = 1):
        super().__init__()
        self.after = after
        self.n_values = 0

    def set_value(self, value, *args, **kwargs):
        super().set_value(value)
        self.n_values += 1
        if self.n_values >= self.after:
            raise lib.OperationCanceled()
