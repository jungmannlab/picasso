"""
picasso.gui.render_worker
~~~~~~~~~~~~~~~~~~~~~~~~~

The asynchronous render worker shared by the views of Picasso: Render
(the main ``View`` and the 3D ``ViewRotation``), plus the preview
subsampling both apply to interactive requests.

Kept in its own module so that ``picasso.gui.rotation`` can use the
worker without importing ``picasso.gui.render`` (which imports the
rotation module).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from math import ceil

import numpy as np
import pandas as pd
from PyQt6 import QtCore

from .. import render


class RenderWorker(QtCore.QObject):
    """Render scenes off the GUI thread, latest request wins.

    ``submit`` (called from the GUI thread) replaces any not-yet-started
    request, so a burst of pan/zoom events collapses into rendering the
    newest state; a render already in flight completes and its result is
    discarded by the receiver when superseded (checked by request id).
    The heavy work runs with the GIL released (the render kernels are
    ``nogil``), so the GUI stays responsive throughout.

    Attributes
    ----------
    finished : QtCore.pyqtSignal
        Emitted with ``(request_id, viewport, qimage, n_locs,
        contrast_limits, raw_image)`` when a render completes.
    """

    finished = QtCore.pyqtSignal(int, object, object, int, object, object)
    _poke = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending = None
        # auto connection: queued once this object lives on its thread
        self._poke.connect(self._process)

    def submit(
        self, request_id: int, scene_kwargs: dict, viewport: tuple
    ) -> None:
        """Queue a render request, replacing any pending one (thread
        safe; called from the GUI thread). ``viewport`` is echoed back
        with the result so superseded frames can still be positioned."""
        with self._lock:
            self._pending = (request_id, scene_kwargs, viewport)
        self._poke.emit()

    @QtCore.pyqtSlot()
    def _process(self) -> None:
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:  # already handled by an earlier poke
            return
        request_id, scene_kwargs, viewport = pending
        qimage, n_locs, contrast_limits, raw_image = render.render_scene(
            **scene_kwargs
        )
        self.finished.emit(
            request_id, viewport, qimage, n_locs, contrast_limits, raw_image
        )


def global_precision_of(
    locs: pd.DataFrame, cache: dict
) -> tuple[float, float] | None:
    """The blur of *Global loc. prec.* for one channel: the median
    ``lpx`` and ``lpy`` (camera pixels) of all its localizations,
    computed once per DataFrame and kept in ``cache`` (keyed on the
    DataFrame's identity and length, so a replaced channel is
    recomputed). None when the channel lacks the precision columns."""
    key = (id(locs), len(locs))
    hit = cache.get(key)
    if hit is not None:
        return hit
    if (
        len(locs) == 0
        or "lpx" not in locs.columns
        or "lpy" not in locs.columns
    ):
        value = None
    else:
        value = (
            float(np.median(locs["lpx"].to_numpy())),
            float(np.median(locs["lpy"].to_numpy())),
        )
    cache.clear()  # one entry per channel is plenty; drop stale ones
    cache[key] = value
    return value


def global_precisions_for(
    prepared, channels: list, cache: dict, checked: Callable[[int], bool]
):
    """``render_scene``'s ``global_precision`` argument for the frames a
    view prepared from its ``channels``: with a single channel every
    frame (a group or property split) gets its precision; with several,
    the frames are the checked channels in order. None when the frames
    cannot be matched to channels (the renderer then takes the median
    of the rows it renders). A single DataFrame gets a single pair."""
    single = isinstance(prepared, pd.DataFrame)
    frames = [prepared] if single else prepared
    caches = cache.setdefault("per_channel", {})
    if len(channels) == 1:
        value = global_precision_of(channels[0], caches.setdefault(0, {}))
        values = [value] * len(frames)
    else:
        selected = [i for i in range(len(channels)) if checked(i)]
        if len(selected) != len(frames):
            return None
        values = [
            global_precision_of(channels[i], caches.setdefault(i, {}))
            for i in selected
        ]
    return values[0] if single else values


def subsample_request(request: dict, target_for: Callable[[int], int]) -> bool:
    """Reduce a ``render_scene`` request to a strided subsample for an
    interactive preview.

    Contrast limits are scaled by the sampled fraction so the preview
    keeps the full render's brightness. A channel carrying a row
    selection (``request["indices"]``, the viewport pyramid's choice)
    is subsampled through it, so the preview targets the visible
    population; the others through a strided view of the DataFrame,
    which a backend with resident uploads renders straight from its
    buffers.

    Parameters
    ----------
    request : dict
        Keyword arguments of ``render.render_scene``; ``locs`` (one
        DataFrame or a list of them), optional ``indices`` and
        ``contrast`` are replaced in place.
    target_for : callable
        Maps the visible population (rows over all channels) to the
        number of localizations a preview should render; 0 disables
        previews.

    Returns
    -------
    bool
        False when subsampling is disabled or not needed (the request
        is left untouched).
    """
    if request.get("blur_method") == "triangulation":
        # a subset would change the method (the paper's random
        # subpopulation variant); the preview is the plain, single
        # unjittered pass instead, which costs a tenth of the average
        if request.get("triangulation_passes", 1) == 0:
            return False
        request["triangulation_passes"] = 0
        return True
    if request.get("blur_method") == "quadtree" and request.get("ang") is None:
        # unrotated, the adaptive histogram renders from the channel's
        # index over all its rows; a strided subset has none, and its
        # overview is cheap enough to render whole (rotated, the tree
        # is rebuilt from the projected rows per frame, so a preview
        # subset is fine and worth it)
        return False
    locs = request["locs"]
    single = isinstance(locs, pd.DataFrame)
    channels = [locs] if single else locs
    indices = request.get("indices")
    if indices is None:
        indices = [None] * len(channels)
    population = sum(
        len(channel) if idx is None else len(idx)
        for channel, idx in zip(channels, indices)
    )
    target = target_for(population)
    if target <= 0 or population <= target:
        return False
    step = ceil(population / target)
    sampled = []
    sampled_indices = []
    n_sampled = 0
    for channel, idx in zip(channels, indices):
        if idx is None:
            channel = channel.iloc[::step]
            n_sampled += len(channel)
        else:
            idx = idx[::step]
            n_sampled += len(idx)
        sampled.append(channel)
        sampled_indices.append(idx)
    fraction = n_sampled / population
    request["locs"] = sampled[0] if single else sampled
    if request.get("indices") is not None:
        request["indices"] = sampled_indices
    if request["contrast"] is not None:
        vmin, vmax = request["contrast"]
        request["contrast"] = (vmin * fraction, vmax * fraction)
    return True
