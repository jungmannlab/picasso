"""
picasso.render.backend
~~~~~~~~~~~~~~~~~~~~~~

Contract between scene composition and the raw splat stage: a splat
backend turns per-channel localization columns into raw grayscale
images. ``splat.CpuBackend`` is the reference implementation and the
universal fallback; a GPU backend implements the same contract.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import abc
import logging
import os
import threading
from typing import Literal, TYPE_CHECKING

from .. import lib

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation

    from .splat import _RenderColumns


class SplatBackendError(Exception):
    """A splat backend failed to initialize or render.

    Raising this (rather than crashing) lets ``scene._render_channels``
    retry the request on the CPU backend.
    """


class SplatBackend(abc.ABC):
    """Renders raw per-channel grayscale images from localization
    columns — the seam between scene composition and the splat stage.

    The interface is whole-request (all channels at once) rather than
    per-channel on purpose: a GPU backend keeps channel buffers resident
    and loops channels on-device with a single readback, while the CPU
    backend schedules row chunks across channels through one thread
    pool. A per-channel interface would forbid both optimizations.

    Every implementation must honor three contracts:

    - **Parity**: per channel, ``(n locs in view, float image)`` equal
      to the CPU reference within the golden-scene tolerances
      (``tests/test_render_goldens.py``: raw intensities within 5e-3
      relative or 2e-3 of the image maximum, histogram counts exact up
      to pixel-boundary rounding, total intensity within 1e-3);
      splatting is additive, so total intensity is conserved. A
      backend need not be bit-reproducible from run to run (the GPU
      sums in hardware-dependent order), but its run-to-run spread
      must stay within the same tolerances.
    - **Thread safety**: ``render_channels`` may be called concurrently
      (the asynchronous GUI render worker and a synchronous render such
      as the rotation window can overlap); implementations must
      serialize or isolate shared resources internally.
    - **Failure**: raise ``SplatBackendError`` on any initialization or
      render failure instead of crashing; the caller then re-renders
      the request on the CPU backend.
    """

    #: short identifier used in logs and settings
    name: str = "abstract"
    #: True when uploads persist across renders (GPU): callers then hand
    #: over whole channels instead of viewport slices, so the resident
    #: buffers are reused and nothing is transferred per view
    persistent_uploads: bool = False

    @abc.abstractmethod
    def render_channels(
        self,
        columns: list[_RenderColumns],
        info: list[list[dict]],
        *,
        disp_px_size: float,
        viewport: tuple[tuple[float, float], tuple[float, float]] | None,
        blur_method: (
            Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
        ),
        min_blur_width: float,
        ang: tuple | Rotation | None,
        quadtree_capacity: int | None = None,
        triangulation_passes: int | None = None,
        triangulation_jitter: float | None = None,
    ) -> list[tuple[int, lib.FloatArray2D]]:
        """Render each channel's raw grayscale image.

        Parameters
        ----------
        columns : list of splat._RenderColumns
            Localization columns, one per channel (already extracted,
            angle in radians and lpz fallback applied).
        info : list of list of dict
            Metadata, one entry per channel.
        disp_px_size : float
            Display pixel size in nm, see ``splat.render``.
        viewport : tuple or None
            Field of view ``((y_min, x_min), (y_max, x_max))`` in
            camera pixels, see ``splat.render``.
        blur_method : {"gaussian", "gaussian_iso", "smooth", \
                "convolve"} or None
            Blur method, see ``splat.render``.
        min_blur_width : float
            Minimum size of blur (camera pixels), see ``splat.render``.
        ang : tuple or scipy.spatial.transform.Rotation or None
            Rotation of the localizations, see ``splat.render``.
        quadtree_capacity : int, optional
            Leaf capacity of the 'quadtree' method, see
            ``splat.render``. Default is None.
        triangulation_passes : int, optional
            Passes of the 'triangulation' method, see
            ``splat.render``. Default is None.
        triangulation_jitter : float, optional
            Jitter width of the 'triangulation' method, see
            ``splat.render``. Default is None.

        Returns
        -------
        renderings : list of (int, lib.FloatArray2D)
            ``(number of locs in view, image)`` per channel, in input
            order.
        """

    def release_uploads(self) -> None:
        """Drop resident localization uploads (no-op by default); the
        GUI calls this when datasets are closed."""

    def describe(self) -> str:
        """Human-readable device description (for the GUI's info)."""
        return self.name

    def close(self) -> None:
        """Release backend resources (no-op by default)."""


_cpu_singleton: SplatBackend | None = None
_gpu_singleton: SplatBackend | None = None
_gpu_unavailable = False
_gpu_adapter: str | None = None  # preference the GPU singleton was built for
_singleton_lock = threading.Lock()  # GUI thread and render worker
_settings_cache = (None, None, None)  # (file mtime, loader, Render section)
_settings_lock = threading.Lock()

_log = logging.getLogger(__name__)


def _render_settings() -> dict:
    """The ``Render`` section of the user settings, re-read only when
    the settings file changed (a YAML parse per render is too slow to
    sit on every request). Keyed on the loader too, so a patched
    ``io.load_user_settings`` takes effect immediately."""
    global _settings_cache
    loader = lib.io.load_user_settings
    try:
        mtime = os.path.getmtime(lib.io._user_settings_filename())
    except OSError:
        mtime = None
    with _settings_lock:
        cached = _settings_cache
        if (
            cached[0] == mtime
            and cached[1] is loader
            and cached[2] is not None
        ):
            return cached[2]
        try:
            section = loader().get("Render", None)
        except Exception:
            section = None
        if not isinstance(section, dict):
            section = {}
        _settings_cache = (mtime, loader, section)
        return section


def gpu_settings() -> dict:
    """Read and validate the GPU rendering settings.

    Invalid values of ``settings["Render"]["gpu"]`` fall back to the
    ``lib.RENDER_GPU_*`` defaults.

    Returns
    -------
    settings : dict
        ``enabled`` (``"auto"``, ``"on"`` or ``"off"``; YAML's bare
        ``on``/``off`` parse as booleans and are accepted), ``adapter``
        (a non-empty string) and ``vram_budget_bytes`` (None =
        unlimited).
    """
    raw = _render_settings().get("gpu", None)
    if not isinstance(raw, dict):
        raw = {}
    enabled = raw.get("enabled", lib.RENDER_GPU_ENABLED_DEFAULT)
    if isinstance(enabled, bool):
        enabled = "on" if enabled else "off"
    enabled = str(enabled).strip().lower()
    if enabled not in ("auto", "on", "off"):
        enabled = lib.RENDER_GPU_ENABLED_DEFAULT
    adapter = raw.get("adapter", lib.RENDER_GPU_ADAPTER_DEFAULT)
    if not isinstance(adapter, str) or not adapter.strip():
        adapter = lib.RENDER_GPU_ADAPTER_DEFAULT
    budget = raw.get("vram_budget_mb", lib.RENDER_VRAM_BUDGET_MB_DEFAULT)
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or budget < 0
    ):
        budget = lib.RENDER_VRAM_BUDGET_MB_DEFAULT
    return {
        "enabled": enabled,
        "adapter": adapter.strip(),
        "vram_budget_bytes": None if budget == 0 else int(budget * 2**20),
    }


def vram_budget_bytes() -> int | None:
    """Return the GPU memory budget for resident uploads.

    Returns
    -------
    budget : int or None
        GPU memory (bytes) the backend may keep resident for uploads
        (see ``gpu_settings``); None means unlimited.
    """
    return gpu_settings()["vram_budget_bytes"]


def render_settings_defaults() -> dict:
    """Return the ``Render`` settings keys rendering reads.

    Returns
    -------
    defaults : dict
        The keys with their defaults (``max_workers`` is optional and
        therefore absent).
    """
    return {
        "cpu_utilization": lib.RENDER_CPU_UTILIZATION_DEFAULT,
        "interaction_subsample": lib.RENDER_INTERACTION_SUBSAMPLE_DEFAULT,
        "max_blur_width": lib.RENDER_MAX_BLUR_WIDTH_DEFAULT,
        "gpu": {
            "enabled": lib.RENDER_GPU_ENABLED_DEFAULT,
            "adapter": lib.RENDER_GPU_ADAPTER_DEFAULT,
            "vram_budget_mb": lib.RENDER_VRAM_BUDGET_MB_DEFAULT,
        },
    }


def _fill_missing(target: dict, defaults: dict) -> bool:
    """Add the keys of ``defaults`` that ``target`` lacks (recursing
    into nested mappings); existing values win. Returns whether
    anything was added."""
    added = False
    for key, value in defaults.items():
        if key not in target:
            target[key] = dict(value) if isinstance(value, dict) else value
            added = True
        elif isinstance(value, dict) and isinstance(target[key], dict):
            added = _fill_missing(target[key], value) or added
    return added


def persist_render_defaults() -> bool:
    """Write the missing ``Render`` settings with their defaults.

    Writes the ``Render`` settings the user settings file does not name
    yet, as the other Picasso settings do — so every key is visible and
    editable in the file. Existing values are kept.

    Nothing is written while the settings file on disk is one that
    could not be read (``io.settings_file_is_broken``): a fresh file
    would replace the user's file before they had a chance to fix it.

    Returns
    -------
    bool
        True when the file was written.
    """
    io = lib.io
    settings = io.load_user_settings()
    if io.settings_file_is_broken():
        return False
    section = settings["Render"]
    if not isinstance(section, dict):
        section = settings["Render"] = {}
    if not _fill_missing(section, render_settings_defaults()):
        return False
    io.save_user_settings(settings)
    return True


def _cpu_backend() -> SplatBackend:
    """The process-wide CPU reference backend (also the fallback)."""
    global _cpu_singleton
    with _singleton_lock:
        if _cpu_singleton is None:
            from .splat import CpuBackend

            _cpu_singleton = CpuBackend()
        return _cpu_singleton


def _gpu_backend(adapter: str, warn: bool) -> SplatBackend | None:
    """The process-wide GPU backend for ``adapter``, or None if it
    cannot start: logged once per adapter preference — as a warning
    when the user asked for the GPU explicitly (``warn``), else as
    information — and not retried until the preference changes."""
    global _gpu_singleton, _gpu_unavailable, _gpu_adapter
    with _singleton_lock:
        if _gpu_adapter == adapter:
            if _gpu_singleton is not None:
                return _gpu_singleton
            if _gpu_unavailable:
                return None
        if _gpu_singleton is not None:  # the preference changed
            _gpu_singleton.close()
            _gpu_singleton = None
        _gpu_adapter = adapter
        try:
            from .gpu import WgpuBackend

            _gpu_singleton = WgpuBackend(adapter=adapter)
            _gpu_unavailable = False
        except Exception as error:
            (_log.warning if warn else _log.info)(
                "GPU rendering unavailable (%s); rendering on the CPU", error
            )
            _gpu_unavailable = True
            return None
        _log.info("Rendering on the GPU: %s", _gpu_singleton.describe())
        return _gpu_singleton


def _get_backend(n_locs: int | None = None) -> SplatBackend:
    """The splat backend for the next render, per
    ``settings["Render"]["gpu"]`` (``enabled``: auto/on/off, ``adapter``):
    the GPU when enabled and available, else the CPU reference. Requests
    of fewer than ``lib.RENDER_GPU_MIN_LOCS`` localizations (``n_locs``)
    stay on the CPU, which is faster for them than the GPU's fixed cost
    per render. ``n_locs`` is a cost measure rather than a count: the
    caller weights localizations that are expensive on the CPU (a
    rotated 3D render, ``lib.RENDER_ROTATED_COST_FACTOR``)."""
    settings = gpu_settings()
    if settings["enabled"] == "off":
        return _cpu_backend()
    if n_locs is not None and n_locs < lib.RENDER_GPU_MIN_LOCS:
        return _cpu_backend()
    backend = _gpu_backend(
        settings["adapter"], warn=settings["enabled"] == "on"
    )
    return backend if backend is not None else _cpu_backend()


#: Why the last render on a non-CPU backend was re-rendered on the CPU
#: (a ``SplatBackendError`` message), or None once a render succeeded
#: on that backend again; see ``note_fallback``.
_last_fallback: str | None = None


def note_fallback(reason: str | None) -> None:
    """Record why a render fell back to the CPU.

    The scene dispatch calls this; the GUI shows the reason in its info
    dialog, since the warning in the log is invisible in the windowed
    application.

    Parameters
    ----------
    reason : str or None
        Why the render fell back to the CPU, or None if the chosen
        backend rendered again.
    """
    global _last_fallback
    _last_fallback = reason


def last_fallback() -> str | None:
    """The reason of the most recent CPU fallback, or None."""
    return _last_fallback


def describe_active() -> str:
    """Describe where large renders currently run.

    Used by the GUI's info dialog.

    Returns
    -------
    description : str
        E.g. ``"GPU (Apple M4 via Metal)"`` or ``"CPU (5 workers)"``.
        When the last render on the GPU fell back to the CPU, the
        reason follows, e.g. ``"GPU (...) - last render on the CPU:
        channel exceeds the GPU storage-binding limit"``.
    """
    backend = _get_backend()
    if backend.persistent_uploads:
        text = f"GPU ({backend.describe()})"
        if _last_fallback:
            text += f" - last render on the CPU: {_last_fallback}"
        return text
    workers = lib.n_workers(
        lib.RENDER_CPU_UTILIZATION_DEFAULT, settings_section="Render"
    )
    return f"CPU ({workers} worker{'s' if workers != 1 else ''})"


def release_uploads() -> None:
    """Drop the resident uploads of the GPU backend, if one is running
    (e.g. when the GUI closes a dataset)."""
    if _gpu_singleton is not None:
        _gpu_singleton.release_uploads()


def close() -> None:
    """Release the GPU backend and its device, if one is running.

    Meant for a deterministic teardown while the interpreter is still
    intact (the GUI calls it once its event loop has ended): leaving the
    device to be collected at interpreter shutdown risks a native crash
    in the driver on the way out. A failure here is logged and swallowed
    - the process is exiting anyway, and the OS reclaims the device.
    """
    global _gpu_singleton, _gpu_adapter, _gpu_unavailable
    with _singleton_lock:
        if _gpu_singleton is None:
            return
        try:
            _gpu_singleton.close()
        except Exception as error:  # pragma: no cover - driver teardown
            _log.info("Closing the GPU backend failed: %s", error)
        finally:
            _gpu_singleton = None
            _gpu_adapter = None
            _gpu_unavailable = False
