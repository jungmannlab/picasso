"""
picasso.io
~~~~~~~~~~

General purpose library for handling input and output of files.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import abc
import datetime
import logging
import re
import json
import os
import threading
import warnings
from typing import Callable, TYPE_CHECKING

import tifffile
import yaml
import h5py
import nd2
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from PyQt6 import QtWidgets

from . import lib, __version__

from .ext import bitplane

if bitplane.IMSWRITER:
    from .ext.bitplane import IMSFile

# Optional vendor readers for Zeiss .czi and Leica .lif movies. Both are
# Christoph Gohlke's BSD-licensed libraries (same author as tifffile)
# and require Python >= 3.12, so they are optional dependencies (extras
# ``czi`` / ``lif``). When absent, the corresponding extensions are
# simply not advertised and the loaders raise a helpful ImportError,
# mirroring how ``.ims`` is gated on ``bitplane.IMSWRITER``.
try:
    import czifile
except ImportError:
    czifile = None
try:
    import liffile
except ImportError:
    liffile = None


# MicroManager OME-TIFF files make tifffile log a couple of benign
# messages that are unused by Picasso (frames and the metadata we read
# are unaffected):
#   * continuation files store a non-ASCII ImageDescription (tag 270),
#     triggering a "coercing invalid ASCII to bytes" warning;
#   * the MicroManagerMetadata tag (50839) can carry a zero value
#     offset, which tifffile reports at ERROR level as
#     "<TiffTag.fromfile> raised TiffFileError(... invalid value
#     offset 0)" while still recovering and reading the file.
# Silence tifffile's logger below CRITICAL so these don't reach the
# console; genuine read failures still raise exceptions in load_tif.
logging.getLogger("tifffile").setLevel(logging.CRITICAL)


# Movie file extensions Picasso can open. TIFF_EXTENSIONS are routed to
# the tifffile-backed reader (load_tif); the others have dedicated
# loaders. ".ome.tif" is covered by ".tif" (os.path.splitext yields
# ".tif").
TIFF_EXTENSIONS = (".tif", ".tiff", ".btf", ".tf8", ".tf2", ".lsm")
# .czi (Zeiss) and .lif (Leica) are only advertised when their optional
# reader libraries are importable (see the guarded imports above).
CZI_EXTENSIONS = (".czi",) if czifile is not None else ()
LIF_EXTENSIONS = (".lif",) if liffile is not None else ()
MOVIE_EXTENSIONS = (
    (".raw", ".ims", ".nd2", ".stk")
    + TIFF_EXTENSIONS
    + CZI_EXTENSIONS
    + LIF_EXTENSIONS
)
# Localizations per block when reading an .hdf5 file, see
# ``_read_locs_dataset``. Blocks make long reads reportable and
# abortable; smaller blocks report more often but read more slowly.
LOCS_BLOCK_SIZE = 1_000_000


class NoMetadataFileError(FileNotFoundError):
    """Raised when a movie or localizations file has no metadata sidecar."""


def _user_settings_filename() -> str:
    """Return the path to the user settings file."""
    home = os.path.expanduser("~")
    return os.path.join(home, ".picasso", "settings.yaml")


def plugins_directory() -> str:
    """Return the user plugins directory (``~/.picasso/plugins``).

    The directory is created if it does not yet exist. It sits next to
    ``~/.picasso/settings.yaml`` so that every install type (one-click
    installer, PyPI, source) shares one stable, user-writable location
    that survives uninstalling Picasso.

    Returns
    -------
    directory : str
        ``~/.picasso/plugins``.
    """
    home = os.path.expanduser("~")
    directory = os.path.join(home, ".picasso", "plugins")
    os.makedirs(directory, exist_ok=True)
    return directory


def notification_sounds_directory() -> str:
    """Return the user notification sounds directory
    (``~/.picasso/notification_sounds``).

    The directory is created (empty) if it does not yet exist. It sits
    next to ``~/.picasso/settings.yaml`` and ``~/.picasso/plugins`` so
    that every install type (one-click installer, PyPI, source) shares
    one stable, user-writable location that survives uninstalling
    Picasso. Users add their own ``.mp3`` or ``.wav`` files here.

    Returns
    -------
    directory : str
        ``~/.picasso/notification_sounds``.
    """
    home = os.path.expanduser("~")
    directory = os.path.join(home, ".picasso", "notification_sounds")
    os.makedirs(directory, exist_ok=True)
    return directory


def load_raw(
    path: str,
    prompt_info: Callable[[None], tuple[dict, bool]] | None = None,
    progress: None = None,
) -> tuple[np.memmap, list[dict]]:
    """Load a raw movie file and its metadata.

    Parameters
    ----------
    path : str
        The path to the raw movie file.
    prompt_info : Callable, optional
        A function to call for additional information if needed.
    progress : None, optional
        A placeholder for progress tracking, not used in this function.

    Returns
    -------
    movie : np.memmap
        A memory-mapped numpy array representing the movie, i.e., an
        array that's only partially loaded into memory.
    info : list of dicts
        A list containing a dictionary with metadata about the movie.
    """
    try:
        info = load_info(path)
    except FileNotFoundError as error:
        if prompt_info is None:
            raise error
        else:
            result = prompt_info()
            if result is None:
                return
            else:
                info, save = result
                info = [info]
                if save:
                    base, ext = os.path.splitext(path)
                    info_path = base + ".yaml"
                    save_info(info_path, info)
    dtype = np.dtype(info[0]["Data Type"])
    shape = (info[0]["Frames"], info[0]["Height"], info[0]["Width"])
    movie = np.memmap(path, dtype, "r", shape=shape)
    if info[0]["Byte Order"] != "<":
        movie = movie.byteswap()
        info[0]["Byte Order"] = "<"
    return movie, info


def load_ims(
    path: str,
    prompt_info: Callable[[list[str]], str] | None = None,
) -> tuple[AbstractPicassoMovie, list[dict]]:
    """Load a Bitplane IMS movie file and its metadata.

    Parameters
    ----------
    path : str
        The path to the IMS movie file.
    prompt_info : Callable, optional
        A function to call for additional information if needed.

    Returns
    -------
    movie : bitplane.MovieMapperStack or bitplane.MovieMapper
        Custom wrapper around IMS file(s).
    info : list of dicts
        A list containing a dictionary with metadata about the movie.
    """
    if not bitplane.IMSWRITER:
        raise ImportError(".ims files are only supported on Windows machines.")
    file = IMSFile(path)

    if len(file.channels) > 1:
        # Default to Channel 0 when causing localizer
        if prompt_info is None:
            channel = "Channel 0"
        else:
            channel = prompt_info(file.channels)
        file.set_channel(channel)

    else:
        channel = "Channel 0"

    file.read_movie()

    info = {}

    info["Frames"] = file.n_frames
    info["Height"] = file.x
    info["Width"] = file.y
    info["Channel"] = channel

    if file.pixelsize is not None:
        info["Pixelsize"] = file.pixelsize

    info["GlobalExtMin0"] = file.ext_min0
    info["GlobalExtMin1"] = file.ext_min1
    info["GlobalExtMin2"] = file.ext_min2

    info["GlobalExtMax0"] = file.ext_max0
    info["GlobalExtMax1"] = file.ext_max1
    info["GlobalExtMax2"] = file.ext_max2

    info["Generated by"] = "IMS Metadata"

    info = [info]

    return file.movie, info


def load_ims_all(path: str) -> tuple[list[np.memmap], list[list[dict]]]:
    """Load all channels of a Bitplane IMS movie file and their
    metadata.

    Parameters
    ----------
    path : str
        The path to the IMS movie file.

    Returns
    -------
    movies : list of np.memmaps
        A list of memory-mapped numpy arrays representing the movie
        channels.
    infos : list of lists of dicts
        A list of lists containing dictionaries with metadata about each
        movie channel.
    """
    file = IMSFile(path)

    movies = []
    infos = []

    for channel in file.channels:
        file.set_channel(channel)

        file.read_movie()

        info = {}
        info["Frames"] = file.n_frames
        info["Height"] = file.x
        info["Width"] = file.y
        info["Channel"] = channel

        if file.pixelsize is not None:
            info["Pixelsize"] = file.pixelsize

        info["ExtMin0"] = file.ext_min0
        info["ExtMin1"] = file.ext_min1
        info["ExtMin2"] = file.ext_min2

        info["ExtMax0"] = file.ext_max0
        info["ExtMax1"] = file.ext_max1
        info["ExtMax2"] = file.ext_max2

        info["Generated by"] = "IMS Metadata"

        info = [info]

        movies.append(file.movie)
        infos.append(info)

    return movies, infos


def save_config(CONFIG: dict) -> None:
    """Save the camera configuration dictionary to the user config file
    (``~/.picasso/config.yaml``). See https://picassosr.readthedocs.io/
    en/latest/localize.html#camera-config.

    Parameters
    ----------
    CONFIG : dict
        The camera configuration dictionary to save.
    """
    from . import config_filename, user_config_dir

    os.makedirs(user_config_dir(), exist_ok=True)
    with open(config_filename(), "w") as config_file:
        yaml.dump(CONFIG, config_file, width=1000)


def save_raw(path: str, movie: lib.IntArray3D, info: dict) -> None:
    """Save a raw movie file and its metadata.

    Parameters
    ----------
    path : str
        The path to the raw movie file.
    movie : lib.IntArray3D
        The raw movie data to save.
    info : dict
        The metadata information to save.
    """
    movie.tofile(path)
    info_path = os.path.splitext(path)[0] + ".yaml"
    save_info(info_path, info)


def save_calibration(path: str, calibration: dict) -> None:
    """Save calibration file to a .yaml path.

    Parameters
    ----------
    path : str
        Where to write the YAML.
    calibration : dict
        The calibration to save.
    """
    with open(path, "w") as f:
        yaml.dump(calibration, f, default_flow_style=False)


def load_calibration(path: str) -> dict:
    """Load 3D astigmatic calibration data from a YAML file.

    Parameters
    ----------
    path : str
        The path to the calibration YAML file.

    Returns
    -------
    calibration : dict
        A dictionary containing the 3D astigmatic calibration data.
    """
    with open(path, "r") as calibration_file:
        try:
            calibration = yaml.full_load(calibration_file)
        except yaml.composer.ComposerError:
            raise ValueError(
                "Invalid calibration file: expected a single-document YAML "
                "file. This does not look like a 3D calibration file."
            )

    if not isinstance(calibration, dict) or (
        "X Coefficients" not in calibration
        or not isinstance(calibration["X Coefficients"], list)
    ):
        raise ValueError(
            "Invalid calibration file: 'X Coefficients' must be present and "
            "be a dictionary."
        )

    return calibration


def _json_default(obj):
    """Coerce values that JSON cannot represent into JSON-serializable
    Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def save_spline_calibration(path: str, calibration: dict) -> None:
    """Save a cubic-spline PSF calibration to an HDF5 file.

    Unlike the astigmatism calibration (a handful of polynomial coefficients
    stored as YAML via ``zfit.calibrate_z``), a spline PSF calibration holds a
    large coefficient table (64 coefficients per interval for a 3D tricubic
    spline, times the number of intervals), so it is stored in HDF5. The
    coefficient array goes into the ``coefficients`` dataset; all remaining
    (scalar / list) metadata is stored as a JSON string in the file attribute
    ``metadata``.

    Parameters
    ----------
    path : str
        Destination HDF5 path (conventionally ``*_spline_calib.hdf5``).
    calibration : dict
        Calibration dictionary. Must contain a numpy array under the key
        ``"coefficients"``; every other key must be JSON-serializable (numpy
        scalars/arrays are coerced automatically).
    """
    if "coefficients" not in calibration:
        raise ValueError(
            "Invalid spline calibration: missing 'coefficients' array."
        )
    coefficients = np.ascontiguousarray(
        calibration["coefficients"], dtype=np.float32
    )
    metadata = {
        key: value
        for key, value in calibration.items()
        if key != "coefficients"
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("coefficients", data=coefficients)
        f.attrs["metadata"] = json.dumps(metadata, default=_json_default)


def load_spline_calibration(path: str) -> dict:
    """Load a cubic-spline PSF calibration saved by
    ``save_spline_calibration``.

    Parameters
    ----------
    path : str
        Path to the spline calibration HDF5 file.

    Returns
    -------
    calibration : dict
        The calibration dictionary, with the coefficient table restored under
        the ``"coefficients"`` key (float32 numpy array) and all metadata
        keys alongside it.
    """
    with h5py.File(path, "r") as f:
        if "coefficients" not in f or "metadata" not in f.attrs:
            raise ValueError(
                "Invalid spline calibration file: expected a 'coefficients' "
                "dataset and a 'metadata' attribute. This does not look like "
                "a Picasso spline PSF calibration."
            )
        calibration = json.loads(f.attrs["metadata"])
        calibration["coefficients"] = f["coefficients"][:].astype(np.float32)
    return calibration


def _is_hdf5_calibration(path: str) -> bool:
    """Whether ``path`` names an HDF5 (spline PSF) calibration rather than
    a YAML one, by extension."""
    return os.path.splitext(path)[1].lower() in (".hdf5", ".h5")


def load_any_calibration(path: str) -> dict:
    """Load a calibration of any kind: a cubic-spline PSF calibration
    (HDF5), a Gaussian astigmatism calibration or a standalone affine
    calibration (both YAML).

    Used wherever a calibration is only handled as a carrier of lateral
    corrections (see ``picasso.lib.lateral_transforms``), so no format is
    assumed. Unlike ``load_calibration`` this does not require the
    astigmatism polynomial coefficients to be present.

    Parameters
    ----------
    path : str
        Path to the calibration file.

    Returns
    -------
    calibration : dict
        The calibration dictionary.
    """
    if _is_hdf5_calibration(path):
        return load_spline_calibration(path)
    with open(path, "r") as calibration_file:
        try:
            calibration = yaml.full_load(calibration_file)
        except yaml.composer.ComposerError:
            raise ValueError(
                "Invalid calibration file: expected a single-document YAML "
                "file. This does not look like a Picasso calibration file."
            )
    if not isinstance(calibration, dict):
        raise ValueError(
            "Invalid calibration file: expected a YAML mapping. This does "
            "not look like a Picasso calibration file."
        )
    return calibration


def save_any_calibration(path: str, calibration: dict) -> None:
    """Save a calibration loaded by ``load_any_calibration`` back to its
    own format: HDF5 for a spline PSF calibration, YAML otherwise.

    Parameters
    ----------
    path : str
        Destination path; its extension selects the format.
    calibration : dict
        The calibration dictionary to write.
    """
    if _is_hdf5_calibration(path):
        save_spline_calibration(path, calibration)
    else:
        save_calibration(path, calibration)


# Map names of an sCMOS camera calibration. ``gain`` is optional: it needs a
# bright illumination series, while ``offset`` and ``variance`` come from the
# dark movie alone. See ``picasso.scmos``.
_CAMERA_CALIBRATION_MAPS = ("offset", "variance", "gain")


def save_camera_calibration(path: str, calibration: dict) -> None:
    """Save a per-pixel sCMOS camera calibration to an HDF5 file.

    Stored exactly like a spline PSF calibration - the large arrays as
    datasets, everything else as a JSON string in the ``metadata`` file
    attribute - because it has the same shape of problem: sensor-sized maps
    that do not belong in a YAML sidecar.

    Parameters
    ----------
    path : str
        Destination HDF5 path (conventionally ``*_scmos_calib.hdf5``).
    calibration : dict
        From ``picasso.scmos.calibrate_scmos``. Must contain the ``offset``
        (ADU) and ``variance`` (ADU squared) maps and may contain ``gain``
        (ADU per photoelectron); every other key must be JSON-serializable
        (numpy scalars/arrays are coerced automatically).
    """
    for name in ("offset", "variance"):
        if name not in calibration:
            raise ValueError(
                f"Invalid camera calibration: missing '{name}' map."
            )
    metadata = {
        key: value
        for key, value in calibration.items()
        if key not in _CAMERA_CALIBRATION_MAPS
    }
    with h5py.File(path, "w") as f:
        for name in _CAMERA_CALIBRATION_MAPS:
            if calibration.get(name) is not None:
                f.create_dataset(
                    name,
                    data=np.ascontiguousarray(
                        calibration[name], dtype=np.float32
                    ),
                )
        f.attrs["metadata"] = json.dumps(metadata, default=_json_default)


def load_camera_calibration(path: str) -> dict:
    """Load a per-pixel sCMOS camera calibration saved by
    :func:`save_camera_calibration`.

    Parameters
    ----------
    path : str
        Path to the camera calibration HDF5 file.

    Returns
    -------
    calibration : dict
        The calibration dictionary, with ``offset`` and ``variance`` (and
        ``gain``, if the file has one) restored as float32 numpy arrays
        alongside the metadata. ``Path`` is set to ``path``, so a calibration
        that has been moved still reports where it was actually loaded from.
    """
    with h5py.File(path, "r") as f:
        if (
            "offset" not in f
            or "variance" not in f
            or "metadata" not in f.attrs
        ):
            raise ValueError(
                "Invalid camera calibration file: expected 'offset' and "
                "'variance' datasets and a 'metadata' attribute. This does "
                "not look like a Picasso sCMOS camera calibration."
            )
        calibration = json.loads(f.attrs["metadata"])
        for name in _CAMERA_CALIBRATION_MAPS:
            if name in f:
                calibration[name] = f[name][:].astype(np.float32)
    calibration["Path"] = path
    return calibration


def _readable_movie_dims(movie: AbstractPicassoMovie) -> dict:
    """Collect the movie dimensions that can be read straight from the
    file structure (frames, height, width), independent of the embedded
    metadata. Used to pre-fill the manual-metadata fallback dialog.

    Parameters
    ----------
    movie : AbstractPicassoMovie
        A movie object whose pixel data could be opened but whose
        metadata could not be parsed.

    Returns
    -------
    dims : dict
        Any of the keys ``Frames``, ``Height`` and ``Width`` that could
        be determined.
    """
    dims = {}
    try:
        dims["Frames"] = int(len(movie))
    except Exception:
        pass
    height = getattr(movie, "height", None)
    width = getattr(movie, "width", None)
    if height is None or width is None:
        # ND2 movies keep their dimensions in a ``sizes`` mapping rather
        # than as plain attributes.
        sizes = getattr(movie, "sizes", None)
        if sizes is not None:
            height = sizes.get("Y", height)
            width = sizes.get("X", width)
    if height is not None:
        dims["Height"] = int(height)
    if width is not None:
        dims["Width"] = int(width)
    return dims


def _movie_metadata_fallback(
    movie: AbstractPicassoMovie,
    path: str,
    prompt_info: Callable[[dict], tuple[dict, bool]] | None,
    cause: BaseException | None = None,
) -> dict | None:
    """Build movie metadata when it could not be read from the file.

    First tries an accompanying ``.yaml`` metadata file (e.g. one saved
    during a previous fallback). If none is found and ``prompt_info`` is
    given, the user is asked to enter the required metadata manually,
    pre-filled with whatever dimensions could still be read. Without a
    prompt callback (e.g. when called programmatically rather than from
    the GUI), a ``NoMetadataFileError`` is raised instead.

    Parameters
    ----------
    movie : AbstractPicassoMovie
        The movie whose pixel data opened but whose metadata could not
        be parsed.
    path : str
        Path to the movie file.
    prompt_info : Callable or None
        Called with the readable dimensions; must return ``(info, save)``
        or None if the user cancels.
    cause : BaseException or None, optional
        The original error raised while reading the metadata, chained
        onto ``NoMetadataFileError`` for context.

    Returns
    -------
    info : dict or None
        The metadata dictionary, or None if the user canceled the
        prompt dialog.

    Raises
    ------
    NoMetadataFileError
        If the metadata could not be read, there is no sidecar ``.yaml``
        file, and no ``prompt_info`` callback was provided to obtain it
        interactively.
    """
    # A sidecar YAML file (possibly saved during an earlier fallback)
    # takes precedence over prompting the user again.
    try:
        return load_info(path)[0]
    except (FileNotFoundError, NoMetadataFileError):
        pass
    if prompt_info is None:
        # No way to obtain the metadata interactively (e.g. programmatic
        # use). Raise rather than silently returning None so the caller
        # gets an informative error instead of an unpack failure.
        raise NoMetadataFileError(
            f"Could not read metadata for movie:\n{path}\n"
            "No accompanying .yaml metadata file was found."
        ) from cause
    result = prompt_info(_readable_movie_dims(movie))
    if result is None:
        return None
    info, save = result
    if save:
        base, _ = os.path.splitext(path)
        save_info(base + ".yaml", [info])
    return info


def _movie_info_or_prompt(
    movie: AbstractPicassoMovie,
    path: str,
    prompt_info: Callable[[dict], tuple[dict, bool]] | None,
) -> dict | None:
    """Return the movie's metadata, falling back to manual entry if it
    cannot be read.

    Returns None only when the metadata could not be read and the user
    canceled the fallback dialog, in which case the caller should abort
    loading. When no ``prompt_info`` callback is available (e.g.
    programmatic use), a ``NoMetadataFileError`` is raised instead of
    returning None.
    """
    try:
        info = movie.info()
    except Exception as error:
        info = None
        cause = error
    else:
        cause = None
    if not info:
        return _movie_metadata_fallback(movie, path, prompt_info, cause)
    return info


def load_tif(
    path: str,
    prompt_info: Callable[[dict], tuple[dict, bool]] | None = None,
    progress=None,
) -> tuple[TiffMultiMap, list[dict]] | None:
    """Load a TIFF movie file and its metadata.

    Parameters
    ----------
    path : str
        The path to the TIFF movie file.
    prompt_info : Callable, optional
        Called with the readable movie dimensions if the embedded
        metadata cannot be parsed, so the user can enter it manually.
        Must return ``(info, save)`` or None if canceled.
    progress : callable, optional
        ``callable(done, total)`` invoked as the per-page IFD scan
        proceeds, so a smooth determinate progress bar can be shown while
        a large movie is opened. Default is None (no reporting).

    Returns
    -------
    movie : TiffMultiMap
        A movie object providing array-like access to TIFF frames.
        Frames are loaded into memory on access.
    info : list[dict]
        A list containing a dictionary with metadata about the movie.

    Returns None if the metadata could not be read and the user
    canceled the manual-metadata fallback dialog.
    """
    # MicroManager can save an acquisition as one single-page TIFF per
    # frame ("separate image files"). If ``path`` is one such frame,
    # assemble the whole folder into a single movie instead of opening a
    # one-frame file.
    separate_paths = _mm_separate_files(path)
    if separate_paths is not None:
        movie = MMSeparateTiffMovie(separate_paths)
    else:
        movie = TiffMultiMap(path, memmap_frames=False, progress=progress)
    info = _movie_info_or_prompt(movie, path, prompt_info)
    if info is None:
        return None
    return movie, [info]


def load_nd2(
    path: str,
    prompt_info: Callable[[dict], tuple[dict, bool]] | None = None,
) -> tuple[ND2Movie, list[dict]] | None:
    """Load a Nikon ND2 movie file and its metadata.

    Parameters
    ----------
    path : str
        The path to the ND2 movie file.
    prompt_info : Callable, optional
        Called with the readable movie dimensions if the embedded
        metadata cannot be parsed, so the user can enter it manually.
        Must return ``(info, save)`` or None if canceled.

    Returns
    -------
    movie : ND2Movie
        The loaded ND2 movie.
    info : list of dicts
        A list containing a dictionary with metadata about the movie.

    Returns None if the metadata could not be read and the user
    canceled the manual-metadata fallback dialog.
    """
    movie = ND2Movie(path)
    info = _movie_info_or_prompt(movie, path, prompt_info)
    if info is None:
        return None
    return movie, [info]


def load_nd2_all(
    path: str,
    prompt_info: Callable[[dict], tuple[dict, bool]] | None = None,
) -> tuple[list[ND2Movie], list[list[dict]]] | None:
    """Load all channels of a Nikon ND2 movie file and their metadata.

    Each channel is returned as an independent ``ND2Movie`` so all
    channels can be read at the same time. Single-channel files yield a
    one-element list (identical to ``load_nd2``).

    Parameters
    ----------
    path : str
        The path to the ND2 movie file.
    prompt_info : Callable, optional
        Called with the readable movie dimensions if the embedded metadata
        cannot be parsed, so the user can enter it manually. Must return
        ``(info, save)`` or None if canceled.

    Returns
    -------
    movies : list of ND2Movie
        One movie object per channel.
    infos : list of lists of dicts
        Per-channel metadata, each carrying a ``"Channel"`` key.

    Returns None if the metadata could not be read and the user canceled
    the manual-metadata fallback dialog.
    """
    probe = ND2Movie(path)
    n_channels = probe.n_channels
    probe.close()
    movies = []
    infos = []
    for i in range(n_channels):
        movie = ND2Movie(path, channel=i)
        info = _movie_info_or_prompt(movie, path, prompt_info)
        if info is None:
            return None
        movies.append(movie)
        infos.append([info])
    return movies, infos


def load_stk(
    path: str,
    prompt_info: Callable[[dict], tuple[dict, bool]] | None = None,
) -> tuple[STKMultiMovie, list[dict]] | None:
    """Load a MetaMorph STK movie file and its metadata.

    If the filename contains a numeric suffix (e.g. ``name_003.stk``),
    all files in the same directory with the same base name and an equal
    or higher suffix are loaded as a single contiguous movie.

    Parameters
    ----------
    path : str
        The path to the STK movie file.
    prompt_info : Callable, optional
        Called with the readable movie dimensions if the embedded
        metadata cannot be parsed, so the user can enter it manually.
        Must return ``(info, save)`` or None if canceled.

    Returns
    -------
    movie : STKMultiMovie
        A movie object providing array-like access to STK frames.
        Frames are loaded into memory on access.
    info : list[dict]
        A list containing a dictionary with metadata about the movie.

    Returns None if the metadata could not be read and the user
    canceled the manual-metadata fallback dialog.
    """
    movie = STKMultiMovie(path)
    info = _movie_info_or_prompt(movie, path, prompt_info)
    if info is None:
        return None
    return movie, [info]


def load_czi(
    path: str,
    prompt_info: Callable[[list[str]], str] | None = None,
) -> tuple[CZIMovie, list[dict]]:
    """Load a Zeiss CZI movie file and its metadata.

    Multi-channel/Z files are reduced to a single-channel ``(T, Y, X)``
    movie; when more than one channel is present, ``prompt_info`` is
    called to choose one (defaulting to the first channel otherwise).

    Parameters
    ----------
    path : str
        The path to the CZI movie file.
    prompt_info : Callable, optional
        Called with the list of channel names to select one.

    Returns
    -------
    movie : CZIMovie
        A movie object providing array-like access to CZI frames.
    info : list[dict]
        A list containing a dictionary with metadata about the movie.
    """
    if czifile is None:
        raise ImportError(
            "Reading .czi files requires the optional 'czifile' package "
            "(needs Python >= 3.12). Install it with: "
            "pip install picassosr[czi]"
        )
    movie = CZIMovie(path, prompt_info=prompt_info)
    return movie, [movie.info()]


def load_czi_all(path: str) -> tuple[list[CZIMovie], list[list[dict]]]:
    """Load all channels of a Zeiss CZI movie file and their metadata.

    Each channel is returned as an independent ``CZIMovie`` so all
    channels can be read at the same time (e.g. for across-channel
    fitting). The file is reopened once per channel; CZI readers are lazy,
    so this is cheap for the handful of channels typical in SMLM.

    Parameters
    ----------
    path : str
        The path to the CZI movie file.

    Returns
    -------
    movies : list of CZIMovie
        One movie object per channel.
    infos : list of lists of dicts
        Per-channel metadata, each carrying a ``"Channel"`` key.
    """
    if czifile is None:
        raise ImportError(
            "Reading .czi files requires the optional 'czifile' package "
            "(needs Python >= 3.12). Install it with: "
            "pip install picassosr[czi]"
        )
    probe = CZIMovie(path)
    channels = list(probe.channels)
    probe.close()
    movies = []
    infos = []
    for i in range(len(channels)):
        movie = CZIMovie(path, channel=i)
        movies.append(movie)
        infos.append([movie.info()])
    return movies, infos


def load_lif(
    path: str,
    prompt_info: Callable[[list[str]], str] | None = None,
) -> tuple[LIFMovie, list[dict]]:
    """Load a Leica LIF movie file and its metadata.

    A LIF file may contain several image series; the one with the most
    time frames is used. Multi-channel files are reduced to a
    single-channel ``(T, Y, X)`` movie via ``prompt_info`` (defaulting to
    the first channel otherwise).

    Parameters
    ----------
    path : str
        The path to the LIF movie file.
    prompt_info : Callable, optional
        Called with the list of channel names to select one.

    Returns
    -------
    movie : LIFMovie
        A movie object providing array-like access to LIF frames.
    info : list[dict]
        A list containing a dictionary with metadata about the movie.
    """
    if liffile is None:
        raise ImportError(
            "Reading .lif files requires the optional 'liffile' package "
            "(needs Python >= 3.12). Install it with: "
            "pip install picassosr[lif]"
        )
    movie = LIFMovie(path, prompt_info=prompt_info)
    return movie, [movie.info()]


def load_lif_all(path: str) -> tuple[list[LIFMovie], list[list[dict]]]:
    """Load all channels of a Leica LIF movie file and their metadata.

    Each channel is returned as an independent ``LIFMovie`` (the image
    series with the most time frames is used, as in ``load_lif``) so all
    channels can be read at the same time. The file is reopened once per
    channel; LIF readers are lazy, so this is cheap.

    Parameters
    ----------
    path : str
        The path to the LIF movie file.

    Returns
    -------
    movies : list of LIFMovie
        One movie object per channel.
    infos : list of lists of dicts
        Per-channel metadata, each carrying a ``"Channel"`` key.
    """
    if liffile is None:
        raise ImportError(
            "Reading .lif files requires the optional 'liffile' package "
            "(needs Python >= 3.12). Install it with: "
            "pip install picassosr[lif]"
        )
    probe = LIFMovie(path)
    channels = list(probe.channels)
    probe.close()
    movies = []
    infos = []
    for i in range(len(channels)):
        movie = LIFMovie(path, channel=i)
        movies.append(movie)
        infos.append([movie.info()])
    return movies, infos


def load_movie(
    path: str,
    prompt_info=None,
    progress=None,
) -> tuple[AbstractPicassoMovie, list[dict]]:
    """Load a movie file based on its extension and returns the movie
    object and its metadata.

    Accepted formats are specified by ``MOVIE_EXTENSIONS``.

    Parameters
    ----------
    path : str
        The path to the movie file.
    prompt_info : Callable, optional
        Format-specific callback used to obtain missing metadata
        interactively (e.g. to select a channel for multi-channel files
        or to enter movie metadata manually when it cannot be read).
    progress : callable, optional
        ``callable(done, total)`` forwarded to the TIFF loader to report
        per-page progress while a movie is opened. Other formats ignore
        it (they open effectively instantly). Default is None.

    Returns
    -------
    movie : AbstractPicassoMovie
        The loaded movie object.
    info : list[dict]
        A list containing a dictionary with metadata about the movie.

    Raises
    ------
    ValueError
        If the file extension is not a supported movie format.
    """
    base, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".raw":
        return load_raw(path, prompt_info=prompt_info)
    elif ext in TIFF_EXTENSIONS:
        return load_tif(path, prompt_info=prompt_info, progress=progress)
    elif ext == ".ims":
        return load_ims(path, prompt_info=prompt_info)
    elif ext == ".nd2":
        return load_nd2(path, prompt_info=prompt_info)
    elif ext == ".stk":
        return load_stk(path, prompt_info=prompt_info)
    elif ext == ".czi":
        return load_czi(path, prompt_info=prompt_info)
    elif ext == ".lif":
        return load_lif(path, prompt_info=prompt_info)
    else:
        raise ValueError(
            f"Unsupported movie format: {ext}. Supported formats are"
            f" {MOVIE_EXTENSIONS}."
        )


def load_movie_all(
    path: str,
    prompt_info=None,
    progress=None,
) -> tuple[list[AbstractPicassoMovie], list[list[dict]]] | None:
    """Load every channel of a movie file as independent movies.

    Mirrors ``load_movie`` but returns parallel lists - one entry per
    channel. Multi-channel formats (.ims, .czi, .lif, .nd2) expose all of
    their channels; single-channel formats return one-element lists. Each
    ``info[0]`` carries a ``"Channel"`` key identifying the channel.

    Parameters
    ----------
    path : str
        The path to the movie file.
    prompt_info : Callable, optional
        Manual-metadata fallback callback, forwarded to the formats that
        may need it (.nd2 and the single-channel formats). Channel
        selection is never prompted here - all channels are loaded.
    progress : None
        Placeholder for progress tracking, not used.

    Returns
    -------
    movies : list of AbstractPicassoMovie
        One movie object per channel.
    infos : list of lists of dicts
        Per-channel metadata.

    Returns None if metadata could not be read and the user canceled the
    manual-metadata fallback dialog.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ims":
        return load_ims_all(path)
    elif ext == ".czi":
        return load_czi_all(path)
    elif ext == ".lif":
        return load_lif_all(path)
    elif ext == ".nd2":
        return load_nd2_all(path, prompt_info=prompt_info)
    else:
        # Single-channel formats: wrap load_movie into one-element lists.
        result = load_movie(path, prompt_info=prompt_info, progress=progress)
        if result is None:
            return None
        movie, info = result
        return [movie], [info]


def load_info(
    path: str,
    qt_parent: QtWidgets.QWidget | None = None,
) -> list[dict]:
    """Load metadata from a YAML file associated with the movie file.

    Parameters
    ----------
    path : str
        The path to the movie file, which is used to derive the metadata
        file name.
    qt_parent : QWidget or None, optional
        The parent widget for any error messages displayed using Qt.
        Default is None.

    Returns
    -------
    info : list of dict
        A list containing a dictionary with metadata about the movie.
    """
    path_base, path_extension = os.path.splitext(path)
    filename = path_base + ".yaml"
    # First, try the sidecar .yaml metadata file.
    try:
        with open(filename, "r") as info_file:
            return list(yaml.load_all(info_file, Loader=yaml.UnsafeLoader))
    except FileNotFoundError:
        pass
    # If absent, fall back to metadata embedded in the HDF5 file itself.
    info = _load_info_from_hdf5(path)
    if info is not None:
        return info
    # Neither the sidecar file nor embedded metadata was found. Files
    # written before Picasso v0.11 carry no embedded metadata, so their
    # .yaml is the only copy.
    message = (
        f"Could not find metadata for:\n{path}\nNeither the metadata "
        f"file:\n{filename}\nnor metadata embedded in the file itself "
        "could be read."
    )
    print(f"\nAn error occured. {message}")
    if qt_parent is not None:
        from PyQt6 import QtWidgets

        QtWidgets.QMessageBox.critical(qt_parent, "An error occured", message)
    raise NoMetadataFileError(filename)


def load_mask(
    path: str,
    qt_parent: QtWidgets.QWidget | None = None,
) -> tuple[lib.FloatArray2D, dict]:
    """Load a mask generated with ``spinna.MaskGenerator``.

    Parameters
    ----------
    path : str
        The path to the mask file.
    qt_parent : QWidget or None, optional
        The parent widget for any error messages displayed using Qt.
        Default is None.

    Returns
    -------
    mask : lib.FloatArray2D
        The loaded mask array.
    info : dict
        A dictionary containing metadata about the mask.
    """
    mask = np.float64(np.load(path))
    mask = mask / mask.sum()
    new_path = os.path.splitext(path)[0] + ".yaml"
    info = load_info(new_path, qt_parent=qt_parent)[0]
    try:
        value = info["Generated by"]
    except KeyError:
        raise TypeError("Incorrect file loaded.")
    if "SPINNA" not in value:
        raise TypeError("Please load a mask provided by Picasso SPINNA")
    return mask, info


def load_picks(  # noqa: C901
    path: str, pixelsize: float | None = None
) -> tuple[list, str, float]:
    """Load picks generated with the Picasso GUI.

    Parameters
    ----------
    path : str
        The path to the picks file.
    pixelsize : float, optional
        Camera pixel size in nm. Used to convert pick size from nm to
        camera pixels (which are the units of localizations coordinates).
        If None, the size will be returned in original units.

    Returns
    -------
    picks : list
        A list of picks.
    shape : str
        The shape of the picks, one of ``lib.PICK_SHAPES``.
    size : float
        The size of the picks in camera pixels (if `pixelsize` is
        provided, otherwise in original units). For circular picks, the
        size is the diameter; for rectangular picks, the size is the
        width; for square picks, the size is the side length. None for
        polygonal, box and brush picks (size not defined).

    Raises
    ------
    ValueError
        If the file is not recognized as a picks file, if the pick
        shape is unknown, or if a shape that needs a size does not
        carry one.
    """
    assert path.endswith(".yaml"), "Picks should be stored in a .yaml file."

    # load the file
    with open(path, "r") as f:
        regions = yaml.full_load(f)

    # Backwards compatibility for old picked region files
    if "Shape" in regions:
        shape = regions["Shape"]
    elif "Centers" in regions and "Diameter" in regions:
        shape = "Circle"
    else:
        raise ValueError("Unrecognized picks file")

    pixelsize = 1 if pixelsize is None else pixelsize

    # assign loaded picks and pick size
    if shape == "Circle":
        picks = regions["Centers"]
        # legacy files store the size in camera pixels, without "(nm)"
        size = _pick_size_from_regions(regions, "Diameter", pixelsize)
    elif shape == "Rectangle":
        picks = regions["Center-Axis-Points"]
        size = _pick_size_from_regions(regions, "Width", pixelsize)
    elif shape == "Polygon":
        picks = regions["Vertices"]
        size = None
    elif shape == "Square":
        picks = regions["Centers"]
        # no backward compatibility here, always in nm
        size = _pick_size_from_regions(regions, "Side Length", pixelsize)
    elif shape == "Box":
        # each box carries its own extent, so there is no size
        picks = regions["Corners"]
        size = None
    elif shape == "Brush":
        # the file holds a flat, ordered list of strokes; which of them
        # form one pick follows from their geometry, so the grouping is
        # re-derived rather than stored
        strokes = [
            (stroke["Width (nm)"] / pixelsize, stroke["Path"])
            for stroke in regions["Strokes"]
        ]
        picks = lib.merge_brush_strokes(strokes)
        size = None
    else:
        raise ValueError(f"Unrecognized pick shape: {shape}")
    return picks, shape, size


def _pick_size_from_regions(
    regions: dict, key: str, pixelsize: float
) -> float:
    """Return a pick size from a loaded picks file, in camera pixels.

    Sizes are stored in nm under ``"<key> (nm)"``; files written by
    older versions of Picasso store them in camera pixels under the bare
    ``key``.

    Parameters
    ----------
    regions : dict
        Contents of the picks .yaml file.
    key : str
        Name of the size entry, e.g., "Diameter".
    pixelsize : float
        Camera pixel size in nm.

    Returns
    -------
    size : float
        Pick size in camera pixels.

    Raises
    ------
    ValueError
        If neither the nm nor the legacy entry is present.
    """
    if f"{key} (nm)" in regions:
        return regions[f"{key} (nm)"] / pixelsize
    elif key in regions:
        return regions[key]
    raise ValueError(
        f"Picks file is missing the pick size entry '{key} (nm)'."
    )


def save_drift(path: str, drift: pd.DataFrame) -> None:
    """Save drift to a .txt file in the format used by the Picasso.

    Parameters
    ----------
    path : str
        The path to the drift file. Must end in .txt.
    drift : pd.DataFrame
        A DataFrame with 'x' and 'y' columns and drift values for each
        frame.
    """
    # Binary handle so the explicit "\r\n" is not itself newline-translated
    # (a text handle writes "\r\r\n" on Windows).
    with open(path, "wb") as f:
        np.savetxt(f, drift, newline="\r\n")


def load_drift(path: str) -> pd.DataFrame | None:
    """Load drift from a .txt file generated with the Picasso GUI.

    Parameters
    ----------
    path : str
        The path to the drift file. Must end in .txt.

    Returns
    -------
    drift_df : pd.DataFrame or None
        A DataFrame containing the drift information with columns 'frame',
        'x', 'y', and optionally 'z'. Returns None if the file cannot be
        loaded.

    Raises
    ------
    ValueError
        If the path does not end with .txt.
    AssertionError
        If the loaded drift data does not have the expected format (2D
        array with 2 or 3 columns).
    """
    if not path.endswith(".txt"):
        raise ValueError("Drift file must end with .txt")
    drift = np.loadtxt(path, delimiter=" ")
    assert drift.ndim == 2 and drift.shape[1] in [2, 3], (
        "Drift must be a 2D array with 2 or 3 columns (x, y, (z)). "
        f"Loaded array has shape {drift.shape}."
    )
    drift_df = pd.DataFrame(drift[:, :2], columns=["x", "y"])
    if drift.shape[1] == 3:
        drift_df["z"] = drift[:, 2]
    return drift_df


def load_user_settings() -> lib.AutoDict:
    """Load user settings from a YAML file containing information such
    as the default directory for loading/saving files, Render color map,
    Localize parameters, etc.

    Returns
    -------
    settings : lib.AutoDict
        The loaded user settings.
    """
    settings_filename = _user_settings_filename()
    settings = None
    try:
        settings_file = open(settings_filename, "r")
    except FileNotFoundError:
        return lib.AutoDict()
    try:
        settings = yaml.load(settings_file, Loader=yaml.FullLoader)
        settings_file.close()
    except Exception as e:
        print(e)
        print("Error reading user settings, Reset.")
    if not settings:
        return lib.AutoDict()
    return lib.AutoDict(settings)


def save_info(
    path: str,
    info: list[dict],
    default_flow_style: bool = False,
) -> None:
    """Save metadata to a YAML file.

    Parameters
    ----------
    path : str
        The path to the YAML file where metadata will be saved.
    info : list of dict
        A list containing a dictionary with metadata about the movie.
    default_flow_style : bool, optional
        If True, the YAML will be written in flow style; otherwise, it
        will be written in block style.
    """
    with open(path, "w") as file:
        yaml.dump_all(info, file, default_flow_style=default_flow_style)


def _to_dict_walk(node: dict) -> dict:
    """Convert mapping objects (subclassed from dict) to actual dict
    objects, including nested ones."""
    node = dict(node)
    for key, val in node.items():
        if isinstance(val, dict):
            node[key] = _to_dict_walk(val)
    return node


def save_user_settings(settings: dict) -> None:
    """Save user settings to ``~/.picasso/settings.yaml``.

    For example, the default directory for loading and saving files.

    Parameters
    ----------
    settings : dict
        The settings to save; nested mappings are converted to plain dicts
        first so PyYAML does not tag them.
    """
    settings = _to_dict_walk(settings)
    settings_filename = _user_settings_filename()
    os.makedirs(os.path.dirname(settings_filename), exist_ok=True)
    with open(settings_filename, "w") as settings_file:
        yaml.dump(dict(settings), settings_file, default_flow_style=False)


def _save_metadata_in_yaml() -> bool:
    """Whether to also write the sidecar ``.yaml`` metadata file when
    saving localizations.

    Metadata is always embedded in the HDF5 file itself (see
    ``_write_metadata_dataset``); this setting only controls whether the
    convenience ``.yaml`` copy is written as well. Defaults to True.
    When the setting is absent, the default is persisted to the user
    settings file so it becomes visible and editable.

    Returns
    -------
    bool
        True if the ``.yaml`` metadata file should be written.
    """
    settings = load_user_settings()
    # cannot rely on truthiness: AutoDict auto-creates an empty (falsy)
    # dict for a missing key, so check membership explicitly.
    if "Save metadata in .yaml" not in settings:
        settings["Save metadata in .yaml"] = True
        save_user_settings(settings)
    return bool(settings["Save metadata in .yaml"])


def _save_picks_in_metadata() -> bool:
    """Whether to embed the picked regions (pick shape, size and
    positions) in the metadata when saving picked localizations from
    Render.

    The regions are stored in the same format as a picks ``.yaml`` file
    (see ``load_picks``), so the picks used to generate a file can be
    recovered from it. This can add a lot of data to the metadata (one
    entry per pick, and polygonal picks store every vertex), so it
    defaults to False. When the setting is absent, the default is
    persisted to the user settings file so it becomes visible and
    editable.

    Returns
    -------
    bool
        True if the picked regions should be saved in the metadata.
    """
    settings = load_user_settings()
    # cannot rely on truthiness: AutoDict auto-creates an empty (falsy)
    # dict for a missing key, so check membership explicitly.
    if "Save picks in metadata" not in settings:
        settings["Save picks in metadata"] = False
        save_user_settings(settings)
    return bool(settings["Save picks in metadata"])


#: File formats that the color bar (LUT) of a rendered property can be
#: saved in, see ``colorbar_export_format``. The first one is the
#: default.
COLORBAR_FORMATS = (".png", ".svg")


def colorbar_export_format() -> str:
    """Format of the color bar (LUT) that Render saves next to an image
    exported while rendering by property.

    ``.png`` (the default) saves it as an image, ``.svg`` as a vector
    graphic whose bands, ticks and text stay editable in figure
    software. Read from ``Colorbar format`` in the ``Render`` section of
    the user settings, next to the colormaps it belongs with; when the
    setting is absent, the default is persisted to the user settings
    file so it becomes visible and editable.

    Returns
    -------
    extension : str
        One of ``COLORBAR_FORMATS``.
    """
    settings = load_user_settings()
    # a "Render" section loaded from the file is a plain dict, so a
    # missing key raises instead of auto-creating; and truthiness cannot
    # be relied on, as AutoDict auto-creates an empty (falsy) dict for
    # the section itself
    render_settings = settings["Render"]
    if "Colorbar format" not in render_settings:
        render_settings["Colorbar format"] = COLORBAR_FORMATS[0]
        save_user_settings(settings)
    extension = str(render_settings["Colorbar format"]).strip().lower()
    if not extension.startswith("."):
        extension = "." + extension
    if extension not in COLORBAR_FORMATS:
        warnings.warn(
            f"Unknown color bar format '{extension}' in the user settings "
            f"(Render -> Colorbar format). Expected one of "
            f"{', '.join(COLORBAR_FORMATS)}; using "
            f"{COLORBAR_FORMATS[0]} instead."
        )
        return COLORBAR_FORMATS[0]
    return extension


#: Key holding the (often very large) block of MicroManager properties
#: read from a movie, see ``_mm_metadata_from_tifffile``.
_MM_METADATA_KEY = "Micro-Manager Metadata"


def _save_mm_metadata() -> bool:
    """Whether to carry the MicroManager metadata block over from a
    movie into the metadata of the localizations fitted from it.

    MicroManager movies carry a large block of microscope properties
    (``Micro-Manager Metadata``), which Picasso copies from the movie
    metadata into the localizations. It is needed to read the camera
    settings while localizing, but it makes the metadata of the saved
    localizations long and hard to read, so users can switch it off.
    Defaults to True, i.e., the block is kept as before. When the
    setting is absent, the default is persisted to the user settings
    file so it becomes visible and editable.

    Returns
    -------
    bool
        True if the MicroManager metadata should be kept.
    """
    settings = load_user_settings()
    # cannot rely on truthiness: AutoDict auto-creates an empty (falsy)
    # dict for a missing key, so check membership explicitly.
    if "Save Micro-Manager metadata" not in settings:
        settings["Save Micro-Manager metadata"] = True
        save_user_settings(settings)
    return bool(settings["Save Micro-Manager metadata"])


def strip_mm_metadata(info: list[dict]) -> list[dict]:
    """Remove the MicroManager metadata block from movie metadata if the
    user settings ask for it (``Save Micro-Manager metadata``).

    The input is not modified: the entries holding the block are copied
    without it, so the movie metadata kept in memory (used, e.g., to
    read the camera settings while localizing) is unaffected.

    Parameters
    ----------
    info : list of dict
        Movie metadata, about to be carried over into the metadata of
        the localizations.

    Returns
    -------
    info : list of dict
        The metadata without the ``Micro-Manager Metadata`` keys, or the
        input unchanged if the setting is on (the default).
    """
    if _save_mm_metadata():
        return info
    return [
        (
            {k: v for k, v in entry.items() if k != _MM_METADATA_KEY}
            if isinstance(entry, dict) and _MM_METADATA_KEY in entry
            else entry
        )
        for entry in info
    ]


def _write_metadata_dataset(hdf_file: h5py.File, info: list[dict]) -> bool:
    """Embed metadata in an open HDF5 file as a JSON string dataset at
    ``/metadata``.

    Parameters
    ----------
    hdf_file : h5py.File
        An HDF5 file opened in write mode.
    info : list of dict
        Metadata to embed.

    Returns
    -------
    bool
        True if the metadata was embedded. False if it could not be
        serialized to JSON, in which case the callers must fall back to
        writing the sidecar ``.yaml`` so that the file stays readable.
    """
    try:
        payload = json.dumps(list(info), default=_json_default)
    except (TypeError, ValueError) as e:
        warnings.warn(
            "Could not embed the metadata in the HDF5 file "
            f"({type(e).__name__}: {e}). Writing the .yaml metadata file "
            "instead; keep it next to the .hdf5 file.",
            stacklevel=3,
        )
        return False
    hdf_file.create_dataset("metadata", data=payload)
    return True


def _load_info_from_hdf5(path: str) -> list[dict] | None:
    """Load metadata embedded in the ``/metadata`` dataset of an HDF5
    file.

    Parameters
    ----------
    path : str
        The path to the HDF5 file.

    Returns
    -------
    info : list of dict or None
        The embedded metadata, or None if the file is not an HDF5 file or
        does not contain a ``/metadata`` dataset.
    """
    try:
        raw = _read_metadata_dataset(path)
    except OSError:
        # HDF5 files on network storage can fail to open with the default
        # file locking; the dataset itself is fine, so retry unlocked.
        try:
            raw = _read_metadata_dataset(path, locking=False)
        except (OSError, KeyError):
            return None
    except KeyError:
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        info = list(json.loads(raw))
    except (ValueError, TypeError) as e:
        warnings.warn(
            f"The metadata embedded in {path} could not be read "
            f"({type(e).__name__}: {e})."
        )
        return None
    return info


def _read_metadata_dataset(path: str, **kwargs) -> bytes | str | None:
    """Return the raw contents of the ``/metadata`` dataset of an HDF5
    file, or None if the file does not have one. ``kwargs`` are passed on
    to ``h5py.File``."""
    with h5py.File(path, "r", **kwargs) as hdf_file:
        if "metadata" not in hdf_file:
            return None
        return hdf_file["metadata"][()]


class AbstractPicassoMovie(abc.ABC):
    """An abstract class defining the minimal interfaces of a
    PicassoMovie used throughout Picasso."""

    @abc.abstractmethod
    def __init__(self):
        self.use_dask = False

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def info(self):
        """Read the movie's metadata.

        Returns
        -------
        info : list of dicts or dict
            Movie metadata, with at least "Byte Order", "Data Type",
            "Frames", "Height" and "Width".
        """
        pass

    @abc.abstractmethod
    def camera_parameters(self, config: dict) -> dict:
        """Get the camera specific parameters:
            * gain
            * quantum efficiency
            * wavelength
        These parameters depend on camera settings (as described in metadata)
        but the values themselves are given in the config.yaml file.
        Each filetype (nd2, ome-tiff, ..) has their own structure of metadata,
        which needs to be matched in the config.yaml description, as detailed
        in the specific child classes.

        Parameters
        ----------
        config : dict
            Description of camera parameters (for all possible settings)
            comes from the config.yaml file.

        Returns
        -------
        parameters : dict
            Keys: gain, qe, wavelength, cam_index, camera. Values are
            lists.
        """
        return {
            "gain": [1],
            "qe": [1],
            "wavelength": [0],
            "cam_index": 0,
            "camera": "None",
        }

    @abc.abstractmethod
    def __getitem__(self, it):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        return self.n_frames

    def close(self):
        """Release the file handle(s) the movie holds. A no-op by default."""
        pass

    @abc.abstractmethod
    def get_frame(self, index: int) -> lib.IntArray2D:
        """Load one frame of the movie.

        Parameters
        ----------
        index : int
            Frame index.

        Returns
        -------
        frame : lib.IntArray2D
            ``(height, width)`` raw camera counts.
        """
        pass

    @abc.abstractmethod
    def tofile(self, file_handle, byte_order=None):
        """Write the whole movie to an open file as raw binary.

        Parameters
        ----------
        file_handle : file object
            Destination, opened in binary write mode.
        byte_order : str, optional
            Byte order to write in (``"<"`` or ``">"``). None keeps the
            movie's own. Default None.
        """
        pass

    @property
    @abc.abstractmethod
    def dtype(self):
        """The dtype of the frames this movie yields."""
        return "u16"


class ND2Movie(AbstractPicassoMovie):
    """Subclass of the AbstractPicassoMovie to implement reading Nikon
    nd2 files.

    This class implements a version which uses only ``nd2``."""

    def __init__(self, path: str, verbose: bool = False, channel: int = 0):
        super().__init__()
        if verbose:
            print("Reading info from {}".format(path))
        self.path = os.path.abspath(path)
        self.nd2file = nd2.ND2File(path)
        self.dask = self.nd2file.to_dask()
        self.sizes = self.nd2file.sizes

        for dim in ["Y", "X"]:  # always required
            if dim not in self.nd2file.sizes.keys():
                raise KeyError(
                    "Required dimension {:s} not in file {:s}".format(
                        dim, self.path
                    )
                )
        # Movies are indexed along T. Files without a time axis (e.g. z
        # stacks) are read along Z instead, treating each slice as a
        # frame.
        if "T" in self.nd2file.sizes:
            self.frame_axis = "T"
        elif "Z" in self.nd2file.sizes:
            self.frame_axis = "Z"
        else:
            raise KeyError(
                "Neither dimension T nor Z in file {:s}".format(self.path)
            )
        # Allow an optional channel (C) axis; reject any other extra
        # dimension (e.g. the unused one of T/Z, or P), as before.
        allowed_dims = {self.frame_axis, "Y", "X", "C"}
        extra_dims = set(self.nd2file.sizes.keys()) - allowed_dims
        if extra_dims:
            raise KeyError(
                "File {:s} has unsupported dimensions {:s}; only T (or Z), "
                "Y, X and C are supported.".format(
                    self.path, str(sorted(extra_dims))
                )
            )

        # Channel selection. Single-channel files default to channel 0, so
        # their behavior is unchanged.
        self.n_channels = int(self.nd2file.sizes.get("C", 1))
        self._channel = channel if 0 <= channel < self.n_channels else 0
        self.channels = [f"Channel {i}" for i in range(self.n_channels)]
        try:
            names = [c.channel.name for c in self.nd2file.metadata.channels]
            if len(names) == self.n_channels and all(names):
                self.channels = [str(n) for n in names]
        except Exception:
            pass

        # Pixel access only needs the dimensions checked above; parsing
        # the (often vendor-specific) metadata may still fail. Keep that
        # failure recoverable so the movie can be loaded with manually
        # entered metadata (info() then returns None).
        try:
            self.meta = self.get_metadata(self.nd2file)
        except Exception:
            self.meta = None
        self._shape = [
            self.nd2file.sizes[self.frame_axis],
            self.nd2file.sizes["X"],
            self.nd2file.sizes["Y"],
        ]

    def info(self) -> dict:
        if self.meta is None:
            return None
        info = dict(self.meta)
        info["Channel"] = self.channels[self._channel]
        return info

    def get_metadata(self, nd2file: nd2.ND2File) -> dict:
        """Bring the file metadata in a readable form, and preprocesses
        it for easier downstream use.

        Parameters
        ----------
        nd2file : nd2.ND2File
            Object holding the image incl metadata.

        Returns
        -------
        info : dict
            Metadata.
        """
        info = {
            # "Byte Order": self._tif_byte_order,
            "File": self.path,
            "Height": nd2file.sizes["Y"],
            "Width": nd2file.sizes["X"],
            "Data Type": nd2file.dtype.name,
            "Frames": nd2file.sizes[self.frame_axis],
        }
        info["Acquisition Comments"] = ""

        mm_info = self.metadata_to_dict(nd2file)
        camera_name = str(
            mm_info.get("description", {})
            .get("Metadata", {})
            .get("Camera Name", "None")
        )
        info["Camera"] = camera_name

        # simulate micro manager camera data for loading config values
        # see picasso/gui/localize:680ff
        # put into camera config
        # 'Sensitivity Categories': ['PixelReadoutRate', 'ReadoutMode']
        # 'Sensitivity':
        #     '540 MHz':
        #         'Rolling Shutter at 16-bit': sensitivityvalue
        # 'Channel Device':
        #     'Name': 'Filter'
        #     'Emission Wavelengths':
        #         '2 (560)': 560
        readout_rate = str(
            mm_info.get("description", {})
            .get("Metadata", {})
            .get("Camera Settings", {})
            .get("Readout Rate", "None")
        )
        readout_mode = str(
            mm_info.get("description", {})
            .get("Metadata", {})
            .get("Camera Settings", {})
            .get("Readout Mode", "None")
        )
        conversion_gain = str(
            mm_info.get("description", {})
            .get("Metadata", {})
            .get("Camera Settings", {})
            .get("Conversion Gain", "None")
        )
        filter = str(
            mm_info.get("description", {})
            .get("Metadata", {})
            .get("Camera Settings", {})
            .get("Microscope Settings", {})
            .get("Nikon Ti2, FilterChanger(Turret-Lo)", "None")
        )

        sensitivity_category = "PixelReadoutRate"
        sensitivity_category2 = "Sensitivity/DynamicRange"
        info["Micro-Manager Metadata"] = {
            camera_name + "-" + sensitivity_category: readout_rate,
            camera_name
            + "-"
            + sensitivity_category2: (readout_mode + " " + conversion_gain),
            "Filter": filter,
        }
        info["Picasso Metadata"] = {
            "Camera": camera_name,
            "PixelReadoutRate": readout_rate,
            "ReadoutMode": readout_mode,
            "ConversionGain": conversion_gain,
            "Filter": filter,
        }
        info["nd2 Metadata"] = mm_info

        return info

    def metadata_to_dict(self, nd2file: nd2.ND2File) -> dict:
        """Extract all types of metadata in the file and returns it in
        a dict.

        Parameters
        ----------
        nd2file : nd2.ND2File
            Object holding the image incl metadata.

        Returns
        -------
        mmmeta : dict
            Metadata.
        """
        mmmeta = {}

        text_info = nd2file.text_info
        try:
            mmmeta["capturing"] = self.nikontext_to_dict(
                text_info["capturing"]
            )
        except Exception:
            pass
        try:
            mmmeta["AcquisitionDate"] = text_info["date"]
        except Exception:
            pass
        try:
            mmmeta["description"] = self.nikontext_to_dict(
                text_info["description"]
            )
        except Exception:
            pass
        try:
            mmmeta["optics"] = self.nikontext_to_dict(text_info["optics"])
        except Exception:
            pass

        mmmeta["custom_data"] = nd2file.custom_data
        mmmeta["attributes"] = nd2file.attributes._asdict()
        mmmeta["metadata"] = self.nd2metadata_to_dict(nd2file.metadata)

        return mmmeta

    @classmethod
    def nikontext_to_dict(cls, text: str) -> dict:
        """Some kinds of Nikon metadata are described with text, using
        newlines and colons. This function restructures the text into
        a dict.

        Parameters
        ----------
        text : str
            Nikon-style metadata description text.

        Returns
        -------
        out : dict
            Restructured text.
        """
        out = {}
        curr_keys = []
        for i, item in enumerate(text.split("\r\n")):
            itparts = item.split(":")
            itparts = [it.strip() for it in itparts if it.strip() != ""]
            if len(itparts) == 1:
                curr_keys.append(itparts[0])
                cls.set_nested_dict_entry(out, curr_keys, {})
            elif len(itparts) == 2:
                cls.set_nested_dict_entry(
                    out, curr_keys + [itparts[0]], itparts[1]
                )
            elif len(itparts) == 3:
                curr_keys.append(itparts[0])
                cls.set_nested_dict_entry(out, curr_keys, {})
                cls.set_nested_dict_entry(
                    out, curr_keys + [itparts[1]], itparts[2]
                )
            elif len(itparts) > 3:
                curr_keys.append(itparts[0])
                cls.set_nested_dict_entry(out, curr_keys, {})
                cls.set_nested_dict_entry(out, curr_keys + [itparts[1]], item)
                # raise KeyError(
                #     'Cannot parse three or more colons between newlines: ' +
                #     item)
        return out

    @classmethod
    def nd2metadata_to_dict(cls, meta: dict) -> dict:
        """Restructure the 'metadata' field from the package nd2 into a
        dict for independent use.
        https://github.com/tlambert03/nd2/blob/main/src/nd2/structures.py

        Parameters
        ----------
        meta : nd2 metadata structure
            The 'metadata' part of nd2 metadata.

        Returns
        -------
        out : dict
            The content as a dict.
        """
        out = {}
        out["contents"] = meta.contents.__dict__
        chans = [{}] * len(meta.channels)
        for i, chan in enumerate(meta.channels):
            chans[i] = chan.__dict__
            metachan = chan.__dict__["channel"].__dict__
            chans[i]["channel"] = {}
            for k, v in metachan.items():
                chans[i]["channel"][str(k)] = str(v)
            chans[i]["loops"] = chan.__dict__["loops"].__dict__
            chans[i]["microscope"] = chan.__dict__["microscope"].__dict__
            chans[i]["volume"] = chan.__dict__["volume"].__dict__
            axints = chans[i]["volume"]["axesInterpretation"]
            chans[i]["volume"]["axesInterpretation"] = [None] * len(axints)
            for j, axes_inter in enumerate(axints):
                chans[i]["volume"]["axesInterpretation"][j] = axes_inter
        out["channels"] = chans
        return out

    @classmethod
    def set_nested_dict_entry(cls, dict: dict, keys: list, val: any) -> None:
        """Set a value (deep) in a nested dict.

        Parameters
        ----------
        dict : dict
            The nested dict.
        keys : list
            The keys leading to the entry to set.
        val : anything
            The value to set.
        """
        currlvl = dict
        for i, key in enumerate(keys[:-1]):
            try:
                currlvl = currlvl[key]
            except KeyError:
                currlvl[key] = {}
                currlvl = currlvl[key]
        currlvl[keys[-1]] = val

    def __enter__(self) -> ND2Movie:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it: int) -> lib.IntArray2D:
        return self.get_frame(it)

    def __iter__(self):
        for i in range(self.sizes[self.frame_axis]):
            yield self[i]

    def __len__(self):
        return self.sizes[self.frame_axis]

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_frames, height, width)`` of the movie."""
        return self._shape

    def close(self):
        self.nd2file.close()

    def get_frame(self, index: int) -> lib.IntArray2D:
        """Load one frame of the movie at the selected channel.

        Parameters
        ----------
        index : int
            The frame index to retrieve.

        Returns
        -------
        frame : lib.IntArray2D
            2D array representing the image data of the frame
        """
        # Index the dask array along each axis in its native order. For a
        # plain (T, Y, X) file this reduces to ``self.dask[index]``.
        selection = []
        for dim in self.sizes:
            if dim == self.frame_axis:
                selection.append(index)
            elif dim == "C":
                selection.append(self._channel)
            else:  # Y, X
                selection.append(slice(None))
        return np.squeeze(self.dask[tuple(selection)].compute())

    def tofile(self, file_handle, byte_order=None):
        raise NotImplementedError("Cannot write .nd2 file.")

    def camera_parameters(self, config):  # noqa: C901
        """Get the camera specific parameters:
            * gain
            * quantum efficiency
            * wavelength
        These parameters depend on camera settings (as described in metadata)
        but the values themselves are given in the config.yaml file.
        Each filetype (nd2, ome-tiff, ..) has their own structure of metadata,
        which needs to be matched in the config.yaml description, as detailed
        in the specific child classes.

        The config file for the corresponding camera should look like this:
          Zyla 4.2:
            Pixelsize: 130
            Baseline: 100
            Quantum Efficiency:
              525: 0.7
              595: 0.72
              700: 0.64
            Sensitivity Categories:
              - PixelReadoutRate
              - ReadoutMode
            Sensitivity:
              540 MHz:
                Rolling Shutter at 16-bit: 7.18
              200 MHz:
                Rolling Shutter at 16-bit: 0.45
            Filter Wavelengths:
                1-R640: 700
                2-G561: 595
                3-B489: 525

        Parameters
        ----------
        config : dict
            Description of camera parameters (for all possible
            settings).

        Returns
        -------
        parameters : dict
            Keys: gain, qe, wavelength, cam_index, camera. Values are
            lists.
        """
        parameters = {}
        info = self.meta

        try:
            assert "Cameras" in config.keys() and "Camera" in info.keys()
        except Exception:
            raise KeyError("'camera' key not found in metadata or config.")

        cameras = config["Cameras"]
        camera = info["Camera"]

        try:
            assert camera in cameras.keys()
        except Exception:
            raise KeyError("camera from metadata not found in config.")

        index = sorted(list(cameras.keys())).index(camera)
        parameters["cam_index"] = index
        parameters["camera"] = camera

        try:
            assert "Picasso Metadata" in info
        except Exception:
            return {"gain": [1], "qe": [1], "wavelength": [0], "cam_index": 0}

        pm_info = info["Picasso Metadata"]
        # mm_info = info["nd2 Metadata"]
        cam_config = config["Cameras"][camera]
        if "Gain Property Name" in cam_config:
            raise NotImplementedError(
                "Extracting Gain from nd2 files is not implemented yet."
            )
        if "gain" not in parameters.keys():
            parameters["gain"] = [1]

        parameters["Sensitivity"] = {}
        if "Sensitivity Categories" in cam_config:
            categories = cam_config["Sensitivity Categories"]
            for _, category in enumerate(categories):
                parameters["Sensitivity"][category] = pm_info[category]
        if "Quantum Efficiency" in cam_config:
            if "Filter Wavelengths" in cam_config:
                channel = pm_info["Filter"]
                channels = cam_config["Filter Wavelengths"]
                if channel in channels:
                    wavelength = channels[channel]
                    parameters["wavelength"] = str(wavelength)
                    parameters["qe"] = cam_config["Quantum Efficiency"][
                        wavelength
                    ]
        if "qe" not in parameters.keys():
            parameters["qe"] = [1]
        if "wavelength" not in parameters.keys():
            parameters["wavelength"] = [0]
        return parameters

    @property
    def dtype(self):
        return np.dtype(self.meta["Data Type"])


class _MultiDimMovie(AbstractPicassoMovie):
    """Shared base for vendor formats (Zeiss .czi, Leica .lif) that store a
    multi-dimensional image which Picasso reduces to a single-channel
    ``(T, Y, X)`` movie.

    Subclasses open the file in ``__init__``, populate ``n_frames``,
    ``height``, ``width`` and ``_dtype``, call :meth:`_select_channel` to
    pick the channel, and implement :meth:`_read_plane` (return the 2D
    image of one time point at the selected channel) and :meth:`info`.
    The array-like interface, channel selection and frame validation are
    provided here. Mirrors the channel-prompt behavior of ``load_ims``.
    """

    def __init__(self):
        super().__init__()
        self.path = None
        self.n_frames = 0
        self.height = 0
        self.width = 0
        self._dtype = np.dtype("uint16")
        self.channels = ["Channel 0"]
        self._channel = 0

    def _select_channel(
        self,
        channels: list[str],
        prompt_info: Callable[[list[str]], str] | None,
        channel: int | None = None,
    ) -> None:
        """Store the available channels and pick one.

        When ``channel`` is given (an index), it is pinned directly and no
        prompt is shown - used to load every channel as an independent
        movie (see ``load_czi_all`` / ``load_lif_all``). Otherwise defaults
        to the first channel when there is only one or when no prompt is
        supplied (e.g. command-line batch processing), matching
        ``load_ims``.
        """
        self.channels = list(channels) if channels else ["Channel 0"]
        if channel is not None:
            self._channel = channel if 0 <= channel < len(self.channels) else 0
        elif len(self.channels) > 1 and prompt_info is not None:
            choice = prompt_info(self.channels)
            if choice in self.channels:
                self._channel = self.channels.index(choice)
            else:
                self._channel = 0
        else:
            self._channel = 0

    def _read_plane(self, index: int) -> np.ndarray:
        """Return the raw 2D image of time point ``index`` at the selected
        channel. Implemented by subclasses."""
        raise NotImplementedError

    def get_frame(self, index: int) -> lib.IntArray2D:
        """Load one frame of the movie as a 2D array."""
        if index < 0:
            index += self.n_frames
        frame = np.squeeze(np.asarray(self._read_plane(index)))
        if frame.ndim != 2:
            raise ValueError(
                f"Expected a 2D frame from {self.path}, got shape "
                f"{frame.shape}. Multi-sample/RGB frames are not supported."
            )
        return frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):
        if isinstance(it, slice):
            return np.stack(
                [self.get_frame(i) for i in range(*it.indices(self.n_frames))]
            )
        return self.get_frame(it)

    def __iter__(self):
        for i in range(self.n_frames):
            yield self.get_frame(i)

    def __len__(self) -> int:
        return self.n_frames

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_frames, height, width)`` of the movie."""
        return (self.n_frames, self.height, self.width)

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    def camera_parameters(self, config: dict) -> dict:
        """No camera-specific calibration is derived from .czi/.lif
        metadata yet; return neutral defaults so localization proceeds with
        the parameters set in the GUI."""
        return {
            "gain": [1],
            "qe": [1],
            "wavelength": [0],
            "cam_index": 0,
            "camera": "None",
        }

    def tofile(self, file_handle, byte_order=None):
        raise NotImplementedError(
            f"Writing {type(self).__name__} data to a raw file is not "
            "supported."
        )

    def close(self):
        pass


class CZIMovie(_MultiDimMovie):
    """Read Zeiss CZI movies via the optional ``czifile`` library,
    presenting a single channel as a ``(T, Y, X)`` movie."""

    def __init__(
        self,
        path: str,
        prompt_info: Callable[[list[str]], str] | None = None,
        channel: int | None = None,
    ):
        super().__init__()
        self.path = os.path.abspath(path)
        self._czi = czifile.CziFile(path)
        # A scene is a dimension-aware CziImage. Single-scene files (the
        # SMLM norm) expose one; fall back to a view over all subblocks.
        scenes = self._czi.scenes
        if scenes:
            self._image = next(iter(scenes.values()))
        else:
            self._image = czifile.CziImage(
                self._czi, self._czi.subblock_directory
            )
        self.sizes = dict(self._image.sizes)
        if "Y" not in self.sizes or "X" not in self.sizes:
            self._czi.close()
            raise ValueError(
                f"CZI file {self.path} has no Y/X image axes; cannot read "
                "it as a movie."
            )
        self.height = int(self.sizes["Y"])
        self.width = int(self.sizes["X"])
        self.n_frames = int(self.sizes.get("T", 1))
        self._dtype = np.dtype(self._image.dtype)
        self._mpp = self._image.mpp  # (mpp-x, mpp-y) in micrometer or None

        n_channels = int(self.sizes.get("C", 1))
        channels = [f"Channel {i}" for i in range(n_channels)]
        try:
            names = list(self._image.channels.keys())
            if len(names) == n_channels:
                channels = [str(n) for n in names]
        except Exception:
            pass
        self._select_channel(channels, prompt_info, channel)

    def _read_plane(self, index: int) -> np.ndarray:
        # Pin every non-spatial axis to a single coordinate so asarray
        # returns one Y/X plane.
        selection = {}
        for dim in self.sizes:
            if dim in ("Y", "X"):
                continue
            elif dim == "T":
                selection[dim] = index
            elif dim == "C":
                selection[dim] = self._channel
            else:
                selection[dim] = 0
        return self._image(**selection).asarray()

    def info(self) -> dict:
        info = {
            "File": self.path,
            "Height": self.height,
            "Width": self.width,
            "Frames": self.n_frames,
            "Data Type": self._dtype.name,
            "Channel": self.channels[self._channel],
            "Generated by": "Picasso Localize (CZI)",
        }
        if self._mpp and self._mpp[0]:
            info["Pixelsize"] = round(float(self._mpp[0]) * 1000, 3)
        try:
            info["CZI Metadata"] = _yaml_safe(self._image.attrs)
        except Exception:
            pass
        return info

    def close(self):
        try:
            self._czi.close()
        except Exception:
            pass


class LIFMovie(_MultiDimMovie):
    """Read Leica LIF movies via the optional ``liffile`` library,
    presenting a single channel of one image series as a ``(T, Y, X)``
    movie."""

    def __init__(
        self,
        path: str,
        prompt_info: Callable[[list[str]], str] | None = None,
        channel: int | None = None,
    ):
        super().__init__()
        self.path = os.path.abspath(path)
        self._lif = liffile.LifFile(path)
        images = list(self._lif.images)
        if not images:
            self._lif.close()
            raise ValueError(f"LIF file {self.path} contains no images.")
        # A LIF file can hold several acquisitions; use the one with the
        # most time points (the actual movie).
        self._image = max(
            images, key=lambda im: (int(im.sizes.get("T", 1)), int(im.size))
        )
        self.image_name = self._image.name
        self.sizes = dict(self._image.sizes)
        if "Y" not in self.sizes or "X" not in self.sizes:
            self._lif.close()
            raise ValueError(
                f"LIF image '{self.image_name}' in {self.path} has no Y/X "
                "axes; cannot read it as a movie."
            )
        self.height = int(self.sizes["Y"])
        self.width = int(self.sizes["X"])
        self.n_frames = int(self.sizes.get("T", 1))
        self._dtype = np.dtype(self._image.dtype)
        # Outer (non-frame) dimensions to index when reading one plane. The
        # frame itself is the innermost Y/X (plus optional S for RGB), so those
        # are excluded here. Derive from ``sizes`` rather than ``frames.dims``
        # for compatibility with older ``liffile`` versions that lack the
        # ``frames`` accessor.
        self._outer_dims = tuple(
            d for d in self.sizes if d not in ("Y", "X", "S")
        )

        n_channels = int(self.sizes.get("C", 1))
        channels = [f"Channel {i}" for i in range(n_channels)]
        try:
            names = self._image.coords.get("C")
            if names is not None and len(names) == n_channels:
                channels = [str(n) for n in list(names)]
        except Exception:
            pass
        self._select_channel(channels, prompt_info, channel)

    def _read_plane(self, index: int) -> np.ndarray:
        indices = {}
        for dim in self._outer_dims:
            if dim == "T":
                indices[dim] = index
            elif dim == "C":
                indices[dim] = self._channel
            else:
                indices[dim] = 0
        return self._image.frame(**indices)

    def info(self) -> dict:
        info = {
            "File": self.path,
            "Height": self.height,
            "Width": self.width,
            "Frames": self.n_frames,
            "Data Type": self._dtype.name,
            "Channel": self.channels[self._channel],
            "Image": self.image_name,
            "Generated by": "Picasso Localize (LIF)",
        }
        pixelsize = self._pixelsize_nm()
        if pixelsize is not None:
            info["Pixelsize"] = pixelsize
        try:
            info["LIF Metadata"] = _yaml_safe(self._image.attrs)
        except Exception:
            pass
        return info

    def _pixelsize_nm(self) -> float | None:
        """Best-effort pixel size in nm from the X coordinate spacing.

        liffile stores physical coordinates in meters; only return a value
        when it is physically plausible to avoid auto-filling the GUI with
        garbage."""
        try:
            xs = self._image.coords.get("X")
            if xs is None or len(xs) < 2:
                return None
            spacing_nm = abs(float(xs[1]) - float(xs[0])) * 1e9
            if 1.0 <= spacing_nm <= 100000.0:
                return round(spacing_nm, 3)
        except Exception:
            pass
        return None

    def close(self):
        try:
            self._lif.close()
        except Exception:
            pass


def _yaml_safe(obj):
    """Coerce arbitrary metadata into YAML/JSON-serializable builtins so it
    can be stored in the movie info dict (which gets written to
    ``_locs.yaml``)."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _mm_metadata_from_tifffile(tif: "tifffile.TiffFile") -> dict:
    """Translate MicroManager metadata from a ``tifffile.TiffFile`` into
    the fields Picasso stores in its info dictionary.

    Returns a dict that may contain ``"Micro-Manager Metadata"``,
    ``"Camera"`` and ``"Micro-Manager Acquisition Comments"``. Missing
    keys simply mean the corresponding metadata was not present (e.g. for
    a non-MicroManager TIFF). All parsing is wrapped defensively so that a
    malformed or absent block never raises."""
    out = {}

    # Per-image MicroManager metadata lives in tag 51123 on the first IFD.
    try:
        raw = None
        tag = tif.pages[0].tags.get(51123)
        if tag is not None:
            raw = tag.value
        if isinstance(raw, (bytes, bytearray)):
            # Strip null bytes which MM 1.4.22 appends, then JSON-decode.
            raw = bytes(raw).strip(b"\0").decode(errors="replace")
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            # Flatten to ensure compatibility with MM 2.0, where every
            # value is nested as {"PropName": ..., "PropVal": ...}.
            mm_info = {}
            for key, val in raw.items():
                if key == "scopeDataKeys":
                    continue
                if isinstance(val, dict):
                    mm_info[key] = val.get("PropVal")
                else:
                    mm_info[key] = val
            out["Micro-Manager Metadata"] = mm_info
            out["Camera"] = mm_info.get("Camera", "None")
    except Exception:
        pass

    # Acquisition comments live in the file-level Comments/Summary block,
    # which tifffile parses into ``micromanager_metadata``.
    try:
        mm_file = tif.micromanager_metadata or {}
        comments_block = mm_file.get("Comments")
        if isinstance(comments_block, dict):
            summary = comments_block.get("Summary")
            # key is left out if the comments is empty
            if isinstance(summary, str) and summary.strip():
                out["Micro-Manager Acquisition Comments"] = summary.split("\n")
    except Exception:
        pass

    return out


class _PerThreadFileHandles:
    """Per-thread binary file handles for offset-based frame reading.

    A single shared handle forces every concurrent reader to serialize on
    one file position (seek + read is stateful), so on network storage the
    per-frame round-trips cannot overlap. Giving each thread its own handle
    lets the OS / network stack keep several reads in flight at once, which
    hides per-frame latency.

    Users must call :meth:`_init_handles` in ``__init__`` (after ``self.path``
    is set) and :meth:`_close_handles` in ``close``

    To put it simply, this architecture allows for reading the movie from
    the network storage even if the access to the latter is interupted.
    """

    def _init_handles(self) -> None:
        self._local = threading.local()
        self._open_handles = []
        self._handles_lock = threading.Lock()

    def _handle(self):
        """Return this thread's private binary file handle, opening one on
        first use."""
        handle = getattr(self._local, "file", None)
        if handle is None:
            handle = open(self.path, "rb")
            self._local.file = handle
            with self._handles_lock:
                self._open_handles.append(handle)
        return handle

    def _drop_handle(self) -> None:
        """Discard this thread's handle so the next read opens a fresh one."""
        handle = getattr(self._local, "file", None)
        if handle is None:
            return
        self._local.file = None
        with self._handles_lock:
            try:
                self._open_handles.remove(handle)
            except ValueError:
                pass
        try:
            handle.close()
        except OSError:
            pass

    def _read_into_at(self, offset: int, array: np.ndarray) -> None:
        """Fill ``array`` with the bytes at ``offset``, retrying once on a
        fresh handle.

        Handles are cached for the lifetime of the movie, so a handle can go
        stale long after it was opened - a network share that dropped its
        session, an external drive that reconnected. Windows reports those as
        a bare ``OSError: [Errno 22] Invalid argument`` (its CRT maps every
        Win32 error it has no entry for onto ``EINVAL``), and without a retry
        the stale handle would keep failing for the rest of the session.

        A handle closed underneath us (``ValueError: seek of closed file``)
        is retried the same way.

        A short read is an error too: ``readinto`` fills only part of a
        freshly allocated (uninitialized) array and returns quietly, which
        would hand uninitialized memory back as image data."""
        view = memoryview(array).cast("B")
        for attempt in (0, 1):
            try:
                handle = self._handle()
                handle.seek(offset)
                n_read = 0
                while n_read < len(view):
                    n = handle.readinto(view[n_read:])
                    if not n:  # end of file: the read is short
                        break
                    n_read += n
            except (OSError, ValueError):
                # Assume the handle went bad; retry once on a new one.
                self._drop_handle()
                if attempt:
                    raise
                continue
            if n_read < len(view):
                raise OSError(
                    f"Truncated read from {self.path}: expected "
                    f"{len(view)} bytes at offset {offset}, got {n_read}. "
                    "The file may be incomplete or still being written."
                )
            return

    def _close_handles(self) -> None:
        with self._handles_lock:
            for handle in self._open_handles:
                handle.close()
            self._open_handles = []
        self._local = threading.local()


def _index_pages(pages, progress=None) -> int:
    """Return the number of IFDs in ``pages``, walking the IFD chain in
    chunks so that a long scan stays reportable and abortable.

    ``len(pages)`` walks the whole chain of a TIFF in one uninterrupt-
    ible call - minutes for a movie with many frames on network storage,
    where every IFD costs a round trip. Reading a page only extends the
    chain up to that index (see ``tifffile.TiffPages._seek``), so the
    walk is done here one chunk of pages at a time, calling
    ``progress(done, 0)`` in between. The total is 0 because the frame
    count is precisely what is not known yet; raising inside ``progress``
    aborts the scan.
    """
    CHUNK = 2000  # IFDs walked per progress report
    if progress is None or not isinstance(pages, tifffile.TiffPages):
        return len(pages)
    if getattr(pages, "_indexed", False):  # chain already walked
        return len(pages)
    done = 0
    while True:
        try:
            pages[done + CHUNK - 1]  # extends the chain up to this page
        except IndexError:  # the chain ended within the last chunk
            break
        done += CHUNK
        progress(done, 0)
    return len(pages)


class TiffMap(_PerThreadFileHandles):
    """Read a single TIFF file and provide array-like access to its frames.

    Backed by :mod:`tifffile`, which robustly parses classic TIFF,
    BigTIFF, OME-TIFF and MicroManager files and others - including
    compressed, tiled and multi-strip variants that were not available
    before v0.11.0.

    Frames are read lazily, one at a time, so resident memory stays at a
    single frame even for multi-gigabyte movies. For the common case of
    an uncompressed, contiguous, single-strip page the frame is read
    directly from its file offset with ``np.fromfile``. Compressed or
    tiled pages fall back to ``page.asarray()``.

    For speed, pages are parsed as lightweight ``tifffile`` frames
    (``useframes``). A few OME-TIFF / ImageJ files append a stray
    trailing IFD that disagrees with the first one (strip count or
    width), which makes tifffile raise ``"incompatible keyframe"``;
    ``__init__`` then falls back to full, independent per-page parsing
    (slower to open but reads the file) and drops the stray IFD so
    ``n_frames`` matches the real number of image planes."""

    def __init__(self, path: str, verbose: bool = False, progress=None):
        """Open the TIFF file with tifffile and extract the geometry,
        data type and per-page layout needed for lazy frame access.

        ``progress`` is an optional ``callable(done, total)`` invoked as
        the per-page IFD scan in ``_build_offsets`` proceeds, so the GUI
        can show a smooth determinate bar while a single large movie is
        opened. It is throttled to at most ~200 calls per file."""
        if verbose:
            print("Reading info from {}".format(path))
        self.path = os.path.abspath(path)
        self._tif = tifffile.TiffFile(self.path)

        # Choose the per-frame list. Zeiss LSM interleaves a
        # reduced-size thumbnail IFD after every image, so tif.pages
        # would double-count; tifffile's LSM series excludes the
        # thumbnails and gives the true plane list. For every other
        # format use the IFDs physically present in this file
        # (TiffMultiMap assembles multi-file OME movies, and tif.series
        # can split compressed stacks into many series, so it is not
        # reliable in general). For tif.pages, parse the IFDs as
        # lightweight TiffFrames (only the essential offset tags per
        # page), which keeps opening a movie fast - important on network
        # storage where each extra per-page read is a round-trip.
        if self._tif.is_lsm:
            self._pages = self._tif.series[0].pages
        else:
            self._tif.pages.useframes = True
            self._pages = self._tif.pages

        page0 = self._pages[0]
        self.height = int(page0.imagelength)
        self.width = int(page0.imagewidth)
        bits = int(page0.bitspersample)
        # A genuine movie frame has the same array shape as page 0;
        # _build_offsets uses this to drop stray trailing IFDs.
        self._page_shape = tuple(page0.shape)

        # Picasso works internally with little-endian unsigned integers; the
        # file may be big-endian, so keep both the file dtype and the target.
        self._tif_byte_order = self._tif.byteorder  # "<" or ">"
        dtype_str = "u" + str(bits // 8)
        self.dtype = np.dtype(dtype_str)
        self._tif_dtype = np.dtype(self._tif_byte_order + dtype_str)

        self.frame_shape = (self.height, self.width)
        self.frame_size = self.height * self.width
        self._frame_nbytes = self.frame_size * self.dtype.itemsize

        # The fast np.fromfile path only applies to uncompressed data.
        self._uncompressed = int(page0.compression) == 1

        # ImageJ writes an uncompressed multi-plane stack as a single IFD
        # followed by every other plane's pixel data laid out
        # contiguously, recording the plane count in the ImageJ metadata
        # rather than as separate pages. tifffile then exposes only one
        # page, so without this the movie would be counted as a single
        # frame. When detected, _build_offsets derives every frame's
        # offset arithmetically from the first plane (see
        # _imagej_contiguous_count).
        self._imagej_planes = self._imagej_contiguous_count(page0)

        # Precompute every frame's byte offset and the true frame count
        # in a single pass over the IFDs. The offset table keeps
        # `get_frame` a pure seek + np.fromfile (one large sequential
        # read per frame) and avoids a per-frame IFD parse, which is
        # costly on network storage; it stays None for compressed /
        # tiled / multi-strip files, which fall back to tifffile's
        # decoder in get_frame.
        #
        # Building this touches pages 1..N as lightweight TiffFrames,
        # each validated against page 0 (the keyframe). Some OME-TIFF /
        # ImageJ files append a stray trailing IFD whose strip count or
        # width disagrees with page 0, which makes tifffile raise
        # "incompatible keyframe". In that case re-parse every IFD as a
        # full, independent TiffPage (no keyframe comparison) and drop
        # the stray IFD from the frame count - slower to open but reads
        # the file with the right number of frames, as the pre-tifffile
        # reader did.
        try:
            self._offsets, self.n_frames = self._build_offsets(progress)
        except RuntimeError:
            # Flipping useframes in place is cache-safe (Picasso never
            # enables tifffile's page cache, so no stale TiffFrames are
            # held) and keeps the already-discovered IFD offset list. The
            # LSM branch above uses full TiffPages and never raises here.
            self._tif.pages.useframes = False
            self._pages = self._tif.pages
            self._offsets, self.n_frames = self._build_offsets(progress)

        # Per-thread binary handles for the fast offset-based read path
        # (see _PerThreadFileHandles). The decode lock only guards the slow
        # tifffile-decode fallback used for compressed / tiled pages, which
        # is not reentrant.
        self._init_handles()
        self._decode_lock = threading.Lock()

    # Read frames concurrently without the shared identify lock; each
    # thread uses its own file handle (see ``_handle``).
    supports_concurrent_reads = True

    def _imagej_contiguous_count(self, page0) -> int | None:
        """Return the plane count of an ImageJ contiguous stack, else None.

        ImageJ stores an uncompressed multi-plane stack as one IFD whose
        pixel data is followed by every other plane's data back-to-back
        in the file, recording the plane count in the ImageJ metadata
        instead of as separate pages. tifffile then reports a single
        page, so the frames would otherwise be miscounted as 1. The true
        plane count comes from ``tif.series[0]`` (parsed from the small
        ImageJ ``ImageDescription``, not the potentially huge per-frame
        ``Labels`` block). Every non-YX axis (z / t / channel) is
        flattened into the frame axis, matching how a page-based reader
        would see the planes.

        Returns None - leaving the normal per-page logic in charge - for
        anything that is not an uncompressed, single-strip, single-page
        ImageJ stack, so genuine single-plane images and per-page stacks
        are unaffected.
        """
        if not getattr(self._tif, "is_imagej", False):
            return None
        if len(self._pages) != 1 or not self._uncompressed:
            return None
        # Fast contiguous reads need exactly one strip of the expected size.
        if (
            len(page0.dataoffsets) != 1
            or int(page0.databytecounts[0]) != self._frame_nbytes
        ):
            return None
        try:
            shape = tuple(int(d) for d in self._tif.series[0].shape)
        except Exception:
            return None
        if len(shape) < 3 or shape[-2:] != self._page_shape:
            return None
        n = 1
        for dim in shape[:-2]:
            n *= dim
        return n if n > 1 else None

    def _build_offsets(self, progress=None) -> tuple[list[int] | None, int]:
        """Return ``(offsets, n_frames)`` for the movie.

        ``n_frames`` is the number of genuine image planes: a stray
        trailing IFD whose array shape differs from page 0 (some
        MicroManager / ImageJ OME-TIFFs append one - the same mismatch
        that makes tifffile raise "incompatible keyframe") is dropped, so
        the count matches the data as the pre-tifffile reader did.

        ``offsets`` is each frame's byte offset for the fast np.fromfile
        path, or ``None`` when any frame is compressed / tiled /
        multi-strip (those are decoded by tifffile in get_frame).

        Iterating the pages creates a TiffFrame per page validated
        against the keyframe, so this is also where the "incompatible
        keyframe" RuntimeError surfaces; __init__ catches it and retries
        with full-page parsing, where the stray IFD has its real shape
        and is dropped below."""
        # Counting the pages walks the IFD chain, which is the whole cost
        # of opening a compressed movie (the branch below returns without
        # touching the pages again), so it is reported in chunks too.
        n_pages = _index_pages(self._pages, progress)

        # Throttle progress reports to at most ~200 per file so a
        # multi-thousand-frame movie does not flood the GUI event queue
        # with one cross-thread signal per page.
        step = max(1, n_pages // 200)

        def report(done: int) -> None:
            if progress is not None and (done % step == 0 or done == n_pages):
                progress(done, n_pages)

        if self._imagej_planes is not None:
            # ImageJ contiguous stack: one IFD, then every plane's data
            # laid out back-to-back from the first plane's offset. Derive
            # each frame's offset arithmetically instead of from
            # (non-existent) per-page IFDs. Guard against a truncated file
            # by keeping only the planes that physically fit.
            base = int(self._pages[0].dataoffsets[0])
            n = self._imagej_planes
            try:
                file_size = os.path.getsize(self.path)
                fit = (file_size - base) // self._frame_nbytes
                if fit < n:
                    n = max(fit, 0)
            except OSError:
                pass
            offsets = [base + i * self._frame_nbytes for i in range(n)]
            if progress is not None:
                progress(n, n)
            return offsets, n

        if not self._uncompressed:
            # Compressed / tiled: no fast offset path. In the lightweight
            # frame mode every frame reports page 0's shape, so a stray
            # IFD cannot be told apart without reading every IFD (costly
            # on network storage). Probe the first and last extra pages
            # so an incompatible one triggers the full-page fallback;
            # with full pages (after that fallback, or for LSM) the
            # shapes are real, so drop trailing mismatched IFDs.
            if (not self._tif.is_lsm) and self._tif.pages.useframes:
                if n_pages > 1:
                    _ = self._pages[1].dataoffsets
                    _ = self._pages[n_pages - 1].dataoffsets
                report(n_pages)
                return None, n_pages
            n_frames = 0
            for i, page in enumerate(self._pages):
                if tuple(page.shape) != self._page_shape:
                    break
                n_frames += 1
                report(i + 1)
            return None, n_frames

        # Uncompressed: one pass collects each frame's byte offset and
        # stops at the first IFD whose shape differs from page 0.
        offsets = []
        n_frames = 0
        fast = True
        for i, page in enumerate(self._pages):
            if tuple(page.shape) != self._page_shape:
                break
            n_frames += 1
            report(i + 1)
            if fast:
                data_offsets = page.dataoffsets
                byte_counts = page.databytecounts
                if (
                    len(data_offsets) == 1
                    and byte_counts[0] == self._frame_nbytes
                ):
                    offsets.append(int(data_offsets[0]))
                else:
                    # Valid frame, but not eligible for the fast path.
                    fast = False
        return (offsets if fast else None), n_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):  # noqa: C901
        # No shared lock here: get_frame is thread-safe on its own (each
        # thread reads through its private handle, and the compressed
        # fallback takes _decode_lock), so concurrent reads can overlap.
        if isinstance(it, tuple):
            if isinstance(it, int) or np.issubdtype(it[0], np.integer):
                return self[it[0]][it[1:]]
            elif isinstance(it[0], slice):
                indices = range(*it[0].indices(self.n_frames))
                stack = np.array([self.get_frame(_) for _ in indices])
                if len(indices) == 0:
                    return stack
                else:
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            elif it[0] == Ellipsis:
                stack = self[it[0]]
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
        elif isinstance(it, slice):
            indices = range(*it.indices(self.n_frames))
            return np.array([self.get_frame(_) for _ in indices])
        elif it == Ellipsis:
            return np.array([self.get_frame(_) for _ in range(self.n_frames)])
        elif isinstance(it, int) or np.issubdtype(it, np.integer):
            return self.get_frame(it)
        raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def info(self) -> dict:
        """Extract metadata from the TIFF file.

        Returns
        -------
        info : dict
            Byte order, file path, height, width, data type, number of
            frames, and - for MicroManager files - the MicroManager metadata,
            camera name and acquisition comments.
        """
        info = {
            "Byte Order": self._tif_byte_order,
            "File": self.path,
            "Height": self.height,
            "Width": self.width,
            "Data Type": self.dtype.name,
            "Frames": self.n_frames,
        }
        mm = _mm_metadata_from_tifffile(self._tif)
        # only carried over if the acquisition actually has a comment
        comments = mm.get("Micro-Manager Acquisition Comments")
        if comments:
            info["Micro-Manager Acquisition Comments"] = comments
        if "Micro-Manager Metadata" in mm:
            info["Micro-Manager Metadata"] = mm["Micro-Manager Metadata"]
            info["Camera"] = mm["Camera"]
        return info

    def get_frame(self, index: int) -> lib.IntArray2D:
        """Lazily load one frame of the TIFF movie (one frame in
        memory).

        Uncompressed, contiguous, single-strip pages are read directly
        from their precomputed file offset with ``np.fromfile`` (one
        large sequential read, no decode overhead and no per-frame IFD
        parse); all other layouts fall back to ``tifffile``'s decoder.

        Parameters
        ----------
        index : int
            Frame index.

        Returns
        -------
        frame : lib.IntArray2D
            ``(height, width)`` raw camera counts.
        """
        if self._offsets is not None:
            # Fast path: pure seek + read on this thread's own handle, no
            # tifffile access and no shared lock, so reads from different
            # threads overlap instead of serializing.
            # ``np.fromfile`` rejects a Python file object on some
            # numpy/Windows builds ("expected str, bytes or os.PathLike
            # object, not BufferedReader"); read into a preallocated
            # array instead, which works with any binary handle.
            frame = np.empty(self.frame_size, dtype=self._tif_dtype)
            self._read_into_at(self._offsets[index], frame)
            frame = frame.reshape(self.frame_shape)
        else:
            # Compressed / tiled / multi-strip pages: let tifffile decode
            # this single page (still lazy - other frames stay on disk).
            # tifffile's reader is not reentrant, so serialize the decode.
            with self._decode_lock:
                frame = np.asarray(self._pages[index].asarray())
        # Downstream code expects little-endian unsigned integers; astype
        # is a no-op (no copy) when the data is already in that order.
        return frame.astype(self.dtype, copy=False)

    def close(self) -> None:
        self._close_handles()
        self._tif.close()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = byte_order != self._tif_byte_order
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)


class STKMovie(AbstractPicassoMovie, _PerThreadFileHandles):
    """Read MetaMorph STK files and provide array-like access to frames.

    STK files are TIFF-based with a single IFD; additional frames are
    stored contiguously after the first frame's pixel data.  The total
    frame count is encoded in the UIC2Tag (tag 33629).

    ``tifffile`` is used once during ``__init__`` to extract metadata
    and the binary offset of the first frame; subsequent frame reads
    bypass tifffile and go directly to the file via offset arithmetic,
    matching the pattern of ``TiffMap``.
    """

    def __init__(self, path: str):
        super().__init__()
        self.path = os.path.abspath(path)

        # Use tifffile to extract metadata from the STK file.
        with tifffile.TiffFile(self.path) as tif:
            if not tif.is_stk:
                raise ValueError(
                    f"File does not appear to be a MetaMorph STK file: {path}"
                )
            meta = tif.stk_metadata
            page = tif.pages[0]

            self.n_frames = int(meta["NumberPlanes"])
            self.height = int(page.shape[0])
            self.width = int(page.shape[1])
            bits = int(page.bitspersample)
            byte_order = tif.byteorder  # '<' or '>'

            # All data offsets for every plane (tifffile resolves these
            # for STK files even though there is only one IFD).
            offsets = page.dataoffsets
            self._first_data_offset = int(offsets[0])
            self._contiguous = len(offsets) == 1

            # Store per-frame offsets (tifffile may already expand them
            # for multi-strip or multi-frame cases).
            # For standard STK files every frame is a single strip, so
            # we compute offsets ourselves from the first one.
            self._frame_bytes = self.height * self.width * (bits // 8)

            self._stk_meta = meta
            self._byte_order = byte_order

        dtype_str = "u" + str(bits // 8)
        self._dtype = np.dtype(dtype_str)  # always little-endian for Picasso
        self._tif_dtype = np.dtype(self._byte_order + dtype_str)
        self.frame_shape = (self.height, self.width)
        self.shape = (self.n_frames, self.height, self.width)

        # Per-thread binary handles for lazy frame reading; see
        # _PerThreadFileHandles.
        self._init_handles()

    # Read frames concurrently without the shared identify lock; each
    # thread uses its own file handle (see ``_handle``).
    supports_concurrent_reads = True

    # ------------------------------------------------------------------
    # AbstractPicassoMovie interface
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):  # noqa: C901
        # No shared lock: get_frame reads through this thread's private
        # handle, so concurrent reads overlap instead of serializing.
        if isinstance(it, tuple):
            if isinstance(it[0], int) or np.issubdtype(it[0], np.integer):
                return self[it[0]][it[1:]]
            elif isinstance(it[0], slice):
                indices = range(*it[0].indices(self.n_frames))
                stack = np.array([self.get_frame(_) for _ in indices])
                if len(indices) == 0:
                    return stack
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
            elif it[0] == Ellipsis:
                stack = self[it[0]]
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
        elif isinstance(it, slice):
            indices = range(*it.indices(self.n_frames))
            return np.array([self.get_frame(_) for _ in indices])
        elif it == Ellipsis:
            return np.array([self.get_frame(_) for _ in range(self.n_frames)])
        elif isinstance(it, int) or np.issubdtype(it, np.integer):
            return self.get_frame(it)
        raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self) -> int:
        return self.n_frames

    def info(self) -> dict:
        """Read the STK file's metadata.

        Returns
        -------
        info : dict
            Byte order, file path, height, width, data type and number of
            frames, plus whatever the STK header carries.
        """
        info = {
            "Byte Order": "<",
            "File": self.path,
            "Height": self.height,
            "Width": self.width,
            "Data Type": self._dtype.name,
            "Frames": self.n_frames,
        }
        meta = self._stk_meta
        if meta.get("SpatialCalibration"):
            x_cal = meta.get("XCalibration")
            units = meta.get("CalibrationUnits", "")
            if x_cal is not None:
                # x_cal is a float (tifffile already divides the rational)
                cal_value = float(x_cal)
                # Convert to nm
                if isinstance(units, bytes):
                    units = units.decode(errors="replace")
                units_lower = units.strip().lower()
                if units_lower in ("um", "µm", "\u00b5m"):
                    cal_nm = cal_value * 1000.0
                elif units_lower == "nm":
                    cal_nm = cal_value
                else:
                    cal_nm = cal_value  # store as-is
                info["Pixelsize"] = cal_nm
        return info

    def camera_parameters(self, config: dict) -> dict:
        return {
            "gain": [1],
            "qe": [1],
            "wavelength": [0],
            "cam_index": 0,
            "camera": "None",
        }

    def get_frame(self, index: int) -> lib.IntArray2D:
        """Load one frame from the STK file by binary offset.

        Parameters
        ----------
        index : int
            Frame index; negative values count from the end.

        Returns
        -------
        frame : lib.IntArray2D
            ``(height, width)`` raw camera counts.

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        if index < 0:
            index = self.n_frames + index
        if not (0 <= index < self.n_frames):
            raise IndexError(
                f"Frame index {index} out of range for movie with "
                f"{self.n_frames} frames."
            )
        offset = self._first_data_offset + index * self._frame_bytes
        # ``np.fromfile`` rejects a Python file object on some
        # numpy/Windows builds ("expected str, bytes or os.PathLike
        # object, not BufferedReader"); read into a preallocated array
        # instead, which works with any binary handle.
        frame = np.empty(self.height * self.width, dtype=self._tif_dtype)
        self._read_into_at(offset, frame)
        frame = frame.reshape(self.frame_shape)
        if self._byte_order == ">":
            frame = frame.byteswap().view(self._dtype)
        return frame

    def close(self) -> None:
        self._close_handles()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = byte_order != self._byte_order
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)

    @property
    def dtype(self):
        return self._dtype


class STKMultiMovie(AbstractPicassoMovie):
    """Read consecutive MetaMorph STK files as a single movie.

    When an STK file with a numeric suffix is opened (e.g.
    ``name_003.stk``), this class automatically discovers all files in
    the same directory that share the same base name and have an equal
    or higher numeric suffix, and presents them as one contiguous movie.
    If the filename does not contain a numeric suffix, only the single
    file is used.
    """

    def __init__(self, path: str):
        super().__init__()
        self.path = os.path.abspath(path)
        self.dir = os.path.dirname(self.path)

        # Detect trailing numeric suffix in the filename stem, e.g.
        # "GluN1_ms_Pos-1_003.stk" → file_base="GluN1_ms_Pos-1", start_idx=3
        stem = os.path.splitext(os.path.basename(self.path))[0]
        m = re.match(r"^(.+)_(\d+)$", stem)
        if m:
            file_base = m.group(1)
            start_idx = int(m.group(2))
            escaped_base = re.escape(os.path.join(self.dir, file_base))
            pattern = re.compile(escaped_base + r"_(\d+)\.stk$", re.IGNORECASE)
            entries = [e.path for e in os.scandir(self.dir) if e.is_file()]
            suffix_path_pairs = [
                (int(pattern.match(e).group(1)), e)
                for e in entries
                if pattern.match(e)
            ]
            self.paths = [
                p for idx, p in sorted(suffix_path_pairs) if idx >= start_idx
            ]
        else:
            self.paths = [self.path]

        self.maps = [STKMovie(p) for p in self.paths]
        self.n_maps = len(self.maps)
        self.n_frames_per_map = [_.n_frames for _ in self.maps]
        self.n_frames = sum(self.n_frames_per_map)
        self.cum_n_frames = np.insert(np.cumsum(self.n_frames_per_map), 0, 0)
        self._dtype = self.maps[0]._dtype
        self.height = self.maps[0].height
        self.width = self.maps[0].width
        self.shape = (self.n_frames, self.height, self.width)

    # Reads dispatch to the underlying STKMovie maps, which each manage
    # their own per-thread handles, so concurrent reads are safe.
    supports_concurrent_reads = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):  # noqa: C901
        if isinstance(it, tuple):
            if it[0] == Ellipsis:
                stack = self[it[0]]
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
            elif isinstance(it[0], slice):
                indices = range(*it[0].indices(self.n_frames))
                stack = np.array([self.get_frame(_) for _ in indices])
                if len(indices) == 0:
                    return stack
                else:
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            if isinstance(it[0], int) or np.issubdtype(it[0], np.integer):
                return self[it[0]][it[1:]]
        elif isinstance(it, slice):
            indices = range(*it.indices(self.n_frames))
            return np.array([self.get_frame(_) for _ in indices])
        elif it == Ellipsis:
            return np.array([self.get_frame(_) for _ in range(self.n_frames)])
        elif isinstance(it, int) or np.issubdtype(it, np.integer):
            return self.get_frame(it)
        raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def close(self):
        for map_ in self.maps:
            map_.close()

    @property
    def dtype(self):
        return self._dtype

    def get_frame(self, index: int) -> lib.IntArray2D:
        for i in range(self.n_maps):
            if self.cum_n_frames[i] <= index < self.cum_n_frames[i + 1]:
                break
        else:
            raise IndexError
        return self.maps[i][index - self.cum_n_frames[i]]

    def info(self) -> dict:
        info = self.maps[0].info()
        info["Frames"] = self.n_frames
        self.meta = info
        return info

    def camera_parameters(self, config: dict) -> dict:
        return {
            "gain": [1],
            "qe": [1],
            "wavelength": [0],
            "cam_index": 0,
            "camera": "None",
        }

    def tofile(self, file_handle, byte_order=None):
        for map_ in self.maps:
            map_.tofile(file_handle, byte_order)


def _scaled_progress(progress, j: int, n: int):
    """Return a per-component progress callback that reports into the
    ``j``-th of ``n`` equal slices of a composite progress bar.

    Movies assembled from several files (a multi-file OME set, or several
    files concatenated across folders) open one component at a time, each
    reporting its own ``(done, total)``. Weighting the slices by frame
    count is not possible before the components are opened, so every one
    gets an equal share. ``SCALE`` sub-steps per component keep a
    single-component movie animating smoothly across ``0..SCALE``.
    Returns None when ``progress`` is None (no reporting).
    """
    SCALE = 1000
    if progress is None:
        return None

    def callback(done, total):
        if total > 0:
            progress(int((j + done / total) * SCALE), n * SCALE)

    return callback


class TiffMultiMap(AbstractPicassoMovie):
    """Read ``.ome.tif`` files created by MicroManager. Single files are
    maxed out at 4GB, so this class orchestrates reading from single
    files, each accessed by ``TiffMap``."""

    def __init__(
        self,
        path: str,
        memmap_frames: bool = False,
        verbose: bool = False,
        progress=None,
    ):
        super().__init__()
        self.path = os.path.abspath(path)
        self.dir = os.path.dirname(self.path)

        # This matches the basename + an appendix of the file number
        filename = os.path.basename(self.path)
        if "NDTiffStack" in filename:
            # only one extension (.tif)
            base, ext = os.path.splitext(self.path)
            base = re.escape(base)
            pattern = re.compile(base + r"_(\d*).tif")
        else:
            # split two extensions as in .ome.tif
            base, ext = os.path.splitext(os.path.splitext(self.path)[0])
            base = re.escape(base)
            pattern = re.compile(base + r"_(\d*).ome.tif")
        entries = [_.path for _ in os.scandir(self.dir) if _.is_file()]
        matches = [re.match(pattern, _) for _ in entries]
        matches = [_ for _ in matches if _ is not None]
        paths_indices = [(int(_.group(1)), _.group(0)) for _ in matches]
        self.paths = [self.path] + [
            path for index, path in sorted(paths_indices)
        ]
        # A multi-file OME movie is opened as several TiffMaps; each gets
        # an equal slice of the composite progress range.
        n_maps = len(self.paths)
        self.maps = [
            TiffMap(
                path,
                verbose=verbose,
                progress=_scaled_progress(progress, j, n_maps),
            )
            for j, path in enumerate(self.paths)
        ]
        self.n_maps = len(self.maps)
        self.n_frames_per_map = [_.n_frames for _ in self.maps]
        self.n_frames = sum(self.n_frames_per_map)
        self.cum_n_frames = np.insert(np.cumsum(self.n_frames_per_map), 0, 0)
        self._dtype = self.maps[0].dtype
        self.height = self.maps[0].height
        self.width = self.maps[0].width
        self.shape = (self.n_frames, self.height, self.width)

    # Reads dispatch to the underlying TiffMap maps, which each manage
    # their own per-thread handles, so concurrent reads are safe.
    supports_concurrent_reads = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):  # noqa: C901
        if isinstance(it, tuple):
            if it[0] == Ellipsis:
                stack = self[it[0]]
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
            elif isinstance(it[0], slice):
                indices = range(*it[0].indices(self.n_frames))
                stack = np.array([self.get_frame(_) for _ in indices])
                if len(indices) == 0:
                    return stack
                else:
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            if isinstance(it[0], int) or np.issubdtype(it[0], np.integer):
                return self[it[0]][it[1:]]
        elif isinstance(it, slice):
            indices = range(*it.indices(self.n_frames))
            return np.array([self.get_frame(_) for _ in indices])
        elif it == Ellipsis:
            return np.array([self.get_frame(_) for _ in range(self.n_frames)])
        elif isinstance(it, int) or np.issubdtype(it, np.integer):
            return self.get_frame(it)
        raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def close(self):
        for map in self.maps:
            map.close()

    @property
    def dtype(self):
        return self._dtype

    def get_frame(self, index: int) -> lib.IntArray2D:
        # TODO deal with negative numbers
        for i in range(self.n_maps):
            if self.cum_n_frames[i] <= index < self.cum_n_frames[i + 1]:
                break
        else:
            raise IndexError
        return self.maps[i][index - self.cum_n_frames[i]]

    def info(self):
        info = self.maps[0].info()
        info["Frames"] = self.n_frames
        self.meta = info
        return info

    def camera_parameters(self, config: dict) -> dict:  # noqa: C901
        """Get the camera specific parameters:
            * gain
            * quantum efficiency
            * wavelength
        These parameters depend on camera settings (as described in metadata)
        but the values themselves are given in the config.yaml file.
        Each filetype (nd2, ome-tiff, ..) has their own structure of metadata,
        which needs to be matched in the config.yaml description, as detailed
        in the specific child classes.
        This code has been moved from localize to here, as it is file type
        specific (HG, April 2022).

        Parameters
        ----------
        config : dict
            Description of camera parameters (for all possible
            settings).

        Returns
        -------
        parameters : dict
            Keys: gain, qe, wavelength, cam_index, camera. Values are
            lists.
        """
        # return {'gain': [1], 'qe': [1], 'wavelength': [0], 'cam_index': 0}
        parameters = {}
        info = self.meta

        try:
            assert "Cameras" in config and "Camera" in info
        except Exception:
            return {"gain": [1], "qe": [1], "wavelength": [0], "cam_index": 0}
            # raise KeyError("'camera' key not found in metadata or config.")

        cameras = config["Cameras"]
        camera = info["Camera"]

        try:
            assert camera in list(cameras.keys())
        except Exception:
            return {"gain": [1], "qe": [1], "wavelength": [0], "cam_index": 0}
            # raise KeyError('camera from metadata not found in config.')

        index = sorted(list(cameras.keys())).index(camera)
        parameters["cam_index"] = index
        parameters["camera"] = camera

        try:
            assert "Micro-Manager Metadata" in info
        except Exception:
            return {"gain": [1], "qe": [1], "wavelength": [0], "cam_index": 0}

        mm_info = info["Micro-Manager Metadata"]
        cam_config = config["Cameras"][camera]
        # The properties named in the config need not be present in the
        # metadata, in which case the default values below are kept
        # instead of raising.
        if "Gain Property Name" in cam_config:
            gain_property_name = cam_config["Gain Property Name"]
            gain_property = camera + "-" + gain_property_name
            if gain_property in mm_info:
                gain = mm_info[gain_property]
                if "EM Switch Property" in cam_config:
                    switch_config = cam_config["EM Switch Property"]
                    switch_property = camera + "-" + switch_config["Name"]
                    if (
                        switch_property in mm_info
                        and mm_info[switch_property] == switch_config[True]
                    ):
                        parameters["gain"] = int(gain)
        if "gain" not in parameters.keys():
            parameters["gain"] = [1]
        parameters["Sensitivity"] = {}
        if "Sensitivity Categories" in cam_config:
            categories = cam_config["Sensitivity Categories"]
            for i, category in enumerate(categories):
                property_name = camera + "-" + category
                if property_name in mm_info:
                    exp_setting = mm_info[camera + "-" + category]
                    parameters["Sensitivity"][category] = exp_setting
        if "Quantum Efficiency" in cam_config:
            if "Channel Device" in cam_config:
                channel_device = cam_config["Channel Device"]
                channel_device_name = channel_device["Name"]
                channels = channel_device["Emission Wavelengths"]
                channel = mm_info.get(channel_device_name)
                if channel in channels:
                    wavelength = channels[channel]
                    parameters["wavelength"] = [str(wavelength)]
                    parameters["qe"] = [
                        cam_config["Quantum Efficiency"][wavelength]
                    ]
        if "qe" not in parameters.keys():
            parameters["qe"] = [1]
        if "wavelength" not in parameters.keys():
            parameters["wavelength"] = [0]
        return parameters

    def tofile(self, file_handle, byte_order=None):
        for map in self.maps:
            map.tofile(file_handle, byte_order)


# MicroManager can save an acquisition as one single-page TIFF per camera
# frame ("separate image files" / "Image files" mode) inside a folder,
# next to a ``metadata.txt``. The two naming schemes are:
#   * MM 2.0: img_channel000_position000_time000000000_z000.tif
#   * MM 1.4: img_000000000_Default_000.tif  (img_<frame>_<channel>_<z>)
# The frame axis is the time index (MM 2.0) or the leading index (MM
# 1.4). Channel, position and z are held fixed (they are part of the
# prefix/suffix below) so a single 2-D time series is assembled and
# multi-channel / multi-position sets are never interleaved into one
# movie.
_MM_SEPARATE_RES = (
    re.compile(
        r"^(?P<prefix>img_channel\d+_position\d+_time)\d+"
        r"(?P<suffix>_z\d+\.tif)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<prefix>img_)\d+(?P<suffix>_.+_\d+\.tif)$",
        re.IGNORECASE,
    ),
)


def _mm_separate_files(path: str) -> list[str] | None:
    """Return the ordered sibling frames of a MicroManager "separate
    image files" acquisition, or ``None`` if ``path`` is not part of one.

    Given one ``img_*.tif`` file, discover every sibling in the same
    folder that differs only in its frame (time) index - channel,
    position and z are held at the selected file's values - and return
    the paths sorted by that index. Returns ``None`` for a name that does
    not match the MicroManager separate-file convention, or when only a
    single matching file is present (a lone TIFF, opened as usual).
    """
    directory = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    for regex in _MM_SEPARATE_RES:
        m = regex.match(name)
        if m is None:
            continue
        # Siblings share the fixed prefix + suffix and vary only in the
        # frame index captured between them.
        sibling_re = re.compile(
            re.escape(m.group("prefix"))
            + r"(\d+)"
            + re.escape(m.group("suffix")),
            re.IGNORECASE,
        )
        indexed = []
        for entry in os.scandir(directory):
            if not entry.is_file():
                continue
            sm = sibling_re.fullmatch(entry.name)
            if sm is not None:
                indexed.append((int(sm.group(1)), entry.path))
        if len(indexed) <= 1:
            # A lone file is not a series; open it as an ordinary TIFF.
            return None
        indexed.sort()
        return [p for _, p in indexed]
    return None


class MMSeparateTiffMovie(TiffMultiMap):
    """Read a MicroManager acquisition saved as *separate image files*.

    In this mode every camera frame is written to its own single-page
    TIFF inside one folder (see :func:`_mm_separate_files` for the naming
    schemes), so a long movie can be tens of thousands of files. Unlike
    ``TiffMultiMap`` - which opens a ``TiffMap`` per file up front - this
    class reads geometry, dtype and metadata from the first frame only
    and takes the frame count from the number of files, so opening the
    movie stays O(1) instead of O(frames). Every other frame's file is
    opened lazily when that frame is first read and closed again right
    away, so the movie holds at most one extra file handle open and never
    exhausts the OS descriptor limit. Frame reads are independent, so
    they are safe to run concurrently.

    Each file is assumed to hold one plane of identical shape and dtype
    (always true for this MicroManager mode); the frame count comes from
    the file list, not from re-parsing each file. The array-like access,
    iteration and ``camera_parameters`` are inherited from
    ``TiffMultiMap``.
    """

    def __init__(self, paths: list[str], verbose: bool = False):
        # Deliberately skip TiffMultiMap.__init__ (which would eagerly
        # open a TiffMap per file); build the composite geometry from the
        # first frame only.
        AbstractPicassoMovie.__init__(self)
        if not paths:
            raise ValueError("No TIFF files given for a separate-files movie.")
        self.paths = list(paths)
        self.path = os.path.abspath(self.paths[0])
        self.dir = os.path.dirname(self.path)
        self.n_frames = len(self.paths)
        # Read geometry, dtype and metadata from the first frame only.
        self._first = TiffMap(self.path, verbose=verbose)
        self._dtype = self._first.dtype
        self.height = self._first.height
        self.width = self._first.width
        self.frame_shape = (self.height, self.width)
        self.shape = (self.n_frames, self.height, self.width)

    # Each read opens its own file handle, so concurrent reads are safe.
    supports_concurrent_reads = True

    def get_frame(self, index: int) -> lib.IntArray2D:
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"Frame {index} out of range for {self.n_frames} frames."
            )
        if index == 0:
            # Frame 0's file stays open for metadata/geometry; reuse it.
            return self._first.get_frame(0)
        # Open every other file lazily and close it right away so the
        # movie never holds thousands of file handles at once.
        with TiffMap(self.paths[index]) as tif:
            return tif.get_frame(0)

    def info(self):
        info = self._first.info()
        info["Frames"] = self.n_frames
        self.meta = info
        return info

    def close(self):
        self._first.close()

    def tofile(self, file_handle, byte_order=None):
        self._first.tofile(file_handle, byte_order)
        for path in self.paths[1:]:
            with TiffMap(path) as tif:
                tif.tofile(file_handle, byte_order)


def find_mm_separate_first(directory: str) -> str | None:
    """Return the first-frame path of a MicroManager "separate image
    files" acquisition inside ``directory``.

    Passing the returned path to :func:`load_tif` (or :func:`load_movie`)
    assembles the whole folder into a single movie, because those loaders
    detect the separate-files layout via :func:`_mm_separate_files`.

    Parameters
    ----------
    directory : str
        Folder to inspect.

    Returns
    -------
    path : str or None
        The first frame's path, or ``None`` if the folder holds no such
        sequence (or cannot be listed).
    """
    try:
        names = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_file()
        )
    except OSError:
        return None
    for name in names:
        if any(regex.match(name) for regex in _MM_SEPARATE_RES):
            candidate = os.path.join(directory, name)
            if _mm_separate_files(candidate) is not None:
                return candidate
    return None


def _open_tif_component(path: str, verbose: bool = False, progress=None):
    """Open one TIFF movie for concatenation, applying the same dispatch
    as :func:`load_tif`: a MicroManager "separate image files" frame
    opens its whole folder, anything else opens as a (possibly
    multi-file) ``TiffMultiMap``."""
    separate_paths = _mm_separate_files(path)
    if separate_paths is not None:
        return MMSeparateTiffMovie(separate_paths, verbose=verbose)
    return TiffMultiMap(
        path, memmap_frames=False, verbose=verbose, progress=progress
    )


class ConcatenatedTiffMovie(TiffMultiMap):
    """Concatenate several TIFF movies along the frame axis.

    One acquisition is sometimes saved as several TIFF files sitting in
    different folders. This opens them as a single movie whose frames run
    through the files in the given order, so it can be localized in one
    go.

    ``TiffMultiMap`` already dispatches frame reads through
    ``cum_n_frames`` into a list of components, so array-like access,
    iteration, ``camera_parameters``, ``tofile`` and ``close`` are all
    inherited. Only the components differ: they are supplied here instead
    of being discovered next to a single file, and each one is itself
    opened with ``TiffMultiMap`` so a component that is a multi-file OME
    set or an ImageJ stack still contributes all of its frames.

    Every component must share the first one's frame shape and dtype;
    otherwise a ``ValueError`` names the offending file. Metadata is
    taken from the first component, with ``Frames`` set to the total and
    the source paths recorded (see :meth:`info`).
    """

    def __init__(self, paths: list[str], verbose: bool = False, progress=None):
        # Deliberately skip TiffMultiMap.__init__, which would discover
        # the components by name next to a single file; here they are
        # given (mirroring how MMSeparateTiffMovie builds its own state).
        AbstractPicassoMovie.__init__(self)
        if not paths:
            raise ValueError("No TIFF files given to concatenate.")
        self.paths = [os.path.abspath(_) for _ in paths]
        self.path = self.paths[0]
        self.dir = os.path.dirname(self.path)
        n_paths = len(self.paths)
        self.maps = [
            _open_tif_component(
                path,
                verbose=verbose,
                progress=_scaled_progress(progress, j, n_paths),
            )
            for j, path in enumerate(self.paths)
        ]
        self.n_maps = len(self.maps)
        self.n_frames_per_map = [_.n_frames for _ in self.maps]
        self.n_frames = sum(self.n_frames_per_map)
        self.cum_n_frames = np.insert(np.cumsum(self.n_frames_per_map), 0, 0)
        self._dtype = self.maps[0].dtype
        self.height = self.maps[0].height
        self.width = self.maps[0].width
        self.shape = (self.n_frames, self.height, self.width)
        self._check_compatible()

    def _check_compatible(self) -> None:
        """Raise if any component's geometry or dtype differs from the
        first one's - concatenating those would silently produce a movie
        whose frames do not all mean the same thing."""
        expected = (self.height, self.width, self._dtype)
        for map_, path in zip(self.maps[1:], self.paths[1:]):
            found = (map_.height, map_.width, map_.dtype)
            if found != expected:
                self.close()
                raise ValueError(
                    f"Cannot concatenate {path}: its frames are "
                    f"{found[0]}x{found[1]} of type {found[2]}, but "
                    f"{self.path} has {expected[0]}x{expected[1]} of type "
                    f"{expected[2]}."
                )

    def info(self):
        """Metadata of the first file, with the total frame count and a
        record of which files were concatenated.

        ``Concatenated Files`` and ``Frames per File`` end up in the
        saved localization metadata, so it stays reconstructable which
        folders went into the movie and which frame range came from
        which file - otherwise unrecoverable after the fact.

        Returns
        -------
        info : dict
            The first file's metadata, with ``Frames`` set to the total,
            plus ``Concatenated Files`` and ``Frames per File``.
        """
        info = self.maps[0].info()
        info["Frames"] = self.n_frames
        info["Concatenated Files"] = list(self.paths)
        info["Frames per File"] = list(self.n_frames_per_map)
        self.meta = info
        return info


def natural_key(text: str) -> list:
    """Sort key that orders embedded numbers numerically, so ``file_2``
    comes before ``file_10``.

    Parameters
    ----------
    text : str
        The string to build a key for.

    Returns
    -------
    key : list
        Alternating lower-cased text and integer parts.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def natural_path_key(path: str) -> tuple:
    """Sort key ordering paths by folder, then file name, with numbers
    compared numerically (see :func:`natural_key`).

    Parameters
    ----------
    path : str
        The path to build a key for.

    Returns
    -------
    key : tuple
        ``(key of the folder, key of the file name)``.
    """
    return (
        natural_key(os.path.dirname(path)),
        natural_key(os.path.basename(path)),
    )


# The continuation files of a multi-file OME set (``*_1.ome.tif``, ...).
# They are not movies in their own right: ``TiffMultiMap`` re-attaches
# them to their first file, so listing them as well repeats frames.
_OME_CONTINUATION_RE = re.compile(r"_\d+\.ome\.tif$", re.IGNORECASE)


def is_ome_continuation(path: str) -> bool:
    """Whether ``path`` is a continuation file of a multi-file OME set
    (``*_1.ome.tif``), i.e. part of another file's movie rather than a
    movie of its own.

    Parameters
    ----------
    path : str
        The path to test.

    Returns
    -------
    is_continuation : bool
    """
    return _OME_CONTINUATION_RE.search(os.path.basename(path)) is not None


def find_tif_movies(
    root: str,
    recursive: bool = True,
    key: Callable[[str], object] | None = None,
) -> list[str]:
    """Find the TIFF movies below ``root``, one path per movie.

    Intended to collect the files of an acquisition that was spread over
    several folders, so they can be concatenated (see
    :func:`load_tif_concatenated`).

    Files that are not movies in their own right are collapsed away, so
    each returned path opens exactly one whole movie:

    * The continuation files of a multi-file OME set (``*_1.ome.tif``,
      ``*_2.ome.tif``, ...) are dropped - ``TiffMultiMap`` re-attaches
      them to their first file, and keeping them would repeat frames.
    * A folder holding a MicroManager "separate image files" acquisition
      (thousands of one-frame ``img_*.tif``) contributes only the first
      frame's path, which opens the whole folder as one movie.

    Parameters
    ----------
    root : str
        Directory to search.
    recursive : bool, optional
        Search sub-folders as well. Default is True.
    key : Callable, optional
        Sort key applied to each path. Default is None, which sorts by
        folder and then file name with numbers compared numerically
        (``run_2`` before ``run_10``). Pass a custom key when the file
        names do not reflect acquisition order.

    Returns
    -------
    paths : list of str
        Absolute, sorted movie paths. Empty if nothing was found.
    """
    paths = []
    walker = os.walk(root) if recursive else [next(os.walk(root))]
    for directory, _, names in walker:
        separate_first = find_mm_separate_first(directory)
        if separate_first is not None:
            # The whole folder is one movie; its individual frame files
            # must not also be listed.
            paths.append(os.path.abspath(separate_first))
            continue
        for name in names:
            if not name.lower().endswith(TIFF_EXTENSIONS):
                continue
            if is_ome_continuation(name):
                continue
            paths.append(os.path.abspath(os.path.join(directory, name)))
    return sorted(paths, key=natural_path_key if key is None else key)


def load_tif_concatenated(
    paths: list[str],
    prompt_info: Callable[[dict], tuple[dict, bool]] | None = None,
    progress=None,
) -> tuple[ConcatenatedTiffMovie, list[dict]] | None:
    """Load several TIFF files as one movie, concatenated along frames.

    Mirrors :func:`load_tif`, but takes an ordered list of files (e.g.
    from :func:`find_tif_movies`) instead of a single path. The files are
    concatenated in the order given.

    Parameters
    ----------
    paths : list of str
        The TIFF movies to concatenate, in the order their frames should
        appear.
    prompt_info : Callable, optional
        Called with the readable movie dimensions if the first file's
        metadata cannot be parsed, so the user can enter it manually.
        Must return ``(info, save)`` or None if canceled.
    progress : callable, optional
        ``callable(done, total)`` invoked as the files are scanned, so a
        determinate progress bar can be shown. Default is None.

    Returns
    -------
    movie : ConcatenatedTiffMovie
        A movie object providing array-like access to all frames.
    info : list[dict]
        A list containing a dictionary with metadata about the movie.

    Returns None if the metadata could not be read and the user
    canceled the manual-metadata fallback dialog.
    """
    movie = ConcatenatedTiffMovie(paths, progress=progress)
    info = _movie_info_or_prompt(movie, movie.path, prompt_info)
    if info is None:
        return None
    return movie, [info]


def save_datasets(path: str, info: dict, **kwargs) -> None:
    """Save multiple datasets to an HDF5 file at the specified path.

    Parameters
    ----------
    path : str
        The file path where the datasets will be saved.
    info : dict
        Metadata information to be saved alongside the datasets.
    **kwargs
        Arbitrary keyword arguments where each key is the name of a
        dataset and each value is a pandas DataFrame containing the data
        to be saved.
    """
    # cannot use df.to_hdf for backward compatibility with older Picasso
    with h5py.File(path, "w") as locs_file:
        for key, val in kwargs.items():
            rec_locs = val.to_records(index=False)
            locs_file.create_dataset(key, data=rec_locs)
        embedded = _write_metadata_dataset(locs_file, info)
    if _save_metadata_in_yaml() or not embedded:
        base, ext = os.path.splitext(path)
        info_path = base + ".yaml"
        save_info(info_path, info)


def save_locs(path: str, locs: pd.DataFrame, info: list[dict]) -> None:
    """Save localization data to an HDF5 file.

    Parameters
    ----------
    path : str
        The path where the localization data will be saved.
    locs : pd.DataFrame
        The localization data to be saved.
    info : list of dict
        Metadata information to be saved alongside the localization
        data.
    """
    locs = lib.ensure_sanity(locs, info)
    # locs.to_hdf(path, key="locs", mode="w", format="fixed")
    # cannot use to_hdf for backward compatibility with older Picasso
    rec_locs = locs.to_records(index=False)
    with h5py.File(path, "w") as locs_file:
        locs_file.create_dataset("locs", data=rec_locs)
        embedded = _write_metadata_dataset(locs_file, info)
    if _save_metadata_in_yaml() or not embedded:
        base, ext = os.path.splitext(path)
        info_path = base + ".yaml"
        save_info(info_path, info)


def _raise_if_truncated(path: str, error: OSError) -> None:
    """Re-raise an HDF5 open error as a readable "file is incomplete".

    HDF5 refuses to open a file whose size on disk is smaller than the
    size recorded in its superblock and reports this as a ``truncated
    file: ... stored_eof = N`` message. Such a file is almost always
    still being downloaded, copied or written by another program, which
    the raw HDF5 message does not convey. Errors of any other kind, and
    files that are not in fact short, are left to the caller.

    Parameters
    ----------
    path : str
        The path to the HDF5 file that could not be opened.
    error : OSError
        The error raised by ``h5py`` when opening ``path``.

    Raises
    ------
    OSError
        If ``path`` is shorter than the size recorded in its superblock.
    """
    match = re.search(r"stored_eof\s*=\s*(\d+)", str(error))
    if match is None:
        return
    expected = int(match.group(1))
    actual = os.path.getsize(path)
    if actual >= expected:
        return
    raise OSError(
        f"File {path} is incomplete: {actual:,} bytes on disk, "
        f"{expected:,} bytes expected. It may still be downloading, "
        "copying, or being written by another program."
    ) from error


def _read_locs_dataset(path: str, progress=None) -> pd.DataFrame:
    """Read the ``locs`` dataset of an .hdf5 file in row blocks.

    Picasso writes the localizations as a single h5py compound dataset
    (see ``save_locs``), which is read here one block of rows at a time
    so that ``progress(done, total)`` can report - and, by raising,
    abort - a long read. Files whose ``locs`` node is not such a dataset
    (e.g. written by ``pandas.DataFrame.to_hdf``) fall back to
    ``pandas.read_hdf``, which reads them in one go.

    Parameters
    ----------
    path : str
        The path to the HDF5 file containing localization data.
    progress : callable or None, optional
        Called as ``progress(done, total)`` after each block of rows
        with the number of localizations read so far. Raising inside it
        aborts the read.

    Returns
    -------
    locs : pd.DataFrame
        The localization data read from the file.
    """
    BLOCK_SIZE = LOCS_BLOCK_SIZE
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    try:
        locs_file = h5py.File(path, "r")
    except OSError as error:
        _raise_if_truncated(path, error)
        raise
    with locs_file:
        if "locs" not in locs_file:
            raise KeyError(f"File: {path} does not contain a 'locs' dataset.")
        dataset = locs_file["locs"]

        # Only Picasso compound datasets can be read block-wise with h5py.
        if isinstance(dataset, h5py.Dataset):
            names = getattr(dataset.dtype, "names", None)

            if names is not None:
                n_locs = len(dataset)
                # fill one contiguous array per column, such that no copy of
                # the whole dataset is needed to build the DataFrame
                columns = {
                    name: np.empty(n_locs, dtype=dataset.dtype[name])
                    for name in names
                }
                for start in range(0, max(n_locs, 1), BLOCK_SIZE):
                    stop = min(start + BLOCK_SIZE, n_locs)
                    block = dataset[start:stop]
                    for name in names:
                        columns[name][start:stop] = block[name]
                    if progress is not None:
                        progress(stop, n_locs)
                return pd.DataFrame(columns, copy=False)

    # Non-compound HDF5 layouts are handled by PyTables/pandas.
    return pd.read_hdf(path, key="locs")


def load_locs(
    path: str,
    qt_parent: QtWidgets.QWidget | None = None,
    progress=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Load localization data from an HDF5 file.

    Parameters
    ----------
    path : str
        The path to the HDF5 file containing localization data.
    qt_parent : QWidget or None, optional
        Parent widget for any Qt-related operations, default is None.
    progress : callable or None, optional
        Called as ``progress(done, total)`` while the localizations are
        read, see ``_read_locs_dataset``. Raising inside it aborts the
        read.

    Returns
    -------
    locs : pd.DataFrame
        The localization data loaded from the file.
    info : list[dict]
        Metadata information loaded from the file, typically a list of
        dictionaries containing various metadata fields.

    Raises
    ------
    ValueError
        If the file path ends with ".csv", indicating that it is a
        ThunderSTORM .csv file, which should be loaded using
        picasso.io.import_ts instead.
    KeyError
        If the "locs" dataset is not found in the HDF5 file, indicating
        that the file does not contain the expected localization data.
    """
    if path.endswith(".csv"):
        raise ValueError(
            "If you wish to load a ThunderSTORM .csv file, use "
            "picasso.io.import_ts instead."
        )
    try:
        locs = _read_locs_dataset(path, progress=progress)
    except KeyError as e:  # if "locs" key not found
        print(
            f"\nAn error occured. File: {path} does not contain a "
            "'locs' dataset."
        )
        if qt_parent is not None:
            from PyQt6 import QtWidgets

            QtWidgets.QMessageBox.critical(
                qt_parent,
                "An error occured",
                f"File: {path} does not contain a 'locs' dataset.",
            )
        raise KeyError(e)
    info = load_info(path, qt_parent=qt_parent)
    locs = lib.ensure_sanity(locs, info)
    return locs, info


def save_identifications(
    path: str, identifications: pd.DataFrame, info: list[dict]
) -> None:
    """Save spot identifications to an HDF5 file.

    Parameters
    ----------
    path : str
        The path where the identifications will be saved.
    identifications : pd.DataFrame
        The identifications to be saved (typically with columns
        ``frame``, ``x``, ``y``, ``net_gradient``, ``n_id``).
    info : list of dict
        Metadata information to be saved alongside the identifications.
    """
    # cannot use df.to_hdf for backward compatibility with older Picasso
    rec_ids = identifications.to_records(index=False)
    with h5py.File(path, "w") as ids_file:
        ids_file.create_dataset("identifications", data=rec_ids)
        embedded = _write_metadata_dataset(ids_file, info)
    if _save_metadata_in_yaml() or not embedded:
        base, ext = os.path.splitext(path)
        info_path = base + ".yaml"
        save_info(info_path, info)


def load_identifications(
    path: str, qt_parent: QtWidgets.QWidget | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    """Load spot identifications from an HDF5 file.

    Parameters
    ----------
    path : str
        The path to the HDF5 file containing the identifications.
    qt_parent : QWidget or None, optional
        Parent widget for any Qt-related operations, default is None.

    Returns
    -------
    identifications : pd.DataFrame
        The identifications loaded from the file.
    info : list[dict]
        Metadata information loaded from the accompanying YAML file.

    Raises
    ------
    KeyError
        If the "identifications" dataset is not found in the HDF5 file.
    """
    try:
        identifications = pd.read_hdf(path, key="identifications")
    except KeyError as e:
        print(
            f"\nAn error occured. File: {path} does not contain an "
            "'identifications' dataset."
        )
        if qt_parent is not None:
            from PyQt6 import QtWidgets

            QtWidgets.QMessageBox.critical(
                qt_parent,
                "An error occured",
                f"File: {path} does not contain an 'identifications' "
                "dataset.",
            )
        raise KeyError(e)
    info = load_info(path, qt_parent=qt_parent)
    return identifications, info


def load_clusters(path: str) -> pd.DataFrame:
    """Load cluster data from an HDF5 file.

    Parameters
    ----------
    path : str
        The path to the HDF5 file containing cluster data.

    Returns
    -------
    clusters : pd.DataFrame
        The cluster data loaded from the file.
    """
    try:
        clusters = pd.read_hdf(path, key="clusters")
    except KeyError:
        clusters = pd.read_hdf(path, key="locs")
    return clusters


def load_filter(
    path: str,
    qt_parent: QtWidgets.QWidget | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Load localization data from an HDF5 file, checking for different
    possible keys for the localization data. This function is used to
    handle files that may contain localization data under different
    keys such as 'locs', 'groups', or 'clusters'.

    Parameters
    ----------
    path : str
        The path to the HDF5 file containing localization data.
    qt_parent : QWidget | None, optional
        Parent widget for any Qt-related operations, default is None.

    Returns
    -------
    locs : pd.DataFrame
        The localization data loaded from the file.
    info : list[dict]
        Metadata information loaded from the file, typically a list of
        dictionaries containing various metadata fields.
    """
    try:
        locs = pd.read_hdf(path, key="locs")
        info = load_info(path, qt_parent=qt_parent)
    except KeyError:
        try:
            locs = pd.read_hdf(path, key="groups")
            info = load_info(path, qt_parent=qt_parent)
        except KeyError:
            locs = pd.read_hdf(path, key="clusters")
            info = []
    return locs, info


def export_txt_imagej(
    path: str, locs: pd.DataFrame, info: list[dict] | None = None
) -> None:
    """Export localizations to a text file compatible with ImageJ.

    Parameters
    ----------
    path : str
        The path where the text file will be saved.
    locs : pd.DataFrame
        The localization data to be exported.
    info : list of dicts, optional
        Metadata dictionaries. Ignored but kept for compatibility with
        other export functions.
    """
    loctxt = locs[["frame", "x", "y"]]
    # Binary handle, as in the other .txt exporters: a text-mode handle would
    # translate the newline of the explicit "\r\n" as well, writing "\r\r\n".
    with open(path, "wb") as f:
        np.savetxt(
            f,
            loctxt.to_records(index=False),
            fmt=["%.1i", "%.5f", "%.5f"],
            newline="\r\n",
            delimiter="   ",
        )


def export_txt_nis(path: str, locs: pd.DataFrame, info: list[dict]) -> None:
    """Export localizations as .txt for NIS.

    Parameters
    ----------
    path : str
        The path where the text file will be saved.
    locs : pd.DataFrame
        The localization data to be exported.
    info : list of dicts
        Metadata dictionaries.
    """
    z_header = b"X\tY\tZ\tChannel\tWidth\tBG\tLength\tArea\tFrame\r\n"
    fmt_z = [
        "%.2f",
        "%.2f",
        "%.2f",
        "%.i",
        "%.2f",
        "%.i",
        "%.i",
        "%.i",
        "%.i",
    ]
    header = b"X\tY\tChannel\tWidth\tBG\tLength\tArea\tFrame\r\n"
    fmt = [
        "%.2f",
        "%.2f",
        "%.i",
        "%.2f",
        "%.i",
        "%.i",
        "%.i",
        "%.i",
    ]
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    columns_original = [
        "x",
        "y",
        "z",
        "sx",
        "bg",
        "photons",
        "frame",
    ]
    if "z" not in locs.columns:
        columns_original.remove("z")
    loctxt = locs[columns_original].copy()
    loctxt["frame"] += 1
    loctxt[["x", "y", "sx"]] *= pixelsize
    loctxt["Channel"] = 1
    loctxt["Length"] = 1
    loctxt["bg"] = loctxt["bg"].round().astype(int)
    loctxt["photons"] = loctxt["photons"].round().astype(int)
    if "z" in locs.columns:
        header = z_header
        fmt = fmt_z
    with open(path, "wb") as f:
        f.write(header)
        np.savetxt(
            f,
            loctxt.to_numpy(),
            fmt=fmt,
            newline="\r\n",
            delimiter="\t",
        )


def export_xyz_chimera(
    path: str, locs: pd.DataFrame, info: list[dict]
) -> None:
    """Export localizations as .xyz for CHIMERA. The file contains
    only x, y, z. Raise a warning if no z coordinate found.

    Parameters
    ----------
    path : str
        The path where the xyz file will be saved.
    locs : pd.DataFrame
        The localization data to be exported.
    info : list of dicts
        Metadata dictionaries.
    """
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    if "z" in locs.columns:
        loctxt = locs[["x", "y", "z"]].copy()
        loctxt["molecule"] = 1
        loctxt[["x", "y"]] *= pixelsize
        loctxt = loctxt[["molecule", "x", "y", "z"]]
        with open(path, "wb") as f:
            f.write(b"Molecule export\r\n")
            np.savetxt(
                f,
                loctxt.to_numpy(),
                fmt=["%i", "%.5f", "%.5f", "%.5f"],
                newline="\r\n",
                delimiter="\t",
            )
    else:
        warnings.warn(
            "No z coordinate found in localizations; cannot export"
            " to .xyz for CHIMERA."
        )


def export_3d_visp(path: str, locs: pd.DataFrame, info: list[dict]) -> None:
    """Export localizations as .3d for ViSP. Show a warning if no z
    coordinate found.

    Parameters
    ----------
    path : str
        The path where the 3d file will be saved.
    locs : pd.DataFrame
        The localization data to be exported.
    info : list of dicts
        Metadata dictionaries.
    """
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    if "z" in locs.columns:
        loctxt = locs[["x", "y", "z", "photons", "frame"]].copy()
        loctxt[["x", "y"]] *= pixelsize
        loctxt["frame"] = loctxt["frame"].astype(int)
        with open(path, "wb") as f:
            np.savetxt(
                f,
                loctxt.to_records(index=False),
                fmt=["%.1f", "%.1f", "%.1f", "%.1f", "%d"],
                newline="\r\n",
            )
    else:
        warnings.warn(
            "No z coordinate found in localizations; cannot export "
            "to .3d for ViSP."
        )


def export_thunderstorm(
    path: str, locs: pd.DataFrame, info: list[dict]
) -> None:
    """Export localizations as .csv for ThunderSTORM.

    Parameters
    ----------
    path : str
        The path where the csv file will be saved.
    locs : pd.DataFrame
        The localization data to be exported.
    info : list of dicts
        Metadata dictionaries.
    """
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    columns_original = [
        "frame",
        "x",
        "y",
        "sx",
        "sy",
        "photons",
        "bg",
        "lpx",
        "lpy",
    ]
    if "z" in locs.columns:
        columns_original.append("z")
    if "len" in locs.columns:
        columns_original.append("len")
    loctxt = locs[columns_original].copy()

    # add the columns
    loctxt["photons"] = loctxt["photons"].astype(np.int32)
    loctxt["bg"] = loctxt["bg"].astype(np.int32)
    loctxt["id"] = np.arange(len(loctxt), dtype=np.int32)
    loctxt[["x", "y", "sx", "sy"]] *= pixelsize
    loctxt["bkgstd [photon]"] = 0
    loctxt["uncertainty_xy [nm]"] = (
        (loctxt["lpx"] + loctxt["lpy"]) / 2 * pixelsize
    )
    column_mapper = {
        "x": "x [nm]",
        "y": "y [nm]",
        "sx": "sigma1 [nm]",
        "sy": "sigma2 [nm]",
        "photons": "intensity [photon]",
        "bg": "offset [photon]",
    }
    if "z" in loctxt.columns:
        column_mapper["z"] = "z [nm]"
    if "len" in loctxt.columns:
        loctxt.rename(columns={"len": "detections"}, inplace=True)
    loctxt.rename(columns=column_mapper, inplace=True)
    loctxt.drop(columns=["lpx", "lpy"], inplace=True)
    # change the order of columns
    columns_final = [
        "id",
        "frame",
        "x [nm]",
        "y [nm]",
        "z [nm]",
        "sigma1 [nm]",
        "sigma2 [nm]",
        "intensity [photon]",
        "offset [photon]",
        "bkgstd [photon]",
        "uncertainty_xy [nm]",
        "detections",
    ]
    if "z [nm]" not in loctxt.columns:
        columns_final.remove("z [nm]")
        columns_final.remove("sigma2 [nm]")
        columns_final[4] = "sigma [nm]"
        loctxt.rename(
            columns={"sigma1 [nm]": "sigma [nm]"},
            inplace=True,
        )
        loctxt.drop(columns=["sigma2 [nm]"], inplace=True)
    if "detections" not in loctxt.columns:
        columns_final.remove("detections")
    loctxt = loctxt[columns_final]
    # save
    loctxt.to_csv(path, index=False)


def _sanitize_column_name(name: str) -> str:
    """Convert an arbitrary column name to a valid identifier (e.g.,
    ``"uncertainty_z [nm]"`` -> ``"uncertainty_z_nm"``) so it survives
    saving as an HDF5 structured array and attribute-style access."""
    name = re.sub(r"\W+", "_", name.strip()).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = "_" + name
    return name


def import_ts(path: str, pixelsize: float) -> tuple[pd.DataFrame, list[dict]]:
    """Import localization data from a ThunderSTORM .csv file.

    Columns that do not map onto predefined Picasso fields are kept
    (numeric columns only), with their names sanitized to valid
    identifiers.

    Parameters
    ----------
    path : str
        The path to the ThunderSTORM .csv file.
    pixelsize : float
        Camera pixel size in nm. Picasso saves xy coordinates in units
        of camera pixels.

    Returns
    -------
    locs : pd.DataFrame
        The localization data imported from the file.
    info : list of dicts
        Minimal metadata information.
    """
    expected_columns = [
        "frame",
        "x [nm]",
        "y [nm]",
        "intensity [photon]",
        "offset [photon]",
        "uncertainty_xy [nm]",
        "sigma [nm]",
    ]
    expected_columns_z = [
        "frame",
        "x [nm]",
        "y [nm]",
        "z [nm]",
        "intensity [photon]",
        "offset [photon]",
        "uncertainty_xy [nm]",
        "sigma1 [nm]",
        "sigma2 [nm]",
    ]
    data = pd.read_csv(path)
    if "z [nm]" in data.columns:
        if not all([col in data.columns for col in expected_columns_z]):
            raise ValueError(
                "Expected columns for 3D ThunderSTORM .csv: "
                f"{expected_columns_z}. Found: {list(data.columns)}."
            )
    else:
        if not all([col in data.columns for col in expected_columns]):
            raise ValueError(
                "Expected columns for 2D ThunderSTORM .csv: "
                f"{expected_columns}. Found: {list(data.columns)}."
            )
    frames = data["frame"].astype(int)
    # make sure frames start at zero:
    frames = frames - np.min(frames)
    x = data["x [nm]"] / pixelsize
    y = data["y [nm]"] / pixelsize
    photons = data["intensity [photon]"].astype(int)

    bg = data["offset [photon]"].astype(int)
    lpx = data["uncertainty_xy [nm]"] / pixelsize
    lpy = data["uncertainty_xy [nm]"] / pixelsize

    if "z [nm]" in data.columns:
        z = data["z [nm]"]
        sx = data["sigma1 [nm]"] / pixelsize
        sy = data["sigma2 [nm]"] / pixelsize
        locs = pd.DataFrame(
            {
                "frame": frames.astype(np.uint32),
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "z": z.astype(np.float32),
                "photons": photons.astype(np.float32),
                "sx": sx.astype(np.float32),
                "sy": sy.astype(np.float32),
                "bg": bg.astype(np.float32),
                "lpx": lpx.astype(np.float32),
                "lpy": lpy.astype(np.float32),
            }
        )
    else:
        sx = data["sigma [nm]"] / pixelsize
        sy = data["sigma [nm]"] / pixelsize
        locs = pd.DataFrame(
            {
                "frame": frames.astype(np.uint32),
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "photons": photons.astype(np.float32),
                "sx": sx.astype(np.float32),
                "sy": sy.astype(np.float32),
                "bg": bg.astype(np.float32),
                "lpx": lpx.astype(np.float32),
                "lpy": lpy.astype(np.float32),
            }
        )

    # Keep any additional (non-predefined) numeric columns from the
    # input file.
    used_columns = set(expected_columns) | set(expected_columns_z)
    for name in data.columns:
        if name in used_columns:
            continue
        values = data[name]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        new_name = _sanitize_column_name(name)
        if new_name in locs.columns:
            continue
        if pd.api.types.is_float_dtype(values):
            values = values.astype(np.float32)
        locs[new_name] = values

    locs.sort_values(kind="quicksort", by="frame", inplace=True)

    img_info = {}
    img_info["Generated by"] = f"Picasso v{__version__} csv2hdf"
    img_info["Frames"] = int(np.max(frames)) + 1
    img_info["Height"] = int(np.ceil(np.max(y)))
    img_info["Width"] = int(np.ceil(np.max(x)))
    img_info["Pixelsize"] = float(pixelsize)

    base, ext = os.path.splitext(path)
    out_path = base + "_locs.hdf5"
    save_locs(out_path, locs, [img_info])
    return locs, [img_info]


# SMAP (https://github.com/jries/SMAP) stores localizations in a
# proprietary ``_sml.mat`` MATLAB file. The localization data lives in
# the ``saveloc.loc`` struct of column arrays (fields such as ``xnm``,
# ``ynm``, ``znm``, ``frame``, ``phot``, ``bg``, ``locprecnm``,
# ``locprecznm``, ``PSFxnm``). Coordinates and precisions are in nm and
# frames are 1-based. The unit mapping below mirrors SMAP's own
# Picasso exporter (``export_picasso_hdf5.m`` / ``savepicasso.m``).
_SMAP_SPLIT_MSG = (
    "This appears to be a SMAP split '_sml_p*.mat' file. Please re-save "
    "it in SMAP as a single file (not split into parts) and try again."
)


def _read_smap_loc_hdf5(path: str) -> dict:
    """Read the ``saveloc.loc`` struct from a MATLAB v7.3 (HDF5) SMAP
    file using h5py (scipy cannot read v7.3 files)."""
    with h5py.File(path, "r") as f:
        if any(key in f for key in ("lds", "partnames", "S")):
            raise ValueError(_SMAP_SPLIT_MSG)
        if "saveloc" in f and "loc" in f["saveloc"]:
            loc_group = f["saveloc"]["loc"]
        elif "loc" in f:
            loc_group = f["loc"]
        else:
            raise ValueError(
                f"Could not find SMAP localizations ('saveloc.loc') in "
                f"{path}. Is this a SMAP _sml.mat file?"
            )
        loc = {}
        for field in loc_group.keys():
            dataset = loc_group[field]
            if not isinstance(dataset, h5py.Dataset):
                continue
            value = np.atleast_1d(np.asarray(dataset).ravel())
            # Skip non-numeric fields (e.g. cell arrays stored as
            # HDF5 object references).
            if value.dtype.kind not in "fiu":
                continue
            loc[field] = value
    return loc


def _read_smap_loc(path: str) -> dict:
    """Read the localization struct (``saveloc.loc``) from a SMAP
    ``_sml.mat`` file as a dictionary mapping field names to 1-D numpy
    arrays.

    Supports both MATLAB v7/v5 (read with scipy) and v7.3/HDF5 (read
    with h5py) single-file saves. SMAP's split ``_sml_p*.mat`` parts
    format is detected and rejected with a clear message.

    Parameters
    ----------
    path : str
        Path to the SMAP ``_sml.mat`` file.

    Returns
    -------
    loc : dict
        Mapping of SMAP field name (e.g. ``"xnm"``) to a 1-D array.
    """
    from scipy.io import loadmat

    # MATLAB v7.3 files are HDF5-based (with an HDF5 user block) and must
    # be read with h5py; v7/v5 files are read with scipy.
    if h5py.is_hdf5(path):
        return _read_smap_loc_hdf5(path)

    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    if any(key in mat for key in ("lds", "partnames", "S")):
        raise ValueError(_SMAP_SPLIT_MSG)
    if "saveloc" in mat:
        loc_struct = getattr(mat["saveloc"], "loc", None)
    elif "loc" in mat:
        loc_struct = mat["loc"]
    else:
        loc_struct = None
    if loc_struct is None or not hasattr(loc_struct, "_fieldnames"):
        raise ValueError(
            f"Could not find SMAP localizations ('saveloc.loc') in "
            f"{path}. Is this a SMAP _sml.mat file?"
        )
    loc = {}
    for field in loc_struct._fieldnames:
        value = np.atleast_1d(np.asarray(getattr(loc_struct, field)).ravel())
        if value.dtype.kind not in "fiu":
            continue
        loc[field] = value
    return loc


def import_smap(
    path: str, pixelsize: float
) -> tuple[pd.DataFrame, list[dict]]:
    """Import localization data from a SMAP ``_sml.mat`` file.

    Fields that do not map onto predefined Picasso columns (e.g.,
    ``channel``, ``numberInGroup``) are kept as extra columns, with
    their names sanitized to valid identifiers.

    Parameters
    ----------
    path : str
        The path to the SMAP ``_sml.mat`` file.
    pixelsize : float
        Camera pixel size in nm. Picasso saves xy coordinates in units
        of camera pixels, while SMAP stores them in nm.

    Returns
    -------
    locs : pd.DataFrame
        The localization data imported from the file.
    info : list of dicts
        Minimal metadata information.
    """
    loc = _read_smap_loc(path)
    missing = [c for c in ("xnm", "ynm", "frame") if c not in loc]
    if missing:
        raise ValueError(
            f"SMAP file {path} is missing required fields: {missing}. "
            f"Found fields: {list(loc.keys())}."
        )
    n = len(loc["xnm"])

    # SMAP frames are 1-based; make them start at zero (robust to any
    # offset, mirroring import_ts).
    frames = loc["frame"].astype(np.int64)
    frames = frames - np.min(frames)
    x = loc["xnm"] / pixelsize
    y = loc["ynm"] / pixelsize

    data = {
        "frame": frames.astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
    }
    # z is stored in nm in both SMAP and Picasso.
    if "znm" in loc:
        data["z"] = loc["znm"].astype(np.float32)

    photons = loc["phot"] if "phot" in loc else np.ones(n)
    data["photons"] = np.broadcast_to(photons, (n,)).astype(np.float32)

    # PSF widths (nm -> px); use a neutral 1 px default when absent.
    if "PSFxnm" in loc:
        sx = loc["PSFxnm"] / pixelsize
    else:
        sx = np.ones(n)
    if "PSFynm" in loc:
        sy = loc["PSFynm"] / pixelsize
    elif "PSFxnm" in loc:
        sy = sx
    else:
        sy = np.ones(n)
    data["sx"] = np.asarray(sx, dtype=np.float32)
    data["sy"] = np.asarray(sy, dtype=np.float32)

    bg = loc["bg"] if "bg" in loc else np.zeros(n)
    data["bg"] = np.broadcast_to(bg, (n,)).astype(np.float32)

    # Localization precision (nm -> px). SMAP stores a single combined
    # value (locprecnm); some files use separate locprecxnm/locprecynm.
    if "locprecnm" in loc:
        lpx = loc["locprecnm"] / pixelsize
        lpy = loc["locprecnm"] / pixelsize
    elif "locprecxnm" in loc:
        lpx = loc["locprecxnm"] / pixelsize
        lpy = loc.get("locprecynm", loc["locprecxnm"]) / pixelsize
    else:
        lpx = np.zeros(n)
        lpy = np.zeros(n)
    data["lpx"] = np.asarray(lpx, dtype=np.float32)
    data["lpy"] = np.asarray(lpy, dtype=np.float32)

    if "znm" in loc and "locprecznm" in loc:
        data["lpz"] = loc["locprecznm"].astype(np.float32)

    # Keep any additional (non-predefined) fields from the SMAP file.
    used_fields = {
        "frame",
        "xnm",
        "ynm",
        "znm",
        "phot",
        "PSFxnm",
        "PSFynm",
        "bg",
        "locprecnm",
        "locprecxnm",
        "locprecynm",
        "locprecznm",
    }
    for field, values in loc.items():
        if field in used_fields or len(values) != n:
            continue
        name = _sanitize_column_name(field)
        if name in data:
            continue
        if values.dtype.kind == "f":
            values = values.astype(np.float32)
        data[name] = values

    locs = pd.DataFrame(data)
    locs.sort_values(kind="quicksort", by="frame", inplace=True)

    img_info = {}
    img_info["Generated by"] = f"Picasso v{__version__} smap2hdf"
    img_info["Frames"] = int(np.max(frames)) + 1
    img_info["Height"] = int(np.ceil(np.max(y)))
    img_info["Width"] = int(np.ceil(np.max(x)))
    img_info["Pixelsize"] = float(pixelsize)

    return locs, [img_info]


def export_smap(path: str, locs: pd.DataFrame, info: list[dict]) -> None:
    """Export localizations as a SMAP ``_sml.mat`` file.

    The file is written as a MATLAB-5 (``-v7``-compatible) file that
    SMAP's loader reads natively: it contains the top-level ``saveloc``
    struct (with ``loc`` and ``info`` sub-structs) and a ``fileformat``
    struct. SMAP recognizes the file as localizations via the ``_sml``
    suffix, which is enforced here.

    Parameters
    ----------
    path : str
        The path where the .mat file will be saved. The ``_sml.mat``
        suffix is enforced.
    locs : pd.DataFrame
        The localization data to be exported.
    info : list of dicts
        Metadata dictionaries.
    """
    from scipy.io import savemat

    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)

    # SMAP recognizes a localization file by the '_sml' suffix.
    base, ext = os.path.splitext(path)
    if not base.endswith("_sml"):
        base = base + "_sml"
    path = base + ".mat"

    n = len(locs)

    def column(name: str, default: float = 0.0) -> np.ndarray:
        """Return a (N, 1) float64 column for a Picasso field, or a
        default-filled column if the field is absent."""
        if name in locs.columns:
            values = locs[name].to_numpy(dtype=np.float64)
        else:
            values = np.full(n, default, dtype=np.float64)
        return values.reshape(-1, 1)

    loc = {
        # SMAP frames are 1-based.
        "frame": column("frame") + 1,
        "xnm": column("x") * pixelsize,
        "ynm": column("y") * pixelsize,
        "phot": column("photons"),
        "bg": column("bg"),
        "locprecnm": (column("lpx") + column("lpy")) / 2 * pixelsize,
        "PSFxnm": column("sx") * pixelsize,
        "PSFynm": column("sy") * pixelsize,
        "channel": np.zeros((n, 1)),
    }
    if "z" in locs.columns:
        loc["znm"] = column("z")  # already in nm
        if "lpz" in locs.columns:
            loc["locprecznm"] = column("lpz")  # already in nm

    width = lib.get_from_metadata(info, "Width")
    height = lib.get_from_metadata(info, "Height")
    if width is None:
        width = int(np.ceil(locs["x"].max()))
    if height is None:
        height = int(np.ceil(locs["y"].max()))
    px_um = float(pixelsize) / 1000.0
    smap_info = {
        "Width": float(width),
        "Height": float(height),
        "roi": np.array([[0.0, 0.0, float(width), float(height)]]),
        "cam_pixelsize_um": np.array([[px_um, px_um]]),
    }

    out = {
        "saveloc": {"loc": loc, "info": smap_info},
        "fileformat": {"name": "sml"},
    }
    savemat(path, out, do_compression=True)
