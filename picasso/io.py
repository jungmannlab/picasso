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
import glob
import logging
import re
import json
import os
import threading
import warnings
from typing import Callable, Literal

import tifffile
import yaml
import h5py
import nd2
import numpy as np
import pandas as pd
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


class NoMetadataFileError(FileNotFoundError):
    pass


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
    """
    home = os.path.expanduser("~")
    directory = os.path.join(home, ".picasso", "plugins")
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
        print(f"Setting channel to {channel}")
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
    """Save the camera configuration dictionary to a YAML file. See
    https://picassosr.readthedocs.io/en/latest/localize.html#camera-config.

    Parameters
    ----------
    CONFIG : dict
        The camera configuration dictionary to save.
    """
    this_file = os.path.abspath(__file__)
    this_directory = os.path.dirname(this_file)
    with open(os.path.join(this_directory, "config.yaml"), "w") as config_file:
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
        calibration = yaml.full_load(calibration_file)
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
        The metadata dictionary, or None if the user cancelled the
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
    cancelled the fallback dialog, in which case the caller should abort
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
        Must return ``(info, save)`` or None if cancelled.
    progress : None, optional
        A placeholder for progress tracking, not used in this function.
        Default is None.

    Returns
    -------
    movie : TiffMultiMap
        A movie object providing array-like access to TIFF frames.
        Frames are loaded into memory on access.
    info : list[dict]
        A list containing a dictionary with metadata about the movie.

    Returns None if the metadata could not be read and the user
    cancelled the manual-metadata fallback dialog.
    """
    movie = TiffMultiMap(path, memmap_frames=False)
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
        Must return ``(info, save)`` or None if cancelled.

    Returns
    -------
    movie : ND2Movie
        The loaded ND2 movie.
    info : list of dicts
        A list containing a dictionary with metadata about the movie.

    Returns None if the metadata could not be read and the user
    cancelled the manual-metadata fallback dialog.
    """
    movie = ND2Movie(path)
    info = _movie_info_or_prompt(movie, path, prompt_info)
    if info is None:
        return None
    return movie, [info]


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
        Must return ``(info, save)`` or None if cancelled.

    Returns
    -------
    movie : STKMultiMovie
        A movie object providing array-like access to STK frames.
        Frames are loaded into memory on access.
    info : list[dict]
        A list containing a dictionary with metadata about the movie.

    Returns None if the metadata could not be read and the user
    cancelled the manual-metadata fallback dialog.
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
    progress : None
        Placeholder for progress tracking, not used in this function.

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
        return load_tif(path, prompt_info=prompt_info)
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
    # Neither the sidecar file nor embedded metadata was found.
    print(f"\nAn error occured. Could not find metadata file:\n{filename}")
    if qt_parent is not None:
        QtWidgets.QMessageBox.critical(
            qt_parent,
            "An error occured",
            f"Could not find metadata file:\n{filename}",
        )
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
) -> tuple[list, Literal["Circle", "Rectangle", "Polygon", "Square"], float]:
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
    shape : Literal["Circle", "Rectangle", "Polygon", "Square"]
        The shape of the picks.
    size : float
        The size of the picks in camera pixels (if `pixelsize` is
        provided, otherwise in original units). For circular picks, the
        size is the diameter; for rectangular picks, the size is the
        width; for square picks, the size is the side length. None for
        polygonal picks (size not defined).
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
        if "Diameter (nm)" in regions:
            size = regions["Diameter (nm)"] / pixelsize
        elif "Diameter" in regions:
            size = regions["Diameter"]
    elif shape == "Rectangle":
        picks = regions["Center-Axis-Points"]
        if "Width (nm)" in regions:
            size = regions["Width (nm)"] / pixelsize
        elif "Width" in regions:
            size = regions["Width"]
    elif shape == "Polygon":
        picks = regions["Vertices"]
        size = None
    elif shape == "Square":
        picks = regions["Centers"]
        # no backward compatibility here, always in nm
        size = regions["Side Length (nm)"] / pixelsize
    else:
        raise ValueError("Unrecognized pick shape")
    return picks, shape, size


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
    np.savetxt(path, drift, newline="\r\n")


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
    """Save user settings, for example, the default directory for
    loading/saving files to a YAML file."""
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


def _write_metadata_dataset(hdf_file: h5py.File, info: list[dict]) -> None:
    """Embed metadata in an open HDF5 file as a JSON string dataset at
    ``/metadata``.

    Parameters
    ----------
    hdf_file : h5py.File
        An HDF5 file opened in write mode.
    info : list of dict
        Metadata to embed.
    """
    hdf_file.create_dataset("metadata", data=json.dumps(list(info)))


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
        with h5py.File(path, "r") as hdf_file:
            if "metadata" not in hdf_file:
                return None
            raw = hdf_file["metadata"][()]
    except (OSError, KeyError):
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return list(json.loads(raw))
    except (ValueError, TypeError):
        return None


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
        pass

    @abc.abstractmethod
    def get_frame(self, index: int) -> lib.IntArray2D:
        pass

    @abc.abstractmethod
    def tofile(self, file_handle, byte_order=None):
        pass

    @property
    @abc.abstractmethod
    def dtype(self):
        return "u16"


class ND2Movie(AbstractPicassoMovie):
    """Subclass of the AbstractPicassoMovie to implement reading Nikon
    nd2 files.

    This class implements a version which uses only ``nd2``."""

    def __init__(self, path: str, verbose: bool = False):
        super().__init__()
        if verbose:
            print("Reading info from {}".format(path))
        self.path = os.path.abspath(path)
        self.nd2file = nd2.ND2File(path)
        self.dask = self.nd2file.to_dask()
        self.sizes = self.nd2file.sizes

        required_dims = ["T", "Y", "X"]  # exactly these, not more
        for dim in required_dims:
            if dim not in self.nd2file.sizes.keys():
                raise KeyError(
                    "Required dimension {:s} not in file {:s}".format(
                        dim, self.path
                    )
                )
        if self.nd2file.ndim != len(required_dims):
            raise KeyError(
                "File {:s} has dimensions {:s} ".format(
                    self.path, str(self.nd2file.sizes.keys())
                )
                + "but should have exactly {:s}.".format(str(required_dims))
            )

        # Pixel access only needs the dimensions checked above; parsing
        # the (often vendor-specific) metadata may still fail. Keep that
        # failure recoverable so the movie can be loaded with manually
        # entered metadata (info() then returns None).
        try:
            self.meta = self.get_metadata(self.nd2file)
        except Exception:
            self.meta = None
        self._shape = [
            self.nd2file.sizes["T"],
            self.nd2file.sizes["X"],
            self.nd2file.sizes["Y"],
        ]

    def info(self) -> dict:
        return self.meta

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
            "Frames": nd2file.sizes["T"],
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
        for i in range(self.sizes["T"]):
            yield self[i]

    def __len__(self):
        return self.sizes["T"]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def close(self):
        self.nd2file.close()

    def get_frame(self, index: int) -> lib.IntArray2D:
        """Load one frame of the movie.

        Parameters
        ----------
        index : int
            The frame index to retrieve.

        Returns
        -------
        frame : lib.IntArray2D
            2D array representing the image data of the frame
        """
        return self.dask[index].compute()

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
    provided here. Mirrors the channel-prompt behaviour of ``load_ims``.
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
    ) -> None:
        """Store the available channels and pick one.

        Defaults to the first channel when there is only one or when no
        prompt is supplied (e.g. command-line batch processing), matching
        ``load_ims``.
        """
        self.channels = list(channels) if channels else ["Channel 0"]
        if len(self.channels) > 1 and prompt_info is not None:
            choice = prompt_info(self.channels)
            if choice in self.channels:
                self._channel = self.channels.index(choice)
            else:
                self._channel = 0
        else:
            self._channel = 0
        print(f"Setting channel to {self.channels[self._channel]}")

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
        self._select_channel(channels, prompt_info)

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
        self._select_channel(channels, prompt_info)

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
    """Coerce arbitrary metadata into YAML/JSON-serialisable builtins so it
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
            if isinstance(summary, str):
                out["Micro-Manager Acquisition Comments"] = summary.split("\n")
    except Exception:
        pass

    return out


class TiffMap:
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

    def __init__(self, path: str, verbose: bool = False):
        """Open the TIFF file with tifffile and extract the geometry,
        data type and per-page layout needed for lazy frame access."""
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
            self._offsets, self.n_frames = self._build_offsets()
        except RuntimeError:
            # Flipping useframes in place is cache-safe (Picasso never
            # enables tifffile's page cache, so no stale TiffFrames are
            # held) and keeps the already-discovered IFD offset list. The
            # LSM branch above uses full TiffPages and never raises here.
            self._tif.pages.useframes = False
            self._pages = self._tif.pages
            self._offsets, self.n_frames = self._build_offsets()

        # A persistent binary handle for the fast offset-based read path.
        self.file = open(self.path, "rb")
        self.lock = threading.Lock()

    def _build_offsets(self) -> tuple[list[int] | None, int]:
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
        n_pages = len(self._pages)

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
                return None, n_pages
            n_frames = 0
            for page in self._pages:
                if tuple(page.shape) != self._page_shape:
                    break
                n_frames += 1
            return None, n_frames

        # Uncompressed: one pass collects each frame's byte offset and
        # stops at the first IFD whose shape differs from page 0.
        offsets = []
        n_frames = 0
        fast = True
        for page in self._pages:
            if tuple(page.shape) != self._page_shape:
                break
            n_frames += 1
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
        with self.lock:  # for reading frames from multiple threads
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
                return np.array(
                    [self.get_frame(_) for _ in range(self.n_frames)]
                )
            elif isinstance(it, int) or np.issubdtype(it, np.integer):
                return self.get_frame(it)
            raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def info(self) -> dict:
        """Extract metadata from the TIFF file and return it as a
        Picasso info dictionary: byte order, file path, height, width,
        data type, number of frames, and - for MicroManager files - the
        MicroManager metadata, camera name and acquisition comments."""
        info = {
            "Byte Order": self._tif_byte_order,
            "File": self.path,
            "Height": self.height,
            "Width": self.width,
            "Data Type": self.dtype.name,
            "Frames": self.n_frames,
        }
        mm = _mm_metadata_from_tifffile(self._tif)
        # The comments key is always present (possibly empty)
        info["Micro-Manager Acquisition Comments"] = mm.get(
            "Micro-Manager Acquisition Comments", ""
        )
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
        parse); all other layouts fall back to ``tifffile``'s decoder."""
        if self._offsets is not None:
            # Fast path: pure seek + read, no tifffile access per frame.
            self.file.seek(self._offsets[index])
            frame = np.fromfile(
                self.file,
                dtype=self._tif_dtype,
                count=self.frame_size,
            ).reshape(self.frame_shape)
        else:
            # Compressed / tiled / multi-strip pages: let tifffile decode
            # this single page (still lazy - other frames stay on disk).
            frame = np.asarray(self._pages[index].asarray())
        # Downstream code expects little-endian unsigned integers; astype
        # is a no-op (no copy) when the data is already in that order.
        return frame.astype(self.dtype, copy=False)

    def close(self) -> None:
        self.file.close()
        self._tif.close()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = byte_order != self._tif_byte_order
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)


class STKMovie(AbstractPicassoMovie):
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

        # Open a persistent binary file handle for lazy frame reading.
        self._file = open(self.path, "rb")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # AbstractPicassoMovie interface
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):  # noqa: C901
        with self._lock:
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
                return np.array(
                    [self.get_frame(_) for _ in range(self.n_frames)]
                )
            elif isinstance(it, int) or np.issubdtype(it, np.integer):
                return self.get_frame(it)
            raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self) -> int:
        return self.n_frames

    def info(self) -> dict:
        """Return Picasso-compatible metadata dictionary."""
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
        """Load one frame from the STK file by binary offset."""
        if index < 0:
            index = self.n_frames + index
        if not (0 <= index < self.n_frames):
            raise IndexError(
                f"Frame index {index} out of range for movie with "
                f"{self.n_frames} frames."
            )
        offset = self._first_data_offset + index * self._frame_bytes
        self._file.seek(offset)
        frame = np.fromfile(
            self._file,
            dtype=self._tif_dtype,
            count=self.height * self.width,
        ).reshape(self.frame_shape)
        if self._byte_order == ">":
            frame = frame.byteswap().view(self._dtype)
        return frame

    def close(self) -> None:
        self._file.close()

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


class TiffMultiMap(AbstractPicassoMovie):
    """Read ``.ome.tif`` files created by MicroManager. Single files are
    maxed out at 4GB, so this class orchestrates reading from single
    files, each accessed by ``TiffMap``."""

    def __init__(
        self,
        path: str,
        memmap_frames: bool = False,
        verbose: bool = False,
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
        self.maps = [TiffMap(path, verbose=verbose) for path in self.paths]
        self.n_maps = len(self.maps)
        self.n_frames_per_map = [_.n_frames for _ in self.maps]
        self.n_frames = sum(self.n_frames_per_map)
        self.cum_n_frames = np.insert(np.cumsum(self.n_frames_per_map), 0, 0)
        self._dtype = self.maps[0].dtype
        self.height = self.maps[0].height
        self.width = self.maps[0].width
        self.shape = (self.n_frames, self.height, self.width)

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

        Args:
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
        if "Gain Property Name" in cam_config:
            gain_property_name = cam_config["Gain Property Name"]
            gain = mm_info[camera + "-" + gain_property_name]
            if "EM Switch Property" in cam_config:
                switch_property_name = cam_config["EM Switch Property"]["Name"]
                switch_property_value = mm_info[
                    camera + "-" + switch_property_name
                ]
                if (
                    switch_property_value
                    == cam_config["EM Switch Property"][True]
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
                channel_device_name = cam_config["Channel Device"]["Name"]
                channel = mm_info[channel_device_name]
                channels = cam_config["Channel Device"]["Emission Wavelengths"]
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


def to_raw_combined(basename: str, paths: list[str]) -> None:
    """Combine multiple TIFF files into a single raw file in the OME
    format.

    Parameters
    ----------
    basename : str
        The base name for the output raw file.
    paths : list of strs
        List of paths to the TIFF files to be combined.
    """
    raw_file_name = basename + ".ome.raw"
    with open(raw_file_name, "wb") as file_handle:
        with TiffMap(paths[0]) as tif:
            tif.tofile(file_handle, "<")
            info = tif.info()
        for path in paths[1:]:
            with TiffMap(path) as tif:
                info_ = tif.info()
                info["Frames"] += info_["Frames"]
                if "Comments" in info_:
                    info["Comments"] = info_["Comments"]
                tif.tofile(file_handle, "<")
        info["Generated by"] = f"Picasso ToRaw v{__version__}"
        info["Byte Order"] = "<"
        info["Original File"] = os.path.basename(info.pop("File"))
        info["Raw File"] = os.path.basename(raw_file_name)
        save_info(basename + ".ome.yaml", [info])


def get_movie_groups(paths: list[str]) -> dict[str, list[str]]:
    """Group movie files by their base name, allowing for an optional
    appendix of the file number. This is useful for handling OME-TIFF
    files that may have multiple parts or versions.

    Parameters
    ----------
    paths : list[str]
        List of file paths to be grouped.

    Returns
    -------
    groups : dict[str, list[str]]
        A dictionary where keys are base names and values are lists of
        file paths that share the same base name.
    """
    groups = {}
    if len(paths) > 0:
        # This matches the basename + an opt appendix of the file number
        pattern = re.compile(r"(.*?)(_(\d*))?.ome.tif")
        matches = [re.match(pattern, path) for path in paths]
        match_infos = [
            {"path": _.group(), "base": _.group(1), "index": _.group(3)}
            for _ in matches
        ]
        for match_info in match_infos:
            if match_info["index"] is None:
                match_info["index"] = 0
            else:
                match_info["index"] = int(match_info["index"])
        basenames = set([_["base"] for _ in match_infos])
        for basename in basenames:
            match_infos_group = [
                _ for _ in match_infos if _["base"] == basename
            ]
            group = [_["path"] for _ in match_infos_group]
            indices = [_["index"] for _ in match_infos_group]
            group = [path for (index, path) in sorted(zip(indices, group))]
            groups[basename] = group
    return groups


def to_raw(path: str, verbose: bool = True) -> None:
    """Convert TIFF files matching the given path pattern into a single
    raw file in the OME format. This function groups files by their base
    name and processes each group to create a combined raw file."""
    paths = glob.glob(path)
    groups = get_movie_groups(paths)
    n_groups = len(groups)
    if n_groups:
        for i, (basename, group) in enumerate(groups.items()):
            if verbose:
                print(
                    "Converting movie {}/{}...".format(i + 1, n_groups),
                    end="\r",
                )
            to_raw_combined(basename, group)
        if verbose:
            print()
    else:
        if verbose:
            print("No files matching {}".format(path))


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
        _write_metadata_dataset(locs_file, info)
    if _save_metadata_in_yaml():
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
        _write_metadata_dataset(locs_file, info)
    if _save_metadata_in_yaml():
        base, ext = os.path.splitext(path)
        info_path = base + ".yaml"
        save_info(info_path, info)


def load_locs(
    path: str, qt_parent: QtWidgets.QWidget | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    """Load localization data from an HDF5 file.

    Parameters
    ----------
    path : str
        The path to the HDF5 file containing localization data.
    qt_parent : QWidget or None, optional
        Parent widget for any Qt-related operations, default is None.

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
        locs = pd.read_hdf(path, key="locs")
    except KeyError as e:  # if "locs" key not found
        print(
            f"\nAn error occured. File: {path} does not contain a "
            "'locs' dataset."
        )
        if qt_parent is not None:
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
        _write_metadata_dataset(ids_file, info)
    if _save_metadata_in_yaml():
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
    np.savetxt(
        path,
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


def import_ts(path: str, pixelsize: float) -> tuple[pd.DataFrame, list[dict]]:
    """Import localization data from a ThunderSTORM .csv file.

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
