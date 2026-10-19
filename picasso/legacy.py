"""
picasso.legacy
~~~~~~~~~~~~~~

Conversion of files written before Picasso 1.0, whose lengths are in
camera pixels, to the current format, whose lengths are in nanometers.

Until Picasso 1.0, localization coordinates (``x``, ``y``), their
precisions (``lpx``, ``lpy``), spot widths (``sx``, ``sy``) and every
quantity derived from them (pick centers, drift, cluster centers, ...)
were stored in camera pixels, while ``z`` was in nm. Since Picasso 1.0
every length is in nm. Picasso tells the two apart by the format marker
``picasso_format`` that it writes into every file (the root attribute of
.hdf5 files, a key of .yaml files, a header line of drift .txt files):
files without it were written before 1.0, in camera pixels.

Picasso converts such files in memory whenever it opens them (see
``picasso.io.load_locs``), so old files keep working; converting them
once with this module (``picasso convert-units`` on the command line,
``File > Convert files to nm`` in Render, or :func:`convert_file` here)
saves the camera pixel size question and makes them readable by other
software as nm.

The conversion needs the camera pixel size in nm. Localization files
usually record it in their metadata (``Pixelsize``); every other file
needs it as an argument. The converted file records the conversion in
its metadata (a ``Convert units`` step) and carries the format marker.

Files and what is converted:

- Localizations (.hdf5 with a ``locs``, ``groups`` or ``clusters``
  dataset, e.g., localizations, cluster centers and pick properties):
  the columns of :data:`LEGACY_PIXEL_COLUMNS` and their ``_mean``,
  ``_std``, ``_min`` and ``_max`` statistics, the Filter ranges of those
  columns in the metadata, the canvas offsets and manual shifts of the
  Move tool, and the spatial index (dropped, Render rebuilds it).
- Pick regions (.yaml): the positions of the picks; the sizes are in nm
  since Picasso 0.6.
- Drift (.txt): ``x`` and ``y``.
- Astigmatism (3D) calibrations (.yaml): the polynomial coefficients
  (spot widths in nm) and the lateral corrections.
- Lateral (affine, chromatic) calibrations (.yaml) and spline PSF
  calibrations (.hdf5): the lateral corrections.
- SPINNA masks (.yaml next to the .npy): the extent of the mask.
- nanoTRON models (.yaml next to the .sav): the pick diameter and the
  rendered pixel size.

Examples
--------
>>> from picasso import legacy
>>> legacy.convert_file("movie_locs.hdf5")  # Pixelsize from the metadata
'movie_locs_nm.hdf5'
>>> legacy.convert_file("movie_locs_picks.yaml", pixelsize=130)
'movie_locs_picks_nm.yaml'
>>> legacy.convert_file("movie_locs_drift.txt", pixelsize=130, in_place=True)
'movie_locs_drift.txt'

:authors: Picasso Developers
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import copy
import logging
import os
import re
import tempfile
from typing import Callable

import h5py
import numpy as np
import pandas as pd

from . import io, lib

__all__ = [
    "LEGACY_PIXEL_COLUMNS",
    "FILE_KINDS",
    "file_kind",
    "needs_conversion",
    "convert_file",
    "convert_files",
    "locs_to_nm",
    "info_to_nm",
    "picks_to_nm",
    "drift_to_nm",
    "calibration_to_nm",
    "lateral_transforms_to_nm",
    "mask_info_to_nm",
    "nanotron_model_to_nm",
    "column_power",
]

_log = logging.getLogger(__name__)

#: Columns that Picasso before 1.0 stored in camera pixels, mapped to
#: the power of the pixel size that converts them to nm: 1 for lengths,
#: 2 for areas, 3 for volumes, 0.5 for ``d_zcalib`` (the root of a
#: width residual). ``convexhull`` is an area for 2D clusters and a
#: volume for 3D ones (None here, see :func:`column_power`). The
#: statistics ``<column>_mean``, ``_std``, ``_min`` and ``_max`` of a
#: length column (pick and track properties) are converted as well.
LEGACY_PIXEL_COLUMNS: dict[str, float | None] = {
    "x": 1,
    "y": 1,
    "lpx": 1,
    "lpy": 1,
    "sx": 1,
    "sy": 1,
    "sx_unc": 1,
    "sy_unc": 1,
    "std_x": 1,
    "std_y": 1,
    "com_x": 1,
    "com_y": 1,
    "x_pick_rot": 1,
    "y_pick_rot": 1,
    "d_zcalib": 0.5,
    "area": 2,
    "volume": 3,
    "convexhull": None,
}
_STATISTIC_SUFFIXES = ("_mean", "_std", "_min", "_max")

#: Kinds of files :func:`convert_file` converts, see the module
#: documentation.
FILE_KINDS = (
    "locs",
    "picks",
    "drift",
    "calibration",
    "spline",
    "mask",
    "nanotron",
)


_CONVERT_STEP = "Convert units"


def _step(pixelsize: float, source: str | None = None) -> dict:
    """The metadata entry recording a conversion."""
    entry = lib.step_info(
        _CONVERT_STEP,
        **{"From": "camera pixels", "To": "nm", "Pixelsize": float(pixelsize)},
    )
    if source is not None:
        entry["Converted from"] = source
    return entry


def _check_pixelsize(pixelsize, what: str) -> float:
    """``pixelsize`` as a positive float, else ``MissingPixelsizeError``."""
    if pixelsize is None:
        raise io.MissingPixelsizeError(
            f"The camera pixel size in nm is needed to convert {what} from "
            "camera pixels to nm. Pass it as pixelsize, e.g., "
            "picasso convert-units --pixelsize 130."
        )
    pixelsize = float(pixelsize)
    if not np.isfinite(pixelsize) or pixelsize <= 0:
        raise ValueError(
            f"The camera pixel size must be positive, got {pixelsize}."
        )
    return pixelsize


# ---------------------------------------------------------------------------
# in-memory conversions
# ---------------------------------------------------------------------------


def column_power(name: str, columns) -> float | None:
    """Power of the pixel size that converts a column written before
    Picasso 1.0 to nm, or None if the column is not a length.

    Parameters
    ----------
    name : str
        Name of the column.
    columns : iterable of str
        All columns of the table; ``convexhull`` is a volume if the table
        describes 3D clusters (it has a ``std_z`` or ``z`` column), else an
        area.

    Returns
    -------
    power : float or None
        1 for lengths, 2 for areas, 3 for volumes, 0.5 for ``d_zcalib``,
        None for columns that are not in camera pixels (e.g., ``z``,
        ``frame``, ``photons``).

    Examples
    --------
    >>> from picasso import legacy
    >>> legacy.column_power("lpx", ["x", "y", "lpx"])
    1
    >>> legacy.column_power("x_std", ["x_mean", "x_std"])
    1
    """
    columns = set(columns)
    if name == "convexhull":
        return 3 if ("std_z" in columns or "z" in columns) else 2
    if name in LEGACY_PIXEL_COLUMNS:
        return LEGACY_PIXEL_COLUMNS[name]
    for suffix in _STATISTIC_SUFFIXES:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if LEGACY_PIXEL_COLUMNS.get(base) == 1:
                return 1
    return None


def locs_to_nm(
    locs: pd.DataFrame, pixelsize: float, *, copy: bool = True
) -> pd.DataFrame:
    """Localizations (or cluster centers, pick properties, ...) written
    before Picasso 1.0 converted from camera pixels to nm.

    Every column with a :func:`column_power` is multiplied by the pixel
    size to that power; the dtypes are kept.

    Parameters
    ----------
    locs : pd.DataFrame
        Table in camera pixels.
    pixelsize : float
        Camera pixel size in nm.
    copy : bool, optional
        Whether to convert a copy (default) or ``locs`` itself.

    Returns
    -------
    locs : pd.DataFrame
        The table in nm.
    """
    pixelsize = _check_pixelsize(pixelsize, "localizations")
    if copy:
        locs = locs.copy()
    for name in locs.columns:
        power = column_power(name, locs.columns)
        if power is None:
            continue
        values = locs[name].to_numpy()
        if values.dtype.kind not in "fiu":
            continue
        dtype = values.dtype if values.dtype.kind == "f" else np.float32
        locs[name] = (values.astype(np.float64) * pixelsize**power).astype(
            dtype
        )
    return locs


def _is_range(value) -> bool:
    """Whether ``value`` is a ``[min, max]`` Filter range."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in value
        )
    )


def info_to_nm(
    info: list[dict],
    pixelsize: float,
    *,
    columns=None,
    source: str | None = None,
) -> list[dict]:
    """Metadata of localizations written before Picasso 1.0 converted
    to nm.

    The Filter ranges ``column: [min, max]`` of the columns in camera
    pixels are scaled and a ``Convert units`` step recording the pixel
    size is appended, so ``Pixelsize`` is present afterwards.

    Parameters
    ----------
    info : list of dicts
        Metadata of the file.
    pixelsize : float
        Camera pixel size in nm.
    columns : iterable of str, optional
        Columns of the table, deciding the power of ``convexhull`` (see
        :func:`column_power`). Default is None, i.e., the names of the
        ranges alone decide.
    source : str, optional
        Path of the converted file, recorded in the step. Default None.

    Returns
    -------
    info : list of dicts
        A deep copy of the metadata in nm.
    """
    pixelsize = _check_pixelsize(pixelsize, "metadata")
    columns = list(columns) if columns is not None else []
    new_info = []
    for entry in copy.deepcopy(info):
        if not isinstance(entry, dict):
            new_info.append(entry)
            continue
        if lib._is_filter_dict(entry):
            for key, value in list(entry.items()):
                power = column_power(str(key), columns or [key])
                if power is not None and _is_range(value):
                    entry[key] = [float(v) * pixelsize**power for v in value]
        new_info.append(entry)
    new_info.append(_step(pixelsize, source))
    return new_info


def _scale_value(value, factor: float):
    """``value`` (a number or a nested list of numbers) times ``factor``."""
    if isinstance(value, (list, tuple)):
        return [_scale_value(v, factor) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) * factor
    return value


def picks_to_nm(regions: dict, pixelsize: float) -> dict:
    """Pick regions written before Picasso 1.0 converted to nm.

    The positions (``Centers``, ``Center-Axis-Points``, ``Vertices``,
    ``Corners`` and the ``Path`` of brush strokes) are scaled; sizes
    stored in camera pixels by Picasso before 0.6 (``Diameter``,
    ``Width``) are scaled and renamed to their ``(nm)`` keys; sizes in nm
    are kept. The format marker is added.

    Parameters
    ----------
    regions : dict
        Contents of a picks .yaml file, see ``io.load_picks``.
    pixelsize : float
        Camera pixel size in nm.

    Returns
    -------
    regions : dict
        A deep copy in nm.
    """
    pixelsize = _check_pixelsize(pixelsize, "pick regions")
    regions = copy.deepcopy(regions)
    for key in ("Centers", "Center-Axis-Points", "Vertices", "Corners"):
        if key in regions:
            regions[key] = _scale_value(regions[key], pixelsize)
    if "Strokes" in regions:
        for stroke in regions["Strokes"]:
            stroke["Path"] = _scale_value(stroke["Path"], pixelsize)
            if "Width" in stroke and "Width (nm)" not in stroke:
                stroke["Width (nm)"] = float(stroke.pop("Width")) * pixelsize
    for key in ("Diameter", "Width", "Side Length"):
        if key in regions:
            value = regions.pop(key)
            regions.setdefault(f"{key} (nm)", float(value) * pixelsize)
    regions[io.FORMAT_VERSION_ATTR] = io.FORMAT_VERSION
    return regions


def drift_to_nm(drift: pd.DataFrame, pixelsize: float) -> pd.DataFrame:
    """Drift written before Picasso 1.0 converted to nm: ``x`` and ``y``
    are scaled, ``z`` (nm already) is kept.

    Parameters
    ----------
    drift : pd.DataFrame
        Drift with columns ``x``, ``y`` and optionally ``z``, see
        ``io.load_drift``.
    pixelsize : float
        Camera pixel size in nm.

    Returns
    -------
    drift : pd.DataFrame
        A copy in nm.
    """
    pixelsize = _check_pixelsize(pixelsize, "drift")
    drift = drift.copy()
    for column in ("x", "y"):
        if column in drift.columns:
            drift[column] = drift[column].to_numpy() * pixelsize
    return drift


def lateral_transforms_to_nm(entries: list, pixelsize: float) -> list:
    """Lateral corrections fitted in camera pixels expressed in nm.

    Each transform is rescaled (see
    ``picasso.transforms.Transform.rescaled``) and its decomposition
    recomputed in nm.

    Parameters
    ----------
    entries : list
        Entries of a calibration's ``Lateral transforms`` list (dicts with
        a ``Transform`` key) or bare transform dicts.
    pixelsize : float
        Camera pixel size in nm.

    Returns
    -------
    entries : list
        Deep copies in nm.
    """
    from . import transforms

    pixelsize = _check_pixelsize(pixelsize, "lateral transforms")
    converted = []
    for entry in copy.deepcopy(entries):
        if isinstance(entry, dict) and "Transform" in entry:
            model = transforms.from_dict(entry["Transform"]).rescaled(
                pixelsize
            )
            entry["Transform"] = model.to_dict()
            if "Decomposition" in entry:
                entry["Decomposition"] = model.decompose()
            entry[io.FORMAT_VERSION_ATTR] = io.FORMAT_VERSION
            converted.append(entry)
        else:
            converted.append(
                transforms.from_dict(entry).rescaled(pixelsize).to_dict()
            )
    return converted


def calibration_to_nm(calibration: dict, pixelsize: float) -> dict:
    """A calibration (astigmatism, lateral or spline PSF) written before
    Picasso 1.0 converted to nm.

    The astigmatism polynomials (``X Coefficients``, ``Y Coefficients``,
    spot widths against z) are multiplied by the pixel size, the lateral
    corrections rescaled (:func:`lateral_transforms_to_nm`) and the
    format marker added. The lateral knots of a spline PSF calibration
    stay on the camera pixel grid, where the fit takes place.

    Parameters
    ----------
    calibration : dict
        The calibration, see ``io.load_any_calibration``.
    pixelsize : float
        Camera pixel size in nm.

    Returns
    -------
    calibration : dict
        A copy in nm.
    """
    pixelsize = _check_pixelsize(pixelsize, "a calibration")
    calibration = dict(calibration)
    for key in ("X Coefficients", "Y Coefficients"):
        if key in calibration:
            calibration[key] = [float(c) * pixelsize for c in calibration[key]]
    if calibration.get(lib.LATERAL_TRANSFORMS_KEY):
        calibration[lib.LATERAL_TRANSFORMS_KEY] = lateral_transforms_to_nm(
            calibration[lib.LATERAL_TRANSFORMS_KEY], pixelsize
        )
    calibration[io.FORMAT_VERSION_ATTR] = io.FORMAT_VERSION
    return calibration


def mask_info_to_nm(info: dict, pixelsize: float | None = None) -> dict:
    """SPINNA mask metadata written before Picasso 1.0 converted to nm:
    the lateral extent (``x_min``, ``x_max``, ``y_min``, ``y_max``) is
    scaled; ``z_min`` and ``z_max`` are in nm already.

    Parameters
    ----------
    info : dict
        The mask metadata, see ``io.load_mask``.
    pixelsize : float, optional
        Camera pixel size in nm. Default is None, i.e., the
        ``Camera pixelsize (nm)`` entry of the metadata.

    Returns
    -------
    info : dict
        A copy in nm.
    """
    if pixelsize is None:
        pixelsize = info.get("Camera pixelsize (nm)")
    pixelsize = _check_pixelsize(pixelsize, "a mask")
    info = dict(info)
    for key in ("x_min", "x_max", "y_min", "y_max"):
        if key in info:
            info[key] = float(info[key]) * pixelsize
    info[io.FORMAT_VERSION_ATTR] = io.FORMAT_VERSION
    return info


def nanotron_model_to_nm(model_info: dict, pixelsize: float) -> dict:
    """nanoTRON model metadata written before Picasso 1.0 converted to
    nm: ``Pick Diameter`` (camera pixels) becomes ``Pick Diameter (nm)``
    and ``Oversampling`` (rendered pixels per camera pixel) becomes
    ``Display pixel size (nm)``.

    Parameters
    ----------
    model_info : dict
        The model metadata (the .yaml next to the .sav model file).
    pixelsize : float
        Camera pixel size in nm the model was trained with.

    Returns
    -------
    model_info : dict
        A copy in nm.
    """
    pixelsize = _check_pixelsize(pixelsize, "a nanoTRON model")
    model_info = dict(model_info)
    if "Pick Diameter" in model_info:
        diameter = float(model_info.pop("Pick Diameter"))
        model_info.setdefault("Pick Diameter (nm)", diameter * pixelsize)
    if "Oversampling" in model_info:
        oversampling = float(model_info.pop("Oversampling"))
        model_info.setdefault(
            "Display pixel size (nm)", pixelsize / oversampling
        )
    model_info[io.FORMAT_VERSION_ATTR] = io.FORMAT_VERSION
    return model_info


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

_HDF5_EXTENSIONS = (".hdf5", ".h5", ".hdf")
_LOCS_DATASETS = ("locs", "groups", "clusters")


def _hdf5_kind(path: str) -> str | None:
    """``"locs"`` or ``"spline"`` for an HDF5 file, None if neither."""
    try:
        with h5py.File(path, "r") as hdf_file:
            if "coefficients" in hdf_file and "metadata" in hdf_file.attrs:
                return "spline"
            if any(key in hdf_file for key in _LOCS_DATASETS):
                return "locs"
    except OSError:
        return None
    return None


def _yaml_kind(path: str, data) -> str | None:
    """Kind of a .yaml file from its parsed content."""
    if isinstance(data, list):
        # the metadata of a localization file, or a nanoTRON model
        if len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        else:
            return None
    if not isinstance(data, dict):
        return None
    if "Shape" in data or "Centers" in data or "Center-Axis-Points" in data:
        return "picks"
    if "X Coefficients" in data or lib.LATERAL_TRANSFORMS_KEY in data:
        return "calibration"
    if "Binsize (nm)" in data and "x_min" in data:
        return "mask"
    if "Pick Diameter" in data or "Pick Diameter (nm)" in data:
        if "Oversampling" in data or "Display pixel size (nm)" in data:
            return "nanotron"
    return None


def file_kind(path: str) -> str | None:
    """Kind of a file :func:`convert_file` can convert.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    kind : str or None
        One of :data:`FILE_KINDS`, or None if the file is not one of
        them. The .yaml metadata next to a localization file counts as
        that localization file (``"locs"``).
    """
    base, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in _HDF5_EXTENSIONS:
        return _hdf5_kind(path)
    if ext == ".txt":
        return "drift" if _read_drift_table(path) is not None else None
    if ext == ".yaml":
        for hdf_ext in _HDF5_EXTENSIONS:
            if os.path.isfile(base + hdf_ext):
                return "locs" if _hdf5_kind(base + hdf_ext) == "locs" else None
        try:
            data = io.load_yaml(path, all_documents=True)
        except Exception:
            return None
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        return _yaml_kind(path, data)
    return None


def _read_drift_table(path: str) -> np.ndarray | None:
    """The numbers of a drift .txt file as a 2D array of 2 or 3 columns,
    None if the file is not one."""
    try:
        table = np.loadtxt(path, ndmin=2)
    except (ValueError, OSError):
        return None
    if table.ndim != 2 or table.shape[1] not in (2, 3):
        return None
    return table


def _has_marker(path: str, kind: str) -> bool:
    """Whether the file carries the format marker, i.e., is in nm."""
    if kind == "locs":
        base, ext = os.path.splitext(path)
        if ext.lower() == ".yaml":
            path = next(
                base + e for e in _HDF5_EXTENSIONS if os.path.isfile(base + e)
            )
        return io.get_format_version(path) is not None
    if kind == "spline":
        # as written: the loader converts files without the marker
        return io.FORMAT_VERSION_ATTR in io._read_spline_calibration(path)
    if kind == "drift":
        return io._drift_file_version(path) is not None
    data = io.load_yaml(path, all_documents=True)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    return isinstance(data, dict) and io.FORMAT_VERSION_ATTR in data


def needs_conversion(path: str) -> bool:
    """Whether a file was written before Picasso 1.0, in camera pixels,
    and :func:`convert_file` would convert it.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    needs : bool
        True for a file of a known kind (see :func:`file_kind`) without
        the format marker.
    """
    kind = file_kind(path)
    if kind is None:
        return False
    return not _has_marker(path, kind)


def _output_path(path: str, output: str | None, in_place: bool) -> str:
    """Where the converted file goes."""
    if output is not None:
        return output
    if in_place:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_nm{ext}"


def _locs_hdf5_path(path: str) -> str:
    """The .hdf5 file of a localization file given by it or by its .yaml."""
    base, ext = os.path.splitext(path)
    if ext.lower() == ".yaml":
        for hdf_ext in _HDF5_EXTENSIONS:
            if os.path.isfile(base + hdf_ext):
                return base + hdf_ext
        raise FileNotFoundError(f"No localization file next to {path}.")
    return path


def _convert_locs_file(
    path: str, pixelsize: float | None, output: str, progress
) -> None:
    """Convert a localization-type HDF5 file (see ``_LOCS_DATASETS``)."""
    with h5py.File(path, "r") as hdf_file:
        keys = [key for key in _LOCS_DATASETS if key in hdf_file]
    if "locs" in keys:
        locs = io._read_locs_dataset(path, progress=progress)
        locs = io._rename_legacy_columns(locs)
    try:
        info = io.load_info(path)
    except io.NoMetadataFileError:
        info = []
    if pixelsize is None:
        pixelsize = lib.get_from_metadata(info, "Pixelsize")
    pixelsize = _check_pixelsize(
        pixelsize, f"{path} (its metadata has no Pixelsize)"
    )
    tables = {}
    for key in keys:
        table = locs if key == "locs" else pd.read_hdf(path, key=key)
        tables[key] = locs_to_nm(table, pixelsize, copy=False)
    columns = tables[keys[0]].columns if keys else []
    new_info = info_to_nm(info, pixelsize, columns=columns, source=path)
    if keys == ["locs"]:
        io.save_locs(output, tables["locs"], new_info)
    else:
        io.save_datasets(output, new_info, **tables)
    # a .yaml next to the input takes precedence over the embedded
    # metadata when loading, so the output gets one as well
    if os.path.isfile(os.path.splitext(path)[0] + ".yaml"):
        io.save_info(os.path.splitext(output)[0] + ".yaml", new_info)


def _convert_yaml_file(path: str, kind: str, pixelsize, output: str) -> None:
    """Convert a picks, calibration, mask or nanoTRON .yaml file."""
    documents = io.load_yaml(path, all_documents=True)
    as_list = isinstance(documents, list) and len(documents) == 1
    data = documents[0] if as_list else documents
    if kind == "picks":
        converted = picks_to_nm(data, pixelsize)
    elif kind == "calibration":
        converted = calibration_to_nm(data, pixelsize)
    elif kind == "mask":
        converted = mask_info_to_nm(data, pixelsize)
    else:
        converted = nanotron_model_to_nm(data, pixelsize)
    if as_list:
        io.save_info(output, [converted])
    else:
        io.save_yaml(output, converted)


def convert_file(
    path: str,
    pixelsize: float | None = None,
    *,
    output: str | None = None,
    in_place: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Convert a file written before Picasso 1.0 from camera pixels to nm.

    See the module documentation for the kinds of files and what is
    converted in each. The .yaml metadata of a localization file is
    converted with it (``path`` may name either).

    Parameters
    ----------
    path : str
        Path to the file.
    pixelsize : float, optional
        Camera pixel size in nm. Default is None: the ``Pixelsize`` of a
        localization file's metadata or the ``Camera pixelsize (nm)`` of
        a mask; every other file needs it.
    output : str, optional
        Path of the converted file. Default is None: the input path with
        ``_nm`` appended to its name (``movie_locs_nm.hdf5``), or the
        input path itself if ``in_place``.
    in_place : bool, optional
        Whether to overwrite the input file (and its .yaml). Default is
        False.
    progress : callable, optional
        Called as ``progress(done, total)`` while the localizations are
        read. Default is None.

    Returns
    -------
    output : str
        Path of the converted file.

    Raises
    ------
    ValueError
        If the file is not of a known kind, or carries the format marker
        already (it is in nm).
    picasso.io.MissingPixelsizeError
        If the pixel size is neither given nor found in the file.
    """
    kind = file_kind(path)
    if kind is None:
        raise ValueError(
            f"{path} is not a file Picasso converts: localizations (.hdf5), "
            "pick regions (.yaml), drift (.txt), calibrations (.yaml, "
            ".hdf5), SPINNA masks (.yaml) or nanoTRON models (.yaml)."
        )
    if _has_marker(path, kind):
        raise ValueError(
            f"{path} is in the current format already (nm); nothing to "
            "convert."
        )
    if kind == "locs":
        path = _locs_hdf5_path(path)
    output = _output_path(path, output, in_place)
    if kind in ("drift", "picks", "calibration", "mask", "nanotron"):
        if kind == "mask" and pixelsize is None:
            pixelsize = io.load_yaml(path).get("Camera pixelsize (nm)")
        pixelsize = _check_pixelsize(pixelsize, path)

    def write(target: str) -> None:
        if kind == "locs":
            _convert_locs_file(path, pixelsize, target, progress)
        elif kind == "drift":
            drift = io.load_drift(path, pixelsize=pixelsize)
            io.save_drift(target, drift)
        elif kind == "spline":
            # the loader converts it, with the pixel size the calibration
            # records unless one is given
            io.save_spline_calibration(
                target, io.load_spline_calibration(path, pixelsize=pixelsize)
            )
        else:
            _convert_yaml_file(path, kind, pixelsize, target)

    if os.path.abspath(output) != os.path.abspath(path):
        write(output)
        return output
    # in place: write next to the file, then replace it (and its .yaml)
    directory = os.path.dirname(os.path.abspath(path))
    base, ext = os.path.splitext(os.path.basename(path))
    handle, temporary = tempfile.mkstemp(
        prefix=base + "_", suffix=ext, dir=directory
    )
    os.close(handle)
    try:
        write(temporary)
        os.replace(temporary, path)
        temporary_yaml = os.path.splitext(temporary)[0] + ".yaml"
        if os.path.isfile(temporary_yaml):
            os.replace(temporary_yaml, os.path.splitext(path)[0] + ".yaml")
    finally:
        for leftover in (temporary, os.path.splitext(temporary)[0] + ".yaml"):
            if os.path.isfile(leftover):
                os.remove(leftover)
    return output


def convert_files(
    paths: list[str],
    pixelsize: float | None = None,
    *,
    in_place: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> list[tuple[str, str | None, str | None]]:
    """Convert several files, see :func:`convert_file`.

    Files that are in nm already or not of a known kind are skipped;
    errors of single files do not stop the others.

    Parameters
    ----------
    paths : list of str
        Paths to the files.
    pixelsize : float, optional
        Camera pixel size in nm for files that do not record it. Default
        None.
    in_place : bool, optional
        Whether to overwrite the input files. Default is False.
    progress : callable, optional
        Called as ``progress(done, total)`` after each file. Default None.

    Returns
    -------
    results : list of tuples
        One ``(path, output, message)`` per file: ``output`` is the
        converted file or None if the file was skipped or failed, and
        ``message`` says why (None on success).
    """
    results = []
    seen = set()
    for i, path in enumerate(paths):
        output = message = None
        try:
            kind = file_kind(path)
            key = os.path.abspath(
                _locs_hdf5_path(path) if kind == "locs" else path
            )
            if kind is None:
                message = "not a file Picasso converts"
            elif key in seen:
                message = "already converted (named twice)"
            elif _has_marker(path, kind):
                message = "already in nm"
            else:
                output = convert_file(path, pixelsize, in_place=in_place)
            seen.add(key)
        except Exception as error:  # report, do not stop the batch
            message = f"{type(error).__name__}: {error}"
        results.append((path, output, message))
        if progress is not None:
            progress(i + 1, len(paths))
    return results
