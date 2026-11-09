"""
__main__.py
~~~~~~~~~~~

Picasso command line interface.

:authors: Rafal Kowalewski, Joerg Schnitzbauer,
    Maximilian Thomas Strauss
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import contextlib
import os.path
import argparse
import logging
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal
import pandas as pd
from . import deprecation, docs_url
from .transforms import MODELS as TRANSFORM_MODELS

if TYPE_CHECKING:  # imported lazily, like every analysis module here
    import numpy as np
    from matplotlib.figure import Figure
    from .localize import (
        IdentificationMethod,
        IdentificationParameters,
        NetGradient,
    )
    from .wavelet import WaveletParameters


def picasso_logo():
    """Print the Picasso logo to the console."""
    print("    ____  _____________   __________ ____ ")
    print("   / __ \\/  _/ ____/   | / ___/ ___// __ \\")
    print("  / /_/ // // /   / /| | \\__ \\\\__ \\/ / / /")
    print(" / _____/ // /___/ ___ |___/ ___/ / /_/ / ")
    print("/_/   /___/\\____/_/  |_/____/____/\\____/  ")
    print("                                          ")


def _csv2hdf(path: str, pixelsize: float) -> None:
    """Convert CSV localization files to HDF5 format.

    Parameters
    ----------
    path : str
        Path to the CSV localization files.
    pixelsize : float
        Camera pixel size in nanometers.
    """
    from glob import glob
    from tqdm import tqdm as _tqdm

    paths = glob(path)
    if paths:
        from .io import import_ts

        for path in _tqdm(paths, desc="Converting from ThunderSTORM"):
            import_ts(path, pixelsize)
    print("Complete.")


def _hdf2csv(path: str) -> None:
    """Convert HDF5 localization files to CSV format (unchanged
    columns)."""
    from glob import glob
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs

        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print(f"Converting {path}")
                out_path = base + ".csv"
                locs = load_locs(path)[0]
                print(f"A total of {len(locs)} rows loaded.")
                locs.to_csv(out_path, sep=",", encoding="utf-8", index=False)
    print("Complete.")


def _hdf2ts(path: str) -> None:
    """Convert HDF5 localization files to ThunderSTORM CSV format."""
    from glob import glob
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs, export_thunderstorm

        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print(f"Converting {path}")
                out_path = base + ".csv"
                locs, info = load_locs(path)
                export_thunderstorm(out_path, locs, info)
    print("Complete.")


def _hdf2imagej(path: str) -> None:
    """Convert HDF5 localization files to ImageJ txt format."""
    from glob import glob
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs, export_txt_imagej

        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print(f"Converting {path}")
                out_path = base + ".txt"
                locs, info = load_locs(path)
                export_txt_imagej(out_path, locs, info)
    print("Complete.")


def _hdf2nis(path: str) -> None:
    """Convert HDF5 localization files to NIS txt format."""
    from glob import glob
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs, export_txt_nis

        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print(f"Converting {path}")
                out_path = base + ".nis.txt"
                locs, info = load_locs(path)
                export_txt_nis(out_path, locs, info)
    print("Complete.")


def _hdf2chimera(path: str) -> None:
    """Convert HDF5 localization files to Chimera .xyz format."""
    from glob import glob
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs, export_xyz_chimera

        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print(f"Converting {path}")
                out_path = base + ".chi.xyz"
                locs, info = load_locs(path)
                export_xyz_chimera(out_path, locs, info)
    print("Complete.")


def _hdf2visp(path: str) -> None:
    """Convert HDF5 localization files to VISP format."""
    from glob import glob
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs, export_3d_visp

        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print(f"Converting {path}")
                out_path = base + ".visp.3d"
                locs, info = load_locs(path)
                export_3d_visp(out_path, locs, info)
    print("Complete.")


def _smap2hdf(path: str, pixelsize: float) -> None:
    """Convert SMAP _sml.mat localization files to HDF5 format.

    Parameters
    ----------
    path : str
        Path (unix style pattern) to the SMAP _sml.mat files.
    pixelsize : float
        Camera pixel size in nanometers.
    """
    from glob import glob
    from tqdm import tqdm as _tqdm

    paths = glob(path)
    if paths:
        import os.path
        from .io import import_smap, save_locs

        for path in _tqdm(paths, desc="Converting from SMAP"):
            locs, info = import_smap(path, pixelsize)
            base, ext = os.path.splitext(path)
            save_locs(base + "_locs.hdf5", locs, info)
    print("Complete.")


def _hdf2smap(path: str) -> None:
    """Convert HDF5 localization files to SMAP _sml.mat format."""
    from glob import glob
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs, export_smap

        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print(f"Converting {path}")
                out_path = base + "_sml.mat"
                locs, info = load_locs(path)
                export_smap(out_path, locs, info)
    print("Complete.")


def _pyme2hdf(path: str, pixelsize: float | None = None) -> None:
    """Convert PyME (python-microscopy) .h5r/.hdf localization files to
    HDF5 format. A file with several channels (``probe`` values) is
    saved as one file per channel, ``<base>_probe<k>_locs.hdf5``.

    Parameters
    ----------
    path : str
        Path (unix style pattern) to the PyME .h5r/.hdf files.
    pixelsize : float or None, optional
        Camera pixel size in nanometers, used for files that do not
        store it. Default is None.
    """
    from glob import glob
    from tqdm import tqdm as _tqdm

    paths = glob(path)
    if paths:
        import os.path
        from .io import import_pyme, save_locs, split_pyme_channels

        for path in _tqdm(paths, desc="Converting from PyME"):
            locs, info = import_pyme(path, pixelsize)
            channels = split_pyme_channels(path, locs, info)
            for channel_path, locs, info in channels:
                base, ext = os.path.splitext(channel_path)
                save_locs(base + "_locs.hdf5", locs, info)
    print("Complete.")


def _minflux2hdf(
    path: str, average_traces: bool = False, origin: str = "data"
) -> None:
    """Convert MINFLUX localizations (pyMINFLUX .pmx, or a .csv table
    such as pyMINFLUX's export) to HDF5 format. A file with several
    fluorophores is saved as one file per fluorophore,
    ``<base>_<fluorophore>_locs.hdf5``.

    Parameters
    ----------
    path : str
        Path (unix style pattern) to the MINFLUX .pmx/.csv files.
    average_traces : bool, optional
        If True, save one localization per MINFLUX trace (the
        photon-weighted mean) instead of every localization, see
        ``picasso.io.import_minflux``. Default is False.
    origin : {"data", "stage"}, optional
        ``"data"`` moves the localizations next to the origin of the
        field of view (the shift is stored in the metadata), ``"stage"``
        keeps MINFLUX's absolute coordinates. Default is ``"data"``.
    """
    from glob import glob
    from tqdm import tqdm as _tqdm

    paths = glob(path)
    if paths:
        import os.path
        from .io import import_minflux, save_locs, split_minflux_channels

        for path in _tqdm(paths, desc="Converting from pyMINFLUX"):
            locs, info = import_minflux(
                path, average_traces=average_traces, origin=origin
            )
            channels = split_minflux_channels(path, locs, info)
            for channel_path, locs, info in channels:
                base, ext = os.path.splitext(channel_path)
                save_locs(base + "_locs.hdf5", locs, info)
    print("Complete.")


def _hdf2pyme(path: str) -> None:
    """Convert HDF5 localization files to PyME (python-microscopy)
    .hdf format, one file per input file."""
    from glob import glob
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs, export_pyme

        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print(f"Converting {path}")
                out_path = base + ".hdf"
                locs, info = load_locs(path)
                export_pyme(out_path, locs, info)
    print("Complete.")


def _link(files: str, d_max: float, tolerance: float) -> None:
    """Link localizations in HDF5 files, see ``postprocess.link`` for
    details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, postprocess

        for path in paths:
            try:
                locs, info = io.load_locs(path)
            except io.NoMetadataFileError:
                continue
            linked_locs, linked_info = postprocess.link(
                locs, info, r_max=d_max, max_dark_time=tolerance
            )
            base, ext = os.path.splitext(path)
            io.save_locs(base + "_link.hdf5", linked_locs, linked_info)


def _clusterfilter(
    files: str,
    clusterfile: str,
    parameter: str,
    minval: float,
    maxval: float,
) -> None:
    """Filter localizations based on cluster parameters.

    Parameters
    ----------
    files : str
        Glob pattern for input files.
    clusterfile : str
        Path to the cluster file.
    parameter : str
        Name of the parameter to filter on.
    minval : float
        Minimum value for the parameter.
    maxval : float
        Maximum value for the parameter.
    """
    from glob import glob

    paths = glob(files)
    if paths:
        from . import io

        for path in paths:
            try:
                locs, info = io.load_locs(path)
            except io.NoMetadataFileError:
                continue

            clusters, _ = io.load_locs(clusterfile)
            if parameter not in clusters.columns:
                print("Error: Field {} not found.".format(parameter))
                continue
            _clusterfilter_locs(
                clusters,
                locs,
                info,
                parameter,
                minval,
                maxval,
                path,
                clusterfile,
            )


def _clusterfilter_locs(
    clusters: pd.DataFrame,
    locs: pd.DataFrame,
    info: list[dict],
    parameter: str,
    minval: float,
    maxval: float,
    path: str,
    clusterfile: str,
) -> None:
    import numpy as np
    from . import io, lib

    selector = (clusters[parameter] > minval) & (clusters[parameter] < maxval)
    if np.sum(selector) == 0:
        print("Error: No localizations in range. Filtering aborted.")
    elif np.sum(selector) == len(selector):
        print("Error: All localizations in range. Filtering aborted.")
    else:
        base, ext = os.path.splitext(path)
        steps = (("in", selector), ("out", ~selector))
        for number, (suffix, selected) in enumerate(steps, 1):
            print(f"Isolating locs.. Step {number}: {suffix} of range")
            groups = clusters["group"][selected]
            filtered = locs[locs["group"].isin(groups)]
            filtered = filtered.sort_values(kind="quicksort", by="frame")
            clusterfilter_info = {
                **lib.step_info(f"Clusterfilter - {suffix}"),
                "Cluster file": clusterfile,
                "Parameter": parameter,
                "Minval": minval,
                "Maxval": maxval,
            }
            out_path = f"{base}_filter_{suffix}.hdf5"
            io.save_locs(out_path, filtered, info + [clusterfilter_info])
            print("Complete. Saved to: {}".format(out_path))


def _undrift_rcc(
    files: str,
    segmentation: int,
    display: bool = False,
    fromfile: str | None = None,
) -> None:
    """Run RCC undrifting on the given files. See
    ``postprocess.undrift`` for details. Alternatively, it can read the
    drift .txt file to apply the drift correction."""
    import glob
    from . import io, lib, postprocess

    paths = glob.glob(files)
    for path in paths:
        try:
            locs, info = io.load_locs(path)
        except io.NoMetadataFileError:
            continue
        if fromfile is not None:
            # the drift is in nm; files written before Picasso 1.0 are
            # converted with the pixel size of the localizations
            drift = io.load_drift(
                fromfile, pixelsize=lib.get_from_metadata(info, "Pixelsize")
            )
            # records the drift file (drift.attrs["Path"])
            locs, info = postprocess.apply_drift(locs, info, drift=drift)

            if display:
                import matplotlib.pyplot as plt

                plt.style.use("ggplot")
                fig = plt.Figure(figsize=(10, 6), constrained_layout=True)
                postprocess.plot_drift(drift, fig)
                plt.show()
        else:
            print(f"Undrifting file {path}")
            try:
                locs, info, drift = postprocess.undrift(
                    locs,
                    info,
                    segmentation=segmentation,
                    display=display,
                    progress="console",
                )
            except ValueError as e:
                raise SystemExit(f"{path}: {e}")

        base, ext = os.path.splitext(path)
        io.save_locs(base + "_undrift.hdf5", locs, info)
        io.save_drift(base + "_drift.txt", drift)


def _undrift_aim(
    files: str,
    segmentation: int,
    intersectdist: float = 20.0,
    roiradius: float = 60.0,
) -> None:
    """Run AIM undrifting on the given files. See ``aim.aim`` for
    details. ``intersectdist`` and ``roiradius`` are in nm."""
    import glob
    from . import io, aim

    paths = glob.glob(files)
    for path in paths:
        try:
            locs, info = io.load_locs(path)
        except io.NoMetadataFileError:
            continue
        print("Undrifting file {}".format(path))
        locs, new_info, drift = aim.aim(
            locs,
            info,
            segmentation=segmentation,
            intersect_d=intersectdist,
            roi_r=roiradius,
            progress="console",
        )
        base, ext = os.path.splitext(path)
        io.save_locs(base + "_aim.hdf5", locs, new_info)
        io.save_drift(base + "_aimdrift.txt", drift)


#: Frames per segment of the RCC that ``picasso undrift-fiducials`` runs
#: before finding the fiducials.
_PRE_RCC_SEGMENTATION = 2000


def _undrift_fiducials(files: str) -> None:
    """Automatically pick fiducials and use them for drift correction.
    See ``postprocess.undrift_from_picked`` and
    ``imageprocess.find_fiducials`` for more details.

    Uses RCC with segments of 2000 frames (fewer for movies shorter than
    8000 frames, so that there are at least 4 segments) before, so that
    fiducials can be detected.
    """
    import glob
    from . import io, lib, postprocess

    paths = glob.glob(files)
    for path in paths:
        try:
            locs, info = io.load_locs(path)
        except io.NoMetadataFileError:
            continue

        print(f"Undrifting file {path}.")
        n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)
        segmentation = min(_PRE_RCC_SEGMENTATION, n_frames // 4)
        if segmentation >= 1:
            locs, info, _ = postprocess.undrift(
                locs,
                info,
                segmentation=segmentation,
                display=False,
                progress="console",
            )
            print("pre-RCC done.")
        else:  # too short to segment, and too short to drift much
            print("pre-RCC skipped, the movie is too short.")
        locs, info, drift = postprocess.undrift_from_fiducials(
            locs, info, progress="console"
        )

        base, ext = os.path.splitext(path)
        io.save_locs(base + "_undrift_fiducials.hdf5", locs, info)
        io.save_drift(base + "_drift_fiducials.txt", drift)
        print("Saved undrifted localizations.")


def _density(files: str, radius: float) -> None:
    """Compute local density of localizations in HDF5 files. See
    ``postprocess.compute_local_density`` for details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, postprocess

        for path in paths:
            locs, info = io.load_locs(path)
            locs, info = postprocess.compute_local_density(
                locs, info, radius=radius
            )
            base, ext = os.path.splitext(path)
            io.save_locs(base + "_density.hdf5", locs, info)


def _filter(files: str, conditions: list[str]) -> None:
    """Keep the localizations in HDF5 files that satisfy conditions,
    applied in the given order. See ``expression.filter_locs`` for
    details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, expression

        conditions = [c.strip() for c in conditions]
        for path in paths:
            locs, info = io.load_locs(path)
            n_before = len(locs)
            try:
                locs, info = expression.filter_locs(
                    locs, info, conditions=conditions
                )
            except expression.ExpressionError as e:
                raise SystemExit(f"{path}: {e}")
            print(f"{path}: kept {len(locs):,} of {n_before:,} localizations.")
            base, _ = os.path.splitext(path)
            io.save_locs(base + "_filter.hdf5", locs, info)


def _dbscan(
    files: str,
    radius: float,
    min_density: float,
    radius_z: float | None = None,
) -> None:
    """Run DBSCAN clustering on localizations in HDF5 files. See
    ``clusterer.dbscan`` for details. The radii are in nm."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, clusterer

        for path in paths:
            print("Loading {} ...".format(path))
            locs, info = io.load_locs(path)
            locs, info = clusterer.dbscan(
                locs,
                info,
                radius=radius,
                min_samples=min_density,
                radius_z=radius_z,
            )
            clusters, clusters_info = clusterer.find_cluster_centers(
                locs, info, progress="console"
            )
            base, _ = os.path.splitext(path)
            io.save_locs(base + "_dbscan.hdf5", locs, info)
            io.save_locs(base + "_dbclusters.hdf5", clusters, clusters_info)
            print(
                "Clustering executed. Results are saved in: \n"
                f"{base}_dbscan.hdf5\n"
                f"{base}_dbclusters.hdf5"
            )


def _hdbscan(
    files: str,
    min_cluster: int,
    min_samples: int,
) -> None:
    """Run HDBSCAN clustering on localizations in HDF5 files. See
    ``clusterer.hdbscan`` for details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, clusterer

        for path in paths:
            print("Loading {} ...".format(path))
            locs, info = io.load_locs(path)
            locs, info = clusterer.hdbscan(
                locs,
                info,
                min_cluster_size=min_cluster,
                min_samples=min_samples,
            )
            clusters, clusters_info = clusterer.find_cluster_centers(
                locs, info, progress="console"
            )
            base, ext = os.path.splitext(path)
            io.save_locs(base + "_hdbscan.hdf5", locs, info)
            io.save_locs(base + "_hdbclusters.hdf5", clusters, clusters_info)
            print(
                "Clustering executed. Results are saved in: \n"
                f"{base}_hdbscan.hdf5\n"
                f"{base}_hdbclusters.hdf5"
            )


def _smlm_clusterer(
    files: str,
    radius: float,
    min_locs: int,
    basic_fa: bool = False,
    radius_z: float | None = None,
) -> None:
    """Run SMLM clustering on localizations in HDF5 files. See
    ``clusterer.cluster`` for details. The radii are in nm."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, clusterer

        params = {
            "radius_xy": radius,
            "radius_z": radius_z,
            "min_locs": min_locs,
            "frame_analysis": basic_fa,
        }
        for path in paths:
            print("Loading {} ...".format(path))
            locs, info = io.load_locs(path)
            locs, info = clusterer.cluster(
                locs, info, **params, progress="console"
            )
            clusters, clusters_info = clusterer.find_cluster_centers(
                locs, info, progress="console"
            )
            base, ext = os.path.splitext(path)
            io.save_locs(base + "_clusters.hdf5", locs, info)
            io.save_locs(
                base + "_cluster_centers.hdf5", clusters, clusters_info
            )
            print(
                "Clustering executed. Results are saved in: \n"
                f"{base}_clusters.hdf5\n"
                f"{base}_cluster_centers.hdf5"
            )


def _nneighbor(files: str) -> None:
    """Calculate the minimum distance to the nearest neighbor for each
    localization in the given HDF5 files. The results are saved in a
    text file with the same name as the input file, but with a
    `_minval.txt` suffix. The distances are calculated using the
    Euclidean distance metric."""
    import glob
    import numpy as np
    from scipy.spatial import KDTree
    from . import io

    paths = glob.glob(files)
    if paths:
        for path in paths:
            print("Loading {} ...".format(path))
            clusters, _ = io.load_locs(path)  # nm, also for old files
            points = clusters[["x", "y"]].to_numpy()
            tree = KDTree(points)
            minvals, _ = tree.query(points, k=2)
            minvals = minvals[:, 1]
            base, ext = os.path.splitext(path)
            out_path = base + "_minval.txt"
            np.savetxt(out_path, minvals, newline="\r\n")
            print("Saved file to: {}".format(out_path))


def _cbc(
    files: list[str],
    r_max: float,
    steps: int,
    all_pairs: bool,
    use_2d: bool,
    names: list[str] | None = None,
    undefined: float | Literal["remove"] = 0.0,
) -> None:
    """Coordinate-based colocalization (CBC) between localization
    files. The first file is scored against all others or, with
    ``all_pairs``, every file against every other file. ``undefined``
    is the CBC of localizations without neighbors, or "remove" to
    remove them. See ``postprocess.cbc`` and
    ``postprocess.cbc_all_pairs`` for details."""
    import warnings
    from . import io, postprocess

    paths = _glob_paths(
        files, 2, "CBC requires at least two localization files."
    )
    names = _channel_names(names, paths)
    columns = _cbc_columns(names)
    locs_infos = [io.load_locs(path) for path in paths]
    _warn_if_unlinked(paths, [locs for locs, _ in locs_infos])
    _print_cbc_plan(paths, columns, all_pairs)
    kwargs = dict(
        r_max=r_max,
        steps=steps,
        use_z=False if use_2d else None,
        undefined=0.0 if undefined == "remove" else undefined,
        remove_undefined=undefined == "remove",
        progress="console",
    )
    all_locs = [locs for locs, _ in locs_infos]
    all_infos = [info for _, info in locs_infos]
    with warnings.catch_warnings():  # unlinked locs reported above
        warnings.filterwarnings("ignore", message=".*not linked")
        if all_pairs:
            out_locs, out_infos = postprocess.cbc_all_pairs(
                all_locs, all_infos, names=names, **kwargs
            )
        else:
            locs, info = postprocess.cbc(
                all_locs[0],
                all_infos[0],
                partner_locs=all_locs[1:],
                partner_names=names[1:],
                **kwargs,
            )
            out_locs, out_infos = [locs], [info]
    for i, (locs, info) in enumerate(zip(out_locs, out_infos)):
        base, ext = os.path.splitext(paths[i])
        io.save_locs(base + "_cbc.hdf5", locs, info)
        print(f"Saved {base}_cbc.hdf5")


def _cbc_columns(names: list[str]) -> list[str]:
    """Return the CBC column of each name; exit if a name is invalid or
    two names give the same column."""
    from . import postprocess

    try:
        columns = [postprocess._cbc_column(name) for name in names]
    except ValueError as e:
        raise SystemExit(str(e))
    if len(set(columns)) < len(columns):
        raise SystemExit(
            f"The names give the same CBC column ({', '.join(columns)}); "
            "please give distinct names with --names."
        )
    return columns


def _warn_if_unlinked(paths: list[str], all_locs: list[pd.DataFrame]) -> None:
    """Print a warning listing the files whose localizations are not
    linked (no 'len' column)."""
    unlinked = [p for p, locs in zip(paths, all_locs) if "len" not in locs]
    if not unlinked:
        return
    print(
        "WARNING: these localizations are not linked (no 'len' column),"
        " so each binding event is counted several times in the CBC:"
    )
    for path in unlinked:
        print(f"  {path}")
    print("Consider linking them first with 'picasso link'.")


def _print_cbc_plan(
    paths: list[str], columns: list[str], all_pairs: bool
) -> None:
    """Print which file is scored against which, and the CBC column
    each partner gives."""
    scored = range(len(paths)) if all_pairs else [0]
    for i in scored:
        print(f"CBC of {paths[i]} against:")
        for j in range(len(paths)):
            if j != i:
                print(f"  {paths[j]} -> column {columns[j]}")


def _parse_bool(value: str) -> bool:
    """Parse a yes/no command line argument: yes/no, true/false, on/off
    or 1/0, in any case."""
    answer = value.strip().lower()
    if answer in ("yes", "true", "on", "1", "y"):
        return True
    if answer in ("no", "false", "off", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected yes or no (or true/false, 1/0), got {value!r}"
    )


def _cbc_undefined(value: str) -> float | Literal["remove"]:
    """Parse the ``-u`` option of ``picasso cbc``: a CBC value between
    -1 and 1, or "remove"."""
    if value.strip().lower() == "remove":
        return "remove"
    try:
        number = float(value)
    except ValueError:
        number = None
    if number is None or not -1 <= number <= 1:
        raise argparse.ArgumentTypeError(
            f"expected a number between -1 and 1 or 'remove', got {value!r}"
        )
    return number


def _pair_statistic(
    statistic: Literal["ripley_h", "rdf"],
    files: list[str],
    picks_path: str | None,
    r_max: float,
    steps: int,
    n_simulations: int,
    edge_correction: bool,
    use_3d: bool = False,
    z_range: tuple[float, float] | None = None,
    observed_z: bool = False,
    partner_path: str | None = None,
    cross_null: Literal["shift", "labels"] = "shift",
) -> None:
    """Ripley's H function (``statistic="ripley_h"``) or the radial
    distribution function (``"rdf"``) of localizations, in the picks
    of ``picks_path`` or in the whole field of view, in 2D or, with
    ``use_3d``, in 3D over ``z_range`` (nm; None for the z range of
    each region's localizations) with the simulated z uniform or, with
    ``observed_z``, shuffled from the localizations. With
    ``partner_path``, the cross function of each file with the
    localizations in that file, simulated by ``cross_null``. Saves the
    functions (``<file>_<statistic>[_cross].csv``), the summary of
    each region (``..._regions.csv``) and a plot (``.png``) next to
    each file. See ``postprocess.ripley_h`` and ``postprocess.rdf``
    for details."""
    import warnings
    import numpy as np
    from . import io, postprocess

    title, compute, plot, tables, key = {
        "ripley_h": (
            "Ripley's H function",
            postprocess.ripley_h,
            postprocess.plot_ripley_h,
            postprocess._ripley_h_tables,
            "h",
        ),
        "rdf": (
            "Radial distribution function",
            postprocess.rdf,
            postprocess.plot_rdf,
            postprocess._rdf_tables,
            "g",
        ),
    }[statistic]
    paths = _glob_paths(files, 0)
    partner_locs = None
    if partner_path is not None:
        partner_locs, _ = io.load_locs(partner_path)
        _warn_if_not_molecules(partner_path, partner_locs)
        # a file matching both is not paired with itself
        paths = [
            p
            for p in paths
            if os.path.abspath(p) != os.path.abspath(partner_path)
        ]
    if not paths:
        raise SystemExit("No localization files found.")
    for path in paths:
        locs, info = io.load_locs(path)
        _warn_if_not_molecules(path, locs)
        picks, shape, size = _load_picks_for(picks_path, info)
        if partner_locs is None:
            print(f"{title} of {path}:")
        else:
            print(f"{title} of {path} and {partner_path}:")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = compute(
                    locs,
                    info,
                    picks=picks,
                    pick_shape=shape,
                    pick_size=size,
                    r_max=r_max,
                    steps=steps,
                    edge_correction=edge_correction,
                    n_simulations=n_simulations,
                    use_z=use_3d,
                    z_range=z_range,
                    simulated_z="observed" if observed_z else "uniform",
                    partner_locs=partner_locs,
                    cross_null=cross_null,
                    progress="console",
                )
        except ValueError as e:
            raise SystemExit(f"{path}: {e}")
        _print_warnings(caught, skip="not molecules")  # see above
        n_used = int((~np.isnan(result[key][:, 0])).sum())
        summary = f"  {n_used} of {len(result['n_locs'])} regions used"
        if statistic == "ripley_h":
            k_max = int(np.argmax(result["h_mean"]))
            summary += (
                f", maximum of the mean H = {result['h_mean'][k_max]:.1f}"
                f" nm at r = {result['r'][k_max]:.0f} nm"
            )
        print(summary)
        base = os.path.splitext(path)[0] + "_" + statistic
        if partner_locs is not None:
            base += "_cross"
        curves, regions = tables(result)
        curves.to_csv(base + ".csv", index=False)
        regions.to_csv(base + "_regions.csv", index=False)
        plot(result).savefig(base + ".png", dpi=150)
        for suffix in (".csv", "_regions.csv", ".png"):
            print(f"  saved {base + suffix}")


def _nnd(
    files: list[str],
    names: list[str] | None,
    picks_path: str | None,
    r_max: float,
    steps: int,
    n_simulations: int,
    use_3d: bool = False,
    z_range: tuple[float, float] | None = None,
    observed_z: bool = False,
    cross_null: Literal["shift", "labels"] = "shift",
) -> None:
    """Distributions of the first nearest neighbor distances within and
    between the channels in ``files`` (in the picks of ``picks_path``
    or the whole field of view), with random simulations. Saves the
    histograms (``<first file>_nnd.csv``), the summary of each pair of
    channels (``<first file>_nnd_summary.csv``) and a plot
    (``<first file>_nnd.png``). See ``postprocess.nnd_distributions``
    for details."""
    import warnings
    from . import io, postprocess

    paths = _glob_paths(files)
    names = _channel_names(names, paths)
    all_locs, infos = zip(*[io.load_locs(path) for path in paths])
    for path, locs in zip(paths, all_locs):
        _warn_if_not_molecules(path, locs)
    picks, shape, size = _load_picks_for(picks_path, infos[0])
    print("Nearest neighbor distances of:")
    for path, name in zip(paths, names):
        print(f"  {path} ({name})")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = postprocess.nnd_distributions(
                list(all_locs),
                infos[0],
                picks=picks,
                pick_shape=shape,
                pick_size=size,
                names=names,
                r_max=r_max,
                steps=steps,
                n_simulations=n_simulations,
                use_z=use_3d,
                z_range=z_range,
                simulated_z="observed" if observed_z else "uniform",
                cross_null=cross_null,
                progress="console",
            )
    except ValueError as e:
        raise SystemExit(str(e))
    _print_warnings(caught, skip="not molecules")  # see above
    curves, summary = postprocess._nnd_tables(result)
    print(summary.to_string(index=False))
    base = os.path.splitext(paths[0])[0] + "_nnd"
    curves.to_csv(base + ".csv", index=False)
    summary.to_csv(base + "_summary.csv", index=False)
    figure = postprocess.plot_nnd_distributions(result)
    figure.set_size_inches(3 + 3 * len(paths), 2.5 + 2.5 * len(paths))
    figure.savefig(base + ".png", dpi=150)
    for suffix in (".csv", "_summary.csv", ".png"):
        print(f"saved {base + suffix}")


def _knn_density(
    files: list[str],
    picks_path: str | None,
    k_max: int,
    k_min: int,
    r_min: float,
    n_simulations: int,
    edge_correction: bool,
    use_3d: bool = False,
    z_range: tuple[float, float] | None = None,
) -> None:
    """Density of the molecules in each of ``files`` fitted to the
    distances to their ``k_max`` nearest neighbors, in the picks of
    ``picks_path`` or in the whole field of view, in 2D or, with
    ``use_3d``, in 3D over ``z_range`` (nm; None for the z range of
    each region's molecules), with p-values from ``n_simulations``
    random simulations. Prints the densities and saves the
    distributions (``<file>_knn_density.csv``), the densities
    (``..._densities.csv``) and a plot (``.png``) next to each file.
    See ``postprocess.knn_density`` for details."""
    import warnings
    from glob import glob
    from itertools import chain
    from . import io, lib, postprocess

    paths = list(chain(*[sorted(glob(_)) for _ in files]))
    if not paths:
        raise SystemExit("No localization files found.")
    for path in paths:
        locs, info = io.load_locs(path)
        _warn_if_not_molecules(path, locs)
        picks, shape, size = None, None, None
        if picks_path is not None:
            # the pixel size converts picks saved before Picasso 1.0
            picks, shape, size = io.load_picks(
                picks_path, lib.get_from_metadata(info, "Pixelsize")
            )
        print(f"Density from the k nearest neighbors of {path}:")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = postprocess.knn_density(
                    locs,
                    info,
                    picks=picks,
                    pick_shape=shape,
                    pick_size=size,
                    k_max=k_max,
                    k_min=k_min,
                    r_min=r_min,
                    edge_correction=edge_correction,
                    n_simulations=n_simulations,
                    use_z=use_3d,
                    z_range=z_range,
                    progress="console",
                )
        except ValueError as e:
            raise SystemExit(f"{path}: {e}")
        for warning in caught:
            if "not molecules" not in str(warning.message):  # see above
                print(f"WARNING: {warning.message}")
        curves, densities = postprocess._knn_density_tables(result)
        print(
            densities.to_string(
                index=False, float_format=lambda value: f"{value:.4g}"
            )
        )
        base = os.path.splitext(path)[0] + "_knn_density"
        curves.to_csv(base + ".csv", index=False)
        densities.to_csv(base + "_densities.csv", index=False)
        figure = postprocess.plot_knn_density(result)
        figure.set_size_inches(11, 4.5)
        figure.savefig(base + ".png", dpi=150)
        for suffix in (".csv", "_densities.csv", ".png"):
            print(f"  saved {base + suffix}")


def _decorrelation(
    files: list[str],
    pixel_sizes: list[float] | None,
    n_radii: int,
    n_filters: int,
    max_image_size: int,
) -> None:
    """Decorrelation resolution of the localizations in each of
    ``files`` over the whole field of view, at ``pixel_sizes`` (nm;
    None scans the default pixel sizes). Prints the resolution at each
    pixel size and saves them (``<file>_decorrelation.csv``), the
    decorrelation functions (``..._curves.csv``) and a plot (``.png``)
    next to each file. See ``postprocess.decorrelation`` for
    details."""
    from glob import glob
    from itertools import chain
    import numpy as np
    from . import io, postprocess

    paths = list(chain(*[sorted(glob(_)) for _ in files]))
    if not paths:
        raise SystemExit("No localization files found.")
    for path in paths:
        locs, info = io.load_locs(path)
        print(f"Decorrelation resolution of {path}:")
        try:
            result = postprocess.decorrelation(
                locs,
                info,
                pixel_size=pixel_sizes,
                n_radii=n_radii,
                n_filters=n_filters,
                max_image_size=max_image_size or None,
                progress="console",
            )
        except ValueError as e:
            raise SystemExit(f"{path}: {e}")
        sizes, curves = postprocess._decorrelation_tables(result)
        print(
            sizes.to_string(
                index=False, float_format=lambda value: f"{value:.4g}"
            )
        )
        skipped = result["skipped_pixel_sizes"]
        if len(skipped):
            print(
                "  skipped pixel sizes (image too large or small): "
                + ", ".join(f"{size:.3g}" for size in skipped)
                + " nm"
            )
        if np.isfinite(result["resolution"]):
            print(
                f"  resolution {result['resolution']:.2f} nm at"
                f" {result['pixel_size']:.3g} nm pixels"
            )
            if result["pixel_size_limited"]:
                print(
                    "WARNING: the resolution is close to twice the smallest"
                    " pixel size rendered, so it may be limited by the pixel"
                    " size; render smaller pixels with -px or a larger"
                    " --max-image-size."
                )
        else:
            print("  no resolution: no decorrelation function has a peak")
        print(
            f"  {result['n_locs']:,} localizations,"
            f" {result['density']:,.0f} per um^2 of sample"
        )
        base = os.path.splitext(path)[0] + "_decorrelation"
        sizes.to_csv(base + ".csv", index=False)
        curves.to_csv(base + "_curves.csv", index=False)
        figure = postprocess.plot_decorrelation(result)
        figure.set_size_inches(11, 4.5)
        figure.savefig(base + ".png", dpi=150)
        for suffix in (".csv", "_curves.csv", ".png"):
            print(f"  saved {base + suffix}")


def _dye_performance_cli(
    files: list[str],
    picks_path: str | None,
    max_dark_time: int,
    cutoff: float,
    fraction: float,
) -> None:
    """Dye performance of the docking sites in each of ``files`` (Steen
    et al., 2024): the docking sites are the picks of ``picks_path`` or,
    without picks, the groups of the localizations (e.g., picked
    localizations saved in Render). Prints the SBR, the localization
    drop and the destroyed docking sites and saves the summary
    (``<file>_dye_performance.csv``), the values of each site
    (``..._sites.csv``) and a plot (``.png``) next to each file. See
    ``postprocess.dye_performance`` for details."""
    import pandas as pd
    from . import io, postprocess

    if max_dark_time < 0:
        raise SystemExit("--max-dark-time must not be negative.")
    if not cutoff > 0:
        raise SystemExit("--cutoff must be positive.")
    if not 0 < fraction <= 0.5:
        raise SystemExit("--fraction must be in (0, 0.5].")
    paths = _glob_paths(files)
    for path in paths:
        locs, info = io.load_locs(path)
        sites, picked = _docking_sites(locs, info, path, picks_path)
        print(f"Dye performance of {path} ({len(picked):,} docking sites):")
        result = postprocess.dye_performance(
            picked,
            info,
            max_dark_time=max_dark_time,
            cutoff=cutoff,
            fraction=fraction,
            progress="console",
        )
        summary = {"n_sites": len(picked)}
        summary |= _dye_performance_summary(result, cutoff, fraction)
        summary |= {
            "max_dark_time": max_dark_time,
            "cutoff": cutoff,
            "fraction": fraction,
        }
        base = os.path.splitext(path)[0] + "_dye_performance"
        pd.DataFrame([summary]).to_csv(base + ".csv", index=False)
        _dye_performance_sites(sites, picked, result["damage"]).to_csv(
            base + "_sites.csv", index=False
        )
        saved = [".csv", "_sites.csv"]
        if result["locs"] is not None:
            _dye_performance_figure(result, info, fraction).savefig(
                base + ".png", dpi=150
            )
            saved.append(".png")
        for suffix in saved:
            print(f"  saved {base + suffix}")


def _docking_sites(
    locs: pd.DataFrame,
    info: list[dict],
    path: str,
    picks_path: str | None,
) -> tuple:
    """Return the docking site labels and the localizations of each
    docking site (list of DataFrames) in ``locs``: the picks of
    ``picks_path`` or, without picks, the groups."""
    import numpy as np
    from . import postprocess

    if picks_path is not None:
        picks, shape, size = _load_picks_for(picks_path, info)
        picked = postprocess.picked_locs(
            locs,
            info,
            picks=picks,
            pick_shape=shape,
            pick_size=size,
            progress="console",
        )
        return np.arange(len(picked)), list(picked)
    if "group" in locs.columns:
        sites, picked = zip(*locs.groupby("group", sort=True))
        return np.asarray(sites), list(picked)
    raise SystemExit(
        f"{path}: each docking site must be a pick (-p) or a group of"
        " the localizations (e.g., picked localizations saved in"
        " Render)."
    )


def _dye_performance_summary(
    result: dict, cutoff: float, fraction: float
) -> dict:
    """Print the SBR, the localization drop and the destroyed docking
    sites of a ``postprocess.dye_performance`` result and return them
    as a summary row."""
    import numpy as np

    errors = result["errors"]
    sbr, damage = result["sbr"], result["damage"]
    drop = result["localization_drop"]
    summary = {}
    if sbr is not None and np.isfinite(sbr).any():
        n_sbr = int(np.isfinite(sbr).sum())
        summary |= {
            "sbr_mean": np.nanmean(sbr),
            "sbr_std": np.nanstd(sbr),
            "n_sbr_localizations": n_sbr,
        }
        print(
            f"  SBR {summary['sbr_mean']:.2f} +/- {summary['sbr_std']:.2f}"
            f" ({n_sbr:,} localizations in the centers of binding"
            " events)"
        )
    else:
        reason = errors.get(
            "sbr",
            "no localization in the center of a binding event of at"
            " least three frames",
        )
        print(f"  SBR not available: {reason}")
    if drop is not None:
        summary["localization_drop_percent"] = 100 * drop
        print(
            f"  localization drop {100 * drop:.1f}% (first vs. last"
            f" {100 * fraction:g}% of the frames)"
        )
    else:
        print(
            "  localization drop not available:"
            f" {errors['localization_drop']}"
        )
    if damage is not None:
        summary |= {
            "n_sites_with_r": damage["n_sites"],
            "mu": damage["mu"],
            "r_cutoff": damage["r_cutoff"],
            "sites_above_cutoff_percent": (
                100 * damage["fraction_above_cutoff"]
            ),
            "destroyed_sites_percent": 100 * damage["fraction_destroyed"],
        }
        print(
            "  destroyed docking sites"
            f" {100 * damage['fraction_destroyed']:.1f}% (r >"
            f" {damage['r_cutoff']:.2f} = {cutoff:g} mu, mu ="
            f" {damage['mu']:.2f}; {damage['n_sites']:,} sites with a"
            " mean dark time)"
        )
    else:
        print(f"  destroyed docking sites not available: {errors['damage']}")
    return summary


def _dye_performance_sites(
    sites: np.ndarray, picked: list[pd.DataFrame], damage: dict | None
) -> pd.DataFrame:
    """Table of the docking sites: the number of localizations and,
    with ``damage``, the end time, the mean dark time and r."""
    site_table = pd.DataFrame(
        {
            "site": sites,
            "n_locs": [len(site_locs) for site_locs in picked],
        }
    )
    if damage is not None:
        site_table["t_end_frames"] = damage["t_end"]
        site_table["dark_time_frames"] = damage["dark"]
        site_table["r"] = damage["r"]
    return site_table


def _dye_performance_figure(
    result: dict, info: list[dict], fraction: float
) -> Figure:
    """Figure of the docking site damage and the localization rate of a
    ``postprocess.dye_performance`` result (axes without a result are
    hidden)."""
    import matplotlib.pyplot as plt
    from . import postprocess

    figure = plt.Figure(constrained_layout=True, figsize=(11, 4.5))
    ax_damage, ax_rate = figure.subplots(1, 2)
    if result["damage"] is not None:
        postprocess.plot_docking_site_damage(result["damage"], ax=ax_damage)
    else:
        ax_damage.set_axis_off()
    if result["localization_drop"] is not None:
        postprocess.plot_localization_rate(
            result["locs"], info, fraction=fraction, ax=ax_rate
        )
    else:
        ax_rate.set_axis_off()
    return figure


def _pick_similar_cli(
    files: list[str],
    picks_path: str | None = None,
    pick_size: float | None = None,
    std_range: float = 2.0,
    n_locs_range: list[float] | None = None,
    rmsd_range: list[float] | None = None,
    classes: bool = False,
    min_n_locs: int | None = None,
    save_candidates: bool = False,
) -> list[str]:
    """Pick similar structures in localization files (``picasso
    pick-similar``), saving one picks file per file (per class with
    ``classes``).

    The structures are similar to the seed picks of ``picks_path`` (see
    ``postprocess.pick_similar``), or are the candidate circular picks
    (``postprocess.pick_similar_candidates``) within ``n_locs_range``
    and/or ``rmsd_range``, or within the ranges of each class of
    ``postprocess.structure_classes``, as Render's Tools > Pick by
    localizations and RMSD finds them: candidates with at least
    ``min_n_locs`` localizations, by default 1% of the frames, and at
    least the lower bound of ``n_locs_range``.

    Returns
    -------
    saved : list of str
        Paths of the picks files written.
    """
    from . import io, postprocess

    by_ranges = n_locs_range is not None or rmsd_range is not None
    if (picks_path is not None) + by_ranges + bool(classes) != 1:
        raise SystemExit(
            "Give exactly one of --picks, --n-locs-range/--rmsd-range and "
            "--classes."
        )
    if picks_path is None and pick_size is None:
        raise SystemExit(
            "--pick-size (the diameter in nm) is needed without --picks."
        )
    paths = _glob_paths(files)
    saved = []
    for path in paths:
        locs, info = io.load_locs(path)
        base = os.path.splitext(path)[0] + "_pick_similar"
        if picks_path is not None:
            saved.append(
                _pick_similar_seeded(
                    locs, info, path, base, picks_path, std_range
                )
            )
            continue
        n_min = _candidate_min_n_locs(info, min_n_locs, n_locs_range)
        print(
            f"{path}: candidate picks of {pick_size:g} nm with at least "
            f"{n_min} localizations"
        )
        candidates = postprocess.pick_similar_candidates(
            locs,
            info,
            pick_size=pick_size,
            min_n_locs=n_min,
            progress="console",
        )
        print(f"{len(candidates)} candidate picks")
        if classes:
            candidates, ranges = _structure_class_ranges(
                candidates, path, base
            )
        else:
            ranges = [(n_locs_range, rmsd_range, base + ".yaml")]
        for n_range, r_range, out in ranges:
            keep = postprocess._in_range(
                candidates["n_locs"], n_range
            ) & postprocess._in_range(candidates["rmsd"], r_range)
            picks = list(zip(candidates["x"][keep], candidates["y"][keep]))
            io.save_picks(out, picks, "Circle", pick_size)
            print(f"saved {len(picks)} picks to {out}")
            saved.append(out)
        if save_candidates:
            out = base + "_candidates.csv"
            candidates.to_csv(out, index=False)
            print(f"saved {out}")
    return saved


def _candidate_min_n_locs(
    info: list[dict],
    min_n_locs: int | None,
    n_locs_range: list[float] | None,
) -> int:
    """Minimum number of localizations of a candidate pick:
    ``min_n_locs`` or, if None, 1% of the frames (at least 2), and at
    least the lower bound of ``n_locs_range``."""
    from . import lib, postprocess

    if min_n_locs is None:
        n_frames = lib.get_from_metadata(info, "Frames", 0)
        n_min = max(2, int(0.01 * n_frames))
    else:
        n_min = min_n_locs
    if n_locs_range is not None:
        # smaller candidates must not exclude the structures they
        # overlap, as in postprocess.pick_similar
        n_min = max(n_min, postprocess._range_min_n_locs(n_locs_range))
    return n_min


def _structure_class_ranges(
    candidates: pd.DataFrame, path: str, base: str
) -> tuple[pd.DataFrame, list[tuple]]:
    """Classify the candidate picks with
    ``postprocess.structure_classes``, print the classes and return
    the candidates with a "class" column and the (n_locs range, RMSD
    range, output path) of each class."""
    from . import postprocess

    if len(candidates) < 2:
        raise SystemExit(f"{path}: too few candidate picks for classes.")
    labels, table = postprocess.structure_classes(candidates)
    candidates = candidates.assign(**{"class": labels + 1})
    if not len(table):
        print("No classes found.")
    else:
        table.index = table.index + 1
        table.index.name = "class"
        print(table.round(1).to_string())
    ranges = [
        (
            (row["n_locs_min"], row["n_locs_max"]),
            (row["rmsd_min"], row["rmsd_max"]),
            f"{base}_class{index}.yaml",
        )
        for index, row in table.iterrows()
    ]
    return candidates, ranges


def _pick_similar_seeded(
    locs: pd.DataFrame,
    info: list[dict],
    path: str,
    base: str,
    picks_path: str,
    std_range: float,
) -> str:
    """``picasso pick-similar --picks``: the seed picks and the picks
    similar to them, saved to ``base``.yaml; the new picks take the
    median size of the seed picks, which keep theirs."""
    import numpy as np

    from . import io, lib, postprocess

    seeds, shape, size = io.load_picks(
        picks_path, lib.get_from_metadata(info, "Pixelsize")
    )
    try:
        picks = postprocess.pick_similar(
            locs,
            info,
            picks=seeds,
            pick_shape=shape,
            pick_size=size,
            std_range=std_range,
        )
    except (ValueError, AssertionError) as e:
        raise SystemExit(f"{path}: {e}")
    shapes, sizes = lib.pick_shapes_and_sizes(seeds, shape, size)
    n_new = len(picks) - len(seeds)
    if shapes and shapes[0] not in lib.PICK_SHAPES_WITHOUT_SIZE:
        sizes = list(sizes) + [float(np.median(sizes))] * n_new
    else:
        sizes = list(sizes) + [None] * n_new
    out = base + ".yaml"
    io.save_picks(out, picks, shapes[0] if shapes else "Circle", sizes)
    print(
        f"{path}: {n_new} picks similar to the {len(seeds)} of "
        f"{picks_path}; saved all {len(picks)} to {out}"
    )
    return out


def _glob_paths(
    files: list[str],
    min_files: int = 1,
    message: str = "No localization files found.",
) -> list[str]:
    """Expand the unix style patterns in ``files`` (each sorted) and
    exit with ``message`` if fewer than ``min_files`` paths match."""
    from glob import glob
    from itertools import chain

    paths = list(chain(*[sorted(glob(_)) for _ in files]))
    if len(paths) < min_files:
        raise SystemExit(message)
    return paths


def _channel_names(names: list[str] | None, paths: list[str]) -> list[str]:
    """Return ``names`` or, if None, the base names of ``paths``; exit
    if the numbers of names and paths differ."""
    if names is None:
        return [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(names) != len(paths):
        raise SystemExit(
            f"{len(names)} names given for {len(paths)} files:\n"
            + "\n".join(f"  {p}" for p in paths)
        )
    return names


def _load_picks_for(picks_path: str | None, info: list[dict]) -> tuple:
    """Load the picks of ``picks_path`` as (picks, shape, size), or
    (None, None, None) without a path. The pixel size in ``info``
    converts picks saved before Picasso 1.0."""
    from . import io, lib

    if picks_path is None:
        return None, None, None
    return io.load_picks(picks_path, lib.get_from_metadata(info, "Pixelsize"))


def _print_warnings(caught: list, skip: str | None = None) -> None:
    """Print the caught warnings, except those whose message contains
    ``skip``."""
    for warning in caught:
        if skip is None or skip not in str(warning.message):
            print(f"WARNING: {warning.message}")


def _warn_if_not_molecules(path: str, locs: pd.DataFrame) -> None:
    """Print a warning if the localizations in ``path`` are not
    molecules, see ``postprocess._are_molecules``."""
    from . import postprocess

    if not postprocess._are_molecules(locs):
        print(
            f"WARNING: {path} does not contain molecules (no"
            " 'n_events' or 'std_frame' column), so each molecule is"
            " counted as many times as it was localized, which appears"
            " as clustering at short distances. Consider finding the"
            " molecules first, e.g., with 'picasso g5m' or as the"
            " cluster centers saved by 'picasso dbscan', 'hdbscan' or"
            " 'smlm_cluster'."
        )


def _add_pair_statistic_arguments(
    parser: argparse.ArgumentParser, steps: int
) -> None:
    """Add the arguments shared by ``picasso ripley_h`` and ``picasso
    rdf`` to ``parser``, with ``steps`` radius steps by default."""
    parser.add_argument(
        "files",
        nargs="+",
        help="hdf5 localization files (unix style path patterns)",
    )
    parser.add_argument(
        "-p",
        "--picks",
        default=None,
        help=(
            "picks (.yaml) saved in Render; the regions of interest in"
            " which the function is computed (default: the whole field"
            " of view)"
        ),
    )
    parser.add_argument(
        "-r",
        "--r-max",
        type=float,
        default=500.0,
        help="maximum radius in nm (default=500)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=steps,
        help=f"number of radius steps up to the maximum radius"
        f" (default={steps})",
    )
    parser.add_argument(
        "-n",
        "--simulations",
        type=int,
        default=99,
        help=(
            "number of simulations of complete spatial randomness for"
            " the envelope, 0 to skip (default=99)"
        ),
    )
    parser.add_argument(
        "--no-edge-correction",
        dest="edge_correction",
        action="store_false",
        help="do not apply Ripley's isotropic edge correction",
    )
    parser.add_argument(
        "--3d",
        dest="use_3d",
        action="store_true",
        help="compute the function in 3D (x, y and z), requires z",
    )
    parser.add_argument(
        "--z-range",
        nargs=2,
        type=float,
        metavar=("ZMIN", "ZMAX"),
        default=None,
        help=(
            "z range in nm over which the regions extend in 3D (default:"
            " the z range of the localizations in each region)"
        ),
    )
    parser.add_argument(
        "--observed-z",
        action="store_true",
        help=(
            "in 3D, shuffle the observed z values for the simulated"
            " points instead of spreading them uniformly over the z range"
        ),
    )
    parser.add_argument(
        "--partner",
        default=None,
        help=(
            "hdf5 localization file of a second channel (aligned); the"
            " cross function of each file with it is computed"
        ),
    )
    parser.add_argument(
        "--cross-null",
        choices=["shift", "labels"],
        default="shift",
        help=(
            "with --partner, how the envelope is simulated: random shifts"
            " of the partner (independent channels; rectangular or"
            " circular regions only) or random labels (both channels"
            " follow the same distribution) (default=shift)"
        ),
    )


def _spt_paths(files: list[str]) -> list[str]:
    """Paths matching the unix style patterns ``files``."""
    from glob import glob
    from itertools import chain

    paths = list(chain(*[sorted(glob(_)) for _ in files]))
    if not paths:
        raise SystemExit("No localization files found.")
    return paths


def _track(
    files: list[str],
    max_jump: float,
    max_gap: int,
    min_length: int,
    frame_time: float | None,
) -> None:
    """Link localizations into tracks (single-particle tracking) and
    save them as ``<base>_tracks.hdf5``. ``max_jump`` is in nm; see
    ``spt.link_tracks`` for details."""
    from . import io, spt

    for path in _spt_paths(files):
        locs, info = io.load_locs(path)
        if "len" in locs.columns:
            print(
                f"Skipped {path}: linked localizations (binding events) "
                "cannot be tracked."
            )
            continue
        try:
            tracked, new_info = spt.link_tracks(
                locs,
                info,
                max_jump=max_jump,
                max_gap=max_gap,
                min_length=min_length,
                frame_time=frame_time,
            )
        except ValueError as e:
            raise SystemExit(str(e))
        out_path = os.path.splitext(path)[0] + "_tracks.hdf5"
        io.save_locs(out_path, tracked, new_info)
        print(
            f"{path}: {new_info[-1]['Number of tracks']} tracks with "
            f"{len(tracked)} localizations, saved as {out_path}"
        )


def _diffusion(
    files: list[str],
    frame_time: float | None,
    max_lag: int,
    n_fit: int,
    populations: list[int],
    sigma: float | Literal["precision"],
    truncate: bool,
    plot: bool,
) -> None:
    """Diffusion analysis of tracked localizations: MSD, jump distances
    and a summary of every track, see ``spt.analyze_tracks``. Saves the
    results next to each file (``spt.save_analysis``) and, with
    ``plot``, PNG images of the plots."""
    from . import io, spt

    for path in _spt_paths(files):
        locs, info = io.load_locs(path)
        if not spt.is_tracked(locs, info):
            print(
                f"Skipped {path}: the localizations are not tracked (see "
                "'picasso track')."
            )
            continue
        try:
            result = spt.analyze_tracks(
                locs,
                info,
                frame_time=frame_time,
                max_lag=max_lag,
                n_fit=n_fit,
                sigma=sigma,
                n_populations=populations,
                truncate=truncate,
                progress="console",
            )
        except ValueError as e:
            raise SystemExit(f"{path}: {e}")
        base = os.path.splitext(path)[0]
        written = spt.save_analysis(base, result)
        print(
            f"{path}: {result['n_tracks']} tracks, "
            f"{len(result['jumps'])} jumps between consecutive frames"
        )
        msd_fit = result["msd_fit"]
        print(
            f"  MSD: D = {msd_fit['D']:.4g} um^2/s, apparent localization "
            f"error = {msd_fit['sigma']:.1f} nm"
        )
        for k, fit in result["jd_fits"].items():
            label = f"{k} population{'s' if k > 1 else ''}"
            if fit is None:
                print(f"  {label}: {result['jd_errors'][k]}")
                continue
            values = ", ".join(
                f"D = {d:.4g} um^2/s ({100 * a:.1f}%)"
                for d, a in zip(fit["D"], fit["fractions"])
            )
            print(f"  {label} (BIC {fit['bic']:.1f}): {values}")
        if plot:
            fits = [f for f in result["jd_fits"].values() if f is not None]
            best = min(fits, key=lambda f: f["bic"]) if fits else None
            figures = {
                "_msd.png": spt.plot_msd(result["msd"], msd_fit),
                "_jump_distances.png": spt.plot_jump_distances(
                    result["jumps"], best
                ),
                "_track_lengths.png": spt.plot_track_lengths(result["table"]),
            }
            for suffix, figure in figures.items():
                figure.savefig(base + suffix, dpi=150)
                written.append(base + suffix)
        for written_path in written:
            print(f"  saved {written_path}")


def _spt_sigma(value: str) -> float | Literal["precision"]:
    """Parse the ``-s`` option of ``picasso diffusion``: a localization
    error in nm, or "lp" for the median localization precision."""
    if value.strip().lower() in ("lp", "precision"):
        return "precision"
    try:
        number = float(value)
    except ValueError:
        number = None
    if number is None or not number >= 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number (nm) or 'lp', got {value!r}"
        )
    return number


def _dark(files: str) -> None:
    """Compute dark times for localizations in HDF5 files. See
    ``postprocess.compute_dark_times`` for details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, postprocess

        for path in paths:
            locs, info = io.load_locs(path)
            locs, info = postprocess.compute_dark_times(locs, info)
            base, ext = os.path.splitext(path)
            io.save_locs(base + "_dark.hdf5", locs, info)


def _align(
    files: str,
    display: bool,
    remove_non_overlapping: bool = False,
    method: str = "rcc",
    max_shift: float = 500.0,
) -> None:
    """Align localization files by RCC or by the displacements between
    their localizations (``method="points"``, searching shifts up to
    ``max_shift`` nm), see ``postprocess.align`` for details. If
    ``remove_non_overlapping``, the localizations outside the region
    where all aligned files overlap are removed."""
    from glob import glob
    from itertools import chain
    from .io import load_locs, save_locs
    from .postprocess import align
    from os.path import splitext

    files = list(chain(*[glob(_) for _ in files]))
    print("Aligning files:")
    for f in files:
        print("  " + f)
    locs_infos = [load_locs(_) for _ in files]
    locs = [_[0] for _ in locs_infos]
    infos = [_[1] for _ in locs_infos]
    try:
        aligned_locs, aligned_infos = align(
            locs,
            infos,
            method=method,
            max_shift=max_shift,
            display=display,
            keep_non_overlapping=not remove_non_overlapping,
            paths=files,
        )
    except ValueError as e:
        raise SystemExit(str(e))
    if method == "points":
        print("Shifts (nm):")
        for file, info in zip(files, aligned_infos):
            entry = info[-1]
            print(
                f"  {file}: x {entry['Shift x (nm)']:.2f} +/-"
                f" {entry['Shift x uncertainty (nm)']:.2f}, y"
                f" {entry['Shift y (nm)']:.2f} +/-"
                f" {entry['Shift y uncertainty (nm)']:.2f}"
            )
    for file, locs_, info in zip(files, aligned_locs, aligned_infos):
        base, ext = splitext(file)
        save_locs(base + "_align.hdf5", locs_, info)


def _join(files: list[str], keep_index: bool = True) -> None:
    """Join multiple localization files into one."""
    from . import lib
    from .io import load_locs, save_locs
    from .lib import merge_locs
    from os.path import splitext

    all_locs = merge_locs(
        [load_locs(file)[0] for file in files],
        increment_frames=(not keep_index),
        increment_groups=False,
    )
    join_info = {
        **lib.step_info("Join"),
        "Files": files,
        "Frames kept": keep_index,
    }
    base, ext = splitext(files[0])
    info = load_locs(files[0])[1]
    info.append(join_info)
    if not keep_index:
        info[0]["Frames"] = all_locs["frame"].max() + 1
    save_locs(base + "_join.hdf5", all_locs, info)


def _stitch(
    files: list[str],
    output: str | None = None,
    search_radius: float = 10.0,
    orientation: str = "auto",
    use_2d: bool = False,
    keep_overlaps: bool = False,
    keep_frames: bool = False,
) -> None:
    """Stitch localization files of tiles recorded at different stage
    positions. Files at the same stage position form one tile and are
    different targets (in the order given); the tiles of each target are
    saved as one file. See ``stitch.stitch`` for details."""
    import warnings
    from . import io, stitch

    paths = _glob_paths(files, 2, "Stitching requires at least two files.")
    locs_infos = [io.load_locs(path) for path in paths]
    locs = [locs_ for locs_, _ in locs_infos]
    infos = [info for _, info in locs_infos]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out_locs, out_infos, report = stitch.stitch(
                locs,
                infos,
                orientation=orientation,
                search_radius_um=search_radius,
                align_z=not use_2d,
                cut_seams=not keep_overlaps,
                offset_frames=not keep_frames,
                paths=paths,
                progress="console",
            )
        except ValueError as e:
            raise SystemExit(str(e))
    _print_warnings(caught)
    _print_stitch_report(report, paths)
    for k, (locs_, info) in enumerate(zip(out_locs, out_infos)):
        if output is None:
            base = os.path.splitext(info[-1]["Stitched files"][0])[0]
            path = base + "_stitched.hdf5"
        elif len(out_locs) == 1:
            path = output
        else:
            base, ext = os.path.splitext(output)
            path = f"{base}_target{k + 1}{ext or '.hdf5'}"
        io.save_locs(path, locs_, info)
        print(f"Saved {path}")


def _print_stitch_report(report: dict, paths: list[str]) -> None:
    """Print the tiles, the stage orientation and the overlaps of a
    ``stitch.stitch`` report."""
    from . import stitch

    tiles = report["tiles"]
    print(f"Stitched {tiles.max() + 1} tiles:")
    for t in range(tiles.max() + 1):
        x, y = report["positions"][t]
        print(
            f"  Tile {t + 1} at ({x:.1f}, {y:.1f}) nm, z offset "
            f"{report['z_offsets'][t]:.1f} nm:"
        )
        for i in (i for i, tile in enumerate(tiles) if tile == t):
            print(f"    {paths[i]}")
    if report["orientation"] is not None:
        print(
            f"Stage orientation: {report['orientation']} ("
            + stitch.describe_orientation(report["orientation"])
            + ")"
        )
    print("Overlaps:")
    for pair in report["pairs"]:
        a, b = (t + 1 for t in pair["tiles"])
        if pair["used"]:
            print(
                f"  Tiles {a} and {b}: {100 * pair['fraction']:.0f}% matched,"
                f" residual {pair['residual']:.1f} nm"
            )
        else:
            print(f"  Tiles {a} and {b}: not used ({pair['status']})")


def _groupprops(files: str) -> None:
    """Calculate group properties for localizations in HDF5 files.
    See ``postprocess.groupprops`` for details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from .io import load_locs, save_datasets
        from .postprocess import groupprops
        from os.path import splitext

        for path in paths:
            locs, info = load_locs(path)
            groups = groupprops(locs, progress="console")
            base, ext = splitext(path)
            save_datasets(
                base + "_groupprops.hdf5",
                info,
                locs=locs,
                groups=groups,
            )


def _pair_correlation(files: str, bin_size: float, r_max: float) -> None:
    """Calculate pair-correlation for localizations in HDF5 files. See
    ``postprocess.pair_correlation`` for details."""
    from glob import glob

    paths = glob(files)
    if paths:
        from .io import load_locs
        from .postprocess import pair_correlation
        from matplotlib.pyplot import plot, style, show, xlabel, ylabel, title

        style.use("ggplot")
        for path in paths:
            print("Loading {}...".format(path))
            locs, info = load_locs(path)
            print("Calculating pair-correlation...")
            bins_lower, pc = pair_correlation(
                locs, info, bin_size=bin_size, r_max=r_max
            )
            plot(bins_lower - bin_size / 2, pc)
            xlabel("r (nm)")
            ylabel("pair-correlation (nm^-2)")
            title(
                f"Pair-correlation. Bin size: {bin_size} nm, R max: {r_max} nm"
            )
            show()


def _start_server(network: bool = False) -> None:
    """Start the Streamlit server for the Picasso GUI.

    Parameters
    ----------
    network : bool, optional
        If True, listen on all network interfaces, so that other
        computers can open the server. Default is False, i.e., only this
        computer (``localhost``) can, because the server has no login
        and its watcher runs custom commands.
    """
    import importlib.util
    import os
    import sys

    missing = [
        name
        for name in ("streamlit", "plotly")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        sys.exit(
            f"Picasso Server needs {' and '.join(missing)}, which are not "
            'installed. Install them with: pip install "picassosr[server]"'
        )
    from streamlit.web import cli as stcli

    print("                                          ")
    picasso_logo()
    print("                 server")
    print("                                          ")

    HOME = os.path.expanduser("~")

    ST_PATH = os.path.join(HOME, ".streamlit")

    for folder in [ST_PATH]:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    _this_file = os.path.abspath(__file__)
    _this_dir = os.path.dirname(_this_file)

    file_path = os.path.join(_this_dir, "server", "app.py")

    # Check if streamlit credentials exists
    ST_CREDENTIALS = os.path.join(ST_PATH, "credentials.toml")
    if not os.path.isfile(ST_CREDENTIALS):
        with open(ST_CREDENTIALS, "w") as file:
            file.write("[general]\n")
            file.write('\nemail = ""')

    theme = []

    theme.append("--theme.backgroundColor=#FFFFFF")
    theme.append("--theme.secondaryBackgroundColor=#f0f2f6")
    theme.append("--theme.textColor=#262730")
    theme.append("--theme.font=sans serif")
    theme.append("--theme.primaryColor=#18212b")

    args = [
        "streamlit",
        "run",
        file_path,
        "--global.developmentMode=false",
        "--server.port=8501",
        "--browser.gatherUsageStats=False",
        # no reloading when the source files change, which only helps while
        # editing them (and would need watchdog, or poll the files instead)
        "--server.fileWatcherType=none",
    ]
    if network:
        print(
            "Listening on all network interfaces: anyone who can reach this "
            "computer can use Picasso Server, including the custom commands "
            "of the watcher.\n"
        )
    else:
        args.append("--server.address=localhost")

    # args.extend(theme)

    sys.argv = args

    sys.exit(stcli.main())


def _check_consecutive_tif(filepath: str) -> list[str]:
    """Return only the first file of each consecutive ome.tif or
    NDTiffStack series found in ``filepath``.

    ``load_movie`` detects consecutive files automatically, so passing
    only the first file avoids redundant reconstructions.
    E.g. folder with file.ome.tif, file_1.ome.tif, file_2.ome.tif
    returns only file.ome.tif.
    """
    import os as _os
    import os.path as _ospath
    import re as _re
    from glob import glob

    files = glob(filepath + "/*.tif")
    newlist = [_ospath.abspath(file) for file in files]
    for file in files:
        path = _ospath.abspath(file)
        directory = _ospath.dirname(path)
        if "NDTiffStack" in path:
            base, ext = _ospath.splitext(path)
            base = _re.escape(base)
            pattern = _re.compile(base + r"_(\d*).tif")
        else:
            base, ext = _ospath.splitext(
                _ospath.splitext(path)[0]
            )  # split two extensions as in .ome.tif
            base = _re.escape(base)
            # This matches the basename + an appendix of the file number
            pattern = _re.compile(base + r"_(\d*).ome.tif")
        entries = [_.path for _ in _os.scandir(directory) if _.is_file()]
        matches = [_re.match(pattern, _) for _ in entries]
        matches = [_ for _ in matches if _ is not None]
        datafiles = [_.group(0) for _ in matches]
        if datafiles != []:
            for element in datafiles:
                newlist.remove(element)
    return newlist


def _localize_collect_paths(files: str) -> list[str]:
    """Resolve ``files`` to a list of movie file paths to process: the
    movies of every format Picasso opens (``io.MOVIE_EXTENSIONS``) in a
    folder, or the files matching a pattern."""
    from glob import glob
    from os.path import isdir, join, splitext
    from .io import MOVIE_EXTENSIONS

    if isdir(files):
        print("Analyzing folder")
        # the first file of each movie split over several .tif files
        tif_files = _check_consecutive_tif(files)
        others = sorted(
            path
            for path in glob(join(files, "*"))
            if splitext(path)[1].lower() in MOVIE_EXTENSIONS
            # .tif files are collected series by series above
            and splitext(path)[1] != ".tif"
        )
        paths = tif_files + others
        print("A total of {} files detected".format(len(paths)))
    else:
        paths = glob(files)
    return paths


def _localize_collect_concat_paths(files: str) -> list[str]:
    """Resolve ``files`` to the TIFF movies to concatenate into one
    movie, in the order their frames will run (``--concat``).

    A folder is searched recursively (see ``io.find_tif_movies``); a glob
    pattern is taken as given, minus the continuation files of split OME
    sets, which belong to their first file. Both are sorted by folder and
    file name with numbers compared numerically, so ``run_2`` comes
    before ``run_10``.
    """
    from glob import glob
    from os.path import abspath, isdir
    from .io import find_tif_movies, is_ome_continuation, natural_path_key

    if isdir(files):
        return find_tif_movies(files)
    paths = [abspath(_) for _ in glob(files)]
    skipped = [_ for _ in paths if is_ome_continuation(_)]
    if skipped:
        print(
            f"Skipping {len(skipped)} continuation file(s) of split "
            "OME-TIFF stacks; they are read together with their first "
            "file."
        )
    paths = [_ for _ in paths if not is_ome_continuation(_)]
    return sorted(paths, key=natural_path_key)


def _prompt_info() -> tuple[dict, bool]:
    """Interactively prompt the user for raw file metadata."""
    info = {}
    info["Byte Order"] = input("Byte Order (< or >): ")
    info["Data Type"] = input('Data Type (e.g. "uint16"): ')
    info["Frames"] = int(input("Frames: "))
    info["Height"] = int(input("Height: "))
    info["Width"] = int(input("Width: "))
    save = input("Use for all remaining raw files in folder (y/n)?") == "y"
    return info, save


def _localize_ensure_raw_yaml(paths: list[str], save_info) -> None:
    """Ensure every ``.raw`` file in ``paths`` has a paired ``.yaml``.

    Prompts the user interactively if a ``.yaml`` is missing.
    """
    import os as _os
    import os.path as _ospath

    save = False
    for path in paths:
        base, ext = _ospath.splitext(path)
        if ext == ".raw":
            if not _os.path.isfile(base + ".yaml"):
                print("No yaml found for {}. Please enter:".format(path))
                if not save:
                    info, save = _prompt_info()
                info_path = base + ".yaml"
                save_info(info_path, [info])


def _resolve_camera_pixelsize(
    pixelsize: float | None, movie, info: list[dict], path: str
) -> float:
    """The camera pixel size (nm) of a movie: the ``--pixelsize`` option
    if given, else the movie metadata's ``Pixelsize``, else the entry of
    the identified camera in the camera configuration (``Cameras`` in
    the config file), else an error asking for ``--pixelsize``.

    Parameters
    ----------
    pixelsize : float or None
        The ``--pixelsize`` option.
    movie : AbstractPicassoMovie
        The movie, whose ``camera_parameters`` names its camera.
    info : list of dicts
        The movie metadata.
    path : str
        Path of the movie, for the error message.

    Returns
    -------
    pixelsize : float
        Camera pixel size in nm.
    """
    from . import CONFIG, lib

    if pixelsize is not None:
        return float(pixelsize)
    value = lib.get_from_metadata(info, "Pixelsize")
    if value:
        return float(value)
    try:
        camera = movie.camera_parameters(CONFIG).get("camera")
    except Exception:
        camera = None
    cameras = CONFIG.get("Cameras") or {}
    if camera in cameras and cameras[camera].get("Pixelsize"):
        return float(cameras[camera]["Pixelsize"])
    raise SystemExit(
        f"The camera pixel size of {path} is unknown: it is neither in the "
        "movie metadata ('Pixelsize') nor in the camera configuration. "
        "Give it in nm with --pixelsize."
    )


def _localize_load_3d_calibration(
    args: argparse.Namespace,
) -> tuple[str, float, dict | None]:
    """Load 3D z-calibration, prompting interactively if needed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``picasso localize`` arguments; ``zc`` (calibration path),
        ``mf`` (magnification factor, 0 to prompt) and ``pixelsize`` are
        read.

    Returns
    -------
    tuple[str, float, dict or None]
        ``(zpath, magnification_factor, z_calibration)``;
        ``z_calibration`` is None for a calibration written before Picasso
        1.0 when ``--pixelsize`` is not given: it is converted with the
        pixel size of each movie instead.
    """
    import os
    from .io import MissingPixelsizeError, load_calibration

    print("------------------------------------------")
    print("Fitting 3D")

    if not os.path.isfile(args.zc):
        print(
            "Given path for calibration file not found."
            " Please enter manually:"
        )
        zpath = input("Path to *.yaml calibration file: ")
    else:
        zpath = args.zc

    if args.mf == 0:
        magnification_factor = float(input("Enter Magnification factor: "))
    else:
        magnification_factor = args.mf

    try:
        # a calibration written before Picasso 1.0 (spot widths in camera
        # pixels) is converted with --pixelsize or, without it, with the
        # pixel size of each movie (see _localize_legacy_calibrations)
        z_calibration = load_calibration(zpath, pixelsize=args.pixelsize)
    except MissingPixelsizeError:
        z_calibration = None
    except Exception as e:
        print(e)
        print("Error loading calibration file.")
        raise

    return zpath, magnification_factor, z_calibration


#: ``--vf-*`` options of the 'vectorfit' fit method and the
#: ``vectorfit.DEFAULT_PARAMETERS`` key each sets.
_VECTORFIT_ARGS = {
    "vf_model": "Fit model",
    "vf_wavelength": "Wavelength",
    "vf_na": "NA",
    "vf_n_medium": "Refractive index medium",
    "vf_n_coverslip": "Refractive index cover slip",
    "vf_n_immersion": "Refractive index immersion",
    "vf_vertical_astigmatism": "Vertical astigmatism",
    "vf_oblique_astigmatism": "Oblique astigmatism",
    "vf_horizontal_coma": "Horizontal coma",
    "vf_vertical_coma": "Vertical coma",
    "vf_spherical_aberration": "Spherical aberration",
}


def _add_vectorfit_args(parser: argparse.ArgumentParser) -> None:
    """Add the settings of the 'vectorfit' fit method. Their defaults are
    left to ``vectorfit.DEFAULT_PARAMETERS``, see
    ``_vectorfit_parameters_from_args``."""
    parser.add_argument(
        "--vf-model",
        choices=["2D", "3D"],
        default=None,
        help="VectorFit: fit model, 3D also fits z (default 2D)",
    )
    for flag, text in (
        ("--vf-wavelength", "emission wavelength in nm (default 680)"),
        ("--vf-na", "numerical aperture (default 1.5)"),
        (
            "--vf-n-medium",
            "refractive index of the sample medium (default 1.406)",
        ),
        (
            "--vf-n-coverslip",
            "refractive index of the cover slip (default 1.525)",
        ),
        (
            "--vf-n-immersion",
            "refractive index of the immersion fluid (default 1.515)",
        ),
        (
            "--vf-vertical-astigmatism",
            "fixed vertical astigmatism Z_2^2 in nm (default 0)",
        ),
        (
            "--vf-oblique-astigmatism",
            "fixed oblique astigmatism Z_2^-2 in nm (default 0)",
        ),
        (
            "--vf-horizontal-coma",
            "fixed horizontal coma Z_3^1 in nm (default 0)",
        ),
        ("--vf-vertical-coma", "fixed vertical coma Z_3^-1 in nm (default 0)"),
        (
            "--vf-spherical-aberration",
            "fixed spherical aberration Z_4^0 in nm (default 0)",
        ),
    ):
        parser.add_argument(
            flag, type=float, default=None, help=f"VectorFit: {text}"
        )
    parser.add_argument(
        "--vf-keep-outliers",
        action="store_true",
        help=(
            "VectorFit: keep the spots VectorFit flags as outliers (not "
            "converged, low likelihood or off-center) instead of discarding "
            "them"
        ),
    )


def _vectorfit_parameters_from_args(args: argparse.Namespace) -> dict:
    """The VectorFit settings given on the command line; the ones left out
    take the defaults of ``vectorfit.DEFAULT_PARAMETERS``."""
    parameters = {
        key: getattr(args, name)
        for name, key in _VECTORFIT_ARGS.items()
        if getattr(args, name, None) is not None
    }
    if getattr(args, "vf_keep_outliers", False):
        parameters["Discard outliers"] = False
    return parameters


def _add_identification_method_args(
    parser: argparse.ArgumentParser, what: str = "spot"
) -> None:
    """Add the arguments that select the identification method (and the
    wavelet settings), shared by every command that identifies spots or
    beads. The wavelet defaults are left to ``wavelet.WaveletParameters``,
    see ``_wavelet_from_args``."""
    parser.add_argument(
        "-im",
        "--identification-method",
        choices=["net-gradient", "wavelet"],
        default="net-gradient",
        help=(
            f"how {what}s are identified: local maxima above the minimum net"
            " gradient (the default), or B-spline wavelet segmentation"
            " (Izeddin et al., Opt. Express, 2012), which ignores --gradient"
        ),
    )
    parser.add_argument(
        "--wavelet-threshold",
        type=float,
        default=None,
        help=(
            "threshold of the wavelet identification in units of the noise"
            " standard deviation; default 0.5 (Izeddin et al. use 0.5-2)"
        ),
    )
    parser.add_argument(
        "--wavelet-noise",
        choices=["image-std", "w1-mad"],
        default=None,
        help=(
            "noise estimate of the wavelet identification: the standard"
            " deviation of the frame (image-std, the default) or the median"
            " absolute deviation of the first wavelet plane (w1-mad), which"
            " is robust to dense spots and uneven background"
        ),
    )
    parser.add_argument(
        "--wavelet-min-area",
        type=int,
        default=None,
        help=(
            "smallest region (in pixels) kept by the wavelet identification;"
            " default 4"
        ),
    )


def _wavelet_from_args(
    args: argparse.Namespace,
) -> WaveletParameters | None:
    """The wavelet identification settings given on the command line, or
    None for the net gradient identification. Read with defaults, since
    ``picasso.server.watcher`` builds its own argument namespace."""
    if getattr(args, "identification_method", None) != "wavelet":
        return None
    from .wavelet import WaveletParameters

    default = WaveletParameters()
    threshold = getattr(args, "wavelet_threshold", None)
    noise = getattr(args, "wavelet_noise", None)
    min_area = getattr(args, "wavelet_min_area", None)
    return WaveletParameters(
        threshold=default.threshold if threshold is None else threshold,
        noise=default.noise if noise is None else noise.replace("-", "_"),
        min_area=default.min_area if min_area is None else min_area,
    )


def _identification_method_from_args(
    args: argparse.Namespace,
    min_net_gradient: float | list[float],
) -> IdentificationMethod | list[NetGradient]:
    """The identification method given on the command line: the wavelet
    settings with ``--identification-method wavelet``, else the minimum
    net gradient (one per ROI region if a list)."""
    from .localize import NetGradient

    method = _wavelet_from_args(args)
    if method is not None:
        return method
    if isinstance(min_net_gradient, (list, tuple)):
        return [NetGradient(_) for _ in min_net_gradient]
    return NetGradient(min_net_gradient)


#: ``--gradient`` value that estimates the minimum net gradient of each
#: movie (see ``_localize_auto_gradient``)
_AUTO_GRADIENT = "auto"


def _gradient_arg(value: str) -> int | str:
    """``--gradient`` of ``picasso localize``: a minimum net gradient or
    'auto'."""
    if value.strip().lower() == _AUTO_GRADIENT:
        return _AUTO_GRADIENT
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid value: {value!r} (an integer or 'auto')"
        ) from None


def _localize_auto_gradient(
    movie,
    args: argparse.Namespace,
    box: int,
    min_net_gradient: int | str | list,
    roi,
    frame_bounds,
) -> int | str | list:
    """``min_net_gradient`` with every ``--gradient auto`` replaced by
    the estimate of ``localize.estimate_min_net_gradient`` for this
    movie, as Localize's 'Estimate' button makes it: on the movie the
    identification searches (filtered), within the ROIs and the frame
    bounds. A single 'auto' is estimated in all ROIs; with one value
    per ROI, each in its own. Left as is when the net gradient is not
    used (wavelet identification, or the multi-emitter fit without
    ``--seed-identification``)."""
    from . import localize

    values = (
        list(min_net_gradient)
        if isinstance(min_net_gradient, list)
        else [min_net_gradient]
    )
    if _AUTO_GRADIENT not in values:
        return min_net_gradient
    if _wavelet_from_args(args) is not None or (
        args.multi_emitter and not getattr(args, "seed_identification", False)
    ):
        return min_net_gradient
    # the filter settings; the threshold plays no part in the filters
    parameters = _identification_parameters_from_args(args, box, 0)
    filtered = localize._identification_filtered_movie(movie, parameters, roi)
    for index, value in enumerate(values):
        if value != _AUTO_GRADIENT:
            continue
        region = roi if len(values) == 1 else [roi[index]]
        result = localize.estimate_min_net_gradient(
            filtered,
            box=box,
            roi=region,
            frame_bounds=frame_bounds,
            progress="console",
        )
        # rounded and at least 0, like the value the button sets
        values[index] = max(int(round(result["min_net_gradient"])), 0)
        where = "" if len(values) == 1 else f" (ROI {index + 1})"
        print(
            f"Min. net gradient estimated{where}: {values[index]} "
            f"(background peak {result['background']:.0f}, width "
            f"{result['background_width']:.0f}; "
            f"{len(result['frames'])} frames)"
        )
    return values if isinstance(min_net_gradient, list) else values[0]


def _identification_parameters_from_args(
    args: argparse.Namespace,
    box: int,
    min_net_gradient: float | list[float],
) -> IdentificationParameters:
    """The identification settings given on the command line, see
    ``_identification_method_from_args``."""
    from .localize import IdentificationParameters

    return IdentificationParameters(
        method=_identification_method_from_args(args, min_net_gradient),
        box=box,
        temporal_median_window=args.temporal_median or 0,
        gaussian_filter_sigma=args.gaussian_filter or 0.0,
    )


def _localize_load_lateral_transforms(
    paths: list[str], pixelsize: float | None
) -> list:
    """The lateral corrections of the ``--affine-calibration`` files, in
    the order given, each counted once.

    Parameters
    ----------
    paths : list of str
        The ``--affine-calibration`` files.
    pixelsize : float or None
        Camera pixel size in nm, which converts files written before
        Picasso 1.0 (camera pixels).

    Returns
    -------
    lateral_transforms : list of dicts
        The corrections, in nm.

    Raises
    ------
    io.MissingPixelsizeError
        If a file was written before Picasso 1.0 and ``pixelsize`` is
        None.
    """
    from . import lib
    from .io import load_any_calibration

    lateral_transforms = []
    for affine_path in paths:
        found = lib.lateral_transforms(
            load_any_calibration(affine_path, pixelsize=pixelsize)
        )
        if not found:
            raise Exception(
                f"No lateral corrections found in {affine_path}. Build one "
                "with the 'Calibrate lateral transform' dialog in "
                "'picasso localize'."
            )
        # The same correction given twice - the flag repeated, or a
        # standalone copy of a transform another file already carries -
        # counts once: applying it twice would correct twice.
        found, duplicates = lib.drop_duplicate_lateral_transforms(
            found, lateral_transforms
        )
        if duplicates:
            print(
                f"Skipping {len(duplicates)} correction(s) from "
                f"{affine_path} already loaded from an earlier "
                "--affine-calibration: "
                + ", ".join(lib.describe_lateral_transforms(duplicates))
            )
        if found:
            lateral_transforms.extend(found)
            print(
                f"Loaded {len(found)} lateral correction(s) from "
                f"{affine_path}: "
                + ", ".join(lib.describe_lateral_transforms(found))
            )
    return lateral_transforms


def _localize_legacy_calibrations(
    args: argparse.Namespace,
    z_params,
    lateral_transforms: list | None,
    pixelsize: float,
) -> tuple:
    """The calibrations of a ``picasso localize`` run that were written
    before Picasso 1.0 (camera pixels) and not converted when they were
    loaded, as ``--pixelsize`` was not given: they are converted with the
    camera pixel size of the movie, which is assumed to be the camera of
    the calibration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``picasso localize`` arguments; ``affine_calibration`` is
        read.
    z_params : tuple or None
        ``(zpath, magnification_factor, z_calibration)`` of an astigmatic
        3D fit, ``z_calibration`` None if not loaded yet; None without a
        3D fit.
    lateral_transforms : list or None
        The ``--affine-calibration`` corrections, None if not loaded yet.
    pixelsize : float
        Camera pixel size of the movie in nm.

    Returns
    -------
    z_params : tuple or None
        As given, with the calibration loaded.
    lateral_transforms : list
        The corrections, in nm.
    """
    from .io import load_calibration

    if z_params is not None and z_params[2] is None:
        zpath, magnification_factor, _ = z_params
        z_calibration = load_calibration(zpath, pixelsize=pixelsize)
        print(
            f"Converted the 3D calibration {zpath}, written before Picasso "
            f"1.0, with the camera pixel size {pixelsize:g} nm."
        )
        z_params = (zpath, magnification_factor, z_calibration)
    if lateral_transforms is None:
        paths = getattr(args, "affine_calibration", []) or []
        if paths:
            print(
                "Converting the lateral corrections written before Picasso "
                f"1.0 with the camera pixel size {pixelsize:g} nm."
            )
        lateral_transforms = _localize_load_lateral_transforms(
            paths, pixelsize
        )
    return z_params, lateral_transforms


def _localize_photon_options(args: argparse.Namespace) -> dict:
    """The keywords of ``io.load_movie`` for photon-counting movies set by
    ``--frame-sum-window`` / ``--frame-sum-stride`` and
    ``--microtime-gate``; empty when none of them is used."""
    options = {}
    window = getattr(args, "frame_sum_window", 1)
    stride = getattr(args, "frame_sum_stride", None)
    if window != 1 or stride not in (None, 1):
        options["frame_sum_window"] = window
        options["frame_sum_stride"] = stride
    gates = getattr(args, "microtime_gate", None)
    if gates:
        options["microtime_gates"] = [tuple(gate) for gate in gates]
    return options


def _localize_check_photon_options(
    args: argparse.Namespace, paths: list[str], concat: bool
) -> None:
    """Exit with a message when scan frames are to be summed or gated for
    movies whose pixel values are not photon counts (anything but
    .ptu)."""
    if not _localize_photon_options(args):
        return
    others = [
        path for path in paths if os.path.splitext(path)[1].lower() != ".ptu"
    ]
    if concat or others:
        listed = ", ".join(others[:3]) + (" ..." if len(others) > 3 else "")
        raise SystemExit(
            "--frame-sum-window, --frame-sum-stride and --microtime-gate "
            "apply to photon-counting movies (.ptu) only: camera frames "
            "carry an offset (the baseline) in every frame, so a sum of "
            "their raw counts is not a count of photons, and they hold no "
            "photon arrival times to gate."
            + (f" Not photon-counting: {listed}" if others else "")
        )


def _localize_movies(paths: list[str], args: argparse.Namespace):
    """The movies of one ``picasso localize`` job, each with the suffix of
    its output files: the files concatenated into one movie
    (``--concat``), every channel of a .ptu file with several channels
    (detectors, or detectors x microtime gates) one after the other as
    ``_<channel>``, else the one movie, without a suffix."""
    from . import io

    if len(paths) > 1:
        movie, info = io.load_tif_concatenated(paths, progress="console")
        yield movie, info, ""
        return
    path = paths[0]
    options = _localize_photon_options(args)
    names = [None]
    if os.path.splitext(path)[1].lower() == ".ptu":
        io._require_ptufile()
        names = io.PTUMovie.channel_names(path, options.get("microtime_gates"))
    if len(names) == 1:
        movie, info = io.load_movie(path, progress="console", **options)
        yield movie, info, ""
        return
    for name in names:
        print(f"Channel {name}")
        movie, info = io.load_movie(
            path,
            prompt_info=lambda channels, name=name: name,
            progress="console",
            **options,
        )
        try:
            yield movie, info, "_" + io._sanitize_filename(name)
        finally:
            movie.close()


def _localize_process_file(
    paths: str | list[str],
    i: int,
    n_total: int,
    args: argparse.Namespace,
    box: int,
    min_net_gradient: float,
    roi,
    frame_bounds,
    camera_info: dict,
    convergence: float,
    max_iterations: int,
    z_params,
    spline_calibration: dict | None = None,
    camera_calibration: dict | None = None,
    lateral_transforms: list | None = None,
    region_fit: dict | None = None,
) -> list[str]:
    """Identify, fit, save and optionally undrift one movie.

    Parameters
    ----------
    paths : str or list of str
        The movie to process. Several paths (``--concat``) are read as
        one movie whose frames run through the files in that order; the
        results are saved next to the first file.
    i : int
        Index of the movie, for the progress message.
    n_total : int
        Total number of movies, for the progress message.
    args : argparse.Namespace
        Parsed ``picasso localize`` arguments.
    box : int
        Side length of the fitting box in camera pixels.
    min_net_gradient : float, "auto" or list of them
        Minimum net gradient for identification; one value per ROI
        region if a list. "auto" is estimated for each movie (see
        ``_localize_auto_gradient``).
    roi : list or None
        ROI regions, each ``[[y_min, x_min], [y_max, x_max]]``, or None
        for the whole frame.
    frame_bounds : list of list of int or None
        Frame segments ``[start, end]`` (0-indexed, inclusive) to
        analyze, or None for all frames.
    camera_info : dict
        Camera parameters (baseline, sensitivity, gain).
    convergence : float
        Fit convergence criterion; 0 uses the fitting method's default.
    max_iterations : int
        Maximum number of fit iterations; 0 uses the fitting method's
        default.
    z_params : tuple or None
        If 3D astigmatism fitting is active, a tuple
        ``(zpath, magnification_factor, z_calibration)``; else ``None``.
        ``z_calibration`` is None for a calibration written before
        Picasso 1.0 that is converted with the pixel size of the movie.
    spline_calibration : dict or None
        Cubic-spline PSF calibration when the fit method is a spline method;
        else ``None``. A 3D spline fit recovers z directly (no ``zfit``).
    camera_calibration : dict or None, optional
        Per-pixel sCMOS camera calibration, or None for a uniform camera.
    lateral_transforms : list or None
        Lateral corrections loaded separately from the calibration used
        for fitting (see ``--affine-calibration``), applied to x/y after
        fitting and, for 3D astigmatism, after the ones the 3D calibration
        carries - so loading one separately gives the same coordinates as
        appending it to the calibration. Corrections the spline or
        astigmatism calibration carries - and therefore applies itself -
        are skipped rather than applied a second time. None loads them
        from ``args.affine_calibration`` with the pixel size of the movie
        (files written before Picasso 1.0, without ``--pixelsize``).
    region_fit : dict or None
        Set by ``--regions-separately``: fit each ``--roi`` region on its
        own and write one file per region. Holds the per-region fitting
        methods and spline calibrations (see ``_localize_regions``).

    Returns
    -------
    saved : list of str
        Paths of the localization files written: one, or one per region
        with ``region_fit``, or one per channel of a .ptu file with
        several channels.
    """
    if isinstance(paths, str):
        paths = [paths]
    path = paths[0]

    print("------------------------------------------")
    print("------------------------------------------")
    if len(paths) > 1:
        print(
            f"Processing {len(paths)} files concatenated into one movie, "
            f"starting with {path}"
        )
        for j, each in enumerate(paths):
            print(f"  {j + 1}. {each}")
    else:
        print(f"Processing {path}, File {i + 1} of {n_total}")
    print("------------------------------------------")
    saved = []
    for movie, info, suffix in _localize_movies(paths, args):
        saved += _localize_movie(
            movie,
            info,
            path,
            suffix,
            args,
            box,
            min_net_gradient,
            roi,
            frame_bounds,
            camera_info,
            convergence,
            max_iterations,
            z_params,
            spline_calibration=spline_calibration,
            camera_calibration=camera_calibration,
            lateral_transforms=lateral_transforms,
            region_fit=region_fit,
        )
    return saved


def _localize_movie(
    movie,
    info: list[dict],
    path: str,
    suffix: str,
    args: argparse.Namespace,
    box: int,
    min_net_gradient: float,
    roi,
    frame_bounds,
    camera_info: dict,
    convergence: float,
    max_iterations: int,
    z_params,
    spline_calibration: dict | None = None,
    camera_calibration: dict | None = None,
    lateral_transforms: list | None = None,
    region_fit: dict | None = None,
) -> list[str]:
    """Identify, fit, save and optionally undrift one loaded movie of
    ``_localize_process_file``, whose arguments it takes; ``suffix`` is
    inserted into the output names before ``_locs`` (a channel of a .ptu
    file)."""
    from .localize import localize, resolve_fit_method

    fitting_method = resolve_fit_method(args.fit_method)[0]
    cam_info = dict(camera_info)
    cam_info["Pixelsize"] = _resolve_camera_pixelsize(
        args.pixelsize, movie, info, path
    )
    print(f"Camera pixel size: {cam_info['Pixelsize']:g} nm")
    z_params, lateral_transforms = _localize_legacy_calibrations(
        args, z_params, lateral_transforms, cam_info["Pixelsize"]
    )
    min_net_gradient = _localize_auto_gradient(
        movie, args, box, min_net_gradient, roi, frame_bounds
    )
    if args.multi_emitter:
        return [
            _localize_multi_emitter(
                movie,
                info,
                path,
                args,
                box,
                min_net_gradient,
                roi,
                frame_bounds,
                cam_info,
                camera_calibration,
                z_params,
                lateral_transforms,
                spline_calibration,
                suffix=suffix,
            )
        ]
    if region_fit is not None:
        return _localize_regions(
            movie,
            info,
            path,
            args,
            box,
            min_net_gradient,
            roi,
            frame_bounds,
            cam_info,
            convergence,
            max_iterations,
            z_params,
            region_fit,
            camera_calibration=camera_calibration,
            lateral_transforms=lateral_transforms,
            suffix=suffix,
        )
    phasor_calibration = None
    if fitting_method == "phasor" and z_params is not None:
        # the phasor fit takes z from its calibration itself, so no z fit
        # follows (and the fit applies the lateral corrections)
        _, magnification_factor, z_calibration = z_params
        phasor_calibration = dict(z_calibration)
        phasor_calibration["Magnification factor"] = magnification_factor
        z_params = None
    with _lateral_report_only():
        locs, info = localize(
            movie,
            camera_info=cam_info,
            identification_parameters=_identification_parameters_from_args(
                args, box, min_net_gradient
            ),
            roi=roi,
            frame_bounds=frame_bounds,
            movie_info=info,
            fitting_method=fitting_method,
            eps=convergence if convergence > 0 else None,
            max_it=max_iterations if max_iterations > 0 else None,
            spline_calibration=spline_calibration,
            vectorfit_parameters=_vectorfit_parameters_from_args(args),
            calibration_3d=phasor_calibration,
            camera_calibration=camera_calibration,
            # in 3D, the z fit applies them (see _localize_finish)
            affine_calibration=None if z_params else lateral_transforms,
            parallel=True,
            progress="console",
        )

    saved = _localize_finish(
        locs,
        info,
        path,
        args,
        z_params,
        lateral_transforms,
        spline_calibration or phasor_calibration,
        suffix=suffix,
    )
    return [saved]


def _check_3d_calibration_kind(
    args: argparse.Namespace, calibration: dict | None
) -> None:
    """Refuse a 3D calibration of the wrong kind before any movie is
    read: the phasor fit ('phasor-3d') needs an astigmatic phasor
    calibration, the Gaussian z fit a Gaussian one."""
    if calibration is None:  # written before Picasso 1.0: Gaussian
        phasor_calibration = False
    else:
        from .zfit import _is_phasor_calibration

        phasor_calibration = _is_phasor_calibration(calibration)
    if args.fit_method == "phasor-3d" and not phasor_calibration:
        raise Exception(
            "'phasor-3d' needs an astigmatic phasor calibration (made in "
            "Localize: Calibration > Calibrate astigmatism (phasor), or with "
            "picasso.zfit.calibrate_z_phasor); this is a Gaussian one."
        )
    if args.fit_method != "phasor-3d" and phasor_calibration:
        raise Exception(
            "This is an astigmatic phasor calibration, which only "
            "'--fit-method phasor-3d' can use; the Gaussian z fit needs a "
            "Gaussian astigmatism calibration."
        )


def _multi_emitter_parameters_from_args(args: argparse.Namespace):
    """The ``localize.MultiEmitterParameters`` given on the command
    line."""
    from .localize import MultiEmitterParameters

    return MultiEmitterParameters(
        psf_model=args.psf_model,
        snr_threshold=args.snr_threshold,
        psf_sigma=args.psf_sigma,
        max_cycles=args.max_cycles,
        peak_radius=args.peak_radius,
        background_sigma=args.background_sigma,
        peak_search_z=tuple(args.peak_search_z),
        fit_region_size=args.fit_region_size,
    )


def _localize_multi_emitter(
    movie,
    info: list[dict],
    path: str,
    args: argparse.Namespace,
    box: int,
    min_net_gradient: float | list[float],
    roi,
    frame_bounds,
    camera_info: dict,
    camera_calibration: dict | None,
    z_params,
    lateral_transforms: list | None,
    spline_calibration: dict | None,
    suffix: str = "",
) -> str:
    """Fit one movie with the multi-emitter fitting (``--multi-emitter``),
    then save it and undrift it as any other fit. With ``z_params`` (an
    elliptical fit, see ``_check_multi_emitter_args``), z is read from the
    fitted sigmas, and the z fit applies the lateral corrections. With
    ``spline_calibration`` (``--psf-model spline``), it is the PSF, and
    its own lateral corrections are not applied again.

    Returns
    -------
    out_path : str
        Path of the localization file written.
    """
    from .io import strip_mm_metadata
    from .localize import localize_multi_emitter

    seed_identification = None
    if args.seed_identification:
        seed_identification = _identification_parameters_from_args(
            args, box, min_net_gradient
        )
    calibration_3d = None
    if z_params is not None:
        _, magnification_factor, z_calibration = z_params
        calibration_3d = dict(z_calibration)
        calibration_3d["Magnification factor"] = magnification_factor
    with _lateral_report_only():
        locs, info = localize_multi_emitter(
            movie,
            # as localize.localize does, see "Save Micro-Manager metadata"
            strip_mm_metadata(info),
            camera_info=camera_info,
            parameters=_multi_emitter_parameters_from_args(args),
            roi=roi,
            frame_bounds=frame_bounds,
            seed_identification=seed_identification,
            camera_calibration=camera_calibration,
            calibration_3d=calibration_3d,
            spline_calibration=spline_calibration,
            lateral_transforms=lateral_transforms,
            parallel=True,
            fit_z_gpu=args.fit_z_gpu,
            progress="console",
        )
    entry = next(e for e in info if e.get("Step") == "Fit multi-emitter")
    if spline_calibration is None:
        print(
            f"PSF sigma: {entry['PSF Sigma (nm)']:.1f} nm"
            + (" (estimated)" if entry["PSF Sigma Estimated"] else "")
            + ("" if args.psf_model == "fixed" else ", the start of the fit")
        )
    if calibration_3d is not None:
        _print_lateral_report(
            info[-1].get("Lateral corrections applied"),
            info[-1].get("Lateral corrections skipped"),
        )
        # applied and reported by the z fit
        lateral_transforms = None
    return _localize_finish(
        locs,
        info,
        path,
        args,
        None,
        lateral_transforms,
        spline_calibration,
        suffix=suffix,
    )


def _localize_region_fit(
    fit_methods: list[str],
    spline_calibrations: list[dict],
    n_regions: int,
) -> dict:
    """The per-region fitting methods and spline calibrations of a
    ``--regions-separately`` run, one entry per region.

    A single ``--fit-method`` / ``--spline-calibration`` applies to every
    region; several must match the ``--roi`` count, so that it is clear
    which region gets which."""
    if n_regions < 1:
        raise Exception(
            "--regions-separately fits each --roi region on its own, so at "
            "least one --roi is needed. Pass the regions, e.g. "
            "--roi 0 0 256 256 --roi 0 256 256 512."
        )
    from .localize import resolve_fit_method

    methods = [resolve_fit_method(method)[0] for method in fit_methods]
    if len(methods) not in (1, n_regions):
        raise Exception(
            f"{len(methods)} fitting methods were given but {n_regions} "
            "ROIs; pass one --fit-method per --roi, or a single one for all "
            "of them."
        )
    if len(methods) == 1:
        methods = methods * n_regions
    calibrations = list(spline_calibrations)
    if calibrations and len(calibrations) not in (1, n_regions):
        raise Exception(
            f"{len(calibrations)} spline calibrations were given but "
            f"{n_regions} ROIs; pass one --spline-calibration per --roi, or "
            "a single one for all of them."
        )
    if len(calibrations) == 1:
        calibrations = calibrations * n_regions
    if not calibrations:
        calibrations = [None] * n_regions
    return {"methods": methods, "spline_calibrations": calibrations}


def _localize_regions(
    movie,
    info: list[dict],
    path: str,
    args: argparse.Namespace,
    box: int,
    min_net_gradient,
    roi: list,
    frame_bounds,
    cam_info: dict,
    convergence: float,
    max_iterations: int,
    z_params,
    region_fit: dict,
    camera_calibration: dict | None = None,
    lateral_transforms: list | None = None,
    suffix: str = "",
) -> list[str]:
    """Fit each ``--roi`` region of one movie on its own
    (``--regions-separately``).

    The regions are channels imaged side by side on one sensor, so the spots
    are identified once over the whole frame and then fitted region by
    region - each with its own fitting method and spline calibration, each
    saved to its own file, in its own coordinates. Nothing is registered or
    linked across the regions; they are connected afterwards (e.g. by
    aligning them in Picasso: Render).

    Returns
    -------
    saved : list of str
        Paths of the localization files written, one per region with
        localizations.
    """
    from .io import strip_mm_metadata
    from .localize import (
        fit_split_fov_independent,
        identify,
        region_label,
    )

    identifications, ids_info = identify(
        movie,
        # as localize.localize does, see "Save Micro-Manager metadata"
        strip_mm_metadata(info),
        parameters=_identification_parameters_from_args(
            args, box, min_net_gradient
        ),
        roi=roi,
        frame_bounds=frame_bounds,
        parallel=True,
        progress="console",
    )
    print(
        f"Identified {len(identifications):,} spots in {len(roi)} regions. "
        "Fitting each region on its own..."
    )
    with _lateral_report_only():
        results = fit_split_fov_independent(
            movie,
            camera_info=cam_info,
            identifications=identifications,
            box=box,
            regions=roi,
            fitting_method=region_fit["methods"],
            info=ids_info,
            eps=convergence if convergence > 0 else None,
            max_it=max_iterations if max_iterations > 0 else None,
            spline_calibration=region_fit["spline_calibrations"],
            vectorfit_parameters=_vectorfit_parameters_from_args(args),
            camera_calibration=camera_calibration,
            # in 3D, the z fit applies them (see _localize_finish)
            lateral_transforms=None if z_params else lateral_transforms,
            parallel=True,
            progress="console",
        )
    saved = []
    for c, (locs, region_info) in enumerate(results):
        label = region_label(c)
        print("------------------------------------------")
        if not len(locs):
            # an empty file would say nothing; the missing one, with this
            # line, says the threshold found nothing in that region
            print(
                f"Region {label}: no spots identified in it, nothing saved. "
                "Lower its --gradient to detect any."
            )
            continue
        print(f"Region {label}: {len(locs):,} localizations")
        saved.append(
            _localize_finish(
                locs,
                region_info,
                path,
                args,
                z_params,
                lateral_transforms,
                region_fit["spline_calibrations"][c],
                suffix=f"{suffix}_{label}",
                fitting_method=region_fit["methods"][c],
            )
        )
    return saved


@contextlib.contextmanager
def _lateral_report_only() -> Iterator[None]:
    """Fit without warning about the lateral corrections skipped because
    the calibration used for fitting applies them already: the console
    report of ``_print_lateral_report`` says so instead."""
    import warnings
    from . import lib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", lib.DuplicateLateralTransformWarning)
        yield


def _print_lateral_report(
    applied: list[str] | None, skipped: list[str] | None
) -> None:
    """Report what happened to the lateral corrections, in the console
    idiom - the same lines whether the fit was 2D or 3D."""
    if applied:
        print("Applied lateral correction(s): " + ", ".join(applied))
    if skipped:
        print(
            "Skipping "
            + ", ".join(skipped)
            + ": the calibration used for fitting already applies this"
            " correction itself; applying it again would correct twice."
        )


def _localize_finish(
    locs,
    info: list[dict],
    path: str,
    args: argparse.Namespace,
    z_params,
    lateral_transforms: list | None,
    spline_calibration: dict | None,
    suffix: str = "",
    fitting_method: str | None = None,
) -> str:
    """Everything that happens to one set of localizations once it is
    fitted: the astigmatic z fit, the separately loaded lateral
    corrections, saving, the quality database and the drift correction.

    Split out of ``_localize_process_file`` so that a run that fits the
    regions separately (``--regions-separately``) does all of it per region,
    each region writing its own file.

    Parameters
    ----------
    locs : pd.DataFrame
        Fitted localizations.
    info : list of dict
        Metadata of the localizations.
    path : str
        Path of the movie; the output is saved next to it.
    args : argparse.Namespace
        Parsed ``picasso localize`` arguments.
    z_params : tuple or None
        If 3D astigmatism fitting is active, a tuple
        ``(zpath, magnification_factor, z_calibration)``; else ``None``.
    lateral_transforms : list or None
        Lateral corrections loaded separately from the calibration used
        for fitting (see ``_localize_process_file``). Without
        ``z_params``, the fit has applied them already, so they are
        only reported here; with it, the z fit applies them.
    spline_calibration : dict or None
        Spline PSF (or phasor) calibration used for fitting, whose own
        lateral corrections are not applied again; None if not a spline
        fit or a phasor fit in 3D.
    suffix : str, optional
        Inserted into the output name before ``_locs``, e.g. ``"_ref"`` for
        one region of a split field of view. Default "" (one file per movie).
    fitting_method : str or None, optional
        The method this set was fitted with, which decides the noise model
        of the astigmatic z fit. None (the default) uses ``args.fit_method``.

    Returns
    -------
    out_path : str
        Path of the localization file written.
    """
    import warnings
    from os.path import splitext
    from . import lib
    from .io import save_locs
    from .localize import add_file_to_db
    from .vectorfit import outliers_discarded

    method_name = fitting_method or args.fit_method
    n_outliers = outliers_discarded(info)
    if n_outliers is not None:
        print(
            f"VectorFit discarded {n_outliers:,} outliers out of "
            f"{len(locs) + n_outliers:,} spots."
        )
    if z_params is not None:
        from . import zfit

        _, magnification_factor, z_calibration = z_params
        print("------------------------------------------")
        print("Fitting 3D...")
        method = "gausslq" if "mle" not in method_name else "gaussmle"
        # The corrections loaded with --affine-calibration go through zfit
        # too, which applies them after the ones the 3D calibration carries
        # and skips any it carries already: loading a correction separately
        # then gives the same coordinates as appending it to the 3D
        # calibration. It says so itself through a warning, which the
        # console message below replaces.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", lib.DuplicateLateralTransformWarning
            )
            locs, info = zfit.zfit(
                locs=locs,
                info=info,
                calibration=z_calibration,
                # --magnification-factor, over the calibration's own
                magnification_factor=magnification_factor,
                fitting_method=method,
                filter=0,
                lateral_transforms=lateral_transforms,
                parallel=not args.fit_z_gpu,
                gpu=args.fit_z_gpu,
                progress="console",
            )
        print("3D fitting complete.")
        print("------------------------------------------")
        _print_lateral_report(
            info[-1].get("Lateral corrections applied"),
            info[-1].get("Lateral corrections skipped"),
        )
    elif lateral_transforms:
        # No z fit to fold them into, so the fit applied them and recorded
        # them in an entry of its own. It skipped the corrections the spline
        # calibration carries, which it applies itself; passing that same
        # file to --affine-calibration would otherwise correct twice.
        extra, duplicates = lib.drop_duplicate_lateral_transforms(
            lateral_transforms, spline_calibration
        )
        _print_lateral_report(
            lib.describe_lateral_transforms(extra),
            lib.describe_lateral_transforms(duplicates),
        )

    base, ext = splitext(path)

    try:
        sfx = args.suffix
    except Exception:
        sfx = ""

    out_path = f"{base}{sfx}{suffix}_locs.hdf5"
    save_locs(out_path, locs, info)
    print("File saved to {}".format(out_path))

    CHECK_DB = getattr(args, "database", False)
    if CHECK_DB:
        print("\n")
        print("Assessing quality and adding to DB")
        add_file_to_db(path, out_path)
        print("Done.")
        print("\n")

    if args.drift > 0:
        print("Undrifting file:")
        print("------------------------------------------")
        try:
            _undrift_rcc(
                out_path,
                args.drift,
                display=False,
                fromfile=None,
            )
        # _undrift_rcc exits for too few frames; the localizations are
        # saved already and the next movie is still localized
        except (Exception, SystemExit) as e:
            print(e)
            print("Drift correction failed for {}".format(out_path))

    print("                                          ")
    return out_path


def _check_multi_emitter_args(args: argparse.Namespace) -> None:
    """Refuse the options the multi-emitter fitting cannot honor: it fits
    by maximum likelihood, on the CPU, one region of one channel at a
    time, takes a spline calibration only as the PSF of --psf-model
    spline, and gives z otherwise only with sigmas in x and y."""
    fit_methods = args.fit_method or ["mle"]
    if fit_methods != ["mle"]:
        raise SystemExit(
            "--multi-emitter fits by maximum likelihood; leave out "
            "--fit-method."
        )
    n_splines = len(args.spline_calibration or [])
    if args.psf_model == "spline" and n_splines != 1:
        raise SystemExit(
            "--multi-emitter --psf-model spline needs one "
            "--spline-calibration, the PSF it fits."
        )
    if args.psf_model != "spline" and n_splines:
        raise SystemExit(
            "--multi-emitter fits a Gaussian unless --psf-model spline; "
            "add it, or leave out --spline-calibration."
        )
    if args.psf_model != "spline" and args.fit_region_size is not None:
        raise SystemExit(
            "--multi-emitter takes the fit region of a Gaussian from its "
            "sigma; leave out --fit-region-size, or add --psf-model spline."
        )
    if args.zc and args.psf_model != "elliptical":
        raise SystemExit(
            "--multi-emitter reads z from the sigmas in x and y it fits "
            "with --psf-model elliptical; add it, or leave out "
            "--z-calibration."
        )
    if args.regions_separately:
        raise SystemExit(
            "--multi-emitter cannot fit regions separately yet; leave out "
            "--regions-separately."
        )


def _localize(args: argparse.Namespace) -> list[str]:  # noqa: C901
    """Localize molecules in microscopy images.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``picasso localize`` arguments. Among others:

        - ``files`` (str): path to the microscopy image files or a
          directory containing image files.
        - ``fit_method`` (list of str): fitting method(s), e.g.
          ``'mle'``, ``'lq-3d'`` or ``'lq-gpu-3d'``; one per ROI region
          with ``regions_separately``.
        - ``box_side_length`` (int): side length of the fitting box.
        - ``gradient`` (list of int or "auto"): minimum net gradient(s),
          one per ROI region or a single one for all; "auto" estimates
          it for each movie.
        - ``roi`` (list of list of int): regions of interest, each
          ``[y_min, x_min, y_max, x_max]``.
        - ``frame_bounds`` (list of list of int): one or more
          ``[start_frame, end_frame]`` segments, 0-indexed and inclusive.
          Several segments restrict the analysis to the union of those
          (disjoint) frame ranges.
        - ``baseline``, ``sensitivity``, ``gain`` (float): camera
          parameters.

    Returns
    -------
    saved : list of str
        Paths of the localization files written, in order, e.g., for the
        custom command of the server's watcher (``$LOCS``).
    """
    from . import localize
    from .io import save_info

    picasso_logo()
    print("Localize - Parameters:")
    print("{:<8} {:<15} {:<10}".format("No", "Label", "Value"))

    # The repeatable options collapse to their single value unless the
    # regions are fitted separately, where they are one per region; the
    # first one stands for the run as a whole (the GPU check, the 3D and
    # spline branches below).
    if getattr(args, "multi_emitter", False):
        _check_multi_emitter_args(args)
    else:
        args.multi_emitter = False
    fit_methods = args.fit_method or ["mle"]
    args.fit_method = fit_methods[0]
    gradients = args.gradient or [5000]
    args.gradient = gradients[0]
    spline_paths = args.spline_calibration or []
    args.spline_calibration = spline_paths[0] if spline_paths else ""
    if any(str(m).startswith("vectorfit") for m in fit_methods):
        # before any movie is read: e.g. 3D needs astigmatism
        from .vectorfit import parameters_with_defaults

        parameters_with_defaults(_vectorfit_parameters_from_args(args))
    if localize.resolve_fit_method(args.fit_method)[0].endswith("-gpu"):
        if args.fit_method == "vectorfit-gpu":
            # VectorFit's own GPU build, with its own requirements
            from .vectorfit import gpu_unavailable_reason

            reason = gpu_unavailable_reason()
            if reason is not None:
                raise Exception(
                    f"VectorFit cannot fit on the GPU. {reason} Aborting."
                )
            print("CUDA GPU found")
        elif localize.CUDA_AVAILABLE:
            print("CUDA GPU found")
        else:
            raise Exception(
                "No CUDA-capable GPU found, so the requested GPU fit method "
                "cannot run. Aborting."
            )

    for index, element in enumerate(vars(args)):
        try:
            print(
                "{:<8} {:<15} {:<10}".format(
                    index + 1, element, getattr(args, element)
                )
            )
        except TypeError:  # if None is default value
            print("{:<8} {:<15} {}".format(index + 1, element, "None"))
    print("------------------------------------------")

    concat = getattr(args, "concat", False)
    if concat:
        paths = _localize_collect_concat_paths(args.files)
    else:
        paths = _localize_collect_paths(args.files)
        _localize_ensure_raw_yaml(paths, save_info)

    if not paths:
        print("Error. No files found.")
        raise FileNotFoundError
    _localize_check_photon_options(args, paths, concat)

    if concat:
        print(
            f"Concatenating {len(paths)} file(s) into a single movie, in "
            "this order:"
        )
        for i, path in enumerate(paths):
            print(f"  {i + 1}. {path}")
        print("------------------------------------------")

    print(args)
    box = args.box_side_length
    min_net_gradient = gradients[0] if len(gradients) == 1 else gradients
    roi = args.roi
    if roi is not None:
        # argparse (action="append", nargs=4) yields a list of 4-int
        # lists; map each to [[y_min, x_min], [y_max, x_max]] and clip so
        # the regions do not overlap.
        from .localize import clip_rois

        rois = [
            [[y_min, x_min], [y_max, x_max]]
            for y_min, x_min, y_max, x_max in roi
        ]
        roi = clip_rois(rois, min_size=box)
    n_regions = len(roi) if roi else 0
    if len(gradients) > 1 and len(gradients) != n_regions:
        raise Exception(
            f"{len(gradients)} minimum net gradients were given but "
            f"{n_regions} ROIs; pass one --gradient per --roi, or a single "
            "one for all of them."
        )
    frame_bounds = args.frame_bounds
    camera_info = {
        "Baseline": args.baseline,
        "Sensitivity": args.sensitivity,
        "Gain": args.gain,
    }

    # 0 means "let the fitting method use its own default", which differs
    # per model and per device - see localize.fit's eps / max_it.
    convergence = args.convergence
    max_iterations = args.max_iterations

    z_params = None
    # the multi-emitter fit reads z from the sigmas of --psf-model
    # elliptical (see _check_multi_emitter_args)
    if "-3d" in args.fit_method or (args.multi_emitter and args.zc):
        z_params = _localize_load_3d_calibration(args)
        _check_3d_calibration_kind(args, z_params[2])
        if args.fit_z_gpu:
            from . import zfit

            if zfit.CUDA_AVAILABLE:
                print("GPU z fitting enabled (numba.cuda)")
            else:
                print(
                    "Warning: GPU z fitting requested (--fit-z-gpu) but no "
                    "CUDA-capable GPU is available. Falling back to "
                    "multiprocessed CPU z fitting."
                )
                args.fit_z_gpu = False

    spline_calibrations = []
    if any(method.startswith("spline") for method in fit_methods) or (
        args.multi_emitter and args.psf_model == "spline"
    ):
        from .io import load_spline_calibration

        if not spline_paths:
            raise Exception(
                "Spline fitting requires --spline-calibration <file.hdf5>. "
                "Build one with 'picasso spline-calibrate'."
            )
        for calibration_path in spline_paths:
            calibration = load_spline_calibration(calibration_path)
            print(
                "Loaded spline PSF calibration "
                f"({calibration.get('model')}) from {calibration_path}"
            )
            spline_calibrations.append(calibration)
    spline_calibration = (
        spline_calibrations[0] if spline_calibrations else None
    )
    region_fit = None
    if args.regions_separately:
        if "phasor-3d" in fit_methods:
            raise Exception(
                "'phasor-3d' cannot be combined with --regions-separately "
                "yet; fit the regions with 'phasor' (2D), or fit them "
                "together."
            )
        region_fit = _localize_region_fit(
            fit_methods, spline_calibrations, n_regions
        )

    # Extra lateral affine corrections, applied after fitting on top of any
    # the 3D / spline calibration carries. Files written before Picasso 1.0
    # (camera pixels) are converted with --pixelsize or, without it, with
    # the pixel size of each movie (see _localize_legacy_calibrations).
    from .io import MissingPixelsizeError

    try:
        lateral_transforms = _localize_load_lateral_transforms(
            getattr(args, "affine_calibration", []) or [], args.pixelsize
        )
    except MissingPixelsizeError:
        lateral_transforms = None

    camera_calibration = None
    if args.camera_calibration:
        from .io import load_camera_calibration

        camera_calibration = load_camera_calibration(args.camera_calibration)
        gain_note = (
            "offset, variance and gain"
            if camera_calibration.get("gain") is not None
            else "offset and variance"
        )
        print(
            f"Loaded sCMOS camera calibration ({gain_note} maps, "
            f"{camera_calibration.get('Height')}x"
            f"{camera_calibration.get('Width')} pixels, "
            f"{camera_calibration.get('Frames')} dark frames) from "
            f"{args.camera_calibration}"
        )

    # Normally one movie per file; with --concat all files together are
    # one movie, so there is a single job.
    jobs = [paths] if concat else [[_] for _ in paths]
    saved = []
    for i, job in enumerate(jobs):
        saved += _localize_process_file(
            job,
            i,
            len(jobs),
            args,
            box,
            min_net_gradient,
            roi,
            frame_bounds,
            camera_info,
            convergence,
            max_iterations,
            z_params,
            spline_calibration=spline_calibration,
            lateral_transforms=lateral_transforms,
            camera_calibration=camera_calibration,
            region_fit=region_fit,
        )
    return saved


def _camera_calibrate(args: argparse.Namespace) -> None:
    """Characterize an sCMOS camera from a dark movie (and optional light)."""
    from os.path import splitext
    from . import scmos
    from .io import load_movie, save_camera_calibration

    picasso_logo()
    print("sCMOS camera calibration")
    print("------------------------------------------")

    dark_movie, _ = load_movie(args.dark)
    print(f"Dark movie: {args.dark} ({len(dark_movie)} frames)")
    bright_movies, bright_paths = [], []
    for path in args.light or []:
        movie, _ = load_movie(path)
        bright_movies.append(movie)
        bright_paths.append(path)
        print(f"Light movie: {path} ({len(movie)} frames)")
    if not bright_movies:
        print(
            "No light movies given, so no gain map: the scalar Sensitivity "
            "is used for the counts-to-photons conversion. Pass -l/--light "
            "several times, at different illumination levels, to measure the "
            "per-pixel gain as well."
        )

    powers = args.power or None
    if powers is not None and len(powers) != len(bright_movies):
        raise ValueError(
            f"Got {len(powers)} -p/--power value(s) for "
            f"{len(bright_movies)} light movie(s). Pass one per movie, in "
            "the same order, or none at all."
        )

    calibration = scmos.calibrate_scmos(
        dark_movie,
        bright_movies or None,
        progress="console",
        dark_path=args.dark,
        bright_paths=bright_paths,
        bright_levels=powers,
        level_unit=args.power_unit,
    )

    out_path = args.output
    if not out_path:
        base, _ = splitext(args.dark)
        out_path = base + "_scmos_calib.hdf5"
    calibration["Path"] = out_path
    save_camera_calibration(out_path, calibration)

    print("------------------------------------------")
    print(f"Frames used:       {calibration['Frames']}")
    print(
        f"Offset (ADU):      median "
        f"{calibration['Offset median (ADU)']:.2f}, range "
        f"{calibration['Offset min (ADU)']:.2f} - "
        f"{calibration['Offset max (ADU)']:.2f}"
    )
    print(
        f"Variance (ADU^2):  median "
        f"{calibration['Variance median (ADU^2)']:.2f}, 99.9th pct "
        f"{calibration['Variance 99.9 percentile (ADU^2)']:.1f}, max "
        f"{calibration['Variance max (ADU^2)']:.1f}"
    )
    print(
        f"Hot pixels:        {calibration['Hot pixels']} above "
        f"{calibration['Hot pixel threshold (ADU^2)']:.1f} ADU^2"
    )
    if calibration.get("gain") is not None:
        print(
            f"Gain (ADU/e-):     median "
            f"{calibration['Gain median (ADU/e-)']:.3f}, range "
            f"{calibration['Gain min (ADU/e-)']:.3f} - "
            f"{calibration['Gain max (ADU/e-)']:.3f} "
            f"({calibration['Gain levels']} illumination levels)"
        )
        signal = calibration.get("Level median signal (ADU)") or []
        if len(signal) > 1:
            # The gain fit assumes the response is linear over the range the
            # series covers, and nothing in the fit itself would complain if
            # a level had saturated.
            print(
                "Level medians:     "
                + ", ".join(f"{value:.1f}" for value in signal)
                + " ADU (mean - offset)"
            )
        if calibration["Gain fallback pixels"]:
            print(
                f"                   {calibration['Gain fallback pixels']} "
                "unresponsive pixel(s) took the chip median"
            )
    print(f"Saved to {out_path}")
    # A dead column, a bright corner or a cluster of hot pixels shows in the
    # maps and in nothing else, so they are written out alongside.
    try:
        plot_path = scmos.save_calibration_plot(
            calibration, scmos.plot_path(out_path)
        )
        print(f"Maps and histograms: {plot_path}")
    except Exception as error:
        print(f"The diagnostic plot could not be saved: {error}")
    print("Use it with: picasso localize <movie> -cm " f"{out_path} -a mle")


def _camera_validate(args: argparse.Namespace) -> None:
    """Check a camera calibration against a short, fresh dark movie."""
    from . import scmos
    from .io import load_camera_calibration, load_movie

    picasso_logo()
    print("sCMOS camera calibration check")
    print("------------------------------------------")

    calibration = load_camera_calibration(args.calibration)
    test_movie, _ = load_movie(args.movie)
    report = scmos.validate_calibration(
        calibration, test_movie, progress="console"
    )

    print(f"Frames tested:     {report['Frames']}")
    print(f"Pixels tested:     {report['Pixels tested']}")
    print(f"Mean p-value:      {report['mean p-value']:.3f} (want 0.5 +- 0.1)")
    print(
        f"Tails:             {100 * report['fraction p < 0.05']:.1f}% below "
        f"0.05, {100 * report['fraction p > 0.95']:.1f}% above 0.95"
    )
    if report["valid"]:
        print("The calibration still describes this camera.")
    else:
        direction = "noisier" if report["mean p-value"] < 0.5 else "quieter"
        print(
            f"The camera looks {direction} than when it was characterized. "
            "Sensor temperature, readout mode and bit depth all change the "
            "maps; recalibrate before trusting the noise model."
        )


def _parse_photon_ratios(args: argparse.Namespace):
    """Parse optional candidate per-channel photon ratios for ratiometric
    color assignment: "0.7,0.3;0.4,0.6" -> [[0.7, 0.3], [0.4, 0.6]]."""
    if not getattr(args, "photon_ratios", None):
        return None
    ratios = [
        [float(v) for v in row.split(",")]
        for row in args.photon_ratios.split(";")
        if row.strip()
    ]
    print(f"  ratiometric: {len(ratios)} candidate ratios")
    return ratios


def _parse_split_fov_regions(split_fov: str) -> list:
    """Parse --split-fov regions: "y0,x0,y1,x1;y0,x0,y1,x1;..." ->
    [[[y0,x0],[y1,x1]], ...]."""
    regions = []
    for row in split_fov.split(";"):
        if not row.strip():
            continue
        v = [int(t) for t in row.split(",")]
        if len(v) != 4:
            raise ValueError(
                "Each --split-fov region needs 4 ints y0,x0,y1,x1; got "
                f"'{row}'."
            )
        regions.append([[v[0], v[1]], [v[2], v[3]]])
    return regions


def _spline_calibrate_split_fov(
    args: argparse.Namespace, files, camera_info, registration, out_path
) -> dict:
    """Calibrate from a single movie whose channels are rectangular FOV
    regions."""
    from . import spline
    from .io import load_movie

    regions = _parse_split_fov_regions(args.split_fov)
    print(f"Split-FOV calibration from {len(regions)} regions of one movie")
    movie, info = load_movie(files[0])
    camera_info = _spline_calibrate_camera_info(
        camera_info, args, movie, info, files[0]
    )
    return spline.calibrate_spline_split_fov(
        movie,
        info=info,
        camera_info=camera_info,
        box=args.box_side_length,
        identification_method=_identification_method_from_args(
            args, args.gradient
        ),
        d=args.step,
        regions=regions,
        reference=getattr(args, "reference", 0) or 0,
        frames_per_step=args.frames_per_step,
        frame_order=args.frame_order,
        z_binning=args.z_binning,
        magnification_factor=args.magnification_factor,
        correct_z_bias=args.correct_z_bias,
        photon_ratios=_parse_photon_ratios(args),
        model=registration,
        path=out_path,
        progress="console",
    )


def _spline_calibrate_single(
    args: argparse.Namespace, files, camera_info, out_path
) -> dict:
    """Calibrate from a single-channel bead z-stack movie."""
    from . import spline
    from .io import load_movie

    movie, info = load_movie(files[0])
    camera_info = _spline_calibrate_camera_info(
        camera_info, args, movie, info, files[0]
    )
    return spline.calibrate_spline(
        movie,
        info=info,
        camera_info=camera_info,
        box=args.box_side_length,
        identification_method=_identification_method_from_args(
            args, args.gradient
        ),
        d=args.step,
        frames_per_step=args.frames_per_step,
        frame_order=args.frame_order,
        z_binning=args.z_binning,
        model=args.model,
        magnification_factor=args.magnification_factor,
        correct_z_bias=args.correct_z_bias,
        path=out_path,
        progress="console",
    )


def _spline_calibrate_multichannel(
    args: argparse.Namespace, files, camera_info, registration, out_path
) -> dict:
    """Calibrate from several single-channel bead z-stack movies."""
    from . import spline
    from .io import load_movie

    print(f"Multichannel calibration from {len(files)} channels")
    movies, infos, camera_infos = [], [], []
    for f in files:
        movie, info = load_movie(f)
        movies.append(movie)
        infos.append(info)
        camera_infos.append(
            _spline_calibrate_camera_info(camera_info, args, movie, info, f)
        )
    return spline.calibrate_spline_multichannel(
        movies,
        infos=infos,
        camera_infos=camera_infos,
        box=args.box_side_length,
        identification_method=_identification_method_from_args(
            args, args.gradient
        ),
        d=args.step,
        frames_per_step=args.frames_per_step,
        frame_order=args.frame_order,
        z_binning=args.z_binning,
        magnification_factor=args.magnification_factor,
        correct_z_bias=args.correct_z_bias,
        photon_ratios=_parse_photon_ratios(args),
        model=registration,
        path=out_path,
        progress="console",
    )


def _spline_calibrate_camera_info(
    camera_info: dict, args: argparse.Namespace, movie, info, path: str
) -> dict:
    """``camera_info`` with the camera pixel size of ``movie`` resolved
    as ``picasso localize`` does (see :func:`_resolve_camera_pixelsize`)."""
    pixelsize = _resolve_camera_pixelsize(args.pixelsize, movie, info, path)
    print(f"Camera pixel size: {pixelsize:g} nm")
    return {**camera_info, "Pixelsize": pixelsize}


def _spline_calibrate(args: argparse.Namespace) -> None:
    """Build a cubic-spline PSF calibration from a bead z-stack movie."""
    from os.path import splitext
    from . import spline

    picasso_logo()
    print("Spline PSF calibration")
    print("------------------------------------------")

    # the camera pixel size is resolved once the movie is loaded
    camera_info = {
        "Baseline": args.baseline,
        "Sensitivity": args.sensitivity,
        "Gain": args.gain,
    }
    registration = getattr(args, "registration_model", None) or "affine"
    files = args.files
    if args.output:
        out_path = args.output
    else:
        base, _ = splitext(files[0])
        out_path = base + "_spline_calib.hdf5"

    split_fov = getattr(args, "split_fov", None)
    if split_fov and len(files) == 1:
        calibration = _spline_calibrate_split_fov(
            args, files, camera_info, registration, out_path
        )
    elif len(files) == 1:
        calibration = _spline_calibrate_single(
            args, files, camera_info, out_path
        )
    else:
        calibration = _spline_calibrate_multichannel(
            args, files, camera_info, registration, out_path
        )

    print("------------------------------------------")
    n_beads = calibration["n_beads"]
    n_used = spline.n_beads_used(calibration)
    built_from = f"{n_beads} beads"
    if n_used < n_beads:
        # the rejected beads are not in the PSF; the gallery next to the
        # calibration shows which ones were dropped and why
        built_from = (
            f"{n_used} of {n_beads} detected beads "
            f"({n_beads - n_used} rejected as outliers, see the "
            "*_beads.png diagnostic)"
        )
    print(
        f"Spline PSF calibration built from {built_from} "
        f"and saved to {out_path}"
    )


def _lateral_calibrate(args: argparse.Namespace) -> None:
    """Fit a lateral (astigmatism / chromatic) correction from two bead
    images and append it to a calibration file."""
    from . import localize
    from .io import load_any_calibration, load_movie, save_any_calibration

    picasso_logo()
    print("Lateral transform calibration")
    print("------------------------------------------")

    movie_ref, _ = load_movie(args.reference)
    movie_target, _ = load_movie(args.target)
    calibration = (
        load_any_calibration(args.calibration, pixelsize=args.pixelsize)
        if args.calibration
        else {}
    )
    out_path = args.output or args.calibration
    if not out_path:
        raise Exception(
            "Pass --calibration (to append to an existing calibration) or "
            "--output (to write a standalone lateral calibration)."
        )

    calibration, qc = localize.fit_lateral_transform(
        movie_ref,
        movie_target,
        calibration=calibration,
        box=args.box_side_length,
        identification_method=_identification_method_from_args(
            args, args.gradient
        ),
        pixelsize=args.pixelsize,
        transform_type=args.type,
        ref_path=args.reference,
        target_path=args.target,
        model=args.model,
    )
    if args.plot:
        localize.plot_lateral_calibration(qc, save_path=args.plot)
    save_any_calibration(out_path, calibration)
    print("------------------------------------------")
    from . import lib

    print(
        f"{args.type} correction fitted on {qc['n_pairs']} bead pairs "
        f"and saved to {out_path}"
    )
    print("  " + ", ".join(lib.describe_lateral_transforms(calibration)))


def _render_many(
    locs,
    info,
    path,
    disp_px_size,
    blur_method,
    min_blur_width,
    vmin,
    vmax,
    scaling,
    cmap,
    silent,
):
    import sys
    from os.path import splitext
    from matplotlib.pyplot import imsave
    from .render import render

    if sys.platform == "win32":
        from os import startfile

    if blur_method == "none":
        blur_method = None
    N, image = render(
        locs,
        info,
        disp_px_size=disp_px_size,
        blur_method=blur_method,
        min_blur_width=min_blur_width,
    )
    base, ext = splitext(path)
    out_path = base + ".png"
    im_max = image.max() / 100
    if scaling == "yes":
        imsave(
            out_path,
            image,
            vmin=vmin * im_max,
            vmax=vmax * im_max,
            cmap=cmap,
        )
    else:
        imsave(out_path, image, vmin=vmin, vmax=vmax, cmap=cmap)
    if not silent and sys.platform == "win32":
        startfile(out_path)


def _render(args: argparse.Namespace) -> None:
    """Render localization files to images.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``picasso render`` arguments:

        - ``files`` (str): path to the localization files or a directory
          containing HDF5 files.
        - ``disp_px_size`` (float): size of the rendered pixel in nm.
        - ``blur_method`` (str): localizations' blur, one of
          ``'gaussian'``, ``'gaussian_iso'``, ``'smooth'``,
          ``'convolve'``. If None, no blurring is applied.
        - ``min_blur_width`` (float): minimum width of the blur kernel
          in pixels.
        - ``vmin``, ``vmax`` (float): minimum and maximum of the color
          scale.
        - ``cmap`` (str): colormap to use for rendering. If None, the
          colormap from user settings is used.
        - ``scaling`` (str): if ``'yes'``, the image is scaled to the
          range [vmin, vmax]; if ``'no'``, it is not scaled.
        - ``silent`` (bool): if True, the rendered images are not opened
          automatically.
    """
    from .lib import locs_glob_map
    from os.path import isdir
    from .io import load_user_settings, save_user_settings
    from tqdm import tqdm
    from glob import glob

    settings = load_user_settings()
    cmap = args.cmap
    if cmap is None:
        # .get: indexing the settings creates a missing entry (an empty
        # dict) instead of raising, e.g., without a settings file
        cmap = settings["Render"].get("Colormap")
        if not isinstance(cmap, str):
            cmap = "viridis"
    settings["Render"]["Colormap"] = cmap
    save_user_settings(settings)

    if isdir(args.files):
        print("Analyzing folder")
        paths = glob(args.files + "/*.hdf5")
        print("A total of {} files detected. Rendering.".format(len(paths)))

        for path in tqdm(paths):
            locs_glob_map(
                _render_many,
                path,
                args=(
                    args.disp_px_size,
                    args.blur_method,
                    args.min_blur_width,
                    args.vmin,
                    args.vmax,
                    args.scaling,
                    cmap,
                    True,
                ),
            )

    else:
        locs_glob_map(
            _render_many,
            args.files,
            args=(
                args.disp_px_size,
                args.blur_method,
                args.min_blur_width,
                args.vmin,
                args.vmax,
                args.scaling,
                cmap,
                args.silent,
            ),
        )


def _g5m(
    files: str,
    min_locs: int = 10,
    loc_prec_handle: Literal["local", "abs"] = "local",
    min_sigma: float = 0.8,
    max_sigma: float = 1.5,
    max_rounds: int = 3,
    bootstrap_sem: bool = False,
    calibration: str = "",
    mode: Literal["astigmatism", "spline"] = "astigmatism",
    covariance_type: str = "auto",
    postprocess: bool = True,
    max_locs: int = 100000,
    parallel: bool = True,
    group_column: Literal["group", "group_input"] = "group",
) -> None:
    """G5M analysis of clustered localizations. See ``picasso.g5m.g5m``
    for details on the parameters."""
    from glob import glob
    from os.path import isdir, splitext
    from .io import load_calibration, load_locs, save_locs
    from .g5m import g5m
    from .lib import get_from_metadata

    if isdir(files):
        print("Analyzing folder")
        paths = glob(files + "/*.hdf5")
    else:
        paths = [files]

    n = len(paths)
    print(f"A total of {n} file{'s' if n > 1 else ''} detected.")
    for path in paths:
        print("------------------------------------------")
        print(f"Processing {path}")
        locs, info = load_locs(path)
        calib = None
        # astigmatism 3D data needs a calibration; spline 3D data
        # recovers z directly and needs none
        if "z" in locs.columns and mode == "astigmatism":
            if calibration == "":
                raise ValueError(
                    "A calibration file (-c/--calibration) is required "
                    "for astigmatism 3D data."
                )
            # a calibration written before Picasso 1.0 (camera pixels) is
            # converted with the pixel size of the localizations
            calib = load_calibration(
                calibration, pixelsize=get_from_metadata(info, "Pixelsize")
            )
        mols, g5m_info, _ = g5m(
            locs,
            info,
            min_locs=min_locs,
            loc_prec_handle=loc_prec_handle,
            sigma_bounds=(min_sigma, max_sigma),
            max_rounds_without_best_bic=max_rounds,
            bootstrap_check=bootstrap_sem,
            calibration=calib,
            mode=mode,
            covariance_type=covariance_type,
            postprocess=postprocess,
            max_locs_per_cluster=max_locs,
            parallel=parallel,
            group_column=group_column,
            progress="console",
        )
        new_path = splitext(path)[0] + "_molmap.hdf5"
        save_locs(new_path, mols, g5m_info)


# =============================================================================
# Plugins
# =============================================================================


def _convert_units(
    files: list[str], pixelsize: float | None, in_place: bool
) -> None:
    """Convert files written before Picasso 1.0 from camera pixels to
    nm, see ``legacy.convert_files``. Prints one line per file: the
    converted file, or why the file was skipped or failed."""
    from glob import glob
    from itertools import chain
    from . import legacy

    paths = list(chain(*[sorted(glob(_)) or [_] for _ in files]))
    results = legacy.convert_files(
        paths, pixelsize, in_place=in_place, progress="console"
    )
    n_converted = 0
    for path, output, message in results:
        if output is not None:
            n_converted += 1
            print(f"{path}: converted, saved as {output}")
        else:
            print(f"{path}: skipped ({message})")
    print(f"Converted {n_converted} of {len(results)} files.")


def _run_plugin_command(args) -> None:
    """Run the handler a plugin subcommand named via ``set_defaults``.

    Kept separate from the built-in dispatch so that a plugin that forgot
    the ``func`` default gets a clear message instead of an
    ``AttributeError`` traceback.
    """
    func = getattr(args, "func", None)
    if func is None:
        print(
            f"The plugin command '{args.command}' does not name a handler. "
            "The plugin must call parser.set_defaults(func=...) when it "
            "registers the command."
        )
        return
    func(args)


def _plugins_confirm(question: str, assume_yes: bool) -> bool:
    """Ask for confirmation on the terminal unless ``--yes`` was given."""
    if assume_yes:
        return True
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


_PLUGIN_TRUST_WARNING = (
    "Plugins are Python files that run with full access to your computer "
    "every time Picasso starts. Only enable files you have reviewed or "
    "whose author you trust."
)


def _plugins_file_origin(name: str, by_file: dict) -> str:
    """Describe where a plugin file came from: local, or registry+version."""
    installed = by_file.get(name)
    if installed is None:
        return "local file"
    pid, record = installed
    version = record.get("version") or "?"
    return f"registry: {pid} {version}"


def _plugins_module_details(module) -> list:
    """Summarize what an enabled plugin module contributes."""
    from . import plugins

    details = []
    if getattr(module, "Plugin", None) is not None:
        details.append("GUI menu entry")
    commands = plugins.plugin_cli_commands(module)
    if commands:
        details.append("commands: " + ", ".join(commands))
    try:
        exports = plugins._api_names(module)
    except Exception:  # noqa: BLE001
        exports = {}
    if exports:
        details.append("API: " + ", ".join(sorted(exports)))
    return details


def _plugins_describe_file(path: str, state: dict, by_file: dict) -> bool:
    """Print one plugin file's status and, if enabled, what it contributes.

    Returns True if the file is enabled but failed to load.
    """
    from . import plugins

    name = os.path.basename(path)
    enabled = plugins.is_enabled(state, name)
    origin = _plugins_file_origin(name, by_file)
    print(f"  {'[x]' if enabled else '[ ]'} {name}  ({origin})")
    if not enabled:
        return False

    # Only an enabled plugin may be imported, so only an enabled one
    # can be asked what it provides.
    try:
        module = plugins._load_module_from_path(path)
    except Exception:  # noqa: BLE001 - reported, not fatal
        import traceback

        print("        failed to load:")
        for line in traceback.format_exc().rstrip().splitlines():
            print(f"          {line}")
        return True

    for detail in _plugins_module_details(module):
        print(f"        {detail}")
    return False


def _plugins_list() -> None:
    """Print every plugin file found, with what it contributes."""
    from . import io, plugins

    directory = io.plugins_directory()
    state = plugins.load_state()
    files = plugins._discover_plugin_files()
    if not files:
        print(f"No plugin files in {directory}")
        return

    by_file = {
        record.get("file"): (pid, record)
        for pid, record in state["plugins"].items()
        if record.get("file")
    }

    print(f"Plugins in {directory}:\n")
    failed = False
    for path in files:
        if _plugins_describe_file(path, state, by_file):
            failed = True

    disabled = plugins.disabled_plugin_files(state)
    if disabled:
        print(
            f"\n{len(disabled)} file(s) not enabled. Review one, then run "
            "'picasso plugins enable <file.py>'."
        )
    if failed:
        print(
            "\nA plugin that fails to load is skipped; the rest of Picasso "
            "is unaffected. Fix the file, or run 'picasso plugins disable "
            "<file.py>' to stop loading it."
        )


def _plugins_set_enabled(filename: str, value: bool, assume_yes: bool) -> None:
    """Enable or disable one plugin file by name."""
    from . import plugins

    if not plugins.is_safe_filename(filename):
        print(f"Not a plugin file name: {filename!r} (expected e.g. 'my.py')")
        return
    path = plugins.plugin_path(filename)
    if not os.path.exists(path):
        print(f"No such plugin file: {path}")
        return

    if value:
        print(_PLUGIN_TRUST_WARNING)
        print(f"\nThe file is at {path}")
        if not _plugins_confirm(f"Enable '{filename}'?", assume_yes):
            print("Nothing was changed.")
            return

    state = plugins.load_state()
    plugins.set_enabled(state, filename, value)
    print(f"{'Enabled' if value else 'Disabled'} {filename}")


def _plugins_install(plugin_id: str, assume_yes: bool) -> None:
    """Download, verify and enable a plugin from the online registry."""
    from . import plugins

    try:
        manifest = plugins.fetch_manifest()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        print(f"Could not reach the online plugin registry: {exc}")
        return
    entry = next((e for e in manifest if e["id"] == plugin_id), None)
    if entry is None:
        print(
            f"No plugin with id {plugin_id!r} in the registry. "
            f"Available: {', '.join(sorted(e['id'] for e in manifest))}"
        )
        return
    if not plugins.is_compatible(entry):
        print(
            f"{plugin_id!r} requires Picasso "
            f"{entry.get('min_picasso_version')} or newer."
        )
        return

    print(f"{entry.get('display_name', plugin_id)} {entry.get('version', '')}")
    if entry.get("description"):
        print(entry["description"])
    print(f"\n{_PLUGIN_TRUST_WARNING}")
    print(
        "The download is checked against the hash published in the registry, "
        "but that only proves the file is the published one — not that it is "
        f"safe. Source: {plugins.REPO_URL}/blob/{plugins.BRANCH}/"
        f"{entry['file']}"
    )
    if not _plugins_confirm(f"Install and enable {plugin_id!r}?", assume_yes):
        print("Nothing was installed.")
        return

    state = plugins.load_state()
    try:
        plugins.install(entry, state)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        print(f"Install failed: {exc}")
        return
    print(f"Installed and enabled {plugin_id!r}.")


def _plugins_uninstall(plugin_id: str) -> None:
    """Delete an installed plugin's file and forget it."""
    from . import plugins

    state = plugins.load_state()
    if plugin_id not in state["plugins"]:
        print(f"{plugin_id!r} is not installed from the registry.")
        return
    plugins.uninstall(plugin_id, state)
    print(f"Uninstalled {plugin_id!r}.")


def _plugins(args) -> None:
    """Dispatch the ``picasso plugins`` subcommands."""
    from . import io, plugins

    action = args.plugins_action
    if action == "list":
        _plugins_list()
    elif action == "path":
        print(io.plugins_directory())
    elif action == "enable":
        _plugins_set_enabled(args.file, True, args.yes)
    elif action == "disable":
        _plugins_set_enabled(args.file, False, args.yes)
    elif action in ("install", "update"):
        _plugins_install(args.id, args.yes)
    elif action == "uninstall":
        _plugins_uninstall(args.id)
    else:
        print(
            "Usage: picasso plugins {list,path,enable,disable,install,"
            "update,uninstall}"
        )
        print(f"\nPlugins folder: {io.plugins_directory()}")
        print(f"Online registry: {plugins.REPO_URL}")


#: Commands renamed in Picasso 1.0 to the style of the other commands
#: (words joined by hyphens); the old names keep working, with a
#: warning, until Picasso 2.0. See docs/versioning.rst.
_RENAMED_COMMANDS = {
    old: (
        new,
        deprecation.register(
            f"picasso {old}",
            since="1.0",
            remove_in="2.0",
            alternative=f"picasso {new}",
        ),
    )
    for old, new in {
        "undrift_fiducials": "undrift-fiducials",
        "nneighbor": "nearest-neighbor",
        "clusterfilter": "cluster-filter",
        "smlm_cluster": "smlm-cluster",
        "groupprops": "group-properties",
        "pc": "pair-correlation",
    }.items()
}


#: Options renamed in Picasso 1.0, per (current) command name: words
#: joined by hyphens, named after the Python parameter they set where
#: there is one (e.g., ``--r-max`` for ``r_max``). The old names keep
#: working, with a warning, until Picasso 2.0.
_RENAMED_OPTIONS = {
    command: {
        old: (
            new,
            deprecation.register(
                f"picasso {command} {old}",
                since="1.0",
                remove_in="2.0",
                alternative=new,
            ),
        )
        for old, new in options.items()
    }
    for command, options in {
        "localize": {
            "--zc": "--z-calibration",
            "--mf": "--magnification-factor",
        },
        "undrift": {"--fromfile": "--from-file"},
        "aim": {
            "--intersectdist": "--intersect-distance",
            "--roiradius": "--roi-radius",
        },
        "cluster-filter": {
            "--clusterfile": "--cluster-file",
            "--minval": "--min-value",
            "--maxval": "--max-value",
        },
        "dbscan": {"--radius_z": "--radius-z"},
        "join": {"--keepindex": "--keep-index"},
        "pair-correlation": {"--binsize": "--bin-size", "--rmax": "--r-max"},
        "spinna": {"--asynch": "--no-parallel"},
        "g5m": {"--asynch": "--no-parallel"},
    }.items()
}


def _current_names(argv: list[str]) -> list[str]:
    """``argv`` (without the program name) with the command and option
    names renamed in Picasso 1.0 replaced by the current ones.

    A notice is printed for every old name. Arguments after ``--`` are
    left alone.

    Parameters
    ----------
    argv : list of str
        The command line arguments, e.g. ``sys.argv[1:]``.

    Returns
    -------
    argv : list of str
        The arguments with the current names.
    """
    argv = list(argv)
    if not argv:
        return argv
    command = argv[0]
    if command in _RENAMED_COMMANDS:
        command, notice = _RENAMED_COMMANDS[command]
        _print_deprecation(notice)
        argv[0] = command
    renamed = _RENAMED_OPTIONS.get(command, {})
    for i in range(1, len(argv)):
        if argv[i] == "--":
            break
        name, equals, value = argv[i].partition("=")
        if name in renamed:
            new, notice = renamed[name]
            _print_deprecation(notice)
            argv[i] = new + equals + value
    return argv


def _print_deprecation(notice: deprecation.Deprecation) -> None:
    """Print the notice of a deprecated command line name."""
    import sys

    print(f"Warning: {notice.text()}", file=sys.stderr)


def _build_parser() -> (
    tuple[argparse.ArgumentParser, argparse._SubParsersAction]
):
    """The parser of the ``picasso`` command line and its subcommands
    (the action plugins add their commands to, see ``main``).

    Built apart from ``main`` so that tests can inspect the commands
    and options, see ``tests/test_stability.py``.
    """
    # Main parser
    parser = argparse.ArgumentParser("picasso")
    subparsers = parser.add_subparsers(dest="command")

    # localize
    localize_parser = subparsers.add_parser(
        "localize", help="identify and fit single molecule spots"
    )
    localize_parser.add_argument(
        "files",
        nargs="?",
        help=(
            "one movie file or a folder containing movie files"
            " specified by a unix style path pattern"
        ),
    )
    localize_parser.add_argument(
        "--concat",
        action="store_true",
        help=(
            "treat all the TIFF movies found as a single movie, with"
            " their frames concatenated in order (folder: searched"
            " recursively; pattern: as matched), instead of localizing"
            " each file separately. Results are saved next to the first"
            " file"
        ),
    )
    localize_parser.add_argument(
        "-b", "--box-side-length", type=int, default=7, help="box side length"
    )
    localize_parser.add_argument(
        "-a",
        "--fit-method",
        choices=[
            "mle",
            "mle-gpu",
            "mle-spherical",
            "mle-spherical-gpu",
            "mle-rotated",
            "mle-rotated-gpu",
            "lq",
            "lq-spherical",
            "lq-spherical-gpu",
            "lq-rotated",
            "lq-rotated-gpu",
            "lq-gpu",
            "lq-3d",
            "lq-gpu-3d",
            "mle-3d",
            "spline",
            "spline-mle",
            "spline-gpu",
            "spline-mle-gpu",
            "avg",
            "vectorfit",
            "vectorfit-gpu",
            "phasor",
            "phasor-3d",
        ],
        action="append",
        default=None,
        help=(
            "fitting method (default 'mle'). 'spline'/'spline-mle' fit an "
            "experimental cubic-spline PSF on the CPU by least squares / "
            "maximum likelihood, 'spline-gpu'/'spline-mle-gpu' do the same "
            "on the GPU (needs a CUDA-capable GPU); all four need "
            "--spline-calibration. 'vectorfit' fits a vectorial PSF model "
            "computed from the optics (see the --vf-* options) with the "
            "optional VectorFit library (pip install picassosr[vectorfit]), "
            "'vectorfit-gpu' does the same on the GPU with VectorFit's GPU "
            "build (pip install vectorfit-py-cu12), which, despite its "
            "name, needs CUDA 13 (NVIDIA driver 580 or newer). 'phasor' "
            "localizes by phasors (Martens et al. 2018; non-iterative, CPU; "
            "its lpx/lpy are Picasso's error-propagation estimate, not part "
            "of the published method), 'phasor-3d' adds z from an "
            "astigmatic phasor calibration (-zc). "
            "With --regions-separately it may be given once per --roi, to "
            "fit each region its own way"
        ),
    )
    _add_vectorfit_args(localize_parser)
    localize_parser.add_argument(
        "-sc",
        "--spline-calibration",
        type=str,
        action="append",
        default=None,
        help=(
            "path to a cubic-spline PSF calibration (.hdf5) for the "
            "'spline', 'spline-mle', 'spline-gpu' and 'spline-mle-gpu' "
            "fit methods. With --regions-separately it may be given once "
            "per --roi: the PSF is measured per channel, so regions "
            "fitted separately usually need one calibration each"
        ),
    )
    localize_parser.add_argument(
        "-ac",
        "--affine-calibration",
        type=str,
        action="append",
        default=[],
        help=(
            "path to a calibration file (.yaml or .hdf5) whose lateral"
            " corrections are applied to the fitted x/y, e.g. a standalone"
            " chromatic-aberration calibration. The transform model stored in"
            " the file (affine, projective or polynomial) is used as saved."
            " Repeat the flag to chain"
            " several; they are applied in the order given, after any"
            " corrections stored in the 3D or spline calibration itself."
            " A correction the 3D or spline calibration already carries is"
            " skipped, so the same file can be passed to both flags"
        ),
    )
    localize_parser.add_argument(
        "-cm",
        "--camera-calibration",
        type=str,
        default="",
        help=(
            "path to a per-pixel sCMOS camera calibration (.hdf5) from "
            "'picasso camera-calibrate'. Its offset map replaces --baseline "
            "and, if it has one, its gain map replaces --sensitivity; "
            "maximum-likelihood fits then use the sCMOS noise model of "
            "Huang et al. Nature Methods, 2013."
        ),
    )
    localize_parser.add_argument(
        "-g",
        "--gradient",
        type=_gradient_arg,
        action="append",
        default=None,
        help=(
            "minimum net gradient (default 5000), or 'auto' to estimate it "
            "for each movie from its background (the local maxima of 40 "
            "frames; see localize.estimate_min_net_gradient). May be given "
            "once per --roi, since regions imaged through different optics "
            "need not share a brightness scale; an 'auto' is then estimated "
            "in its own region"
        ),
    )
    localize_parser.add_argument(
        "-rs",
        "--regions-separately",
        action="store_true",
        help=(
            "fit each --roi region on its own and save one file per region "
            "(<movie>_ref_locs.hdf5, <movie>_ch1_locs.hdf5, ...), in that "
            "region's own coordinates. Use it when the regions are channels "
            "imaged side by side on one sensor and are to be analyzed - and "
            "connected - as separate channels. --fit-method, --gradient and "
            "--spline-calibration may then be given once per region"
        ),
    )
    localize_parser.add_argument(
        "-tm",
        "--temporal-median",
        type=int,
        default=0,
        help=(
            "window length (in frames) of the temporal median filter applied"
            " before spot identification; it subtracts a per-pixel rolling"
            " median background, which suppresses inhomogeneous background"
            " and static structures. 0 (the default) disables it. Fitting"
            " always uses the raw movie, so the minimum net gradient needs"
            " re-tuning when this is switched on"
        ),
    )
    localize_parser.add_argument(
        "-gf",
        "--gaussian-filter",
        type=float,
        default=0.0,
        help=(
            "standard deviation (in camera pixels) of a spatial Gaussian"
            " filter applied before spot identification; it merges the"
            " several local maxima of a non-Gaussian spot into one, so the"
            " spot is easier to identify. 0 (the default) disables it. "
            "Fitting always uses the raw movie, so the minimum net gradient "
            "needs re-tuning when this is changed"
        ),
    )
    _add_identification_method_args(localize_parser)
    localize_parser.add_argument(
        "-cc",
        "--convergence",
        type=float,
        default=0,
        help=(
            "convergence criterion: the fit stops once the chi-square"
            " changes by less than this, relative to its own magnitude."
            " 0 (the default) uses the selected fit method's own value"
            " (1e-5 for 'mle', 0.01 for 'lq' and every GPU Gaussian, and"
            " for the splines 1e-4 with the axial multi-start, 1e-2"
            " without)"
        ),
    )
    localize_parser.add_argument(
        "-mi",
        "--max-iterations",
        type=int,
        default=0,
        help=(
            "maximum number of iterations per spot. 0 (the default) uses"
            " the selected fit method's own value (100 for 'mle', 200 for"
            " 'lq', 20 for every GPU Gaussian, and for the splines 100 with"
            " the axial multi-start, 20 without)"
        ),
    )
    localize_parser.add_argument(
        "-d",
        "--drift",
        type=int,
        default=1000,
        help="segmentation size for subsequent RCC, 0 to deactivate",
    )
    localize_parser.add_argument(
        "-r",
        "--roi",
        type=int,
        nargs=4,
        action="append",
        default=None,
        help=(
            "ROI (y_min, x_min, y_max, x_max) in camera pixels; note the\n"
            "origin of the image is in the top left corner. May be given\n"
            "multiple times to analyze several regions, e.g.\n"
            "--roi 10 10 100 100 --roi 200 200 300 300. Overlapping ROIs\n"
            "are corrected automatically so that they do not overlap."
        ),
    )
    localize_parser.add_argument(
        "-fb",
        "--frame-bounds",
        type=int,
        nargs=2,
        action="append",
        default=None,
        help=(
            "frame bounds (start_frame, end_frame), 0-indexed and\n"
            "inclusive. May be given multiple times to analyze several\n"
            "disjoint frame segments, e.g.\n"
            "--frame-bounds 0 100 --frame-bounds 200 300."
        ),
    )
    localize_parser.add_argument(
        "--frame-sum-window",
        type=int,
        default=1,
        help=(
            "photon-counting movies (.ptu) only: number of consecutive\n"
            "scan frames summed into one frame, for confocal scans too\n"
            "dim to localize scan frame by scan frame. Default 1 (no\n"
            "summing)"
        ),
    )
    localize_parser.add_argument(
        "--frame-sum-stride",
        type=int,
        default=None,
        help=(
            "photon-counting movies (.ptu) only: number of scan frames\n"
            "between the starts of two sums, at most --frame-sum-window.\n"
            "Default: --frame-sum-window, i.e., sums that do not overlap\n"
            "(every photon is used once); 1 gives a sliding window"
        ),
    )
    localize_parser.add_argument(
        "--microtime-gate",
        type=float,
        nargs=2,
        action="append",
        default=None,
        metavar=("START", "END"),
        help=(
            "photon-counting movies (.ptu) only: a microtime gate, i.e.,\n"
            "photons arriving START to END ns after the sync (laser)\n"
            "pulse, e.g., one gate per laser of pulsed interleaved\n"
            "excitation (PIE). May be given several times; each detector\n"
            "then gives one channel per gate. A .ptu file with several\n"
            "channels (detectors or gates) is localized channel by\n"
            "channel, into <movie>_<channel>_locs.hdf5"
        ),
    )
    localize_parser.add_argument(
        "-bl", "--baseline", type=int, default=0, help="camera baseline"
    )
    localize_parser.add_argument(
        "-s", "--sensitivity", type=float, default=1, help="camera sensitivity"
    )
    localize_parser.add_argument(
        "-ga", "--gain", type=int, default=1, help="camera gain"
    )
    localize_parser.add_argument(
        "-qe",
        "--qe",
        type=float,
        default=1,
        # accepted so that existing scripts keep working, but ignored: the
        # quantum efficiency has not been used since Picasso 0.6.0 and was
        # finally completely removed in Picasso 1.0.0
        help=argparse.SUPPRESS,
    )
    localize_parser.add_argument(
        "-mf",
        "--magnification-factor",
        dest="mf",
        type=float,
        default=0,
        help="magnification factor (3D only)",
    )
    localize_parser.add_argument(
        "-px",
        "--pixelsize",
        type=float,
        default=None,
        help=(
            "camera pixel size in nm; the localizations are saved in nm."
            " Default: the movie metadata, else the camera configuration"
        ),
    )
    localize_parser.add_argument(
        "-zc",
        "--z-calibration",
        dest="zc",
        type=str,
        default="",
        help=(
            "path to 3D calibration file (3D only): a Gaussian astigmatism "
            "calibration for the '-3d' Gaussian methods, an astigmatic "
            "phasor calibration for 'phasor-3d'"
        ),
    )
    localize_parser.add_argument(
        "-zg",
        "--fit-z-gpu",
        action="store_true",
        help=(
            "fit z coordinates on a CUDA-capable GPU (numba.cuda);" " 3D only"
        ),
    )

    localize_parser.add_argument(
        "-sf",
        "--suffix",
        type=str,
        default="",
        help="suffix to add to output files",
    )

    localize_parser.add_argument(
        "-db",
        "--database",
        action="store_true",
        help="add the run to the local database",
    )
    multi_emitter = localize_parser.add_argument_group(
        "multi-emitter fitting",
        "fit all emitters of a frame at once, so that emitters closer than"
        " about 2 PSF sigma are localized separately (3D-DAOSTORM,"
        " Babcock et al., Optical Nanoscopy 2012). The peaks are found in"
        " the residual of the fit, so the identification options are not"
        " used, except with --seed-identification",
    )
    multi_emitter.add_argument(
        "-me",
        "--multi-emitter",
        action="store_true",
        help="fit overlapping emitters instead of one emitter per box",
    )
    multi_emitter.add_argument(
        "--psf-model",
        choices=["fixed", "spherical", "elliptical", "spline"],
        default="fixed",
        help=(
            "the PSF of each emitter: a Gaussian whose sigma is fixed, the"
            " same for all (default), spherical, one fitted sigma per"
            " emitter, or elliptical, a fitted sigma in x and one in y per"
            " emitter, whose z --z-calibration then gives (storm-analysis"
            " models 2dfixed, 2d and 3d); or spline, the PSF of the"
            " --spline-calibration, which gives z directly for a 3D one"
        ),
    )
    multi_emitter.add_argument(
        "--snr-threshold",
        type=float,
        default=15.0,
        help=(
            "peak threshold, in units of the signal-to-noise ratio, and the"
            " minimum significance of an emitter (default 15)"
        ),
    )
    multi_emitter.add_argument(
        "--psf-sigma",
        type=float,
        default=None,
        help=(
            "standard deviation of the PSF in nm: fixed in the fit, or the"
            " start of a fitted one. Default: estimated from the movie"
        ),
    )
    multi_emitter.add_argument(
        "--max-cycles",
        type=int,
        default=20,
        help=(
            "most cycles of finding, fitting and subtracting per frame"
            " (default 20)"
        ),
    )
    multi_emitter.add_argument(
        "--peak-radius",
        type=float,
        default=5.0,
        help=(
            "radius (camera pixels) within which a new peak must be the"
            " maximum of the residual (default 5)"
        ),
    )
    multi_emitter.add_argument(
        "--background-sigma",
        type=float,
        default=8.0,
        help=(
            "sigma (camera pixels) of the Gaussian low pass that estimates"
            " the background (default 8)"
        ),
    )
    multi_emitter.add_argument(
        "--peak-search-z",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "z (nm) at which --psf-model spline with a 3D calibration"
            " filters the residual to find peaks; a new emitter starts at"
            " the z where it is brightest (default 0)"
        ),
    )
    multi_emitter.add_argument(
        "--fit-region-size",
        type=int,
        default=None,
        help=(
            "side length (camera pixels) of the disc each emitter is fitted"
            " in with --psf-model spline, cropped from the calibration: at"
            " most its box less 2. Default: from the calibration's PSF"
            " sigma, 8 sigma (12 for a 3D calibration)"
        ),
    )
    multi_emitter.add_argument(
        "--seed-identification",
        action="store_true",
        help=(
            "start the fit of each frame from the spots the identification"
            " options find (--gradient, --box-side-length, the wavelet and"
            " filter options), instead of the peak finder's first cycle"
        ),
    )

    # camera-calibrate: per-pixel sCMOS offset, variance and gain maps
    camera_calib_parser = subparsers.add_parser(
        "camera-calibrate",
        help=(
            "characterize an sCMOS camera: per-pixel offset, readout "
            "variance and (optionally) gain maps"
        ),
    )
    camera_calib_parser.add_argument(
        "dark",
        help=(
            "dark movie: frames recorded with no light on the sensor. Huang "
            "et al. (2013) used 60,000; fewer than 10,000 warns"
        ),
    )
    camera_calib_parser.add_argument(
        "-l",
        "--light",
        type=str,
        action="append",
        help=(
            "movie at a quasi-uniform illumination level, same camera "
            "settings as the dark movie; repeat for several levels (the "
            "paper used 15, spanning 20-200 photons per pixel) to measure "
            "the per-pixel gain. Omit for offset and variance only"
        ),
    )
    camera_calib_parser.add_argument(
        "-p",
        "--power",
        type=float,
        action="append",
        help=(
            "illumination each -l/--light movie was recorded at (laser "
            "power, exposure time, ...), repeated once per light movie in "
            "the same order. Not used by the gain fit; it lets the "
            "diagnostic plot show the response against what was actually "
            "set, which is the only way to judge linearity when the levels "
            "are not evenly spaced"
        ),
    )
    camera_calib_parser.add_argument(
        "--power-unit",
        type=str,
        default="mW",
        help="unit of -p/--power, for the plot axis (default: mW)",
    )
    camera_calib_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help=(
            "output calibration path (.hdf5); default alongside the dark "
            "movie"
        ),
    )

    camera_check_parser = subparsers.add_parser(
        "camera-validate",
        help="check an sCMOS calibration against a fresh dark movie",
    )
    camera_check_parser.add_argument(
        "calibration", help="camera calibration (.hdf5)"
    )
    camera_check_parser.add_argument(
        "movie",
        help="short fresh dark movie; about 1,000 frames is plenty",
    )

    # spline-calibrate: build a cubic-spline PSF calibration from a bead
    # z-stack movie for later use with the 'spline' fit methods.
    spline_calib_parser = subparsers.add_parser(
        "spline-calibrate",
        help="build a cubic-spline PSF calibration from a bead z-stack",
    )
    spline_calib_parser.add_argument(
        "files",
        nargs="+",
        help=(
            "bead z-stack movie file(s); pass several (one per channel) to "
            "build a multichannel calibration"
        ),
    )
    spline_calib_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="output calibration path (.hdf5); default alongside the movie",
    )
    spline_calib_parser.add_argument(
        "-b", "--box-side-length", type=int, default=13, help="box side length"
    )
    spline_calib_parser.add_argument(
        "-g",
        "--gradient",
        type=int,
        default=5000,
        help="minimum net gradient for bead detection",
    )
    _add_identification_method_args(spline_calib_parser, what="bead")
    spline_calib_parser.add_argument(
        "-s",
        "--step",
        type=float,
        required=True,
        help="z step size in nm between consecutive stage positions",
    )
    spline_calib_parser.add_argument(
        "-fps",
        "--frames-per-step",
        type=int,
        default=1,
        help="number of frames acquired per z position (multi-FOV)",
    )
    spline_calib_parser.add_argument(
        "-fo",
        "--frame-order",
        choices=["fov", "z"],
        default="fov",
        help="acquisition order when frames-per-step > 1",
    )
    spline_calib_parser.add_argument(
        "-zb",
        "--z-binning",
        type=int,
        default=1,
        help=(
            "number of consecutive z steps averaged into one slice of the PSF"
            " model, so its axial knots are z-binning * step apart; e.g. 10"
            " turns a 5 nm scan into 50 nm slices, which suppresses stage"
            " jitter that is correlated between neighboring steps. Default 1"
            " (no binning)"
        ),
    )
    spline_calib_parser.add_argument(
        "-rm",
        "--registration-model",
        choices=list(TRANSFORM_MODELS),
        default="affine",
        help=(
            "how the channels are registered to the reference in a"
            " multichannel or split-FOV calibration: translation (2 DOF, a"
            " pure xy shift), rigid (3 DOF, a rotation and a shift), affine"
            " (6 DOF, the default), projective (8 DOF, adds the perspective"
            " term) or"
            " polynomial2 / polynomial3 (a smooth field-distortion warp,"
            " needing 6 / 10 well-spread beads). Ignored for a"
            " single-channel calibration"
        ),
    )
    spline_calib_parser.add_argument(
        "-m",
        "--model",
        choices=["spline-3d", "spline-2d"],
        default="spline-3d",
        help="build a 3D (z-recovering) or 2D (single-plane) spline PSF",
    )
    spline_calib_parser.add_argument(
        "-mf",
        "--magnification-factor",
        type=float,
        default=0.79,
        help=(
            "magnification factor applied to the fitted z at localization "
            "time (refractive-index mismatch), as in the astigmatism fit"
        ),
    )
    spline_calib_parser.add_argument(
        "-cz",
        "--correct-z-bias",
        action="store_true",
        help=(
            "define z = 0 at the axial intensity peak of the averaged PSF "
            "(astigmatism); corrects a potential z bias in the stage scan"
        ),
    )
    spline_calib_parser.add_argument(
        "-pr",
        "--photon-ratios",
        type=str,
        default="",
        help=(
            "multichannel only: candidate per-channel photon ratios for "
            "ratiometric color assignment, hypotheses separated by ';' and "
            "channels by ',' (e.g. '0.7,0.3;0.4,0.6'). Only relative values "
            "matter; stored in the calibration for "
            "fit_spline_multichannel_ratiometric"
        ),
    )
    spline_calib_parser.add_argument(
        "-sf",
        "--split-fov",
        type=str,
        default="",
        help=(
            "single-movie multichannel: treat rectangular field-of-view "
            "regions of ONE movie as the channels. Regions separated by ';' "
            "and given as 'y0,x0,y1,x1' (all the same size); the first (or "
            "--reference) region is the reference channel. e.g. "
            "'0,0,512,256;0,256,512,512'"
        ),
    )
    spline_calib_parser.add_argument(
        "-rf",
        "--reference",
        type=int,
        default=0,
        help="split-fov only: index of the reference region (default 0)",
    )
    spline_calib_parser.add_argument(
        "-bl", "--baseline", type=float, default=0, help="camera baseline"
    )
    spline_calib_parser.add_argument(
        "-se",
        "--sensitivity",
        type=float,
        default=1,
        help="camera sensitivity",
    )
    spline_calib_parser.add_argument(
        "-ga", "--gain", type=int, default=1, help="camera gain"
    )
    spline_calib_parser.add_argument(
        "-px",
        "--pixelsize",
        type=float,
        default=None,
        help=(
            "camera pixel size in nm, recorded in the calibration."
            " Default: the movie metadata, else the camera configuration"
        ),
    )

    lateral_parser = subparsers.add_parser(
        "lateral-calibrate",
        help=(
            "fit a lateral (astigmatism / chromatic) x-y correction from two "
            "bead images"
        ),
    )
    lateral_parser.add_argument(
        "reference",
        type=str,
        help=(
            "reference bead image: without the cylindrical lens"
            " (astigmatism), or in the reference color channel (chromatic)"
        ),
    )
    lateral_parser.add_argument(
        "target",
        type=str,
        help="bead image to be mapped onto the reference",
    )
    lateral_parser.add_argument(
        "-t",
        "--type",
        choices=["astigmatism", "chromatic"],
        default="astigmatism",
        help="what the transform corrects",
    )
    lateral_parser.add_argument(
        "-m",
        "--model",
        choices=list(TRANSFORM_MODELS),
        default="affine",
        help=(
            "transform model: translation (2 DOF, a pure xy shift), rigid"
            " (3 DOF, a rotation and a shift), affine (6 DOF, the default),"
            " projective (8 DOF), polynomial2 or polynomial3, needing at"
            " least 1 / 2 / 3 / 4 / 6 / 10 bead pairs respectively"
        ),
    )
    lateral_parser.add_argument(
        "-c",
        "--calibration",
        type=str,
        default="",
        help=(
            "existing calibration (.yaml or .hdf5) to append the correction"
            " to; omit to write a standalone one with --output"
        ),
    )
    lateral_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="where to write the calibration; defaults to --calibration",
    )
    lateral_parser.add_argument(
        "-b", "--box-side-length", type=int, default=7, help="box side length"
    )
    lateral_parser.add_argument(
        "-g",
        "--gradient",
        type=int,
        default=5000,
        help="minimum net gradient for bead detection",
    )
    _add_identification_method_args(lateral_parser, what="bead")
    lateral_parser.add_argument(
        "-px",
        "--pixelsize",
        type=float,
        required=True,
        help=(
            "camera pixel size in nm; the correction is fitted on the bead"
            " images (camera pixels) and stored in nm"
        ),
    )
    lateral_parser.add_argument(
        "-p",
        "--plot",
        type=str,
        default="",
        help="write the diagnostic figure to this path",
    )

    filter_parser = subparsers.add_parser(
        "filter",
        help=(
            "filter localizations (GUI), or keep the localizations of files"
            " that satisfy conditions, e.g. 'photons > 500 and 100 < sx < 160'"
        ),
    )
    filter_parser.add_argument(
        "files",
        nargs="?",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    filter_parser.add_argument(
        "conditions",
        nargs="*",
        help=(
            "one or more conditions on the columns, applied in this order;"
            " statistics such as median(photons) are computed over the"
            " localizations left by the previous conditions"
        ),
    )

    # render
    render_parser = subparsers.add_parser(
        "render", help="render localization based images"
    )
    render_parser.add_argument(
        "files",
        nargs="?",
        help=(
            "one or multiple localization files"
            " specified by a unix style path pattern"
        ),
    )
    render_parser.add_argument(
        "-px",
        "--disp-px-size",
        type=float,
        default=10.0,
        help="the size of the rendered pixel in nm",
    )
    render_parser.add_argument(
        "-b",
        "--blur-method",
        choices=["none", "convolve", "gaussian"],
        default="convolve",
    )
    render_parser.add_argument(
        "-w",
        "--min-blur-width",
        type=float,
        default=0.0,
        help="minimum blur width if blur is applied",
    )
    render_parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="minimum colormap level in range 0-100 or absolute value",
    )
    render_parser.add_argument(
        "--vmax",
        type=float,
        default=20.0,
        help="maximum colormap level in range 0-100 or absolute value",
    )
    render_parser.add_argument(
        "--scaling",
        choices=["yes", "no"],
        default="yes",
        help="if scaling the colormap value is relative in the range 0-100",
    )
    render_parser.add_argument(
        "-c",
        "--cmap",
        choices=["viridis", "inferno", "plasma", "magma", "hot", "gray"],
        help="the colormap to be applied",
    )
    render_parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="do not open the image file",
    )

    # design
    subparsers.add_parser("design", help="design RRO DNA origami structures")

    # simulate
    subparsers.add_parser(
        "simulate",
        help="simulate single molecule fluorescence data",
    )

    # server
    server_parser = subparsers.add_parser(
        "server", help="picasso server workflow management system"
    )
    server_parser.add_argument(
        "--network",
        action="store_true",
        help=(
            "listen on all network interfaces, so that other computers can "
            "open the server; anyone who can reach this computer can then "
            "use it, including the custom commands of the watcher (default: "
            "only this computer)"
        ),
    )

    # spinna
    _spinna_docs_url = docs_url("spinna.html#spinna-batch-analysis")
    spinna_parser = subparsers.add_parser(
        "spinna",
        help=(
            "picasso single protein investigation via nearest neighbor "
            "analysis"
        ),
        description=(
            "Run SPINNA. Without -p, the GUI is launched. With -p, batch "
            "analysis is run on the rows of the given .csv file. For the "
            "full .csv column reference, run `picasso spinna --columns` "
            f"or see {_spinna_docs_url}."
        ),
        epilog=f"Documentation: {_spinna_docs_url}",
    )
    spinna_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        help=(
            ".csv file containing the parameters for spinna batch analysis."
            " Run `picasso spinna --columns` for the column reference."
        ),
    )
    spinna_parser.add_argument(
        "-a",
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="fit in this process instead of a pool of worker processes",
    )
    spinna_parser.add_argument(
        "-b",
        "--bootstrap",
        action="store_true",
        help="perform bootstrapping",
    )
    spinna_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display progress bar for each row",
    )
    spinna_parser.add_argument(
        "--columns",
        action="store_true",
        help=("print the .csv column reference for batch analysis and exit"),
    )

    # nanotron
    subparsers.add_parser("nanotron", help="segmentation with deep learning")

    # average
    subparsers.add_parser("average", help="particle averaging (GUI)")

    # undrift RCC
    undrift_rcc_parser = subparsers.add_parser(
        "undrift", help="correct localization coordinates for drift using RCC"
    )
    undrift_rcc_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    undrift_rcc_parser.add_argument(
        "-s",
        "--segmentation",
        type=float,
        default=1000,
        help=(
            "the number of frames to be combined"
            " for one temporal segment (default=1000)"
        ),
    )
    undrift_rcc_parser.add_argument(
        "-f",
        "--from-file",
        type=str,
        help="apply drift from specified file instead of computing it",
    )
    undrift_rcc_parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="display estimated drift",
    )

    # undrift AIM
    undrift_aim_parser = subparsers.add_parser(
        "aim", help="correct localization coordinates for drift with AIM"
    )
    undrift_aim_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    undrift_aim_parser.add_argument(
        "-s",
        "--segmentation",
        type=float,
        default=100,
        help=(
            "the number of frames to be combined"
            " for one temporal segment (default=100)"
        ),
    )
    undrift_aim_parser.add_argument(
        "-i",
        "--intersect-distance",
        type=float,
        default=20.0,
        help=(
            "max. distance (nm) between localizations in consecutive"
            " segments to be considered as intersecting (default=20)"
        ),
    )
    undrift_aim_parser.add_argument(
        "-r",
        "--roi-radius",
        type=float,
        default=60.0,
        help="max. drift (nm) between two consecutive segments (default=60)",
    )

    # undrift by fiducials parser
    undrift_fiducial_parser = subparsers.add_parser(
        "undrift-fiducials",
        help="correct localization coordinates for drift with fiducials",
    )
    undrift_fiducial_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )

    # link parser
    link_parser = subparsers.add_parser(
        "link", help="link localizations in consecutive frames"
    )
    link_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    link_parser.add_argument(
        "-d",
        "--distance",
        type=float,
        default=10.0,
        help=(
            "maximum distance between localizations to consider them the same "
            "binding event in nm (default=10)"
        ),
    )
    link_parser.add_argument(
        "-t",
        "--tolerance",
        type=int,
        default=1,
        help=(
            "maximum dark time between localizations"
            " to still consider them the same binding event (default=1)"
        ),
    )

    # single-particle tracking
    track_parser = subparsers.add_parser(
        "track",
        help="link localizations into tracks (single-particle tracking)",
    )
    track_parser.add_argument(
        "files",
        nargs="+",
        help=(
            "one or multiple hdf5 localization files, specified by unix"
            " style path patterns"
        ),
    )
    track_parser.add_argument(
        "-d",
        "--max-jump",
        type=float,
        default=500.0,
        help=(
            "maximum distance between consecutive localizations of a track"
            " in nm (default=500)"
        ),
    )
    track_parser.add_argument(
        "-g",
        "--max-gap",
        type=int,
        default=1,
        help="maximum number of frames a track may skip (default=1)",
    )
    track_parser.add_argument(
        "-m",
        "--min-length",
        type=int,
        default=2,
        help="minimum number of localizations of a track (default=2)",
    )
    track_parser.add_argument(
        "-t",
        "--frame-time",
        type=float,
        default=None,
        help=(
            "time between two consecutive frames in ms, stored in the"
            " metadata for the analysis of the tracks"
        ),
    )
    diffusion_parser = subparsers.add_parser(
        "diffusion",
        help=(
            "diffusion analysis of tracks: mean squared displacement, jump"
            " distances and a summary of every track"
        ),
    )
    diffusion_parser.add_argument(
        "files",
        nargs="+",
        help=(
            "one or multiple tracked hdf5 localization files (see 'picasso"
            " track'), specified by unix style path patterns"
        ),
    )
    diffusion_parser.add_argument(
        "-t",
        "--frame-time",
        type=float,
        default=None,
        help=(
            "time between two consecutive frames in ms (default: from the"
            " metadata)"
        ),
    )
    diffusion_parser.add_argument(
        "-l",
        "--max-lag",
        type=int,
        default=10,
        help="largest time lag of the MSD in frames (default=10)",
    )
    diffusion_parser.add_argument(
        "-n",
        "--n-fit",
        type=int,
        default=4,
        help="number of time lags fitted in the MSD analysis (default=4)",
    )
    diffusion_parser.add_argument(
        "-k",
        "--populations",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help=(
            "number(s) of diffusing populations fitted to the jump"
            " distances (default=1 2 3)"
        ),
    )
    diffusion_parser.add_argument(
        "-s",
        "--sigma",
        type=_spt_sigma,
        default=0.0,
        help=(
            "localization error in nm for the jump distance fit, or 'lp' for"
            " the median localization precision (default=0, i.e., apparent"
            " diffusion coefficients)"
        ),
    )
    diffusion_parser.add_argument(
        "--truncate",
        action="store_true",
        help=(
            "correct the jump distance fit for the maximum jump distance of"
            " the tracking"
        ),
    )
    diffusion_parser.add_argument(
        "--plot",
        action="store_true",
        help="save the plots of the MSD, jump distances and track lengths",
    )

    # nneighbors
    nneighbor_parser = subparsers.add_parser(
        "nearest-neighbor",
        help="calculate nearest neighbor of a clustered dataset",
    )
    nneighbor_parser.add_argument(
        "files",
        nargs="?",
        help=(
            "one or multiple hdf5 clustered files"
            " specified by a unix style path pattern"
        ),
    )

    # coordinate-based colocalization
    cbc_parser = subparsers.add_parser(
        "cbc",
        help=(
            "coordinate-based colocalization (CBC) of localizations"
            " with other localization files"
        ),
    )
    cbc_parser.add_argument(
        "files",
        nargs="+",
        help=(
            "two or more hdf5 localization files (unix style path"
            " patterns); the first file is scored against all others"
        ),
    )
    cbc_parser.add_argument(
        "-r",
        "--r-max",
        type=float,
        default=200.0,
        help="maximum radius in nm (default=200)",
    )
    cbc_parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=100,
        help="number of radii up to the maximum radius (default=100)",
    )
    cbc_parser.add_argument(
        "-a",
        "--all-pairs",
        action="store_true",
        help=(
            "score every file against every other file, i.e., both"
            " directions of each pair (the CBC is not symmetric); with"
            " two files, this gives both directions"
        ),
    )
    cbc_parser.add_argument(
        "--2d",
        dest="use_2d",
        action="store_true",
        help="ignore z and compute the CBC in 2D",
    )
    cbc_parser.add_argument(
        "-n",
        "--names",
        nargs="+",
        help=(
            "short names of the files in the order given, used for the"
            " column names cbc_<name> (default: the file names)"
        ),
    )
    cbc_parser.add_argument(
        "-u",
        "--undefined",
        type=_cbc_undefined,
        default=0.0,
        help=(
            "CBC given to localizations without neighbors within the"
            " maximum radius, between -1 and 1, or 'remove' to remove"
            " them (default=0)"
        ),
    )

    # Ripley's H function and radial distribution function
    ripley_h_parser = subparsers.add_parser(
        "ripley-h",
        help=(
            "Ripley's H function of localizations: are they clustered,"
            " random or dispersed"
        ),
    )
    _add_pair_statistic_arguments(ripley_h_parser, steps=100)
    rdf_parser = subparsers.add_parser(
        "rdf",
        help=(
            "radial distribution function of localizations: density of"
            " neighbors at each distance, relative to the mean density"
        ),
    )
    _add_pair_statistic_arguments(rdf_parser, steps=50)

    # nearest neighbor distance distributions
    nnd_parser = subparsers.add_parser(
        "nnd",
        help=(
            "distributions of the 1st nearest neighbor distances within"
            " and between channels, with random simulations"
        ),
    )
    nnd_parser.add_argument(
        "files",
        nargs="+",
        help="hdf5 localization files of the channels (unix style path"
        " patterns), aligned",
    )
    nnd_parser.add_argument(
        "--names",
        nargs="+",
        help="names of the channels in the order of the files (default:"
        " the file names)",
    )
    nnd_parser.add_argument(
        "-p",
        "--picks",
        default=None,
        help=(
            "picks (.yaml) saved in Render; the regions of interest in"
            " which the distances are computed (default: the whole field"
            " of view)"
        ),
    )
    nnd_parser.add_argument(
        "-r",
        "--r-max",
        type=float,
        default=200.0,
        help="largest distance of the histograms in nm (default=200)",
    )
    nnd_parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=40,
        help="number of histogram bins up to rmax (default=40)",
    )
    nnd_parser.add_argument(
        "-n",
        "--simulations",
        type=int,
        default=99,
        help="number of random simulations, 0 to skip (default=99)",
    )
    nnd_parser.add_argument(
        "--3d",
        dest="use_3d",
        action="store_true",
        help="compute the distances in 3D (x, y and z), requires z",
    )
    nnd_parser.add_argument(
        "--z-range",
        nargs=2,
        type=float,
        metavar=("ZMIN", "ZMAX"),
        default=None,
        help=(
            "z range in nm over which the regions extend in 3D (default:"
            " the z range of the localizations in each region)"
        ),
    )
    nnd_parser.add_argument(
        "--observed-z",
        action="store_true",
        help=(
            "in 3D, shuffle the observed z values for the simulated"
            " points within a channel instead of spreading them uniformly"
        ),
    )
    nnd_parser.add_argument(
        "--cross-null",
        choices=["shift", "labels"],
        default="shift",
        help=(
            "how pairs of channels are simulated: random shifts of the"
            " second channel (independent channels; rectangular or"
            " circular regions only) or random labels (both channels"
            " follow the same distribution) (default=shift)"
        ),
    )

    # decorrelation resolution
    decorrelation_parser = subparsers.add_parser(
        "decorrelation",
        help=(
            "resolution of localizations by decorrelation analysis, over"
            " the whole field of view"
        ),
    )
    decorrelation_parser.add_argument(
        "files",
        nargs="+",
        help="hdf5 localization files (unix style path patterns)",
    )
    decorrelation_parser.add_argument(
        "-px",
        "--pixel-size",
        nargs="+",
        type=float,
        default=None,
        help=(
            "pixel size or sizes in nm at which the localizations are"
            " rendered; the smallest resolution is reported (default:"
            " 2 to 64 nm in steps of a factor sqrt(2))"
        ),
    )
    decorrelation_parser.add_argument(
        "--n-radii",
        type=int,
        default=50,
        help="number of radii of the decorrelation functions (default=50)",
    )
    decorrelation_parser.add_argument(
        "--n-filters",
        type=int,
        default=10,
        help="number of high-pass filters (default=10)",
    )
    decorrelation_parser.add_argument(
        "--max-image-size",
        type=int,
        default=8192,
        help=(
            "largest side length of the rendered images in pixels; smaller"
            " pixel sizes are skipped, 0 for no limit (default=8192)"
        ),
    )

    # dye performance of picked docking sites
    dye_parser = subparsers.add_parser(
        "dye-performance",
        help=(
            "signal-to-background ratio, localization drop and destroyed"
            " docking sites of picked docking sites (Steen et al., 2024)"
        ),
    )
    dye_parser.add_argument(
        "files",
        nargs="+",
        help=(
            "hdf5 localization files (unix style path patterns); without"
            " --picks, each group of the localizations is a docking site,"
            " e.g., in picked localizations saved in Render"
        ),
    )
    dye_parser.add_argument(
        "-p",
        "--picks",
        default=None,
        help="picks file (.yaml) with one docking site per pick",
    )
    dye_parser.add_argument(
        "-t",
        "--max-dark-time",
        type=int,
        default=0,
        help=(
            "largest number of frames without a localization within a"
            " binding event; 0 links only consecutive frames, as Steen et"
            " al. (default=0)"
        ),
    )
    dye_parser.add_argument(
        "--cutoff",
        type=float,
        default=4.0,
        help=(
            "cut-off of the ratio r (time after the last binding event over"
            " the mean dark time) for destroyed sites, in units of its"
            " fitted mean (default=4)"
        ),
    )
    dye_parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help=(
            "fraction of the frames at the beginning and the end compared"
            " for the localization drop (default=0.2)"
        ),
    )

    # density from the k nearest neighbors
    knn_parser = subparsers.add_parser(
        "knn-density",
        help=(
            "density of molecules fitted to the distances to their k"
            " nearest neighbors: are they randomly distributed"
        ),
    )
    knn_parser.add_argument(
        "files",
        nargs="+",
        help="hdf5 localization files (unix style path patterns)",
    )
    knn_parser.add_argument(
        "-p",
        "--picks",
        default=None,
        help=(
            "picks (.yaml) saved in Render; the regions of interest in"
            " which the neighbors are searched (default: the whole field"
            " of view)"
        ),
    )
    knn_parser.add_argument(
        "-k",
        "--k-max",
        type=int,
        default=5,
        help="highest order of the neighbors (default=5)",
    )
    knn_parser.add_argument(
        "--k-min",
        type=int,
        default=1,
        help=(
            "lowest order of the neighbors of the pooled density, e.g., 2"
            " to leave out the nearest neighbors (default=1)"
        ),
    )
    knn_parser.add_argument(
        "--r-min",
        type=float,
        default=0.0,
        help=(
            "shortest distance to a neighbor used in nm; shorter ones are"
            " left out, e.g., molecules counted twice (default=0)"
        ),
    )
    knn_parser.add_argument(
        "-n",
        "--simulations",
        type=int,
        default=99,
        help=(
            "number of simulations of complete spatial randomness for the"
            " p-values and errors, 0 to skip (default=99)"
        ),
    )
    knn_parser.add_argument(
        "--no-edge-correction",
        dest="edge_correction",
        action="store_false",
        help=(
            "use the distances as measured, also those longer than the"
            " distance to the region boundary"
        ),
    )
    knn_parser.add_argument(
        "--3d",
        dest="use_3d",
        action="store_true",
        help="measure the distances in 3D (x, y and z), requires z",
    )
    knn_parser.add_argument(
        "--z-range",
        nargs=2,
        type=float,
        metavar=("ZMIN", "ZMAX"),
        default=None,
        help=(
            "z range in nm over which the regions extend in 3D (default:"
            " the z range of the molecules in each region)"
        ),
    )

    pick_similar_parser = subparsers.add_parser(
        "pick-similar",
        help=(
            "pick structures similar to seed picks, or by ranges of their"
            " number of localizations and RMSD without seed picks"
        ),
    )
    pick_similar_parser.add_argument(
        "files",
        nargs="+",
        help="one or more hdf5 localization files (unix style path patterns)",
    )
    pick_similar_parser.add_argument(
        "-p",
        "--picks",
        default=None,
        help=(
            "seed picks (.yaml) saved in Render; the picks similar to"
            " them are found (circles, squares, rectangles or boxes)"
        ),
    )
    pick_similar_parser.add_argument(
        "-d",
        "--pick-size",
        type=float,
        default=None,
        help="diameter (nm) of the circular picks, without --picks",
    )
    pick_similar_parser.add_argument(
        "-s",
        "--std-range",
        type=float,
        default=2.0,
        help=(
            "with --picks, the range around the mean of the seed picks in"
            " standard deviations (default=2)"
        ),
    )
    pick_similar_parser.add_argument(
        "--n-locs-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="range of the number of localizations of the picks",
    )
    pick_similar_parser.add_argument(
        "--rmsd-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="range of the RMSD (nm) of the picks",
    )
    pick_similar_parser.add_argument(
        "--classes",
        action="store_true",
        help=(
            "find classes of structures by their number of localizations"
            " and RMSD and save the picks of each class to its own file"
        ),
    )
    pick_similar_parser.add_argument(
        "--min-n-locs",
        type=int,
        default=None,
        help=(
            "without --picks, the minimum number of localizations of a"
            " candidate pick (default: 1%% of the frames; at least the"
            " minimum of --n-locs-range)"
        ),
    )
    pick_similar_parser.add_argument(
        "--save-candidates",
        action="store_true",
        help=(
            "without --picks, also save the candidate picks with their"
            " number of localizations and RMSD (.csv)"
        ),
    )

    clusterfilter_parser = subparsers.add_parser(
        "cluster-filter",
        help="filter localizations by properties of their clusters",
    )
    clusterfilter_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    clusterfilter_parser.add_argument(
        "-c", "--cluster-file", help="a hdf5 file of cluster centers"
    )
    clusterfilter_parser.add_argument(
        "-p", "--parameter", type=str, help="parameter to be filtered"
    )
    clusterfilter_parser.add_argument(
        "--min-value",
        type=float,
        help="lower boundary",
    )
    clusterfilter_parser.add_argument(
        "--max-value",
        type=float,
        help="upper boundary",
    )

    # local densitydd
    density_parser = subparsers.add_parser(
        "density", help="compute the local density of localizations"
    )
    density_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    density_parser.add_argument(
        "radius",
        type=float,
        help=(
            "maximal distance (nm) between two localizations"
            " to be considered local"
        ),
    )

    # DBSCAN
    dbscan_parser = subparsers.add_parser(
        "dbscan",
        help="cluster localizations with the dbscan clustering algorithm",
    )
    dbscan_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    dbscan_parser.add_argument(
        "radius",
        type=float,
        help=(
            "maximal distance (nm) between two localizations"
            " to be considered local"
        ),
    )
    dbscan_parser.add_argument(
        "density",
        type=int,
        help=(
            "minimum local density for localizations"
            " to be assigned to a cluster"
        ),
    )
    dbscan_parser.add_argument(
        "--radius-z",
        type=float,
        help=(
            "DBSCAN epsilon in z (nm). If set, enables anisotropic 3D "
            "clustering."
        ),
        default=None,
    )

    # HDBSCAN
    hdbscan_parser = subparsers.add_parser(
        "hdbscan",
        help="cluster localizations with the hdbscan clustering algorithm",
    )
    hdbscan_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    hdbscan_parser.add_argument(
        "min_cluster",
        type=int,
        help=("smallest size grouping that is considered a cluster"),
    )
    hdbscan_parser.add_argument(
        "min_samples",
        type=int,
        help=("the higher the more points are considered noise"),
    )

    # SMLM clusterer
    smlm_cluster_parser = subparsers.add_parser(
        "smlm-cluster",
        help="cluster localizations with the custom SMLM clustering algorithm",
    )
    smlm_cluster_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    smlm_cluster_parser.add_argument(
        "radius",
        type=float,
        help=("clustering radius (nm)"),
    )
    smlm_cluster_parser.add_argument(
        "min_locs",
        type=int,
        help=("minimum number of localizations in a cluster"),
    )
    smlm_cluster_parser.add_argument(
        "basic_fa",
        nargs="?",
        type=_parse_bool,
        help=(
            "whether or not perform basic frame analysis (sticking event "
            "removal): yes/no, true/false or 1/0 (default: no)"
        ),
        default=False,
    )
    smlm_cluster_parser.add_argument(
        "radius_z",
        nargs="?",
        type=float,
        help=("clustering radius (nm) in axial direction (required for 3D)"),
        default=None,
    )

    g5m_parser = subparsers.add_parser(
        "g5m",
        help=(
            "Gaussian Mixture Modeling with Modifications for Molecular "
            "Mapping\n"
            "For more details see https://doi.org/10.1038/s41467-026-70198-5"
        ),
    )
    g5m_parser.add_argument(
        "files",
        nargs="?",
        help=(
            "path clustered localization file(s) (.hdf5) specified by a unix "
            "style path pattern or a path to the folder in which all .hdf5 "
            "files will be analyzed"
        ),
    )
    g5m_parser.add_argument(
        "-ml",
        "--min-locs",
        type=int,
        default=10,
        help="min. number of locs per molecule",
    )
    g5m_parser.add_argument(
        "-lph",
        "--loc-prec-handle",
        type=str,
        default="local",
        help="loc. precision handle, either 'local' or 'abs'",
    )
    g5m_parser.add_argument(
        "--min-sigma",
        type=float,
        default=0.8,
        help="minimum sigma factor/value",
    )
    g5m_parser.add_argument(
        "--max-sigma",
        type=float,
        default=1.5,
        help="maximum sigma factor/value",
    )
    g5m_parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="max. rounds without BIC improvement to terminate",
    )
    g5m_parser.add_argument(
        "--bootstrap-sem",
        action="store_true",
        help="bootstrap to estimate SEM of molecule positions",
    )
    g5m_parser.add_argument(
        "-c",
        "--calibration",
        type=str,
        default="",
        help=(
            "path to astigmatism calibration file, used and required "
            "only for astigmatism 3D data"
        ),
    )
    g5m_parser.add_argument(
        "--mode",
        type=str,
        choices=["astigmatism", "spline"],
        default="astigmatism",
        help=(
            "fitting mode of the input 3D localizations: 'astigmatism' "
            "(couples x/y widths via the calibration, requires -c) or "
            "'spline' (plain diagonal 3D model, reads z/lpz from the "
            "locs, no calibration needed); ignored for 2D data"
        ),
    )
    g5m_parser.add_argument(
        "--covariance-type",
        type=str,
        choices=["auto", "spherical", "diagonal", "rotated"],
        default="auto",
        help=(
            "shape of the G5M components: 'rotated' gives the xy "
            "covariance a rotation read from the 'angle' column, for 3D "
            "astigmatism data localized with a rotated elliptical "
            "Gaussian; 'diagonal' is the axis-aligned 3D model and "
            "'spherical' the isotropic 2D one. "
            "'auto' (default) picks 'rotated' for 3D astigmatism locs "
            "that carry an 'angle' column and otherwise keeps the "
            "established model"
        ),
    )
    g5m_parser.add_argument(
        "-p",
        "--postprocess",
        action="store_false",
        help=(
            "do not postprocess results to remove sticking events and"
            " low-quality fits"
        ),
    )
    g5m_parser.add_argument(
        "--max-locs",
        type=int,
        default=100000,
        help=(
            "maximum number of localizations to process per cluster; "
            "useful for excluding fiducials"
        ),
    )
    g5m_parser.add_argument(
        "-a",
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="fit in this process instead of a pool of worker processes",
    )
    g5m_parser.add_argument(
        "--group-column",
        type=str,
        choices=["group", "group_input"],
        default="group",
        help=(
            "column used to group localizations into clusters; use "
            "'group_input' if 'group' was overwritten but the original "
            "cluster ids are kept in 'group_input'"
        ),
    )

    # Dark time
    dark_parser = subparsers.add_parser(
        "dark", help="compute the dark time for grouped localizations"
    )
    dark_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )

    # align
    align_parser = subparsers.add_parser(
        "align", help="align one localization file to another"
    )
    align_parser.add_argument(
        "-m",
        "--method",
        choices=["rcc", "points"],
        default="rcc",
        help=(
            "how the shifts are found: rcc, the cross-correlation of the"
            " rendered images, or points, the peak of the displacements"
            " between the localizations, with an uncertainty"
            " (default=rcc)"
        ),
    )
    align_parser.add_argument(
        "--max-shift",
        type=float,
        default=500.0,
        help=(
            "largest shift (nm) between two files searched for by"
            " --method points (default=500)"
        ),
    )
    align_parser.add_argument(
        "-d",
        "--display",
        help="display the mean shift per RCC iteration",
        action="store_true",
    )
    align_parser.add_argument(
        "-r",
        "--remove-non-overlapping",
        help=(
            "remove the localizations outside the region where all aligned"
            " files overlap (by default, all localizations are kept)"
        ),
        action="store_true",
    )
    # align_parser.add_argument('-a', '--affine',
    # help='include affine transformations (may take long time)',
    # action='store_true')
    align_parser.add_argument(
        "file", help="one or multiple hdf5 localization files", nargs="+"
    )

    # join
    join_parser = subparsers.add_parser(
        "join",
        help=(
            "join hdf5 localization lists. frame numbers of consecutive files "
            "will be reindexed."
        ),
    )
    join_parser.add_argument(
        "file", nargs="+", help="the hdf5 localization files to be joined"
    )
    join_parser.add_argument(
        "-k",
        "--keep-index",
        help="do not change frame numbers",
        action="store_true",
    )

    # stitch
    stitch_parser = subparsers.add_parser(
        "stitch",
        help=(
            "stitch localization files of tiles recorded at different"
            " stage positions (Micro-Manager metadata) into one file per"
            " target"
        ),
    )
    stitch_parser.add_argument(
        "files",
        nargs="+",
        help=(
            "hdf5 localization files of the tiles (unix style path"
            " patterns); files at the same stage position are different"
            " targets, matched across tiles in the order given"
        ),
    )
    stitch_parser.add_argument(
        "-o",
        "--output",
        help=(
            "output file (default: <first file>_stitched.hdf5); with"
            " several targets, _target<n> is appended"
        ),
    )
    stitch_parser.add_argument(
        "-r",
        "--search-radius",
        type=float,
        default=10.0,
        help=(
            "how far the true offset of two tiles may be from the one"
            " given by the stage positions, in um (default=10)"
        ),
    )
    stitch_parser.add_argument(
        "--orientation",
        default="auto",
        help=(
            "image x and y axes in terms of the stage X and Y axes, i.e.,"
            " where a tile at a larger stage position lies in the image:"
            " e.g., '-x, -y' if a larger stage X is to the left and a"
            " larger stage Y above (the image y axis points down), 'y, x'"
            " if the stage X axis runs along the image y axis; one of"
            " 'x, y', '-x, y', 'x, -y', '-x, -y', 'y, x', '-y, x', 'y, -x',"
            " '-y, -x', or 'auto' to find the best one (default=auto)"
        ),
    )
    stitch_parser.add_argument(
        "--2d",
        dest="use_2d",
        action="store_true",
        help="do not correct the z offsets of the tiles",
    )
    stitch_parser.add_argument(
        "--keep-overlaps",
        action="store_true",
        help=(
            "keep the localizations of all tiles in the overlaps instead"
            " of cutting them at the seams"
        ),
    )
    stitch_parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="do not offset the frames of consecutive tiles",
    )

    # group properties
    groupprops_parser = subparsers.add_parser(
        "group-properties",
        help=(
            "calculate kinetics "
            "and various properties of localization groups"
        ),
    )
    groupprops_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )

    # Pair correlation
    pc_parser = subparsers.add_parser(
        "pair-correlation",
        help="calculate the pair-correlation of localizations",
    )
    pc_parser.add_argument(
        "-b",
        "--bin-size",
        type=float,
        default=10.0,
        help="the bin size in nm (default=10)",
    )
    pc_parser.add_argument(
        "-r",
        "--r-max",
        type=float,
        default=1000.0,
        help=(
            "the maximum distance in nm to calculate the pair-correlation"
            " (default=1000)"
        ),
    )
    pc_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )

    # export/import conversions
    csv2hdf_parser = subparsers.add_parser(
        "csv2hdf", help="convert csv (ThunderSTORM) to hdf5 format"
    )
    csv2hdf_parser.add_argument("files")
    csv2hdf_parser.add_argument(
        "-p",
        "--pixelsize",
        help="camera pixel size in nm",
        type=float,
        required=True,
    )

    hdf2csv_parser = subparsers.add_parser(
        "hdf2csv", help="convert hdf5 to csv format"
    )
    hdf2csv_parser.add_argument("files", help="one or multiple hdf5 files")

    hdf2ts_parser = subparsers.add_parser(
        "hdf2ts", help="convert hdf5 to ThunderSTORM csv format"
    )
    hdf2ts_parser.add_argument("files", help="one or multiple hdf5 files")

    hdf2imagej_parser = subparsers.add_parser(
        "hdf2imagej", help="convert hdf5 to ImageJ .txt format (frame, x, y)"
    )
    hdf2imagej_parser.add_argument("files", help="one or multiple hdf5 files")

    hdf2nis_parser = subparsers.add_parser(
        "hdf2nis", help="convert hdf5 to NIS .txt format"
    )
    hdf2nis_parser.add_argument("files", help="one or multiple hdf5 files")

    hdf2chimera_parser = subparsers.add_parser(
        "hdf2chimera", help="convert hdf5 to Chimera .xyz format"
    )
    hdf2chimera_parser.add_argument("files", help="one or multiple hdf5 files")

    hdf2visp_parser = subparsers.add_parser(
        "hdf2visp", help="convert hdf5 to visp format"
    )
    hdf2visp_parser.add_argument("files", help="one or multiple hdf5 files")

    smap2hdf_parser = subparsers.add_parser(
        "smap2hdf", help="convert SMAP _sml.mat to hdf5 format"
    )
    smap2hdf_parser.add_argument(
        "files", help="one or multiple _sml.mat files"
    )
    smap2hdf_parser.add_argument(
        "-p",
        "--pixelsize",
        help="camera pixel size in nm",
        type=float,
        required=True,
    )

    hdf2smap_parser = subparsers.add_parser(
        "hdf2smap", help="convert hdf5 to SMAP _sml.mat format"
    )
    hdf2smap_parser.add_argument("files", help="one or multiple hdf5 files")

    pyme2hdf_parser = subparsers.add_parser(
        "pyme2hdf", help="convert PyME .h5r/.hdf to hdf5 format"
    )
    pyme2hdf_parser.add_argument(
        "files", help="one or multiple PyME .h5r/.hdf files"
    )
    pyme2hdf_parser.add_argument(
        "-p",
        "--pixelsize",
        help="camera pixel size in nm, used for files that do not store it",
        type=float,
        default=None,
    )

    hdf2pyme_parser = subparsers.add_parser(
        "hdf2pyme", help="convert hdf5 to PyME .hdf format"
    )
    hdf2pyme_parser.add_argument("files", help="one or multiple hdf5 files")

    minflux2hdf_parser = subparsers.add_parser(
        "minflux2hdf",
        help="convert MINFLUX .pmx/.csv (pyMINFLUX) to hdf5 format",
    )
    minflux2hdf_parser.add_argument(
        "files", help="one or multiple MINFLUX .pmx/.csv files"
    )
    minflux2hdf_parser.add_argument(
        "--average-traces",
        action="store_true",
        help=(
            "save one localization per MINFLUX trace (photon-weighted"
            " mean) instead of every localization"
        ),
    )
    minflux2hdf_parser.add_argument(
        "--origin",
        choices=("data", "stage"),
        default="data",
        help=(
            "'data' (default) moves the localizations next to the origin of"
            " the field of view, 'stage' keeps the absolute coordinates"
        ),
    )

    # files written before Picasso 1.0 (camera pixels) to nm
    convert_units_parser = subparsers.add_parser(
        "convert-units",
        help=(
            "convert files written before Picasso 1.0 from camera pixels to"
            " nm: localizations (.hdf5), picks (.yaml), drift (.txt),"
            " calibrations and SPINNA masks (.yaml)"
        ),
    )
    convert_units_parser.add_argument(
        "files",
        nargs="+",
        help="the files to convert (unix style path patterns)",
    )
    convert_units_parser.add_argument(
        "-px",
        "--pixelsize",
        type=float,
        default=None,
        help=(
            "camera pixel size in nm, for files that do not record it"
            " (default: the Pixelsize of the file's metadata)"
        ),
    )
    convert_units_parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help=(
            "overwrite the files (by default, the converted file is saved"
            " next to the original with the suffix _nm)"
        ),
    )

    # plugins
    plugins_parser = subparsers.add_parser(
        "plugins", help="manage Picasso plugins"
    )
    plugins_actions = plugins_parser.add_subparsers(dest="plugins_action")
    plugins_actions.add_parser(
        "list", help="list the plugin files found and what they provide"
    )
    plugins_actions.add_parser("path", help="print the plugins folder")
    plugins_enable_parser = plugins_actions.add_parser(
        "enable", help="allow a plugin file to be loaded and run"
    )
    plugins_enable_parser.add_argument(
        "file", help="the plugin file name, e.g. my_plugin.py"
    )
    plugins_disable_parser = plugins_actions.add_parser(
        "disable", help="stop loading a plugin file, without deleting it"
    )
    plugins_disable_parser.add_argument(
        "file", help="the plugin file name, e.g. my_plugin.py"
    )
    plugins_install_parser = plugins_actions.add_parser(
        "install", help="download a plugin from the online registry"
    )
    plugins_install_parser.add_argument("id", help="the plugin id")
    plugins_update_parser = plugins_actions.add_parser(
        "update", help="re-download an installed plugin at its latest version"
    )
    plugins_update_parser.add_argument("id", help="the plugin id")
    plugins_uninstall_parser = plugins_actions.add_parser(
        "uninstall", help="delete an installed plugin"
    )
    plugins_uninstall_parser.add_argument("id", help="the plugin id")
    for _p in (
        plugins_enable_parser,
        plugins_disable_parser,
        plugins_install_parser,
        plugins_update_parser,
    ):
        _p.add_argument(
            "--yes",
            action="store_true",
            help="skip the confirmation prompt",
        )
    return parser, subparsers


class _ConsoleLogHandler(logging.StreamHandler):
    """Write the messages of Picasso's loggers to the current
    ``sys.stderr``: information as it is, warnings and errors after
    their level, e.g., ``Warning: ...``."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self):
        # the current stream, also after it was replaced
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if record.levelno >= logging.WARNING:
            text = f"{record.levelname.capitalize()}: {text}"
        return text


def _log_to_console() -> None:
    """Show the messages of Picasso's loggers (``picasso.*``) from
    information on in the console, unless logging is set up already,
    e.g., by a program that runs ``main``."""
    logger = logging.getLogger("picasso")
    if logger.handlers or logging.getLogger().handlers:
        return
    logger.addHandler(_ConsoleLogHandler())
    logger.setLevel(logging.INFO)


def main():  # noqa: C901
    """Entry point of the ``picasso`` command line interface.

    Builds the argument parser for every subcommand - the GUIs, the
    calibrations, ``localize``, ``render``, the undrift, clustering and
    file-conversion tools - parses ``sys.argv`` and dispatches to the matching
    handler. Called by the ``picasso`` console script and by
    ``python -m picasso``. Run ``picasso -h`` for the full list.
    """
    from .diagnostics import ensure_std_streams, install_excepthooks

    # in the windowed one-click build there is no console: give the
    # process usable streams and log uncaught exceptions to
    # ~/.picasso/logs/picasso.log instead of dropping them
    ensure_std_streams()
    install_excepthooks()

    parser, subparsers = _build_parser()

    # Plugins may add their own subcommands. Done last so that a plugin can
    # never shadow a built-in command, and skipped entirely when no plugin is
    # enabled, so the common case costs one directory listing.
    #
    # "picasso plugins ..." is skipped too: it needs no plugin-contributed
    # command, and it reports a plugin that will not load in full itself, so
    # importing here would only precede that report with a terse warning
    # about the very thing it is about to explain.
    import sys

    plugin_commands: set[str] = set()
    if sys.argv[1:2] != ["plugins"]:
        from .plugins import register_cli_plugins

        plugin_commands = register_cli_plugins(subparsers)

    # Parse (the names renamed in Picasso 1.0 still work, with a notice)
    args = parser.parse_args(_current_names(sys.argv[1:]))
    if args.command:
        # check for updates while the command runs, see CliUpdateCheck
        from .updater import CliUpdateCheck

        cli_update_check = True
        update_check = None
        gui_apps = [
            "localize",
            "filter",
            "render",
            "average",
            "nanotron",
            "simulate",
            "design",
            "spinna",
        ]
        if args.command in gui_apps:
            if len(sys.argv) == 2:  # only the gui is opened
                cli_update_check = False
        if cli_update_check:
            # a command, not a GUI: its messages go to the console
            _log_to_console()
            update_check = CliUpdateCheck()

        if args.command in plugin_commands:
            _run_plugin_command(args)
        elif args.command == "plugins":
            _plugins(args)
        elif args.command == "localize":
            if args.files:
                _localize(args)
            else:
                from picasso.gui import localize

                localize.main()
        elif args.command == "camera-calibrate":
            _camera_calibrate(args)
        elif args.command == "camera-validate":
            _camera_validate(args)
        elif args.command == "spline-calibrate":
            _spline_calibrate(args)
        elif args.command == "lateral-calibrate":
            _lateral_calibrate(args)
        elif args.command == "filter":
            if args.files:
                if not args.conditions:
                    parser.error("filter: give at least one condition")
                _filter(args.files, args.conditions)
            else:
                from .gui import filter

                filter.main()
        elif args.command == "render":
            if args.files:
                _render(args)
            else:
                from .gui import render

                render.main()
        elif args.command == "average":
            from .gui import average

            average.main()
        elif args.command == "nanotron":
            from .gui import nanotron

            nanotron.main()
        elif args.command == "simulate":
            from .gui import simulate

            simulate.main()
        elif args.command == "design":
            from .gui import design

            design.main()
        elif args.command == "server":
            _start_server(args.network)
        elif args.command == "spinna":
            if args.columns:
                import inspect

                from .spinna import batch_analysis

                print(inspect.getdoc(batch_analysis))
            elif args.parameters:
                from .spinna import batch_analysis

                batch_analysis(
                    args.parameters,
                    parallel=args.parallel,
                    bootstrap=args.bootstrap,
                    progress="console" if args.verbose else None,
                )
            else:
                from .gui import spinna

                spinna.main()
        elif args.command == "link":
            _link(args.files, args.distance, args.tolerance)
        elif args.command == "track":
            _track(
                args.files,
                args.max_jump,
                args.max_gap,
                args.min_length,
                args.frame_time,
            )
        elif args.command == "diffusion":
            _diffusion(
                args.files,
                args.frame_time,
                args.max_lag,
                args.n_fit,
                args.populations,
                args.sigma,
                args.truncate,
                args.plot,
            )
        elif args.command == "cluster-filter":
            _clusterfilter(
                args.files,
                args.cluster_file,
                args.parameter,
                args.min_value,
                args.max_value,
            )
        elif args.command == "undrift":
            _undrift_rcc(
                args.files,
                args.segmentation,
                args.display,
                args.from_file,
            )
        elif args.command == "aim":
            _undrift_aim(
                args.files,
                args.segmentation,
                args.intersect_distance,
                args.roi_radius,
            )
        elif args.command == "undrift-fiducials":
            _undrift_fiducials(args.files)
        elif args.command == "density":
            _density(args.files, args.radius)
        elif args.command == "dbscan":
            _dbscan(
                args.files,
                args.radius,
                args.density,
                args.radius_z,
            )
        elif args.command == "hdbscan":
            _hdbscan(
                args.files,
                args.min_cluster,
                args.min_samples,
            )
        elif args.command == "smlm-cluster":
            _smlm_clusterer(
                args.files,
                args.radius,
                args.min_locs,
                args.basic_fa,
                args.radius_z,
            )
        elif args.command == "g5m":
            _g5m(
                args.files,
                args.min_locs,
                args.loc_prec_handle,
                args.min_sigma,
                args.max_sigma,
                args.max_rounds,
                args.bootstrap_sem,
                args.calibration,
                args.mode,
                args.covariance_type,
                args.postprocess,
                args.max_locs,
                args.parallel,
                args.group_column,
            )
        elif args.command == "nearest-neighbor":
            _nneighbor(args.files)
        elif args.command == "cbc":
            _cbc(
                args.files,
                args.r_max,
                args.steps,
                args.all_pairs,
                args.use_2d,
                args.names,
                args.undefined,
            )
        elif args.command == "nnd":
            _nnd(
                args.files,
                args.names,
                args.picks,
                args.r_max,
                args.steps,
                args.simulations,
                args.use_3d,
                None if args.z_range is None else tuple(args.z_range),
                args.observed_z,
                args.cross_null,
            )
        elif args.command == "decorrelation":
            _decorrelation(
                args.files,
                args.pixel_size,
                args.n_radii,
                args.n_filters,
                args.max_image_size,
            )
        elif args.command == "dye-performance":
            _dye_performance_cli(
                args.files,
                args.picks,
                args.max_dark_time,
                args.cutoff,
                args.fraction,
            )
        elif args.command == "knn-density":
            _knn_density(
                args.files,
                args.picks,
                args.k_max,
                args.k_min,
                args.r_min,
                args.simulations,
                args.edge_correction,
                args.use_3d,
                None if args.z_range is None else tuple(args.z_range),
            )
        elif args.command == "pick-similar":
            _pick_similar_cli(
                args.files,
                picks_path=args.picks,
                pick_size=args.pick_size,
                std_range=args.std_range,
                n_locs_range=args.n_locs_range,
                rmsd_range=args.rmsd_range,
                classes=args.classes,
                min_n_locs=args.min_n_locs,
                save_candidates=args.save_candidates,
            )
        elif args.command in ("ripley-h", "rdf"):
            _pair_statistic(
                "ripley_h" if args.command == "ripley-h" else "rdf",
                args.files,
                args.picks,
                args.r_max,
                args.steps,
                args.simulations,
                args.edge_correction,
                args.use_3d,
                None if args.z_range is None else tuple(args.z_range),
                args.observed_z,
                args.partner,
                args.cross_null,
            )
        elif args.command == "dark":
            _dark(args.files)
        elif args.command == "align":
            _align(
                args.file,
                args.display,
                args.remove_non_overlapping,
                args.method,
                args.max_shift,
            )
        elif args.command == "join":
            _join(args.file, args.keep_index)
        elif args.command == "stitch":
            _stitch(
                args.files,
                args.output,
                args.search_radius,
                args.orientation,
                args.use_2d,
                args.keep_overlaps,
                args.keep_frames,
            )
        elif args.command == "group-properties":
            _groupprops(args.files)
        elif args.command == "pair-correlation":
            _pair_correlation(args.files, args.bin_size, args.r_max)
        elif args.command == "csv2hdf":
            _csv2hdf(args.files, args.pixelsize)
        elif args.command == "hdf2csv":
            _hdf2csv(args.files)
        elif args.command == "hdf2ts":
            _hdf2ts(args.files)
        elif args.command == "hdf2imagej":
            _hdf2imagej(args.files)
        elif args.command == "hdf2nis":
            _hdf2nis(args.files)
        elif args.command == "hdf2chimera":
            _hdf2chimera(args.files)
        elif args.command == "hdf2visp":
            _hdf2visp(args.files)
        elif args.command == "smap2hdf":
            _smap2hdf(args.files, args.pixelsize)
        elif args.command == "hdf2smap":
            _hdf2smap(args.files)
        elif args.command == "pyme2hdf":
            _pyme2hdf(args.files, args.pixelsize)
        elif args.command == "hdf2pyme":
            _hdf2pyme(args.files)
        elif args.command == "minflux2hdf":
            _minflux2hdf(args.files, args.average_traces, args.origin)
        elif args.command == "convert-units":
            _convert_units(args.files, args.pixelsize, args.in_place)
    else:
        parser.print_help()
        return

    # tell about an update once the command is done
    if update_check is not None:
        update_check.finish()


if __name__ == "__main__":
    main()
