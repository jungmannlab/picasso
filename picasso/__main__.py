"""
__main__.py
~~~~~~~~~~~

Picasso command line interface.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os.path
import argparse
from typing import TYPE_CHECKING, Literal
import pandas as pd
from . import __version__, docs_url
from .transforms import MODELS as TRANSFORM_MODELS

if TYPE_CHECKING:  # imported lazily, like every analysis module here
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
                locs.to_csv(out_path, sep=",", encoding="utf-8")
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
    """Convert HDF5 localization files to NIS txt format."""
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


def _link(files: str, d_max: float, tolerance: float) -> None:
    """Link localizations in HDF5 files, see ``postprocess.link`` for
    details."""
    import glob
    import h5py
    import numpy as _np
    from tqdm import tqdm as _tqdm

    paths = glob.glob(files)
    if paths:
        from . import io, postprocess

        for path in paths:
            try:
                locs, info = io.load_locs(path)
            except io.NoMetadataFileError:
                continue
            linked_locs = postprocess.link(locs, info, d_max, tolerance)
            base, ext = os.path.splitext(path)
            link_info = {
                "Maximum Distance": d_max,
                "Maximum Transient Dark Time": tolerance,
                "Generated by": f"Picasso v{__version__} Link",
            }
            info.append(link_info)
            io.save_locs(base + "_link.hdf5", linked_locs, info)

            try:
                # Check if there is a _clusters.hdf5 file present
                # if yes update this file
                cluster_path = base[:-7] + "_clusters.hdf5"
                print(cluster_path)
                clusters = io.load_clusters(cluster_path)
                print("Clusterfile detected. Updating entries.")

                n_after_link = []
                linked_len = []
                linked_n = []
                linked_photonrate = []

                for group in _tqdm(_np.unique(clusters["groups"])):
                    temp = linked_locs[linked_locs["group"] == group]
                    if len(temp) > 0:
                        n_after_link.append(len(temp))
                        linked_len.append(_np.mean(temp["len"]))
                        linked_n.append(_np.mean(temp["n"]))
                        linked_photonrate.append(_np.mean(temp["photon_rate"]))

                clusters["n_after_link"] = _np.array(
                    n_after_link,
                    dtype=_np.int32,
                )
                clusters["linked_len"] = _np.array(linked_len, dtype=_np.int32)
                clusters["linked_n"] = _np.array(linked_n, dtype=_np.int32)
                clusters["linked_photonrate"] = _np.array(
                    linked_photonrate,
                    dtype=_np.float32,
                )
                # clusters.to_hdf(cluster_path, "clusters", mode="a")
                # cannot use to_hdf for backward compatibility with older
                # Picasso
                rec_clusters = clusters.to_records(index=False)
                with h5py.File(cluster_path, "w") as locs_file:
                    locs_file.create_dataset("clusters", data=rec_clusters)
            except Exception:
                print("No clusterfile found for updating.")
                continue


def _cluster_combine(files: str) -> None:
    """Combine clusters in HDF5 files. See
    ``postprocess.cluster_combine`` for details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, postprocess

        for path in paths:
            try:
                locs, info = io.load_locs(path)
            except io.NoMetadataFileError:
                continue
            combined_locs = postprocess.cluster_combine(locs)
            base, ext = os.path.splitext(path)
            combined_info = {"Generated by": f"Picasso v{__version__} Combine"}
            info.append(combined_info)
            io.save_locs(base + "_comb.hdf5", combined_locs, info)


def _cluster_combine_dist(files: str) -> None:
    """Combine clusters in HDF5 files based on distance. See
    ``postprocess.cluster_combine_dist`` for details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, postprocess

        for path in paths:
            try:
                locs, info = io.load_locs(path)
            except io.NoMetadataFileError:
                continue
            px = info[1].get("Pixelsize", None)
            combinedist_locs = postprocess.cluster_combine_dist(locs, px)
            base, ext = os.path.splitext(path)
            cluster_combine_dist_info = {
                "Generated by": f"Picasso v{__version__} CombineDist"
            }
            info.append(cluster_combine_dist_info)
            io.save_locs(base + "_cdist.hdf5", combinedist_locs, info)


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

            clusters = io.load_clusters(clusterfile)
            try:
                _clusterfilter_locs(
                    clusters, locs, info, parameter, minval, maxval, path
                )
            except ValueError:
                print("Error: Field {} not found.".format(parameter))


def _clusterfilter_locs(
    clusters: pd.DataFrame,
    locs: pd.DataFrame,
    info: list[dict],
    parameter: str,
    minval: float,
    maxval: float,
    path: str,
) -> None:
    import numpy as np
    from tqdm import tqdm
    from . import io

    selector = (clusters[parameter] > minval) & (clusters[parameter] < maxval)
    if np.sum(selector) == 0:
        print("Error: No localizations in range. Filtering aborted.")
    elif np.sum(selector) == len(selector):
        print("Error: All localizations in range. Filtering aborted.")
    else:
        print("Isolating locs.. Step 1: in range")
        groups = clusters["groups"][selector]
        first = True
        for group in tqdm(groups):
            if first:
                all_locs = locs[locs["group"] == group]
                first = False
            else:
                all_locs = np.append(
                    all_locs,
                    locs[locs["group"] == group],
                )

        base, ext = os.path.splitext(path)
        clusterfilter_info = {
            "Generated by": (f"Picasso v{__version__} Clusterfilter - in"),
            "Parameter": parameter,
            "Minval": minval,
            "Maxval": maxval,
        }
        info.append(clusterfilter_info)
        all_locs.sort_values(
            kind="quicksort",
            by="frame",
            inplace=True,
        )
        out_path = base + "_filter_in.hdf5"
        io.save_locs(out_path, all_locs, info)
        print("Complete. Saved to: {}".format(out_path))

        print("Isolating locs.. Step 2: out of range")
        groups = clusters["groups"][~selector]
        first = True
        for group in tqdm(groups):
            if first:
                all_locs = locs[locs["group"] == group]
                first = False
            else:
                all_locs = np.append(
                    all_locs,
                    locs[locs["group"] == group],
                )

        base, ext = os.path.splitext(path)
        clusterfilter_info = {
            "Generated by": (f"Picasso v{__version__} Clusterfilter - out"),
            "Parameter": parameter,
            "Minval": minval,
            "Maxval": maxval,
        }
        info.append(clusterfilter_info)
        all_locs.sort_values(
            kind="quicksort",
            by="frame",
            inplace=True,
        )
        out_path = base + "_filter_out.hdf5"
        io.save_locs(out_path, all_locs, info)
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
    undrift_info = {"Generated by": f"Picasso v{__version__} Undrift"}
    if fromfile is not None:
        undrift_info["From File"] = fromfile
        drift = io.load_drift(fromfile)
    else:
        undrift_info["Segmentation"] = segmentation
    for path in paths:
        try:
            locs, info = io.load_locs(path)
        except io.NoMetadataFileError:
            continue
        if fromfile is not None:
            # this works for mingjies drift files but not for the own ones
            locs.x -= drift.loc[locs.frame, "x"].to_numpy()
            locs.y -= drift.loc[locs.frame, "y"].to_numpy()

            if display:
                import matplotlib.pyplot as plt

                plt.style.use("ggplot")
                fig = plt.Figure(figsize=(10, 6), constrained_layout=True)
                pixelsize = lib.get_from_metadata(info, "Pixelsize", 1.0)
                postprocess.plot_drift(drift, pixelsize, fig)
                plt.show()
        else:
            print(f"Undrifting file {path}")
            drift, locs = postprocess.undrift(
                locs,
                info,
                segmentation,
                display=display,
            )

            undrift_info["Drift X"] = float(drift["x"].mean())
            undrift_info["Drift Y"] = float(drift["y"].mean())

        info.append(undrift_info)
        base, ext = os.path.splitext(path)
        io.save_locs(base + "_undrift.hdf5", locs, info)
        io.save_drift(base + "_drift.txt", drift)


def _undrift_aim(
    files: str,
    segmentation: int,
    intersectdist: float = 20 / 130,
    roiradius: float = 60 / 130,
) -> None:
    """Run AIM undrifting on the given files. See ``aim.aim`` for
    details."""
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
            segmentation,
            intersectdist,
            roiradius,
            progress="console",
        )
        base, ext = os.path.splitext(path)
        io.save_locs(base + "_aim.hdf5", locs, new_info)
        io.save_drift(base + "_aimdrift.txt", drift)


def _undrift_fiducials(files: str) -> None:
    """Automatically pick fiducials and use them for drift correction.
    See ``postprocess.undrift_from_picked`` and
    ``imageprocess.find_fiducials`` for more details.

    Uses RCC with segmentation of 2000 before so that fiducials can be
    detected.
    """
    import glob
    from . import io, postprocess

    segmentation = 2000

    paths = glob.glob(files)
    undrift_info = {
        "Generated by": f"Picasso v{__version__} Undrift by fiducials",
        "pre-RCC segmentation": segmentation,
    }
    for path in paths:
        try:
            locs, info = io.load_locs(path)
        except io.NoMetadataFileError:
            continue

        print(f"Undrifting file {path}.")
        drift, locs = postprocess.undrift(
            locs, info, segmentation, display=False
        )
        print("pre-RCC done.")
        locs, new_info, drift = postprocess.undrift_from_fiducials(locs, info)

        undrift_info["Drift X"] = float(drift["x"].mean())
        undrift_info["Drift Y"] = float(drift["y"].mean())
        undrift_info["Number of picks"] = new_info[-1]["Number of picks"]
        undrift_info["Pick radius (nm)"] = new_info[-1]["Pick radius (nm)"]

        info.append(undrift_info)
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
            locs = postprocess.compute_local_density(locs, info, radius)
            base, ext = os.path.splitext(path)
            density_info = {
                "Generated by": f"Picasso v{__version__} Density",
                "Radius": radius,
            }
            info.append(density_info)
            io.save_locs(base + "_density.hdf5", locs, info)


def _dbscan(
    files: str,
    radius: float,
    min_density: float,
    pixelsize: float | None = None,
    radius_z: float | None = None,
) -> None:
    """Run DBSCAN clustering on localizations in HDF5 files. See
    ``clusterer.dbscan`` for details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, clusterer

        for path in paths:
            print("Loading {} ...".format(path))
            locs, info = io.load_locs(path)
            locs, dbscan_info = clusterer.dbscan(
                locs,
                radius,
                min_density,
                pixelsize=pixelsize,
                radius_z=radius_z,
            )
            clusters = clusterer.find_cluster_centers(locs, pixelsize)
            base, _ = os.path.splitext(path)
            info.append(dbscan_info)
            io.save_locs(base + "_dbscan.hdf5", locs, info)
            io.save_locs(base + "_dbclusters.hdf5", clusters, info)
            print(
                "Clustering executed. Results are saved in: \n"
                f"{base}_dbscan.hdf5\n"
                f"{base}_dbclusters.hdf5"
            )


def _hdbscan(
    files: str,
    min_cluster: int,
    min_samples: int,
    pixelsize: float | None = None,
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
            locs, hdbscan_info = clusterer.hdbscan(
                locs, min_cluster, min_samples, pixelsize
            )
            clusters = clusterer.find_cluster_centers(locs, pixelsize)
            base, ext = os.path.splitext(path)
            info.append(hdbscan_info)
            io.save_locs(base + "_hdbscan.hdf5", locs, info)
            io.save_locs(base + "_hdbclusters.hdf5", clusters, info)
            print(
                "Clustering executed. Results are saved in: \n"
                f"{base}_hdbscan.hdf5\n"
                f"{base}_hdbclusters.hdf5"
            )


def _smlm_clusterer(
    files: str,
    radius: float,
    min_locs: int,
    pixelsize: float | None = None,
    basic_fa: bool = False,
    radius_z: float | None = None,
) -> None:
    """Run SMLM clustering on localizations in HDF5 files. See
    ``clusterer.cluster`` for details."""
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
            locs, smlm_cluster_info = clusterer.cluster(
                locs, **params, pixelsize=pixelsize, progress="console"
            )
            clusters = clusterer.find_cluster_centers(
                locs, pixelsize, progress="console"
            )
            base, ext = os.path.splitext(path)
            info.append(smlm_cluster_info)
            io.save_locs(base + "_clusters.hdf5", locs, info)
            io.save_locs(base + "_cluster_centers.hdf5", clusters, info)
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
    import pandas as pd
    from scipy.spatial import KDTree

    paths = glob.glob(files)
    if paths:
        for path in paths:
            print("Loading {} ...".format(path))
            clusters = pd.read_hdf(path, key="clusters")
            points = clusters[["com_x", "com_y"]].to_numpy()
            tree = KDTree(points)
            minvals, _ = tree.query(points, k=2)
            minvals = minvals[:, 1]
            base, ext = os.path.splitext(path)
            out_path = base + "_minval.txt"
            np.savetxt(out_path, minvals, newline="\r\n")
            print("Saved filest o: {}".format(out_path))


def _dark(files: str) -> None:
    """Compute dark times for localizations in HDF5 files. See
    ``postprocess.compute_dark_times`` for details."""
    import glob

    paths = glob.glob(files)
    if paths:
        from . import io, postprocess

        for path in paths:
            locs, info = io.load_locs(path)
            locs = postprocess.compute_dark_times(locs)
            base, ext = os.path.splitext(path)
            d_info = {"Generated by": f"Picasso v{__version__} Dark"}
            info.append(d_info)
            io.save_locs(base + "_dark.hdf5", locs, info)


def _align(files: str, display: bool) -> None:
    """Align localization files using RCC, see ``postprocess.align``
    for details."""
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
    aligned_locs = align(locs, infos, display=display)
    align_info = {
        "Generated by": f"Picasso v{__version__} Align",
        "Files": files,
    }
    for file, locs_, info in zip(files, aligned_locs, infos):
        info.append(align_info)
        base, ext = splitext(file)
        save_locs(base + "_align.hdf5", locs_, info)


def _join(files: list[str], keep_index: bool = True) -> None:
    """Join multiple localization files into one."""
    from .io import load_locs, save_locs
    from .lib import merge_locs
    from os.path import splitext

    all_locs = merge_locs(
        [load_locs(file)[0] for file in files],
        increment_frames=(not keep_index),
        increment_groups=False,
    )
    join_info = {
        "Generated by": f"Picasso v{__version__} Join",
        "Files": files,
    }
    base, ext = splitext(files[0])
    info = load_locs(files[0])[1]
    info.append(join_info)
    if not keep_index:
        info[0]["Frames"] = all_locs["frame"].max() + 1
    save_locs(base + "_join.hdf5", all_locs, info)


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
            groups = groupprops(locs)
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
            bins_lower, pc = pair_correlation(locs, info, bin_size, r_max)
            plot(bins_lower - bin_size / 2, pc)
            xlabel("r (pixel)")
            ylabel("pair-correlation (pixel^-2)")
            title(f"Pair-correlation. Bin size: {bin_size}, R max: {r_max}")
            show()


def _start_server() -> None:
    """Start the Streamlit server for the Picasso GUI."""
    import os
    import sys
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
    ]

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
    """Resolve ``files`` to a list of movie file paths to process."""
    from glob import glob
    from os.path import isdir

    if isdir(files):
        print("Analyzing folder")
        tif_files = _check_consecutive_tif(files)
        paths = (
            tif_files
            + glob(files + "/*.raw")
            + glob(files + "/*.nd2")
            + glob(files + "/*.stk")
            + glob(files + "/*.czi")
            + glob(files + "/*.lif")
        )
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


def _localize_load_3d_calibration(
    args: argparse.Namespace,
) -> tuple[str, float, dict]:
    """Load 3D z-calibration, prompting interactively if needed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``picasso localize`` arguments; ``zc`` (calibration path)
        and ``mf`` (magnification factor, 0 to prompt) are read.

    Returns
    -------
    tuple[str, float, dict]
        ``(zpath, magnification_factor, z_calibration)``
    """
    import os
    from .io import load_calibration

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
        z_calibration = load_calibration(zpath)
    except Exception as e:
        print(e)
        print("Error loading calibration file.")
        raise

    return zpath, magnification_factor, z_calibration


_FIT_METHOD_MAP = {
    "lq": "gausslq",
    "lq-spherical": "gausslq-spherical",
    "lq-spherical-gpu": "gausslq-spherical-gpu",
    "lq-rotated": "gausslq-rotated",
    "lq-rotated-gpu": "gausslq-rotated-gpu",
    "lq-3d": "gausslq",
    "lq-gpu": "gausslq-gpu",
    "lq-gpu-3d": "gausslq-gpu",
    "mle": "gaussmle",
    "mle-gpu": "gaussmle-gpu",
    "mle-spherical": "gaussmle-spherical",
    "mle-spherical-gpu": "gaussmle-spherical-gpu",
    "mle-rotated": "gaussmle-rotated",
    "mle-rotated-gpu": "gaussmle-rotated-gpu",
    "mle-3d": "gaussmle",
    "spline": "spline",
    "spline-mle": "spline-mle",
    "spline-gpu": "spline-gpu",
    "spline-mle-gpu": "spline-mle-gpu",
    "avg": "avg",
}


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
) -> None:
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
    min_net_gradient : float or list of float
        Minimum net gradient for identification; one value per ROI
        region if a list.
    roi : list or None
        ROI regions, each ``[[y_min, x_min], [y_max, x_max]]``, or None
        for the whole frame.
    frame_bounds : list of list of int or None
        Frame segments ``[start, end]`` (0-indexed, inclusive) to
        analyze, or None for all frames.
    camera_info : dict
        Camera parameters (baseline, sensitivity, gain, QE).
    convergence : float
        Fit convergence criterion; 0 uses the fitting method's default.
    max_iterations : int
        Maximum number of fit iterations; 0 uses the fitting method's
        default.
    z_params : tuple or None
        If 3D astigmatism fitting is active, a tuple
        ``(zpath, magnification_factor, z_calibration)``; else ``None``.
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
        are skipped rather than applied a second time.
    region_fit : dict or None
        Set by ``--regions-separately``: fit each ``--roi`` region on its
        own and write one file per region. Holds the per-region fitting
        methods and spline calibrations (see ``_localize_regions``).
    """
    from .io import load_movie, load_tif_concatenated
    from .localize import localize

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
    if len(paths) > 1:
        movie, info = load_tif_concatenated(paths)
    else:
        movie, info = load_movie(path)

    fitting_method = _FIT_METHOD_MAP[args.fit_method]
    cam_info = dict(camera_info)
    cam_info["Pixelsize"] = args.pixelsize
    if region_fit is not None:
        _localize_regions(
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
        )
        return
    parameters = {
        "Min. Net Gradient": min_net_gradient,
        "Box Size": box,
        "Temporal Median Window": args.temporal_median,
        "Gaussian Filter Sigma": args.gaussian_filter,
    }
    wavelet = _wavelet_from_args(args)
    if wavelet is not None:
        from .localize import IDENTIFY_METHOD_WAVELET

        parameters["Identification Method"] = IDENTIFY_METHOD_WAVELET
        parameters.update(wavelet.to_info())

    locs, info = localize(
        movie,
        camera_info=cam_info,
        identification_parameters=parameters,
        roi=roi,
        frame_bounds=frame_bounds,
        movie_info=info,
        fitting_method=fitting_method,
        eps=convergence if convergence > 0 else None,
        max_it=max_iterations if max_iterations > 0 else None,
        spline_calibration=spline_calibration,
        camera_calibration=camera_calibration,
        threaded=True,
        identification_progress_callback="console",
        fit_progress_callback="console",
    )

    _localize_finish(
        locs,
        info,
        path,
        args,
        z_params,
        lateral_transforms,
        spline_calibration,
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
    methods = [_FIT_METHOD_MAP[method] for method in fit_methods]
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
) -> None:
    """Fit each ``--roi`` region of one movie on its own
    (``--regions-separately``).

    The regions are channels imaged side by side on one sensor, so the spots
    are identified once over the whole frame and then fitted region by
    region - each with its own fitting method and spline calibration, each
    saved to its own file, in its own coordinates. Nothing is registered or
    linked across the regions; they are connected afterwards (e.g. by
    aligning them in Picasso: Render).
    """
    from .localize import (
        fit_split_fov_independent,
        identify,
        region_label,
    )

    identifications, identify_info = identify(
        movie,
        min_net_gradient,
        box,
        roi=roi,
        frame_bounds=frame_bounds,
        threaded=True,
        temporal_median_window=args.temporal_median,
        gaussian_filter_sigma=args.gaussian_filter,
        wavelet=_wavelet_from_args(args),
        progress_callback="console",
    )
    print(
        f"Identified {len(identifications):,} spots in {len(roi)} regions. "
        "Fitting each region on its own..."
    )
    results = fit_split_fov_independent(
        movie,
        cam_info,
        identifications,
        box,
        roi,
        region_fit["methods"],
        movie_info=info,
        eps=convergence if convergence > 0 else None,
        max_it=max_iterations if max_iterations > 0 else None,
        spline_calibration=region_fit["spline_calibrations"],
        camera_calibration=camera_calibration,
        multiprocess=True,
        progress_callback="console",
    )
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
        # as ``localize`` builds it: the movie's metadata, then the
        # identification, then the fit
        region_info = region_info[:-1] + [identify_info] + [region_info[-1]]
        _localize_finish(
            locs,
            region_info,
            path,
            args,
            z_params,
            lateral_transforms,
            region_fit["spline_calibrations"][c],
            suffix=f"_{label}",
            fitting_method=region_fit["methods"][c],
        )


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
) -> None:
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
        for fitting (see ``_localize_process_file``).
    spline_calibration : dict or None
        Spline PSF calibration used for fitting, whose own lateral
        corrections are not applied again; None if not a spline fit.
    suffix : str, optional
        Inserted into the output name before ``_locs``, e.g. ``"_ref"`` for
        one region of a split field of view. Default "" (one file per movie).
    fitting_method : str or None, optional
        The method this set was fitted with, which decides the noise model
        of the astigmatic z fit. None (the default) uses ``args.fit_method``.
    """
    import warnings
    from os.path import splitext
    from . import lib
    from .io import save_locs
    from .localize import add_file_to_db

    method_name = fitting_method or args.fit_method
    if z_params is not None:
        from . import zfit

        zpath, magnification_factor, z_calibration = z_params
        z_calibration["Magnification Factor"] = magnification_factor
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
                fitting_method=method,
                filter=0,
                lateral_transforms=lateral_transforms,
                multiprocess=not args.fit_z_gpu,
                gpu=args.fit_z_gpu,
                progress_callback="console",
            )
        info[-1]["Z Calibration Path"] = zpath
        print("3D fitting complete.")
        print("------------------------------------------")
        _print_lateral_report(
            info[-1].get("Lateral corrections applied"),
            info[-1].get("Lateral corrections skipped"),
        )
    elif lateral_transforms:
        # No z fit to fold them into, so apply them here. Corrections the
        # spline calibration carries were applied by ``localize`` itself;
        # passing that same file to --affine-calibration would otherwise
        # correct the coordinates twice.
        extra, duplicates = lib.drop_duplicate_lateral_transforms(
            lateral_transforms, spline_calibration
        )
        if extra:
            locs = lib.apply_lateral_transforms(locs, extra)
            info[-1]["Lateral corrections applied"] = info[-1].get(
                "Lateral corrections applied", []
            ) + lib.describe_lateral_transforms(extra)
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
        print("Assesing quality and adding to DB")
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
        except Exception as e:
            print(e)
            print("Drift correction failed for {}".format(out_path))

    print("                                          ")


def _localize(args: argparse.Namespace) -> None:  # noqa: C901
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
        - ``gradient`` (list of float): minimum net gradient(s), one per
          ROI region or a single one for all.
        - ``roi`` (list of list of int): regions of interest, each
          ``[y_min, x_min, y_max, x_max]``.
        - ``frame_bounds`` (list of list of int): one or more
          ``[start_frame, end_frame]`` segments, 0-indexed and inclusive.
          Several segments restrict the analysis to the union of those
          (disjoint) frame ranges.
        - ``baseline``, ``sensitivity``, ``gain`` (float): camera
          parameters.
        - ``qe`` (float): not used in the calculations.
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
    fit_methods = args.fit_method or ["mle"]
    args.fit_method = fit_methods[0]
    gradients = args.gradient or [5000]
    args.gradient = gradients[0]
    spline_paths = args.spline_calibration or []
    args.spline_calibration = spline_paths[0] if spline_paths else ""
    if _FIT_METHOD_MAP[args.fit_method].endswith("-gpu"):
        if localize.CUDA_AVAILABLE:
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
        "Qe": args.qe,
    }

    # 0 means "let the fitting method use its own default", which differs
    # per model and per device - see localize.fit's eps / max_it.
    convergence = args.convergence
    max_iterations = args.max_iterations

    z_params = None
    if "-3d" in args.fit_method:
        z_params = _localize_load_3d_calibration(args)
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
    if any(method.startswith("spline") for method in fit_methods):
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
        region_fit = _localize_region_fit(
            fit_methods, spline_calibrations, n_regions
        )

    # Extra lateral affine corrections, applied after fitting on top of any
    # the 3D / spline calibration carries.
    lateral_transforms = []
    for affine_path in getattr(args, "affine_calibration", []) or []:
        from . import lib
        from .io import load_any_calibration

        found = lib.lateral_transforms(load_any_calibration(affine_path))
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
    for i, job in enumerate(jobs):
        _localize_process_file(
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
        progress_callback="console",
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
        calibration, test_movie, progress_callback="console"
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
    return spline.calibrate_spline_split_fov(
        movie,
        info=info,
        camera_info=camera_info,
        box=args.box_side_length,
        minimum_ng=args.gradient,
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
        progress_callback=lambda i: print(f"  step {i}/3"),
        wavelet=_wavelet_from_args(args),
    )


def _spline_calibrate_single(
    args: argparse.Namespace, files, camera_info, out_path
) -> dict:
    """Calibrate from a single-channel bead z-stack movie."""
    from . import spline
    from .io import load_movie

    movie, info = load_movie(files[0])
    return spline.calibrate_spline(
        movie,
        info=info,
        camera_info=camera_info,
        box=args.box_side_length,
        minimum_ng=args.gradient,
        d=args.step,
        frames_per_step=args.frames_per_step,
        frame_order=args.frame_order,
        z_binning=args.z_binning,
        model=args.model,
        magnification_factor=args.magnification_factor,
        correct_z_bias=args.correct_z_bias,
        path=out_path,
        progress_callback=lambda i: print(f"  step {i}/3"),
        wavelet=_wavelet_from_args(args),
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
        camera_infos.append(dict(camera_info))
    return spline.calibrate_spline_multichannel(
        movies,
        infos=infos,
        camera_infos=camera_infos,
        box=args.box_side_length,
        minimum_ng=args.gradient,
        d=args.step,
        frames_per_step=args.frames_per_step,
        frame_order=args.frame_order,
        z_binning=args.z_binning,
        magnification_factor=args.magnification_factor,
        correct_z_bias=args.correct_z_bias,
        photon_ratios=_parse_photon_ratios(args),
        model=registration,
        path=out_path,
        progress_callback=lambda i: print(f"  step {i}/3"),
        wavelet=_wavelet_from_args(args),
    )


def _spline_calibrate(args: argparse.Namespace) -> None:
    """Build a cubic-spline PSF calibration from a bead z-stack movie."""
    from os.path import splitext
    from . import spline

    picasso_logo()
    print("Spline PSF calibration")
    print("------------------------------------------")

    camera_info = {
        "Baseline": args.baseline,
        "Sensitivity": args.sensitivity,
        "Gain": args.gain,
        "Pixelsize": args.pixelsize,
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
        load_any_calibration(args.calibration) if args.calibration else {}
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
        calibration,
        box=args.box_side_length,
        minimum_ng=args.gradient,
        pixelsize=args.pixelsize,
        transform_type=args.type,
        ref_path=args.reference,
        target_path=args.target,
        model=args.model,
        wavelet=_wavelet_from_args(args),
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
        try:
            cmap = settings["Render"]["Colormap"]
        except KeyError:
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


def _parse_float_list(value) -> list:
    """Parse a CSV cell into a list of floats. Accepts a numeric scalar
    or a comma-separated string like ``"3,4,5"``."""
    if isinstance(value, bool):
        raise ValueError("Expected a number or comma-separated string.")
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _spinna_targets_from_row(row) -> list:
    """Derive target names from the ``exp_data_*`` columns of a CSV row.

    LE fitting does not load a structures yaml, so targets come from
    column names. The first non-empty ``exp_data_*`` column maps to
    ``target_a`` in ``spinna.fit_le``.
    """
    import pandas as pd

    prefix = "exp_data_"
    targets = [
        c[len(prefix) :]
        for c in row.index
        if c.startswith(prefix) and pd.notna(row[c])
    ]
    if len(targets) != 2:
        raise ValueError(
            "LE fitting requires exactly two targets (two non-empty "
            f"exp_data_* columns); found: {targets}"
        )
    return targets


def _spinna_parse_distances(row) -> list:
    """Parse the ``distances`` column of an LE-fitting row into a list
    of candidate heterodimer distances (nm)."""
    import pandas as pd

    if "distances" not in row.index or pd.isna(row["distances"]):
        raise ValueError("Column 'distances' is required when le_fitting=1.")
    distances = _parse_float_list(row["distances"])
    if not distances:
        raise ValueError("'distances' must contain at least one value.")
    return distances


def _spinna_validate_parameters(
    parameters_filename: str,
) -> tuple:
    """Validate the parameters file and create a result directory name.

    Also checks that all required columns are present; the result
    directory name is made unique.

    Parameters
    ----------
    parameters_filename : str
        Path to the ``.csv`` file with the SPINNA fitting parameters.

    Returns
    -------
    tuple[pd.DataFrame, str]
        ``(parameters, result_dir)``
    """
    import os
    import pandas as pd

    if not isinstance(parameters_filename, str):
        raise TypeError(
            "parameters_filename must be a string ending with .csv"
        )
    elif not parameters_filename.endswith(".csv"):
        raise TypeError("parameters_filename must end with .csv")

    parameters = pd.read_csv(parameters_filename)

    path, ext = os.path.splitext(parameters_filename)
    result_dir = path + "__fitting_results"
    if os.path.isdir(result_dir):
        i = 1
        while True:
            result_dir_ = result_dir + f"_{i}"
            if not os.path.isdir(result_dir_):
                result_dir = result_dir_
                break
            else:
                i += 1

    for column in [
        "granularity",
        "save_filename",
        "NND_bin",
        "NND_maxdist",
        "sim_repeats",
    ]:
        if column not in parameters.columns:
            raise ValueError(
                f"Column {column} not found in the parameters file."
            )

    return parameters, result_dir


def _spinna_check_target_columns(row, target: str, le_fitting: bool) -> None:
    """Raise if a target's required columns are missing from the row."""
    for col_name in [f"{_}_{target}" for _ in ["label_unc", "exp_data"]]:
        if col_name not in row.index:
            raise ValueError(
                f"Column {col_name} not found in the parameters file."
            )
    if not le_fitting and f"le_{target}" not in row.index:
        raise ValueError(
            f"Column le_{target} not found in the parameters file."
        )


def _spinna_parse_target_uncertainty(
    row, target: str, le_fitting: bool
) -> tuple:
    """Parse ``label_unc_TARGET`` and ``le_TARGET`` for one target."""
    if le_fitting:
        label_unc = _parse_float_list(row[f"label_unc_{target}"])
        if not label_unc:
            raise ValueError(
                f"label_unc_{target} must contain at least one value."
            )
        return label_unc, 1.0
    return float(row[f"label_unc_{target}"]), float(row[f"le_{target}"]) / 100


def _spinna_target_pixelsize(info) -> float:
    """Recover the pixel size (nm) from a locs info list, defaulting to
    130 when not found."""
    for element in info:
        # in newer versions it's Picasso vX.Y.Z Localize
        if "Picasso" in element.values() and "Localize" in element.values():
            if "Pixelsize" in element:
                return element["Pixelsize"]
    return 130


def _spinna_stack_exp_data(locs, pixelsize: float) -> tuple:
    """Stack loc coordinates (scaled to nm) into an (N, dim) array."""
    import numpy as np

    if "z" in locs.columns:
        return (
            np.stack((locs.x * pixelsize, locs.y * pixelsize, locs.z)).T,
            3,
        )
    return np.stack((locs.x * pixelsize, locs.y * pixelsize)).T, 2


def _spinna_load_target_data(
    row,
    targets: list,
    io,
    le_fitting: bool = False,
) -> tuple:
    """Load per-target experimental data and parameters from a CSV row.

    When ``le_fitting`` is True, ``label_unc_TARGET`` is parsed as a
    comma-separated list of candidates (single values are still accepted),
    ``le_TARGET`` is not read (LE is what is being fit), and
    ``n_simulated`` is the raw localization count (no LE division).

    Parameters
    ----------
    row : pd.Series
        One row of the parameters file.
    targets : list of str
        Names of the targets.
    io : module
        The ``picasso.io`` module, used to load the localizations.
    le_fitting : bool, optional
        Whether the labeling efficiency is fitted. Default is False.

    Returns
    -------
    tuple[dict, dict, dict, dict, int, dict]
        ``(label_unc, le, exp_data, n_simulated, dim, infos)``
    """
    label_unc: dict = {}
    le: dict = {}
    exp_data: dict = {}
    n_simulated: dict = {}
    infos: dict = {}
    dim = 2

    for target in targets:
        _spinna_check_target_columns(row, target, le_fitting)
        label_unc[target], le[target] = _spinna_parse_target_uncertainty(
            row, target, le_fitting
        )

        locs, info = io.load_locs(str(row[f"exp_data_{target}"]))
        infos[target] = info
        pixelsize = _spinna_target_pixelsize(info)
        exp_data[target], dim = _spinna_stack_exp_data(locs, pixelsize)

        if le_fitting:
            n_simulated[target] = len(locs)
        else:
            n_simulated[target] = int(len(locs) / le[target])

    return label_unc, le, exp_data, n_simulated, dim, infos


def _spinna_resolve_roi_3d(row) -> tuple:
    """Resolve a homogeneous 3D ROI (volume, z_range) from a row.

    Parameters
    ----------
    row : pd.Series
        One row of the parameters file.

    Returns
    -------
    tuple[float | None, float | None, bool]
        ``(volume, z_range, apply_mask)``
    """
    if "volume" not in row.index:
        return None, None, True
    volume = float(row["volume"])
    if "z_range" not in row.index:
        raise ValueError(
            "Column z_range not found in the parameters file."
            " 3D simulation was specified with homogeneous"
            " distribution. Please specify z_range."
        )
    return volume, float(row["z_range"]), False


def _spinna_resolve_roi_2d(row, targets: list, infos: dict | None) -> tuple:
    """Resolve a homogeneous 2D ROI (area) from a row.

    If the ``area`` column is missing or empty, the area is recovered from
    the experimental data metadata key ``"Area (um^2)"`` (taken from the
    first target's info).

    Parameters
    ----------
    row : pd.Series
        One row of the parameters file.
    targets : list of str
        Names of the targets.
    infos : dict or None
        Metadata of the experimental data per target.

    Returns
    -------
    tuple[float | None, bool]
        ``(area, apply_mask)``
    """
    import pandas as pd
    from . import lib

    if "area" in row.index and pd.notna(row["area"]):
        return float(row["area"]), False
    if infos:
        meta_area = lib.get_from_metadata(infos[targets[0]], "Area (um^2)")
        if meta_area is not None:
            return float(meta_area), False
    return None, True


def _spinna_resolve_mask_paths(row, targets: list) -> dict:
    """Collect and validate the per-target mask filenames from a row."""
    mask_paths = {}
    for target in targets:
        if f"mask_filename_{target}" not in row.index:
            raise ValueError(
                f"Column mask_filename_{target} not found in the"
                " parameters file."
            )
        mask_paths[target] = row[f"mask_filename_{target}"]
    return mask_paths


def _spinna_resolve_roi(
    row, dim: int, targets: list, infos: dict | None = None
) -> tuple:
    """Determine ROI parameters for a row: homogeneous or masked.

    Parameters
    ----------
    row : pd.Series
        One row of the parameters file.
    dim : int
        Dimensionality of the data, 2 or 3.
    targets : list of str
        Names of the targets.
    infos : dict or None, optional
        Metadata of the experimental data per target, used to recover
        the 2D area. Default is None.

    Returns
    -------
    tuple[bool, dict, float | None, float | None, float | None]
        ``(apply_mask, mask_paths, area, volume, z_range)``
    """
    apply_mask = True
    area = volume = z_range = None

    if dim == 3:
        volume, z_range, apply_mask = _spinna_resolve_roi_3d(row)
    elif dim == 2:
        area, apply_mask = _spinna_resolve_roi_2d(row, targets, infos)

    mask_paths = _spinna_resolve_mask_paths(row, targets) if apply_mask else {}
    return apply_mask, mask_paths, area, volume, z_range


def _spinna_compute_roi(
    targets: list,
    apply_mask: bool,
    mask_paths: dict,
    dim: int,
    area,
    volume,
    z_range,
) -> tuple:
    """Resolve the simulation ROI for a row.

    Parameters
    ----------
    targets : list of str
        Names of the targets.
    apply_mask : bool
        Whether masks are used instead of a homogeneous ROI.
    mask_paths : dict
        Path of the ``.npy`` mask per target.
    dim : int
        Dimensionality of the data, 2 or 3.
    area : float or None
        Area of the 2D homogeneous ROI in um^2.
    volume : float or None
        Volume of the 3D homogeneous ROI in um^3.
    z_range : float or None
        Depth of the 3D homogeneous ROI in nm.

    Returns
    -------
    tuple[dict | None, float | None, float | None, float | None]
        ``(mask_dict, width, height, depth)``
    """
    import os
    import yaml
    import numpy as np

    if apply_mask:
        masks: dict = {}
        mask_info: dict = {}
        width = height = depth = None
        for target in targets:
            masks[target] = np.load(mask_paths[target])
            mask_path = os.path.splitext(mask_paths[target])[0] + ".yaml"
            mask_info[target] = yaml.load(
                open(mask_path, "r"),
                Loader=yaml.FullLoader,
            )
        mask_dict = {"mask": masks, "info": mask_info}
    else:
        mask_dict = None
        if dim == 2:
            width = height = np.sqrt(area * 1e6)
            depth = None
        else:  # dim == 3
            depth = z_range
            width = height = np.sqrt(volume * 1e9 / depth)

    return mask_dict, width, height, depth


def _spinna_build_mixer(
    spinna,
    structures: list,
    targets: list,
    label_unc: dict,
    le: dict,
    random_rot_mode: str,
    apply_mask: bool,
    mask_paths: dict,
    dim: int,
    area,
    volume,
    z_range,
):
    """Build a ``StructureMixer`` from resolved ROI and target data."""
    mask_dict, width, height, depth = _spinna_compute_roi(
        targets,
        apply_mask,
        mask_paths,
        dim,
        area,
        volume,
        z_range,
    )
    return spinna.StructureMixer(
        structures=structures,
        label_unc=label_unc,
        le=le,
        mask_dict=mask_dict,
        width=width,
        height=height,
        depth=depth,
        random_rot_mode=random_rot_mode,
    )


def _spinna_roi_results(
    row, targets: list, apply_mask: bool, dim: int, area, volume, z_range
) -> dict:
    """Report the ROI (mask paths, or area/volume/z-range) used for a row."""
    if apply_mask:
        return {
            "File location of masks": [
                row[f"mask_filename_{target}"] for target in targets
            ]
        }
    if dim == 2:
        return {"Area (um^2)": area}
    if dim == 3:
        return {"Volume (um^3)": volume, "Z range (nm)": z_range}
    return {}


def _spinna_collect_le_fit_results(
    row,
    targets: list,
    structures: list,
    opt_props,
    score,
    label_unc: dict,
    random_rot_mode: str,
    dim: int,
    granularity,
    sim_repeats: int,
    apply_mask: bool,
    area,
    volume,
    z_range,
    label_unc_search: dict | None,
    distances_search: list | None,
    best_distance: float | None,
    le_values: dict | None,
) -> dict:
    """Assemble the results dict for an LE-fitting row."""
    results: dict = {
        "Molecular targets": targets,
        "File location of experimental data": [
            str(row[f"exp_data_{target}"]) for target in targets
        ],
        "Parameters search space granularity": granularity,
        "Dimensionality": f"{dim}D",
        "Rotation mode": random_rot_mode,
        "Number of simulation repeats": sim_repeats,
    }
    if label_unc_search is not None:
        for target in targets:
            results[f"Label-uncertainty search space (nm) for {target}"] = (
                ", ".join(f"{float(v):.2f}" for v in label_unc_search[target])
            )
    for target in targets:
        results[f"Fitted label uncertainty (nm) for {target}"] = (
            f"{float(label_unc[target]):.4f}"
        )
    if distances_search is not None:
        results["Heterodimer distance search space (nm)"] = ", ".join(
            f"{float(v):.2f}" for v in distances_search
        )
    if best_distance is not None:
        results["Fitted heterodimer distance (nm)"] = (
            f"{float(best_distance):.4f}"
        )
    if le_values is not None:
        for target in targets:
            results[f"Fitted labeling efficiency (%) for {target}"] = (
                f"{float(le_values[target]):.2f}"
            )
    results["Best fitting structure proportions (%)"] = ", ".join(
        f"{s.title}: {float(p):.2f}" for s, p in zip(structures, opt_props)
    )
    results["Modified Kolmogorov-Smirnov score"] = score
    results.update(
        _spinna_roi_results(
            row, targets, apply_mask, dim, area, volume, z_range
        )
    )
    return results


def _spinna_relative_proportions(
    targets: list, structures: list, mixer, opt_props, n_simulated: dict
) -> dict:
    """Report each target's relative proportion across structures."""
    import numpy as np

    opt_props_ = opt_props[0] if isinstance(opt_props, tuple) else opt_props
    results = {}
    for target in targets:
        rel_props = mixer.convert_props_for_target(
            opt_props_, target, n_simulated
        )
        idx_valid = np.where(rel_props != np.inf)[0]
        value = ", ".join(
            [f"{structures[i].title}: {rel_props[i]:.2f}%" for i in idx_valid]
        )
        results[f"Relative proportions of {target} in"] = value
    return results


def _spinna_collect_structure_results(
    row,
    targets: list,
    structures: list,
    mixer,
    opt_props,
    score,
    label_unc: dict,
    le: dict,
    random_rot_mode: str,
    dim: int,
    granularity,
    N_structures: dict,
    sim_repeats: int,
    apply_mask: bool,
    area,
    volume,
    z_range,
    n_simulated: dict,
) -> dict:
    """Assemble the results dict for a structure-fitting row."""
    results: dict = {
        "File location of structures": row["structures_filename"],
        "Molecular targets": targets,
        "File location of experimenal data": [
            str(row[f"exp_data_{target}"]) for target in targets
        ],
        "Labeling efficiency (%)": [le[target] * 100 for target in targets],
        "Label uncertainty (nm)": list(label_unc.values()),
        "Rotation mode": random_rot_mode,
        "Dimensionality": f"{dim}D",
        "Parameters search space granularity": granularity,
        "Fitted structures names": list(N_structures.keys()),
        "Number of simulation repeats": sim_repeats,
    }

    if isinstance(opt_props, tuple):
        props_mean, props_std = opt_props
        results["Modified Kolmogorov-Smirnov score +/- s.d."] = score
        results["Fitted proportions of structures"] = ", ".join(
            [
                f"{props_mean[i]:.2f} +/- {props_std[i]:.2f}%"
                for i in range(len(props_mean))
            ]
        )
    else:
        results["Modified Kolmogorov-Smirnov score"] = score
        results["Fitted proportions of structures"] = opt_props

    if len(targets) > 1:
        results.update(
            _spinna_relative_proportions(
                targets, structures, mixer, opt_props, n_simulated
            )
        )

    results.update(
        _spinna_roi_results(
            row, targets, apply_mask, dim, area, volume, z_range
        )
    )
    return results


def _spinna_collect_results(
    row,
    targets: list,
    structures: list,
    mixer,
    opt_props,
    score,
    label_unc: dict,
    le: dict,
    random_rot_mode: str,
    dim: int,
    granularity,
    N_structures: dict,
    sim_repeats: int,
    apply_mask: bool,
    mask_paths: dict,
    area,
    volume,
    z_range,
    n_simulated: dict,
    spinna,
    le_fitting: bool = False,
    label_unc_search: dict | None = None,
    distances_search: list | None = None,
    best_distance: float | None = None,
    le_values: dict | None = None,
) -> dict:
    """Assemble the full results dict from fitting output.

    When ``le_fitting`` is True, the dict reports the recovered LE
    values, fitted label uncertainty and heterodimer distance (plus the
    search spaces used), mirroring the GUI's Fit LE summary keys.
    """
    from datetime import datetime

    results = {"Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    if le_fitting:
        results.update(
            _spinna_collect_le_fit_results(
                row,
                targets,
                structures,
                opt_props,
                score,
                label_unc,
                random_rot_mode,
                dim,
                granularity,
                sim_repeats,
                apply_mask,
                area,
                volume,
                z_range,
                label_unc_search,
                distances_search,
                best_distance,
                le_values,
            )
        )
    else:
        results.update(
            _spinna_collect_structure_results(
                row,
                targets,
                structures,
                mixer,
                opt_props,
                score,
                label_unc,
                le,
                random_rot_mode,
                dim,
                granularity,
                N_structures,
                sim_repeats,
                apply_mask,
                area,
                volume,
                z_range,
                n_simulated,
            )
        )

    return results


def _spinna_plot_nnd(
    spinna,
    mixer,
    targets: list,
    exp_data: dict,
    opt_props,
    n_simulated: dict,
    sim_repeats: int,
    NND_bin: float,
    NND_maxdist: float,
    nn_plotted: int,
    save_filename: str,
) -> None:
    """Compute and save NND plots for all target pairs."""
    import matplotlib.pyplot as plt

    nn_counts = {
        f"{t1}-{t2}": nn_plotted
        for i, t1 in enumerate(targets)
        for t2 in targets[i:]
    }
    mixer.nn_counts = nn_counts
    n_total = sum(n_simulated.values())

    opt_for_counts = (
        opt_props[0] if isinstance(opt_props, tuple) else opt_props
    )
    dist_sim = spinna.get_NN_dist_simulated(
        mixer.convert_props_to_counts(opt_for_counts, n_total),
        sim_repeats,
        mixer,
        duplicate=True,
    )

    for i, (t1, t2, _) in enumerate(mixer.get_neighbor_idx(duplicate=True)):
        fig, ax = spinna.plot_NN(
            dist=dist_sim[i],
            mode="plot",
            show_legend=False,
            return_fig=True,
            figsize=(4.947, 3.71),
            alpha=1.0,
            binsize=NND_bin,
            xlim=[0, NND_maxdist],
            title=f"Nearest Neighbors Distances: {t1} -> {t2}",
        )
        fig, ax = spinna.plot_NN(
            data1=exp_data[t1],
            data2=exp_data[t2],
            n_neighbors=nn_plotted,
            show_legend=False,
            fig=fig,
            ax=ax,
            mode="hist",
            return_fig=True,
            binsize=NND_bin,
            xlim=[0, NND_maxdist],
            title=f"Nearest Neighbors Distances: {t1} -> {t2}",
            savefig=[
                f"{save_filename}_NND_{t1}_{t2}.{_}" for _ in ["png", "svg"]
            ],
        )
        # release the figure; otherwise pyplot keeps every figure of
        # every batch row in memory (growing RAM + "too many figures")
        plt.close(fig)


def _spinna_row_rotation_mode(row) -> str:
    """Read ``rotation_mode`` from a row, defaulting to "2D"."""
    if "rotation_mode" in row.index:
        if not isinstance(row["rotation_mode"], str):
            print("Invalid rotation_mode. Using default: 2D")
        else:
            return str(row["rotation_mode"])
    return "2D"


def _spinna_row_nn_plotted(row) -> int:
    """Read ``nn_plotted`` from a row, defaulting to 4."""
    if "nn_plotted" in row.index:
        if not isinstance(row["nn_plotted"], int):
            print("Invalid nn_plotted. Using default: 4")
        else:
            return int(row["nn_plotted"])
    return 4


def _spinna_row_fitting_mode(row) -> str | None:
    """Read ``fitting_mode`` from a row.

    None means "use the per-branch default" (bayesian for standard
    SPINNA, coarse-to-fine for LE fitting).
    """
    import pandas as pd

    if "fitting_mode" not in row.index or pd.isna(row["fitting_mode"]):
        return None
    mode = str(row["fitting_mode"]).strip()
    if mode in ("coarse-to-fine", "bayesian", "brute-force"):
        return mode
    print(
        f"Invalid fitting_mode '{mode}'. Must be one of "
        "'coarse-to-fine', 'bayesian', 'brute-force'. Using default."
    )
    return None


def _spinna_row_targets_and_structures(
    index: int, row, spinna, le_fitting: bool
) -> tuple:
    """Resolve ``(structures, targets)`` for a row.

    ``structures`` is None when ``le_fitting`` is True, since LE fitting
    does not use a structures file.
    """
    import pandas as pd

    if le_fitting:
        return None, _spinna_targets_from_row(row)
    if "structures_filename" not in row.index or pd.isna(
        row["structures_filename"]
    ):
        raise ValueError(
            f"Row {index}: structures_filename is required when "
            "le_fitting != 1."
        )
    return spinna.load_structures(row["structures_filename"])


def _spinna_process_row(
    index: int,
    row,
    parameters,
    result_dir: str,
    io,
    spinna,
    asynch: bool,
    bootstrap: bool,
    verbose: bool,
) -> dict:
    """Run a single SPINNA analysis row and return the results dict."""
    import os
    import pandas as pd

    print(f"Running SPINNA on row {index+1} out of {len(parameters)}.")

    le_fitting = (
        "le_fitting" in row.index
        and pd.notna(row["le_fitting"])
        and int(row["le_fitting"]) == 1
    )

    granularity = row["granularity"]
    NND_bin = row["NND_bin"]
    NND_maxdist = row["NND_maxdist"]
    sim_repeats = row["sim_repeats"]
    save_filename, _ = os.path.splitext(row["save_filename"])
    save_filename = os.path.join(result_dir, os.path.basename(save_filename))

    random_rot_mode = _spinna_row_rotation_mode(row)
    nn_plotted = _spinna_row_nn_plotted(row)
    fitting_mode = _spinna_row_fitting_mode(row)

    structures, targets = _spinna_row_targets_and_structures(
        index, row, spinna, le_fitting
    )

    label_unc, le, exp_data, n_simulated, dim, infos = (
        _spinna_load_target_data(
            row,
            targets,
            io,
            le_fitting=le_fitting,
        )
    )
    apply_mask, mask_paths, area, volume, z_range = _spinna_resolve_roi(
        row, dim, targets, infos
    )

    if le_fitting:
        return _spinna_process_row_le(
            row=row,
            targets=targets,
            label_unc=label_unc,
            exp_data=exp_data,
            n_simulated=n_simulated,
            dim=dim,
            granularity=granularity,
            sim_repeats=sim_repeats,
            NND_bin=NND_bin,
            NND_maxdist=NND_maxdist,
            nn_plotted=nn_plotted,
            apply_mask=apply_mask,
            mask_paths=mask_paths,
            area=area,
            volume=volume,
            z_range=z_range,
            random_rot_mode=random_rot_mode,
            save_filename=save_filename,
            asynch=asynch,
            verbose=verbose,
            fitting_mode=fitting_mode,
            spinna=spinna,
        )

    N_structures = spinna.generate_N_structures(
        structures, n_simulated, granularity
    )

    mixer = _spinna_build_mixer(
        spinna,
        structures,
        targets,
        label_unc,
        le,
        random_rot_mode,
        apply_mask,
        mask_paths,
        dim,
        area,
        volume,
        z_range,
    )

    opt_props, score = spinna.SPINNA(
        mixer=mixer,
        gt_coords=exp_data,
        N_sim=sim_repeats,
    ).fit_stoichiometry(
        N_structures,
        fitting_mode=fitting_mode if fitting_mode is not None else "bayesian",
        save=f"{save_filename}_fit_scores.csv",
        asynch=asynch,
        bootstrap=bootstrap,
        callback="console" if verbose else None,
    )

    results = _spinna_collect_results(
        row,
        targets,
        structures,
        mixer,
        opt_props,
        score,
        label_unc,
        le,
        random_rot_mode,
        dim,
        granularity,
        N_structures,
        sim_repeats,
        apply_mask,
        mask_paths,
        area,
        volume,
        z_range,
        n_simulated,
        spinna,
    )

    with open(f"{save_filename}_fit_summary.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Results saved to {save_filename}_fit_summary.txt")

    _spinna_plot_nnd(
        spinna,
        mixer,
        targets,
        exp_data,
        opt_props,
        n_simulated,
        sim_repeats,
        NND_bin,
        NND_maxdist,
        nn_plotted,
        save_filename,
    )

    return results


def _spinna_process_row_le(
    *,
    row,
    targets: list,
    label_unc: dict,
    exp_data: dict,
    n_simulated: dict,
    dim: int,
    granularity,
    sim_repeats: int,
    NND_bin: float,
    NND_maxdist: float,
    nn_plotted: int,
    apply_mask: bool,
    mask_paths: dict,
    area,
    volume,
    z_range,
    random_rot_mode: str,
    save_filename: str,
    asynch: bool,
    verbose: bool,
    fitting_mode: str | None,
    spinna,
) -> dict:
    """LE-fitting branch of ``_spinna_process_row``: builds monomer/
    heterodimer structures via ``spinna.fit_le`` and recovers per-target
    LE from the fitted structure proportions."""
    import os

    distances = _spinna_parse_distances(row)
    mask_dict, width, height, depth = _spinna_compute_roi(
        targets,
        apply_mask,
        mask_paths,
        dim,
        area,
        volume,
        z_range,
    )

    # snapshot the search-space inputs before calling fit_le —
    # compare_models mutates label_unc in place
    label_unc_input = {t: list(v) for t, v in label_unc.items()}
    distances_input = list(distances)

    (
        le_values,
        fitted_label_unc,
        best_distance,
        score,
        best_props,
        best_mixer,
    ) = spinna.fit_le(
        target_a=targets[0],
        target_b=targets[1],
        exp_data=exp_data,
        granularity=int(granularity),
        label_unc=label_unc,
        distances=distances,
        N_sim=int(sim_repeats),
        mask_dict=mask_dict,
        width=width,
        height=height,
        depth=depth,
        random_rot_mode=random_rot_mode,
        asynch=asynch,
        savedir=os.path.dirname(save_filename),
        callback="console" if verbose else None,
        fitting_mode=(
            fitting_mode if fitting_mode is not None else "bayesian"
        ),
    )

    structures = best_mixer.structures
    results = _spinna_collect_results(
        row,
        targets,
        structures,
        best_mixer,
        best_props,
        score,
        fitted_label_unc,
        {t: 1.0 for t in targets},
        random_rot_mode,
        dim,
        granularity,
        {s.title: None for s in structures},
        sim_repeats,
        apply_mask,
        mask_paths,
        area,
        volume,
        z_range,
        n_simulated,
        spinna,
        le_fitting=True,
        label_unc_search=label_unc_input,
        distances_search=distances_input,
        best_distance=best_distance,
        le_values=le_values,
    )

    with open(f"{save_filename}_fit_summary.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Results saved to {save_filename}_fit_summary.txt")

    _spinna_plot_nnd(
        spinna,
        best_mixer,
        targets,
        exp_data,
        best_props,
        n_simulated,
        sim_repeats,
        NND_bin,
        NND_maxdist,
        nn_plotted,
        save_filename,
    )

    return results


def _spinna_batch_analysis(
    parameters_filename: str,
    asynch: bool = True,
    bootstrap: bool = False,
    verbose: bool = False,
) -> None:
    """SPINNA batch analysis. Results are automatically saved in the
    a new subfolder named "parameters_filename_fitting_results" in the
    folder where the parameters file is located. Parameters should be
    provided in .csv file. Each row in the file specifies one analysis
    run. The parameters (columns) are:

    - "structures_filename" : Name of the files with structures saved
        (.yaml). Required unless ``le_fitting=1``, in which case the
        monomer/heterodimer structures are built internally and targets
        are taken from the two ``exp_data_TARGET`` columns.
    - "exp_data_TARGET" : Name of the file with experimental data
        (.hdf5). Each target in the structures must have a
        corresponding column, for example, "exp_data_EGFR".
    - "le_TARGET" : Labeling efficiency (%) for each target. Each
        target in the structures must have a corresponding column,
        for example, "le_EGFR". Ignored when ``le_fitting=1``.
    - "label_unc_TARGET" : Label uncertainty (nm) for each target. Each
        target in the structures must have a corresponding column,
        for example: "label_unc_EGFR". When ``le_fitting=1``, this may
        be a comma-separated list of candidates (e.g. ``"3,4,5,6"``);
        a single value disables the per-target search.
    - "granularity" : Granularity used in parameters search space
        generation. The higher the value the more combinations of
        structure counts will be tested.
    - "sim_repeats" : Number of simulation repeats used for obtaining
        smoother NND histograms.
    - "save_filename" : Name of the file where the results will be
        saved.
    - "NND_bin" : Bin size (nm) for the nearest neighbor distance (NND)
        histogram (plotting only).
    - "NND_maxdist" : Maximum distance (nm) for the nearest neighbor
        distance (NND) histogram (plotting only).

    Depending on whether a homo- or heterogeneous (masked) distribution
    is used, the following columns must be present:
    * For homogeneous distribution:

    - "area" or "volume" : Area (2D simulation) or volume (3D
        simulation) of the simulated ROI (um^2 or um^3). For 2D rows,
        "area" is optional: if omitted, the area is read from the
        experimental data metadata key "Area (um^2)" (written by Picasso
        when picks/areas are saved).
    - "z_range" : Applicable only when "volume" is provided. Defines
        the range of z coordinates (nm) of simulated molecular targets.

    * For heterogeneous distribution:
    - "mask_filename_TARGET" : Name of the .npy file with the mask saved for
        each molecular target. Each target in the structures must have
        a corresponding column, for example, "mask_EGFR".

    Optional columns are:

    - "rotation_mode" : Random rotations mode used in analysis. Values
        must be one of {"3D", "2D", "None"}. Default: "2D".
    - "nn_plotted" : Number of nearest neighbors plotted in the NND.
        Only integer values are accepted. Default: 4.
    - "fitting_mode" : Optimization method used to fit the structure
        counts. Values must be one of {"coarse-to-fine", "bayesian",
        "brute-force"}. Default: "bayesian".
    - "le_fitting" : 0 if standard SPINNA is ran, 1 if labeling
        efficiency fitting is to be performed. If the column is not
        provided, standard SPINNA is ran. When set to 1, the batch
        calls ``picasso.spinna.fit_le``: monomer A, monomer B and
        heterodimer(d) structures are built internally for each
        candidate ``distances`` value, label uncertainty is fit per
        target from the comma-separated candidates in
        ``label_unc_TARGET``, and the per-target LE is recovered from
        the fitted structure proportions. Exactly two ``exp_data_*``
        columns must be present; the first column maps to ``target_a``.
        ``-b/--bootstrap`` is ignored on LE-fitting rows. For more
        details about the LE fitting, see Hellmeier, Strauss, et al.
        Nature Methods, 2024.
    - "distances" : Comma-separated list of candidate heterodimer
        distances in nm (e.g. ``"5,10,15,20"``). A single value fixes
        the distance. Required when ``le_fitting=1``; ignored
        otherwise.

    When saving, each analysis run index is used as the prefix for
    filename, for example, "analysis_run1_fit_summary.txt".

    Parameters
    ----------
    parameters_filename : str
        Path to the parameters file.
    asynch : bool (default=True)
        If True, multiprocessing is used.
    bootstrap : bool (default=False)
        If True, bootstrapping is used.
    verbose : bool (default=True)
        If True, progress bar for each row is printed to the console.
    """
    import os
    import pandas as pd
    from . import io, spinna

    parameters, result_dir = _spinna_validate_parameters(parameters_filename)
    os.makedirs(result_dir, exist_ok=True)

    summary = []
    for index, row in parameters.iterrows():
        results = _spinna_process_row(
            index,
            row,
            parameters,
            result_dir,
            io,
            spinna,
            asynch,
            bootstrap,
            verbose,
        )
        summary.append(results)

    summary = pd.DataFrame(summary)
    summary.to_csv(
        os.path.join(result_dir, "summary_results.csv"),
        index=False,
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
    asynch: bool = True,
    group_column: Literal["group", "group_input"] = "group",
) -> None:
    """G5M analysis of clustered localizations. See ``picasso.g5m.g5m``
    for details on the parameters."""
    from glob import glob
    from os.path import isdir, splitext
    import yaml
    from .io import load_locs, save_locs
    from .g5m import g5m

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
            with open(calibration, "r") as f:
                calib = yaml.full_load(f)
        mols, _, g5m_info = g5m(
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
            asynch=asynch,
            group_column=group_column,
            callback_parent="console",
        )
        new_path = splitext(path)[0] + "_molmap.hdf5"
        save_locs(new_path, mols, g5m_info)


# =============================================================================
# Plugins
# =============================================================================


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
        ],
        action="append",
        default=None,
        help=(
            "fitting method (default 'mle'). 'spline'/'spline-mle' fit an "
            "experimental cubic-spline PSF on the CPU by least squares / "
            "maximum likelihood, 'spline-gpu'/'spline-mle-gpu' do the same "
            "on the GPU (needs a CUDA-capable GPU); all four need "
            "--spline-calibration. With --regions-separately it may be "
            "given once per --roi, to fit each region its own way"
        ),
    )
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
        type=int,
        action="append",
        default=None,
        help=(
            "minimum net gradient (default 5000). May be given once per "
            "--roi, since regions imaged through different optics need not "
            "share a brightness scale"
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
        "-bl", "--baseline", type=int, default=0, help="camera baseline"
    )
    localize_parser.add_argument(
        "-s", "--sensitivity", type=float, default=1, help="camera sensitivity"
    )
    localize_parser.add_argument(
        "-ga", "--gain", type=int, default=1, help="camera gain"
    )
    localize_parser.add_argument(
        "-qe", "--qe", type=float, default=1, help="camera quantum efficiency"
    )
    localize_parser.add_argument(
        "-mf",
        "--mf",
        type=float,
        default=0,
        help="magnification factor (3D only)",
    )
    localize_parser.add_argument(
        "-px", "--pixelsize", type=int, default=130, help="pixelsize in nm"
    )
    localize_parser.add_argument(
        "-zc",
        "--zc",
        type=str,
        default="",
        help="path to 3D calibration file (3D only)",
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

    # spline-calibrate: build a cubic-spline PSF calibration from a bead
    # z-stack movie for later use with the 'spline' fit methods.
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
            " pure xy shift), affine (6 DOF, the default), projective (8 DOF,"
            " adds the perspective term) or"
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
        "-px", "--pixelsize", type=int, default=130, help="pixelsize in nm"
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
            "transform model: translation (2 DOF, a pure xy shift), affine"
            " (6 DOF, the default), projective (8 DOF), polynomial2 or"
            " polynomial3, needing at least 1 / 3 / 4 / 6 / 10 bead pairs"
            " respectively"
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
        default=None,
        help="camera pixel size in nm, for the reported shift in nm",
    )
    lateral_parser.add_argument(
        "-p",
        "--plot",
        type=str,
        default="",
        help="write the diagnostic figure to this path",
    )

    subparsers.add_parser("filter", help="filter raw files based on SNR (GUI)")

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
    subparsers.add_parser(
        "server", help="picasso server workflow management system"
    )

    # spinna
    _spinna_docs_url = docs_url("spinna.html#command-window-batch-analysis")
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
        "--asynch",
        action="store_false",
        help="do not perform fitting asynchronously (multiprocessing)",
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
    subparsers.add_parser(
        "average3",
        help="three-dimensional particle averaging (to be deprecated in 1.0)",
    )  # TODO: deprecate in 1.0

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
        "--fromfile",
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
        "--intersectdist",
        type=float,
        default=20 / 130,
        help=(
            "max. distance (cam. pixels) between localizations in"
            " consecutive segments to be considered as intersecting"
        ),
    )
    undrift_aim_parser.add_argument(
        "-r",
        "--roiradius",
        type=float,
        default=60 / 130,
        help="max. drift (cam. pixels) between two consecutive segments",
    )

    # undrift by fiducials parser
    undrift_fiducial_parser = subparsers.add_parser(
        "undrift_fiducials",
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
        default=1.0,
        help=(
            "maximum distance between localizations to consider them the same "
            "binding event in units of camera pixels (default=1.0)"
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

    # nneighbors
    nneighbor_parser = subparsers.add_parser(
        "nneighbor", help="calculate nearest neighbor of a clustered dataset"
    )
    nneighbor_parser.add_argument(
        "files",
        nargs="?",
        help=(
            "one or multiple hdf5 clustered files"
            " specified by a unix style path pattern"
        ),
    )

    clusterfilter_parser = subparsers.add_parser(
        "clusterfilter",
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
        "-c", "--clusterfile", help="a hdf5 clusterfile"
    )
    clusterfilter_parser.add_argument(
        "-p", "--parameter", type=str, help="parameter to be filtered"
    )
    clusterfilter_parser.add_argument(
        "--minval",
        type=float,
        help="lower boundary",
    )
    clusterfilter_parser.add_argument(
        "--maxval",
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
            "maximal distance between two localizations"
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
            "maximal distance (camera pixels) between two localizations"
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
        "pixelsize",
        type=int,
        help=("camera pixel size in nm (required for 3D localizations only)"),
        default=None,
    )
    dbscan_parser.add_argument(
        "--radius_z",
        type=float,
        help=(
            "DBSCAN epsilon in z (camera pixels). If set, enables "
            "anisotropic 3D clustering."
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
    hdbscan_parser.add_argument(
        "pixelsize",
        type=int,
        help=("camera pixel size in nm (required for 3D localizations only)"),
        default=None,
    )

    # SMLM clusterer
    smlm_cluster_parser = subparsers.add_parser(
        "smlm_cluster",
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
        help=("clustering radius (in camera pixels)"),
    )
    smlm_cluster_parser.add_argument(
        "min_locs",
        type=int,
        help=("minimum number of localizations in a cluster"),
    )
    smlm_cluster_parser.add_argument(
        "pixelsize",
        type=int,
        help=("camera pixel size in nm (required for 3D localizations only)"),
        default=None,
    )
    smlm_cluster_parser.add_argument(
        "basic_fa",
        type=bool,
        help=(
            "whether or not perform basic frame analysis (sticking event "
            "removal)"
        ),
        default=False,
    )
    smlm_cluster_parser.add_argument(
        "radius_z",
        type=float,
        help=("clustering radius in axial direction (MUST BE SET FOR 3D!!!)"),
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
        "--asynch",
        action="store_false",
        help="do not perform fitting asynchronously (multiprocessing)",
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
        "-d", "--display", help="display correlation", action="store_true"
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
        "--keepindex",
        help="do not change frame numbers",
        action="store_true",
    )

    # group properties
    groupprops_parser = subparsers.add_parser(
        "groupprops",
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
        "pc", help="calculate the pair-correlation of localizations"
    )
    pc_parser.add_argument(
        "-b",
        "--binsize",
        type=float,
        default=0.1,
        help="the bin size (camera pixels)",
    )
    pc_parser.add_argument(
        "-r",
        "--rmax",
        type=float,
        default=10,
        help="The maximum distance to calculate the pair-correlation",
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

    cluster_combine_parser = subparsers.add_parser(
        "cluster_combine",
        help=(
            "combine localization in each cluster of a group "
            "(to be deprecated in 1.0)"
        ),
    )
    cluster_combine_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )

    cluster_combine_dist_parser = subparsers.add_parser(
        "cluster_combine_dist",
        help=(
            "calculate the nearest neighbor for each combined cluster "
            "(to be deprecated in 1.0)"
        ),
    )
    cluster_combine_dist_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
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

    # Parse
    args = parser.parse_args()
    if args.command:
        # check for updates and print in the console if available
        from .updater import cli_notify_update, check_and_notify

        cli_update_check = True
        update_thread = None
        gui_apps = [
            "localize",
            "filter",
            "render",
            "average",
            "nanotron",
            "average3",
            "simulate",
            "design",
            "spinna",
        ]
        if args.command in gui_apps:
            if len(sys.argv) == 2:  # only the gui is opened
                cli_update_check = False
        if cli_update_check:
            update_thread = check_and_notify(cli_notify_update)

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
        elif args.command == "average3":
            from .gui import average3

            average3.main()
        elif args.command == "simulate":
            from .gui import simulate

            simulate.main()
        elif args.command == "design":
            from .gui import design

            design.main()
        elif args.command == "server":
            _start_server()
        elif args.command == "spinna":
            if args.columns:
                import inspect

                print(inspect.getdoc(_spinna_batch_analysis))
            elif args.parameters:
                _spinna_batch_analysis(
                    args.parameters,
                    args.asynch,
                    args.bootstrap,
                    args.verbose,
                )
            else:
                from .gui import spinna

                spinna.main()
        elif args.command == "link":
            _link(args.files, args.distance, args.tolerance)
        elif args.command == "clusterfilter":
            _clusterfilter(
                args.files,
                args.clusterfile,
                args.parameter,
                args.minval,
                args.maxval,
            )
        elif args.command == "undrift":
            _undrift_rcc(
                args.files,
                args.segmentation,
                args.display,
                args.fromfile,
            )
        elif args.command == "aim":
            _undrift_aim(
                args.files,
                args.segmentation,
                args.intersectdist,
                args.roiradius,
            )
        elif args.command == "undrift_fiducials":
            _undrift_fiducials(args.files)
        elif args.command == "density":
            _density(args.files, args.radius)
        elif args.command == "dbscan":
            _dbscan(
                args.files,
                args.radius,
                args.density,
                args.pixelsize,
                args.radius_z,
            )
        elif args.command == "hdbscan":
            _hdbscan(
                args.files,
                args.min_cluster,
                args.min_samples,
                args.pixelsize,
            )
        elif args.command == "smlm_cluster":
            _smlm_clusterer(
                args.files,
                args.radius,
                args.min_locs,
                args.pixelsize,
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
                args.asynch,
                args.group_column,
            )
        elif args.command == "nneighbor":
            _nneighbor(args.files)
        elif args.command == "dark":
            _dark(args.files)
        elif args.command == "align":
            _align(args.file, args.display)
        elif args.command == "join":
            _join(args.file, args.keepindex)
        elif args.command == "groupprops":
            _groupprops(args.files)
        elif args.command == "pc":
            _pair_correlation(args.files, args.binsize, args.rmax)
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
        elif args.command == "cluster_combine":
            _cluster_combine(args.files)
        elif args.command == "cluster_combine_dist":
            _cluster_combine_dist(args.files)
    else:
        parser.print_help()
        return

    # wait for the update check to finish before exiting
    if update_thread is not None:
        update_thread.join(timeout=6)


if __name__ == "__main__":
    main()
