"""
    __main__.py
    ~~~~~~~~~~~

    Picasso command line interface.

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss
    :copyright: Copyright (c) 2016-2019 Jungmann Lab, MPI of Biochemistry
"""

import os.path
import argparse
from . import __version__


def picasso_logo():
    """Print the Picasso logo to the console."""
    print("    ____  _____________   __________ ____ ")
    print("   / __ \\/  _/ ____/   | / ___/ ___// __ \\")
    print("  / /_/ // // /   / /| | \\__ \\\\__ \\/ / / /")
    print(" / _____/ // /___/ ___ |___/ ___/ / /_/ / ")
    print("/_/   /___/\\____/_/  |_/____/____/\\____/  ")
    print("                                          ")


def _average(args: argparse.Namespace) -> None:
    """Run Picasso: Average.

    Parameters
    ----------
    iterations : int
        Number of iterations for the averaging algorithm.
    oversampling : int
        Number of super-resolution pixels per camera pixel.
    file : list of str
        List of file paths to the localization files to be averaged.
    """
    from glob import glob
    from .io import load_locs, NoMetadataFileError
    from picasso.gui import average

    kwargs = {"iterations": args.iterations, "oversampling": args.oversampling}
    paths = glob(args.file)
    if paths:
        for path in paths:
            print("Averaging {}".format(path))
            try:
                locs, info = load_locs(path)
            except NoMetadataFileError:
                continue
            kwargs["path_basename"] = os.path.splitext(path)[0] + "_avg"
            average(locs, info, **kwargs)


def _hdf2visp(path: str, pixel_size: float) -> None:
    """Convert HDF5 localization files to VISP format.

    Parameters
    ----------
    path : str
        Path to the HDF5 localization files. The file will be saved
        under the same name with a `.3d` extension.
    pixel_size : float
        Camera pixel size in nanometers.
    """
    from glob import glob

    paths = glob(path)
    if paths:
        from .io import load_locs
        import os.path
        from numpy import savetxt

        for path in paths:
            print("Converting {}".format(path))
            locs, info = load_locs(path)
            locs = locs[["x", "y", "z", "photons", "frame"]].copy()
            locs.x *= pixel_size
            locs.y *= pixel_size
            outname = os.path.splitext(path)[0] + ".3d"
            savetxt(
                outname,
                locs,
                fmt=["%.1f", "%.1f", "%.1f", "%.1f", "%d"],
                newline="\r\n",
            )


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
    import pandas as pd

    paths = glob(path)
    if paths:
        from .io import save_locs
        import os.path
        import numpy as _np

        for path in _tqdm(paths):
            print("Converting {}".format(path))
            data = pd.read_csv(path)

            try:
                frames = data["frame"].astype(int)
                # make sure frames start at zero:
                frames = frames - _np.min(frames)
                x = data["x [nm]"] / pixelsize
                y = data["y [nm]"] / pixelsize
                photons = data["intensity [photon]"].astype(int)

                bg = data["offset [photon]"].astype(int)
                lpx = data["uncertainty_xy [nm]"] / pixelsize
                lpy = data["uncertainty_xy [nm]"] / pixelsize

                if "z_nm" in list(data):
                    # TODO update other column labels
                    z = data["z_nm"] / pixelsize
                    sx = data["sigma1_nm"] / pixelsize
                    sy = data["sigma2_nm"] / pixelsize

                    LOCS_DTYPE = [
                        ("frame", "u4"),
                        ("x", "f4"),
                        ("y", "f4"),
                        ("z", "f4"),
                        ("photons", "f4"),
                        ("sx", "f4"),
                        ("sy", "f4"),
                        ("bg", "f4"),
                        ("lpx", "f4"),
                        ("lpy", "f4"),
                    ]

                    locs = _np.rec.array(
                        (frames, x, y, z, photons, sx, sy, bg, lpx, lpy),
                        dtype=LOCS_DTYPE,
                    )

                else:
                    sx = data["sigma [nm]"] / pixelsize
                    sy = data["sigma [nm]"] / pixelsize

                    LOCS_DTYPE = [
                        ("frame", "u4"),
                        ("x", "f4"),
                        ("y", "f4"),
                        ("photons", "f4"),
                        ("sx", "f4"),
                        ("sy", "f4"),
                        ("bg", "f4"),
                        ("lpx", "f4"),
                        ("lpy", "f4"),
                    ]

                    locs = _np.rec.array(
                        (frames, x, y, photons, sx, sy, bg, lpx, lpy),
                        dtype=LOCS_DTYPE,
                    )

                locs.sort(kind="mergesort", order="frame")

                img_info = {}
                img_info["Generated by"] = f"Picasso v{__version__} csv2hdf"
                img_info["Frames"] = int(_np.max(frames)) + 1
                img_info["Height"] = int(_np.ceil(_np.max(y)))
                img_info["Width"] = int(_np.ceil(_np.max(x)))

                info = []
                info.append(img_info)

                base, ext = os.path.splitext(path)
                out_path = base + "_locs.hdf5"
                save_locs(out_path, locs, info)
                print("Saved to {}.".format(out_path))
            except Exception as e:
                print(e)
                print("Error. Datatype not understood.")
                raise e


def _hdf2csv(path: str) -> None:
    """Convert HDF5 localization files to CSV format."""
    from glob import glob
    import pandas as pd
    from tqdm import tqdm as _tqdm
    from os.path import isdir

    if isdir(path):
        paths = glob(path + "/*.hdf5")
    else:
        paths = glob(path)
    if paths:
        import os.path
        from .io import load_locs

        for path in _tqdm(paths):
            base, ext = os.path.splitext(path)
            if ext == ".hdf5":
                print("Converting {}".format(path))
                out_path = base + ".csv"
                locs = load_locs(path)[0]
                df = pd.DataFrame(locs)
                print("A total of {} rows loaded".format(len(locs)))
                df.to_csv(out_path, sep=",", encoding="utf-8")
    print("Complete.")


def _link(files: str, d_max: float, tolerance: float) -> None:
    """Link localizations in HDF5 files, see ``postprocess.link`` for
    details."""
    import numpy as _np
    from tqdm import tqdm as _tqdm
    from . import lib as _lib
    from h5py import File

    import glob

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

                clusters = _lib.append_to_rec(
                    clusters,
                    _np.array(n_after_link, dtype=_np.int32),
                    "n_after_link",
                )
                clusters = _lib.append_to_rec(
                    clusters,
                    _np.array(linked_len, dtype=_np.int32),
                    "linked_len",
                )
                clusters = _lib.append_to_rec(
                    clusters, _np.array(linked_n, dtype=_np.int32), "linked_n"
                )
                clusters = _lib.append_to_rec(
                    clusters,
                    _np.array(linked_photonrate, dtype=_np.float32),
                    "linked_photonrate",
                )
                with File(cluster_path, "w") as clusters_file:
                    clusters_file.create_dataset("clusters", data=clusters)
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
            combinedist_locs = postprocess.cluster_combine_dist(locs)
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
    from tqdm import tqdm
    import numpy as np

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
                selector = (clusters[parameter] > minval) & (
                    clusters[parameter] < maxval
                )
                if np.sum(selector) == 0:
                    print(
                        "Error: No localizations in range. Filtering aborted."
                    )
                elif np.sum(selector) == len(selector):
                    print(
                        "Error: All localizations in range. Filtering aborted."
                    )
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
                                all_locs, locs[locs["group"] == group],
                            )

                    base, ext = os.path.splitext(path)
                    clusterfilter_info = {
                        "Generated by": (
                            f"Picasso v{__version__} Clusterfilter - in"
                        ),
                        "Parameter": parameter,
                        "Minval": minval,
                        "Maxval": maxval,
                    }
                    info.append(clusterfilter_info)
                    all_locs.sort(kind="mergesort", order="frame")
                    all_locs = all_locs.view(np.recarray)
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
                                all_locs, locs[locs["group"] == group],
                            )

                    base, ext = os.path.splitext(path)
                    clusterfilter_info = {
                        "Generated by": (
                            f"Picasso v{__version__} Clusterfilter - out"
                        ),
                        "Parameter": parameter,
                        "Minval": minval,
                        "Maxval": maxval,
                    }
                    info.append(clusterfilter_info)
                    all_locs.sort(kind="mergesort", order="frame")
                    all_locs = all_locs.view(np.recarray)
                    out_path = base + "_filter_out.hdf5"
                    io.save_locs(out_path, all_locs, info)
                    print("Complete. Saved to: {}".format(out_path))

            except ValueError:
                print("Error: Field {} not found.".format(parameter))


def _undrift(
    files: str,
    segmentation: int,
    display: bool = True,
    fromfile: str | None = None,
) -> None:
    """Run RCC undrifting on the given files. See
    ``postprocess.undrift`` for details. Alternatively, it can read the
    drift .txt file to apply the drift correction."""
    import glob
    from . import io, postprocess
    from numpy import genfromtxt, savetxt

    paths = glob.glob(files)
    undrift_info = {"Generated by": f"Picasso v{__version__} Undrift"}
    if fromfile is not None:
        undrift_info["From File"] = fromfile
        drift = genfromtxt(fromfile)
    else:
        undrift_info["Segmentation"] = segmentation
    for path in paths:
        try:
            locs, info = io.load_locs(path)
        except io.NoMetadataFileError:
            continue
        if fromfile is not None:
            # this works for mingjies drift files but not for the own ones
            locs.x -= drift[:, 1][locs.frame]
            locs.y -= drift[:, 0][locs.frame]
            if display:
                import matplotlib.pyplot as plt

                plt.style.use("ggplot")
                plt.figure(figsize=(17, 6))
                plt.suptitle("Estimated drift")
                plt.subplot(1, 2, 1)
                plt.plot(drift[:, 1], label="x")
                plt.plot(drift[:, 0], label="y")
                plt.legend(loc="best")
                plt.xlabel("Frame")
                plt.ylabel("Drift (pixel)")
                plt.subplot(1, 2, 2)
                plt.plot(
                    drift[:, 1],
                    drift[:, 0],
                    color=list(plt.rcParams["axes.prop_cycle"])[2]["color"],
                )
                plt.axis("equal")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.show()
        else:
            print("Undrifting file {}".format(path))
            drift, locs = postprocess.undrift(
                locs, info, segmentation, display=display,
            )

            undrift_info["Drift X"] = float(drift["x"].mean())
            undrift_info["Drift Y"] = float(drift["y"].mean())

        info.append(undrift_info)
        base, ext = os.path.splitext(path)
        io.save_locs(base + "_undrift.hdf5", locs, info)
        savetxt(base + "_drift.txt", drift, header="dx\tdy", newline="\r\n")


def _undrift_aim(
    files: str,
    segmentation: int,
    intersectdist: float = 20/130,
    roiradius: float = 60/130
) -> None:
    """Run AIM undrifting on the given files. See ``aim.aim`` for
    details."""
    import glob
    from . import io, aim
    from numpy import savetxt

    paths = glob.glob(files)
    for path in paths:
        try:
            locs, info = io.load_locs(path)
        except io.NoMetadataFileError:
            continue
        print("Undrifting file {}".format(path))
        locs, new_info, drift = aim.aim(
            locs, info, segmentation, intersectdist, roiradius,
        )
        base, ext = os.path.splitext(path)
        io.save_locs(base + "_aim.hdf5", locs, new_info)
        savetxt(base + "_aimdrift.txt", drift, header="dx\tdy", newline="\r\n")


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
            locs = clusterer.dbscan(locs, radius, min_density, pixelsize)
            clusters = clusterer.find_cluster_centers(locs, pixelsize)
            base, _ = os.path.splitext(path)
            dbscan_info = {
                "Generated by": f"Picasso v{__version__} DBSCAN",
                "Radius": radius,
                "Minimum local density": min_density,
            }
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
            locs = clusterer.hdbscan(locs, min_cluster, min_samples, pixelsize)
            clusters = clusterer.find_cluster_centers(locs, pixelsize)
            base, ext = os.path.splitext(path)
            hdbscan_info = {
                "Generated by": f"Picasso v{__version__} HDBSCAN",
                "Min. cluster": min_cluster,
                "Min. samples": min_samples,
            }
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
            locs = clusterer.cluster(locs, **params, pixelsize=pixelsize)
            clusters = clusterer.find_cluster_centers(locs, pixelsize)
            base, ext = os.path.splitext(path)
            smlm_cluster_info = {
                "Generated by": f"Picasso v{__version__} SMLM clusterer",
                "Radius_xy": radius,
                "Radius_z": radius_z,
                "Min locs": min_locs,
                "Basic frame analysis": basic_fa,
            }
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
    import h5py as _h5py
    import numpy as np
    from scipy.spatial import distance

    paths = glob.glob(files)
    if paths:
        for path in paths:
            print("Loading {} ...".format(path))
            with _h5py.File(path, "r") as locs_file:
                locs = locs_file["clusters"][...]
            clusters = np.rec.array(locs, dtype=locs.dtype)
            points = np.array(clusters[["com_x", "com_y"]].tolist())
            alldist = distance.cdist(points, points)
            alldist[alldist == 0] = float("inf")
            minvals = np.amin(alldist, axis=0)
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
        "Generated by": f"Picasso v{__version__} Align", "Files": files,
    }
    for file, locs_, info in zip(files, aligned_locs, infos):
        info.append(align_info)
        base, ext = splitext(file)
        save_locs(base + "_align.hdf5", locs_, info)


def _join(files: list[str], keep_index: bool = True) -> None:
    """Join multiple localization files into one."""
    from .io import load_locs, save_locs
    from os.path import splitext
    from numpy import append
    import numpy as np

    locs, info = load_locs(files[0])
    total_frames = info[0]["Frames"]
    join_info = {
        "Generated by": f"Picasso v{__version__} Join", "Files": [files[0]],
    }
    for path in files[1:]:
        locs_, info_ = load_locs(path)
        try:
            n_frames = info[0]["Frames"]
            total_frames += n_frames
            if not keep_index:
                locs_["frame"] += total_frames - n_frames
            locs = append(locs, locs_)
            join_info["Files"].append(path)
        except TypeError:
            print(
                "An error occured.\n"
                "Unable to join files."
                " Make sure they have the same columns."
            )
    base, ext = splitext(files[0])
    info.append(join_info)
    if not keep_index:
        info[0]["Frames"] = total_frames
    locs.sort(kind="mergesort", order="frame")
    locs = locs.view(np.recarray)
    save_locs(base + "_join.hdf5", locs, info)


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
                base + "_groupprops.hdf5", info, locs=locs, groups=groups,
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


def _nanotron(args: argparse.Namespace) -> None:
    """Run nanoTRON prediction on localization files.

    Parameters
    ----------
    files : str
        Path to the localization files or a directory containing
        HDF5 files.
    model_path : str
        Path to the nanoTRON model file.
    model_pth : str
        Path to the nanoTRON model weights file.
    """
    from glob import glob
    from os.path import isdir
    from .io import load_locs, NoMetadataFileError

    files = args.files

    if isdir(files):
        print("Analyzing folder")
        paths = glob(files + "/*.hdf5")
        print("A total of {} files detected".format(len(paths)))
    else:
        paths = glob(files)

    if paths:
        for path in paths:
            print("nanoTRON predicting {}".format(path))
            try:
                locs, info = load_locs(path)
            except NoMetadataFileError:
                continue
            # TODO: Include call to proper prediction routine
            raise NotImplementedError
            # predict(locs, info, **kwargs)


def _localize(args: argparse.Namespace) -> None:
    """Localize molecules in microscopy images.

    Parameters
    ----------
    files : str
        Path to the microscopy image files or a directory containing
        image files.
    fit_method : str
        Method to use for fitting localizations. Options are:
        - 'mle': Maximum Likelihood Estimation
        - 'lq-3d': LQ 3D fitting
        - 'lq-gpu-3d': LQ GPU 3D fitting
    box_side_length : int
        Side length of the box used for localization.
    gradient : float
        Minimum net gradient for localization.
    roi : list of int
        Region of interest defined as [y_min, x_min, y_max, x_max].
    baseline : float
        Baseline value for the camera.
    sensitivity : float
        Sensitivity of the camera.
    gain : float
        Gain of the camera.
    qe : float
        Not used in the calculations.
    """
    files = args.files
    from glob import glob
    from .io import load_movie, save_locs, save_info
    from .localize import (
        get_spots,
        identify_async,
        identifications_from_futures,
        fit_async,
        locs_from_fits,
        add_file_to_db,
    )
    from os.path import splitext, isdir
    from time import sleep
    from . import gausslq, avgroi
    import os.path as _ospath
    import re as _re
    import os as _os
    import yaml as yaml

    picasso_logo()
    print("Localize - Parameters:")
    print("{:<8} {:<15} {:<10}".format("No", "Label", "Value"))

    if args.fit_method == "lq-gpu":
        if gausslq.gpufit_installed:
            print("GPUfit installed")
        else:
            raise Exception("GPUfit not installed. Aborting.")

    for index, element in enumerate(vars(args)):
        try:
            print("{:<8} {:<15} {:<10}".format(
                index + 1, element, getattr(args, element)
            ))
        except TypeError:  # if None is default value
            print("{:<8} {:<15} {}".format(index + 1, element, "None"))
    print("------------------------------------------")

    def check_consecutive_tif(filepath):
        """
        Function to only return the first file of a consecutive ome.tif series
        to not reconstruct all of them as load_movie automatically detects
        consecutive files. E.g. have a folder with file.ome.tif,
        file_1.ome.tif, file_2.ome.tif, will return only file.ome.tif
        Or NDTiffStacks where files have format file.tif, file_1.tif, etc.
        """
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

    if isdir(files):
        print("Analyzing folder")

        tif_files = check_consecutive_tif(files)

        paths = tif_files + glob(files + "/*.raw") + glob(files + "/*.nd2")
        print("A total of {} files detected".format(len(paths)))
    else:
        paths = glob(files)

    # Check for raw files: make sure that each contains a yaml file
    def prompt_info():
        info = {}
        info["Byte Order"] = input("Byte Order (< or >): ")
        info["Data Type"] = input('Data Type (e.g. "uint16"): ')
        info["Frames"] = int(input("Frames: "))
        info["Height"] = int(input("Height: "))
        info["Width"] = int(input("Width: "))
        save = input("Use for all remaining raw files in folder (y/n)?") == "y"
        return info, save

    save = False
    for path in paths:
        base, ext = _ospath.splitext(path)
        if ext == ".raw":
            if not _os.path.isfile(base + ".yaml"):
                print("No yaml found for {}. Please enter:".format(path))
                if not save:
                    info, save = prompt_info()
                info_path = base + ".yaml"
                save_info(info_path, [info])

    if paths:
        print(args)
        box = args.box_side_length
        min_net_gradient = args.gradient
        roi = args.roi
        if roi is not None:
            y_min, x_min, y_max, x_max = roi
            roi = [[y_min, x_min], [y_max, x_max]]
        camera_info = {}
        camera_info["Baseline"] = args.baseline
        camera_info["Sensitivity"] = args.sensitivity
        camera_info["Gain"] = args.gain
        camera_info["Qe"] = args.qe

        if args.fit_method == "mle":
            # use default settings
            convergence = 0.001
            max_iterations = 1000
        else:
            convergence = 0
            max_iterations = 0

        if args.fit_method == "lq-3d" or args.fit_method == "lq-gpu-3d":
            from . import zfit

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
                magnification_factor = float(
                    input("Enter Magnification factor: ")
                )
            else:
                magnification_factor = args.mf

            try:
                with open(zpath, "r") as f:
                    z_calibration = yaml.full_load(f)
            except Exception as e:
                print(e)
                print("Error loading calibration file.")
                raise

        for i, path in enumerate(paths):
            print("------------------------------------------")
            print("------------------------------------------")
            print(f"Processing {path}, File {i + 1} of {len(paths)}")
            print("------------------------------------------")
            movie, info = load_movie(path)
            current, futures = identify_async(
                movie, min_net_gradient, box, roi=roi,
            )
            n_frames = len(movie)
            while current[0] < n_frames:
                print(
                    f"Identifying in frame {current[0] + 1} of {n_frames}",
                    end="\r",
                )
                sleep(0.2)
            print(f"Identifying in frame {n_frames} of {n_frames}")
            ids = identifications_from_futures(futures)

            if args.fit_method == "lq" or args.fit_method == "lq-3d":
                spots = get_spots(movie, ids, box, camera_info)
                theta = gausslq.fit_spots_parallel(spots, asynch=False)
                locs = gausslq.locs_from_fits(ids, theta, box, args.gain)
            elif args.fit_method == "lq-gpu" or args.fit_method == "lq-gpu-3d":
                spots = get_spots(movie, ids, box, camera_info)
                theta = gausslq.fit_spots_gpufit(spots)
                em = camera_info["Gain"] > 1
                locs = gausslq.locs_from_fits_gpufit(ids, theta, box, em)
            elif args.fit_method == "mle":
                current, thetas, CRLBs, likelihoods, iterations = fit_async(
                    movie, camera_info, ids, box, convergence, max_iterations
                )
                n_spots = len(ids)
                while current[0] < n_spots:
                    print(
                        f"Fitting spot {current[0] + 1} of {n_spots}",
                        end="\r",
                    )
                    sleep(0.2)
                print(f"Fitting spot {n_spots} of {n_spots}")
                locs = locs_from_fits(
                    ids, thetas, CRLBs, likelihoods, iterations, box,
                )

            elif args.fit_method == "avg":
                spots = get_spots(movie, ids, box, camera_info)
                theta = avgroi.fit_spots_parallel(spots, asynch=False)
                locs = avgroi.locs_from_fits(ids, theta, box, args.gain)

            else:
                print("This should never happen...")

            try:
                px = args.pixelsize
            except Exception:
                px = None

            localize_info = {
                "Generated by": f"Picasso v{__version__} Localize",
                "ROI": None,  # TODO: change if ROI is given
                "Box Size": box,
                "Min. Net Gradient": min_net_gradient,
                "Pixelsize": px,
                "Fit method": args.fit_method
            }
            localize_info.update(camera_info)
            if args.fit_method == "mle":
                localize_info["Convergence Criterion"] = convergence
                localize_info["Max. Iterations"] = max_iterations

            if args.fit_method == "lq-3d" or args.fit_method == "lq-gpu-3d":
                print("------------------------------------------")
                print("Fitting 3D...", end="")
                fs = zfit.fit_z_parallel(
                    locs,
                    info,
                    z_calibration,
                    magnification_factor,
                    filter=0,
                    asynch=True,
                )
                locs = zfit.locs_from_futures(fs, filter=0)
                localize_info["Z Calibration Path"] = zpath
                localize_info["Z Calibration"] = z_calibration
                print("complete.")
                print("------------------------------------------")

            info.append(localize_info)
            info.append(camera_info)

            base, ext = splitext(path)

            try:
                sfx = args.suffix
            except Exception:
                sfx = ""

            out_path = f"{base}{sfx}_locs.hdf5"
            save_locs(out_path, locs, info)
            print("File saved to {}".format(out_path))

            if hasattr(args, "database"):
                CHECK_DB = args.database
            else:
                CHECK_DB = False

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
                    _undrift(
                        out_path, args.drift, display=False, fromfile=None,
                    )
                except Exception as e:
                    print(e)
                    print("Drift correction failed for {}".format(out_path))

            print("                                          ")
    else:
        print("Error. No files found.")
        raise FileNotFoundError


def _render(args: argparse.Namespace) -> None:
    """Render localization files to images.

    Parameters
    ----------
    files : str
        Path to the localization files or a directory containing
        HDF5 files.
    oversampling : int
        Number of super-resolution pixels per camera pixel.
    blur_method : str
        Defines localizations' blur. The string has to be one of
        'gaussian', 'gaussian_iso', 'smooth', 'convolve'. If None, no
        blurring is applied.
    min_blur_width : float
        Minimum width of the blur kernel in pixels.
    vmin : float
        Minimum value for the color scale.
    vmax : float
        Maximum value for the color scale.
    cmap : str
        Colormap to use for rendering. If None, the colormap from
        user settings is used.
    scaling : str
        If 'yes', the image is scaled to the range [vmin, vmax].
        If 'no', the image is not scaled.
    silent : bool
        If True, the rendered images are not opened automatically.
    """
    from .lib import locs_glob_map
    from .render import render
    from os.path import splitext
    from matplotlib.pyplot import imsave
    from os import startfile
    from os.path import isdir
    from .io import load_user_settings, save_user_settings
    from tqdm import tqdm
    from glob import glob

    def render_many(
        locs,
        info,
        path,
        oversampling,
        blur_method,
        min_blur_width,
        vmin,
        vmax,
        scaling,
        cmap,
        silent,
    ):
        if blur_method == "none":
            blur_method = None
        N, image = render(
            locs,
            info,
            oversampling,
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
        if not silent:
            startfile(out_path)

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
                render_many,
                path,
                args=(
                    args.oversampling,
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
            render_many,
            args.files,
            args=(
                args.oversampling,
                args.blur_method,
                args.min_blur_width,
                args.vmin,
                args.vmax,
                args.scaling,
                cmap,
                args.silent,
            ),
        )


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
        (.yaml).
    - "exp_data_TARGET" : Name of the file with experimental data
        (.hdf5). Each target in the structures must have a
        corresponding column, for example, "exp_data_EGFR".
    - "le_TARGET" : Labeling efficiency (%) for each target. Each
        target in the structures must have a corresponding column,
        for example, "le_EGFR".
    - "label_unc_TARGET" : Label uncertainty (nm) for each target. Each
        target in the structures must have a corresponding column,
        for example: "label_unc_EGFR".
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
        simulation) of the simulated ROI (um^2 or um^3).
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
    - "le_fitting" : 0 if standard SPINNA is ran, 1 if labeling
        efficiency fitting is to be performed. Then, 100% LE is used in
        the pipeline and different output file is saved. If the column
        is not provided, standard SPINNA is ran. For more details about
        the LE fitting, see Hellmeier, Strauss, et al. Nature Methods
        2024.

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
    import yaml
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from . import io, spinna

    # open the parameters file
    if not isinstance(parameters_filename, str):
        raise TypeError(
            "parameters_filename must be a string ending with .csv"
        )
    elif not parameters_filename.endswith(".csv"):
        raise TypeError("parameters_filename must end with .csv")

    parameters = pd.read_csv(parameters_filename)

    # find the folder name for saving results
    result_dir = parameters_filename.replace(".csv", "_fitting_results")
    if os.path.isdir(result_dir):
        i = 1
        while True:
            result_dir_ = result_dir + f"_{i}"
            if not os.path.isdir(result_dir_):
                result_dir = result_dir_
                break
            else:
                i += 1

    # check that all columns (non-target specific) are present
    for column in [
        "structures_filename",
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

    # summary list of results for each simulation
    summary = []

    # run each row (analysis) one by one:
    for index, row in parameters.iterrows():
        print(f"Running SPINNA on row {index+1} out of {len(parameters)}.")
        # start by reading structures filename and creating the structures
        structures_filename = row["structures_filename"]
        structures, targets = spinna.load_structures(structures_filename)

        # get the target-independent parameters
        granularity = row["granularity"]
        NND_bin = row["NND_bin"]
        NND_maxdist = row["NND_maxdist"]
        sim_repeats = row["sim_repeats"]
        save_filename, _ = os.path.splitext(row["save_filename"])
        save_filename = os.path.join(result_dir, save_filename)

        # get the optional arguments
        random_rot_mode = "2D"
        if "rotation_mode" in row.index:
            if not isinstance(row["rotation_mode"], str):
                print("Invalid rotation_mode. Using default: 2D")
            else:
                random_rot_mode = str(row["rotation_mode"])

        nn_plotted = 4
        if "nn_plotted" in row.index:
            if not isinstance(row["nn_plotted"], int):
                print("Invalid nn_plotted. Using default: 4")
            else:
                nn_plotted = int(row["nn_plotted"])

        # initialize the input dictionaries that are target-specific
        label_unc = {}
        le = {}
        exp_data = {}
        n_simulated = {}

        # load data and parameters for each molecular target
        for target in targets:
            for col_name in [
                f"{_}_{target}"
                for _ in ["label_unc", "exp_data"]
            ]:
                if col_name not in row.index:
                    raise ValueError(
                        f"Column {col_name} not found in the parameters file."
                    )
            if (
                f"le_{target}" not in row.index
                and ("le_fitting" in row.index and row["le_fitting"] == 0)
            ):
                raise ValueError(
                    f"Column le_{target} not found in the parameters file."
                )

            # load label uncertainy and labeling efficiency
            label_unc[target] = float(row[f"label_unc_{target}"])
            le[target] = float(row[f"le_{target}"]) / 100

            # load experimental data
            locs, info = io.load_locs(str(row[f"exp_data_{target}"]))
            pixelsize = 130
            for element in info:
                if "Picasso Localize" in element.values():
                    if "Pixelsize" in element:
                        pixelsize = element["Pixelsize"]
                        break
            if hasattr(locs, "z"):
                exp_data[target] = np.stack(
                    (locs.x * pixelsize, locs.y * pixelsize, locs.z)
                ).T
                dim = 3
            else:
                exp_data[target] = np.stack(
                    (locs.x * pixelsize, locs.y * pixelsize)
                ).T
                dim = 2

            # number of simulated molecules (after labeling efficiency
            # correction)
            n_simulated[target] = int(len(locs) / le[target])

        # check if the distribution is homogeneous or heterogeneous
        apply_mask = True
        if dim == 3:  # 3D simulation
            area = None
            if "volume" in row.index:
                volume = float(row["volume"])
                apply_mask = False
                # find z range
                if "z_range" not in row.index:
                    raise ValueError(
                        "Column z_range not found in the parameters file."
                        " 3D simulation was specified with homogeneous"
                        " distribution. Please specify z_range."
                    )
                else:
                    z_range = float(row["z_range"])
        elif dim == 2:  # 2D simulation
            volume = None
            if "area" in row.index:
                area = float(row["area"])
                apply_mask = False

        # extract masks if area/volume is not provided
        if apply_mask:
            mask_paths = {}
            for target in targets:
                if f"mask_filename_{target}" not in row.index:
                    raise ValueError(
                        f"Column mask_filename_{target} not found in the"
                        " parameters file."
                    )
                else:
                    mask_paths[target] = row[f"mask_filename_{target}"]

        # if le fitting is ran (see the docstring above), set LE to 100%
        if "le_fitting" in row.index and row["le_fitting"] == 1:
            # check that the structures are valid

            le = {target: 1.0 for target in targets}

        # generate search space for fitting
        N_structures = spinna.generate_N_structures(
            structures, n_simulated, granularity
        )

        # set up StructureMixer
        if apply_mask:
            masks = {}
            mask_info = {}
            width = height = depth = None
            for target in targets:
                masks[target] = np.load(mask_paths[target])
                mask_info[target] = yaml.load(
                    open(mask_paths[target].replace(".npy", ".yaml"), "r"),
                    Loader=yaml.FullLoader,
                )
            mask_dict = {"mask": masks, "info": mask_info}
        else:
            mask_dict = None
            if dim == 2:
                width = height = np.sqrt(area * 1e6)
                depth = None
            elif dim == 3:
                depth = z_range
                width = height = np.sqrt(volume * 1e9 / depth)

        mixer = spinna.StructureMixer(
            structures=structures,
            label_unc=label_unc,
            le=le,
            mask_dict=mask_dict,
            width=width, height=height, depth=depth,
            random_rot_mode=random_rot_mode,
        )

        if not os.path.isdir(result_dir):  # create the save folder
            os.mkdir(result_dir)

        # set up and run fitting
        opt_props, score = spinna.SPINNA(
            mixer=mixer,
            gt_coords=exp_data,
            N_sim=sim_repeats,
        ).fit_stoichiometry(
            N_structures,
            save=f"{save_filename}_fit_scores.csv",
            asynch=asynch,
            bootstrap=bootstrap,
            callback="console" if verbose else None,
        )

        # save the results
        results = {}
        results["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results["File location of structures"] = structures_filename
        results["Molecular targets"] = targets
        results["File location of experimenal data"] = [
            str(row[f"exp_data_{target}"]) for target in targets
        ]
        results["Labeling efficiency (%)"] = [
            le[target] * 100 for target in targets
        ]
        results["Label uncertainty (nm)"] = list(label_unc.values())
        results["Rotation mode"] = random_rot_mode
        results["Dimensionality"] = f"{dim}D"
        results["Parameters search space granularity"] = granularity
        results["Fitted structures names"] = list(N_structures.keys())
        results["Number of simulation repeats"] = sim_repeats
        if isinstance(opt_props, tuple):
            props_mean, props_std = opt_props
            results["Modified Kolmogorov-Smirnov score +/- s.d."] = score
            results["Fitted proportions of structures"] = ", ".join([
                f"{props_mean[i]:.2f} +/- {props_std[i]:.2f}%"
                for i in range(len(props_mean))
            ])
        else:
            results["Modified Kolmogorov-Smirnov score"] = score
            results["Fitted proportions of structures"] = opt_props

        # relative proportions of structures for each target
        if len(targets) > 1:
            for target in targets:
                if isinstance(opt_props, tuple):
                    opt_props_ = opt_props[0]
                else:
                    opt_props_ = opt_props
                rel_props = mixer.convert_props_for_target(
                    opt_props_, target, n_simulated,
                )
                idx_valid = np.where(rel_props != np.inf)[0]
                value = ", ".join([
                    f"{structures[i].title}: {rel_props[i]:.2f}%"
                    for i in idx_valid
                ])
                results[f"Relative proportions of {target} in"] = value

        if apply_mask:
            results["File location of masks"] = [
                row[f"mask_filename_{target}"] for target in targets
            ]
        else:
            if dim == 2:
                results["Area (um^2)"] = area
            elif dim == 3:
                results["Volume (um^3)"] = volume
                results["Z range (nm)"] = z_range

        # if le fitting was ran, output the result
        if "le_fitting" in row.index and row["le_fitting"] == 1:
            le_values = spinna.get_le_from_props(structures, opt_props)
            results["Labeling efficiency fitting"] = (
                f"LE {targets[0]}: {le_values[targets[0]]:.1f}%,"
                f" LE {targets[1]}: {le_values[targets[1]]:.1f}%"
            )

        # save .txt with summary of the results
        with open(f"{save_filename}_fit_summary.txt", "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        print(f"Results saved to {save_filename}_fit_summary.txt")
        summary.append(results)

        # plot and save the NND plots
        nn_counts = {}
        for i, t1 in enumerate(targets):
            for t2 in targets[i:]:
                nn_counts[f"{t1}-{t2}"] = nn_plotted
        mixer.nn_counts = nn_counts
        n_total = sum(n_simulated.values())
        if isinstance(opt_props, tuple):
            dist_sim = spinna.get_NN_dist_simulated(
                mixer.convert_props_to_counts(opt_props[0], n_total),
                sim_repeats,
                mixer,
                duplicate=True,
            )
        else:
            dist_sim = spinna.get_NN_dist_simulated(
                mixer.convert_props_to_counts(opt_props, n_total),
                sim_repeats,
                mixer,
                duplicate=True,
            )
        for i, (t1, t2, _) in enumerate(
            mixer.get_neighbor_idx(duplicate=True)
        ):
            fig, ax = spinna.plot_NN(
                dist=dist_sim[i], mode='plot', show_legend=False,
                return_fig=True, figsize=(4.947, 3.71), alpha=1.0,
                binsize=NND_bin, xlim=[0, NND_maxdist],
                title=f"Nearest Neighbors Distances: {t1} -> {t2}",
            )
            exp1 = exp_data[t1]
            exp2 = exp_data[t2]
            fig, ax = spinna.plot_NN(
                data1=exp1, data2=exp2,
                n_neighbors=nn_plotted,
                show_legend=False, fig=fig, ax=ax, mode='hist',
                return_fig=True, binsize=NND_bin, xlim=[0, NND_maxdist],
                title=f"Nearest Neighbors Distances: {t1} -> {t2}",
                savefig=[
                    f"{save_filename}_NND_{t1}_{t2}.{_}"
                    for _ in ["png", "svg"]
                ],
            )

    # save the summary as .csv file
    summary = pd.DataFrame(summary)
    summary.to_csv(
        os.path.join(result_dir, "summary_results.csv"), index=False,
    )


def main():
    # Main parser
    parser = argparse.ArgumentParser("picasso")
    subparsers = parser.add_subparsers(dest="command")

    for command in ["toraw", "filter"]:
        subparsers.add_parser(command)

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
            "maximum distance between localizations"
            " to consider them the same binding event (default=1.0)"
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

    cluster_combine_parser = subparsers.add_parser(
        "cluster_combine",
        help="combine localization in each cluster of a group",
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
        help="calculate the nearest neighbor for each combined cluster",
    )
    cluster_combine_dist_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
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
    clusterfilter_parser.add_argument("clusterfile", help="a hdf5 clusterfile")
    clusterfilter_parser.add_argument(
        "parameter", type=str, help="parameter to be filtered"
    )
    clusterfilter_parser.add_argument(
        "minval", type=float, help="lower boundary",
    )
    clusterfilter_parser.add_argument(
        "maxval", type=float, help="upper boundary",
    )

    # undrift parser
    undrift_parser = subparsers.add_parser(
        "undrift", help="correct localization coordinates for drift"
    )
    undrift_parser.add_argument(
        "files",
        help=(
            "one or multiple hdf5 localization files"
            " specified by a unix style path pattern"
        ),
    )
    # undrift_parser.add_argument(
    #     "-m",
    #     "--mode",
    #     default="render",
    #     help='"std", "render" or "framepair")',
    # )
    undrift_parser.add_argument(
        "-s",
        "--segmentation",
        type=float,
        default=1000,
        help=(
            "the number of frames to be combined"
            " for one temporal segment (default=1000)"
        ),
    )
    undrift_parser.add_argument(
        "-f",
        "--fromfile",
        type=str,
        help="apply drift from specified file instead of computing it",
    )
    undrift_parser.add_argument(
        "-d",
        "--nodisplay",
        action="store_false",
        help="do not display estimated drift",
    )

    # undrift by AIM parser
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
        default=20/130,
        help=(
            "max. distance (cam. pixels) between localizations in"
            " consecutive segments to be considered as intersecting"
        ),
    )
    undrift_aim_parser.add_argument(
        "-r",
        "--roiradius",
        type=float,
        default=60/130,
        help=(
            "max. drift (cam. pixels) between two consecutive"
            " segments"
        ),
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
            "maximal distance between to localizations"
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
            "maximal distance (camera pixels) between to localizations"
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
        help=(
            "clustering radius in axial direction (MUST BE SET FOR 3D!!!)"
        ),
        default=None,
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
        "-b", "--binsize", type=float, default=0.1, help="the bin size"
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
            "specified by a unix style path pattern"
        ),
    )

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
        "-b", "--box-side-length", type=int, default=7, help="box side length"
    )
    localize_parser.add_argument(
        "-a",
        "--fit-method",
        choices=["mle", "lq", "lq-gpu", "lq-3d", "lq-gpu-3d", "avg"],
        default="mle",
    )
    localize_parser.add_argument(
        "-g", "--gradient", type=int, default=5000, help="minimum net gradient"
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
        default=None,
        help=(
            "ROI (y_min, x_min, y_max, x_max) in camera pixels;\n"
            "note the origin of the image is in the top left corner"
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
        help="Magnification factor (only 3d)",
    )
    localize_parser.add_argument(
        "-px", "--pixelsize", type=int, default=130, help="pixelsize in nm"
    )
    localize_parser.add_argument(
        "-zc",
        "--zc",
        type=str,
        default="",
        help="Path to 3d calibration file (only 3d)",
    )

    localize_parser.add_argument(
        "-sf",
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to files",
    )

    localize_parser.add_argument(
        "-db",
        "--database",
        action="store_true",
        help="do not add to database",
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
        "-o",
        "--oversampling",
        type=float,
        default=1.0,
        help="the number of super-resolution pixels per camera pixels",
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
        "simulate", help="simulate single molecule fluorescence data",
    )

    # nanotron
    nanotron_parser = subparsers.add_parser(
        "nanotron", help="segmentation with deep learning"
    )
    nanotron_parser.add_argument(
        "-m",
        "--model",
        nargs="?",
        help="a model file for prediction",
    )

    nanotron_parser.add_argument(
        "files",
        nargs="?",
        help="one localization file or a folder containing localization files"
        " specified by a unix style path pattern",
    )

    # average
    average_parser = subparsers.add_parser(
        "average", help="particle averaging",
    )
    average_parser.add_argument(
        "-o",
        "--oversampling",
        type=float,
        default=10,
        help=(
            "oversampling of the super-resolution images"
            " for alignment evaluation"
        ),
    )
    average_parser.add_argument("-i", "--iterations", type=int, default=20)
    average_parser.add_argument(
        "files",
        nargs="?",
        help="a localization file with grouped localizations",
    )

    subparsers.add_parser(
        "average3", help="three-dimensional particle averaging"
    )  # TODO: depracate in 1.0

    hdf2visp_parser = subparsers.add_parser("hdf2visp")
    hdf2visp_parser.add_argument("files")
    hdf2visp_parser.add_argument("pixelsize", type=float)

    csv2hdf_parser = subparsers.add_parser("csv2hdf")
    csv2hdf_parser.add_argument("files")
    csv2hdf_parser.add_argument("pixelsize", type=float)

    hdf2csv_parser = subparsers.add_parser("hdf2csv")
    hdf2csv_parser.add_argument("files")

    subparsers.add_parser(
        "server", help="picasso server workflow management system"
    )

    spinna_parser = subparsers.add_parser(
        "spinna",
        help=(
            "picasso single protein investigation via nearest neighbor "
            "analysis"
        ),
    )
    spinna_parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        help=(
            ".csv file containing the parameters for spinna batch analysis;"
            " see the documentation for details explaining the .csv file"
            " structure."
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

    # Parse
    args = parser.parse_args()
    if args.command:
        if args.command == "toraw":
            from .gui import toraw

            toraw.main()
        elif args.command == "localize":
            if args.files:
                _localize(args)
            else:
                from picasso.gui import localize

                localize.main()
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
            if args.files:
                _average(args)
            else:
                from .gui import average

                average.main()
        elif args.command == "nanotron":
            if args.files:
                _nanotron(args)
            else:
                from .gui import nanotron

                nanotron.main()
        elif args.command == "average3":
            from .gui import average3

            average3.main()
        elif args.command == "link":
            _link(args.files, args.distance, args.tolerance)
        elif args.command == "cluster_combine":
            _cluster_combine(args.files)
        elif args.command == "cluster_combine_dist":
            _cluster_combine_dist(args.files)
        elif args.command == "clusterfilter":
            _clusterfilter(
                args.files,
                args.clusterfile,
                args.parameter,
                args.minval,
                args.maxval,
            )
        elif args.command == "undrift":
            _undrift(
                args.files, args.segmentation, args.nodisplay, args.fromfile,
            )
        elif args.command == "aim":
            _undrift_aim(
                args.files,
                args.segmentation,
                args.intersectdist,
                args.roiradius,
            )
        elif args.command == "density":
            _density(args.files, args.radius)
        elif args.command == "dbscan":
            _dbscan(args.files, args.radius, args.density, args.pixelsize)
        elif args.command == "hdbscan":
            _hdbscan(
                args.files, args.min_cluster, args.min_samples, args.pixelsize,
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
        elif args.command == "simulate":
            from .gui import simulate

            simulate.main()
        elif args.command == "design":
            from .gui import design

            design.main()
        elif args.command == "hdf2visp":
            _hdf2visp(args.files, args.pixelsize)
        elif args.command == "csv2hdf":
            _csv2hdf(args.files, args.pixelsize)
        elif args.command == "hdf2csv":
            _hdf2csv(args.files)
        elif args.command == "server":
            _start_server()
        elif args.command == "spinna":
            if args.parameters:
                _spinna_batch_analysis(
                    args.parameters, args.asynch, args.bootstrap,
                )
            else:
                from .gui import spinna

                spinna.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
