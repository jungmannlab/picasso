"""
picasso.clusterer
~~~~~~~~~~~~~~~~~

Clusterer for single molecules optimized for DNA-PAINT.
Additionally, contains implementations of DBSCAN and HDBSCAN and other
clustering-related functions.

SMLM clusterer is based on:
* Schlichthaerle, et al. Nature Comm, 2021
  (DOI: 10.1038/s41467-021-22606-1)
* Reinhardt, Masullo, Baudrexel, Steen, et al. Nature, 2023
  (DOI: 10.1038/s41586-023-05925-9)

:authors: Rafal Kowalewski, Susanne Reinhardt,
    Thomas Schlichthaerle, 2020-2025
:copyright: Copyright (c) 2022-2025 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import ConvexHull, KDTree, QhullError
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN, HDBSCAN

from . import lib, masking


def _frame_analysis(frame: pd.SeriesGroupBy, n_frames: int) -> int:
    """Verify which clusters pass basic frame analysis. Reject clusters
    whose mean frame is outside of the [20, 80] % (max frame) range or
    any 1/20th of measurement's time contains more than 80 % of
    localizations.

    Assumes frame to be a ``pandas.SeriesGroupBy`` object, grouped by
    cluster ids.

    Parameters
    ----------
    frame : pandas.SeriesGroupBy
        Frame number for a given cluster; grouped by cluster ids.
    n_frames : int
        Acquisition time given in frames.

    Returns
    -------
    passed : int
        1 if passed frame analysis, 0 otherwise.
    """
    passed = 1

    # get mean frame
    mean_frame = frame.values.mean()

    # get maximum number of locs in a 1/20th of acquisition time
    n_locs = len(frame)
    locs_binned = np.histogram(
        frame.values, bins=np.linspace(0, n_frames, 21)
    )[0]
    max_locs_bin = locs_binned.max()

    # test if frame analysis passed
    if (
        (mean_frame < 0.2 * n_frames)
        or (mean_frame > 0.8 * n_frames)
        or (max_locs_bin > 0.8 * n_locs)
    ):
        passed = 0

    return passed


def frame_analysis(labels: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Perform basic frame analysis on clustered localizations. Reject
    clusters whose mean frame is outside of the [20, 80] % (max frame)
    range or any 1/20th of measurement's time contains more than 80 % of
    localizations.

    Uses ``pandas`` for fast calculations using groupby().

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels (-1 means no cluster assigned).
    frame : np.ndarray
        Frame number for each localization.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    # group frames by cluster ids
    frame_pd = pd.Series(frame, index=labels)
    frame_grouped = frame_pd.groupby(frame_pd.index)

    # perform frame analysis
    true_cluster = frame_grouped.apply(_frame_analysis, frame.max() + 1)

    # cluster ids that did not pass frame analysis
    discard = true_cluster.index[true_cluster == 0].values
    # change labels of these clusters to -1
    labels[np.isin(labels, discard)] = -1

    return labels


def _cluster(
    X: np.ndarray,
    radius: float,
    min_locs: int,
    frame: np.ndarray | None = None,
) -> np.ndarray:
    """Cluster points given by X with a given clustering radius and
    minimum number of localizations within that radius using KDTree.

    The general workflow is as follows:
    1. Build a KDTree from the points in X.
    2. For each point, find its neighbors within the given radius.
    3. Identify local maxima, i.e., points with the most neighbors
       within their neighborhood.
    4. Assign cluster labels to all points. If two local maxima are
       within the radius from each other, combine such clusters.

    Based on the algorithm published by Schichthaerle et al.
    Nature Comm, 2021 (10.1038/s41467-021-22606-1) and first implemented
    in Reinhardt, Masullo, Baudrexel, Steen, et al. Nature, 2023
    (10.1038/s41586-023-05925-9).

    Parameters
    ----------
    X : np.ndarray
        Array of points of shape (n_points, n_dim) to be clustered.
    radius : float
        Clustering radius.
    min_locs : int
        Minimum number of localizations in a cluster.
    frame : np.ndarray, optional
        Frame number of each localization. If None, no frame analysis
        is performed.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    # build kdtree (use cKDTree in case user did not update scipy)
    tree = KDTree(X)

    # find neighbors for each point within radius
    neighbors = tree.query_ball_tree(tree, radius)

    # find local maxima, i.e., points with the most neighbors within
    # their neighborhood
    lm = np.zeros(X.shape[0], dtype=np.int8)
    for i in range(len(lm)):
        idx = neighbors[i]  # indeces of points that are neighbors of i
        n = len(idx)  # number of neighbors of i
        if n > min_locs:  # note that i is included in its neighbors
            # if i has the most neighbors in its neighborhood
            if n == max([len(neighbors[_]) for _ in idx]):
                lm[i] = 1

    # assign cluster labels to all points (-1 means no cluster)
    # if two local maxima are within radius from each other, combine
    # such clusters
    labels = -1 * np.ones(X.shape[0], dtype=np.int32)  # cluster labels
    lm_idx = np.where(lm == 1)[0]  # indeces of local maxima

    for count, i in enumerate(lm_idx):  # for each local maximum
        label = labels[i]
        if label == -1:  # if lm not assigned yet
            labels[neighbors[i]] = count
        else:
            # indeces of locs that were not assigned to any cluster
            idx = [
                neighbors[i][_]
                for _ in np.where(labels[neighbors[i]] == -1)[0]
            ]
            if len(idx):  # if such a loc exists, assign it to a cluster
                labels[idx] = label

    # check for number of locs per cluster to be above min_locs
    values, counts = np.unique(labels, return_counts=True)
    # labels to discard if has fewer locs than min_locs
    to_discard = values[counts < min_locs]
    # substitute this with -1
    labels[np.isin(labels, to_discard)] = -1

    if frame is not None:
        labels = frame_analysis(labels, frame)

    return labels


def cluster_2D(
    x: np.ndarray,
    y: np.ndarray,
    frame: np.ndarray,
    radius: float,
    min_locs: int,
    fa: bool,
) -> np.ndarray:
    """Prepare 2D input to be used by ``_cluster``.

    Parameters
    ----------
    x, y : np.ndarray
        x and y coordinates to be clustered.
    frame : np.ndarray
        Frame number for each localization.
    radius : float
        Clustering radius.
    min_locs : int
        Minimum number of localizations in a cluster.
    fa : bool
        True, if basic frame analysis is to be performed.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    X = np.stack((x, y)).T

    if not fa:
        frame = None

    labels = _cluster(X, radius, min_locs, frame)

    return labels


def cluster_3D(x, y, z, frame, radius_xy, radius_z, min_locs, fa):
    """Prepare 3D input to be used by ``_cluster``.

    Scales z coordinates by radius_xy / radius_z

    Parameters
    ----------
    x, y, z : np.ndarray
        x, y, and z coordinates to be clustered.
    frame : np.ndarray
        Frame number for each localization.
    radius_xy : float
        Clustering radius in x and y directions.
    radius_z : float
        Clustering radius in z direction.
    min_locs : int
        Minimum number of localizations in a cluster.
    fa : bool
        True, if basic frame analysis is to be performed.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    radius = radius_xy
    X = np.stack((x, y, z * radius_xy / radius_z)).T

    if not fa:
        frame = None

    labels = _cluster(X, radius, min_locs, frame)

    return labels


def cluster(
    locs: pd.DataFrame,
    radius_xy: float,
    min_locs: int,
    frame_analysis: bool,
    radius_z: float | None = None,
    pixelsize: float | None = None,
) -> pd.DataFrame:
    """Cluster localizations from single molecules (SMLM clusterer).

    The general workflow is as follows:
    1. Build a KDTree from the points in X.
    2. For each point, find its neighbors within the given radius.
    3. Identify local maxima, i.e., points with the most neighbors
       within their neighborhood.
    4. Assign cluster labels to all points. If two local maxima are
       within the radius from each other, combine such clusters.

    Based on the algorithm published by Schichthaerle et al.
    Nature Comm, 2021 (10.1038/s41467-021-22606-1) and first implemented
    in Reinhardt, Masullo, Baudrexel, Steen, et al. Nature, 2023
    (10.1038/s41586-023-05925-9).

    The recommended parameters are ``radius`` of 2*NeNA and ``min_locs``
    of 10. Keep in mind that the parameters may vary between
    applications, so we encourage you to experiment with them when
    needed. Especially ``min_locs`` may vary between datasets.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered.
    radius_xy : float
        Clustering radius in xy plane (camera pixels).
    min_locs : int
        Minimum number of localizations in a cluster.
    frame_analysis : bool
        If True, performs basic frame analysis.
    radius_z : float, optional
        Clustering radius in z plane (camera pixels). Only used for
        3D clustering.
    pixelsize : int, optional
        Camera pixel size in nm. Only needed for 3D clustering.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    """
    if hasattr(locs, "z"):  # 3D
        if pixelsize is None or radius_z is None:
            raise ValueError(
                "Camera pixel size and clustering radius in z must be"
                " specified for 3D clustering."
            )
        labels = cluster_3D(
            locs["x"].values,
            locs["y"].values,
            locs["z"].values / pixelsize,  # convert from nm to px
            locs["frame"].values,
            radius_xy,
            radius_z,
            min_locs,
            frame_analysis,
        )
    else:
        labels = cluster_2D(
            locs["x"].values,
            locs["y"].values,
            locs["frame"].values,
            radius_xy,
            min_locs,
            frame_analysis,
        )
    locs = extract_valid_labels(locs, labels)
    return locs


def _dbscan(
    X: np.ndarray,
    radius: float,
    min_density: int,
    min_locs: int = 0,
) -> np.ndarray:
    """Find DBSCAN cluster labels, given data points and parameters.

    See Ester, et al. Inkdd, 1996. (Vol. 96, No. 34, pp. 226-231).

    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, D), with N being the number of data points
        and D the number of dimensions.
    radius : float
        DBSCAN search radius, often referred to as "epsilon".
    min_density : int
        Number of points within radius to consider a given point a core
        sample.
    min_locs : int, optional
        Minimum number of localizations in a cluster. Clusters with
        fewer localizations will be removed. Default is 0.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each point. Shape: (N,). -1 means no cluster
        assigned.
    """
    db = DBSCAN(eps=radius, min_samples=min_density).fit(X)
    labels = db.labels_.astype(np.int32)
    unique_clusters, counts = np.unique(labels, return_counts=True)
    to_discard = unique_clusters[counts < min_locs]
    labels[np.isin(labels, to_discard)] = -1
    return labels


def dbscan(
    locs: pd.DataFrame,
    radius: float,
    min_samples: int,
    min_locs: int = 10,
    pixelsize: int | None = None,
) -> pd.DataFrame:
    """Perform DBSCAN on localizations.

    See Ester, et al. Inkdd, 1996. (Vol. 96, No. 34, pp. 226-231).

    Parameters
    ---------
    locs : pd.DataFrame
        Localizations to be clustered.
    radius : float
        DBSCAN search radius, often referred to as "epsilon". Same units
        as locs.
    min_samples : int
        Number of localizations within radius to consider a given point
        a core sample.
    min_locs : int, optional
        Minimum number of localizations in a cluster. Clusters with
        fewer localizations will be removed. Default is 0.
    pixelsize : int, optional
        Camera pixel size in nm. Only needed for 3D.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    """
    if hasattr(locs, "z"):
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size must be specified as an integer for 3D"
                " clustering."
            )
        X = np.vstack(
            (locs["x"].values, locs["y"].values, locs["z"].values / pixelsize)
        ).T
    else:
        X = np.vstack((locs["x"].values, locs["y"].values)).T
    labels = _dbscan(X, radius, min_samples, min_locs)
    locs = extract_valid_labels(locs, labels)
    return locs


def _hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_eps: float = 0,
) -> np.ndarray:
    """Find HDBSCAN cluster labels, given data points and parameters.

    See Campello, et al. PAKDD, 2013 (DOI: 10.1007/978-3-642-37456-2_14).

    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, D), with N being the number of data points
        and D the number of dimensions.
    min_cluster_size : int
        Minimum number of points in cluster.
    min_samples : int
        Number of points within radius to consider a given point a core
        sample.
    cluster_eps : float, optional
        Distance threshold. Clusters below this value will be merged.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each point. Shape: (N,). -1 means no cluster
        assigned.
    """
    hdb = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_eps,
    ).fit(X)
    return hdb.labels_.astype(np.int32)


def hdbscan(
    locs: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int,
    pixelsize: int | None = None,
    cluster_eps: float = 0.0,
) -> pd.DataFrame:
    """Perform HDBSCAN on localizations.

    See Campello, et al. PAKDD, 2013 (DOI: 10.1007/978-3-642-37456-2_14).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered.
    min_cluster_size : int
        Minimum number of localizations in cluster.
    min_samples : int
        Number of localizations within radius to consider a given point
        a core sample.
    pixelsize : int, optional
        Camera pixel size in nm. Only needed for 3D.
    cluster_eps : float, optional
        Distance threshold. Clusters below this value will be merged.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    """
    if hasattr(locs, "z"):
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size (nm) must be specified as an integer for 3D"
                " clustering."
            )
        X = np.vstack(
            (locs["x"].values, locs["y"].values, locs["z"].values / pixelsize)
        ).T
    else:
        X = np.vstack((locs["x"].values, locs["y"].values)).T
    labels = _hdbscan(
        X, min_cluster_size, min_samples, cluster_eps=cluster_eps
    )
    locs = extract_valid_labels(locs, labels)
    return locs


def extract_valid_labels(
    locs: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Extract localizations based on clustering results. Localizations
    that were not clustered are excluded.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be filtered.
    labels : np.ndarray
        Array of cluster labels for each localization. -1 means no
        cluster assignment.

    Returns
    -------
    locs : pd.DataFrame
        Localization list with "group" column appended, providing
        cluster label.
    """
    # add cluster id to locs, as "group"
    locs["group"] = labels

    # -1 means no cluster assigned to a loc
    locs = locs[locs["group"] != -1]
    return locs


# def error_sums_wtd(x: float, w: float) -> float:
#     """Find "localization precision" for cluster centers, i.e., weighted
#     standard error of the mean of the localizations in the given
#     cluster.

#     Parameters
#     ----------
#     x : float
#         x or y coordinate of the cluster center.
#     w : float
#         weight (inverse localization precision squared).

#     Returns
#     -------
#     lp : float
#         Weighted standard error of the mean of the cluster center.
#     """
#     lp = (w * (x - (w * x).sum() / w.sum())**2).sum() / w.sum()
#     return lp


def find_cluster_centers(
    locs: pd.DataFrame,
    pixelsize: float | None = None,
) -> pd.DataFrame:
    """Calculate cluster centers.

    Uses ``pandas.groupby`` to quickly run across all cluster ids.

    Parameters
    ----------
    locs : pd.DataFrame
        Clustered localizations (contain group info)
    pixelsize : int, optional
        Camera pixel size (used for finding volume and 3D convex hull).
        Only required for 3D localizations.

    Returns
    -------
    centers : pd.DataFrame
        Cluster centers saved in the format of localizations.
    """
    # group locs by their cluster id (group)
    grouplocs = locs.groupby(locs["group"])

    # get cluster centers
    res = grouplocs.apply(cluster_center, pixelsize, include_groups=False)
    centers_ = res.values

    # convert to DataFrame and save
    frame = np.array([_[0] for _ in centers_])
    std_frame = np.array([_[1] for _ in centers_])
    x = np.array([_[2] for _ in centers_])
    y = np.array([_[3] for _ in centers_])
    std_x = np.array([_[4] for _ in centers_])
    std_y = np.array([_[5] for _ in centers_])
    photons = np.array([_[6] for _ in centers_])
    sx = np.array([_[7] for _ in centers_])
    sy = np.array([_[8] for _ in centers_])
    bg = np.array([_[9] for _ in centers_])
    lpx = np.array([_[10] for _ in centers_])
    lpy = np.array([_[11] for _ in centers_])
    ellipticity = np.array([_[12] for _ in centers_])
    net_gradient = np.array([_[13] for _ in centers_])
    n = np.array([_[14] for _ in centers_])
    n_events = np.array([_[15] for _ in centers_])  # number of locs in cluster

    if hasattr(locs, "z"):
        z = np.array([_[16] for _ in centers_])
        std_z = np.array([_[17] for _ in centers_])
        volume = np.array([_[18] for _ in centers_])
        convexhull = np.array([_[19] for _ in centers_])
        centers = pd.DataFrame(
            {
                "frame": frame.astype(np.float32),
                "std_frame": std_frame.astype(np.float32),
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "std_x": std_x.astype(np.float32),
                "std_y": std_y.astype(np.float32),
                "z": z.astype(np.float32),
                "photons": photons.astype(np.float32),
                "sx": sx.astype(np.float32),
                "sy": sy.astype(np.float32),
                "bg": bg.astype(np.float32),
                "lpx": lpx.astype(np.float32),
                "lpy": lpy.astype(np.float32),
                "std_z": std_z.astype(np.float32),
                "ellipticity": ellipticity.astype(np.float32),
                "net_gradient": net_gradient.astype(np.float32),
                "n": n.astype(np.uint32),
                "n_events": n_events.astype(np.int32),
                "volume": volume.astype(np.float32),
                "convexhull": convexhull.astype(np.float32),
                "group": res.index.values.astype(np.int32),  # group id
            }
        )
    else:
        area = np.array([_[16] for _ in centers_])
        convexhull = np.array([_[17] for _ in centers_])
        centers = pd.DataFrame(
            {
                "frame": frame.astype(np.float32),
                "std_frame": std_frame.astype(np.float32),
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "std_x": std_x.astype(np.float32),
                "std_y": std_y.astype(np.float32),
                "photons": photons.astype(np.float32),
                "sx": sx.astype(np.float32),
                "sy": sy.astype(np.float32),
                "bg": bg.astype(np.float32),
                "lpx": lpx.astype(np.float32),
                "lpy": lpy.astype(np.float32),
                "ellipticity": ellipticity.astype(np.float32),
                "net_gradient": net_gradient.astype(np.float32),
                "n": n.astype(np.uint32),
                "n_events": n_events.astype(np.int32),
                "area": area.astype(np.float32),
                "convexhull": convexhull.astype(np.float32),
                "group": res.index.values.astype(np.int32),  # group id
            }
        )
    if hasattr(locs, "group_input"):
        group_input = np.array([_[-1] for _ in centers_])
        centers["group_input"] = group_input.astype(np.int32)

    return centers


def cluster_center(
    grouplocs: pd.SeriesGroupBy,
    pixelsize: float | None = None,
    separate_lp: bool = False,
) -> pd.Series:
    """Find cluster centers and their attributes, such as mean number
    of photons per localization, etc.

    Assumes locs to be a ``pandas.SeriesGroupBy`` object, grouped by
    cluster ids.

    Parameters
    ----------
    grouplocs : pandas.SeriesGroupBy
        Localizations grouped by cluster ids.
    pixelsize : int, optional
        Camera pixel size (used for finding volume and 3D convex hull).
        Only required for 3D localizations.
    separate_lp : bool, optional
        If True, localization precision in x and y will be calculated
        separately. Otherwise, the mean of the two is taken.

    Returns
    -------
    results : pd.Series
        Cluster center attributes. For each group, a list of values is
        returned: x, y, (z, optional), etc.
    """
    # mean and std frame
    frame = grouplocs.frame.mean()
    std_frame = grouplocs.frame.std()
    # average x and y, weighted by lpx, lpy
    # x = np.average(grouplocs.x, weights=1/(grouplocs.lpx)**2)
    # y = np.average(grouplocs.y, weights=1/(grouplocs.lpy)**2)
    x = np.mean(grouplocs.x)
    y = np.mean(grouplocs.y)
    std_x = grouplocs.x.std()
    std_y = grouplocs.y.std()
    # mean values
    photons = grouplocs.photons.mean()
    sx = grouplocs.sx.mean()
    sy = grouplocs.sy.mean()
    bg = grouplocs.bg.mean()
    # weighted mean loc precision
    # lpx = np.sqrt(
    #     error_sums_wtd(grouplocs.x, grouplocs.lpx)
    #     / (len(grouplocs) - 1)
    # )
    # lpy = np.sqrt(
    #     error_sums_wtd(grouplocs.y, grouplocs.lpy)
    #     / (len(grouplocs) - 1)
    # )
    lpx = np.std(grouplocs.x) / len(grouplocs) ** 0.5
    lpy = np.std(grouplocs.y) / len(grouplocs) ** 0.5
    if not separate_lp:
        lpx = (lpx + lpy) / 2
        lpy = lpx
    # other attributes
    ellipticity = sx / sy
    net_gradient = grouplocs.net_gradient.mean()
    # n_locs in cluster
    n = len(grouplocs)
    # number of binding events
    split_idx = np.where(np.diff(grouplocs.frame) > 3)[0] + 1  # split locs by
    # consecutive frames
    x_events = np.split(grouplocs.x.values, split_idx)
    n_events = len(x_events)  # number of binding events
    if hasattr(grouplocs, "z"):
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size must be specified as an integer for 3D"
                " cluster centers calculation."
            )
        z = np.average(
            grouplocs.z,
            weights=1 / ((grouplocs.lpx + grouplocs.lpy) ** 2),
        )  # take lpz = 2 * mean(lpx, lpy)
        std_z = grouplocs.z.std()
        # lpz = std_z
        volume = (
            np.power((std_x + std_y + std_z / pixelsize) / 3 * 2, 3) * 4.18879
        )
        try:
            X = np.stack(
                (grouplocs.x, grouplocs.y, grouplocs.z / pixelsize),
                axis=0,
            ).T
            hull = ConvexHull(X)
            convexhull = hull.volume
        except QhullError:
            convexhull = 0
        result = [
            frame,
            std_frame,
            x,
            y,
            std_x,
            std_y,
            photons,
            sx,
            sy,
            bg,
            lpx,
            lpy,
            ellipticity,
            net_gradient,
            n,
            n_events,
            z,
            std_z,
            # lpz,
            volume,
            convexhull,
        ]
    else:
        area = np.power(std_x + std_y, 2) * np.pi
        try:
            X = np.stack((grouplocs.x, grouplocs.y), axis=0).T
            hull = ConvexHull(X)
            convexhull = hull.volume
        except QhullError:
            convexhull = 0
        result = [
            frame,
            std_frame,
            x,
            y,
            std_x,
            std_y,
            photons,
            sx,
            sy,
            bg,
            lpx,
            lpy,
            ellipticity,
            net_gradient,
            n,
            n_events,
            area,
            convexhull,
        ]

    if hasattr(grouplocs, "group_input"):
        # assumes only one group input!
        result.append(np.unique(grouplocs.group_input)[0])
    return result


def _cluster_area(X: np.ndarray, lp: np.ndarray) -> float:
    """Calculate cluster area (2D) or volume (3D). Uses Otsu
    thresholding of the images of the clusters to find areas/volumes.

    Parameters
    ----------
    X : np.ndarray
        Array of points of shape (n_points, n_dim).
    lp : np.ndarray
        Localization precision in x and y of the cluster. The median
        is taken to define the pixel size for image rendering

    Returns
    -------
    area : float
        Cluster area (2D) or volume (3D) in units of LP.
    """
    # get image
    bin_size = np.median(lp) / 2  # pixel size for rendering
    if X.shape[1] == 3:  # 3D
        bin_size_z = bin_size * 2.5  # just a rough estimate
        edges = [
            np.arange(X[:, 0].min(), X[:, 0].max() + bin_size, bin_size),
            np.arange(X[:, 1].min(), X[:, 1].max() + bin_size, bin_size),
            np.arange(X[:, 2].min(), X[:, 2].max() + bin_size_z, bin_size_z),
        ]
    else:  # 2D
        edges = [
            np.arange(X[:, 0].min(), X[:, 0].max() + bin_size, bin_size),
            np.arange(X[:, 1].min(), X[:, 1].max() + bin_size, bin_size),
        ]
    image = np.histogramdd(X, bins=edges)[0]
    image = gaussian_filter(image, sigma=2)  # smooth image with sigma = LP
    # threshold the image and calculate area/volume
    thresh = masking.threshold_otsu(image.reshape(-1))
    if X.shape[1] == 3:  # 3D
        area = np.sum(image >= thresh) / (16 / 5)  # volume in LP^3, bins are 0.5 LP in xy and 1.25 LP in z
    else:  # 2D
        area = np.sum(image >= thresh) / 4  # area in LP^2, bins are 0.5 LP
    return area


def cluster_areas(
    locs: pd.DataFrame,
    info: list[dict],
    progress: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """Calculate cluster areas (2D) or volumes (3D).

    Uses Otsu thresholding of the images of the clusters to find areas/
    volumes.

    Parameters
    ----------
    locs : pd.DataFrame
        Clustered localizations (contain group info).
    info : list of dict
        Localization metadata, see `picasso.io.load_locs`.
    progress : picasso.lib.ProgressDialog, optional
        Progress dialog. If None, progress is displayed with into the
        console. Default is None.

    Returns
    -------
    areas : pd.DataFrame
        Cluster areas/volumes for each cluster.
    """
    assert hasattr(
        locs, "group"
    ), "Localizations must contain 'group' column."

    # get pixel size from info
    pixelsize = lib.get_from_metadata(info, "Pixelsize", default=None)
    assert (
        isinstance(pixelsize, (int, float))
    ), "Pixelsize not found in info."

    groups = np.unique(locs["group"])
    area_key = "Area (LP^2)" if not hasattr(locs, "z") else "Volume (LP^3)"
    areas = {
        "group": groups.astype(np.int32), 
        area_key: np.zeros(len(groups), dtype=np.float32)
    }
    if progress is None:
        iterator = tqdm(range(len(groups)), desc="Calculating cluster areas")
    else:
        iterator = range(len(groups))

    for idx in iterator:
        group_id = groups[idx]
        grouplocs = locs[locs["group"] == group_id]
        if not len(grouplocs):
            continue
        lp = locs[["lpx", "lpy"]].values.mean(axis=1)
        if hasattr(grouplocs, "z"):
            X = grouplocs[["x", "y", "z"]].values
            X[:, 2] /= pixelsize  # convert z to pixels
        else:
            X = grouplocs[["x", "y"]].values
        areas[area_key][idx] = _cluster_area(X, lp)
        if progress is not None:
            progress(idx + 1)
    return pd.DataFrame(areas)
