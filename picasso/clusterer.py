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
    Thomas Schlichthaerle
:copyright: Copyright (c) 2022-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.spatial import ConvexHull, KDTree, QhullError
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN, HDBSCAN

from . import lib, masking

#: scikit-learn's HDBSCAN fails with a TypeError for cluster_eps > 0 on
#: some data with NumPy 2.4 and later, which no longer converts a
#: one-element array to a number (scikit-learn issue 33219, unfixed in
#: scikit-learn 1.9.1). The one-click installers ship NumPy 2.3.
_HDBSCAN_EPSILON_BUG = np.lib.NumpyVersion(np.__version__) >= "2.4.0"
_HDBSCAN_EPSILON_ISSUE = (
    "https://github.com/scikit-learn/scikit-learn/issues/33219"
)

__all__ = [
    "cluster",
    "dbscan",
    "hdbscan",
    "find_cluster_centers",
    "cluster_areas",
    "test_subclustering",
]


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
    mean_frame = frame.mean()

    # get maximum number of locs in a 1/20th of acquisition time
    n_locs = len(frame)
    locs_binned = np.histogram(frame, bins=np.linspace(0, n_frames, 21))[0]
    max_locs_bin = locs_binned.max()

    # test if frame analysis passed
    if (
        (mean_frame < 0.2 * n_frames)
        or (mean_frame > 0.8 * n_frames)
        or (max_locs_bin > 0.8 * n_locs)
    ):
        passed = 0

    return passed


def frame_analysis(
    labels: lib.IntArray1D, frame: lib.IntArray1D
) -> lib.IntArray1D:
    """Perform basic frame analysis on clustered localizations. Reject
    clusters whose mean frame is outside of the [20, 80] % (max frame)
    range or any 1/20th of measurement's time contains more than 80 % of
    localizations.

    Uses ``pandas`` for fast calculations using groupby().

    Parameters
    ----------
    labels : lib.IntArray1D
        Cluster labels (-1 means no cluster assigned).
    frame : lib.IntArray1D
        Frame number for each localization.

    Returns
    -------
    labels : lib.IntArray1D
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    # group frames by cluster ids
    frame_pd = pd.Series(frame, index=labels)
    frame_grouped = frame_pd.groupby(frame_pd.index)

    # perform frame analysis
    true_cluster = frame_grouped.apply(_frame_analysis, frame.max() + 1)

    # cluster ids that did not pass frame analysis
    discard = true_cluster.index[true_cluster == 0]
    # change labels of these clusters to -1
    labels[np.isin(labels, discard)] = -1

    return labels


# number of points whose neighbors are queried from the KDTree at once;
# bounds peak memory and provides progress granularity in ``_cluster``
_NEIGHBOR_BATCH_SIZE = 100_000


def _build_neighbor_graph(
    tree: KDTree,
    X: lib.FloatArray2D,
    radius: float,
    progress: lib.Progress,
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Build a compact (CSR-like) neighbor graph for all points in X.

    For each point, finds the indices of all points within ``radius``
    (including the point itself). Rather than keeping scipy's
    list-of-lists (Python ints, very memory heavy for millions of
    points), neighbors are stored as two flat NumPy arrays:

    * ``indptr`` of shape (n_points + 1,); the neighbors of point ``i``
      are ``indices[indptr[i]:indptr[i + 1]]``.
    * ``indices``, the concatenated neighbor indices (int32).

    The tree is queried in batches of ``_NEIGHBOR_BATCH_SIZE`` points,
    so only one batch of Python lists is alive at any time and progress
    can be reported.

    Parameters
    ----------
    tree : scipy.spatial.KDTree
        KDTree built from ``X``.
    X : lib.FloatArray2D
        Array of points of shape (n_points, n_dim).
    radius : float
        Clustering radius.
    progress : lib.Progress
        Normalized progress tracker, set to the cumulative number of
        points processed after each batch.

    Returns
    -------
    indptr : lib.IntArray1D
        Neighbor offsets, shape (n_points + 1,), dtype int64.
    indices : lib.IntArray1D
        Concatenated neighbor indices, dtype int32.
    """
    n_points = X.shape[0]
    counts = np.empty(n_points, dtype=np.int32)
    index_chunks = []

    n_batches = int(np.ceil(n_points / _NEIGHBOR_BATCH_SIZE))

    for b in range(n_batches):
        start = b * _NEIGHBOR_BATCH_SIZE
        end = min(start + _NEIGHBOR_BATCH_SIZE, n_points)
        # neighbors for this batch as a list of lists; discarded once
        # flattened into compact arrays below
        nb = tree.query_ball_point(X[start:end], radius, workers=-1)
        batch_counts = np.fromiter(
            (len(n) for n in nb), dtype=np.int32, count=len(nb)
        )
        counts[start:end] = batch_counts
        total = int(batch_counts.sum())
        if total:
            index_chunks.append(
                np.fromiter(
                    itertools.chain.from_iterable(nb),
                    dtype=np.int32,
                    count=total,
                )
            )
        progress.set_value(end)

    indptr = np.zeros(n_points + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    if index_chunks:
        indices = np.concatenate(index_chunks)
    else:
        indices = np.empty(0, dtype=np.int32)
    return indptr, indices


def _cluster(
    X: lib.FloatArray2D,
    radius: float,
    min_locs: int,
    frame: pd.Series | None,
    progress: lib.Progress,
) -> lib.IntArray1D:
    """Cluster points given by X with a given clustering radius and
    minimum number of localizations within that radius using KDTree.

    The general workflow is as follows:

    1. Build a KDTree from the points in X.
    2. For each point, find its neighbors within the given radius.
    3. Identify local maxima, i.e., points with the most neighbors
       within their neighborhood.
    4. Assign cluster labels to all points. If two local maxima are
       within the radius from each other, combine such clusters.

    The neighbor graph is built in batches and stored as compact NumPy
    arrays (see ``_build_neighbor_graph``) so peak memory stays bounded
    even for tens of millions of localizations.

    Based on the algorithm published by Schichthaerle et al.
    Nature Comm, 2021 (10.1038/s41467-021-22606-1) and first implemented
    in Reinhardt, Masullo, Baudrexel, Steen, et al. Nature, 2023
    (10.1038/s41586-023-05925-9).

    Parameters
    ----------
    X : lib.FloatArray2D
        Array of points of shape (n_points, n_dim) to be clustered.
    radius : float
        Clustering radius.
    min_locs : int
        Minimum number of localizations in a cluster.
    frame : pd.Series or None
        Frame number of each localization. If None, no frame analysis
        is performed.
    progress : lib.Progress
        Normalized progress tracker (see ``lib.normalize_progress``);
        reports "Clustering" over the localizations while building the
        neighbor graph (the main, O(n) step).

    Returns
    -------
    labels : lib.IntArray1D
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    n_points = X.shape[0]

    # build kdtree and a compact, batched neighbor graph (bounded memory)
    tree = KDTree(X)
    lib.start_phase(progress, "Clustering", n_points, unit="loc")
    progress.set_value(0)
    indptr, indices = _build_neighbor_graph(tree, X, radius, progress)
    progress.set_value(n_points)

    # number of neighbors of each point (point is its own neighbor)
    counts = np.diff(indptr).astype(np.int32)

    # find local maxima, i.e., points with the most neighbors within
    # their neighborhood. ``reduceat`` computes, per point, the maximum
    # neighbor count over that point's neighborhood in one vectorized
    # pass (no Python loop over all points).
    if n_points:
        neighbor_max = np.maximum.reduceat(counts[indices], indptr[:-1])
    else:
        neighbor_max = np.empty(0, dtype=np.int32)
    # note that a point is included in its own neighbors
    lm = (counts > min_locs) & (counts == neighbor_max)

    # assign cluster labels to all points (-1 means no cluster)
    # if two local maxima are within radius from each other, combine
    # such clusters
    labels = -1 * np.ones(n_points, dtype=np.int32)  # cluster labels
    lm_idx = np.where(lm)[0]  # indeces of local maxima

    for count, i in enumerate(lm_idx):  # for each local maximum
        neighbors_i = indices[indptr[i] : indptr[i + 1]]
        label = labels[i]
        if label == -1:  # if lm not assigned yet
            labels[neighbors_i] = count
        else:
            # locs in the neighborhood not yet assigned to any cluster
            idx = neighbors_i[labels[neighbors_i] == -1]
            if idx.size:  # if such a loc exists, assign it to a cluster
                labels[idx] = label

    # check for number of locs per cluster to be above min_locs
    values, counts = np.unique(labels, return_counts=True)
    # labels to discard if has fewer locs than min_locs
    to_discard = values[counts < min_locs]
    # substitute this with -1
    labels[np.isin(labels, to_discard)] = -1

    if frame is not None:
        # must convert frames to an array, do not change!
        labels = frame_analysis(labels, frame.to_numpy())

    return labels


def cluster_2D(
    locs: pd.DataFrame,
    radius: float,
    min_locs: int,
    fa: bool,
    progress: lib.Progress | Literal["console"] | None = None,
) -> lib.IntArray1D:
    """Prepare 2D input to be used by ``_cluster``.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered.
    radius : float
        Clustering radius.
    min_locs : int
        Minimum number of localizations in a cluster.
    fa : bool
        True, if basic frame analysis is to be performed.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Clustering" over the
        localizations. Default is None.

    Returns
    -------
    labels : lib.IntArray1D
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    X = locs[["x", "y"]].to_numpy()

    if not fa:
        frame = None
    else:
        frame = locs["frame"]

    with lib.tracked_progress(progress, "Clustering", unit="loc") as progress:
        labels = _cluster(X, radius, min_locs, frame, progress)

    return labels


def cluster_3D(
    locs: pd.DataFrame,
    radius_xy: float,
    radius_z: float,
    min_locs: int,
    fa: bool,
    progress: lib.Progress | Literal["console"] | None = None,
) -> lib.IntArray1D:
    """Prepare 3D input to be used by ``_cluster``.

    Scales z coordinates by ``radius_xy / radius_z`` so that a Euclidean
    neighborhood search with radius ``radius_xy`` in the scaled space
    corresponds to an ellipsoidal neighborhood with semi-axes
    ``(radius_xy, radius_xy, radius_z)`` in the original space.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered.
    radius_xy : float
        Clustering radius in x and y directions (nm).
    radius_z : float
        Clustering radius in z direction (nm).
    min_locs : int
        Minimum number of localizations in a cluster.
    fa : bool
        True, if basic frame analysis is to be performed.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Clustering" over the
        localizations. Default is None.

    Returns
    -------
    labels : lib.IntArray1D
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    radius = radius_xy
    # a copy, since z is scaled in place (pandas 3 returns read-only
    # views of the columns)
    X = locs[["x", "y", "z"]].to_numpy(copy=True)
    X[:, 2] *= radius_xy / radius_z

    if not fa:
        frame = None
    else:
        frame = locs["frame"]

    with lib.tracked_progress(progress, "Clustering", unit="loc") as progress:
        labels = _cluster(X, radius, min_locs, frame, progress)

    return labels


def cluster(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    radius_xy: float,
    min_locs: int,
    frame_analysis: bool,
    radius_z: float | None = None,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
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
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    radius_xy : float
        Clustering radius in xy plane (nm).
    min_locs : int
        Minimum number of localizations in a cluster.
    frame_analysis : bool
        If True, performs basic frame analysis.
    radius_z : float, optional
        Clustering radius in z (nm). Required for 3D clustering.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Clustering" over the
        localizations while building the neighbor graph (the main, O(n)
        step). Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    info : list of dicts
        Metadata of the clustered localizations: ``info`` with an entry
        recording the clustering appended, see ``lib.append_metadata``.
    """
    locs = locs.copy()
    n_raw = len(locs)
    if "z" in locs.columns and radius_z is None:  # 3D
        raise ValueError(
            "The clustering radius in z must be specified for 3D"
            " clustering."
        )
    with lib.tracked_progress(progress, "Clustering", unit="loc") as progress:
        if "z" in locs.columns:  # 3D
            labels = cluster_3D(
                locs,
                radius_xy,
                radius_z,
                min_locs,
                frame_analysis,
                progress,
            )
        else:
            labels = cluster_2D(
                locs,
                radius_xy,
                min_locs,
                frame_analysis,
                progress,
            )
    locs = extract_valid_labels(locs, labels)
    n_clusters = len(locs)
    entry = {
        **lib.step_info("SMLM clusterer"),
        "Number of clusters": len(np.unique(locs["group"])),
        "Min. cluster size": min_locs,
        "Performed basic frame analysis": frame_analysis,
        "Fraction of rejected locs (%)": 100 * (n_raw - n_clusters) / n_raw,
    }
    if "z" in locs.columns:
        entry["Clustering radius xy (nm)"] = radius_xy
        entry["Clustering radius z (nm)"] = radius_z
    else:
        entry["Clustering radius (nm)"] = radius_xy
    return locs, lib.append_metadata(info, entry)


def _dbscan(
    X: lib.FloatArray2D,
    radius: float,
    min_density: int,
    min_locs: int = 0,
) -> lib.IntArray1D:
    """Find DBSCAN cluster labels, given data points and parameters.

    See Ester, et al. Inkdd, 1996. (Vol. 96, No. 34, pp. 226-231).

    Parameters
    ----------
    X : lib.FloatArray2D
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
    labels : lib.IntArray1D
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
    info: list[dict] | None = None,
    *,
    radius: float,
    min_samples: int,
    min_locs: int = 10,
    radius_z: float | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Perform DBSCAN on localizations.

    See Ester, et al. Inkdd, 1996. (Vol. 96, No. 34, pp. 226-231).

    For 3D data with ``radius_z`` set, anisotropic clustering is used:
    z coordinates are scaled by ``radius / radius_z`` so that the
    isotropic DBSCAN search with epsilon ``radius`` corresponds to an
    ellipsoidal neighborhood with semi-axes
    ``(radius, radius, radius_z)`` in the original space (same approach
    as ``cluster_3D``).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered.
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    radius : float
        DBSCAN search radius in the xy plane (nm), usually referred to
        as "epsilon".
    min_samples : int
        Number of localizations within radius to consider a given point
        a core sample.
    min_locs : int, optional
        Minimum number of localizations in a cluster. Clusters with
        fewer localizations will be removed. Default is 10.
    radius_z : float, optional
        DBSCAN search radius in z (nm). If None (default),
        the clustering is isotropic and uses ``radius`` in all
        dimensions. Only used for 3D.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    info : list of dicts
        Metadata of the clustered localizations: ``info`` with an entry
        recording the clustering appended, see ``lib.append_metadata``.
    """
    locs = locs.copy()
    n_raw = len(locs)
    if "z" in locs.columns:
        # a copy, since z is scaled in place (pandas 3 returns read-only
        # views of the columns)
        X = locs[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)
        if radius_z is not None:
            X[:, 2] *= radius / radius_z
    else:
        X = locs[["x", "y"]].to_numpy()
    labels = _dbscan(X, radius, min_samples, min_locs)
    locs = extract_valid_labels(locs, labels)
    n_clusters = len(locs)
    entry = {
        **lib.step_info("DBSCAN"),
        "Number of clusters": len(np.unique(locs["group"])),
        "Radius (nm)": radius,
        "Minimum local density": min_samples,
        "Min. localizations per cluster": min_locs,
        "Fraction of rejected locs (%)": 100 * (n_raw - n_clusters) / n_raw,
    }
    if "z" in locs.columns and radius_z is not None:
        entry["Radius z (nm)"] = radius_z
    return locs, lib.append_metadata(info, entry)


def _hdbscan(
    X: lib.FloatArray2D,
    min_cluster_size: int,
    min_samples: int,
    cluster_eps: float = 0,
) -> lib.IntArray1D:
    """Find HDBSCAN cluster labels, given data points and parameters.

    See Campello, et al. PAKDD, 2013 (DOI: 10.1007/978-3-642-37456-2_14).

    Parameters
    ----------
    X : lib.FloatArray2D
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
    labels : lib.IntArray1D
        Cluster labels for each point. Shape: (N,). -1 means no cluster
        assigned.

    Raises
    ------
    RuntimeError
        If ``cluster_eps`` > 0 runs into scikit-learn's HDBSCAN bug with
        NumPy 2.4 and later (scikit-learn issue 33219).
    """
    try:
        hdb = HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_eps,
            copy=False,
        ).fit(X)
    except TypeError as error:
        if cluster_eps > 0 and _HDBSCAN_EPSILON_BUG:
            raise RuntimeError(
                "HDBSCAN with a cluster selection epsilon above 0 fails "
                f"with NumPy {np.__version__} because of a bug in "
                f"scikit-learn ({_HDBSCAN_EPSILON_ISSUE}). Set the epsilon "
                "to 0, or use NumPy older than 2.4, e.g., "
                '"pip install numpy<2.4".'
            ) from error
        raise
    return hdb.labels_.astype(np.int32)


def hdbscan(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    min_cluster_size: int,
    min_samples: int,
    cluster_eps: float = 0.0,
) -> tuple[pd.DataFrame, list[dict]]:
    """Perform HDBSCAN on localizations.

    See Campello, et al. PAKDD, 2013 (DOI: 10.1007/978-3-642-37456-2_14).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered.
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    min_cluster_size : int
        Minimum number of localizations in cluster.
    min_samples : int
        Number of localizations within radius to consider a given point
        a core sample.
    cluster_eps : float, optional
        Distance threshold (nm). Clusters below this value will be
        merged. Default is 0.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    info : list of dicts
        Metadata of the clustered localizations: ``info`` with an entry
        recording the clustering appended, see ``lib.append_metadata``.

    Raises
    ------
    RuntimeError
        If ``cluster_eps`` > 0 runs into scikit-learn's HDBSCAN bug with
        NumPy 2.4 and later (scikit-learn issue 33219).
    """
    locs = locs.copy()
    n_raw = len(locs)
    if "z" in locs.columns:
        X = locs[["x", "y", "z"]].to_numpy()
    else:
        X = locs[["x", "y"]].to_numpy()
    labels = _hdbscan(
        X, min_cluster_size, min_samples, cluster_eps=cluster_eps
    )
    locs = extract_valid_labels(locs, labels)
    n_clusters = len(locs)
    entry = {
        **lib.step_info("HDBSCAN"),
        "Number of clusters": len(np.unique(locs["group"])),
        "Min. cluster": min_cluster_size,
        "Min. samples": min_samples,
        "Intercluster distance (nm)": cluster_eps,
        "Fraction of rejected locs (%)": 100 * (n_raw - n_clusters) / n_raw,
    }
    return locs, lib.append_metadata(info, entry)


def extract_valid_labels(
    locs: pd.DataFrame,
    labels: lib.IntArray1D,
) -> pd.DataFrame:
    """Extract localizations based on clustering results. Localizations
    that were not clustered are excluded.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be filtered.
    labels : lib.IntArray1D
        Array of cluster labels for each localization. -1 means no
        cluster assignment.

    Returns
    -------
    locs : pd.DataFrame
        Localization list with "group" column appended, providing
        cluster label. If ``locs`` was already grouped, the previous
        grouping is preserved in the ``group_input`` column.
    """
    # add cluster id to locs, as "group"
    locs = lib.append_group(locs, labels)

    # -1 means no cluster assigned to a loc
    locs = locs[locs["group"] != -1]
    return locs


# Column names that ``find_cluster_centers`` derives itself. Any input
# column not listed here and not part of the coordinate/frame set is
# simply averaged per cluster, so custom columns carry over.
RESERVED_CENTER_COLUMNS = frozenset(
    {
        "group",
        "group_input",
        "std_frame",
        "std_x",
        "std_y",
        "std_z",
        "lpx",
        "lpy",
        "lpz",
        "ellipticity",
        "n_locs",
        "n_events",
        "area",
        "volume",
        "convexhull",
    }
)


def _aggregate_cluster_stats(
    locs: pd.DataFrame, has_z: bool
) -> tuple[pd.core.groupby.DataFrameGroupBy, dict]:
    """One vectorized pass for per-group means, stds and sizes.

    Every numeric column present in ``locs`` is averaged, except the
    ones ``find_cluster_centers`` derives itself
    (``RESERVED_CENTER_COLUMNS``).

    Returns the underlying ``groupby`` object (for downstream use such
    as ``group_input.first()``) and a dict of plain NumPy arrays, one
    per statistic, indexed positionally by sorted group id. The names of
    the averaged non-coordinate columns are returned as ``extra_cols``.
    """
    coord_cols = ["frame", "x", "y", "z"] if has_z else ["frame", "x", "y"]
    extra_cols = [
        c
        for c in locs.columns
        if c not in coord_cols
        and c not in RESERVED_CENTER_COLUMNS
        and is_numeric_dtype(locs[c])
    ]
    mean_cols = coord_cols + extra_cols
    std_cols = ["frame", "x", "y", "z"] if has_z else ["frame", "x", "y"]

    gb = locs.groupby("group", sort=True)
    means = gb[mean_cols].mean()
    stds = gb[std_cols].std()

    stats = {f"{c}_mean": means[c].to_numpy() for c in mean_cols}
    stats.update({f"{c}_std": stds[c].to_numpy() for c in std_cols})
    stats["n_locs"] = gb.size().to_numpy()
    stats["unique_groups"] = means.index.to_numpy()
    stats["extra_cols"] = extra_cols
    return gb, stats


def _count_binding_events(
    group_arr: lib.IntArray1D, frame_arr: lib.IntArray1D
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.IntArray1D]:
    """Number of binding events per cluster.

    A new event starts whenever consecutive frames within a cluster are
    more than 3 frames apart. Vectorized across all groups with one
    stable sort + one diff pass.

    Parameters
    ----------
    group_arr : lib.IntArray1D
        Cluster (group) id of each localization.
    frame_arr : lib.IntArray1D
        Frame number of each localization.

    Returns
    -------
    n_events : lib.IntArray1D
        One value per sorted unique group.
    order : lib.IntArray1D
        Stable argsort by group id; reused for the convex hull pass.
    group_s : lib.IntArray1D
        ``group_arr`` reindexed by ``order``.
    """
    order = np.argsort(group_arr, kind="stable")
    group_s = group_arr[order]
    frame_s = frame_arr[order]
    new_event = np.empty(len(frame_s), dtype=bool)
    new_event[:1] = True  # no localizations, no events
    new_event[1:] = (group_s[1:] != group_s[:-1]) | (
        (frame_s[1:] - frame_s[:-1]) > 3
    )
    n_events = (
        pd.Series(new_event).groupby(group_s, sort=True).sum().to_numpy()
    )
    return n_events, order, group_s


def _cluster_convex_hulls(
    locs: pd.DataFrame,
    order: lib.IntArray1D,
    group_s: lib.IntArray1D,
    unique_groups: lib.IntArray1D,
    has_z: bool,
    progress: lib.Progress,
) -> lib.FloatArray1D:
    """Convex-hull area (2D, nm^2) or volume (3D, nm^3) per cluster.

    The only per-cluster Python loop in ``find_cluster_centers``; runs
    on raw NumPy slices of a group-sorted coordinate array. For datasets
    with many clusters this is the run-time bottleneck, so the
    normalized tracker ``progress`` reports "Calculating cluster
    centers" over the clusters.
    """
    coord_cols = ["x", "y", "z"] if has_z else ["x", "y"]
    coords_sorted = (
        locs[coord_cols].to_numpy()[order].astype(np.float64, copy=True)
    )
    group_offsets = np.searchsorted(group_s, unique_groups, side="left")
    group_offsets = np.append(group_offsets, len(group_s))
    n_clusters = len(unique_groups)
    convexhull = np.zeros(n_clusters, dtype=np.float64)
    lib.start_phase(
        progress, "Calculating cluster centers", n_clusters, unit="cluster"
    )
    progress.set_value(0)
    for i in range(n_clusters):
        X = coords_sorted[group_offsets[i] : group_offsets[i + 1]]
        try:
            convexhull[i] = ConvexHull(X).volume
        except QhullError:
            convexhull[i] = 0.0
        progress.set_value(i + 1)
    progress.set_value(n_clusters)
    return convexhull


def _weighted_z_means(
    locs: pd.DataFrame, group_arr: lib.IntArray1D
) -> lib.FloatArray1D:
    """Per-cluster z mean weighted by 1/(lpx + lpy)^2 (per-row weights).

    Falls back to the unweighted mean if ``lpx``/``lpy`` are missing.
    """
    if not {"lpx", "lpy"}.issubset(locs.columns):
        return (
            pd.Series(locs["z"].to_numpy())
            .groupby(group_arr, sort=True)
            .mean()
            .to_numpy()
        )
    w = 1.0 / (locs["lpx"].to_numpy() + locs["lpy"].to_numpy()) ** 2
    wz = (
        pd.Series(locs["z"].to_numpy() * w).groupby(group_arr, sort=True).sum()
    )
    ws = pd.Series(w).groupby(group_arr, sort=True).sum()
    return (wz / ws).to_numpy()


def find_cluster_centers(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Calculate cluster centers.

    Aggregations are computed in vectorized pandas/NumPy passes; the
    only per-cluster Python loop is the convex hull, which operates on
    raw NumPy slices.

    Parameters
    ----------
    locs : pd.DataFrame
        Clustered localizations (contain group info). Must contain
        "frame", "x", "y" and "group" (and "z" for 3D). Any other
        numeric column present is averaged per cluster. Other numeric
        columns are carried over as means.
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating cluster centers"
        over the clusters (the per-cluster convex-hull pass, the
        run-time bottleneck for datasets with many clusters). Default is
        None.

    Returns
    -------
    centers : pd.DataFrame
        Cluster centers saved in the format of localizations: positions,
        their standard deviations (``std_x``, ``std_y``, ``std_z``) and
        precisions (``lpx``, ``lpy``, ``lpz``) in nm, ``area`` and 2D
        ``convexhull`` in nm^2, ``volume`` and 3D ``convexhull`` in
        nm^3.
    info : list of dicts
        Metadata of the cluster centers: ``info`` with an entry
        recording the step appended, see ``lib.append_metadata``.
    """
    has_z = "z" in locs.columns

    group_arr = locs["group"].to_numpy()
    frame_arr = locs["frame"].to_numpy()

    gb, s = _aggregate_cluster_stats(locs, has_z)

    lpx = s["x_std"] / np.sqrt(s["n_locs"])
    lpy = s["y_std"] / np.sqrt(s["n_locs"])
    n_events, order, group_s = _count_binding_events(group_arr, frame_arr)
    with lib.tracked_progress(
        progress, "Calculating cluster centers", unit="cluster"
    ) as progress:
        convexhull = _cluster_convex_hulls(
            locs, order, group_s, s["unique_groups"], has_z, progress
        )

    columns = {
        "frame": s["frame_mean"].astype(np.float32),
        "std_frame": s["frame_std"].astype(np.float32),
        "x": s["x_mean"].astype(np.float32),
        "y": s["y_mean"].astype(np.float32),
        "std_x": s["x_std"].astype(np.float32),
        "std_y": s["y_std"].astype(np.float32),
    }
    if has_z:
        columns["z"] = _weighted_z_means(locs, group_arr).astype(np.float32)
    # per-cluster means of every other numeric column found in locs
    for col in s["extra_cols"]:
        columns[col] = s[f"{col}_mean"].astype(np.float32)
    columns.update(
        {
            "lpx": lpx.astype(np.float32),
            "lpy": lpy.astype(np.float32),
        }
    )
    if has_z:
        columns["lpz"] = (s["z_std"] / np.sqrt(s["n_locs"])).astype(np.float32)
        columns["std_z"] = s["z_std"].astype(np.float32)
    if {"sx", "sy"}.issubset(s["extra_cols"]):
        columns["ellipticity"] = (s["sx_mean"] / s["sy_mean"]).astype(
            np.float32
        )
    columns.update(
        {
            "n_locs": s["n_locs"].astype(np.uint32),
            "n_events": n_events.astype(np.int32),
        }
    )
    if has_z:
        volume = (
            np.power((s["x_std"] + s["y_std"] + s["z_std"]) / 3 * 2, 3)
            * 4.18879
        )  # assume radius = 2 * std_xyz
        columns["volume"] = volume.astype(np.float32)
    else:
        # assume radius = 2 * std_xy
        area = np.power(s["x_std"] + s["y_std"], 2) * np.pi
        columns["area"] = area.astype(np.float32)
    columns["convexhull"] = convexhull.astype(np.float32)
    columns["group"] = s["unique_groups"].astype(np.int32)

    if "group_input" in locs.columns:
        columns["group_input"] = (
            gb["group_input"].first().to_numpy().astype(np.int32)
        )

    entry = lib.step_info(
        "Cluster centers", **{"Number of clusters": len(s["unique_groups"])}
    )
    return pd.DataFrame(columns), lib.append_metadata(info, entry)


def _cluster_area(X: lib.FloatArray2D, lp: float) -> float:
    """Calculate cluster area (2D) or volume (3D). Uses Otsu
    thresholding of the images of the clusters to find areas/volumes.

    Parameters
    ----------
    X : lib.FloatArray2D
        Array of points of shape (n_points, n_dim).
    lp : float
        Median localization precision in x and y of the dataset. Used to
        define the pixel size for image rendering

    Returns
    -------
    area : float
        Cluster area (2D) or volume (3D) in units of LP.
    """
    # get image
    bin_size = lp / 2  # pixel size for rendering
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
        area = np.sum(image >= thresh) / (
            16 / 5
        )  # volume in LP^3, bins are 0.5 LP in xy and 1.25 LP in z
    else:  # 2D
        area = np.sum(image >= thresh) / 4  # area in LP^2, bins are 0.5 LP
    return area


def cluster_areas(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    progress: lib.Progress | Literal["console"] | None = None,
) -> pd.DataFrame:
    """Calculate cluster areas (2D) or volumes (3D).

    Uses Otsu thresholding of the images of the clusters to find areas/
    volumes.

    Parameters
    ----------
    locs : pd.DataFrame
        Clustered localizations (contain group info).
    info : list of dicts or None, optional
        Metadata of ``locs``. Not used; accepted so that the call
        matches the other functions on localizations. Default is None.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating cluster areas"
        over the clusters. Default is None.

    Returns
    -------
    areas : pd.DataFrame
        Cluster areas/volumes for each cluster.
    """
    assert (
        "group" in locs.columns
    ), "Localizations must contain 'group' column."

    groups = np.unique(locs["group"])
    area_key = "Area (LP^2)" if "z" not in locs.columns else "Volume (LP^3)"
    areas = {
        "group": groups.astype(np.int32),
        area_key: np.zeros(len(groups), dtype=np.float32),
    }
    n_groups = len(groups)
    lp = np.median(locs[["lpx", "lpy"]].mean(axis=1))
    with lib.tracked_progress(
        progress, "Calculating cluster areas", unit="cluster"
    ) as progress:
        lib.start_phase(
            progress, "Calculating cluster areas", n_groups, unit="cluster"
        )
        progress.set_value(0)
        for idx in range(n_groups):
            group_id = groups[idx]
            grouplocs = locs[locs["group"] == group_id]
            if len(grouplocs):
                if "z" in grouplocs.columns:
                    X = grouplocs[["x", "y", "z"]].to_numpy()
                else:
                    X = grouplocs[["x", "y"]].to_numpy()
                areas[area_key][idx] = _cluster_area(X, lp)
            progress.set_value(idx + 1)
        progress.set_value(n_groups)
    return pd.DataFrame(areas)


def test_subclustering(
    mols: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    clustering_dist: float = 25,
    sparse_dist: float = 80,
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Extract number of events from molecular maps based on their
    numbers of binding events assigned.

    The reasoning is that 'subclustered' molecules will tend to have
    fewer binding events assigned to them since multiple molecules
    are assigned to a real single molecule. Thus, we can compare the
    distribution of the number of events for the populations of
    molecules that have close neighbors (clustered) and those that
    do not (sparse). If subclustering occurs, the clustered molecules
    should have a lower number of events on average.

    Introduced in Kowalewski, Reinhardt, et al. Nature Comms, 2026.
    DOI: https://doi.org/10.1038/s41467-026-70198-5

    Parameters
    ----------
    mols : pd.DataFrame
        List of molecules, must contain n_events.
    info : list of dicts or None, optional
        Metadata of ``mols``. Not used; accepted so that the call
        matches the other functions on localizations. Default is None.
    clustering_dist : float, optional
        Maximum distance between molecules (nm) to consider them as
        clustered. Default is 25.
    sparse_dist : float, optional
        Minimum distance between molecules (nm) to consider them as sparse.
        Default is 80.

    Returns
    -------
    clustered_nevents : lib.IntArray1D
        Number of events for clustered molecules.
    sparse_nevents : lib.IntArray1D
        Number of events for sparse molecules.
    """
    assert (
        "n_events" in mols.columns
    ), "The input molecules must have n_events attribute."
    assert sparse_dist > clustering_dist, (
        "The sparse distance must be larger than the clustering " "distance."
    )

    # get 1st nearest neighbor distances
    if "z" in mols.columns:
        coords = mols[["x", "y", "z"]].to_numpy()
    else:
        coords = mols[["x", "y"]].to_numpy()
    tree = KDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nnd1 = distances[:, 1]

    # split molecules into clustered and monomeric
    close_nnd_idx = np.where(nnd1 < clustering_dist)[0]
    far_nnd_idx = np.where(nnd1 >= sparse_dist)[0]
    close_mols = mols.iloc[close_nnd_idx]
    far_mols = mols.iloc[far_nnd_idx]
    clustered_nevents = close_mols["n_events"].to_numpy()
    sparse_nevents = far_mols["n_events"].to_numpy()
    return clustered_nevents, sparse_nevents
