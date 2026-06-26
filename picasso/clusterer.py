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
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import ConvexHull, KDTree, QhullError
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN, HDBSCAN

from . import lib, masking, __version__


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
    progress: Callable[[int], None] | None = None,
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
    progress : callable or None, optional
        Called with the cumulative number of points processed after
        each batch. If None, a tqdm progress bar is shown.

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
    iterator = range(n_batches)
    if progress is None:
        iterator = tqdm(iterator, desc="Clustering (finding neighbors)")

    for b in iterator:
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
        if progress is not None:
            progress(end)

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
    frame: pd.Series | None = None,
    progress: Callable[[int], None] | None = None,
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
    frame : pd.Series or None, optional
        Frame number of each localization. If None, no frame analysis
        is performed.
    progress : callable or None, optional
        Called with the cumulative number of localizations processed
        while building the neighbor graph (the main, O(n) step). If
        None, a tqdm progress bar is shown.

    Returns
    -------
    labels : lib.IntArray1D
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    n_points = X.shape[0]

    # build kdtree and a compact, batched neighbor graph (bounded memory)
    tree = KDTree(X)
    indptr, indices = _build_neighbor_graph(tree, X, radius, progress)

    # number of neighbors of each point (point is its own neighbor)
    counts = np.diff(indptr).astype(np.int32)

    # find local maxima, i.e., points with the most neighbors within
    # their neighborhood. ``reduceat`` computes, per point, the maximum
    # neighbor count over that point's neighborhood in one vectorised
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
    progress: Callable[[int], None] | None = None,
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
    progress : callable or None, optional
        Called with the cumulative number of localizations processed
        while building the neighbor graph. If None, a tqdm progress bar
        is shown.

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

    labels = _cluster(X, radius, min_locs, frame, progress)

    return labels


def cluster_3D(
    locs: pd.DataFrame,
    radius_xy: float,
    radius_z: float,
    min_locs: int,
    fa: bool,
    progress: Callable[[int], None] | None = None,
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
        Clustering radius in x and y directions, in the same units as
        ``locs["x"]`` / ``locs["y"]`` (camera pixels).
    radius_z : float
        Clustering radius in z direction, in the same units as
        ``locs["z"]`` (camera pixels after ``cluster()``'s nm-to-px
        conversion).
    min_locs : int
        Minimum number of localizations in a cluster.
    fa : bool
        True, if basic frame analysis is to be performed.
    progress : callable or None, optional
        Called with the cumulative number of localizations processed
        while building the neighbor graph. If None, a tqdm progress bar
        is shown.

    Returns
    -------
    labels : lib.IntArray1D
        Cluster labels for each localization (-1 means no cluster
        assigned).
    """
    radius = radius_xy
    X = locs[["x", "y", "z"]].to_numpy()
    X[:, 2] *= radius_xy / radius_z

    if not fa:
        frame = None
    else:
        frame = locs["frame"]

    labels = _cluster(X, radius, min_locs, frame, progress)

    return labels


def cluster(
    locs: pd.DataFrame,
    radius_xy: float,
    min_locs: int,
    frame_analysis: bool,
    radius_z: float | None = None,
    pixelsize: float | None = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
    progress: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
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
    return_info : bool, optional
        If True, returns a tuple of (locs, info), where locs is the
        clustered localizations and info is a dictionary containing
        clustering information.
    return_info : bool, optional
        If True, returns a tuple of (locs, info), where locs is the
        clustered localizations and info is a dictionary containing
        clustering information. Will be removed in v0.12.0 and both
        locs and metadata will be returned.
    progress : callable or None, optional
        Called with the cumulative number of localizations processed
        while building the neighbor graph (the main, O(n) step). Useful
        for wiring a progress bar in a GUI. If None, a tqdm progress bar
        is shown in the console.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    info : dict, optional
        Dictionary containing clustering information, only returned if
        return_info is True.
    """
    if not return_info:
        lib.deprecation_warning(
            "In v0.12.0, return_info will not be an argument and"
            "cluster will always return both locs and cluster info."
        )
    locs = locs.copy()
    n_raw = len(locs)
    if "z" in locs.columns:  # 3D
        if pixelsize is None or radius_z is None:
            raise ValueError(
                "Camera pixel size and clustering radius in z must be"
                " specified for 3D clustering."
            )
        locs["z"] /= pixelsize  # convert from nm to px
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
    if "z" in locs.columns:
        locs["z"] *= pixelsize  # convert back to nm
    n_clusters = len(locs)
    info = {
        "Generated by": f"Picasso v{__version__} SMLM clusterer",
        "Number of clusters": len(np.unique(locs["group"])),
        "Min. cluster size": min_locs,
        "Performed basic frame analysis": frame_analysis,
        "Fraction of rejected locs (%)": 100 * (n_raw - n_clusters) / n_raw,
    }
    unit = "nm" if pixelsize is not None else "px"
    pixelsize = pixelsize if pixelsize is not None else 1
    if "z" in locs.columns:
        info[f"Clustering radius xy ({unit})"] = radius_xy * pixelsize
        info[f"Clustering radius z ({unit})"] = radius_z * pixelsize
    else:
        info[f"Clustering radius ({unit})"] = radius_xy * pixelsize
    if return_info:
        return locs, info
    else:
        return locs


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
    radius: float,
    min_samples: int,
    min_locs: int = 10,
    pixelsize: float | None = None,
    radius_z: float | None = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
    """Perform DBSCAN on localizations.

    See Ester, et al. Inkdd, 1996. (Vol. 96, No. 34, pp. 226-231).

    For 3D data with ``radius_z`` set, anisotropic clustering is used:
    z coordinates are scaled by ``radius / radius_z`` so that the
    isotropic DBSCAN search with epsilon ``radius`` corresponds to an
    ellipsoidal neighborhood with semi-axes
    ``(radius, radius, radius_z)`` in the original space (same approach
    as ``cluster_3D``).

    Parameters
    ---------
    locs : pd.DataFrame
        Localizations to be clustered.
    radius : float
        DBSCAN search radius in the xy plane, usually referred to as
        "epsilon". Same units as ``locs["x"]`` / ``locs["y"]``
        (camera pixels).
    min_samples : int
        Number of localizations within radius to consider a given point
        a core sample.
    min_locs : int, optional
        Minimum number of localizations in a cluster. Clusters with
        fewer localizations will be removed. Default is 0.
    pixelsize : float, optional
        Camera pixel size in nm. Only needed for 3D.
    radius_z : float, optional
        DBSCAN search radius in z (camera pixels). If None (default),
        the clustering is isotropic and uses ``radius`` in all
        dimensions. Only used for 3D.
    return_info : bool, optional
        If True, returns a tuple of (locs, info), where locs is the
        clustered localizations and info is a dictionary containing
        clustering information. Will be removed in v0.12.0 and both
        locs and metadata will be returned.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    info : dict, optional
        Dictionary containing clustering information, only returned if
        return_info is True.
    """
    if not return_info:
        lib.deprecation_warning(
            "In v0.12.0, return_info will not be an argument and"
            "dbscan will always return both locs and cluster info."
        )
    locs = locs.copy()
    n_raw = len(locs)
    if "z" in locs.columns:
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size must be specified as an integer for 3D"
                " clustering."
            )
        X = locs[["x", "y", "z"]].to_numpy()
        X[:, 2] /= pixelsize
        if radius_z is not None:
            X[:, 2] *= radius / radius_z
    else:
        X = locs[["x", "y"]].to_numpy()
    labels = _dbscan(X, radius, min_samples, min_locs)
    locs = extract_valid_labels(locs, labels)
    n_clusters = len(locs)
    unit = "nm" if pixelsize is not None else "px"
    pixelsize_unit = pixelsize if pixelsize is not None else 1
    info = {
        "Generated by": f"Picasso v{__version__} DBSCAN",
        "Number of clusters": len(np.unique(locs["group"])),
        f"Radius ({unit})": radius * pixelsize_unit,
        "Minimum local density": min_samples,
        "Min. localizations per cluster": min_locs,
        "Fraction of rejected locs (%)": 100 * (n_raw - n_clusters) / n_raw,
    }
    if "z" in locs.columns and radius_z is not None:
        info[f"Radius z ({unit})"] = radius_z * pixelsize_unit
    if return_info:
        return locs, info
    else:
        return locs


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
    """
    hdb = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_eps,
        copy=False,
    ).fit(X)
    return hdb.labels_.astype(np.int32)


def hdbscan(
    locs: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int,
    pixelsize: float | None = None,
    cluster_eps: float = 0.0,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
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
    pixelsize : float, optional
        Camera pixel size in nm. Only needed for 3D.
    cluster_eps : float, optional
        Distance threshold. Clusters below this value will be merged.
    return_info : bool, optional
        If True, returns a tuple of (locs, info), where locs is the
        clustered localizations and info is a dictionary containing
        clustering information. Will be removed in v0.12.0 and both
        locs and metadata will be returned.

    Returns
    -------
    locs : pd.DataFrame
        Clusterered localizations, with column 'group' added, which
        specifies cluster label for each localization. Noise (label -1)
        is removed.
    info : dict, optional
        Dictionary containing clustering information, only returned if
        return_info is True.
    """
    if not return_info:
        lib.deprecation_warning(
            "In v0.12.0, return_info will not be an argument and"
            "cluster will always return both locs and cluster info."
        )
    locs = locs.copy()
    n_raw = len(locs)
    if "z" in locs.columns:
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size (nm) must be specified as an integer for 3D"
                " clustering."
            )
        X = locs[["x", "y", "z"]].to_numpy()
        X[:, 2] /= pixelsize
    else:
        X = locs[["x", "y"]].to_numpy()
    labels = _hdbscan(
        X, min_cluster_size, min_samples, cluster_eps=cluster_eps
    )
    locs = extract_valid_labels(locs, labels)
    n_clusters = len(locs)
    info = {
        "Generated by": f"Picasso v{__version__} HDBSCAN",
        "Number of clusters": len(np.unique(locs["group"])),
        "Min. cluster": min_cluster_size,
        "Min. samples": min_samples,
        "Intercluster distance": cluster_eps,
        "Fraction of rejected locs (%)": 100 * (n_raw - n_clusters) / n_raw,
    }
    if return_info:
        return locs, info
    else:
        return locs


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
        cluster label.
    """
    # add cluster id to locs, as "group"
    locs["group"] = labels

    # -1 means no cluster assigned to a loc
    locs = locs[locs["group"] != -1]
    return locs


def _aggregate_cluster_stats(
    locs: pd.DataFrame, has_z: bool
) -> tuple[pd.core.groupby.DataFrameGroupBy, dict]:
    """One vectorised pass for per-group means, stds and sizes.

    Returns the underlying ``groupby`` object (for downstream use such
    as ``group_input.first()``) and a dict of plain NumPy arrays, one
    per statistic, indexed positionally by sorted group id."""
    mean_cols = [
        "frame",
        "x",
        "y",
        "photons",
        "sx",
        "sy",
        "bg",
        "net_gradient",
    ]
    std_cols = ["frame", "x", "y"]
    if has_z:
        mean_cols.append("z")
        std_cols.append("z")

    gb = locs.groupby("group", sort=True)
    means = gb[mean_cols].mean()
    stds = gb[std_cols].std()

    stats = {f"{c}_mean": means[c].to_numpy() for c in mean_cols}
    stats.update({f"{c}_std": stds[c].to_numpy() for c in std_cols})
    stats["n_locs"] = gb.size().to_numpy()
    stats["unique_groups"] = means.index.to_numpy()
    return gb, stats


def _count_binding_events(
    group_arr: lib.IntArray1D, frame_arr: lib.IntArray1D
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.IntArray1D]:
    """Number of binding events per cluster.

    A new event starts whenever consecutive frames within a cluster are
    more than 3 frames apart. Vectorised across all groups with one
    stable sort + one diff pass.

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
    new_event[0] = True
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
    pixelsize: float | None,
    progress: Callable[[int], None] | None = None,
) -> lib.FloatArray1D:
    """Convex-hull area (2D) or volume (3D) per cluster.

    The only per-cluster Python loop in ``find_cluster_centers``; runs
    on raw NumPy slices of a group-sorted coordinate array. For datasets
    with many clusters this is the run-time bottleneck, so an optional
    ``progress`` callable (called with the number of clusters processed)
    can be passed for feedback.
    """
    coord_cols = ["x", "y", "z"] if has_z else ["x", "y"]
    coords_sorted = (
        locs[coord_cols].to_numpy()[order].astype(np.float64, copy=True)
    )
    if has_z:
        coords_sorted[:, 2] /= pixelsize
    group_offsets = np.searchsorted(group_s, unique_groups, side="left")
    group_offsets = np.append(group_offsets, len(group_s))
    convexhull = np.zeros(len(unique_groups), dtype=np.float64)
    for i in range(len(unique_groups)):
        X = coords_sorted[group_offsets[i] : group_offsets[i + 1]]
        try:
            convexhull[i] = ConvexHull(X).volume
        except QhullError:
            convexhull[i] = 0.0
        if progress is not None:
            progress(i + 1)
    return convexhull


def _weighted_z_means(
    locs: pd.DataFrame, group_arr: lib.IntArray1D
) -> lib.FloatArray1D:
    """Per-cluster z mean weighted by 1/(lpx + lpy)^2 (per-row weights)."""
    w = 1.0 / (locs["lpx"].to_numpy() + locs["lpy"].to_numpy()) ** 2
    wz = (
        pd.Series(locs["z"].to_numpy() * w).groupby(group_arr, sort=True).sum()
    )
    ws = pd.Series(w).groupby(group_arr, sort=True).sum()
    return (wz / ws).to_numpy()


def find_cluster_centers(
    locs: pd.DataFrame,
    pixelsize: float | None = None,
    progress: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """Calculate cluster centers.

    Aggregations are computed in vectorised pandas/NumPy passes; the
    only per-cluster Python loop is the convex hull, which operates on
    raw NumPy slices.

    Parameters
    ----------
    locs : pd.DataFrame
        Clustered localizations (contain group info)
    pixelsize : float, optional
        Camera pixel size (used for finding volume and 3D convex hull).
        Only required for 3D localizations.
    progress : callable or None, optional
        Called with the cumulative number of clusters processed during
        the per-cluster convex-hull pass (the run-time bottleneck for
        datasets with many clusters). Useful for wiring a progress bar
        in a GUI. If None (default), no progress is reported.

    Returns
    -------
    centers : pd.DataFrame
        Cluster centers saved in the format of localizations.
    """
    has_z = "z" in locs.columns
    if has_z and pixelsize is None:
        raise ValueError(
            "Camera pixel size must be specified as an integer for 3D"
            " cluster centers calculation."
        )

    group_arr = locs["group"].to_numpy()
    frame_arr = locs["frame"].to_numpy()

    gb, s = _aggregate_cluster_stats(locs, has_z)

    lpx = s["x_std"] / np.sqrt(s["n_locs"])
    lpy = s["y_std"] / np.sqrt(s["n_locs"])
    ellipticity = s["sx_mean"] / s["sy_mean"]
    n_events, order, group_s = _count_binding_events(group_arr, frame_arr)
    convexhull = _cluster_convex_hulls(
        locs, order, group_s, s["unique_groups"], has_z, pixelsize, progress
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
    columns.update(
        {
            "photons": s["photons_mean"].astype(np.float32),
            "sx": s["sx_mean"].astype(np.float32),
            "sy": s["sy_mean"].astype(np.float32),
            "bg": s["bg_mean"].astype(np.float32),
            "lpx": lpx.astype(np.float32),
            "lpy": lpy.astype(np.float32),
        }
    )
    if has_z:
        columns["lpz"] = (s["z_std"] / np.sqrt(s["n_locs"])).astype(np.float32)
        columns["std_z"] = s["z_std"].astype(np.float32)
    columns.update(
        {
            "ellipticity": ellipticity.astype(np.float32),
            "net_gradient": s["net_gradient_mean"].astype(np.float32),
            "n_locs": s["n_locs"].astype(np.uint32),
            "n_events": n_events.astype(np.int32),
        }
    )
    if has_z:
        volume = (
            np.power(
                (s["x_std"] + s["y_std"] + s["z_std"] / pixelsize) / 3 * 2, 3
            )
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

    return pd.DataFrame(columns)


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
    progress : callable or None, optional
        Callable accepting an int (progress count). If None, progress
        is displayed in the console. Default is None.

    Returns
    -------
    areas : pd.DataFrame
        Cluster areas/volumes for each cluster.
    """
    assert (
        "group" in locs.columns
    ), "Localizations must contain 'group' column."

    # get pixel size from info
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)

    groups = np.unique(locs["group"])
    area_key = "Area (LP^2)" if "z" not in locs.columns else "Volume (LP^3)"
    areas = {
        "group": groups.astype(np.int32),
        area_key: np.zeros(len(groups), dtype=np.float32),
    }
    if progress is None:
        iterator = tqdm(range(len(groups)), desc="Calculating cluster areas")
    else:
        iterator = range(len(groups))

    lp = np.median(locs[["lpx", "lpy"]].mean(axis=1))
    for idx in iterator:
        group_id = groups[idx]
        grouplocs = locs[locs["group"] == group_id]
        if not len(grouplocs):
            continue
        if "z" in grouplocs.columns:
            X = grouplocs[["x", "y", "z"]].to_numpy()
            X[:, 2] /= pixelsize  # convert z to pixels
        else:
            X = grouplocs[["x", "y"]].to_numpy()
        areas[area_key][idx] = _cluster_area(X, lp)
        if progress is not None:
            progress(idx + 1)
    return pd.DataFrame(areas)


def test_subclustering(
    mols: pd.DataFrame,
    info: list[dict],
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
    info : list of dict
        Molecules metadata.
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
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)

    # get 1st nearest neighbor distances
    if "z" in mols.columns:
        coords = mols[["x", "y", "z"]].to_numpy()
        coords[:, 2] /= pixelsize
    else:
        coords = mols[["x", "y"]].to_numpy()
    tree = KDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nnd1 = distances[:, 1]

    # split molecules into clustered and monomeric
    close_nnd_idx = np.where(nnd1 < clustering_dist / pixelsize)[0]
    far_nnd_idx = np.where(nnd1 >= sparse_dist / pixelsize)[0]
    close_mols = mols.iloc[close_nnd_idx]
    far_mols = mols.iloc[far_nnd_idx]
    clustered_nevents = close_mols["n_events"].to_numpy()
    sparse_nevents = far_mols["n_events"].to_numpy()
    return clustered_nevents, sparse_nevents
