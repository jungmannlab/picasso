"""
picasso.clusterer
~~~~~~~~~~~~~~~~~

Clustering of localizations and analysis of the resulting clusters.

- ``cluster``: the SMLM clusterer, optimized for resolving single
  molecules in DNA-PAINT data. Based on Schlichthaerle et al. (2021)
  and Reinhardt et al. (2023).
- ``dbscan``, ``hdbscan``: wrappers around scikit-learn's DBSCAN and
  HDBSCAN.
- ``find_cluster_centers``: one localization-like row per cluster
  (position, spread, size, number of binding events).
- ``cluster_areas``: cluster areas/volumes from thresholded images.
- ``test_subclustering``: checks a molecular map for molecules that
  were wrongly split into several.

All clustering functions add a ``group`` column with the cluster id
and remove unclustered localizations.

References
----------
Schlichthaerle, T., Lindner, C. & Jungmann, R. Super-resolved
visualization of single DNA-based tension sensors in cell adhesion. Nat
Commun 12, 2510 (2021).
https://doi.org/10.1038/s41467-021-22606-1

Reinhardt, S. C. M. et al. Ångström-resolution fluorescence microscopy.
Nature 617, 711-716 (2023).
https://doi.org/10.1038/s41586-023-05925-9

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
    """Check whether one cluster passes basic frame analysis.

    A cluster fails if its mean frame lies outside 20-80 % of
    ``n_frames``, or if more than 80 % of its localizations fall into
    one twentieth of ``n_frames``. Both indicate sticking events or
    other non-repetitive signal rather than repeated DNA-PAINT binding.

    Parameters
    ----------
    frame : pandas.Series
        Frame numbers of the localizations of one cluster (one group of
        a ``SeriesGroupBy``, as passed by ``groupby().apply``).
    n_frames : int
        Number of frames the time range spans.

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
    """Reject clusters that fail basic frame analysis.

    Repeated DNA-PAINT binding spreads a cluster's localizations over
    the whole acquisition, while sticking events are concentrated in
    time. A cluster is rejected if its mean frame lies outside 20-80 %
    of the time range, or if more than 80 % of its localizations fall
    into one twentieth of it. The time range is taken as frame 0 to the
    last frame in ``frame``.

    Parameters
    ----------
    labels : lib.IntArray1D
        Cluster label of each localization (-1 means no cluster
        assigned). Modified in place.
    frame : lib.IntArray1D
        Frame number of each localization.

    Returns
    -------
    labels : lib.IntArray1D
        ``labels`` with the rejected clusters set to -1.
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
    """Cluster points with the SMLM clusterer.

    The general workflow is as follows:

    1. Build a KDTree from the points in X.
    2. For each point, find its neighbors within ``radius`` (the point
       itself included).
    3. Identify local maxima: points that have more than ``min_locs``
       neighbors and no neighbor with more neighbors than themselves.
    4. Assign each local maximum and its neighbors to a cluster. A
       local maximum that already belongs to another one's cluster
       adds its unassigned neighbors to that cluster instead of
       starting a new one.
    5. Remove clusters with fewer than ``min_locs`` points and, if
       ``frame`` is given, clusters failing ``frame_analysis``.

    The neighbor graph is built in batches and stored as compact NumPy
    arrays (see ``_build_neighbor_graph``) so peak memory stays bounded
    even for tens of millions of localizations.

    Based on the algorithm published by Schlichthaerle et al. (2021) and
    first implemented in Reinhardt et al. (2023).

    Parameters
    ----------
    X : lib.FloatArray2D
        Array of points of shape (n_points, n_dim) to be clustered.
    radius : float
        Clustering radius, in the units of ``X``.
    min_locs : int
        Minimum number of localizations in a cluster, also used as the
        neighbor threshold for local maxima (step 3).
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

    References
    ----------
    Schlichthaerle, T., Lindner, C. & Jungmann, R. Super-resolved
    visualization of single DNA-based tension sensors in cell adhesion.
    Nat Commun 12, 2510 (2021).
    https://doi.org/10.1038/s41467-021-22606-1

    Reinhardt, S. C. M. et al. Ångström-resolution fluorescence
    microscopy. Nature 617, 711-716 (2023).
    https://doi.org/10.1038/s41586-023-05925-9
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
    """Cluster 2D localizations with the SMLM clusterer and return the
    labels.

    Low-level helper of ``cluster``, which also adds the labels to the
    localizations and records metadata.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered. Must contain "x" and "y", and
        "frame" if ``fa`` is True.
    radius : float
        Clustering radius (nm).
    min_locs : int
        Minimum number of localizations in a cluster, see ``cluster``.
    fa : bool
        If True, clusters failing basic frame analysis are rejected,
        see ``frame_analysis``.
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
    """Cluster 3D localizations with the SMLM clusterer and return the
    labels.

    Low-level helper of ``cluster``, which also adds the labels to the
    localizations and records metadata.

    Scales z coordinates by ``radius_xy / radius_z`` so that a Euclidean
    neighborhood search with radius ``radius_xy`` in the scaled space
    corresponds to an ellipsoidal neighborhood with semi-axes
    ``(radius_xy, radius_xy, radius_z)`` in the original space.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered. Must contain "x", "y" and "z",
        and "frame" if ``fa`` is True.
    radius_xy : float
        Clustering radius in x and y directions (nm).
    radius_z : float
        Clustering radius in z direction (nm).
    min_locs : int
        Minimum number of localizations in a cluster, see ``cluster``.
    fa : bool
        If True, clusters failing basic frame analysis are rejected,
        see ``frame_analysis``.
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
    """Cluster localizations of single molecules (SMLM clusterer).

    Designed to group the repeated localizations of one DNA-PAINT
    target into one cluster. The general workflow is as follows:

    1. For each localization, find its neighbors within ``radius_xy``
       (the localization itself included). In 3D, the neighborhood is
       an ellipsoid with semi-axes ``(radius_xy, radius_xy,
       radius_z)``.
    2. Identify local maxima: localizations that have more than
       ``min_locs`` neighbors and no neighbor with more neighbors than
       themselves.
    3. Assign each local maximum and its neighbors to a cluster. A
       local maximum that already belongs to another one's cluster
       adds its unassigned neighbors to that cluster instead of
       starting a new one.
    4. Remove clusters with fewer than ``min_locs`` localizations and,
       if ``frame_analysis`` is True, clusters whose localizations are
       concentrated in time (see below).

    Based on the algorithm published by Schlichthaerle et al. (2021) and
    first implemented in Reinhardt et al. (2023).

    The recommended (starting) parameters are ``radius_xy`` of 2*NeNA and
    ``min_locs`` of 10. Keep in mind that the parameters may vary
    between applications, so we encourage you to experiment with them
    when needed. Especially ``min_locs`` may vary between datasets.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered. 3D clustering is used if "z" is
        present.
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    radius_xy : float
        Clustering radius in xy plane (nm).
    min_locs : int
        Minimum number of localizations in a cluster, also used as the
        neighbor threshold for local maxima (step 2).
    frame_analysis : bool
        If True, rejects clusters whose mean frame lies outside 20-80 %
        of the acquisition, or with more than 80 % of their
        localizations in one twentieth of it. Such clusters usually
        come from sticking events rather than repeated binding.
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
        Clustered localizations, with the cluster id of each
        localization in the ``group`` column. Ids are not consecutive.
        Unclustered localizations are removed. An existing ``group``
        column is moved to ``group_input``.
    info : list of dicts
        Metadata of the clustered localizations: ``info`` with an entry
        recording the clustering appended, see ``lib.append_metadata``.

    Raises
    ------
    ValueError
        If ``locs`` is 3D and ``radius_z`` is None.

    See Also
    --------
    find_cluster_centers : Summarizes each cluster as one localization.

    References
    ----------
    Schlichthaerle, T., Lindner, C. & Jungmann, R. Super-resolved
    visualization of single DNA-based tension sensors in cell adhesion.
    Nat Commun 12, 2510 (2021).
    https://doi.org/10.1038/s41467-021-22606-1

    Reinhardt, S. C. M. et al. Ångström-resolution fluorescence
    microscopy. Nature 617, 711-716 (2023).
    https://doi.org/10.1038/s41586-023-05925-9
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

    See Ester et al. (1996).

    Parameters
    ----------
    X : lib.FloatArray2D
        Array of shape (N, D), with N being the number of data points
        and D the number of dimensions.
    radius : float
        DBSCAN search radius, often referred to as "epsilon".
    min_density : int
        Number of points within radius (the point itself included) for
        a point to be a core point.
    min_locs : int, optional
        Minimum number of localizations in a cluster. Clusters with
        fewer localizations will be removed. Default is 0.

    Returns
    -------
    labels : lib.IntArray1D
        Cluster labels for each point. Shape: (N,). -1 means no cluster
        assigned.

    References
    ----------
    Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. A density-based
    algorithm for discovering clusters in large spatial databases with
    noise. In Proc. 2nd Int. Conf. Knowledge Discovery and Data Mining
    (KDD-96) 226-231 (AAAI Press, 1996).
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

    See Ester et al. (1996). Localizations with at least
    ``min_samples`` localizations within ``radius`` are core points;
    core points within ``radius`` of each other form one cluster, which
    also takes in the non-core localizations within ``radius`` of its
    core points. Uses scikit-learn's ``DBSCAN``.

    For 3D data with ``radius_z`` set, anisotropic clustering is used:
    z coordinates are scaled by ``radius / radius_z`` so that the
    isotropic DBSCAN search with epsilon ``radius`` corresponds to an
    ellipsoidal neighborhood with semi-axes
    ``(radius, radius, radius_z)`` in the original space (same approach
    as ``cluster_3D``).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered. 3D clustering is used if "z" is
        present.
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    radius : float
        DBSCAN search radius in the xy plane (nm), usually referred to
        as "epsilon".
    min_samples : int
        Number of localizations within ``radius`` (the localization
        itself included) for a localization to be a core point.
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
        Clustered localizations, with the cluster id of each
        localization in the ``group`` column. Unclustered localizations
        are removed. An existing ``group`` column is moved to
        ``group_input``.
    info : list of dicts
        Metadata of the clustered localizations: ``info`` with an entry
        recording the clustering appended, see ``lib.append_metadata``.

    See Also
    --------
    find_cluster_centers : Summarizes each cluster as one localization.

    References
    ----------
    Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. A density-based
    algorithm for discovering clusters in large spatial databases with
    noise. In Proc. 2nd Int. Conf. Knowledge Discovery and Data Mining
    (KDD-96) 226-231 (AAAI Press, 1996).
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

    See Campello et al. (2013).

    Parameters
    ----------
    X : lib.FloatArray2D
        Array of shape (N, D), with N being the number of data points
        and D the number of dimensions.
    min_cluster_size : int
        Minimum number of points in a cluster.
    min_samples : int
        Number of nearest neighbors (the point itself included) whose
        farthest distance defines a point's core distance.
    cluster_eps : float, optional
        Clusters that would split at distances below this threshold
        are kept merged. Default is 0 (no merging).

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

    References
    ----------
    Campello, R. J. G. B., Moulavi, D. & Sander, J. Density-based
    clustering based on hierarchical density estimates. In Advances in
    Knowledge Discovery and Data Mining (PAKDD 2013) 160-172 (Springer,
    2013).
    https://doi.org/10.1007/978-3-642-37456-2_14
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

    See Campello et al. (2013). Unlike DBSCAN, HDBSCAN needs no search
    radius: it builds a hierarchy of clusters over all density levels
    and keeps the most stable ones, so clusters of different densities
    are found in one run. Uses scikit-learn's ``HDBSCAN``. In 3D, x, y
    and z are weighted equally.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be clustered. 3D clustering is used if "z" is
        present.
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    min_cluster_size : int
        Minimum number of localizations in a cluster.
    min_samples : int
        Number of nearest neighbors (the localization itself included)
        whose farthest distance defines a localization's core distance.
        Larger values make the clustering more conservative: more
        localizations are labeled as noise.
    cluster_eps : float, optional
        Distance threshold (nm). Clusters that would split at distances
        below it are kept merged, so dense structures are not broken
        into many small clusters. Default is 0 (no merging).

    Returns
    -------
    locs : pd.DataFrame
        Clustered localizations, with the cluster id of each
        localization in the ``group`` column. Unclustered localizations
        are removed. An existing ``group`` column is moved to
        ``group_input``.
    info : list of dicts
        Metadata of the clustered localizations: ``info`` with an entry
        recording the clustering appended, see ``lib.append_metadata``.

    Raises
    ------
    RuntimeError
        If ``cluster_eps`` > 0 runs into scikit-learn's HDBSCAN bug with
        NumPy 2.4 and later (scikit-learn issue 33219).

    See Also
    --------
    find_cluster_centers : Summarizes each cluster as one localization.

    References
    ----------
    Campello, R. J. G. B., Moulavi, D. & Sander, J. Density-based
    clustering based on hierarchical density estimates. In Advances in
    Knowledge Discovery and Data Mining (PAKDD 2013) 160-172 (Springer,
    2013).
    https://doi.org/10.1007/978-3-642-37456-2_14
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
    """Add cluster labels to localizations as the ``group`` column and
    remove the unclustered ones (label -1).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be filtered. Modified in place (the ``group``
        column is assigned); pass a copy to keep the original.
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
    """Per-cluster means, standard deviations and sizes in one
    vectorized pass.

    Every numeric column present in ``locs`` is averaged, except the
    ones ``find_cluster_centers`` derives itself
    (``RESERVED_CENTER_COLUMNS``). Standard deviations are computed for
    the frame and the coordinates only.

    Returns the ``groupby`` object (for downstream use such as
    ``gb["group_input"].first()``) and a dict of plain NumPy arrays
    indexed positionally by sorted group id: ``"<col>_mean"``,
    ``"<col>_std"``, ``"n_locs"`` and ``"unique_groups"``, plus the
    names of the averaged non-coordinate columns as ``"extra_cols"``.
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
    on raw NumPy slices of a group-sorted coordinate array. Clusters
    whose points do not span a hull (too few or collinear/coplanar
    points) get 0. ``progress`` reports "Calculating cluster centers"
    over the clusters.
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
    """Per-cluster mean of z, weighted by ``1 / (lpx + lpy)^2`` of each
    localization.

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
    """Summarize each cluster of localizations as one localization.

    Every value of the ``group`` column is treated as one cluster and
    turned into one row of ``centers``. That row has the cluster's
    mean position, its spread and size, and the per-cluster means of
    the other localization columns. The result is a localization
    table, so it can be saved with ``io.save_locs``, opened in Render
    and Filter, and passed on to analyses that work on localizations
    (e.g. nearest-neighbor or Ripley's analysis of the centers).

    Typical input is the output of a clustering function in this
    module (``dbscan``, ``hdbscan``, ``cluster``) or picked
    localizations, where every pick is one group.

    Parameters
    ----------
    locs : pd.DataFrame
        Clustered localizations. Must contain "frame", "x", "y" and
        "group", and "z" for 3D data (3D is detected from the presence
        of "z"). Every row counts towards its group, so remove
        unclustered localizations (``group == -1``) first. The
        clustering functions in this module already do so.
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
        One row per cluster, sorted by group id, with the columns:

        - ``frame``, ``std_frame``: mean and standard deviation of the
          frames of the cluster's localizations.
        - ``x``, ``y``: mean position (nm).
        - ``z`` (3D only): mean axial position (nm), weighted by
          ``1 / (lpx + lpy)^2`` of each localization. Unweighted if
          ``locs`` has no ``lpx``/``lpy``.
        - ``std_x``, ``std_y``, ``std_z`` (3D only): standard
          deviations of the localization positions (nm), i.e. the
          cluster's spatial extent.
        - ``lpx``, ``lpy``, ``lpz`` (3D only): precision of the center
          position as the standard error of the mean,
          ``std / sqrt(n_locs)`` (nm). These are not the mean
          localization precisions of the input.
        - Any other numeric column of ``locs`` (e.g. ``photons``,
          ``bg``, ``sx``, ``sy``): its per-cluster mean.
        - ``ellipticity``: ``sx / sy`` of the cluster means, only if
          ``locs`` has ``sx`` and ``sy``.
        - ``n_locs``: number of localizations in the cluster.
        - ``n_events``: number of binding events. A new event starts
          whenever two consecutive localizations of the cluster are
          more than 3 frames apart. Consecutive means in the row order
          of ``locs``, so ``locs`` should be sorted by frame (as
          Picasso saves localizations).
        - ``area`` (2D, nm^2) or ``volume`` (3D, nm^3): size of a disk
          or sphere whose radius is twice the mean standard deviation,
          i.e. ``pi * (std_x + std_y)^2`` in 2D and
          ``4/3 * pi * (2 * (std_x + std_y + std_z) / 3)^3`` in 3D.
        - ``convexhull``: area (2D, nm^2) or volume (3D, nm^3) of the
          convex hull of the cluster's localizations. 0 for clusters
          too small or too flat to span a hull (e.g. fewer than 3
          localizations in 2D).
        - ``group``: the cluster's group id.
        - ``group_input``: copied from ``locs`` if present (the
          grouping before clustering, e.g. pick ids), taken from the
          cluster's first localization.

        Input columns named like any of the derived columns above are
        not averaged; the derived values replace them.
    info : list of dicts
        Metadata of the cluster centers: ``info`` with an entry
        recording the step and the number of clusters appended, see
        ``lib.append_metadata``.

    See Also
    --------
    dbscan, hdbscan, cluster : Clustering functions whose output is
        the usual input here.
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
    """Calculate the area (2D) or volume (3D) of one cluster from a
    thresholded image.

    The points are binned with a pixel size of ``lp / 2`` (``1.25 *
    lp`` in z), blurred with a Gaussian of sigma ``lp`` (2 pixels in
    every dimension) and thresholded with Otsu's method. The pixels at
    or above the threshold make up the area/volume.

    Parameters
    ----------
    X : lib.FloatArray2D
        Array of points of shape (n_points, n_dim), n_dim 2 or 3.
    lp : float
        Median localization precision in x and y of the dataset (nm).
        Sets the pixel size for rendering and the unit of the result.

    Returns
    -------
    area : float
        Cluster area in units of ``lp^2`` (2D) or volume in units of
        ``lp^3`` (3D).
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
    """Calculate cluster areas (2D) or volumes (3D) from thresholded
    images.

    Each cluster is rendered with a pixel size of half the localization
    precision ``LP`` (``1.25 * LP`` in z), blurred with a Gaussian of
    sigma ``LP`` and thresholded with Otsu's method; the pixels above
    the threshold make up the area/volume. ``LP`` is the median of
    ``(lpx + lpy) / 2`` over all localizations. Unlike the hull-based
    and std-based sizes of ``find_cluster_centers``, this follows the
    shape of the localization density.

    Parameters
    ----------
    locs : pd.DataFrame
        Clustered localizations. Must contain "x", "y", "lpx", "lpy"
        and "group" (and "z" for 3D).
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
        One row per cluster, sorted by group id, with the columns
        ``group`` and ``Area (LP^2)`` (2D) or ``Volume (LP^3)`` (3D).
        The sizes are in units of ``LP``, not nm; multiply by ``LP^2``
        or ``LP^3`` to convert.

    Raises
    ------
    AssertionError
        If ``locs`` has no ``group`` column.
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
    """Split a molecular map by nearest-neighbor distance to test for
    subclustering.

    Subclustering means that the localizations of one real molecule
    were split into several molecules. Each of these then gets only
    part of the molecule's binding events, and they lie close to each
    other. So, if subclustering occurs, molecules with a close nearest
    neighbor ("clustered") have fewer binding events on average than
    isolated ("sparse") ones. This function returns the numbers of
    binding events of both populations so that their distributions can
    be compared, e.g. with histograms or their means. Molecules with a
    nearest-neighbor distance between ``clustering_dist`` and
    ``sparse_dist`` are in neither population.

    Introduced in Kowalewski et al. (2026).

    Parameters
    ----------
    mols : pd.DataFrame
        Molecules, e.g. the output of ``find_cluster_centers`` or
        G5M. Must contain "x", "y" and "n_events" (and "z" for 3D,
        which makes the distances 3D).
    info : list of dicts or None, optional
        Metadata of ``mols``. Not used; accepted so that the call
        matches the other functions on localizations. Default is None.
    clustering_dist : float, optional
        Molecules whose nearest neighbor is closer than this (nm) are
        clustered. Default is 25.
    sparse_dist : float, optional
        Molecules whose nearest neighbor is at least this far (nm) are
        sparse. Must be larger than ``clustering_dist``. Default is 80.

    Returns
    -------
    clustered_nevents : lib.IntArray1D
        ``n_events`` of the clustered molecules.
    sparse_nevents : lib.IntArray1D
        ``n_events`` of the sparse molecules.

    Raises
    ------
    AssertionError
        If ``mols`` has no ``n_events`` column, or if ``sparse_dist``
        is not larger than ``clustering_dist``.

    References
    ----------
    Kowalewski, R. et al. Molecular mapping in DNA-PAINT via modified
    Gaussian Mixture Modeling. Nat Commun 17, 2315 (2026).
    https://doi.org/10.1038/s41467-026-70198-5
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
