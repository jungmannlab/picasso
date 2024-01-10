"""
    picasso.clusterer
    ~~~~~~~~~~~~~~~~~

    Clusterer optimized for DNA PAINT, as well as DBSCAN and HDBSCAN.

    Based on the work of Thomas Schlichthaerle and Susanne Reinhardt.
    :authors: Thomas Schlichthaerle, Susanne Reinhardt, 
        Rafal Kowalewski, 2020-2022
    :copyright: Copyright (c) 2022 Jungmann Lab, MPI of Biochemistry
"""

import numpy as _np
import pandas as _pd
from scipy.spatial import cKDTree as _cKDTree
from scipy.spatial import ConvexHull as _ConvexHull
from sklearn.cluster import DBSCAN as _DBSCAN

from . import lib as _lib

CLUSTER_CENTERS_DTYPE_2D = [
    ("frame", "f4"),
    ("std_frame", "f4"),
    ("x", "f4"),
    ("y", "f4"),
    ("std_x", "f4"),
    ("std_y", "f4"),
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
    ("ellipticity", "f4"),
    ("net_gradient", "f4"),
    ("n", "u4"),
    ("area", "f4"),
    ("convexhull", "f4"),
    ("group", "i4"),
] 
CLUSTER_CENTERS_DTYPE_3D = [
    ("frame", "f4"),
    ("std_frame", "f4"),
    ("x", "f4"),
    ("y", "f4"),
    ("std_x", "f4"),
    ("std_y", "f4"),
    ("z", "f4"),
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
    ("std_z", "f4"),
    ("ellipticity", "f4"),
    ("net_gradient", "f4"),
    ("n", "u4"),
    ("volume", "f4"),
    ("convexhull", "f4"),
    ("group", "i4"),
]

def _frame_analysis(frame, n_frames):
    """
    Verifies which clusters pass basic frame analysis.

    Rejects clusters whose mean frame is outside of the 
    [20, 80] % (max frame) range or any 1/20th of measurement's time
    contains more than 80 % of localizations.

    Assumes frame to be a pandas.SeriesGruopBy object, grouped by
    cluster ids.

    Parameters
    ----------
    frame : pandas.SeriesGruopBy
        Frame number for a given cluster; grouped by cluster ids
    n_frames : int
        Acquisition time given in frames

    Returns
    -------
    int
        1 if passed frame analysis, 0 otheriwse
    """

    passed = 1

    # get mean frame
    mean_frame = frame.values.mean()

    # get maximum number of locs in a 1/20th of acquisition time
    n_locs = len(frame)
    locs_binned = _np.histogram(
        frame.values, bins=_np.linspace(0, n_frames, 21)
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


def frame_analysis(labels, frame):
    """
    Performs basic frame analysis on clustered localizations.

    Rejects clusters whose mean frame is outside of the 
    [20, 80] % (max frame) range or any 1/20th of measurement's time
    contains more than 80 % of localizations.

    Uses pandas for fast calculations using groupby().

    Parameters
    ----------
    labels : np.array
        Cluster labels (-1 means no cluster assigned)
    frame : np.array
        Frame number for each localization

    Returns
    -------
    np.array
        Cluster labels for each localization (-1 means no cluster 
        assigned)
    """

    # group frames by cluster ids
    frame_pd = _pd.Series(frame, index=labels)
    frame_grouped = frame_pd.groupby(frame_pd.index)

    # perform frame analysis
    true_cluster = frame_grouped.apply(_frame_analysis, frame.max()+1)

    # cluster ids that did not pass frame analysis
    discard = true_cluster.index[true_cluster == 0].values
    # change labels of these clusters to -1
    labels[_np.isin(labels, discard)] = -1

    return labels


def _cluster(X, radius, min_locs, frame=None):
    """
    Clusters points given by X with a given clustering radius and 
    minimum number of localizaitons withing that radius using KDTree

    Parameters
    ----------
    X : np.array
        Array of points of shape (n_points, n_dim) to be clustered
    radius : float
        Clustering radius
    min_locs : int
        Minimum number of localizations in a cluster
    frame : np.array (default=None)
        Frame number of each localization. If None, no frame analysis
        is performed

    Returns
    -------
    np.array
        Cluster labels for each localization (-1 means no cluster 
        assigned)
    """

    ## build kdtree (use cKDTree in case user did not update scipy)
    tree = _cKDTree(X)

    ## find neighbors for each point within radius
    neighbors = tree.query_ball_tree(tree, radius)

    ## find local maxima, i.e., points with the most neighbors within
    ## their neighborhood
    lm = _np.zeros(X.shape[0], dtype=_np.int8)
    for i in range(len(lm)):
        idx = neighbors[i] # indeces of points that are neighbors of i
        n = len(idx) # number of neighbors of i
        if n > min_locs: # note that i is included in its neighbors
            # if i has the most neighbors in its neighborhood
            if n == max([len(neighbors[_]) for _ in idx]):
                lm[i] = 1

    ## assign cluster labels to all points (-1 means no cluster)
    ## if two local maxima are within radius from each other, combine
    ## such clusters
    labels = -1 * _np.ones(X.shape[0], dtype=_np.int32) # cluster labels
    lm_idx = _np.where(lm == 1)[0] # indeces of local maxima

    for count, i in enumerate(lm_idx): # for each local maximum
        label = labels[i]
        if label == -1: # if lm not assigned yet
            labels[neighbors[i]] = count
        else:
            # indeces of locs that were not assigned to any cluster
            idx = [
                neighbors[i][_]
                for _ in _np.where(labels[neighbors[i]] == -1)[0]
            ]
            if len(idx): # if such a loc exists, assign it to a cluster
                labels[idx] = label

    ## check for number of locs per cluster to be above min_locs
    values, counts = _np.unique(labels, return_counts=True)
    # labels to discard if has fewer locs than min_locs
    to_discard = values[counts < min_locs]
    # substitute this with -1
    labels[_np.isin(labels, to_discard)] = -1

    if frame is not None:
        labels = frame_analysis(labels, frame)

    return labels


def cluster_2D(x, y, frame, radius, min_locs, fa):
    """
    Prepares 2D input to be used by _cluster()

    Parameters
    ----------
    x : np.array
        x coordinates to be clustered
    y : np.array
        y coordinates to be clustered
    frame : np.array
        Frame number for each localization 
    radius : float
        Clustering radius
    min_locs : int
        Minimum number of localizations in a cluster
    fa : bool
        True, if basic frame analysis is to be performed

    Returns
    -------
    np.array
        Cluster labels for each localization (-1 means no cluster 
        assigned)
    """

    X = _np.stack((x, y)).T

    if not fa:
        frame = None

    labels = _cluster(X, radius, min_locs, frame)

    return labels


def cluster_3D(x, y, z, frame, radius_xy, radius_z, min_locs, fa):
    """
    Prepares 3D input to be used by _cluster()

    Scales z coordinates by radius_xy / radius_z

    Parameters
    ----------
    x : np.array
        x coordinates to be clustered
    y : np.array
        y coordinates to be clustered
    z : np.array
        z coordinates to be clustered
    frame : np.array
        Frame number for each localization 
    radius_xy : float
        Clustering radius in x and y directions
    radius_z : float
        Clutsering radius in z direction
    min_locs : int
        Minimum number of localizations in a cluster
    fa : bool
        True, if basic frame analysis is to be performed

    Returns
    -------
    np.array
        Cluster labels for each localization (-1 means no cluster 
        assigned)
    """

    radius = radius_xy
    X = _np.stack((x, y, z * radius_xy / radius_z)).T

    if not fa:
        frame = None

    labels = _cluster(X, radius, min_locs, frame)
    
    return labels


def cluster(locs, params, pixelsize=None):
    """
    Clusters localizations given user parameters using KDTree.

    Finds if localizations are 2D or 3D.

    Paramaters
    ----------
    locs : np.recarray
        Localizations to be clustered
    params : tuple
        SMLM clustering parameters
    pixelsize : int (default=None)
        Camera pixel size in nm. Only needed for 3D clustering.

    Returns
    -------
    np.array
        Cluster labels for each localization (-1 means no cluster 
        assigned)
    """

    if hasattr(locs, "z"): # 3D
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size must be specified as an integer for 3D"
                " clustering."
            )
        radius_xy, radius_z, min_locs, _, fa, _ = params
        labels = cluster_3D(
            locs.x,
            locs.y,
            locs.z / pixelsize, # convert z coordinates from nm to px
            locs.frame,
            radius_xy,
            radius_z,
            min_locs,
            fa
        )
    else:
        radius, min_locs, _, fa, _ = params
        labels = cluster_2D(
            locs.x,
            locs.y,
            locs.frame,
            radius,
            min_locs,
            fa
        )
    locs = extract_valid_labels(locs, labels)
    return locs


def _dbscan(X, radius, min_density):
    """ 
    Finds DBSCAN cluster labels, given data points and parameters.
    
    Parameters
    ----------
    X : np.array
        Array of shape (N, D), with N being the number of data points
        and D the number of dimensions.
    radius : float
        DBSCAN search radius, often referred to as "epsilon"
    min_density : int
        Number of points within radius to consider a given point a core
        sample

    Returns
    -------
    labels : np.array
        Cluster labels for each point. Shape: (N,). -1 means no cluster
        assigned.
    """

    db = _DBSCAN(eps=radius, min_samples=min_density).fit(X)
    return db.labels_.astype(_np.int32)


def dbscan(locs, radius, min_density, pixelsize=None):
    """
    Performs DBSCAN on localizations.
    
    Paramters
    ---------
    locs : np.recarray
        Localizations to be clustered.
    radius : float
        DBSCAN search radius, often referred to as "epsilon". Same units
        as locs.
    min_density : int
        Number of localizations within radius to consider a given point 
        a core sample.
    pixelsize : int (default=None)
        Camera pixel size in nm. Only needed for 3D.
    
    Returns
    -------
    locs : np.recarray
        Clustered localizations; cluster labels are assigned to the
        "group" column    
    """
    
    if hasattr(locs, "z"):
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size must be specified as an integer for 3D"
                " clustering."
            )
        X = _np.vstack((locs.x, locs.y, locs.z / pixelsize)).T
    else:
        X = _np.vstack((locs.x, locs.y)).T
    labels = _dbscan(X, radius, min_density)
    locs = extract_valid_labels(locs, labels)
    return locs


def _hdbscan(X, min_cluster_size, min_samples, cluster_eps=0):
    """
    Finds HDBSCAN cluster labels, given data points and parameters.

    Parameters
    ----------
    X : np.array
        Array of shape (N, D), with N being the number of data points
        and D the number of dimensions.
    min_cluster_size : int
        Minimum number of points in cluster
    min_samples : int
        Number of points within radius to consider a given point a core
        sample
    cluster_eps : float (default=0.)
        Distance threshold. Clusters below this value will be merged
    
    Returns
    -------
    labels : np.array
        Cluster labels for each point. Shape: (N,). -1 means no cluster
        assigned.
    """
    
    from sklearn.cluster import HDBSCAN as _HDBSCAN
    hdb = _HDBSCAN(
        min_samples=min_samples, 
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_eps,
    ).fit(X)
    return hdb.labels_.astype(_np.int32)


def hdbscan(
    locs, min_cluster_size, min_samples, pixelsize=None, cluster_eps=0.0
):
    """
    Performs HDBSCAN on localizations.
    
    Paramters
    ---------
    locs : np.recarray
        Localizations to be clustered
    min_cluster_size : int
        Minimum number of localizations in cluster
    min_samples : int
        Number of localizations within radius to consider a given point 
        a core sample
    pixelsize : int (default=None)
        Camera pixel size in nm
    cluster_eps : float (default=0.)
        Distance threshold. Clusters below this value will be merged
    
    Returns
    -------
    locs : np.recarray
        Clustered localizations; cluster labels are assigned to the
        "group" column    
    """

    if hasattr(locs, "z"):
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size (nm) must be specified as an integer for 3D"
                " clustering."
            )
        X = _np.vstack((locs.x, locs.y, locs.z / pixelsize)).T
    else:
        X = _np.vstack((locs.x, locs.y)).T
    labels = _hdbscan(
        X, min_cluster_size, min_samples, cluster_eps=cluster_eps
    )
    locs = extract_valid_labels(locs, labels)
    return locs


def extract_valid_labels(locs, labels):
    """
    Extracts localizations based on clustering results.

    Localizations that were not clustered are excluded.

    Parameters
    ----------
    locs : np.recarray
        Localizations used for clustering
    labels : np.array
        Array of cluster labels for each localization. -1 means no
        cluster assignment.

    Returns
    -------
    locs : np.recarray
        Localization list with "group" column appended, providing
        cluster label. 
    """
    
    # add cluster id to locs, as "group"
    locs = _lib.append_to_rec(locs, labels, "group")

    # -1 means no cluster assigned to a loc
    locs = locs[locs.group != -1]
    return locs


def error_sums_wtd(x, w):
    """ 
    Function used for finding localization precision for cluster 
    centers.

    Parameters
    ----------
    x : float
        x or y coordinate of the cluster center
    w : float
        weight (inverse localization precision squared)

    Returns
    -------
    float
        weighted localizaiton precision of the cluster center
    """

    return (w * (x - (w * x).sum() / w.sum())**2).sum() / w.sum()


def find_cluster_centers(locs, pixelsize=None):
    """
    Calculates cluster centers. 

    Uses pandas.groupby to quickly run across all cluster ids.

    Parameters
    ----------
    locs : np.recarray
        Clustered localizations (contain group info)
    pixelsize : int (default=None)
        Camera pixel size (used for finding volume and 3D convex hull).
        Only required for 3D localizations.

    Returns
    -------
    np.recarray
        Cluster centers saved as localizations
    """

    # group locs by their cluster id (group)
    locs_pd = _pd.DataFrame(locs)
    grouplocs = locs_pd.groupby(locs_pd.group)

    # get cluster centers
    res = grouplocs.apply(cluster_center, pixelsize)
    centers_ = res.values

    # convert to recarray and save
    frame = _np.array([_[0] for _ in centers_])
    std_frame = _np.array([_[1] for _ in centers_])
    x = _np.array([_[2] for _ in centers_])
    y = _np.array([_[3] for _ in centers_])
    std_x = _np.array([_[4] for _ in centers_])
    std_y = _np.array([_[5] for _ in centers_])
    photons = _np.array([_[6] for _ in centers_])
    sx = _np.array([_[7] for _ in centers_])
    sy = _np.array([_[8] for _ in centers_])
    bg = _np.array([_[9] for _ in centers_])
    lpx = _np.array([_[10] for _ in centers_])
    lpy = _np.array([_[11] for _ in centers_])
    ellipticity = _np.array([_[12] for _ in centers_])
    net_gradient = _np.array([_[13] for _ in centers_])
    n = _np.array([_[14] for _ in centers_])

    if hasattr(locs, "z"):
        z = _np.array([_[15] for _ in centers_])
        std_z = _np.array([_[16] for _ in centers_])
        volume = _np.array([_[17] for _ in centers_])
        convexhull = _np.array([_[18] for _ in centers_])
        centers = _np.rec.array(
            (
                frame,
                std_frame,
                x,
                y,
                std_x,
                std_y,
                z,
                photons,
                sx,
                sy,
                bg,
                lpx,
                lpy,
                std_z,
                ellipticity,
                net_gradient,
                n,
                volume,
                convexhull,
                res.index.values, # group id
            ),
            dtype=CLUSTER_CENTERS_DTYPE_3D,
        )
    else:
        area = _np.array([_[13] for _ in centers_])
        convexhull = _np.array([_[14] for _ in centers_])
        centers = _np.rec.array(
            (
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
                area,
                convexhull,
                res.index.values, # group id
            ),
            dtype=CLUSTER_CENTERS_DTYPE_2D,
        )

    if hasattr(locs, "group_input"):
        group_input = _np.array([_[-1] for _ in centers_])
        centers = _lib.append_to_rec(centers, group_input, "group_input")

    return centers


def cluster_center(grouplocs, pixelsize=None, separate_lp=False):
    """
    Finds cluster centers and their attributes.

    Assumes locs to be a pandas.SeriesGroupBy object, grouped by
    cluster ids.

    Paramaters
    ----------
    grouplocs : pandas.SeriesGroupBy
        Localizations grouped by cluster ids
    pixelsize : int (default=None)
        Camera pixel size (used for finding volume and 3D convex hull).
        Only required for 3D localizations.
    separate_lp : bool (default=False)
        If True, localization precision in x and y will be calculated
        separately. Otherwise, the mean of the two is taken
    
    Returns
    -------
    list
        Attributes used for saving the given cluster as .hdf5
        (frame, x, y, etc)
    """

    # mean and std frame
    frame = grouplocs.frame.mean()
    std_frame = grouplocs.frame.std()
    # average x and y, weighted by lpx, lpy
    x = _np.average(grouplocs.x, weights=1/(grouplocs.lpx)**2)
    y = _np.average(grouplocs.y, weights=1/(grouplocs.lpy)**2)
    std_x = grouplocs.x.std()
    std_y = grouplocs.y.std()
    # mean values
    photons = grouplocs.photons.mean()
    sx = grouplocs.sx.mean()
    sy = grouplocs.sy.mean()
    bg = grouplocs.bg.mean()
    # weighted mean loc precision
    lpx = _np.sqrt(
        error_sums_wtd(grouplocs.x, grouplocs.lpx)
        / (len(grouplocs) - 1)
    )
    lpy = _np.sqrt(
        error_sums_wtd(grouplocs.y, grouplocs.lpy)
        / (len(grouplocs) - 1)
    )
    if not separate_lp:
        lpx = (lpx + lpy) / 2
        lpy = lpx
    # other attributes
    ellipticity = sx / sy
    net_gradient = grouplocs.net_gradient.mean()
    # n_locs in cluster
    n = len(grouplocs)
    if hasattr(grouplocs, "z"):
        if pixelsize is None:
            raise ValueError(
                "Camera pixel size must be specified as an integer for 3D"
                " cluster centers calculation."
            )
        z = _np.average(
            grouplocs.z, 
            weights=1/((grouplocs.lpx+grouplocs.lpy)**2),
        ) # take lpz = 2 * mean(lpx, lpy)
        std_z = grouplocs.z.std()
        # lpz = std_z
        volume = _np.power(
            (std_x + std_y + std_z / pixelsize) / 3 * 2, 3
        ) * 4.18879
        try:
            X = _np.stack(
                (grouplocs.x, grouplocs.y, grouplocs.z / pixelsize),
                axis=0,
            ).T
            hull = _ConvexHull(X)
            convexhull = hull.volume
        except:
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
            z,
            std_z,
            # lpz,
            volume,
            convexhull,
        ]
    else:
        area = _np.power(std_x + std_y, 2) * _np.pi
        try:
            X = _np.stack((grouplocs.x, grouplocs.y), axis=0).T
            hull = _ConvexHull(X)
            convexhull = hull.volume
        except:
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
            area,
            convexhull,
        ]

    if hasattr(grouplocs, "group_input"):
        # assumes only one group input!
        result.append(_np.unique(grouplocs.group_input)[0]) 
    return result