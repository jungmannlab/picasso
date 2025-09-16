"""
    picasso.spinna
    ~~~~~~~~~~~~~~

    Single protein simulations in DNA-PAINT for recovery of
    stoichiometries of oligomerization states.

    :authors: Luciano A Masullo, Rafal Kowalewski, 2022-2025
    :copyright: Copyright (c) 2022-2025 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import time
from typing import Literal
from concurrent import futures
from multiprocessing import cpu_count
from itertools import product as it_prod
from copy import deepcopy
from numbers import Number

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from scipy.stats import ks_2samp
from tqdm import tqdm

from . import io, lib, render, __version__


LOCS_DTYPE_2D = [
    ("frame", "u4"),
    ("x", "f4"),
    ("y", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
]
LOCS_DTYPE_3D = [
    ("frame", "u4"),
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
]
NN_COLORS = ['#2880C4', '#97D8C4', '#F4B942', '#363636']
N_TASKS = 100
N_BOOTSTRAPS = 20
BOOTSTRAP_DISTANCE = 30.0
BOOTSTRAP_DISTANCE_METRIC = 1.0


def rref(M: np.ndarray) -> np.ndarray:
    """Convert a given matrix to its reduced row echelon form (RREF)
    using Gaussian elimination. Used for solving sets of linear
    equations.

    Based on the pseudocode from Wikipedia:
    https://en.wikipedia.org/wiki/Row_echelon_form

    Parameters
    ----------
    M : np.ndarray
        The matrix to be transformed.

    Returns
    -------
    M : np.ndarray
        The matrix in the reduced row echelon form.
    """
    M = M.copy()

    lead = 0  # leading column
    n_rows, n_cols = M.shape
    for r in range(n_rows):
        if n_cols <= lead:
            break
        i = r
        # find the pivot column that is non-zero
        while M[i, lead] == 0:
            i += 1
            if n_rows == i:
                i = r
                lead += 1
                if n_cols == lead:
                    break
        M[[r, i], :] = M[[i, r], :]  # swap rows i and r
        # transfor the pivot to 1
        if M[r, lead] != 0:
            M[r, :] = M[r, :] / M[r, lead]
        # eliminate all entries in the pivot column below the row
        for i in range(n_rows):
            if i != r:
                M[i, :] -= M[i, lead] * M[r, :]
        lead += 1

    return M


def find_target_counts(
    targets: list[str],
    structures: list[Structure],
) -> np.ndarray:
    """Find the number of each molecular target in structures.

    Parameters
    ----------
    targets : list of strs
        Names of the molecular targets to be counted.
    structures : list of Structure's
        Structures that are to be simulated.

    Returns
    -------
    t_counts : np.ndarray
        Array of shape (len(targets), len(structures)) specifying the
        number of each target in each structures.
    """
    n_t = len(targets)
    n_s = len(structures)
    t_counts = np.zeros((n_t, n_s), dtype=np.float32)
    for ii, structure in enumerate(structures):
        t_counts[:, ii] = structure.get_ind_target_count(targets)
    return t_counts


def get_structures_permutation(t_counts: np.ndarray) -> np.ndarray:
    """Find a permutation that ensures that the numbers of structures
    can be found using ``generate_N_structures``.

    Gaussian elimination provides the means to find the free variables
    in the system of linear equations that is used here. For more
    details, please see: Polyanin AD, Manzhirov AV. Handbook of
    Mathematics for Engineers and Scientists. Boca Raton, FL: CRC; 2007.

    Parameters
    ----------
    t_counts : np.ndarray
        Array specifying the counts of each molecular target in each
        structure, see ``generate_N_structures``. Shape (T, S), where
        T is the number of molecular targets and S is the number of
        structures.

    Returns
    -------
    perm : np.ndarray
        The permutation array, shape (S,).
    """
    n_t, n_s = t_counts.shape
    perm = np.arange(n_s)  # initiate the permutation array

    # run reduced row echelon form to check if the matrix is valid
    red = rref(t_counts)
    # if the order of columns is incorrect, the diagonal element is
    # non-zero, and this needs to be switched with one of the
    # free-parameter columns, i.e., we shift the pivot columns to the
    # left such that they correspond to the dependent parameters, see
    # documentation
    lpc = n_t  # last permutated column
    for i in range(n_t):
        if red[i, i] != 1:
            perm[i] = lpc
            perm[lpc] = i
            lpc += 1
    return perm


def generate_N_structures(
    structures: list[Structure],
    N_total: dict,
    granularity: int,
    save: str = '',
) -> dict:
    """Generate combinations of numbers of structures to be simulated
    in NN_scorer. In other words, generate the parameter search space
    for NN_scorer.

    Parameters
    ----------
    structures : list of Structure's
        Structures that are to be simulated.
    N_total : dict
        Keys give the names of the targets and the corresponding values
        give the total number of molecules to be simulated for each
        species. The numbers specify the total (not observed!) number of
        molecules to be simulated for each species, i.e., corrected for
        labeling efficiency: n_observed / LE.
    granularity : int
        Controls how many structure counts are
        generated overall. The higher granularity, the more combinations
        of numbers of structures are generated.
    save : str (default='')
        Path to save a .csv file with the number of structures
        generated. If '' is given, no file is saved.

    Returns
    -------
    structure_counts : dict
        Specifies what structure counts are to be simulated for each
        iteration. Keys are the names of the structures and values
        are lists of integers.
    """
    # extract the unique names of molecular targets in structures
    targets = []
    for structure in structures:
        for target in structure.targets:
            if target not in targets:
                targets.append(target)

    # number of molecular targets in each structure; each row gives one
    # target species and each column gives one structure
    n_t = len(targets)
    n_s = len(structures)
    if n_s <= n_t:
        raise ValueError(
            "To generate the search space, the number of unique molecular"
            " targets must be lower than the number of structures that are"
            " investigated. Otherwise, the numbers of structures to be"
            " simulated is constant."
        )
    t_counts = find_target_counts(targets, structures)

    # ensure that the order of structures is correct, i.e., the free
    # paramters in the system of linear equations are on the right side
    p = get_structures_permutation(t_counts.copy())
    t_counts = t_counts[:, p]
    structures = [structures[_] for _ in p]

    # convert N_total to a 1D array for simplicity, keep only the
    # targets that are present in the structures
    N_total = np.asarray([N_total[target] for target in targets])

    # extend t_counts to have each target's total count (from N_total)
    # on the right side; this matrix specifies the augmented matrix for
    # the set of linear equations that specify the number of structures
    eqs = np.hstack((t_counts, N_total.reshape(-1, 1)))

    # use Gaussian elimination to find the reduced row echelon form
    # that will give the expressions for the numbers of structures
    eqs = np.float32(rref(eqs))

    # separate structures into two groups - free and dependent; free
    # structures are those whose counts can be freely changed, i.e.,
    # the free parameters in the set of linear equations, while
    # dependent structures are those whose counts are dependent on the
    # free structure counts, see eqs above. The number of free
    # structures is equal to the number of structure species minus the
    # number of unique molecular targets
    t_free = t_counts[:, n_t:]

    # find the maximum values of the free structure counts that will be
    # examined; this upper bound is given by the total number of
    # targets of each species to be simulated divided by the number of
    # targets of these species in the structures; out of all
    # structures, the lowest value is taken. This is done to reduce the
    # number of combinations of structure counts that are examined in
    # the later stages
    max_vals = N_total.max() * np.ones_like(t_free)
    np.divide(N_total.reshape(-1, 1), t_free, out=max_vals, where=t_free != 0)
    max_vals = max_vals.min(axis=0).astype(np.int32)

    # now we find the free structure counts that will be examined; to
    # do this, we find granularity-many equally spaced numbers that lie
    # between 0 and max_vals for each free structure. These numbers are
    # combined into a 2D array, where each row gives one combination of
    # the free structure counts.

    # unique counts for the free structures
    bases = [np.linspace(0, m, granularity) for m in max_vals]

    # combined numbers of free structures
    free_structures = np.array(list(it_prod(*bases)))
    N_structures = np.hstack((
        np.zeros((free_structures.shape[0], n_t)),
        free_structures,
    ))

    # based on the numbers of free structures, we can calculate the
    # numbers of dependent structures using the set of linear equations
    # that were found above, see eqs.
    # by taking the last row of eqs, we can find the counts of
    # the first of the dependent structures, see documentation. Once
    # this is done, the next dependent structure can be found by
    # repeating the process with the second to last row of eqs, and so
    # on.
    for i in range(n_t):  # iterate over each dependent structure
        # take the coefficients from eqs; we start from the last row
        # and take only the coefficients that are to the right side of
        # the leading one
        formula = eqs[n_t-i-1][(n_t-i):]
        # the last element in the formula is the constant coefficient;
        # to get the value of the dependent structure count, subtract
        # the term taken from the dot product of the coefficients from the
        # the constant coefficient (all taken from formula and the
        # structure counts)
        N_structures[:, n_t-i-1] = (
            formula[-1] - (N_structures[:, (n_t-i):] @ formula[:-1])
        )

    # the last step is to delete the unphysical values, i.e., the
    # rows where the structure counts are negative; rows with repeating
    # counts are also filtered out
    mask = np.any(N_structures < 0, axis=1)
    N_structures = N_structures[~mask].astype(np.int32)

    # convert the resulting combinations of numbers of structures to a
    # dictionary where keys specify the names of the structures and
    # values specify the numbers of structures to be simulated
    structure_counts = {}
    for i, structure in enumerate(structures):
        structure_counts[structure.title] = N_structures[:, i]

    if save:  # if path for saving was provided
        # find proportions first, just like in
        # StructureMixer.convert_counts_to_props
        props = np.zeros(N_structures.shape, dtype=np.float32)
        for i, structure in enumerate(structures):
            N_str_total = np.zeros(N_structures.shape[0], dtype=np.float32)
            N_per_target = structure.get_ind_target_count(targets)
            for N_mol in N_per_target:
                N_str_total = N_str_total + N_mol * N_structures[:, i]
            prop = np.round(100 * N_str_total / N_total, 2)
            props[:, i] = prop
        # if rounding error occurs, delete from the first non-zero element
        rows_to_correct = np.where(np.sum(props, axis=1) != 100)[0]
        for row in rows_to_correct:
            first_non_zero_idx = next(
                i for i, prop in enumerate(props[row, :]) if prop > 0
            )
            props[row, first_non_zero_idx] -= np.sum(props[row, :]) - 100

        # save as a .csv file
        df = pd.DataFrame(
            np.hstack((N_structures, props)),
            columns=[
                f"N_{_.title}" for _ in structures
            ]+[
                f"Prop_{_.title}" for _ in structures
            ],
        )
        df.to_csv(save, header=True, index=False)

    return structure_counts


def otsu(image: np.ndarray) -> float:
    """Apply Otsu's thresholding algorithm to the input image to
    segment it into a binary image.

    Taken from scikit-image and reduced to the specific case used in
    SPINNA.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    thresh : float
        Otsu's threshold value.
    """
    # histogram the image and converts bin edges to bin centers
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype('float32', copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    thresh = bin_centers[idx]
    return thresh


def random_rotation_matrices(
    num: int,
    mode: Literal["3D", "2D"] | None = "2D",
) -> np.ndarray:
    """Generate num-many random rotation matrices. By default, 2D
    rotations are generated, although 3D rotations around the z axis
    are supported too.

    Parameters
    ----------
    num : int
        Number of rotations to be generated.
    mode : {'3D', '2D', None}
        Mode of rotation. If '3D', random rotations in 3D are
        generated; if '2D' random rotations around the z axis are
        generated; if None, no rotations are provided (i.e., identity
        matrices).

    Returns
    -------
    rots : np.ndarray
        Array of shape (num, 3, 3) specifying num-many random rotation
        matrices.
    """
    if not isinstance(num, int) or num <= 0:
        raise TypeError("Number of rotations must be a positive integer.")

    if mode == '3D':
        rots = Rotation.random(num=num).as_matrix().astype(np.float32)
    elif mode == '2D':
        # rotate only around z
        angles = np.random.uniform(0, 2*np.pi, size=(num,))
        rots = Rotation.from_euler('z', angles).as_matrix().astype(np.float32)
    elif mode is None:
        rots = Rotation.identity(num=num).as_matrix().astype(np.float32)
    else:
        raise ValueError("Argument mode must be one of {'3D', '2D', None}.")
    return rots


def coords_to_locs(
    coords: np.ndarray,
    lp: float = 1.,
    pixelsize: int = 130,
) -> np.recarray:
    """Convert ``coords`` array into localization list that can be read
    in Picasso Render.

    Parameters
    ----------
    coords: np.ndarray
        Coordinates of localizations to be converted. All coordinates
        are in nm. Shape (N, 2) or (N, 3), where N is the number of
        localizations.
    lp : float (default=1)
        Localization precision to be added to the localizations.
    pixelsize : int (default=130)
        Camera pixelsize in nm. Lateral coordinates are expressed in
        units of camera pixelsize but axial coordinates are
        expressed in nm (picasso).

    Returns
    -------
    locs : np.rec.array
        Localization list compatible with Picasso Render.
    """
    # x, y and localization precision in Picasso are in camera pixels
    x = coords[:, 0] / pixelsize
    y = coords[:, 1] / pixelsize
    lpx = lp * np.ones(len(x)) / pixelsize
    lpy = lpx
    # dummy value to avoid errors in Picasso Render
    frame = np.ones(len(x))
    if coords.shape[1] == 3:
        z = coords[:, 2]
        locs = np.rec.array(
            (frame, x, y, z, lpx, lpy),
            dtype=LOCS_DTYPE_3D,
        )
    else:
        locs = np.rec.array(
            (frame, x, y, lpx, lpy),
            dtype=LOCS_DTYPE_2D,
        )
    return locs


def plot_NN(
    data1: np.ndarray | None = None,
    data2: np.ndarray | None = None,
    n_neighbors: int = 1,
    dist: np.ndarray | None = None,
    mode: Literal['hist', 'plot'] = 'hist',
    fig: plt.Figure | None = None,
    ax: plt.axes | None = None,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 200,
    binsize: float = 4.0,
    xlim: tuple[float, float] | None = (0, 200),
    ylim: tuple[float, float] | None = None,
    colors: list = NN_COLORS,
    title: str = "Nearest neighbor distances",
    xlabel: str = "Distances (nm)",
    ylabel: str = "Norm. frequency",
    show_legend: bool = True,
    alpha: float = 0.6,
    edgecolor: str = 'black',
    show: bool = False,
    return_fig: bool = False,
    savefig: str = '',
) -> tuple[plt.Figure, plt.axes] | None:
    """Plot a nearest neighbor distances histogram.

    Parameters
    ----------
    data1, data2 : np.ndarrays
        Coordinates of two datasets to be compared and whose NND
        (nearest neighbor distribution) is plotted. If None, dist must
        be provided.
    dist : np.array
        Contains the NN distances (obtained with get_NN_dist). If None,
        the distances are calculated from data1 and data2. Otherwise,
        the NND calculation is skipped.
    mode : {'hist', 'plot'} (default='hist')
        Mode of plotting. If 'hist', histogram is plotted. If 'plot'
        NNDs are histogramed and a line is plotted.
    fig, ax : plt.Figure, plt.Axes (default=None,None)
        Figure and Axes to be used for plotting. If None, new figure
        and axes are created.
    figsize : tuple of ints (default=(6,6))
        Figure size, used when new fig and ax are created.
    binsize : float (default=2.5)
        Binsize used for histograming NNDs.
    colors : list
        List specifying the colors of the histogram bins or plotted
        lines. If the number of neighbors is larger than the number of
        colors, the colors are repeated. Each element must be specified
        as in:
        https://matplotlib.org/stable/tutorials/colors/colors.html.
    title, xlabel, ylabel : strs
        Title and label of x and y axes, respectively.
    xlim, ylim : tuples of floats (default=None, None)
        Limits in which x and y axes are plotted. If None, the
        automatic limits are used.
    alpha : float (default=0.6)
        Alpha (transparency) of histogram bins (not applied to
        lineplot).
    edgecolor : str (default='black')
        Histogram bin edgecolor (not applied to lineplot).
    show : bool (default=True)
        If True, the plot is shown using plt.show()
    show_legend : bool (default=True)
        If True, legend is shown.
    return_fig : bool (default=False)
        If True, fig and ax are returned and can be used for further
        processing.
    savefig : str (default='')
        Path to save the plot. If '', the plot is not saved.
    """
    # initiate figure and axis
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            1, figsize=figsize, constrained_layout=True, dpi=dpi
        )

    # calculate NNDs if not provided directly
    if dist is None:
        if data1 is None or data2 is None:
            raise ValueError(
                "If no NN distribution is given, please provide spatial"
                " coordinates to calculate the NNDs."
            )
        else:
            dist = get_NN_dist(data1, data2, n_neighbors)
    else:
        n_neighbors = dist.shape[1] if dist.ndim == 2 else 0

    if n_neighbors > len(colors):
        colors = colors * (n_neighbors // len(colors) + 1)

    # plot histogram / line
    for i in range(n_neighbors):
        data = dist[:, i]
        if mode == 'hist':
            ax.hist(
                data,
                bins=np.arange(xlim[0], xlim[1]+binsize, binsize),
                edgecolor=edgecolor,
                color=colors[i],
                label=f"exp {i+1}th NN",
                alpha=alpha,
                linewidth=0.4,  # 0.1
                density=True,
            )
        elif mode == 'plot':
            counts, bin_edges = np.histogram(
                data,
                bins=np.arange(xlim[0], xlim[1]+binsize, binsize),
                density=True,
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.plot(
                bin_centers,
                counts,
                color=colors[i],
                linewidth=0.9,  # 2
                alpha=alpha,
                label=f"sim {i+1}th NN",
            )

    # display parameters
    if show_legend:
        ax.legend()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # save figure(s)
    if savefig:
        if isinstance(savefig, str):
            plt.savefig(savefig)
        elif isinstance(savefig, list) or isinstance(savefig, tuple):
            for name in savefig:
                plt.savefig(name)

    # display and/or return figure
    if show:
        plt.show()
    if return_fig:
        return fig, ax


def get_NN_dist(
    data1: np.ndarray,
    data2: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Find nearest neighbors distances between data1 and data2 for
    n_neighbors closest neighbors.

    Parameters
    ----------
    data1 : np.ndarray
        Array of points from which distances are measured. Should have
        shape (N, 2) or (N, 3) for 2D/3D case, respectively, where N
        is the number of points.
    data2 : np.ndarray
        Array of points to which distances are measured. May contain a
        different number of points but of the same dimensionality.
    n_neighbors : int
        Number of neighbors to consider.

    Returns
    -------
    dist : np.ndarray
        Array with distances of N-th neighbors for each point in data1.
        Shape: (N, n_neighbors)
    """
    # if empty list passed, return empty array
    if len(data1) == 0 or len(data2) == 0:
        return np.array([])

    # check that data1 and data2 have the same dimensionalities
    if data1.shape[1] != data2.shape[1]:
        raise ValueError(
            "data1 and data2 must have the same number of dimensions."
        )

    # avoid self-counting if data1 and data2 are the same
    reduce = 1 if np.array_equal(data1, data2) else 0

    # find distances
    tree = KDTree(data2)
    dist, _ = tree.query(data1, k=n_neighbors+reduce)

    # adjust the shape of the output if needed
    if n_neighbors + reduce == 1:
        dist = np.expand_dims(dist, 1)

    # avoid self-counting if data1 and data2 are the same
    if reduce:
        dist = dist[:, 1:]

    return dist


def get_NN_dist_experimental(
    coords: dict,
    mixer: StructureMixer,
    duplicate: bool = False,
) -> list[np.ndarray]:
    """Calculate nearest neighbor distances for experimental data.

    Parameters
    ----------
    coords : dict
        Dictionary with the names of the molecular targets as keys and
        the coordinates of the localizations as values.
    mixer : StructureMixer
        Instance of the structure mixer used in simulation(s).
    duplicate : bool (default=False)
        If True, the NN distances are found for each pair of molecular
        targets in both directions, e.g., CD80 -> CD86 as well as
        CD86 -> CD80. If False, only one direction is considered.

    Returns
    -------
    dists : list of np.2darrays
        Lists of arrays of shape (N, n_neighbors) where N is the
        number of distances measured and n_neighbors is the number of
        neighbors considered. The list has the same length as
        mixer.get_neighbor_idx(), i.e. the number of molecular target
        species pairs that are considered. If duplicate is True, the
        length of the list increases.
    """
    neighbor_idx = mixer.get_neighbor_idx(duplicate=duplicate)
    dists = [[] for (_, _, nn_count) in neighbor_idx if nn_count]
    current_idx = 0
    for t1, t2, n in neighbor_idx:
        if n:
            dist = get_NN_dist(coords[t1], coords[t2], n)
            dists[current_idx].append(dist)
            current_idx += 1
    # for consistency with get_NN_dist_simulated:
    dists = [np.concatenate(_) if _ else [] for _ in dists]
    return dists


def get_NN_dist_simulated(
    N_str: list[np.ndarray],
    N_sim: int,
    mixer: StructureMixer,
    duplicate: bool = False,
) -> list[np.ndarray]:
    """Calculate nearest neighbor distances across many simulations
    with the same settings. Simulations are repeated ``N_sim`` times and
    the NN distances are calculated for each simulation.

    Parameters
    ----------
    N_str : list of np.ndarrays
        Numbers of structures to be simulated.
    N_sim : int
        Number of times the simulation is repeated.
    mixer : StructureMixer
        Instance of the structure mixer used in simulation(s).
    duplicate : bool (default=False)
        If True, the NN distances are found for each pair of molecular
        targets in both directions, e.g., CD80 -> CD86 as well as
        CD86 -> CD80. If False, only one direction is considered.

    Returns
    -------
    dists : list of np.2darrays
        Lists of arrays of shape (N, n_neighbors) where N is the
        number of distances measured and n_neighbors is the number of
        neighbors considered. The list has the same length as
        mixer.get_neighbor_idx(), i.e. the number of molecular target
        species pairs that are considered. If duplicate is True, the
        length of the list increases.
    """
    neighbor_idx = mixer.get_neighbor_idx(duplicate=duplicate)

    # empty lists for each of neighbors indeces
    dists = [[] for (_, _, nn_count) in neighbor_idx if nn_count]
    # run simulations N_sim times
    for _ in range(N_sim):
        coords = mixer.run_simulation(N_str)
        # find NN distances for each pair of molecular targets
        current_idx = 0
        for t1, t2, n in neighbor_idx:
            if n:
                dist = get_NN_dist(coords[t1], coords[t2], n)
                dists[current_idx].append(dist)
                current_idx += 1

    # combine the results from all simulations into one array
    dists = [np.concatenate(_) if _ else [] for _ in dists]
    return dists


def NND_score(dists1: list[np.ndarray], dists2: list[np.ndarray]) -> float:
    """Score the two datasets of nearest neighbor distances (NND)
    using the Kolmogorov-Smirnov test.

    Parameters
    ----------
    dists1, dists2: list of np.ndarray
        Lists of arrays of shape (N, n_neighbors) where N is the
        number of distances measured and n_neighbors is the number
        of neighbors considered. See get_NN_dist_simulated and
        get_NN_dist_experimental, which return such dists arrays.

    Returns
    -------
    score : float
        Sum of KS test statistics. Ranges between 0 (perfect fit)
        and 1 (worst fit).
    """
    scores = []
    norm = 0
    for d1, d2 in zip(dists1, dists2):  # iterate over each pair of molecules
        for n in range(d1.shape[1]):
            scores.append(ks_2samp(d1[:, n], d2[:, n]).statistic)
            norm += 1
    score = np.sum(scores) / norm
    return score


def load_structures(path: str) -> tuple[list[Structure], list[str]]:
    """Load structures (``Structure``'s) saved in a .yaml file.

    Parameters
    ----------
    path : str
        Path to the .yaml file with structures.

    Returns
    -------
    structures : list of Structure's
        List of structures loaded from the file.
    targets : list of strs
        List of all unique molecular targets in the structures.
    """
    with open(path, 'r') as file:
        try:
            info = list(yaml.load_all(file, Loader=yaml.FullLoader))
        except TypeError:
            raise TypeError(
                "Incorrect file. Please choose a file that was created"
                " that was created with Picasso SPINNA."
            )
        if "Structure title" not in info[0]:
            raise TypeError(
                "Incorrect file. Please choose a file that was created"
                " that was created with Picasso SPINNA."
            )
        # continue if the correct file is loaded
        structures = []
        targets = []
        for m_info in info:
            structure = Structure(m_info["Structure title"])
            for target in m_info["Molecular targets"]:
                x = m_info[f"{target}_x"]
                y = m_info[f"{target}_y"]
                z = m_info[f"{target}_z"]
                structure.define_coordinates(target, x, y, z)
                if target not in targets:
                    targets.append(target)
            structures.append(structure)
    return structures, targets


class MaskGenerator():
    """Interface for mask generation based on a Picasso .hdf5 file.

    To generate a mask, run the following command with arguments of
    choice:

    ``mask =  MaskGenerator(*args1).generate_mask(*args2).mask``

    ...

    Attributes
    ----------
    binsize : float
        Binsize used for histograming localizations (nm).
    locs : np.recarray
        Localizations list used for creating the mask (Picasso format).
    locs_path : str
        Path to .hdf5 with locs used for masking.
    mask : np.array
        Mask giving probability mass function of finding a structure in
        each pixel/voxel.
    ndim : int (default=None)
        Dimensionality of the mask. If None, it is deduced from
        localizations (if z given, 3D mask is created). Otherwise,
        it can be specified that 3D localizations are used for
        generating a 2D mask.
    pixelsize : float (default=130)
        Camera pixel size (nm).
    roi : float
        ROI width/height (nm). Calculated from localizations' metadata.
    sigma : float
        Sigma used in gaussian filtering in nm
    thresh : float
        Otsu threshold (see, scikit-image) used for masking and
        area/volume calculation.
    x_min : float
        Starting value of localizations' x coordinate in camera pixels.
        (default=0.0)
    x_max : float
        Highest value of localizations' x coordinate in camera pixels.
    y_min : float
        Starting value of localizations' y coordinate in camera pixels .
        (default=0.0)
    y_max : float
        Highest value of localizations' y coordinate in camera pixels.
    z_min : float
        Starting value of localizations' z coordinate in nm.
    z_max : float
        Highest value of localizations' z coordinate in nm.

    Parameters
    ----------
    locs_path : str
        Path to the .hdf5 file with localizations/molecules.
    binsize : int, optional
        Binsize used for histograming localizations (nm). Default is
        130.
    sigma : int, optional
        Sigma used for gaussian filtering (nm). Default is 65.
    ndim : int, optional
        Dimensionality of the mask (2 or 3). If None, the dimensionality
        is taken from the loaded localizations/molecules. Default is
        None.
    run_checks : bool, optional
        Whether to run input checks during initialization. Default is
        False.
    """

    def __init__(
        self,
        locs_path: str,
        binsize: int = 130,
        sigma: int = 65,
        ndim: int | None = None,
        run_checks: bool = False,
    ) -> None:
        self.locs_path = locs_path
        self.binsize = binsize
        self.sigma = sigma
        self.ndim = ndim

        self.mask = None
        self.thresh = None

        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None
        self.z_min = None
        self.z_max = None

        # open localizations
        locs, info = io.load_locs(self.locs_path)
        if hasattr(locs, "z"):
            self.locs = locs[['x', 'y', 'z']]
        else:
            self.locs = locs[['x', 'y']]

        if run_checks:
            self.ensure_correct_inputs()

        # camera pixelsize (like in Picasso Render.View.add())
        for element in info:
            if (
                "Picasso" in element.values() and
                "Localize" in element.values()
            ):
                if "Pixelsize" in element:
                    self.pixelsize = element["Pixelsize"]
                    break
        else:
            self.pixelsize = 130

        # deduce roi from localizations metadata
        self.roi = [
            info[0]["Width"] * self.pixelsize,
            info[0]["Height"] * self.pixelsize,
        ]

    def ensure_correct_inputs(self) -> None:
        """Ensure correct data types used in initialization."""
        # path
        if not isinstance(self.locs_path, str):
            raise TypeError(
                "Path to localizations must be a string ending with .hdf5."
            )
        elif not self.locs_path.endswith(".hdf5"):
            raise TypeError(
                "Path to localizations must be a string ending with .hdf5."
            )
        # binsize
        if not (
            isinstance(self.binsize, float) or isinstance(self.binsize, int)
        ):
            raise TypeError("Binsize must be a number.")
        # sigma
        if not (
            isinstance(self.sigma, float) or isinstance(self.sigma, int)
        ):
            raise TypeError("Sigma must be a number.")
        # ndim
        if self.ndim is not None:
            if hasattr(self.locs, "z") and self.ndim not in [2, 3]:
                raise ValueError(
                    "Dimensionality of the mask must be either 2 or 3 for 3D"
                    " localizations."
                )
            elif not hasattr(self.locs, "z") and self.ndim != 2:
                raise ValueError(
                    "Dimensionality of the mask must be 2 for 2D"
                    " localizations."
                )

    def render_locs(self) -> np.ndarray:
        """Render localizations histogram (2D or 3D), no blur.

        Uses ``picasso.render`` after preparing inputs."""
        # prepare inputs for picasso.render
        oversampling = self.pixelsize / self.binsize
        self.x_min = 0
        self.x_max = self.roi[0] / self.pixelsize
        self.y_min = 0
        self.y_max = self.roi[1] / self.pixelsize

        # 2D image
        if self.ndim == 2 or not hasattr(self.locs, "z"):
            _, image = render.render_hist(
                self.locs, oversampling,
                self.y_min, self.x_min, self.y_max, self.x_max,
            )
        # 3D image
        else:
            self.z_min = self.locs.z.min()
            self.z_max = self.locs.z.max()
            _, image = render.render_hist3d(
                self.locs, oversampling,
                self.y_min, self.x_min, self.y_max, self.x_max,
                self.z_min, self.z_max,
                self.pixelsize,
            )
        return image

    def generate_mask(
        self,
        apply_thresh: bool = False,
        mode: Literal["loc_den", "binary"] = "loc_den",
        verbose: bool = False,
    ) -> MaskGenerator:
        """Generate a mask (available after class initialization). The
        mask provides the probability mass function for finding a
        molecule in each mask pixel/voxel.

        Parameters
        ----------
        apply_thresh : bool (default=False)
            Whether or not apply Otsu thresholding to the density map
            mask. Does not apply to binary mask.
        mode : {'loc_den', 'binary'}
            If 'loc_den', mask giving probability mass function is
            created. If 'binary', a binary mask is created (i.e., each
            pixel/voxel specifies if a molecule can be found at the
            given region or not)

        Returns
        -------
        self : MaskGenerator
        """
        if verbose:
            print(f"Generating a mask in {self.ndim}D.")
            print("Rendering localizations... (1/3)")
        image = self.render_locs()
        if verbose:
            print("Applying gaussian filter... (2/3)")
        if self.sigma > 0:
            sigma = self.sigma / self.binsize
            image = gaussian_filter(image, sigma=sigma, mode='constant')
        if verbose:
            print("Thresholding... (3/3)")
        image = np.float64(image / image.sum())
        self.thresh = otsu(image)

        if mode == "loc_den":
            if apply_thresh:
                image[image < self.thresh] = 0
            self.mask = image
        elif mode == "binary":
            self.mask = np.zeros_like(image, dtype=np.float64)
            self.mask[image > self.thresh] = 1
            self.mask = self.mask / self.mask.sum()
        else:
            raise ValueError("mode must be either 'loc_den' or 'binary'.")
        return self

    def save_mask(self, path: str, save_png: bool = False) -> None:
        """Save the result in self.mask and/or in .npy/.png files.

        If .npy is saved, it is accompanied by a metadata .yaml file
        used for reading the mask in StructureSimulator.

        save_png : bool (default=False)
            Whether or not save the mask as .png (3D mask will be
            summed along z axis).
        """
        if self.mask is None:
            return

        if not path.endswith(".npy"):
            raise ValueError("Path for saving mask must end with .npy")

        np.save(path, self.mask)
        self.save_mask_info(path)

        if save_png:
            from PIL import Image
            outpath = path.replace(".npy", ".png")
            if self.mask.ndim == 3:
                mask_ = np.sum(self.mask, axis=2)
                mask_ /= mask_.max()  # normalize to save image
            else:
                mask_ = self.mask.copy()
                mask_ /= mask_.max()  # normalize to save image
            im = Image.fromarray(np.uint8(mask_*255))
            im.save(outpath)

    def save_mask_info(self, path: str) -> None:
        """Save info about the mask in .yaml format.

        Parameters
        ----------
        path : str
            Path to save the mask.
        """
        # basic info (both 2D and 3D)
        info = {
            "Generated by": f"Picasso v{__version__} SPINNA",
            "Size (GB)": self.mask.nbytes / (1024**3),
            "File": path,
            "Binsize (nm)": self.binsize,
            "Generated from": self.locs_path,
            "Gaussian blur (nm)": self.sigma,
            "Camera pixelsize (nm)": self.pixelsize,
            "x_min":  self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "Number of molecules": len(self.locs),
            "Dimensionality": f"{self.mask.ndim}D",
            "Threshold value": float(self.thresh),
        }
        if self.mask.ndim == 3:  # 3D mask
            info["z_min"] = float(self.z_min)
            info["z_max"] = float(self.z_max)
            info["Volume (um^3)"] = float(
                1e-9 * self.binsize ** 3 * (self.mask > self.thresh).sum()
            )
        else:  # 2D mask
            info["Area (um^2)"] = float(
                1e-6 * self.binsize ** 2 * (self.mask > self.thresh).sum()
            )

        # save
        outpath = path.replace(".npy", ".yaml")
        with open(outpath, 'w') as file:
            yaml.dump(info, file)


class Structure():
    """Specify a structure (hetero/homomultimer).

    ...

    Attributes
    ----------
    title : str
        The name of the structure, must be defined at initialization.
    targets : list
        Names of molecular targets in the structure.
    x : dict
        x coordinates of each molecular target's molecules.
    y : dict
        y coordinates of each molecular target's molecules.
    z : dict
        z coordinates of each molecular target's molecules.

    Parameters
    ----------
    Same as attributes.
    """

    def __init__(self, title: str) -> None:
        self.title = title
        self.targets = []
        self.x = {}
        self.y = {}
        self.z = {}

    def __repr__(self) -> str:
        display = [(
            f"Type: Structure, Title: {self.title}\n"
            f"Coordinates below: x, y, z (nm)\n"
        )]
        for target in self.x.keys():
            display.append(f"{target}:")
            for x, y, z in zip(
                self.x[target], self.y[target], self.z[target]
            ):
                display.append(f"{x}, {y}, {z}")

        return "\n".join(display) + "\n"

    def define_coordinates(
        self,
        target: str,
        x: list[float],
        y: list[float],
        z: list[float] | None = None,
    ) -> Structure:
        """Manually define coordinates of a given molecular target.

        Parameters
        ----------
        target : str
            Name of the molecular target.
        x : list of floats
            x coordinates of the molecular targets in nm.
        y : list of floats
            y coordinates of the molecular targets in nm.
        z : list of floats (default=None)
            z coordinates of the molecular targets in nm. If None, 2D
            structure is simulated.

        Returns
        -------
        self: Structure
        """
        if z is not None:  # 3D
            # assert equal lengths
            if not ((len(x) == len(y)) and (len(x) == len(z))):
                raise ValueError(
                    "x, y and z coordinates must have equal length."
                )
        else:  # 2D
            if not (len(x) == len(y)):  # assert equal lengths
                raise ValueError(
                    "x and y coordinates must have equal length."
                )
            z = [0] * len(x)

        # add name to self.targets if not present already
        if target not in self.targets:
            self.targets.append(target)
            self.x[target] = x
            self.y[target] = y
            self.z[target] = z
        else:  # append to the target info
            self.x[target].extend(x)
            self.y[target].extend(y)
            self.z[target].extend(z)
        return self

    def delete_target(self, target: str) -> None:
        """Delete a molecular target's information.

        Parameters
        ----------
        target : str
            Name of the molecular target to be deleted.
        """
        if target in self.targets:
            self.targets.remove(target)
            del self.x[target]
            del self.y[target]
            del self.z[target]

    def get_all_targets_count(self) -> int:
        """Find the number of all molecular targets in the structure.

        Returns
        -------
        n : int
            Number of all molecular targets in the structure.
        """
        n = sum([len(coords) for coords in self.x.values()])
        return n

    def get_ind_target_count(self, targets: str) -> list[int]:
        """Find the number of each molecular target in the structure.

        Parameters
        ----------
        targets : list of strs
            Names of molecular targets to be counted, If the target is
            not present, 0 is returned.

        Returns
        -------
        n : list
            Number of targets of each species in the structure. The
            same order as the input argument targets.
        """
        n = []
        for t in targets:
            if t in self.targets:
                n.append(len(self.x[t]))
            else:
                n.append(0)
        return n

    def get_max_nn(self, target1: str, target2: str) -> int:
        """Find the maximum number of nearest neighbors between two
        molecular targets in the structure.

        Parameters
        ----------
        target1, target2 : str
            Names of the molecular targets.

        Returns
        -------
        n : int
            Maximum number of nearest neighbors between the two targets.
        """
        if target1 not in self.targets or target2 not in self.targets:
            return 0
        elif target1 == target2:
            return max(len(self.x[target1]) - 1, 0)
        else:
            n1 = len(self.x[target1])
            n2 = len(self.x[target2])
            return min(n1, n2)

    def restart(self) -> Structure:
        """Delete all molecular targets, reset the structure but keep
        its title."""
        self.targets = []
        self.x = {}
        self.y = {}
        self.z = {}
        return self


class StructureSimulator():
    """Simulate positions of one structure using CSR, taking into
    account labeling efficiency and label uncertainty for each
    molecular target as well as the number of structures to be
    simulated. Rotates each structure randomly as a rigid body in 2D or
    3D.

    Mask can be applied to structure centers after generating it with
    MaskGenerator. This provides mask metadata required for the
    simulation. Masking consists of drawing the given number of
    structures from a multinomial distribution (given by the mask) to
    distribute centers across mask pixels/voxels. The mask can be a
    binary mask or a local density map. Next, the structure centers are
    CSR distributed within the mask pixels/voxels.

    2D and 3D simulations and masking are available.

    ...

    Attributes
    ----------
    c_pos : np.array
        Positions of centers of structures (nm).
    N : int
        Number of structures to be simulated. Note: this is usually
        not equal to the number of molecular targets simulated.
    depth : float (default=None)
        Depth of the simulated ROI (nm). If None and width and height
        are provided, 2D ROI is simulated. If specified and no mask is
        given, 3D ROI is simulated. If mask is provided, ROI is
        overwritten from mask.
    height : float (default=None)
        Height of the simulated ROI (nm). If None, mask must be
        provided. If mask is provided, ROI is overwritten from mask.
    label_unc : float or list of floats
        Label uncertainty of each molecular target (nm). Must follow
        the order specified in self.structures.targets. Lies in the range
        (0, inf).
    le : float or list of floats
        labeling efficiency of each molecular target simulated. Must
        follow the order specified in self.structures.targets. Lies in the
        range [0, 1].
    mask : np.ndarray
        Array specifying expected number of structures to be simulated
        in each mask pixel/voxel. If None, width, height and optionally
        depth must be provided to generate a rectangular ROI.
    mask_info : dict
        Mask metadata, containing info about size of masked ROI, etc.
        Must come from the .yaml metadata file created by
        MaskGenerator.
    N : int
        Number of structures to be simulated.
    structures : Structure
        Instance of the Structure class, determining molecular
        targets' names and their positions for a single structure.
    pos : dict
        Simulated positions of all molecular target species, offset by
        structure centers and label uncertainty (before labeling
        efficiency).
    pos_obs : dict
        Observed positions of all molecular targets' species (after
        labeling efficiency).
    random_rot_mode : {"3D", "2D", None} (default="2D")
        Mode of random rotation of structures. If "3D", structures are
        rotated randomly in 3D. If "2D", structures are rotated
        randomly in 2D about z axis. If None, structures are not
        rotated.
    width : float (default=None)
        Width of the simulated ROI (nm). If None, mask must be
        provided. If mask is provided, ROI is overwritten from mask.
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Lowest/highets value of structures' x/y/z coordinate in nm.
        z_min and z_max are specified for 3D simulation only.

    Parameters
    ----------
    structure : Structure
        Instance of the ``Structure`` class, determining molecular
        targets' names and their positions for a single structure.
    N_structures : int
        Number of structures to be simulated.
    le : float or list of floats
        labeling efficiency of each molecular target (nm). Must
        follow the order specified in self.structures.targets. Lies in
        the range [0, 1].
    label_unc : float or list of floats
        Label uncertainty of each molecular target (nm). Must follow the
        order specified in self.structures.targets. Lies in the range
        (0, inf).
    mask : np.ndarray or None, optional
        Mask to specify the region of interest (ROI) for the simulation.
        Default is None.
    mask_info : dict or None, optional
        Information about the mask, such as its size and position, see
        class Attributes. Default is None.
    width, height, depth : float or None, optional
        Dimensions of the region of interest (ROI) for the simulation in
        nm. Depth is required only for 3D data. If ``mask`` is not
        specified, ROI must be provided. Default is None.
    random_rot_data : {'3D', '2D'} or None, optional
        Random rotation mode for the simulation. If None, no rotations
        are applied. Default is '2D'.
    """

    def __init__(
        self,
        structure: Structure,
        N_structures: int,
        le: float | list[float],
        label_unc: float | list[float],
        mask: np.ndarray | None = None,
        mask_info: dict | None = None,
        width: float | None = None,
        height: float | None = None,
        depth: float | None = None,
        random_rot_mode: Literal["2D", "3D"] | None = "2D",
    ) -> None:
        self.structure = structure
        self.le = le
        self.N = N_structures
        self.label_unc = label_unc
        self.random_rot_mode = random_rot_mode

        self.c_pos = None
        self.pos = {}
        self.pos_obs = {}
        self.read_mask_and_ROI(mask, mask_info, width, height, depth)

    def read_mask_and_ROI(
        self,
        mask: np.ndarray | None = None,
        mask_info: dict | None = None,
        width: float | None = None,
        height: float | None = None,
        depth: float | None = None,
    ) -> None:
        """Read mask and/or ROI.

        By default, one of the two must be specified. If both are
        given, mask overwrites the ROI."""
        # ROI: width, height and depth #
        if mask is None:
            self.mask = None
            self.mask_info = None

            self.width = width
            self.height = height
            self.depth = depth

            self.x_min = 0
            self.x_max = width
            self.y_min = 0
            self.y_max = height

            if depth is not None:
                self.z_min = - depth / 2
                self.z_max = depth / 2
            else:
                self.z_min = None
                self.z_max = None

            return

        # Mask is given #
        elif mask is not None and mask_info is not None:
            # Load the ROI and mask attributes
            pixelsize = mask_info["Camera pixelsize (nm)"]
            self.x_min = mask_info["x_min"] * pixelsize
            self.x_max = mask_info["x_max"] * pixelsize
            self.y_min = mask_info["y_min"] * pixelsize
            self.y_max = mask_info["y_max"] * pixelsize

            self.width = self.x_max - self.x_min  # width in nm
            self.height = self.y_max - self.y_min  # height in nm

            if mask.ndim == 3:
                # z coordiantes are given in nm already (picasso)
                self.z_min = mask_info["z_min"]
                self.z_max = mask_info["z_max"]
                self.depth = (self.z_max - self.z_min)  # depth in nm
            else:
                self.z_min = None
                self.z_max = None
                self.depth = None

            # convert the mask to probabilities + set attributes
            mask = mask.astype(np.float64)
            self.mask = mask / mask.sum()
            self.mask_info = mask_info

        elif mask is not None and not mask_info:
            raise ValueError(
                "If mask is given, mask_info must be given as well."
            )

        else:
            raise ValueError(
                "Please provide information for mask or ROI."
            )

    def simulate_centers(self) -> None:
        """Simulate positions of centers of structures in 2D or 3D with
        or without masking."""
        self.c_pos = None

        # no need to run anything if no structures are provided
        if self.N == 0:
            return

        if self.mask is None:
            self.simulate_centers_CSR()
        else:
            self.simulate_centers_mask()

    def simulate_centers_CSR(self) -> None:
        """Simulate CSR distributed structure center positions on a
        rectangular ROI (2D or 3D)."""
        # simulate x, y and z coordinates (in nm)
        x = np.random.uniform(self.x_min, self.x_max, self.N)
        y = np.random.uniform(self.y_min, self.y_max, self.N)
        if self.depth is not None:
            z = np.random.uniform(self.z_min, self.z_max, self.N)
            self.c_pos = np.stack((x, y, z)).T
        else:
            self.c_pos = np.stack((x, y)).T

    def simulate_centers_mask(self) -> None:
        """Simulate CSR distributed structure centers within the mask
        in 2D or 3D."""
        if self.mask.ndim == 2:
            self.simulate_centers_mask_2D()
        elif self.mask.ndim == 3:
            self.simulate_centers_mask_3D()
        else:
            raise IndexError("Incorrect mask dimensionality.")

    def simulate_centers_mask_2D(self) -> None:
        """Simulate CSR distributed structure centers within the mask
        in 2D.

        Draw numbers of structures in each mask pixel using a
        multinomial distribution (provided by the mask). Then,
        generate random positions of structure centers within each
        mask pixel."""
        # Get mask pixel size #
        binsize = self.mask_info["Binsize (nm)"]

        # Draw from multinomial distribution #
        rng = np.random.default_rng()
        # find the number of structures to be simulated in each mask
        # pixel; numpy function only allows 1D arrays, thus this needs
        # to be later modified
        counts = rng.multinomial(self.N, pvals=self.mask.ravel())

        # CSR positions are generated using np.random.uniform();     #
        # It takes in inputs low and high specifying the min and max #
        # value of the random float generated.                       #
        # The first step is to generate a meshgrid; this will be used
        # when counting number of locs per mask pixel/voxel;
        # Lastly, let us consider the 'low' argument at first, since
        # creating high will be trivial, see below.
        bins_x_left = np.arange(self.x_min, self.x_max, binsize)
        bins_y_left = np.arange(self.y_min, self.y_max, binsize)
        bins_x_left, bins_y_left = np.meshgrid(bins_x_left, bins_y_left)

        # Then, 'low' argument is found by copying left bins edges
        # counts many times
        lows_x = np.repeat(bins_x_left.ravel(), counts)
        lows_y = np.repeat(bins_y_left.ravel(), counts)

        # 'high' is simply shifted by binsize in nm
        highs_x = lows_x + binsize
        highs_y = lows_y + binsize

        # generate random positions within mask pixels
        x = np.random.uniform(lows_x, highs_x)
        y = np.random.uniform(lows_y, highs_y)

        # Save center positions #
        self.c_pos = np.stack((x, y)).T

    def simulate_centers_mask_3D(self) -> None:
        """Simulate CSR distributed structure centers within the mask
        in 3D.

        Similar to 2D; see comments in self.simulate_centers_mask_2D
        for code explanation."""
        binsize = self.mask_info["Binsize (nm)"]
        rng = np.random.default_rng()
        counts = rng.multinomial(self.N, pvals=self.mask.ravel())

        bins_x_left = np.arange(self.x_min, self.x_max, binsize)
        bins_y_left = np.arange(self.y_min, self.y_max, binsize)
        bins_z_left = np.arange(self.z_min, self.z_max, binsize)
        bxl, byl, bzl = np.meshgrid(bins_x_left, bins_y_left, bins_z_left)

        lows_x = np.repeat(bxl.ravel(), counts.ravel())
        lows_y = np.repeat(byl.ravel(), counts.ravel())
        lows_z = np.repeat(bzl.ravel(), counts.ravel())
        highs_x = lows_x + binsize
        highs_y = lows_y + binsize
        highs_z = lows_z + binsize

        x = np.random.uniform(lows_x, highs_x)
        y = np.random.uniform(lows_y, highs_y)
        z = np.random.uniform(lows_z, highs_z)
        self.c_pos = np.stack((x, y, z)).T

    def simulate_all_targets(self) -> None:
        """Simulate and randomly rotate all molecular targets, given
        center positions. Takes into account label uncertainty but does
        not simulate labeling efficiency."""
        if self.c_pos is not None:
            self.pos = {}
            # generate random rotations
            rotations = random_rotation_matrices(
                len(self.c_pos), mode=self.random_rot_mode
            )

            # iterate over each molecular target species separately
            for i, target in enumerate(self.structure.targets):
                x = self.structure.x[target]
                y = self.structure.y[target]
                z = self.structure.z[target]  # required for 3D rotation

                # initialize x,y,z coordinates for each structure
                coords = self.initialize_coordinates(x, y, z)
                # rotate each structure separetely
                coords = self.rotate_structures(coords, rotations)
                # extract x and y coordinates in case of 2D simulation
                if self.depth is None:
                    coords = coords[:, :, :2]
                # shift rotated molecules by center positions
                coords += np.expand_dims(self.c_pos, 1)
                # add label uncertainty
                coords = np.random.normal(loc=coords, scale=self.label_unc[i])
                # reshape the resulting array to x,y,z coordinates
                coords = self.reshape_coordinates(coords)

                # save positions of the molecular targets
                self.pos[target] = coords

    def initialize_coordinates(
        self,
        x: list[float],
        y: list[float],
        z: list[float],
    ) -> np.ndarray:
        """Initialize coordinates of molecular targets as a 3D array.

        Parameters
        ----------
        x, y, z : list
            x/y/z coordinates of a single molecular target species.

        Returns
        -------
        coords : np.array
            Array of shape (N, M, 2) for 2D or (N, M, 3) for 3D, where
            N is number of structures and M is the number of molecular
            targets in the structure.
        """
        # initialize single structure
        coords = np.stack((x, y, z)).astype(np.float32).T
        # extend to N structures
        coords = np.tile(coords, reps=(self.N, 1, 1))
        return coords

    def rotate_structures(
        self,
        coords: np.ndarray,
        rotations: np.ndarray,
    ) -> np.ndarray:
        """Rotate coordinates of each molecular target with a defined
        rotation.

        Parameters
        ----------
        coords : np.3darray
            Array of shape (N, M, 3) specifying x,y,z coordinates of
            each rotated molecular target.
            N - number of structures/structures;
            M - number of molecular targets in the structure.
        rotations : np.3darray
            Array of shape (N, 3, 3) specifying rotation matrices for
            each structure.

        Returns
        -------
        coords_rot : np3darray
            Array of shape (N, M, 3) with the rotated coordinates.
        """
        N, M, _ = coords.shape
        # reshape matrices to allow matrix multiplication
        coords_ = coords.reshape(N*M, 3)
        # rotation matrix for each molecule (not structure)
        rotations_ = np.repeat(rotations, M, axis=0)

        # apply the rotations and reshape
        coords_rot = Rotation.from_matrix(rotations_).apply(coords_)
        coords_rot = coords_rot.reshape(N, M, 3)
        return coords_rot

    def reshape_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Reshape x,y,z coordinates to a 2D array for saving
        molecular targets' positions.

        Parameters
        ----------
        coords : np.array
            Array of shape N, M, 2) (2D) or (N, M, 3) (3D), specifying
            x,y,z coordinates of molecular targets, where N is the
            number of structures and M is the number of molecular
            targets in a single structure.

        Returns
        -------
        coords : np.array
            Reshaped array.
        """
        N, M, ndim = coords.shape
        output_shape = (N * M, ndim)
        return coords.reshape(output_shape)

    def simulate_le(self) -> None:
        """Simulate labeling efficiency by randomly choosing molecular
        targets."""
        if self.pos:
            self.pos_obs = {}
            # iterate over all molecular targets species
            for i, target in enumerate(self.pos.keys()):
                N = len(self.pos[target])  # total number of molecules
                # indeces to be kept after LE correction
                le_idx = np.random.choice(
                    N, size=int(N * self.le[i]), replace=False
                )
                # extract molecules (LE)
                self.pos_obs[target] = self.pos[target][le_idx, :]

    def run(
        self,
        save_centers: bool = False,
        save_all_mol: bool = False,
        save_obs_mol: bool = False,
        path_base: str | None = None,
    ) -> StructureSimulator:
        """Simulate positions of structure centers, arrangement of
        molecular targets around structures and labeling efficiency.

        Allows saving of molecular targets and structure centers.

        Parameters
        ----------
        save_centers : bool (defaul=False)
            Whether or not to save centers of structures.
        save_all_mol : bool (default=False)
            Whether or not to save all simulated molecular targets'
            positions.
        save_obs_mol : bool (default=False)
            Whether or not to save observed positions of molecular
            targets, i.e., after labeling efficiency correction.
        path_base : str (default=None)
            Path base for saving files. Suffixes are automatically
            generated for each molecular target species.

        Returns
        -------
        self : StructureSimulator
        """
        self.simulate_centers()
        self.simulate_all_targets()
        self.simulate_le()

        if any([save_centers, save_all_mol, save_obs_mol]):
            if path_base is not None:
                self.save(
                    path_base=path_base,
                    centers=save_centers,
                    all_mol=save_all_mol,
                    obs_mol=save_obs_mol,
                )
            else:
                raise ValueError(
                    "Please specify path_base for saving."
                )

        return self

    def save(
        self,
        path_base: str,
        centers: bool = True,
        all_mol: bool = True,
        obs_mol: bool = True,
        pixelsize: int = 130,
    ) -> None:
        """Save center positions, all molecular targets and the observed
        ones in .hdf5 format compatible to read with Picasso: Render.

        Parameters
        ----------
        path_base : str
            Path base for saving files. Suffixes are automatically
            generated.
        centers : bool (default=True)
            Whether or not to save centers of structures.
        all_mol : bool (default=True)
            Whether or not to save all simulated molecular targets'
            positions.
        obs_mol : bool (default=True)
            Whether or not to save observed positions of molecular
            targets, i.e., after labeling efficiency correction.
        pixelsize : int (default=130)
            Camera pixel size to be used to save in Picasso format.
            Should be specified if no mask is given. Otherwise, the
            mask metadata should specify it.
        """
        if self.mask_info is not None:
            pixelsize = self.mask_info["Camera pixelsize (nm)"]

        # yaml metadata
        info = [
            {
                "Height": int(self.height / pixelsize),
                "Width": int(self.width / pixelsize),
                "Frames": 1,  # some arbitrary number
            },
            {
                "Pixelsize": pixelsize,
                "Generated by": "Picasso Localize",
            },  # Render requires the second element to read the file
        ]

        # centers
        if centers and self.c_pos is not None:
            path = f"{path_base}_centers.hdf5"
            frame = np.ones(self.N)
            lpx = np.ones(self.N) / pixelsize
            lpy = lpx
            x = self.c_pos[:, 0] / pixelsize
            y = self.c_pos[:, 1] / pixelsize
            if self.depth is not None:
                z = self.c_pos[:, 2]  # not scaled for picasso compatibility
                locs = np.rec.array(
                    (frame, x, y, z, lpx, lpy),
                    dtype=LOCS_DTYPE_3D,
                )
            else:
                locs = np.rec.array(
                    (frame, x, y, lpx, lpy),
                    dtype=LOCS_DTYPE_2D,
                )
            io.save_locs(path, locs, info)

        for i, name in enumerate(self.structure.targets):
            # all molecular targets
            if all_mol and self.pos is not None:
                pos = self.pos[name]
                path = f"{path_base}_{name}_all_mols.hdf5"
                frame = np.ones(self.N)
                lpx = self.label_unc[i] * np.ones(self.N) / pixelsize
                lpy = lpx
                x = pos[:, 0] / pixelsize
                y = pos[:, 1] / pixelsize
                if pos.shape[1] == 3:
                    z = pos[:, 2]  # not scaled for picasso compatibility
                    locs = np.rec.array(
                        (frame, x, y, z, lpx, lpy),
                        dtype=LOCS_DTYPE_3D,
                    )
                else:
                    locs = np.rec.array(
                        (frame, x, y, lpx, lpy),
                        dtype=LOCS_DTYPE_2D,
                    )
                io.save_locs(path, locs, info)

            # observed molecular targets
            if obs_mol and self.pos_obs is not None:
                pos_obs = self.pos_obs[name]
                N = len(pos_obs)
                path = f"{path_base}_{name}_obs_mols.hdf5"
                frame = np.ones(N)
                lpx = self.label_unc[i] * np.ones(N) / pixelsize
                lpy = lpx
                x = pos_obs[:, 0] / pixelsize
                y = pos_obs[:, 1] / pixelsize
                if pos_obs.shape[1] == 3:
                    z = pos_obs[:, 2]  # not scaled for picasso compatibility
                    locs = np.rec.array(
                        (frame, x, y, z, lpx, lpy),
                        dtype=LOCS_DTYPE_3D,
                    )
                else:
                    locs = np.rec.array(
                        (frame, x, y, lpx, lpy),
                        dtype=LOCS_DTYPE_2D,
                    )
                io.save_locs(path, locs, info)


class StructureMixer():
    """Interface for mixing different structures for simulations.

    ...

    Attributes
    ----------
    label_unc : dict
        Dictionary with molecular target names as keys and labeling
        uncertainty (due to the probe size) of these targets in nm as
        values. These are used as sigmas of Gaussian distributions
        that are used to shift the coordinates of the molecules. The
        molecular target names given must be the same as in
        self.structures. Alternatively, "ALL" can be used as the key and
        this will be applied to all molecular targets.
    le : dict
        Dictionary with molecular target names as keys and labeling
        efficiencies of these targets as values (float between 0 and
        1). The molecular target names given must be the same as in
        self.structures. Alternatively, "ALL" can be used as the key and
        this will be applied to all molecular targets.
    mask_dict : dict
        Dictionary of the form {"mask": mask, "info": mask_info}, where
        mask and mask_info are defined below.
    mask : dict
        Dictionary with molecular target names as keys and masks of
        these targets as values. The masks are 2D or 3D numpy.arrays
        giving the probabilites of finding a molecule of the given
        target in each mask pixel (2D) or voxel (3D). The molecular
        target names given must be the same as in self.structures.
        Alternatively, "ALL" can be used as the key and this will be
        applied to all molecular targets.
    mask_info : dict
        Dictionary with molecular target names as keys and masks'
        metadata as values. The masks should be created using
        MaskGenerator. Metadata specify the dimensions of the mask,
        its pixel/voxel size, etc. The molecular target names given
        must be the same as in self.structures. Alternatively, "ALL"
        can be used as the key and this will be applied to all
        molecular targets.
    nn_counts : dict or "auto"
        Dictionary with pairs of molecular target names as keys and
        the number of nearest neighbors between these pairs that are
        considered, for example, at fitting, as values. The keys must
        have the format "target1-target2". If "auto" the values are
        found automatically, see documentation.
    structures : list of SingleStructures
        List of structures that are to be simulated. These specify the
        molecular target names for each structure as well as the
        spatial coordinates of the molecular targets within each
        structure.
    targets : list of strs
        List of all unique molecular target names present across all
        structures in self.structures.
    random_rot_mode : {"3D", "2D", None} (default="2D")
        Mode of random rotation of structures. If "3D", structures are
        rotated randomly in 3D. If "2D", structures are rotated
        randomly in 2D. If None, structures are not rotated.
    simulators : list of StructureSimulators
        List of StructureSimulator instances, one for each structure in
        self.structures.
    roi : list of floats
        Width, height and depth of the simulated ROI in nm. If mask is
        provided, ROI is overwritten using the mask metadata. If width,
        height and depth are None, mask_dict must be specified. If
        width and height are given, but no mask and depth is provided,
        2D simulation will be conducted. If width, height and depth are
        given but no mask is provided, 3D simulation will be conducted.

    Parameters
    ----------
    structures : list of Structure
        List of structures to be simulated.
    label_unc : dict
        Dictionary with molecular target names as keys and their
        labeling uncertainties in nm as values.
    le : dict
        Dictionary with molecular target names as keys and their
        localisation errors in nm as values.
    mask_dict : dict or None, optional
        Dictionary with molecular target names as keys and masks of
        these targets as values. The masks are 2D or 3D numpy.arrays
        giving the probabilites of finding a molecule of the given
        target in each mask pixel (2D) or voxel (3D). The molecular
        target names given must be the same as in self.structures.
        Alternatively, "ALL" can be used as the key and this will be
        applied to all molecular targets. Default is None.
    width, height, depth : float or None
        Dimensions of the simulated region of interest (ROI) in nm.
        If mask is provided, ROI is overwritten using the mask metadata.
        If width, height and depth are None, mask_dict must be
        specified. If width and height are given, but no mask and depth
        is provided, 2D simulation will be conducted. If width, height
        and depth are given but no mask is provided, 3D simulation will
        be conducted.
    random_rot_mode : {'2D', '3D'} or None
        Mode of random rotation of structures. If "3D", structures are
        rotated randomly in 3D. If "2D", structures are rotated
        randomly in 2D. If None, structures are not rotated. Default
        is '2D'.
    nn_counts : dict or "auto"
        Dictionary with pairs of molecular target names as keys and
        the number of nearest neighbors between these pairs that are
        considered, for example, at fitting, as values. The keys must
        have the format "target1-target2". If "auto" the values are
        found automatically, see documentation.
    """

    def __init__(
        self,
        structures: list[Structure],
        label_unc: dict,
        le: dict,
        mask_dict: dict | None = None,
        width: float | None = None,
        height: float | None = None,
        depth: float | None = None,
        random_rot_mode: Literal["2D", "3D"] | None = "2D",
        nn_counts: Literal["auto"] | dict = "auto",
    ) -> None:
        self.structures = structures
        self.label_unc = label_unc
        self.le = le
        self.mask_dict = mask_dict
        self.roi = [width, height, depth]
        self.random_rot_mode = random_rot_mode
        self.nn_counts = nn_counts
        # StructureSimulator instances (one for each structure)
        self.simulators = []

        # check for correct input types
        self.ensure_correct_input()

        # get all unique molecular targets names given in
        # self.structures
        self.targets = self.get_target_names()

        # check if all inputs are compatible with each other, i.e.,
        # contain info for all molecular targets in self.structures
        self.ensure_consistency()

    def get_target_names(self) -> list[str]:
        """Extract unique molecular target names given in
        ``self.structures`` into ``self.targets``.

        Returns
        -------
        targets : list of strs
            List of all unique molecular target names present across all
            structures in self.structures.
        """
        targets = []
        for structure in self.structures:
            for target in structure.targets:
                if target not in targets:
                    targets.append(target)
        return targets

    def ensure_correct_input(self) -> None:
        """Ensure correct formats are input at initialization."""
        self.check_structures()
        self.check_label_unc()
        self.check_le()
        self.check_mask_and_roi()

    def check_structures(self) -> None:
        """Check if self.structures has correct format."""
        if isinstance(self.structures, Structure):  # single structure case
            self.structures = [self.structures]
        elif not isinstance(self.structures, list):
            raise TypeError("Please input structures as a list.")
        elif any([not isinstance(m, Structure) for m in self.structures]):
            raise TypeError(
                "All structures must be instances of the Structure class."
            )

    def check_label_unc(self) -> None:
        """Check if self.label_unc has correct format."""
        if not isinstance(self.label_unc, dict):
            raise TypeError(
                "labeling uncertainties must be input as a dictionary.\n"
                "Please provide molecular target name(s) as key(s) and"
                " labeling uncertainties in nm as corresponding values,"
                " e.g.:\n"
                '{"CD80", 2.3, "PDL1", 1.89}.'
            )
        if not all([
            isinstance(label_unc, Number)
            for label_unc in self.label_unc.values()
        ]):
            raise TypeError(
                "labeling uncertainties must be positive numbers."
            )
        if not all([label_unc >= 0 for label_unc in self.label_unc.values()]):
            raise ValueError(
                "labeling uncertainties must be positive numbers."
            )

    def check_le(self) -> None:
        """Check if ``self.le`` has correct format."""
        if not isinstance(self.le, dict):
            raise TypeError(
                "labeling efficiencies must be input as a dictionary.\n"
                "Please provide molecular target name(s) as key(s) and"
                " labeling efficiencies (between 0.0 and 1.0) as"
                " corresponding values, e.g.:\n"
                '{"CD80", 0.53, "PDL1", 0.6}.'
            )
        if not all([isinstance(le, Number) for le in self.le.values()]):
            raise TypeError(
                "labeling efficiencies must be floats between zero and one."
            )
        if not all([0 < le <= 1 for le in self.le.values()]):
            raise ValueError(
                "labeling efficiencies must be floats between zero and one."
            )

    def check_mask_and_roi(self) -> None:
        """Check if ``self.mask_dict`` and ``self.roi`` have correct
        format."""
        if self.mask_dict is None:
            if self.roi[0] is not None and self.roi[1] is not None:
                self.mask = None
                self.mask_info = None
                self.ensure_correct_roi()
                return
            else:
                raise TypeError(
                    "If no mask is provided, ROI dimensions must be provided."
                )
        else:
            self.roi = [None, None, None]
            self.mask = self.mask_dict["mask"]
            self.mask_info = self.mask_dict["info"]

    def ensure_correct_roi(self) -> None:
        """Check if ``self.roi`` has correct format."""
        width, height, depth = self.roi
        try:
            width = float(width)
            height = float(height)
        except ValueError:
            raise TypeError("Width and height must be positive numbers.")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive numbers.")
        if depth is not None:
            try:
                depth = float(depth)
            except ValueError:
                raise TypeError("Depth must be a positive numbers.")
            if depth <= 0:
                raise ValueError("Depth must be a positive numbers.")

    def ensure_consistency(self) -> None:
        """Check if all input parameters are consistent with each
        other, i.e., that the molecular targets listed in
        ``self.structures`` are found in all input dictionaries given in
        ``self.__init__()``."""
        for target in self.targets:
            if (
              "ALL" not in self.label_unc.keys()
              and target not in self.label_unc.keys()
            ):
                raise KeyError(
                    f"{target} defined in the model was not specified in"
                    " self.label_unc."
                )
            if (
              "ALL" not in self.le.keys()
              and target not in self.le.keys()
            ):
                raise KeyError(
                    f"{target} defined in the model was not specified in"
                    " self.le."
                )
        if self.nn_counts != "auto":
            if isinstance(self.nn_counts, dict):
                for i, target1 in enumerate(self.targets):
                    for target2 in self.targets[i:]:
                        if f"{target1}-{target2}" not in self.nn_counts.keys():
                            raise KeyError(
                                f"{target1}-{target2} defined in"
                                " the model was not specified in"
                                " self.nn_counts."
                            )
            else:
                raise TypeError(
                    "Please input nearest neighbor counts as a dictionary or"
                    " choose the automatic values by inputting 'auto'."
                )

    def run_simulation(
        self,
        N_structures: list | np.ndarray,
        path: str = '',
    ) -> dict:
        """Run a simulation with the given numbers of structures.

        Parameters
        ----------
        N_structures : list or 1D np.ndarray
            Each element gives the number of structures to be simulated.
            Must have the same number of elements as self.structures as
            well as the same ordering.
        path : str (default='')
            Path to save simulated molecules. If '' is given, the
            molecules are not saved. When saving molecules, "_sim_" and
            molecular targets' names will be appended to path.

        Returns
        -------
        all_locs : dict
            Dictionary with simulated molecular targets' names as keys
            and spatial coordinates of localizations (from combined
            simulations for each structure) as values.
        """
        if (
            not isinstance(N_structures, list)
            and (
                not isinstance(N_structures, np.ndarray)
                and N_structures.ndim != 1
            )
        ):
            raise TypeError(
                "Please input numbers of structures as a list or 1D array."
            )
        if any([N < 0 for N in N_structures]):
            raise ValueError("Numbers of structures must be positive numbers.")

        # list to store coordinates of simulated molecules
        sim_results = []
        self.simulators = []
        for i, structure in enumerate(self.structures):
            # get inputs for random distribution #
            targets = structure.targets
            # labeling efficiency
            if "ALL" in self.le.keys():
                le = [self.le["ALL"] for _ in targets]
            else:
                le = [self.le[t] for t in targets]
            # label uncertainty
            if "ALL" in self.label_unc.keys():
                label_unc = [self.label_unc["ALL"] for _ in targets]
            else:
                label_unc = [self.label_unc[t] for t in targets]
            # mask and metadata
            mask, mask_info = self.extract_mask(structure)
            # ROI
            width, height, depth = self.roi

            # generate random distribution
            simulator = StructureSimulator(
                structure=structure,
                N_structures=N_structures[i],
                le=le,
                label_unc=label_unc,
                mask=mask, mask_info=mask_info,
                width=width, height=height, depth=depth,
                random_rot_mode=self.random_rot_mode,
            ).run()
            self.simulators.append(simulator)
            sim_results.append(simulator.pos_obs)

        # convert simulation results into correct format
        all_locs = self.convert_sim_results(sim_results)

        if path:
            self.save(path, all_locs)

        return all_locs

    def extract_mask(
        self,
        structure: Structure,
    ) -> tuple[np.ndarray, dict] | tuple[None, None]:
        """Extract masks and metadata for the given structure.

        If a heteromultimer is simulated, weighted average of masks is
        used. In this case, it is assumed that masks have the same
        shape.

        Parameters
        ----------
        structure : Structure
            Structure to be simulated.

        Returns
        -------
        mask : np.ndarray or None
            Mask for the given molecular targets.
        mask_info : dict or None
            Metadata for the mask.
        """
        targets = structure.targets
        if self.mask is not None:
            if "ALL" in self.mask.keys():
                mask = self.mask["ALL"]
                mask_info = self.mask_info["ALL"]
            else:
                if len(targets) == 1:
                    mask = self.mask[targets[0]]
                else:  # multiplied masks
                    mask = np.ones_like(self.mask[targets[0]])
                    n_molecules = structure.get_ind_target_count(targets)
                    for n, target in zip(n_molecules, targets):
                        mask *= n * self.mask[target]
                    mask /= mask.sum()  # normalize
                mask_info = self.mask_info[targets[0]]
        else:
            mask = None
            mask_info = None
        return mask, mask_info

    def convert_sim_results(self, sim_results: list[np.ndarray]) -> dict:
        """Convert sim_results calculated by multiple
        ``StructureSimulator``'s into a dictionary with molecules
        ordered by their molecular targets' names.

        Parameters
        ----------
        sim_results : list of arrays
            Each element contains spatial coordinates of simulated
            molecules for each simulated structure.

        Returns
        -------
        all_locs : dict
            Dictionary with simulated molecular targets' names as keys
            and spatial coordinates of localizations (from combined
            simulations for each structure) as values.
        """
        # initialize output
        all_locs = {t: [] for t in self.targets}

        # append coordinates to each
        for result in sim_results:
            targets = result.keys()
            for target in targets:
                coords = result[target]
                all_locs[target].append(coords)

        # concatenate coordinates for each molecular target species
        for target in all_locs:
            locs = all_locs[target]
            if len(locs):
                locs = np.concatenate(locs)
            all_locs[target] = locs

        return all_locs

    def save(
        self,
        path: str,
        all_locs: dict,
        lp: float = 1.0,
        pixelsize: int | None = None,
    ) -> None:
        """Save observed molecules. Each molecular target is saved in
        a separate .hdf5 file. The saved files can be read in Picasso
        Render.

        Parameters
        ----------
        path : str
            Path to save molecules. Should end with ".hdf5". Each saved
            file will be added the suffix _TARGETNAME.
        all_locs : dict
            Dictionary with molecular target names as keys and
            np.ndarrays with spatial coordinates of the molecules to be
            saved. Each of the arrays must have shape (N, 2) or (N, 3),
            where N is the number of molecules of the given molecular
            target species to be saved.
        N_structures : list or np.ndarray
            Numbers of structures that were simulated.
        lp : float (default=1.0)
            Localization precision in nm to be assigned to saved
            molecules.
        pixelsize : float (default=130.0)
            Camera pixelsize in nm to be saved. Required for
            compatibility with Picasso Render.
        """
        # prepare path for saving
        if not path.endswith(".hdf5"):
            path = f"{path}.hdf5"

        # prepare metadata
        if pixelsize is None:
            if self.mask_info is not None:
                pixelsize = list(
                    self.mask_info.values()
                )[0]["Camera pixelsize (nm)"]
            else:
                pixelsize = 130

        if self.mask is not None:
            binsize = list(self.mask_info.values())[0]["Binsize (nm)"]
            height = list(self.mask.values())[0].shape[1] * binsize
            width = list(self.mask.values())[0].shape[0] * binsize
        else:
            width, height, _ = self.roi

        # save each molecular target separately
        for tname in self.targets:
            coords = all_locs[tname]
            if len(coords):
                locs = coords_to_locs(coords, lp=lp, pixelsize=pixelsize)
                info = self.get_metadata(tname, width, height, pixelsize)
                outpath = path.replace(".hdf5", f"_{tname}.hdf5")
                io.save_locs(outpath, locs, info)

    def get_metadata(
        self,
        target: str,
        width: float,
        height: float,
        pixelsize: int,
    ) -> list[dict]:
        """Return metadata for saving molecules in Picasso format, i.e.,
        the data that will be saved in .yaml format.

        Parameters
        ----------
        target : str
            Name of the molecular target to be saved.
        width, height : float
            Width/height of the ROI in nm.
        pixelsize : int
            Camera pixel size (nm).

        Returns
        -------
        info : list of dict
            Metadata to be saved.
        """
        if "ALL" in self.label_unc.keys():
            label_unc = self.label_unc["ALL"]
        else:
            label_unc = self.label_unc[target]

        if "ALL" in self.le.keys():
            le = self.le["ALL"]
        else:
            le = self.le[target]

        info = [
            {
                "Height": int(height / pixelsize),
                "Width": int(width / pixelsize),
                "Frames": 1,
            },  # necessary to use in Picasso Render
            {
                "Pixelsize": pixelsize,
                "Generated by": "Picasso Localize",
            },  # necessary to use in Picasso Render
            {
                "Generated by": f"Picasso v{__version__} SPINNA",
                "Label uncertainty (nm)": label_unc,
                "labeling efficiency (%)": int(100 * le),
                "Rotations mode": self.random_rot_mode,
            },  # simulation info
        ]
        return info

    def get_neighbor_counts(self, target1: str, target2: str) -> int:
        """Find maximum number of neighbors that two molecular target
        species form in ``self.structures`` such that their nearest
        neighbor distances are not expected to follow the distances of
        complete spatial randomness. For example, if molecular targets
        A and B form dimers, the maximum number of neighbors is 1.

        Parameters
        ----------
        target1, target2 : str
            Names of molecular targets to consider. Assumes that both
            target1 and target2 are present in self.targets.

        Returns
        -------
        count : int
            Maximum number of neighbors in a structure.
        """
        if self.nn_counts == "auto":
            count = 0
            for structure in self.structures:
                current_count = structure.get_max_nn(target1, target2)
                count = max(count, current_count)
        else:
            count = self.nn_counts[f"{target1}-{target2}"]
        return count

    def get_neighbor_idx(self, duplicate: bool = False) -> list[tuple]:
        """Find which kth nearest neighbors and cross neighbors are
        relevant in the mixture.

        For each molecular target pair, assignes n_neighbors to
        consider in a tuple. For each tuple, the first two elements
        (str) give names of molecuar targets to consider and the third
        element specifies n_neighbors.

        For example, if target1 forms monomers and dimers and target2
        forms dimers with target1, the following list of tuples is
        given:
        [(target1, target2, 2),
         (target1, target2, 1)]

        Parameters
        ----------
        duplicate : bool (default=False)
            If True, two directions in pairs of molecular species are
            considered, i.e., (target1, target2) and
            (target2, target1).

        Returns
        -------
        neighbor_idx : list of tuples
            Custom indexing of pairs of molecular targets to consider.
        """
        if self.targets is None:
            return

        neighbor_idx = []

        # iterate over each possible pair of molecular targets
        # (including pairing with the same molecular target)
        for i, target1 in enumerate(self.targets):
            for target2 in self.targets[i:]:
                # how many neighbors in a single structure?
                n_neighbors = self.get_neighbor_counts(target1, target2)
                neighbor_idx.append((target1, target2, n_neighbors))
                # duplicate if requested
                if duplicate and target1 != target2:
                    neighbor_idx.append((target2, target1, n_neighbors))
        return neighbor_idx

    def get_structure_names(self) -> list[str]:
        """Return names of all structures in a list."""
        return [m.title for m in self.structures]

    def convert_props_for_target(
        self,
        props: np.ndarray,
        target: str,
        n_mols: dict,
    ) -> np.ndarray:
        """Convert the given proportions of structures to the relative
        proportions of the given molecular target.

        Parameters
        ----------
        props : np.ndarray
            Relative proportions of structures (0 to 100). Can be 1D or
            2D.
        target : str
            Name of the molecular target for which the proportions are
            to be converted.
        n_mols : dict
            Dictionary with molecular target names as keys and numbers
            of molecules of these targets as values. The molecular
            target names given must be the same as in self.structures.

        Returns
        -------
        props_target : np.ndarray
            Relative proportions of the given molecular target.
        """
        targets_per_str = [_.get_all_targets_count() for _ in self.structures]
        t_counts = find_target_counts([target], self.structures).reshape(-1)
        n_target = n_mols[target]
        n_total = sum(list(n_mols.values()))
        n_str = props * n_total / targets_per_str
        props_target = n_str * t_counts / n_target
        # ignore the structures where the target is not present
        props_target[t_counts == 0] = np.inf
        return props_target

    def convert_counts_to_props(
        self,
        N_structures: list | np.ndarray,
    ) -> np.ndarray:
        """Convert numbers of structures to their relative
        proportions (%).

        The proportions are defined as the ratios of all
        targets arising from the given structure compared to the total
        number of targets simulated. The proportions sum to 100%.

        Parameters
        ----------
        N_structures : list or np.ndarray
            Each element (1D) or row (2D) gives the number of
            structures to be simulated. Must have the same number of
            elements (1D) or columns (2D) as self.structures as well as
            the same ordering.

        Returns
        -------
        props : np.ndarray
            Resulting proportions (0 to 100).
        """
        N_structures = deepcopy(N_structures)
        if isinstance(N_structures, list):
            N_structures = np.int32(N_structures)
        elif isinstance(N_structures, dict):
            N = len(list(N_structures.values())[0])  # number of simulations
            N_structures_ = np.zeros(
                (N, len(self.mixer.structures)),
                dtype=np.int32,
            )
            for i, structure in enumerate(self.mixer.structures):
                N_structures_[:, i] = N_structures[structure.title]
            N_structures = N_structures_

        if N_structures.ndim == 1:
            N_structures = N_structures.reshape(1, -1)
        elif N_structures.ndim != 2:
            raise TypeError(
                "Please input numbers of structures as a list or 1D/2D array."
            )

        if N_structures.shape[1] != len(self.structures):
            raise ValueError(
                "Please input numbers of structures for each structure"
                " in self.structures."
            )

        # initialize proportions array
        props = np.zeros(N_structures.shape, dtype=np.float32)

        # find total numbers of targets for each row
        N_total = np.zeros(N_structures.shape[0], dtype=np.int32)
        for i, structure in enumerate(self.structures):
            N_str_total = np.zeros(N_structures.shape[0], dtype=np.float32)
            N_per_target = structure.get_ind_target_count(self.targets)
            for N_mol in N_per_target:
                # n all targets given n structures
                N_str_total = N_str_total + N_mol * N_structures[:, i]
            N_total = N_total + np.int32(N_str_total)

        # find proportions
        for i, structure in enumerate(self.structures):
            N_str_total = np.zeros(N_structures.shape[0], dtype=np.float32)
            N_per_target = structure.get_ind_target_count(self.targets)
            for N_mol in N_per_target:
                N_str_total = N_str_total + N_mol * N_structures[:, i]
            prop = np.round(100 * N_str_total / N_total, 2)
            props[:, i] = prop

        # if rounding error occurs, delete from the first non-zero element
        rows_to_correct = np.where(np.sum(props, axis=1) != 100)[0]
        for row in rows_to_correct:
            first_non_zero_idx = next(
                i for i, prop in enumerate(props[row, :]) if prop > 0
            )
            props[row, first_non_zero_idx] -= np.sum(props[row, :]) - 100

        if props.shape[0] == 1:
            props = props.reshape(-1)

        return props

    def convert_props_to_counts(
        self,
        proportions: list | np.ndarray,
        N_total: int | np.ndarray,
    ) -> np.ndarray:
        """Convert relative proportions (%) of structures to their
        absolute counts.

        Input proportions are assumed to sum to 100%.

        Parameters
        ----------
        proportions : list or np.ndarray
            Each element (1D) or row (2D) gives the relative proportion
            of the given structure (0 to 100). Must have the same
            number of elements (1D) or columns (2D) as self.structures
            as well as the same ordering.
        N_total : int or np.ndarray
            Total number of molecular targets (if different molecular
            species are present, they should be summed together in this
            value).

        Returns
        -------
        N_structures : np.ndarray
            Resulting numbers of structures.
        """
        proportions = deepcopy(proportions)
        if isinstance(proportions, list):
            proportions = np.float32(proportions)

        if proportions.ndim == 1:
            proportions = proportions.reshape(1, -1)
        elif proportions.ndim != 2:
            raise TypeError(
                "Please input proportions of structures as a list or 1D/2D"
                " array."
            )

        if proportions.shape[1] != len(self.structures):
            raise ValueError(
                "Please input proportions of structures for each structure"
                " in self.structures."
            )

        N_total = np.int32(N_total)  # assert numpy array
        N_structures = np.zeros(proportions.shape, dtype=np.int32)
        for i, structure in enumerate(self.structures):
            N_structures[:, i] = np.int32(
                N_total
                * proportions[:, i] / 100
                / structure.get_all_targets_count()
            )
        if N_structures.shape[0] == 1:
            N_structures = N_structures.reshape(-1)

        return N_structures

    @property
    def roi_size(self) -> float:
        """Returns the size of the ROI in um^2 or um^3."""

        if self.mask is not None:  # extract mask area/volume
            mask_info = list(self.mask_info.values())[0]
            if mask_info["Dimensionality"] == "2D":
                return mask_info["Area (um^2)"]
            else:
                return mask_info["Volume (um^3)"]
        else:  # CSR simualtion, use width, height, and optionally depth
            if self.roi[2] is None:
                return self.roi[0] * self.roi[1] * 1e-6
            else:
                return self.roi[0] * self.roi[1] * self.roi[2] * 1e-9


class SPINNA():
    """Fit simulations produced from ``StructureMixer`` to the
    experimental data by comparing their nearest neighbors distances
    (NND) across different combinations of the numbers of the
    structures.

    Provides support for multiprocessing (80% CPU cores used by
    default). Uses 2-sample Kolmogorov-Smirnov test to compare the
    fitting scores.

    ...

    Attributes
    ----------
    mixer : StructureMixer
        Structure mixture used for simulations.
    dists_gt : np.array
        NNDs of ground-truth data.
    N_sim : int
        Defines how many times each simulation is repeated with the
        same settings to obtain smoother NND histograms.

    Parameters
    ----------
    mixer : StructureMixer
        Instance of the structure mixer used for simulations.
    gt_coords : dict
        Dictionary with ground-truth molecular targets' names as
        keys and spatial coordinates (nm) of the observed molecules
        as values.
    N_sim : int, optional
        Specifies how many times each simulation with the given
        structure count should be repeated. Default is 1.
    progress_title : str, optional
        Title of the progress bar displayed during fitting simulations.
        Default is "Spinning structures".
    """

    def __init__(
        self,
        mixer: StructureMixer,
        gt_coords: dict,
        N_sim: int = 1,
        progress_title: str = "Spinning structures",
    ) -> None:
        if not isinstance(mixer, StructureMixer):
            raise TypeError("Initialize the class with StructureMixer.")

        self.mixer = mixer
        self.N_sim = N_sim
        self.progress_title = progress_title

        # NND of the ground truth coordinates. Also, in case of 3D exp
        # data but 2D simulation, the third dimension is ignored. But if
        # masking is used, we assume that the dimensions are the same
        # for both exp. and sim. data.
        if mixer.roi[0] is not None and mixer.roi[2] is None:  # 2D CSR sim.
            gt_coords = {k: v[:, :2] for k, v in gt_coords.items()}
        self.gt_coords = gt_coords
        self.dists_gt = get_NN_dist_experimental(gt_coords, mixer)

    def fit(
        self,
        N_structures: np.ndarray | dict,
        save: str = '',
        asynch: bool = True,
        bootstrap: bool = False,
        callback: lib.ProgressDialog | Literal["console"] | None = None,
    ) -> (
        tuple[np.ndarray, float] |
        tuple[tuple[np.ndarray, ...], tuple[float, ...]]
    ):
        """Find fitting error for every combination of ``N_structures``
        using NND comparison to ground truth. Applies multiprocessing
        and provides progress tracking. Outputs the optimal combination
        of structures and the correspodning score. Uses 2-sample
        Kolmogorov-Smirnov to compare the fitting scores.

        Parameters
        ----------
        N_structures : np.2darray or dict
            Specifies what combinations of structures  are to be
            simulated for each iteration. Shape (N, M), where N is the
            number of simulations to be tested and M is the number of
            structures in mixer. If dict, keys are structure names and
            values are lists of numbers of structures to be simulated.
        save : str, optional
            Path to save numbers of structures tested and their
            corresponding scores as a .csv file. If '' is given, the
            file is not saved. Default is ''.
        asynch : bool, optional
            If True, multiprocessing is used (80% of CPU cores are
            used). Else, single thread is used for fitting. Default is
            True.
        bootstrap : bool, optional
            If True, bootstrapping is used to estimate the fitting
            error. Default is False.
        callback : {lib.ProgressDialog, "console", None}, optional
            Progress bar to track fitting progress. If "console", the
            progress bar is displayed in the console. If None, no
            progress bar is displayed. Default is None.

        Returns
        -------
        opt_proportions : np.ndarray or tuple of np.ndarrays
            The stoichiometry of structures that gives the best fit to
            ground truth.
        score : float or tuple of floats
            KS2 score of the best fit.
        """
        return self.fit_stoichiometry(
            N_structures,
            save=save,
            asynch=asynch,
            bootstrap=bootstrap,
            callback=callback,
        )

    def fit_stoichiometry(
        self,
        N_structures: np.ndarray | dict,
        save: str = '',
        asynch: bool = True,
        bootstrap: bool = False,
        callback: lib.ProgressDialog | Literal["console"] | None = None,
    ) -> (
        tuple[np.ndarray, float] |
        tuple[tuple[np.ndarray, ...], tuple[float, ...]]
    ):
        """Alias for ``self.fit()``."""
        assert callback is None or isinstance(callback, lib.ProgressDialog) \
            or callback == "console", "callback must be a ProgressDialog," \
            " 'console', or None."
        if callback is None:
            callback = lib.MockProgress()

        # check and optionally convert N_structures
        if isinstance(N_structures, dict):
            N = len(list(N_structures.values())[0])  # number of simulations
            N_structures_ = np.zeros(
                (N, len(self.mixer.structures)),
                dtype=np.int32,
            )
            for i, structure in enumerate(self.mixer.structures):
                N_structures_[:, i] = N_structures[structure.title]
            N_structures = N_structures_
        elif (
            not isinstance(N_structures, np.ndarray)
            or len(N_structures.shape) != 2
        ):
            raise TypeError("N_structures must be a 2D array or a dictionary.")

        if asynch:  # fit with multiprocessing
            fs = self.fit_stoichiometry_parallel(N_structures)
            N = len(fs)
            N_ = N_structures.shape[0]
            if callback == "console":
                progress_bar = tqdm(range(N_), desc=self.progress_title)
            while self.n_futures_done(fs) < N:  # display progress
                fd = self.n_futures_done(fs)
                fd_ = int(fd * N_ / N)
                if fd > 0 and callback != "console":
                    callback.setLabelText(f"{self.progress_title} {fd_}/{N_}")
                    callback.set_value(fd_)
                elif fd > 0 and callback == "console":
                    progress_bar.update(fd_ - progress_bar.n)
                time.sleep(0.1)
            if callback != "console":
                callback.set_value(N_)
            else:
                progress_bar.update(fd_ - progress_bar.n)
                progress_bar.close()
            N_structures, scores = self.scores_from_futures(fs)
        else:  # fit in a single thread
            N_structures, scores = self.NN_scorer(
                N_structures, callback=callback
            )

        if save:
            props = self.mixer.convert_counts_to_props(N_structures)
            df = pd.DataFrame(
                np.hstack((N_structures, props, scores.reshape(-1, 1))),
                columns=[
                    f"N_{name}" for name in self.mixer.get_structure_names()
                ]+[
                    f"Prop_{name}" for name in self.mixer.get_structure_names()
                ]+['Kolmogorov-Smirnov statistic'],
            )
            df.to_csv(save, header=True, index=False)

        index = np.argmin(scores)
        score = scores[index]
        opt_N_structures = N_structures[index]
        opt_proportions = self.mixer.convert_counts_to_props(opt_N_structures)

        if bootstrap:
            exp_dists_gt = deepcopy(self.dists_gt)

            N_structures_subset = self.get_subset_N_structures(
                N_structures, opt_N_structures,
            )

            # initialize bootstrapping
            if callback != "console":
                callback.setMaximum(len(N_structures_subset))
            scores = []
            boot_props = []
            for i in range(N_BOOTSTRAPS):
                self.progress_title = (
                    f"Bootstrapping {i+1}/{N_BOOTSTRAPS}; spinning structures"
                )
                # gt_coords_boot = self.mixer.run_simulation(opt_N_structures_)
                gt_coords_boot = self.mixer.run_simulation(opt_N_structures)
                self.dists_gt = get_NN_dist_experimental(
                    gt_coords_boot, self.mixer
                )
                N_structures_boot, scores_boot = self.NN_scorer(
                    N_structures_subset, callback=callback
                )
                index_boot = np.argmin(scores_boot)
                score_boot = scores_boot[index_boot]
                scores.append(score_boot)
                boot_props.append(self.mixer.convert_counts_to_props(
                    N_structures_boot[index_boot]
                ))

            self.dists_gt = exp_dists_gt
            score_std = np.std(scores)
            props_std = np.std(boot_props, axis=0)
            return (opt_proportions, props_std), (score, score_std)
        else:
            return opt_proportions, score

    def fit_stoichiometry_parallel(self, N_structures: np.ndarray) -> list:
        """Apply multiprocessing to find best fitting combination of
        structures.

        Parameters
        ----------
        N_structures : np.2darray
            Specifies what combinations of structures  are to be
            simulated for each iteration. Shape (N, M), where N is the
            number of simulations to be tested and M is the number of
            structures in mixer.

        Returns
        -------
        fs : list of Futures
            Contain the scoring for each combination of structures
        """
        # get number of threads and tasks
        n_workers = min(
            60, max(1, int(0.75 * cpu_count()))
        )  # Python crashes when using >64 cores

        # split N_structures into groups that are tested by each Process
        # separately
        N = N_structures.shape[0]
        structures_per_task = [
            int(N / N_TASKS + 1) if _ < N % N_TASKS else int(N / N_TASKS)
            for _ in range(N_TASKS)
        ]
        start_indices = np.cumsum([0] + structures_per_task[:-1])
        fs = []
        executor = futures.ProcessPoolExecutor(n_workers)
        # call NN_scorer for each group of N_structures
        for i, n_neighbors_task in zip(start_indices, structures_per_task):
            fs.append(executor.submit(
                self.NN_scorer,
                N_structures[i:i + n_neighbors_task],
            ))
        return fs

    def NN_scorer(
        self,
        N_structures: np.ndarray,
        callback: (
            lib.ProgressDialog
            | Literal["console"]
            | lib.MockProgress
        ) = lib.MockProgress(),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score the simulations similarity to the ground truth dataset
        based on their nearest neighbor distances distribution using
        Kolmogorov-Smirnov 2 sample test.

        Uses least squares to determine the best fitting structure
        counts.

        Parameters
        ----------
        N_structures : np.2darray
            Specifies what combinations of structures  are to be
            simulated for each iteration. Shape (N, M), where N is the
            number of simulations to be tested and M is the number of
            structures in mixer.
        callback : {lib.ProgressDialog, "console", lib.MockProgress}, optional
            Progress bar to track fitting progress. If "console", the
            progress bar is displayed in the console. If MockProgress,
            no progress bar is displayed. Default is MockProgress.

        Returns
        -------
        N_structures : np.ndarray
            Same as the input N_structures.
        scores : np.ndarray
            1D array with fit scores for each combination of structures.
        """
        # Run simulations for each structure count and score them #
        scores = np.zeros((N_structures.shape[0],))
        if callback == "console":
            iterator = tqdm(
                range(N_structures.shape[0]), desc=self.progress_title
            )
        else:
            iterator = range(N_structures.shape[0])

        for ii in iterator:
            N = N_structures[ii]
            # calculate NNDs over self.N_sim repeated simulations
            dists_sim = get_NN_dist_simulated(
                N, self.N_sim, self.mixer, duplicate=False
            )
            # score the simulation results
            scores[ii] = NND_score(dists_sim, self.dists_gt)
            if callback != "console":
                callback.setLabelText(
                    f"{self.progress_title} {ii+1}/{N_structures.shape[0]}"
                )
                callback.set_value(ii)
        return N_structures, scores

    def get_subset_N_structures(
        self,
        N_structures: np.ndarray,
        center_N_structures: np.ndarray,
        radius: float = BOOTSTRAP_DISTANCE,
        p: float = BOOTSTRAP_DISTANCE_METRIC,
    ) -> np.ndarray:
        """Find a subset of N_structures that are within a given radius
        from the center_proportions.

        Parameters
        ----------
        N_structures : np.ndarray
            Array where each row specifies each structures count tested.
        center_N_structures : np.ndarray
            Array with the numbers of the structures that are considered
            as the center of the subset (ground-truth).
        radius : float (default=30.0)
            Radius of the hypersphere around the center_proportions.
        p : float (default=1.0)
            Power parameter for the Minkowski metric.

        Returns
        -------
        N_structures_subset : np.ndarray
            Subset of N_structures that are within the radius from the
            center_proportions.
        """
        if isinstance(N_structures, dict):
            N = len(list(N_structures.values())[0])  # number of simulations
            N_structures_ = np.zeros(
                (N, len(self.mixer.structures)),
                dtype=np.int32,
            )
            for i, structure in enumerate(self.mixer.structures):
                N_structures_[:, i] = N_structures[structure.title]
            N_structures = N_structures_

        proportions = self.mixer.convert_counts_to_props(N_structures)
        center_proportions = (
            self.mixer.convert_counts_to_props(center_N_structures)
        )

        proportions_tree = KDTree(proportions)
        indices = proportions_tree.query_ball_point(
            center_proportions, r=radius, p=p
        )
        if len(indices) > len(N_structures):
            return N_structures
        else:
            N_structures_subset = N_structures[indices]
            return N_structures_subset

    def n_futures_done(self, fs: list) -> int:
        """Find the number of tasks finished in the futures.

        Parameters
        ----------
        fs : list of concurrent.Futures
        """
        return sum([_.done() for _ in fs])

    def scores_from_futures(self, fs: list) -> tuple[np.ndarray, np.ndarray]:
        """Convert futures resulting from fitting N_structures with
        multiprocessing.

        Parameters
        ----------
        futures : list of concurrent.Futures
            Futures provided from ProcessPoolExecutor after NN fitting.

        Returns
        -------
        N_structures : np.ndarray
            Array where each row specifies each structures count tested.
        scores : np.ndarray
            Array with the corresponding fitting scores.
        """
        res_list = [f.result() for f in fs]
        N_structures = np.vstack([res[0] for res in res_list])
        scores = np.concatenate([res[1] for res in res_list])
        return N_structures, scores


def compare_models(
    models: list[list[Structure]],
    exp_data: dict,
    granularity: int,
    label_unc: dict,
    le: dict,
    N_sim: int = 1,
    mask_dict: dict | None = None,
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
    random_rot_mode: Literal["2D", "3D"] | None = "2D",
    asynch: bool = True,
    savedir: str = "",
    callback: lib.ProgressDialog | Literal["console"] | None = None,
) -> tuple[float, int, dict, StructureMixer, np.ndarray]:
    """Compare different models, i.e., ``StructureMixer``'s with label
    uncertainties given the experimental dataset and
    stoichiometries-search-space.

    Parameters
    ----------
    models : list of lists of SingleStructures
        Each element contains the list of SingleStructures that form
        a SPINNA model.
    exp_data : dict
        Dictionary with molecular targets' names as keys and spatial
        coordinates of the observed molecules as values.
    granularity : int
        Granularity as in spinna.generate_N_structures.
    label_unc : dict
        Dictionary specifying the label uncertainty for each molecular
        target species. Keys specify the species and values are lists
        of floats.
    le : dict
        Dictionary specifying the labeling efficiency for each
        molecular target species. Keys specify the species and values
        are floats.
    N_sim : int, optional
        Number of times each simulation is repeated to obtain smoother
        NND histograms. Default is 1.
    mask_dict : dict, optional
        Dictionary of the form {"mask": mask, "info": mask_info}, where
        mask and mask_info are defined as in StructureMixer. Only used
        when masking is used in simulations. Default is None
    width, height, depth: float, optional
        Width, height and depth of the simulated ROI in nm. If mask is
        provided, ROI is overwritten using the mask metadata. If width,
        height and depth are None, mask_dict must be specified. If
        width and height are given, but no mask and depth is provided,
        2D CSR simulation will be conducted. If width, height and depth
        are given but no mask is provided, 3D CSR simulation will be
        conducted. Default is None
    random_rot_mode : {"2D", "3D"} or None, optional
        Mode of random rotation of structures. If "3D", structures are
        rotated randomly in 3D. If "2D", structures are rotated
        randomly in 2D about z axis. If None, structures are not
        rotated. Default is "2D".
    asynch : bool, optional
        If True, multiprocessing is used for fitting. Else, single
        thread is used. Default is True.
    savedir : str, optional
        Path to the directory where the fitting scores are saved as
        .csv files. If "" is given, the files are not saved. Default
        is "".
    callback : {lib.ProgressDialog, "console", None}, optional
        Progress bar to track fitting progress. If "console", the
        progress bar is displayed in the console. If None, no
        progress bar is displayed. Default is None.

    Returns
    -------
    best_score : float
        The best fitting score.
    best_idx : int
        Index of the best fitting model in the models list.
    label_unc : dict
        The best fitting label uncertainties for each molecular target
        species.
    best_mixer : StructureMixer
        The best fitting StructureMixer.
    best_props : np.ndarray
        The stoichiometry of structures that gives the best fit to the
        data.
    """
    # extract the molecular targets from the models
    targets = []
    for structure in models[0]:
        for target in structure.targets:
            if target not in targets:
                targets.append(target)

    nn_counts = {}
    for ii, target1 in enumerate(targets):
        for target2 in targets[ii:]:
            nn_counts[f"{target1}-{target2}"] = 0

    # used for the fitting of label uncertainties, each target must be
    # specified
    label_unc_input_ = {target: lunc[0] for target, lunc in label_unc.items()}

    # First we fit label_unc for each target, where we only focus on the
    # structures that contain the target. The fitting can be skipped if
    # label_unc is already provided without the search space.
    for target in targets:
        best_score = np.inf
        best_l_unc = 5.0
        l_unc = label_unc[target]
        if len(l_unc) == 1:  # no search space for label uncertainty
            label_unc[target] = l_unc[0]
            continue

        # extract the models that contain the target only
        target_models = []
        for model in models:
            target_structures = []
            for structure in model:
                if [target] == structure.targets:
                    target_structures.append(structure)
            target_models.append(target_structures)

        # specify nn counts to be considered for fitting (only the
        # target to itself, 1st NN, the rest is ignored)
        nn_counts = {key: 0 for key in nn_counts.keys()}
        nn_counts[f"{target}-{target}"] = 1

        # test the range of label uncertainties for the target
        for l_unc_ in l_unc:
            progress_title = (
                f"Spinning with label uncertainty {l_unc_:.2f} nm for {target}"
            )
            label_unc_input = deepcopy(label_unc_input_)
            label_unc_input[target] = l_unc_
            score = compare_models_given_label_unc(
                models=target_models,
                exp_data=exp_data,
                granularity=granularity,
                label_unc=label_unc_input,
                le=le,
                mask_dict=mask_dict,
                width=width,
                height=height,
                depth=depth,
                random_rot_mode=random_rot_mode,
                nn_counts=nn_counts,
                N_sim=N_sim,
                asynch=asynch,
                savedir=savedir,
                callback=callback,
                progress_title=progress_title,
            )[0]
            if score < best_score:
                best_score = score
                best_l_unc = l_unc_

        # save the best fitting label uncertainty for the given target
        label_unc[target] = best_l_unc

    # test the models with the best fitting label uncertainties; note
    # that here we'd like to pay the attention to the NNDs that are
    # present in all models, i.e., if a simpler model does not contain
    # a structure with a dimer of a certain species, it should still aim
    # to fit the "dimer" NNDs.
    for ii, target1 in enumerate(targets):
        for target2 in targets[ii:]:
            key = f"{target1}-{target2}"
            for model in models:
                for structure in model:
                    max_nn_count = structure.get_max_nn(target1, target2)
                    nn_counts[key] = max(nn_counts[key], max_nn_count)

    # compare the models
    progress_title = f"Spinning with label uncertainties: {label_unc}"
    (
        best_score, best_idx, best_mixer, best_props
    ) = compare_models_given_label_unc(
        models=models,
        exp_data=exp_data,
        granularity=granularity,
        label_unc=label_unc,
        le=le,
        mask_dict=mask_dict,
        width=width,
        height=height,
        depth=depth,
        random_rot_mode=random_rot_mode,
        nn_counts=nn_counts,
        N_sim=N_sim,
        asynch=asynch,
        savedir=savedir,
        callback=callback,
        progress_title=progress_title,
    )
    return best_score, best_idx, label_unc, best_mixer, best_props


def compare_models_given_label_unc(
    models: list[list[Structure]],
    exp_data: dict,
    granularity: int,
    label_unc: dict,
    le: dict,
    mask_dict: dict | None = None,
    width: float | None = None,
    height: float | None = None,
    depth: float | None = None,
    random_rot_mode: Literal["2D", "3D"] | None = "2D",
    nn_counts: dict | str = "auto",
    N_sim: int = 1,
    asynch: bool = True,
    savedir: str = "",
    callback: lib.ProgressDialog | Literal["console"] | None = None,
    progress_title: str = "Spinning structures",
) -> tuple[float, int, StructureMixer, np.ndarray]:
    """Compare different models, i.e., ``StructureMixer``'s given the
    experimental dataset, stoichiometries-search-space and label
    position uncertainty.

    Parameters
    ----------
    models : list of lists of SingleStructures
        Each element contains the list of SingleStructures that form
        a SPINNA model.
    exp_data : dict
        Dictionary with molecular targets' names as keys and spatial
        coordinates of the observed molecules as values.
    granularity : int
        Granularity as in spinna.generate_N_structures.
    label_unc : dict
        Dictionary specifying the label uncertainty for each molecular
        target species. Keys specify the species and values are floats.
    le : dict
        Dictionary specifying the labeling efficiency for each
        molecular target species. Keys specify the species and values
        are floats.
    mask_dict : dict, optional
        Dictionary of the form {"mask": mask, "info": mask_info}, where
        mask and mask_info are defined as in StructureMixer. Only used
        when masking is used in simulations. Default is None.
    width, height, depth: float, optional
        Width, height and depth of the simulated ROI in nm. If mask is
        provided, ROI is overwritten using the mask metadata. If width,
        height and depth are None, mask_dict must be specified. If
        width and height are given, but no mask and depth is provided,
        2D CSR simulation will be conducted. If width, height and depth
        are given but no mask is provided, 3D CSR simulation will be
        conducted. Default is None.
    random_rot_mode : {"2D", "3D", None}, optional
        Mode of random rotation of structures. If "3D", structures are
        rotated randomly in 3D. If "2D", structures are rotated
        randomly in 2D about z axis. If None, structures are not
        rotated. Default is "2D".
    nn_counts : {dict, "auto"}, optional
        Dictionary specifying the number of nearest neighbors that two
        molecular target species form in the structures. Keys are
        strings of the form "target1-target2" and values are integers.
        If "auto", the number of neighbors is determined automatically
        from the structures. If None, no nearest neighbors are
        considered. Default is "auto".
    N_sim : int, optional
        Number of times each simulation is repeated to obtain smoother
        NND histograms. Default is 1.
    asynch : bool, optional
        If True, multiprocessing is used for fitting. Else, single
        thread is used. Default is True.
    savedir : str, optional
        Path to the directory where the fitting scores are saved as
        .csv files. If "" is given, the files are not saved. Default
        is "".
    callback : {lib.ProgressDialog, "console", None}, optional
        Progress bar to track fitting progress. If "console", the
        progress bar is displayed in the console. If None, no
        progress bar is displayed. Default is None.
    progress_title : str, optional
        Title of the progress bar displayed during fitting simulations.
        Default is "Spinning structures".

    Returns
    -------
    best_score : float
        Best fitting score (KS2).
    best_idx : int
        Index of the best fitting model in the models list.
    best_mixer : StructureMixer
        The best fitting StructureMixer.
    best_props : np.ndarray
        The stoichiometry of structures that gives the best fit to the
        data.
    """
    # initialize
    best_mixer = None
    best_idx = None
    best_score = np.inf
    N_total = {
        target: int(len(exp_data[target]) / le[target])
        for target in exp_data.keys()
    }

    for i, model in enumerate(models):
        search_space = generate_N_structures(model, N_total, granularity)
        mixer = StructureMixer(
            structures=model,
            label_unc=label_unc,
            le=le,
            mask_dict=mask_dict,
            width=width,
            height=height,
            depth=depth,
            random_rot_mode=random_rot_mode,
            nn_counts=nn_counts,
        )
        spinner = SPINNA(
            mixer=mixer,
            gt_coords=exp_data,
            N_sim=N_sim,
            progress_title=(
                f"{progress_title} and model nr {i+1}/{len(models)}"
            ),
        )
        savepath = ""
        if savedir:
            suffix = ("_").join([
                f"{target}_{lunc:2f}_nm" for target, lunc in label_unc.items()
            ])
            savepath = os.path.join(
                savedir, f"fit_scores_model_{i+1}_label_unc_{suffix}.csv"
            )
        # adjust the progress dialog
        if isinstance(callback, lib.ProgressDialog):
            callback.setMaximum(len(list(search_space.values())[0]))
        elif callback == "console":
            print(f"Model {i+1}/{len(models)}")
        opt_props, score = spinner.fit_stoichiometry(
            N_structures=search_space,
            save=savepath,
            asynch=asynch,
            callback=callback,
        )
        if score < best_score:
            best_score = score
            best_mixer = mixer
            best_idx = i
            best_props = opt_props
    return best_score, best_idx, best_mixer, best_props


def check_structures_valid_for_fitting(structures: list[Structure]) -> bool:
    """Check if the structures loaded can be used for finding LE.

    This means:
        * 2 molecular target species loaded.
        * The structures loaded are: monomer A, monomer B,
            heterodimer.

    Returns
    -------
    bool
        True if structures loaded can be used for finding LE.
    """
    targets = list(set([structure.targets[0] for structure in structures]))
    # 2 targets are present
    if len(targets) != 2:
        return False

    # 3 structures are present
    if len(structures) != 3:
        return False

    # search through structures - monomers and heterodimer
    flag_le_structures = {"A": False, "B": False, "AB": False}
    target_a = targets[0]
    target_b = targets[1]
    for structure in structures:
        # monomer A?
        if (
            len(structure.targets) == 1
            and structure.targets[0] == target_a
            and len(structure.x[target_a]) == 1
        ):
            flag_le_structures["A"] = True
        # monomer B?
        if (
            len(structure.targets) == 1
            and structure.targets[0] == target_b
            and len(structure.x[target_b]) == 1
        ):
            flag_le_structures["B"] = True
        # heterodimer?
        if (
            len(structure.targets) == 2
            and target_a in structure.targets
            and target_b in structure.targets
            and len(structure.x[target_a]) == 1
            and len(structure.x[target_b]) == 1
        ):
            flag_le_structures["AB"] = True
    return all(flag_le_structures.values())


def get_le_from_props(
    structures: list[Structure],
    opt_props: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> dict:
    """Based on the fitted proportions of structures, extract the
    LE values.

    Parameters
    ----------
    structures : list of Structure
        List of the structures used for fitting.
    opt_props : np.ndarray or tuple
        Fitted proportions of the structures. If bootstraping was used,
        the tuple is accepted and only the mean value is used.

    Returns
    -------
    le_values : dict
        Dictionary with LE values for the two targets.

    Raises
    ------
    ValueError
        If the structures are not valid for fitting. See
        ``check_structures_valid_for_fitting`` for details.
    """
    if not check_structures_valid_for_fitting(structures):
        raise ValueError("Invalid structures for fitting.")

    targets = list(set([structure.targets[0] for structure in structures]))
    props_ = {}
    target_a = targets[0]
    target_b = targets[1]
    if isinstance(opt_props, tuple):
        opt_props = opt_props[0]
    for idx, structure in enumerate(structures):
        # monomer A?
        if (
            len(structure.targets) == 1
            and structure.targets[0] == target_a
            and len(structure.x[target_a]) == 1
        ):
            props_["A"] = opt_props[idx]
        # monomer B?
        if (
            len(structure.targets) == 1
            and structure.targets[0] == target_b
            and len(structure.x[target_b]) == 1
        ):
            props_["B"] = opt_props[idx]
        # heterodimer?
        if (
            len(structure.targets) == 2
            and target_a in structure.targets
            and target_b in structure.targets
            and len(structure.x[target_a]) == 1
            and len(structure.x[target_b]) == 1
        ):
            props_["AB"] = opt_props[idx]
    # we need the proportion of structures, not molecules in
    # structures
    props_["AB"] = props_["AB"] / 2
    le_values = {
        target_a: props_["AB"] / (props_["B"] + props_["AB"]) * 100,
        target_b: props_["AB"] / (props_["A"] + props_["AB"]) * 100,
    }
    return le_values
