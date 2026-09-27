"""
picasso.g5m
~~~~~~~~~~~

Gaussian Mixture Modeling with Modifications for Molecular Mapping
(G5M). Published in: Kowalewski, Reinhardt, et al. Nature Comms, 2026.
DOI: https://doi.org/10.1038/s41467-026-70198-5.

G5M is based on the sklearn implementation of Gaussian Mixture Modeling
(GMM) with numba optimizations for fitting, as well as for kmeans++
initialization. Several modifications for molecular mapping in DNA-PAINT
are added, for example, localization cloud shape modeling.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2023-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import chain as itchain
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import erf
from sklearn.utils import check_random_state

from . import lib, zfit, __version__

if TYPE_CHECKING:
    from PyQt6 import QtWidgets  # only used in type annotations

# default min. number of localizations per molecule
MIN_LOCS = 10
# default number of rounds without BIC improvement to terminate the
# search for n_components
MAX_ROUNDS_WITHOUT_BEST_BIC = 3
# default min. sigma factor for each G5M component
# (min_sigma = MIN_SIGMA_FACTOR * loc_prec)
MIN_SIGMA_FACTOR = 0.8
# default max. sigma factor for each G5M component
# (max_sigma = MAX_SIGMA_FACTOR * loc_prec)
MAX_SIGMA_FACTOR = 1.5
# default number of tasks for parallel processing
N_TASKS = 500
# to avoid spending eternity on fitting too large clusters
N_COMPONENTS_MAX = 100
# available shapes of the G5M components: "spherical" (isotropic, 2D
# only), "diagonal" (axis-aligned, 3D only) and "rotated" (like diagonal
# but with xy-plane rotation)
COVARIANCE_TYPES = ("spherical", "diagonal", "rotated")


# helper functions for numba operations along axes
fastmath = True


@njit(fastmath=fastmath)
def _max_along_axis1(
    X: lib.FloatArray2D, final_shape: tuple[int]
) -> lib.FloatArray1D:
    output = np.zeros(final_shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        output[i] = np.max(X[i])
    return output


@njit(fastmath=fastmath)
def _sum_along_axis0(
    X: lib.FloatArray2D, final_shape: tuple[int]
) -> lib.FloatArray1D:
    output = np.zeros(final_shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        output += X[i]
    return output


@njit(fastmath=fastmath)
def _sum_along_axis1(
    X: lib.FloatArray2D, final_shape: tuple[int]
) -> lib.FloatArray1D:
    output = np.zeros(final_shape, dtype=X.dtype)
    for i in range(X.shape[1]):
        output += X[:, i]
    return output


@njit(fastmath=fastmath)
def _mean_along_axis1(
    X: lib.FloatArray2D, final_shape: tuple[int]
) -> lib.FloatArray1D:
    output = _sum_along_axis1(X, final_shape)
    return output / X.shape[1]


@njit(fastmath=fastmath)
def _logsumexp_axis1(
    X: lib.FloatArray2D, final_shape: tuple[int]
) -> lib.FloatArray1D:
    """njit implementation of ``scipy.special.logsumexp``. Note that we
    cannot use ``np.log(np.sum(np.exp(X), axis=1))`` because it will
    cause overflow for large numbers. Thus, we use the ``logsumexp``
    formula: ``log(sum(exp(X))) = log(sum(exp(X - max(X))) + max(X)``
    where ``max(X)`` is subtracted from X to avoid overflow."""
    max_val = _max_along_axis1(X, final_shape)
    exp_values = np.exp(X - max_val[:, np.newaxis])
    exp_sum = _sum_along_axis1(exp_values, final_shape)
    output = np.log(exp_sum) + max_val
    return output


@njit(fastmath=fastmath)
def _matmul(a: lib.FloatArray2D, b: lib.FloatArray2D) -> lib.FloatArray2D:
    """Matrix multiplication, assuming that the shapes are
    compatible."""
    n, m = a.shape
    m, p = b.shape
    c = np.zeros((n, p), dtype=a.dtype)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                c[i, j] += a[i, k] * b[k, j]
    return c


@njit(fastmath=fastmath)
def _square_elements_1d(X: lib.FloatArray1D) -> lib.FloatArray1D:
    output = np.zeros(X.shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        output[i] = X[i] ** 2
    return output


@njit(fastmath=fastmath)
def _square_elements_2d(X: lib.FloatArray2D) -> lib.FloatArray2D:
    m, n = X.shape
    output = np.zeros((m, n), dtype=X.dtype)
    for i in range(m):
        for j in range(n):
            output[i, j] = X[i, j] ** 2
    return output


@njit(fastmath=fastmath)
def _poly1d(
    coeffs: lib.FloatArray1D, xs: lib.FloatArray1D
) -> lib.FloatArray1D:
    """Use Horner's method to evaluate a polynomial with coefficients
    `coeffs` at points `xs`. Coefficients are in the form [a_n, a_{n-1},
    ..., a_0] for the polynomial a_n*x^n + a_{n-1}*x^{n-1} + ... + a_0.
    """
    out = np.empty(len(xs))
    for i in range(len(xs)):
        result = 0.0
        x = xs[i]
        for c in coeffs:
            result = result * x + c
        out[i] = result
    return out


@njit(fastmath=fastmath)
def _circular_weighted_mean_angle(
    resp: lib.FloatArray2D,  # shape (n_samples, n_components)
    angles: lib.FloatArray1D,  # shape (n_samples,), radians
) -> lib.FloatArray1D:
    """Responsibility-weighted mean of axial angles, per component.

    The rotated elliptical PSF angle is *axial*, i.e. it has a period of
    pi (an ellipse at +89 deg and one at -89 deg point in almost the same
    direction). A plain weighted mean of such data is wrong: it would
    average +89 and -89 to 0, which is perpendicular to the truth.
    Averaging the doubled angle and halving the result avoids this.

    Parameters
    ----------
    resp : np.ndarray
        Responsibilities of the G5M components, shape (n_samples,
        n_components).
    angles : np.ndarray
        Angle of each localization in radians, shape (n_samples,).

    Returns
    -------
    theta : np.ndarray
        Mean angle per component in radians, shape (n_components,).
    """
    n_components = resp.shape[1]
    theta = np.zeros(n_components, dtype=np.float64)
    for k in range(n_components):
        sin_sum = 0.0
        cos_sum = 0.0
        for i in range(resp.shape[0]):
            sin_sum += resp[i, k] * np.sin(2.0 * angles[i])
            cos_sum += resp[i, k] * np.cos(2.0 * angles[i])
        # the (identical) weight normalization cancels in arctan2
        theta[k] = 0.5 * np.arctan2(sin_sum, cos_sum)
    return theta


@njit(fastmath=fastmath)
def _assemble_covs_3D_rot(
    cov_maj: lib.FloatArray1D,
    cov_min: lib.FloatArray1D,
    cov_z: lib.FloatArray1D,
    theta: lib.FloatArray1D,
) -> lib.FloatArray3D:
    """Build block covariance matrices from principal-axis variances.

    Each component gets ``blockdiag(R(theta) @ diag(cov_maj, cov_min) @
    R(theta).T, cov_z)``, i.e. a full 2x2 xy block rotated by ``theta``
    and an independent z variance.

    Parameters
    ----------
    cov_maj, cov_min : np.ndarray
        Variances along the major and minor principal axes in xy, shape
        (n_components,).
    cov_z : np.ndarray
        Variances in z, shape (n_components,).
    theta : np.ndarray
        Rotation angle of the major axis in radians, shape
        (n_components,).

    Returns
    -------
    covs : np.ndarray
        Covariance matrices, shape (n_components, 3, 3).
    """
    n_components = len(cov_maj)
    covs = np.zeros((n_components, 3, 3), dtype=np.float64)
    for k in range(n_components):
        ct = np.cos(theta[k])
        st = np.sin(theta[k])
        covs[k, 0, 0] = cov_maj[k] * ct * ct + cov_min[k] * st * st
        covs[k, 1, 1] = cov_maj[k] * st * st + cov_min[k] * ct * ct
        covs[k, 0, 1] = (cov_maj[k] - cov_min[k]) * ct * st
        covs[k, 1, 0] = covs[k, 0, 1]
        covs[k, 2, 2] = cov_z[k]
    return covs


@njit(fastmath=fastmath)
def _precision_chol_3D_rot(covs: lib.FloatArray3D) -> lib.FloatArray3D:
    """Cholesky decomposition of the precision matrices, for the block
    covariance produced by ``_assemble_covs_3D_rot``.

    Follows sklearn's convention for full covariances: the returned C is
    upper triangular and satisfies ``C @ C.T == inv(covs)``. For a 2x2
    xy block ``[[a, b], [b, c]]`` with ``d = a*c - b**2`` this has the
    closed form used below, so no matrix routines are needed inside njit.
    Note that for ``b == 0`` it reduces to the plain ``1 / sqrt(cov)``
    used by the diagonal 3D model.

    Parameters
    ----------
    covs : np.ndarray
        Covariance matrices, shape (n_components, 3, 3).

    Returns
    -------
    precisions_chol : np.ndarray
        Shape (n_components, 3, 3), upper triangular per component.
    """
    n_components = covs.shape[0]
    precisions_chol = np.zeros((n_components, 3, 3), dtype=np.float64)
    for k in range(n_components):
        a = covs[k, 0, 0]
        b = covs[k, 0, 1]
        c = covs[k, 1, 1]
        d = a * c - b * b  # det of the xy block; > 0, see _m_step_3D_rot
        precisions_chol[k, 0, 0] = 1.0 / np.sqrt(a)
        precisions_chol[k, 0, 1] = -b / np.sqrt(a * d)
        precisions_chol[k, 1, 1] = np.sqrt(a / d)
        precisions_chol[k, 2, 2] = 1.0 / np.sqrt(covs[k, 2, 2])
    return precisions_chol


# In sklearn's GaussianMixture implementation, the term in the nominator
# of the exponential term ((x - mu)^2 / sigma^2) is calculated as
# (x^2 - 2*x*mu + mu^2). This can cause numerical instability when x and
# mu are large, which is the case for our data. To avoid this, we first
# calculate (x - mu) and then square it (x and mu values are similar,
# thus the instability is avoided). Note: precision = 1/sigma**2
@njit(fastmath=fastmath)
def _gauss_exponential_term_2D(
    X: lib.FloatArray2D,  # shape (n_samples, 2)
    means: lib.FloatArray2D,  # shape (n_components, 2)
    precision: lib.FloatArray1D,  # shape (n_components,)
) -> lib.FloatArray2D:
    n_samples = X.shape[0]
    n_components = means.shape[0]
    sq_diff = np.zeros((n_samples, n_components), dtype=X.dtype)
    for i in range(n_samples):
        for j in range(n_components):
            for k in range(2):
                sq_diff[i, j] += (X[i, k] - means[j, k]) ** 2 * precision[j]
    return sq_diff


@njit(fastmath=fastmath)
def _gauss_exponential_term_3D(
    X: lib.FloatArray2D,  # shape (n_samples, 3)
    means: lib.FloatArray2D,  # shape (n_components, 3)
    precision: lib.FloatArray2D,  # shape (n_components, 3)
) -> lib.FloatArray2D:
    """Same as ``_gauss_exponential_term_2D`` but precision has shape
    (K, 3), where K is the number of components."""
    n_samples = X.shape[0]
    n_components = means.shape[0]
    sq_diff = np.zeros((n_samples, n_components), dtype=X.dtype)
    for i in range(n_samples):
        for j in range(n_components):
            for k in range(3):
                sq_diff[i, j] += (X[i, k] - means[j, k]) ** 2 * precision[j, k]
    return sq_diff


@njit(fastmath=fastmath)
def _gauss_exponential_term_3D_rot(
    X: lib.FloatArray2D,  # shape (n_samples, 3)
    means: lib.FloatArray2D,  # shape (n_components, 3)
    precisions_chol: lib.FloatArray3D,  # shape (n_components, 3, 3)
) -> lib.FloatArray2D:
    """Same as ``_gauss_exponential_term_3D`` but for the rotated xy
    block model, where the precision is stored as its (upper triangular)
    Cholesky factor C with ``C @ C.T = inv(cov)``.

    Computes ``|(x - mu) @ C|^2 = (x - mu) inv(cov) (x - mu).T``. As in
    the other kernels, the difference is taken before squaring, which
    is what keeps the result stable for large coordinates."""
    n_samples = X.shape[0]
    n_components = means.shape[0]
    sq_diff = np.zeros((n_samples, n_components), dtype=X.dtype)
    for i in range(n_samples):
        for j in range(n_components):
            dx = X[i, 0] - means[j, 0]
            dy = X[i, 1] - means[j, 1]
            dz = X[i, 2] - means[j, 2]
            y0 = dx * precisions_chol[j, 0, 0]
            y1 = dx * precisions_chol[j, 0, 1] + dy * precisions_chol[j, 1, 1]
            y2 = dz * precisions_chol[j, 2, 2]
            sq_diff[i, j] = y0 * y0 + y1 * y1 + y2 * y2
    return sq_diff


# kmeans++ init, adopted from sklearn, numba implementation #
@njit
def _euclidean_distances(
    X: lib.FloatArray2D,
    Y: lib.FloatArray2D,
    X_norm_squared: lib.FloatArray2D | None = None,
    Y_norm_squared: lib.FloatArray2D | None = None,
) -> lib.FloatArray2D:
    """njit implementation of
    ``sklearn.metrics.pairwise._euclidean_distances`` with
    ``squared=True``."""
    if X_norm_squared is not None:
        XX = X_norm_squared.reshape(-1, 1)
    else:
        XX = _sum_along_axis1(_square_elements_2d(X), (X.shape[0],))[
            :, np.newaxis
        ]

    if Y is X:
        YY = None if XX is None else XX.T
    else:
        if Y_norm_squared is not None:
            YY = Y_norm_squared.reshape(1, -1)
        else:
            YY = _sum_along_axis1(_square_elements_2d(Y), (Y.shape[0],))[
                :, np.newaxis
            ]

    distances = -2 * _matmul(X, Y.T)
    distances += XX
    distances += YY
    distances = np.maximum(distances, 0)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances  # squared distances


@njit
def _kmeans_plusplus(
    X: lib.FloatArray2D,
    n_components: int,
    random_state: int,
) -> lib.IntArray1D:
    """njit implementation of ``sklearn.cluster._kmeans_plusplus``. Used
    for initializing ``G5M``'s."""
    np.random.seed(random_state)

    n_samples, n_dimensions = X.shape
    centers = np.empty((n_components, n_dimensions), dtype=X.dtype)
    n_local_trials = 2 + int(np.log(n_components))
    x_squared_norms = _sum_along_axis1(_square_elements_2d(X), (X.shape[0],))

    # Pick first center randomly and track index of point
    center_id = np.random.choice(n_samples)
    indices = np.full(n_components, -1, dtype=np.int64)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms
    ).flatten()
    current_pot = np.sum(closest_dist_sq)

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_components):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.empty(n_local_trials, dtype=X.dtype)
        for i in range(n_local_trials):
            rand_vals[i] = np.random.uniform() * current_pot
        candidate_ids = np.searchsorted(
            np.cumsum(closest_dist_sq.flatten()), rand_vals
        )
        # numerical imprecision can result in a candidate_id out of range
        max_value = closest_dist_sq.size - 1
        for i in range(len(candidate_ids)):
            if candidate_ids[i] > max_value:
                candidate_ids[i] = max_value

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms
        )

        # update closest distances squared and potential for each candidate
        distance_to_candidates = np.minimum(
            closest_dist_sq, distance_to_candidates
        )
        candidates_pot = _sum_along_axis1(
            distance_to_candidates, (distance_to_candidates.shape[0],)
        )

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return indices


# G5M abstract class #
class G5M(metaclass=ABCMeta):
    """Parent class for G5M in 2D and 3D with numba implementations of
    initialization and fitting. Based on the implementation of sklearn.

    ...

    Attributes
    ----------
    calibration : dict
        Calibration dictionary with x and y coefficients and
        magnification factor. Required for 3D data only. See
        https://picassosr.readthedocs.io/en/latest/localize.html#d-calibration.
    converged : bool
        True if the G5M converged, False otherwise.
    covariances_ : np.ndarray
        Covariances of the G5M components, shape (n_components,).
    covariances : np.ndarray
        Same as covariances_ but only valid components (based on
        min_locs) are indexed.
    loc_prec_handle : {"local", "abs"}
        How to handle sigma bounds. If "local", localization
        precisions of points around each component are used to bound
        sigmas. Else, sigma_bounds specifies the absolute bounds on
        sigmas.
    means_init : np.ndarray, optional
        Initial means of the G5M components. If None, the means are
        initialized using kmeans++. Default is None.
    means_ : np.ndarray
        Means of the G5M components, shape (n_components, n_dimensions).
    means : np.ndarray
        Same as means_ but only valid components (based on min_locs) are
        indexed.
    min_locs : int
        Minimum number of localizations per component. Used to filter
        out components with too few localizations that likely represent
        background/noise.
    n_components : int
        Number of components in the G5M (may include invalid components
        that were rejected due to low localization count).
    n_dimensions : int
        Number of dimensions in the data.
    n_init : int
        Number of initializations.
    n_locs : np.ndarray
        Number of localizations per component (applied after fitting).
    precisions_cholesky_ : np.ndarray
        Cholesky decomposition of the precision matrices of the G5M
        components, shape (n_components, n_dimensions).
    precisions_cholesky : np.ndarray
        Same as precisions_cholesky_ but only valid components (based
        on min_locs) are indexed.
    random_state : int
        Random seed for reproducibility.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma.
    valid_idx : np.ndarray
        Indices of valid components (based on min_locs), applied after
        fitting. Its length gives the number of valid components.
    weights_ : np.ndarray
        Weights of the G5M components, shape (n_components,).
    weights : np.ndarray
        Same as weights_ but only valid components (based on min_locs)
        are indexed. Renormalized to sum to 1.

    Parameters
    ----------
    n_components : int
        Number of components in the model.
    min_locs : int
        Minimum number of localizations per component.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma.
    covariance_type : {"spherical", "diagonal", "rotated"}
        Covariance model of the Gaussian components (keyword-only).
    means_init : np.ndarray or None, optional
        Initial means (mu) of the Gaussian components. If None, the
        means are initialized using kmeans++.
    """

    def __init__(
        self,
        n_components: int,
        min_locs: int,
        sigma_bounds: tuple[float, float],
        *,
        covariance_type: Literal["spherical", "diagonal", "rotated"],
        means_init: np.ndarray | None = None,
    ) -> None:

        assert sigma_bounds[0] >= 0.0 and sigma_bounds[1] >= 0.0
        assert sigma_bounds[1] >= sigma_bounds[0]
        assert covariance_type in COVARIANCE_TYPES, (
            f"covariance_type must be one of {COVARIANCE_TYPES}, got "
            f"'{covariance_type}'."
        )

        self.n_components = int(n_components)
        self.min_locs = int(min_locs)
        self.sigma_bounds = sigma_bounds
        self.covariance_type = covariance_type
        self.n_init = max(int(n_components), 3)
        self.random_state = 42
        self.converged = False
        self.means_init = means_init
        self.loc_prec_handle = "local"

        # for 3D compatibility
        self.calibration = None

        # indices for valid components (based on min_locs), applied
        # after fitting
        self.valid_idx = np.arange(n_components).astype(int)
        # number of locs per component (applied after fitting)
        self.n_locs = np.zeros(n_components, dtype=int)

    def bic(self, X: lib.FloatArray2D) -> float:
        """Bayesian Information Criterion (BIC) for the G5M.

        Parameters
        ----------
        X : lib.FloatArray2D
            ``(n_locs, n_dim)`` localization coordinates.

        Returns
        -------
        bic : float
            The lower, the better the model explains ``X`` for its number of
            parameters.
        """
        # shift coordinates by their mean (numerical stability)
        bic = (
            self.n_parameters() * np.log(X.shape[0])
            - 2 * self.score_samples(X).mean() * X.shape[0]
        )
        return bic

    @property
    def covariances(self) -> np.ndarray:
        """Valid covariance."""
        return self.covariances_[self.valid_idx]

    @abstractmethod
    def estimate_log_prob(self, X: lib.FloatArray2D) -> lib.FloatArray2D:
        """Calculate the log probabilities of the data X under the G5M,
        without weights.

        Parameters
        ----------
        X : lib.FloatArray2D
            ``(n_locs, n_dim)`` localization coordinates.

        Returns
        -------
        log_prob : lib.FloatArray2D
            ``(n_locs, n_valid_components)`` log probabilities.
        """
        pass

    def estimate_weighted_log_prob(
        self, X: lib.FloatArray2D
    ) -> lib.FloatArray2D:
        """Calculate the log probabilities of the data X under the G5M,
        with weights.

        Parameters
        ----------
        X : lib.FloatArray2D
            ``(n_locs, n_dim)`` localization coordinates.

        Returns
        -------
        log_prob : lib.FloatArray2D
            ``(n_locs, n_valid_components)`` weighted log probabilities.
        """
        return self.estimate_log_prob(X) + np.log(self.weights)

    def fit(
        self,
        X: lib.FloatArray2D,
        lp: lib.FloatArray1D | lib.FloatArray2D,
        loc_prec_handle: Literal["local", "abs"] = "local",
        angles: lib.FloatArray1D | None = None,
    ) -> G5M | None:
        """Fit G5M to data X. Return None if fitting failed.

        Parameters
        ----------
        X : np.ndarray
            Data points, shape (n_samples, n_dimensions).
        lp : np.ndarray
            Localization precision for each localization. Only used if
            loc_prec_handle is "local". Shape (n_samples,) for 2D and
            (n_samples, 3) for 3D.
        loc_prec_handle : {"local", "abs"}, optional
            How to handle sigma bounds. If "local", localization
            precisions of points around each component are used to bound
            sigmas. Else, sigma_bounds specifies the absolute bounds on
            sigmas. Default is "local".
        angles : np.ndarray or None, optional
            Angle of each localization in radians, shape (n_samples,).
            Required for ``covariance_type="rotated"``, ignored
            otherwise. Default is None.

        Returns
        -------
        self : G5M
            Fitted model.
        """
        assert X.shape[1] == self.n_dimensions, (
            "The number of dimensions in X must match the number of "
            f"dimensions in the G5M class ({self.n_dimensions})."
        )

        X = np.ascontiguousarray(np.float64(X))
        lp = np.ascontiguousarray(np.float64(lp))
        self.n_samples = X.shape[0]
        self.loc_prec_handle = loc_prec_handle

        if angles is None:
            # empty float sentinel; unused unless the covariance type is
            # "rotated", but required so the njit m-step's angles stays
            # typed as a float array
            angles = np.array([])
        angles = np.ascontiguousarray(np.float64(angles))

        if self.n_dimensions == 2:
            initialize_G5M = _initialize_G5M_2D
        elif self.n_dimensions == 3:
            if self.covariance_type == "rotated":
                initialize_G5M = _initialize_G5M_3D_rot
            else:
                initialize_G5M = _initialize_G5M_3D
        else:
            raise ValueError("Only 2D and 3D data are supported.")

        init_weights, init_means, init_precisions_cholesky = initialize_G5M(
            X,
            self.n_init,
            self.n_components,
            self.random_state,
        )

        if self.means_init is not None:
            init_means = np.tile(self.means_init, (self.n_init, 1, 1))

        if self.calibration is None:
            cx = np.array([])
            cy = np.array([])
            # float sentinel; unused when cx/cy are empty, but required
            # so the njit m-step's mag_factor stays typed as a float
            mag_factor = 0.79
        else:
            # convert to numpy arrays so the njit m-step can use .size
            # (calibration coefficients come from YAML as Python lists)
            cx = np.asarray(
                self.calibration["X Coefficients"], dtype=np.float64
            )
            cy = np.asarray(
                self.calibration["Y Coefficients"], dtype=np.float64
            )
            mag_factor = self.calibration["Magnification factor"]
        (w, m, c, pc), converged, valid_idx = _fit_G5M(
            X,
            min_locs=self.min_locs,
            init_weights=init_weights,
            init_means=init_means,
            init_precisions_cholesky=init_precisions_cholesky,
            sigma_bounds=self.sigma_bounds,
            lp=lp,
            loc_prec_handle=loc_prec_handle,
            cx=cx,
            cy=cy,
            mag_factor=mag_factor,
            angles=angles,
        )
        if w is None:
            return None
        self.set_parameters(w, m, c, pc, converged, valid_idx=valid_idx)

        # set valid indices based on number of localizations per
        # component
        n = np.round(w * len(X)).astype(int)  # N_locs per component
        self.n_locs = n[self.valid_idx]
        return self

    @property
    def means(self) -> np.ndarray:
        """Valid means."""
        return self.means_[self.valid_idx]

    @abstractmethod
    def n_parameters(self) -> int:
        """Return the number of parameters."""
        pass

    @property
    def precisions_cholesky(self) -> np.ndarray:
        """Valid precision."""
        return self.precisions_cholesky_[self.valid_idx]

    @property
    def sqrt_det_covariances(self) -> np.ndarray:
        """``sqrt(det(covariance))`` per valid component.

        Used for the analytic expected log-likelihood behind ``p_val``.
        """
        covariances = self.covariances
        if self.covariance_type == "spherical":
            # covariance is sigma**2, so sqrt(det) = sigma**2 exactly
            return covariances
        elif self.covariance_type == "diagonal":
            return np.sqrt(covariances).prod(1)
        # rotated: blockdiag(2x2 xy, z)
        det_xy = (
            covariances[:, 0, 0] * covariances[:, 1, 1]
            - covariances[:, 0, 1] ** 2
        )
        return np.sqrt(det_xy) * np.sqrt(covariances[:, 2, 2])

    def predict(self, X: lib.FloatArray2D) -> lib.IntArray1D:
        """Predict the cluster labels for the data X.

        Parameters
        ----------
        X : lib.FloatArray2D
            ``(n_locs, n_dim)`` localization coordinates.

        Returns
        -------
        labels : lib.IntArray1D
            Index of the most likely component for each localization.
        """
        return self.estimate_weighted_log_prob(X).argmax(axis=1)

    @abstractmethod
    def sample(
        self, n_samples: int = 1
    ) -> tuple[lib.FloatArray2D, lib.IntArray1D]:
        """Sample data points from the G5M.

        Parameters
        ----------
        n_samples : int, optional
            Number of points to draw. Default 1.

        Returns
        -------
        X : lib.FloatArray2D
            ``(n_samples, n_dim)`` sampled coordinates.
        y : lib.IntArray1D
            Index of the component each point was drawn from.
        """
        pass

    def set_parameters(
        self,
        weights: lib.FloatArray1D,
        means: lib.FloatArray2D,
        covs: lib.FloatArray1D | lib.FloatArray2D,
        precisions_cholesky: lib.FloatArray1D | lib.FloatArray2D,
        converged: bool,
        valid_idx: lib.IntArray1D | None = None,
    ) -> None:
        """Set the G5M parameters, used after fitting.

        Parameters
        ----------
        weights : lib.FloatArray1D
            Mixture weights; normalized to sum to 1.
        means : lib.FloatArray2D
            ``(n_components, n_dim)`` component centers.
        covs : lib.FloatArray1D or lib.FloatArray2D
            Component covariances, in the model's own layout.
        precisions_cholesky : lib.FloatArray1D or lib.FloatArray2D
            Cholesky factors of the precision matrices.
        converged : bool
            Whether the fit converged.
        valid_idx : lib.IntArray1D, optional
            Indices of the components that passed the ``min_locs`` filter.
            None keeps every component.
        """
        self.weights_ = weights / weights.sum()
        self.means_ = means
        self.covariances_ = covs
        self.precisions_cholesky_ = precisions_cholesky
        self.converged = converged
        if self.valid_idx is not None:
            self.valid_idx = valid_idx
        else:
            self.valid_idx = np.arange(len(weights))

    def score_samples(self, X: lib.FloatArray2D) -> lib.FloatArray1D:
        """Compute the log-likelihood of the data X under the G5M.

        Parameters
        ----------
        X : lib.FloatArray2D
            ``(n_locs, n_dim)`` localization coordinates.

        Returns
        -------
        log_likelihood : lib.FloatArray1D
            Per-localization log-likelihood.
        """
        weighted_log_prob = self.estimate_weighted_log_prob(X)
        final_shape = (weighted_log_prob.shape[0],)
        return _logsumexp_axis1(weighted_log_prob, final_shape)

    @property
    def weights(self) -> np.ndarray:
        """Valid weights."""
        w = self.weights_[self.valid_idx]
        # return w / w.sum()
        return w


# 2D G5M functions and classes #
@njit
def _check_G5M_resolution_2D(
    means: lib.FloatArray2D,
    weights: lib.FloatArray1D,
    precisions_chol: lib.FloatArray1D,
) -> bool:
    """Check if Sparrow limit is passed for all components of the
    ``G5M_2D``.

    Sparrow resolution limit is violated if there is not local minimum
    between two signals.

    Parameters
    ----------
    means : np.ndarray
        Means of the G5M components, shape (n_components, 2).
    weights : np.ndarray
        Weights of the G5M components, shape (n_components,).
    precisions_chol : np.ndarray
        Cholesky decomposition of the precision matrices of the G5M
        components, shape (n_components,).

    Returns
    -------
    bool
        True if the G5M components are well separated, False otherwise.
    """
    n_valid_components = means.shape[0]
    if n_valid_components == 0:  # if no component is valid
        return False
    elif n_valid_components == 1:
        return True

    # iterate over all pairs of components
    for i in range(n_valid_components):
        for j in range(i + 1, n_valid_components):
            # extract the parameters of the two components
            prec_chol_ = np.array([precisions_chol[i], precisions_chol[j]])
            weights_ = np.array([weights[i], weights[j]])
            means_ = np.zeros((2, 2), dtype=np.float64)
            means_[0, :] = means[i]
            means_[1, :] = means[j]

            # get the straight line between the two components
            direction_vector = means_[1, :] - means_[0, :]
            t = np.linspace(0, 1, 40)
            x = means_[0, 0] + direction_vector[0] * t
            y = means_[0, 1] + direction_vector[1] * t

            # get the PDF of all components along the line
            X = np.stack((x, y)).T
            ll = _estimate_log_gaussian_prob_2D(
                X, means_, prec_chol_
            ) + np.log(weights_)
            pdf = _sum_along_axis1(np.exp(ll), ll.shape[0])

            # find if there is at least one local minimum (may be more
            # if components in between align)
            if not len(lib.find_local_minima(pdf)):
                return False

    # if all components are well separated
    return True


@njit
def _initialize_G5M_2D(
    X: lib.FloatArray2D, n_init: int, n_components: int, random_state: int
) -> tuple[lib.FloatArray2D, lib.FloatArray3D, lib.FloatArray2D]:
    """Initialize the 2D G5M parameters using kmeans++."""
    n_samples = X.shape[0]
    init_weights = np.zeros((n_init, n_components), dtype=np.float64)
    init_means = np.zeros((n_init, n_components, 2), dtype=np.float64)
    init_precisions_cholesky = np.zeros(
        (n_init, n_components), dtype=np.float64
    )
    for ii in range(n_init):
        # initialize responsibilities using kmeans++ (e-step-like)
        resp = np.zeros((n_samples, n_components), dtype=np.float64)
        indices = _kmeans_plusplus(X, n_components, random_state)  # kmeans++

        for i in range(n_components):
            resp[indices[i], i] = 1

        # initialize G5M parameters (m-step-like)
        weights, means, covariances = _estimate_gaussian_parameters_2D(X, resp)
        weights /= n_samples
        init_weights[ii] = weights
        init_means[ii] = means
        init_precisions_cholesky[ii] = 1.0 / np.sqrt(covariances)

        random_state += 1

    return (
        np.asarray(init_weights, dtype=np.float64),
        np.asarray(init_means, dtype=np.float64),
        np.asarray(init_precisions_cholesky, dtype=np.float64),
    )


@njit
def _estimate_gaussian_parameters_2D(
    X: lib.FloatArray2D,
    resp: lib.FloatArray2D,
) -> tuple[lib.FloatArray1D, lib.FloatArray2D, lib.FloatArray1D]:
    nk, means, covariances = _estimate_gaussian_parameters_diag_cov(X, resp)
    covariances = _mean_along_axis1(covariances, final_shape=(len(nk),))
    return (
        np.asarray(nk, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(covariances, dtype=np.float64),
    )


@njit
def _estimate_log_gaussian_prob_2D(
    X: lib.FloatArray2D,
    means: lib.FloatArray2D,
    precisions_chol: lib.FloatArray1D,
) -> lib.FloatArray2D:
    log_det = 2 * np.log(precisions_chol)
    precisions = _square_elements_1d(precisions_chol)
    log_prob = _gauss_exponential_term_2D(X, means, precisions)
    return -0.5 * (2 * np.log(2 * np.pi) + log_prob) + log_det


@njit
def _e_step_2D(
    X: lib.FloatArray2D,
    weights: lib.FloatArray1D,
    means: lib.FloatArray2D,
    precisions_cholesky: lib.FloatArray1D,
) -> tuple[float, lib.FloatArray2D]:
    weighted_log_prob = _estimate_log_gaussian_prob_2D(
        X, means, precisions_cholesky
    ) + np.log(weights)
    log_prob_norm = _logsumexp_axis1(weighted_log_prob, (X.shape[0],))
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    return np.mean(log_prob_norm), log_resp.astype(np.float64)


@njit
def _m_step_2D(
    X: lib.FloatArray2D,
    log_resp: lib.FloatArray2D,
    sigma_bounds: tuple[float, float],
    lp: lib.FloatArray1D,
    loc_prec_handle: Literal["local", "abs"],
    cx: dict | None = None,  # for 3D consistency
    cy: dict | None = None,  # for 3D consistency
    mag_factor: float | None = None,  # for 3D consistency
    angles: lib.FloatArray1D = np.array([]),  # for rotated 3D consistency
) -> tuple[
    lib.FloatArray1D, lib.FloatArray2D, lib.FloatArray1D, lib.FloatArray1D
]:
    """2D m step. cx, cy, mag_factor and angles are not used and are here
    for compatibility with the 3D m steps."""
    min_cov = sigma_bounds[0] ** 2
    max_cov = sigma_bounds[1] ** 2
    resp = np.exp(log_resp)
    weights, means, covs = _estimate_gaussian_parameters_2D(X, resp)
    # clip covariances (numba does not support np.clip or multidim.
    # indexing)
    if loc_prec_handle == "local":  # local sigma bounds
        # take weighted (based on resp) avg. loc. prec. per component
        lp_ = np.reshape(lp, (-1, 1))
        mean_lp_per_component = _sum_along_axis0(
            resp * lp_, (resp.shape[1])
        ) / _sum_along_axis0(resp, (resp.shape[1],))
        mean_cov_per_component = _square_elements_1d(mean_lp_per_component)
        min_covs = min_cov * mean_cov_per_component
        max_covs = max_cov * mean_cov_per_component
    else:
        min_covs = np.full(len(weights), min_cov)
        max_covs = np.full(len(weights), max_cov)

    for i in range(len(weights)):
        if covs[i] < min_covs[i]:
            covs[i] = min_covs[i]
        elif covs[i] > max_covs[i]:
            covs[i] = max_covs[i]
    weights /= weights.sum()
    precisions_cholesky = 1.0 / np.sqrt(covs)

    return weights, means, covs, precisions_cholesky


def _find_optimal_G5M_2D(
    X: lib.FloatArray2D,
    min_locs: int,
    sigma_bounds: tuple[float, float],
    *,
    lp: lib.FloatArray1D,
    loc_prec_handle: Literal["local", "abs"] = "local",
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
) -> G5M_2D:
    """Find the optimal G5M for the given 2D dataset.

    Parameters
    ----------
    X : lib.FloatArray2D
        2D array of localizations, shape (n_samples, 2).
    min_locs : int
        Minimum number of localizations per component.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma.
    lp : lib.FloatArray1D
        Localization precision for each localization. Only used if
        loc_prec_handle is "local". Shape (n_samples,).
    loc_prec_handle : {"local", "abs"}, optional
        How to handle sigma bounds. If "local", localization precisions
        of points around each component are used to bound sigmas. Else,
        sigma_bounds specifies the absolute bounds on sigmas. Default
        is "local".
    max_rounds_without_best_bic : int, optional
        Maximum number of rounds without BIC improvement to terminate
        the search for the optimal G5M n_components. Default is
        `MAX_ROUNDS_WITHOUT_BEST_BIC`.

    Returns
    -------
    g5m : G5M_2D
        Fitted G5M. Returns None if fitting failed.
    """
    assert isinstance(lp, np.ndarray)
    assert loc_prec_handle in ["local", "abs"]
    assert len(lp) == len(X), (
        "Length of localization precision must match the number of "
        "localizations."
    )

    n_components = 1
    rounds_without_best_bic = 0
    best_bic = np.inf
    n_components_max = min(N_COMPONENTS_MAX, len(X) // min_locs)

    g5ms = []
    bics = []
    while (
        n_components <= n_components_max
        and rounds_without_best_bic < max_rounds_without_best_bic
    ):
        g5m = G5M_2D(
            n_components=n_components,
            min_locs=min_locs,
            sigma_bounds=sigma_bounds,
        ).fit(X, lp=lp, loc_prec_handle=loc_prec_handle)
        if g5m is None or not _check_G5M_resolution_2D(
            g5m.means, g5m.weights, g5m.precisions_cholesky
        ):
            current_bic = np.inf
            rounds_without_best_bic += 1
        else:
            current_bic = g5m.bic(X)
            if current_bic < best_bic:
                best_bic = current_bic
                rounds_without_best_bic = 0
            else:
                rounds_without_best_bic += 1
            g5ms.append(g5m)
            bics.append(current_bic)
        n_components += 1

    # select the best result
    if len(g5ms):
        best_bic_idx = np.argmin(bics)
        return g5ms[best_bic_idx]


def _run_g5m_group_2D(
    locs_group: pd.DataFrame,
    *,
    min_locs: int = MIN_LOCS,
    loc_prec_handle: Literal["local", "abs"] = "local",
    sigma_bounds: tuple[float, float] = (MIN_SIGMA_FACTOR, MAX_SIGMA_FACTOR),
    pixelsize: float = 130.0,
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
    bootstrap_check: bool = False,
    max_locs_per_cluster: int = np.inf,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Run G5M for a given group of localizations (by default
    one DBSCAN cluster of localizations) in 2D.

    Parameters
    ----------
    locs_group : pd.DataFrame
        Localizations.
    min_locs : int, optional
        Minimum number of localizations per component. Default is
        `MIN_LOCS`.
    loc_prec_handle : {"local", "abs"}, optional
        How to handle sigma bounds. If "local", localization precisions
        of points around each component are used to bound sigmas. Else,
        sigma_bounds specifies the absolute bounds on sigmas. Default is
        "local".
    sigma_bounds : tuple, optional
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma. Default is (`MIN_SIGMA_FACTOR`,
        `MAX_SIGMA_FACTOR`).
    pixelsize : float, optional
        Camera pixel size in nm. Default is 130.0.
    max_rounds_without_best_bic : int, optional
        Maximum number of rounds without BIC improvement to terminate
        the search for optimal G5M n_components. Default is
        `MAX_ROUNDS_WITHOUT_BEST_BIC`.
    bootstrap_check : bool, optional
        If True, the standard error of the means (SEM) is calculated
        using bootstrapping. If False, the standard, single Gaussian
        SEM is used. Default is False.
    max_locs_per_cluster : int, optional
        Maximum number of localizations per cluster accepted for G5M.
        Used to avoid fitting to fiducial markers. Such clusters are
        ignored. Default is np.inf.

    Returns
    -------
    centers : pd.DataFrame
        Centers of the G5M components in the format of localizations.
    clustered_locs : pd.DataFrame
        Localizations with assigned cluster labels, based on the G5M
        components.
    """
    assert loc_prec_handle in [
        "local",
        "abs",
    ], "loc_prec_handle must be 'local'  or 'abs'."
    assert (
        len(sigma_bounds) == 2
    ), "sigma_bounds must be a tuple of two values."

    # check that the number of localizations is within the limits
    n_locs = len(locs_group)
    if n_locs < min_locs or n_locs > max_locs_per_cluster:
        return None, None

    if loc_prec_handle == "local":
        lp = locs_group[["lpx", "lpy"]].mean(axis=1).to_numpy()
    else:
        lp = np.ones(len(locs_group))  # dummy
    X = locs_group[["x", "y"]].to_numpy().astype(np.float64)

    g5m = _find_optimal_G5M_2D(
        X,
        min_locs=min_locs,
        sigma_bounds=sigma_bounds,
        lp=lp,
        loc_prec_handle=loc_prec_handle,
        max_rounds_without_best_bic=max_rounds_without_best_bic,
    )
    if g5m is None or len(g5m.valid_idx) == 0:
        return None, None

    return _convert_G5M_results(g5m, locs_group, pixelsize, bootstrap_check)


class G5M_2D(G5M):
    """G5M for 2D data. See ``G5M`` for more details.

    Parameters
    ----------
    n_components : int
        Number of components in the model.
    min_locs : int
        Minimum number of localizations per component.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma.
    means_init : lib.FloatArray2D | None, optional
        Initial means (mu) of the Gaussian components. If None, the
        means are initialized using kmeans++. Default is None.
    """

    def __init__(
        self,
        n_components: int,
        min_locs: int,
        sigma_bounds: tuple[float, float],
        *,
        means_init: lib.FloatArray2D | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            min_locs=min_locs,
            sigma_bounds=sigma_bounds,
            covariance_type="spherical",
            means_init=means_init,
        )
        self.n_dimensions = 2

    def estimate_log_prob(self, X: lib.FloatArray2D) -> lib.FloatArray2D:
        """Calculate the log probabilities of the data X under the G5M,
        without weights.

        Parameters
        ----------
        X : lib.FloatArray2D
            ``(n_locs, n_dim)`` localization coordinates.

        Returns
        -------
        log_prob : lib.FloatArray2D
            ``(n_locs, n_valid_components)`` log probabilities.
        """
        return _estimate_log_gaussian_prob_2D(
            X,
            self.means,
            self.precisions_cholesky,
        )

    def n_parameters(self) -> int:
        """Find the number of parameters in the G5M.

        Returns
        -------
        n_params : int
            Free parameters of the valid components: their means, covariances
            and weights.
        """
        n_valid = len(self.valid_idx)
        cov_params = n_valid
        mean_params = 2 * n_valid
        weight_params = n_valid - 1
        return int(cov_params + mean_params + weight_params)

    def sample(
        self, n_samples: int = 1
    ) -> tuple[lib.FloatArray2D, lib.IntArray1D]:
        """Sample data points from the G5M.

        Parameters
        ----------
        n_samples : int, optional
            Number of points to draw. Default 1.

        Returns
        -------
        X : lib.FloatArray2D
            ``(n_samples, n_dim)`` sampled coordinates.
        y : lib.IntArray1D
            Index of the component each point was drawn from.
        """
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights)

        X = np.vstack(
            [
                mean
                + rng.standard_normal(size=(sample, 2)) * np.sqrt(covariance)
                for (mean, covariance, sample) in zip(
                    self.means, self.covariances, n_samples_comp
                )
            ]
        )

        y = np.concatenate(
            [
                np.full(sample, j, dtype=int)
                for j, sample in enumerate(n_samples_comp)
            ]
        )
        return X, y


# 3D G5M functions and classes #
@njit
def _check_G5M_resolution_3D(
    means: lib.FloatArray2D,
    weights: lib.FloatArray1D,
    precisions_chol: lib.FloatArray2D,
) -> bool:
    """Check if Sparrow limit is passed for all components of the
    ``G5M_3D``.

    Sparrow resolution limit is violated if there is not local minimum
    between two signals.

    Parameters
    ----------
    means : np.ndarray
        Means of the G5M components, shape (n_components, 3).
    weights : np.ndarray
        Weights of the G5M components, shape (n_components,).
    precisions_chol : np.ndarray
        Cholesky decomposition of the precision matrices of the G5M
        components, shape (n_components, 3).

    Returns
    -------
    bool
        True if the G5M components are well separated, False otherwise.
    """
    n_valid_components = means.shape[0]
    if n_valid_components == 0:  # if no component is valid
        return False
    elif n_valid_components == 1:
        return True

    # iterate over all pairs of components
    for i in range(n_valid_components):
        for j in range(i + 1, n_valid_components):
            # extract the parameters of the two components
            prec_chol_ = np.zeros((2, 3), dtype=np.float64)
            prec_chol_[0, :] = precisions_chol[i]
            prec_chol_[1, :] = precisions_chol[j]
            weights_ = np.array([weights[i], weights[j]])
            means_ = np.zeros((2, 3), dtype=np.float64)
            means_[0, :] = means[i]
            means_[1, :] = means[j]

            # get the straight line between the two components
            direction_vector = means_[1, :] - means_[0, :]
            t = np.linspace(0, 1, 40)  # parameter for the line
            x = means_[0, 0] + direction_vector[0] * t
            y = means_[0, 1] + direction_vector[1] * t
            z = means_[0, 2] + direction_vector[2] * t

            # get the PDF of all components along the line
            X = np.stack((x, y, z)).T
            ll = _estimate_log_gaussian_prob_3D(
                X, means_, prec_chol_
            ) + np.log(weights_)
            pdf = _sum_along_axis1(np.exp(ll), ll.shape[0])

            # find if there is at least one local minimum (may be more
            # if components in between align)
            if not len(lib.find_local_minima(pdf)):
                return False

    # if all components are well separated
    return True


@njit
def _initialize_G5M_3D(
    X: lib.FloatArray2D, n_init: int, n_components: int, random_state: int
) -> tuple[lib.FloatArray2D, lib.FloatArray3D, lib.FloatArray3D]:
    """Initialize the 3D G5M parameters using kmeans++."""
    n_samples = X.shape[0]
    init_weights = np.zeros((n_init, n_components), dtype=np.float64)
    init_means = np.zeros((n_init, n_components, 3), dtype=np.float64)
    init_precisions_cholesky = np.zeros(
        (n_init, n_components, 3), dtype=np.float64
    )
    for ii in range(n_init):
        # initialize responsibilities using kmeans++ (e-step-like)
        resp = np.zeros((n_samples, n_components), dtype=np.float64)
        indices = _kmeans_plusplus(X, n_components, random_state)  # kmeans++

        for i in range(n_components):
            resp[indices[i], i] = 1

        # initialize G5M parameters (m-step-like)
        weights, means, covariances = _estimate_gaussian_parameters_3D(X, resp)
        weights /= n_samples
        init_weights[ii] = weights
        init_means[ii] = means
        init_precisions_cholesky[ii] = 1.0 / np.sqrt(covariances)

        random_state += 1

    return (
        np.asarray(init_weights, dtype=np.float64),
        np.asarray(init_means, dtype=np.float64),
        np.asarray(init_precisions_cholesky, dtype=np.float64),
    )


@njit
def _estimate_gaussian_parameters_3D(
    X: lib.FloatArray2D,
    resp: lib.FloatArray2D,
) -> tuple[lib.FloatArray1D, lib.FloatArray2D, lib.FloatArray2D]:
    return _estimate_gaussian_parameters_diag_cov(X, resp)


@njit
def _estimate_log_gaussian_prob_3D(
    X: lib.FloatArray2D,
    means: lib.FloatArray2D,
    precisions_chol: lib.FloatArray2D,
) -> lib.FloatArray2D:
    log_det = _sum_along_axis1(
        np.log(precisions_chol),
        (precisions_chol.shape[0],),
    )
    precisions = _square_elements_2d(precisions_chol)
    log_prob = _gauss_exponential_term_3D(X, means, precisions)
    return -0.5 * (3 * np.log(2 * np.pi) + log_prob) + log_det


@njit
def _e_step_3D(
    X: lib.FloatArray2D,
    weights: lib.FloatArray1D,
    means: lib.FloatArray2D,
    precisions_cholesky: lib.FloatArray2D,
) -> tuple[float, lib.FloatArray2D]:
    weighted_log_prob = _estimate_log_gaussian_prob_3D(
        X, means, precisions_cholesky
    ) + np.log(weights)
    log_prob_norm = _logsumexp_axis1(weighted_log_prob, (X.shape[0],))
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    return np.mean(log_prob_norm), log_resp.astype(np.float64)


@njit
def _clip_covs_3D(
    covs,
    min_cov_x,
    max_cov_x,
    min_cov_y,
    max_cov_y,
    min_cov_z,
    max_cov_z,
):
    for i in range(len(covs)):
        if covs[i, 0] < min_cov_x[i]:
            covs[i, 0] = min_cov_x[i]
        elif covs[i, 0] > max_cov_x[i]:
            covs[i, 0] = max_cov_x[i]
        if covs[i, 1] < min_cov_y[i]:
            covs[i, 1] = min_cov_y[i]
        elif covs[i, 1] > max_cov_y[i]:
            covs[i, 1] = max_cov_y[i]
        if covs[i, 2] < min_cov_z[i]:
            covs[i, 2] = min_cov_z[i]
        if covs[i, 2] > max_cov_z[i]:
            covs[i, 2] = max_cov_z[i]


@njit
def _m_step_3D(
    X: lib.FloatArray2D,
    log_resp: lib.FloatArray2D,
    sigma_bounds: tuple[float, float],
    lp: lib.FloatArray2D,
    loc_prec_handle: Literal["local", "abs"],
    cx: lib.FloatArray1D = np.array([]),
    cy: lib.FloatArray1D = np.array([]),
    mag_factor: float = 0.79,
    angles: lib.FloatArray1D = np.array([]),  # for rotated 3D consistency
) -> tuple[
    lib.FloatArray1D, lib.FloatArray2D, lib.FloatArray2D, lib.FloatArray2D
]:
    """Modified m-step to handle astigmatism in 3D G5M. ``angles`` is not
    used and is here for compatibility with the rotated 3D m step.

    The astigmatism modification handles the astigmatism effect in
    3D DNA-PAINT data. As in sklearn's implementation, the weights,
    means and diagonal covariance matrices are estimated first, together
    with the min/max constrainsts, like in ``_m_step_2D``. In the
    next step, sigma bound are imposed and then the ratio of the spot
    width and height is extracted from calibration, based on the z
    position, for each component and imposed on the covariances' x and
    y values.

    When ``cx``/``cy`` are empty (spline fitting), the astigmatism
    coupling step is skipped and the x/y/z covariances stay independent
    (plain diagonal 3D model), bounded by the local loc. precisions."""
    resp = np.exp(log_resp)
    weights, means, covs = _estimate_gaussian_parameters_3D(X, resp)

    # find the min. and max. covariances in each dimension
    if loc_prec_handle == "local":
        # extract loc. precisions and convert to covariances
        lpx = np.ascontiguousarray(lp[:, 0]).reshape(-1, 1)
        lpy = np.ascontiguousarray(lp[:, 1]).reshape(-1, 1)
        lpz = np.ascontiguousarray(lp[:, 2]).reshape(-1, 1)

        mean_lpx_per_component = _sum_along_axis0(
            resp * lpx, (resp.shape[1])
        ) / _sum_along_axis0(resp, (resp.shape[1]))
        mean_lpy_per_component = _sum_along_axis0(
            resp * lpy, (resp.shape[1])
        ) / _sum_along_axis0(resp, (resp.shape[1]))
        mean_lpz_per_component = _sum_along_axis0(
            resp * lpz, (resp.shape[1])
        ) / _sum_along_axis0(resp, (resp.shape[1]))

        mean_covx_per_component = _square_elements_1d(mean_lpx_per_component)
        mean_covy_per_component = _square_elements_1d(mean_lpy_per_component)
        mean_covz_per_component = _square_elements_1d(mean_lpz_per_component)

        min_cov_x = sigma_bounds[0] ** 2 * mean_covx_per_component
        max_cov_x = sigma_bounds[1] ** 2 * mean_covx_per_component
        min_cov_y = sigma_bounds[0] ** 2 * mean_covy_per_component
        max_cov_y = sigma_bounds[1] ** 2 * mean_covy_per_component
        min_cov_z = sigma_bounds[0] ** 2 * mean_covz_per_component

        # decrease max z cov because the lpz is already pretty high
        max_cov_z = (
            (sigma_bounds[1] - 1.0) * 0.5 + 1.0
        ) ** 2 * mean_covz_per_component
    elif loc_prec_handle == "abs":
        min_cov_x = np.full(covs.shape[0], sigma_bounds[0] ** 2)
        max_cov_x = np.full(covs.shape[0], sigma_bounds[1] ** 2)
        min_cov_y = np.full(covs.shape[0], sigma_bounds[0] ** 2)
        max_cov_y = np.full(covs.shape[0], sigma_bounds[1] ** 2)
        # roughly account for worse z precision
        min_cov_z = np.full(covs.shape[0], sigma_bounds[0] ** 2 * 2.0**2)
        max_cov_z = np.full(covs.shape[0], sigma_bounds[1] ** 2 * 2.5**2)

    _clip_covs_3D(
        covs,
        min_cov_x,
        max_cov_x,
        min_cov_y,
        max_cov_y,
        min_cov_z,
        max_cov_z,
    )

    # impose the astigmatism coupling between the x and y covariances
    # only when calibration polynomials are provided (astigmatism mode).
    # For spline fitting cx/cy are empty and the covariances stay
    # independent (plain diagonal 3D model).
    if cx.size > 0 and cy.size > 0:
        z_position_calib = means[:, 2] / mag_factor
        spot_width = _poly1d(cx, z_position_calib)
        spot_height = _poly1d(cy, z_position_calib)
        ratio = spot_width / spot_height

        covs_xy = np.empty((covs.shape[0], 2))
        covs_xy[:, 0] = covs[:, 0]
        covs_xy[:, 1] = covs[:, 1]
        mean_xy_covs = _mean_along_axis1(covs_xy, (covs_xy.shape[0],))
        covs[:, 0] = mean_xy_covs * ratio
        covs[:, 1] = mean_xy_covs / ratio
    weights /= weights.sum()
    precisions_cholesky = 1.0 / np.sqrt(covs)
    return weights, means, covs, precisions_cholesky


# 3D G5M with a rotated xy covariance block #
#
# Used when the localizations were fitted with the rotated elliptical
# Gaussian PSF model The component covariance is blockdiag(R(theta) @
# diag(cov_u, cov_v) @ R(theta).T, cov_z), where cov_u is the variance
# along the axis at angle theta.
#
# Angle is not a free parameter and is taken as the mean value from the
# surrounding localizations.
#
# The u axis corresponds to the PSF fit's sx and the v axis to sy: the
# rotated PSF model in picasso.fitting.gaussfit projects onto
# (cos a, -sin a) with internal angle a, and localize saves
# angle = -rad2deg(a), so the saved angle is already the standard-
# convention angle of the sx axis and needs no sign flip here.
@njit
def _estimate_gaussian_parameters_3D_rot(
    X: lib.FloatArray2D,
    resp: lib.FloatArray2D,
    theta: lib.FloatArray1D,
    reg_covar: float = 1e-6,
) -> tuple[lib.FloatArray1D, lib.FloatArray2D, lib.FloatArray2D]:
    """MLE parameters for the rotated 3D model.

    Second moments are taken in each component's own rotated frame, so
    the returned covariances are the principal-axis variances
    ``(cov_u, cov_v, cov_z)`` rather than camera-axis ones.

    Parameters
    ----------
    X : lib.FloatArray2D
        Data points, shape (n_samples, 3).
    resp : lib.FloatArray2D
        Responsibilities of the components, shape (n_samples,
        n_components).
    theta : lib.FloatArray1D
        Rotation angle of each component's major axis in radians, shape
        (n_components,).
    reg_covar : float, optional
        Non-negative regularization added to the variances. Default is
        1e-6.

    Returns
    -------
    nk : lib.FloatArray1D
        Number of localizations per component, shape (n_components,).
    means : lib.FloatArray2D
        Means of the components, shape (n_components, 3).
    covariances : lib.FloatArray2D
        Principal-axis variances of the components, shape
        (n_components, 3).
    """
    nk = (
        _sum_along_axis0(resp, (resp.shape[1],))
        + 10.0 * np.finfo(resp.dtype).eps
    )
    means = _matmul(resp.T, X) / nk[:, np.newaxis]
    covariances = np.zeros((resp.shape[1], 3), dtype=np.float64)
    for k in range(resp.shape[1]):
        ct = np.cos(theta[k])
        st = np.sin(theta[k])
        s_uu = 0.0
        s_vv = 0.0
        s_zz = 0.0
        for i in range(X.shape[0]):
            r = resp[i, k]
            dx = X[i, 0] - means[k, 0]
            dy = X[i, 1] - means[k, 1]
            dz = X[i, 2] - means[k, 2]
            du = dx * ct + dy * st
            dv = -dx * st + dy * ct
            s_uu += r * du * du
            s_vv += r * dv * dv
            s_zz += r * dz * dz
        covariances[k, 0] = s_uu / nk[k] + reg_covar
        covariances[k, 1] = s_vv / nk[k] + reg_covar
        covariances[k, 2] = s_zz / nk[k] + reg_covar
    return (
        np.asarray(nk, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(covariances, dtype=np.float64),
    )


@njit
def _estimate_log_gaussian_prob_3D_rot(
    X: lib.FloatArray2D,
    means: lib.FloatArray2D,
    precisions_chol: lib.FloatArray3D,
) -> lib.FloatArray2D:
    n_components = precisions_chol.shape[0]
    log_det = np.zeros(n_components, dtype=np.float64)
    for k in range(n_components):
        log_det[k] = (
            np.log(precisions_chol[k, 0, 0])
            + np.log(precisions_chol[k, 1, 1])
            + np.log(precisions_chol[k, 2, 2])
        )
    log_prob = _gauss_exponential_term_3D_rot(X, means, precisions_chol)
    return -0.5 * (3 * np.log(2 * np.pi) + log_prob) + log_det


@njit
def _e_step_3D_rot(
    X: lib.FloatArray2D,
    weights: lib.FloatArray1D,
    means: lib.FloatArray2D,
    precisions_cholesky: lib.FloatArray3D,
) -> tuple[float, lib.FloatArray2D]:
    weighted_log_prob = _estimate_log_gaussian_prob_3D_rot(
        X, means, precisions_cholesky
    ) + np.log(weights)
    log_prob_norm = _logsumexp_axis1(weighted_log_prob, (X.shape[0],))
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    return np.mean(log_prob_norm), log_resp.astype(np.float64)


@njit
def _check_G5M_resolution_3D_rot(
    means: lib.FloatArray2D,
    weights: lib.FloatArray1D,
    precisions_chol: lib.FloatArray3D,
) -> bool:
    """Sparrow limit check for ``G5M_3D`` with a rotated xy block. Same
    procedure as ``_check_G5M_resolution_3D``."""
    n_valid_components = means.shape[0]
    if n_valid_components == 0:
        return False
    elif n_valid_components == 1:
        return True

    for i in range(n_valid_components):
        for j in range(i + 1, n_valid_components):
            prec_chol_ = np.zeros((2, 3, 3), dtype=np.float64)
            prec_chol_[0, :, :] = precisions_chol[i]
            prec_chol_[1, :, :] = precisions_chol[j]
            weights_ = np.array([weights[i], weights[j]])
            means_ = np.zeros((2, 3), dtype=np.float64)
            means_[0, :] = means[i]
            means_[1, :] = means[j]

            direction_vector = means_[1, :] - means_[0, :]
            t = np.linspace(0, 1, 40)
            x = means_[0, 0] + direction_vector[0] * t
            y = means_[0, 1] + direction_vector[1] * t
            z = means_[0, 2] + direction_vector[2] * t

            X = np.stack((x, y, z)).T
            ll = _estimate_log_gaussian_prob_3D_rot(
                X, means_, prec_chol_
            ) + np.log(weights_)
            pdf = _sum_along_axis1(np.exp(ll), ll.shape[0])

            if not len(lib.find_local_minima(pdf)):
                return False

    return True


@njit
def _initialize_G5M_3D_rot(
    X: lib.FloatArray2D, n_init: int, n_components: int, random_state: int
) -> tuple[lib.FloatArray2D, lib.FloatArray3D, lib.FloatArray4D]:
    """Initialize the rotated 3D G5M parameters using kmeans++.

    The angle is seeded at zero (an axis-aligned block); the first m-step
    replaces it with the measured mean angle of each component."""
    n_samples = X.shape[0]
    init_weights = np.zeros((n_init, n_components), dtype=np.float64)
    init_means = np.zeros((n_init, n_components, 3), dtype=np.float64)
    init_precisions_cholesky = np.zeros(
        (n_init, n_components, 3, 3), dtype=np.float64
    )
    zero_theta = np.zeros(n_components, dtype=np.float64)
    for ii in range(n_init):
        resp = np.zeros((n_samples, n_components), dtype=np.float64)
        indices = _kmeans_plusplus(X, n_components, random_state)

        for i in range(n_components):
            resp[indices[i], i] = 1

        weights, means, covariances = _estimate_gaussian_parameters_3D_rot(
            X, resp, zero_theta
        )
        weights /= n_samples
        init_weights[ii] = weights
        init_means[ii] = means
        init_precisions_cholesky[ii] = _precision_chol_3D_rot(
            _assemble_covs_3D_rot(
                covariances[:, 0],
                covariances[:, 1],
                covariances[:, 2],
                zero_theta,
            )
        )

        random_state += 1

    return (
        np.asarray(init_weights, dtype=np.float64),
        np.asarray(init_means, dtype=np.float64),
        np.asarray(init_precisions_cholesky, dtype=np.float64),
    )


@njit
def _m_step_3D_rot(
    X: lib.FloatArray2D,
    log_resp: lib.FloatArray2D,
    sigma_bounds: tuple[float, float],
    lp: lib.FloatArray2D,
    loc_prec_handle: Literal["local", "abs"],
    cx: lib.FloatArray1D = np.array([]),
    cy: lib.FloatArray1D = np.array([]),
    mag_factor: float = 0.79,
    angles: lib.FloatArray1D = np.array([]),
) -> tuple[
    lib.FloatArray1D, lib.FloatArray2D, lib.FloatArray3D, lib.FloatArray3D
]:
    """M-step for the 3D G5M with a rotated xy covariance block.

    Identical in structure to ``_m_step_3D``, but the second moments,
    the sigma bounds and the astigmatism coupling are all applied in each
    component's own rotated frame instead of the camera frame. The angle
    is measured from the localizations rather than fitted."""
    resp = np.exp(log_resp)
    # angle of each component, measured from the localizations
    theta = _circular_weighted_mean_angle(resp, angles)
    weights, means, covs = _estimate_gaussian_parameters_3D_rot(X, resp, theta)

    if loc_prec_handle == "local":
        # in-plane precision is a single scalar: once the frame is
        # rotated, lpx and lpy no longer match the principal axes
        lpxy = np.ascontiguousarray(0.5 * (lp[:, 0] + lp[:, 1])).reshape(-1, 1)
        lpz = np.ascontiguousarray(lp[:, 2]).reshape(-1, 1)

        resp_sum = _sum_along_axis0(resp, (resp.shape[1]))
        mean_lpxy_per_component = (
            _sum_along_axis0(resp * lpxy, (resp.shape[1])) / resp_sum
        )
        mean_lpz_per_component = (
            _sum_along_axis0(resp * lpz, (resp.shape[1])) / resp_sum
        )

        mean_covxy_per_component = _square_elements_1d(mean_lpxy_per_component)
        mean_covz_per_component = _square_elements_1d(mean_lpz_per_component)

        min_cov_u = sigma_bounds[0] ** 2 * mean_covxy_per_component
        max_cov_u = sigma_bounds[1] ** 2 * mean_covxy_per_component
        min_cov_v = min_cov_u.copy()
        max_cov_v = max_cov_u.copy()
        min_cov_z = sigma_bounds[0] ** 2 * mean_covz_per_component

        # decrease max z cov because the lpz is already pretty high
        max_cov_z = (
            (sigma_bounds[1] - 1.0) * 0.5 + 1.0
        ) ** 2 * mean_covz_per_component
    elif loc_prec_handle == "abs":
        min_cov_u = np.full(covs.shape[0], sigma_bounds[0] ** 2)
        max_cov_u = np.full(covs.shape[0], sigma_bounds[1] ** 2)
        min_cov_v = np.full(covs.shape[0], sigma_bounds[0] ** 2)
        max_cov_v = np.full(covs.shape[0], sigma_bounds[1] ** 2)
        # roughly account for worse z precision
        min_cov_z = np.full(covs.shape[0], sigma_bounds[0] ** 2 * 2.0**2)
        max_cov_z = np.full(covs.shape[0], sigma_bounds[1] ** 2 * 2.5**2)

    # covs holds (cov_u, cov_v, cov_z), so the diagonal clipper applies
    # unchanged - it is agnostic to which frame the axes are in
    _clip_covs_3D(
        covs,
        min_cov_u,
        max_cov_u,
        min_cov_v,
        max_cov_v,
        min_cov_z,
        max_cov_z,
    )

    # impose the astigmatism coupling, now between the two principal
    # axes rather than the camera axes
    if cx.size > 0 and cy.size > 0:
        z_position_calib = means[:, 2] / mag_factor
        spot_width = _poly1d(cx, z_position_calib)
        spot_height = _poly1d(cy, z_position_calib)
        ratio = spot_width / spot_height

        covs_uv = np.empty((covs.shape[0], 2))
        covs_uv[:, 0] = covs[:, 0]
        covs_uv[:, 1] = covs[:, 1]
        mean_uv_covs = _mean_along_axis1(covs_uv, (covs_uv.shape[0],))
        covs[:, 0] = mean_uv_covs * ratio
        covs[:, 1] = mean_uv_covs / ratio
    weights /= weights.sum()
    covs_full = _assemble_covs_3D_rot(
        covs[:, 0], covs[:, 1], covs[:, 2], theta
    )
    precisions_cholesky = _precision_chol_3D_rot(covs_full)
    return weights, means, covs_full, precisions_cholesky


def _find_optimal_G5M_3D(
    X: lib.FloatArray2D,
    min_locs: int,
    sigma_bounds: tuple[float, float],
    *,
    calibration: dict | None,
    lp: lib.FloatArray2D,
    loc_prec_handle: Literal["local", "abs"] = "local",
    mode: Literal["astigmatism", "spline"] = "astigmatism",
    covariance_type: Literal["diagonal", "rotated"] = "diagonal",
    angles: lib.FloatArray1D | None = None,
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
) -> G5M_3D:
    """Find optimal G5M for given 3D data X.

    Parameters
    ----------
    X : np.ndarray
        2D array of localizations, shape (n_samples, 3).
    min_locs : int
        Minimum number of localizations per component.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma.
    calibration : dict or None
        Astigmatism calibration dictionary with the keys "X
        Coefficients", "Y Coefficients" and "Magnification factor".
        Required for ``mode="astigmatism"``, may be None for spline.
        See https://picassosr.readthedocs.io/en/latest/localize.html#d-calibration.  # noqa: E501
    lp : lib.FloatArray2D
        Localization precision for each localization in x, y and z. Only
        used if loc_prec_handle is "local". Shape (n_samples, 3).
    loc_prec_handle : {"local", "abs"}, optional
        How to handle sigma bounds. If "local", localization precisions
        of points around each component are used to bound sigmas. Else,
        sigma_bounds specifies the absolute bounds on sigmas. Default
        is "local".
    mode : {"astigmatism", "spline"}, optional
        Fitting mode of the input localizations. Default is
        "astigmatism".
    covariance_type : {"diagonal", "rotated"}, optional
        Shape of the G5M components. "diagonal" is axis-aligned;
        "rotated" gives the xy block a rotation measured from `angles`.
        Default is "diagonal".
    angles : np.ndarray or None, optional
        Angle of each localization in radians, shape (n_samples,).
        Required for ``covariance_type="rotated"``. Default is None.
    max_rounds_without_best_bic : int, optional
        Maximum number of rounds without BIC improvement to terminate
        the search for optimal G5M n_components. Default is
        `MAX_ROUNDS_WITHOUT_BEST_BIC`.

    Returns
    -------
    g5m : G5M_3D
        Fitted G5M. Returns None if fitting failed.
    """
    assert isinstance(lp, np.ndarray)
    assert loc_prec_handle in ["local", "abs"]
    assert lp.shape == (len(X), 3), (
        "Localization precisions (lp) must have the shape of (N, 3) "
        "where N is the number of localizations."
    )
    if covariance_type == "rotated":
        assert angles is not None and len(angles) == len(X), (
            "Angles must be provided for each localization when "
            "covariance_type is 'rotated'."
        )
    if mode == "astigmatism":
        for key in [
            "X Coefficients",
            "Y Coefficients",
            "Magnification factor",
        ]:
            assert calibration is not None and key in calibration, (
                "Calibration dictionary must contain the keys 'X "
                "Coefficients', 'Y Coefficients' and 'Magnification "
                "factor'"
            )

    n_components = 1
    rounds_without_best_bic = 0
    best_bic = np.inf
    n_components_max = min(N_COMPONENTS_MAX, len(X) // min_locs)

    g5ms = []
    bics = []
    while (
        n_components <= n_components_max
        and rounds_without_best_bic < max_rounds_without_best_bic
    ):
        g5m = G5M_3D(
            n_components=n_components,
            min_locs=min_locs,
            sigma_bounds=sigma_bounds,
            calibration=calibration,
            mode=mode,
            covariance_type=covariance_type,
        ).fit(
            X,
            lp=lp,
            loc_prec_handle=loc_prec_handle,
            angles=angles,
        )
        check_resolution = (
            _check_G5M_resolution_3D_rot
            if covariance_type == "rotated"
            else _check_G5M_resolution_3D
        )
        if g5m is None or not check_resolution(
            g5m.means, g5m.weights, g5m.precisions_cholesky
        ):
            current_bic = np.inf
            rounds_without_best_bic += 1
        else:
            current_bic = g5m.bic(X)
            if current_bic < best_bic:
                best_bic = current_bic
                rounds_without_best_bic = 0
            else:
                rounds_without_best_bic += 1
            g5ms.append(g5m)
            bics.append(current_bic)
        n_components += 1

    # select the best result
    if len(g5ms):
        best_bic_idx = np.argmin(bics)
        return g5ms[best_bic_idx]


def _run_g5m_group_3D(
    locs_group: pd.DataFrame,
    calibration: dict | None,
    *,
    min_locs: int = MIN_LOCS,
    loc_prec_handle: Literal["local", "abs"] = "local",
    sigma_bounds: tuple[float, float] = (MIN_SIGMA_FACTOR, MAX_SIGMA_FACTOR),
    pixelsize: float = 130.0,
    mode: Literal["astigmatism", "spline"] = "astigmatism",
    covariance_type: Literal["diagonal", "rotated"] = "diagonal",
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
    bootstrap_check: bool = False,
    max_locs_per_cluster: int = np.inf,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Run G5M for a given group of localizations (by default one
    DBSCAN cluster of localizations) in 3D.

    Parameters
    ----------
    locs_group : pd.DataFrame
        Localizations.
    calibration : dict or None
        Astigmatism calibration dictionary with the keys "X
        Coefficients", "Y Coefficients" and "Magnification factor".
        Required for ``mode="astigmatism"``, may be None for spline.
        See https://picassosr.readthedocs.io/en/latest/localize.html#d-calibration.  # noqa: E501
    min_locs : int, optional
        Minimum number of localizations per component. Default is
        `MIN_LOCS`.
    loc_prec_handle : {"local", "abs"}, optional
        How to handle sigma bounds. If "local", localization precisions
        of points around each component are used to bound sigmas. Else,
        sigma_bounds specifies the absolute bounds on sigmas. Default
        is "local".
    sigma_bounds : tuple, optional
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If loc_prec_handle is "local", the bounds specify
        the margin of error in units of localization precision. Else,
        the bounds specify the absolute bounds on sigma. Default is
        (`MIN_SIGMA_FACTOR`, `MAX_SIGMA_FACTOR`).
    pixelsize : float, optional
        Camera pixel size in nm. Default is 130.0.
    mode : {"astigmatism", "spline"}, optional
        Fitting mode of the input localizations. "spline" uses a plain
        diagonal 3D model and reads lpz directly from the locs. Default
        is "astigmatism".
    covariance_type : {"diagonal", "rotated"}, optional
        Shape of the G5M components. "diagonal" is axis-aligned;
        "rotated" gives the xy block a rotation measured from the
        ``"angle"`` column of ``locs_group`` (in degrees). Default is
        "diagonal".
    max_rounds_without_best_bic : int, optional
        Maximum number of rounds without BIC improvement to terminate
        the search for optimal G5M n_components. Default is
        `MAX_ROUNDS_WITHOUT_BEST_BIC`.
    bootstrap_check : bool, optional
        If True, the standard error of the means (SEM) is calculated
        using bootstrapping. If False, the standard, single Gaussian
        SEM is used. Default is False.
    max_locs_per_cluster : int, optional
        Maximum number of localizations per cluster accepted for G5M.
        Used to avoid fitting to fiducial markers. Such clusters are
        ignored. Default is np.inf.

    Returns
    -------
    centers : pd.DataFrame
        Centers of the G5M components in the format of localizations.
    clustered_locs : pd.DataFrame
        Localizations with assigned cluster labels, based on the G5M
        components.
    """
    assert loc_prec_handle in [
        "local",
        "abs",
    ], "loc_prec_handle must be 'local' or 'abs'."
    assert (
        len(sigma_bounds) == 2
    ), "sigma_bounds must be a tuple of two values."
    # make sure lpz is available. For astigmatism (assume gauss
    # least-squares used for localization) it can be derived from the
    # calibration; spline fitting already provides lpz directly.
    if "lpz" not in locs_group.columns:
        if mode == "spline":
            raise ValueError(
                "Spline mode requires an 'lpz' column in the "
                "localizations (produced by the spline 3D fit)."
            )
        locs_group = locs_group.copy()
        locs_group["lpz"] = zfit.axial_localization_precision(
            locs_group, [{"Pixelsize": pixelsize}], calibration, "gausslq"
        )
    # check that the number of localizations is within the limits
    n_locs = len(locs_group)
    if n_locs < min_locs or n_locs > max_locs_per_cluster:
        return None, None

    if loc_prec_handle == "local":
        lp = locs_group[["lpx", "lpy", "lpz"]].to_numpy()
    else:
        lp = np.ones((len(locs_group), 3))  # dummy
    X = locs_group[["x", "y", "z"]].to_numpy()
    X[:, 2] /= pixelsize  # convert z to camera pixels
    lp[:, 2] /= pixelsize  # convert lpz to camera pixels

    if covariance_type == "rotated":
        # the "angle" column is written by picasso.localize in degrees;
        # the m-step works in radians
        angles = np.deg2rad(locs_group["angle"].to_numpy())
    else:
        angles = None

    g5m = _find_optimal_G5M_3D(
        X,
        min_locs=min_locs,
        sigma_bounds=sigma_bounds,
        lp=lp,
        loc_prec_handle=loc_prec_handle,
        calibration=calibration,
        mode=mode,
        covariance_type=covariance_type,
        angles=angles,
        max_rounds_without_best_bic=max_rounds_without_best_bic,
    )
    if g5m is None or len(g5m.valid_idx) == 0:
        return None, None

    return _convert_G5M_results(g5m, locs_group, pixelsize, bootstrap_check)


class G5M_3D(G5M):
    """G5M for 3D data (astigmatism or spline). See ``G5M`` for more
    details.

    Parameters
    ----------
    n_components : int
        Number of components in the model.
    min_locs : int
        Minimum number of localizations per component.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma.
    calibration : dict or None
        Astigmatism calibration dictionary with the keys "X
        Coefficients", "Y Coefficients" and "Magnification factor".
        Required for ``mode="astigmatism"`` and ignored (may be None)
        for ``mode="spline"``.
        See https://picassosr.readthedocs.io/en/latest/localize.html#d-calibration.  # noqa: E501
    mode : {"astigmatism", "spline"}, optional
        Fitting mode of the input localizations. "astigmatism" couples
        the x/y covariances via the calibration polynomials; "spline"
        uses a plain diagonal 3D model (independent x/y/z covariances).
        Default is "astigmatism".
    covariance_type : {"diagonal", "rotated"}, optional
        Shape of the G5M components. "diagonal" is axis-aligned;
        "rotated" rotates the xy block of each component and requires
        ``mode="astigmatism"``. Default is "diagonal".
    means_init : np.ndarray or None, optional
        Initial means (mu) of the Gaussian components. If None, the
        means are initialized using kmeans++. Default is None.
    """

    def __init__(
        self,
        n_components: int,
        min_locs: int,
        sigma_bounds: tuple[float, float],
        *,
        calibration: dict | None,
        mode: Literal["astigmatism", "spline"] = "astigmatism",
        covariance_type: Literal["diagonal", "rotated"] = "diagonal",
        means_init: np.ndarray | None = None,
    ) -> None:
        assert mode in [
            "astigmatism",
            "spline",
        ], "mode must be 'astigmatism' or 'spline'."
        assert covariance_type in (
            "diagonal",
            "rotated",
        ), "covariance_type must be 'diagonal' or 'rotated' for 3D data."
        # the rotated model couples the two principal axes via the
        # calibration, so it is only defined in astigmatism mode
        assert not (
            covariance_type == "rotated" and mode == "spline"
        ), "covariance_type='rotated' requires mode='astigmatism'."
        if mode == "astigmatism":
            for key in [
                "X Coefficients",
                "Y Coefficients",
                "Magnification factor",
            ]:
                assert calibration is not None and key in calibration, (
                    "Calibration dictionary must contain the keys 'X "
                    "Coefficients', 'Y Coefficients' and 'Magnification "
                    "factor'"
                )
        super().__init__(
            n_components=n_components,
            min_locs=min_locs,
            sigma_bounds=sigma_bounds,
            covariance_type=covariance_type,
            means_init=means_init,
        )
        self.calibration = calibration
        self.mode = mode
        self.n_dimensions = 3

    def estimate_log_prob(self, X: lib.FloatArray2D) -> lib.FloatArray2D:
        """Calculate the log probabilities of the data X under the G5M,
        without weights.

        Parameters
        ----------
        X : lib.FloatArray2D
            ``(n_locs, n_dim)`` localization coordinates.

        Returns
        -------
        log_prob : lib.FloatArray2D
            ``(n_locs, n_valid_components)`` log probabilities.
        """
        if self.covariance_type == "rotated":
            return _estimate_log_gaussian_prob_3D_rot(
                X,
                self.means,
                self.precisions_cholesky,
            )
        return _estimate_log_gaussian_prob_3D(
            X,
            self.means,
            self.precisions_cholesky,
        )

    def n_parameters(self) -> int:
        """Return the number of free parameters in the model.

        Note that in astigmatism mode the modification reduces the number of
        free parameters for each component by one (cov. in y depends on cov.
        in x), whereas in spline mode all three covariances are free.

        Returns
        -------
        n_params : int
            Free parameters of the valid components.
        """
        n_valid = len(self.valid_idx)
        # astigmatism: cov. in y depends on cov. in x (2 free per comp.);
        # spline: x/y/z covariances are independent (3 free per comp.)
        cov_params = n_valid * (3 if self.mode == "spline" else 2)
        mean_params = 3 * n_valid
        weight_params = n_valid - 1
        return int(cov_params + mean_params + weight_params)

    def sample(
        self, n_samples: int = 1
    ) -> tuple[lib.FloatArray2D, lib.IntArray1D]:
        """Sample data points from the G5M.

        Parameters
        ----------
        n_samples : int, optional
            Number of points to draw. Default 1.

        Returns
        -------
        X : lib.FloatArray2D
            ``(n_samples, n_dim)`` sampled coordinates.
        y : lib.IntArray1D
            Index of the component each point was drawn from.
        """
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights)

        if self.covariance_type == "rotated":
            samples = []
            for mean, covariance, sample in zip(
                self.means, self.covariances, n_samples_comp
            ):
                eigvals, eigvecs = np.linalg.eigh(covariance)
                eigvals = np.maximum(eigvals, 0.0)
                samples.append(
                    mean
                    + (
                        rng.standard_normal(size=(sample, 3))
                        * np.sqrt(eigvals)
                    )
                    @ eigvecs.T
                )
            X = np.vstack(samples)
        else:
            X = np.vstack(
                [
                    mean
                    + rng.standard_normal(size=(sample, 3))
                    * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        self.means, self.covariances, n_samples_comp
                    )
                ]
            )

        y = np.concatenate(
            [
                np.full(sample, j, dtype=int)
                for j, sample in enumerate(n_samples_comp)
            ]
        )
        return (X, y)


# G5M (2D/3D) functions and classes #
@njit
def _estimate_gaussian_parameters_diag_cov(
    X: lib.FloatArray2D,
    resp: lib.FloatArray2D,
    reg_covar: float = 1e-6,
) -> tuple[lib.FloatArray1D, lib.FloatArray2D, lib.FloatArray2D]:
    """Calculate the MLE parameters for a G5M. Assumes diagonal
    covariance matrices.

    Parameters
    ----------
    X : np.ndarray
        Data points.
    resp : np.ndarray
        Responsibilities of the G5M components, shape (n_samples,
        n_components).
    reg_covar : float, optional
        Regularization term for the covariance matrices. Default is
        1e-6.

    Returns
    -------
    nk, means, covariances : tuple
        Number of localizations per component, means and covariances.
    """
    nk = (
        _sum_along_axis0(resp, (resp.shape[1],))
        + 10.0 * np.finfo(resp.dtype).eps
    )
    means = _matmul(resp.T, X) / nk[:, np.newaxis]
    covariances = np.zeros((resp.shape[1], X.shape[1]), dtype=np.float64)
    for i in range(resp.shape[1]):
        for j in range(X.shape[1]):
            covariances[i, j] = (
                np.sum(resp[:, i] * (X[:, j] - means[i, j]) ** 2) / nk[i]
                + reg_covar
            )
    return (
        np.asarray(nk, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(covariances, dtype=np.float64),
    )


def _approximate_sem(g5m: G5M, locs: pd.DataFrame) -> np.ndarray:
    """Return the standard error of the means (SEM) in the G5M.

    Note: this is only an approximation since we treat each component
    independently and ignore the covariance between the
    components. The standard, single Gaussian SEM is used, i.e.,
    ``sigma / sqrt(n)``.

    Parameters
    ----------
    g5m : G5M
        Fitted G5M.
    locs : pd.DataFrame
        Localizations that g5m was fitted to.

    Returns
    -------
    sem : np.ndarray
        Array of standard errors of the means
        (n_components, n_dimensions).
    """
    weights = g5m.weights
    covariances = g5m.covariances

    if g5m.covariance_type == "spherical":
        covariances = np.repeat(covariances, 2).reshape(-1, 2)
    elif g5m.covariance_type == "rotated":
        # lpx/lpy/lpz are per camera axis, so the marginal variances
        # (the diagonal) are the right quantity even though the xy block
        # is rotated
        covariances = np.diagonal(covariances, axis1=1, axis2=2)
    N = len(locs) * weights.reshape(len(weights), -1)
    sem = np.sqrt(covariances / N)
    return sem


def _bootstrap_sem(
    g5m: G5M, locs: pd.DataFrame, n_bootstraps: int = 20
) -> np.ndarray:
    """Return the standard error of the means (SEM) for the G5M using
    bootstrapping.

    Parameters
    ----------
    g5m : G5M
        Fitted G5M.
    locs : pd.DataFrame
        Localizations that g5m was fitted to.
    n_bootstraps : int, optional
        Number of bootstrap rounds to perform. Default is 20.

    Returns
    -------
    sem : np.ndarray
        Array of standard errors of the means
        (n_components, n_dimensions).
    """
    np.random.seed(42)
    old_random_state = g5m.random_state
    g5m.random_state = None
    boot_means = []
    for i in range(n_bootstraps):
        X_boot = g5m.sample(len(locs))[0]
        angles = None
        if "z" in locs.columns:
            g5m_boot = G5M_3D(
                n_components=len(g5m.valid_idx),
                min_locs=g5m.min_locs,
                sigma_bounds=g5m.sigma_bounds,
                calibration=g5m.calibration,
                mode=g5m.mode,
                covariance_type=g5m.covariance_type,
                means_init=g5m.means,
            )
            lp = locs[["lpx", "lpy", "lpz"]].to_numpy()
            if g5m.covariance_type == "rotated":
                angles = np.deg2rad(locs["angle"].to_numpy())
        else:
            g5m_boot = G5M_2D(
                n_components=len(g5m.valid_idx),
                min_locs=g5m.min_locs,
                sigma_bounds=g5m.sigma_bounds,
                means_init=g5m.means,
            )
            lp = locs[["lpx", "lpy"]].mean(axis=1).to_numpy()
        g5m_boot.fit(
            X_boot,
            lp=lp,
            loc_prec_handle=g5m.loc_prec_handle,
            angles=angles,
        )
        if hasattr(g5m_boot, "means_"):  # converged
            boot_means.append(g5m_boot.means_)
    sem = np.std(boot_means, axis=0)
    g5m.random_state = old_random_state
    return sem


def _select_X_and_estep(
    g5m: G5M, locs_group: pd.DataFrame, pixelsize: float
) -> tuple:
    """Localization coordinates and the E-step matching the G5M's shape.

    Parameters
    ----------
    g5m : G5M
        Fitted G5M.
    locs_group : pd.DataFrame
        Localizations that g5m was fitted to.
    pixelsize : float
        Camera pixel size in nm.

    Returns
    -------
    X : np.ndarray
        ``(n_locs, 2)`` or ``(n_locs, 3)`` coordinates, z converted to
        camera pixels.
    e_step : callable
        ``_e_step_2D``, ``_e_step_3D`` or ``_e_step_3D_rot``, matching
        ``g5m.covariance_type``.
    """
    if "z" in locs_group.columns:
        X = locs_group[["x", "y", "z"]].to_numpy()
        X[:, 2] /= pixelsize  # convert z to camera pixels
        if g5m.covariance_type == "rotated":
            e_step = _e_step_3D_rot
        else:
            e_step = _e_step_3D
    else:
        X = locs_group[["x", "y"]].to_numpy()
        e_step = _e_step_2D
    return X, e_step


def _shape_columns(
    is_3d: bool,
    rotated: bool,
    means: np.ndarray,
    covariances: np.ndarray,
    sem: np.ndarray,
    resp: np.ndarray,
    rsum: np.ndarray,
    locs_group: pd.DataFrame,
    pixelsize: float,
) -> dict:
    """Fitted/relative width columns for the centers DataFrame.

    Handles the 2D/3D and diagonal/rotated covariance layouts; see
    :func:`_convert_G5M_results`.
    """
    if not is_3d:
        sigma = np.sqrt(covariances) * pixelsize
        lp = locs_group[["lpx", "lpy"]].mean(axis=1).to_numpy()
        weighted_lp = ((resp * lp.reshape(-1, 1)).sum(0) / rsum).reshape(-1)
        rel_sigma = sigma / weighted_lp / pixelsize
        return {"fitted_sigma": sigma, "rel_sigma": rel_sigma}

    z = means[:, 2] * pixelsize
    if rotated:
        # marginal (camera-axis) widths, so that fitted_sigma_x/y/z keep
        # the same meaning as for the diagonal model
        marginal = np.diagonal(covariances, axis1=1, axis2=2)
        sigma_x = np.sqrt(marginal[:, 0]) * pixelsize
        sigma_y = np.sqrt(marginal[:, 1]) * pixelsize
        sigma_z = np.sqrt(marginal[:, 2]) * pixelsize
        # principal axes of the xy block; eigh returns ascending
        # eigenvalues, so the major axis is the second one
        eigvals, eigvecs = np.linalg.eigh(covariances[:, :2, :2])
        eigvals = np.maximum(eigvals, 0.0)
        sigma_minor = np.sqrt(eigvals[:, 0]) * pixelsize
        sigma_major = np.sqrt(eigvals[:, 1]) * pixelsize
        axis_ratio = sigma_major / sigma_minor
        # orientation of the major axis, in the same convention as the
        # "angle" column written by picasso.localize: degrees wrapped
        # into [-90, 90)
        angle = np.rad2deg(np.arctan2(eigvecs[:, 1, 1], eigvecs[:, 0, 1]))
        angle = np.mod(angle + 90.0, 180.0) - 90.0
    else:
        sigma_x = np.sqrt(covariances[:, 0]) * pixelsize
        sigma_y = np.sqrt(covariances[:, 1]) * pixelsize
        sigma_z = np.sqrt(covariances[:, 2]) * pixelsize

    lpz = sem[:, 2] * pixelsize
    weighted_lpx = (
        (resp * locs_group["lpx"].to_numpy().reshape(-1, 1)).sum(0) / rsum
    ).reshape(-1)
    weighted_lpy = (
        (resp * locs_group["lpy"].to_numpy().reshape(-1, 1)).sum(0) / rsum
    ).reshape(-1)
    weighted_lpz = (
        (resp * locs_group["lpz"].to_numpy().reshape(-1, 1)).sum(0) / rsum
    ).reshape(-1)
    columns = {
        "z": z,
        "lpz": lpz,
        "fitted_sigma_x": sigma_x,
        "fitted_sigma_y": sigma_y,
        "fitted_sigma_z": sigma_z,
        "rel_sigma_x": sigma_x / weighted_lpx / pixelsize,
        "rel_sigma_y": sigma_y / weighted_lpy / pixelsize,
        "rel_sigma_z": sigma_z / weighted_lpz,
    }
    if rotated:
        # the principal axes do not correspond to the camera axes, so
        # they are normalized by the mean in-plane precision - the same
        # scalar the m-step bounds them with
        weighted_lpxy = 0.5 * (weighted_lpx + weighted_lpy)
        columns.update(
            fitted_sigma_major=sigma_major,
            fitted_sigma_minor=sigma_minor,
            rel_sigma_major=sigma_major / weighted_lpxy / pixelsize,
            rel_sigma_minor=sigma_minor / weighted_lpxy / pixelsize,
            axis_ratio=axis_ratio,
            angle=angle,
        )
    return columns


def _binding_event_counts(
    g5m: G5M, locs_group: pd.DataFrame, pixelsize: float, is_3d: bool
) -> np.ndarray:
    """Number of binding events assigned to each G5M component.

    A binding event links localizations that are contiguous in frame
    (up to 3 frames of no signal allowed) and is assigned to the G5M
    component closest to its center of mass.

    Parameters
    ----------
    g5m : G5M
        Fitted G5M.
    locs_group : pd.DataFrame
        Localizations of one cluster, expected in frame order.
    pixelsize : float
        Camera pixel size in nm, used to convert z to camera pixels.
    is_3d : bool
        Whether the localizations are 3D.

    Returns
    -------
    n_events : np.ndarray
        ``(n_components,)`` binding event count per component.
    """
    split_idx = np.where(np.diff(locs_group["frame"].to_numpy()) > 3)[0] + 1
    x_events = [
        np.mean(_) for _ in np.split(locs_group["x"].to_numpy(), split_idx)
    ]
    y_events = [
        np.mean(_) for _ in np.split(locs_group["y"].to_numpy(), split_idx)
    ]
    if is_3d:
        z_events = [
            np.mean(_) / pixelsize
            for _ in np.split(locs_group["z"].to_numpy(), split_idx)
        ]
        X_events = np.stack((x_events, y_events, z_events)).T
    else:
        X_events = np.stack((x_events, y_events)).T
    labels = g5m.predict(X_events)
    expected_labels = np.arange(len(g5m.valid_idx))
    found_labels, counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(found_labels, counts))
    return np.array([count_dict.get(_, 0) for _ in expected_labels])


def _add_mean_extra_columns(
    centers: pd.DataFrame,
    locs_group: pd.DataFrame,
    resp: np.ndarray,
    rsum: np.ndarray,
    ignore_columns: list,
) -> None:
    """Add the ``{col}_mean`` weighted-average columns to ``centers`` in place.

    Covers extra localization columns (e.g. photons) that would otherwise
    be lost in the conversion to centers.
    """
    for col in locs_group.columns:
        if col not in ignore_columns:
            centers[f"{col}_mean"] = (
                (resp * locs_group[col].to_numpy().reshape(-1, 1)).sum(0)
                / rsum
            ).reshape(-1)


def _ordered_center_columns(
    is_3d: bool, base: dict, shape_columns: dict
) -> dict:
    """Interleave ``base``/``shape_columns`` into the historical column order.

    ``base`` holds the columns common to 2D/3D (already computed by the
    caller); ``shape_columns`` is the output of :func:`_shape_columns`.
    """
    columns = {
        "frame": base["frame"],
        "std_frame": base["std_frame"],
        "x": base["x"],
        "y": base["y"],
    }
    if is_3d:
        columns["z"] = shape_columns["z"]
    columns["lpx"] = base["lpx"]
    columns["lpy"] = base["lpy"]
    if is_3d:
        for key in (
            "lpz",
            "fitted_sigma_x",
            "fitted_sigma_y",
            "fitted_sigma_z",
            "rel_sigma_x",
            "rel_sigma_y",
            "rel_sigma_z",
        ):
            columns[key] = shape_columns[key]
    else:
        columns["fitted_sigma"] = shape_columns["fitted_sigma"]
        columns["rel_sigma"] = shape_columns["rel_sigma"]
    for key in (
        "p_val",
        "mol_log_likelihood",
        "group_log_likelihood",
        "n_locs",
        "n_events",
        "group_input",
    ):
        columns[key] = base[key]
    return columns


def _convert_G5M_results(
    g5m: G5M,
    locs_group: pd.DataFrame,
    pixelsize: float = 130.0,
    bootstrap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract G5M components as ``pd.DataFrame`` in the format of
    localizations - frame, spatial coordinates, standard errors of the
    means, fitted sigma and number of localizations corresponding to
    the centers, etc.

    Parameters
    ----------
    g5m : G5M
        Fitted G5M.
    locs_group : pd.DataFrame
        Localizations that g5m was fitted to.
    pixelsize : float, optional
        Camera pixel size in nm. Default is 130.0.
    bootstrap : bool, optional
        If True, the standard error of the means (SEM) is calculated
        using bootstrapping. If False, the standard, single Gaussian
        SEM is used. Default is False.

    Returns
    -------
    centers : pd.DataFrame
        Centers of the G5M components in the format of localizations.
    clustered_locs : pd.DataFrame
        Localizations with assigned cluster labels, based on the G5M
        components.
    """
    locs_group = locs_group.copy()  # to avoid SettingWithCopyWarning
    means = g5m.means
    covariances = g5m.covariances
    weights = g5m.weights
    # find responsibilites which are used for weighted averaging of
    # properties per component
    X, e_step = _select_X_and_estep(g5m, locs_group, pixelsize)
    is_3d = X.shape[1] == 3
    log_prob = g5m.estimate_weighted_log_prob(X)
    sample_scores = _logsumexp_axis1(log_prob, (X.shape[0],))
    # average LL
    group_ll = np.ones(len(g5m.valid_idx)) * np.mean(sample_scores)

    _, log_resp = e_step(
        X,
        g5m.weights_,
        g5m.means_,
        g5m.precisions_cholesky_,
    )
    resp = np.exp(log_resp[:, g5m.valid_idx])  # only valid components
    rsum = resp.sum(0)
    # molecule log likelihood - weighted mean log likelihood of
    # localizations for each component
    mol_ll = (resp * log_prob).sum(0) / rsum

    # valid probability of the components - we know the expected value
    # and the standard deviation of the mean log likelihood of each
    # component, whose distirbution follows the normal distribution (due
    # to the central limit theorem). The valid probability is then
    # calculated as the cumulative distribution function of the normal
    # distribution with mu and sigma as the expected value and standard
    # deviation of the mean log likelihood of the component.
    # For x drawn from component k, E[log(w_k * N(x; mu_k, cov_k))] is
    # log(w_k / ((2*pi)**(D/2) * sqrt(det cov_k))) - D/2, since
    # E[chi2_D] = D. sqrt(det cov)
    n_dim = X.shape[1]
    norm = 2 * np.pi if n_dim == 2 else (2 * np.pi) ** 1.5
    expected = np.log(weights / (norm * g5m.sqrt_det_covariances)) - n_dim / 2
    stdev = np.sqrt(X.shape[1] * 0.5 / (len(X) * weights))
    # gauss CDF
    p_val = (
        0.5 * (1 + erf((mol_ll - expected) / (stdev * np.sqrt(2))))
    ).reshape(-1)

    # extract position of the centers
    x = means[:, 0]
    y = means[:, 1]

    # standard errors of the means, saved as loc. prec.
    if bootstrap:
        sem = _bootstrap_sem(g5m, locs_group)
    else:
        sem = _approximate_sem(g5m, locs_group)
    lpx = sem[:, 0]
    lpy = sem[:, 1]

    rotated = g5m.covariance_type == "rotated"
    shape_columns = _shape_columns(
        is_3d,
        rotated,
        means,
        covariances,
        sem,
        resp,
        rsum,
        locs_group,
        pixelsize,
    )

    # extract frame info and group_input
    frames_locs = np.reshape(locs_group["frame"].to_numpy(), (-1, 1))
    # weighted average of the frame
    frame = (resp * frames_locs).sum(0) / rsum
    # weighted std of the frame
    std_frame = np.sqrt(
        (resp * (frames_locs - frame) ** 2).sum(0)
        / ((resp.shape[0] - 1) * rsum / resp.shape[0])
    )
    labels = g5m.predict(X)
    # dbscan group id
    group_input = locs_group["group"].iloc[0] * np.ones(len(frame), dtype=int)
    locs_group["group_input"] = locs_group["group"].iloc[0] * np.ones(
        len(locs_group), dtype=int
    )
    # assign cluster labels to localizations
    locs_group["group"] = labels

    # assign log_likelihood and cluster labels to localizations
    log_likelihood = g5m.score_samples(X)
    locs_group["log_likelihood"] = log_likelihood

    # extract the number of binding events, i.e., link localizations
    # and assign them to molecules - sticky events will likely have only
    # one or two such events associated, then find the closest G5M
    # component to each event, accounting for a component with none
    n_events = _binding_event_counts(g5m, locs_group, pixelsize, is_3d)

    # convert to DataFrame
    base_columns = {
        "frame": frame.astype(np.float32),
        "std_frame": std_frame.astype(np.float32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
        "p_val": p_val.astype(np.float32),
        "mol_log_likelihood": mol_ll.astype(np.float32),
        "group_log_likelihood": group_ll.astype(np.float32),
        "n_locs": g5m.n_locs.astype(np.int32),
        "n_events": n_events.astype(np.int32),
        "group_input": group_input.astype(np.int32),
    }
    shape_columns = {
        key: value.astype(np.float32) for key, value in shape_columns.items()
    }
    centers = pd.DataFrame(
        _ordered_center_columns(is_3d, base_columns, shape_columns)
    )
    if rotated:
        # extra shape columns for the rotated model
        for key in (
            "fitted_sigma_major",
            "fitted_sigma_minor",
            "rel_sigma_major",
            "rel_sigma_minor",
            "axis_ratio",
            "angle",
        ):
            centers[key] = shape_columns[key]
    # add mean values of extra columns from locs_group so that
    # the info is not lost, e.g., mean photons
    ignore_columns = [
        "frame",
        "x",
        "y",
        "z",
        "lpx",
        "lpy",
        "lpz",
        "group",
        "group_input",
        "angle",
    ]
    _add_mean_extra_columns(centers, locs_group, resp, rsum, ignore_columns)
    return centers, locs_group


def sum_G5Ms(g5ms: list[G5M]) -> G5M:
    """Sum and normalize G5Ms. Assumes that all G5Ms gave the same
    input parameters, i.e., min_locs, min_sigma, max_sigma.

    Parameters
    ----------
    g5ms : list of G5M
        List of G5Ms to sum.

    Returns
    -------
    sum_g5m : G5M
        Summed G5Ms.
    """
    # check that all G5Ms are instances of G5m
    if not all(isinstance(_, G5M) for _ in g5ms):
        raise ValueError("All G5Ms must be instances of G5M.")

    # check that all G5Ms belong to the same class (2D/3D)
    if not all(isinstance(_, g5ms[0].__class__) for _ in g5ms):
        raise ValueError("All G5Ms must be of the same class (2D/3D).")

    # get weights
    n_locs = []
    for gm in g5ms:
        for n in gm.n_locs:
            n_locs.append(n)
    n_locs = np.array(n_locs).astype(float)
    weights = n_locs / n_locs.sum()
    # check that all G5Ms use the same component shape
    if not all(_.covariance_type == g5ms[0].covariance_type for _ in g5ms):
        raise ValueError("All G5Ms must have the same covariance_type.")

    # get means
    means = np.vstack([_.means for _ in g5ms])
    # get covariances, note that the shape of the covs array depends on
    # the covariance type: spherical -> (n_components,), diagonal ->
    # (n_components, 3), rotated -> (n_components, 3, 3)
    covariance_type = g5ms[0].covariance_type
    if covariance_type == "spherical":
        covs = np.hstack([_.covariances for _ in g5ms])
        pc = 1 / np.sqrt(covs)
    elif covariance_type == "diagonal":
        covs = np.stack([_.covariances for _ in g5ms]).reshape(len(weights), 3)
        pc = 1 / np.sqrt(covs)
    else:  # rotated
        covs = np.concatenate([_.covariances for _ in g5ms], axis=0)
        pc = _precision_chol_3D_rot(covs)

    kwargs = {}
    if isinstance(g5ms[0], G5M_3D):
        # G5M_2D takes no calibration/mode/covariance_type
        kwargs = {
            "calibration": g5ms[0].calibration,
            "mode": g5ms[0].mode,
            "covariance_type": covariance_type,
        }
    sum_g5m = g5ms[0].__class__(
        n_components=len(weights),
        min_locs=g5ms[0].min_locs,
        sigma_bounds=g5ms[0].sigma_bounds,
        **kwargs,
    )

    # set parameters (just like after fitting)
    valid_idx = np.arange(len(weights))
    sum_g5m.set_parameters(weights, means, covs, pc, True, valid_idx)
    sum_g5m.n_locs = n_locs
    return sum_g5m


@njit
def _fit_G5M(
    X: np.ndarray,
    min_locs: int,
    init_weights: np.ndarray,
    init_means: np.ndarray,
    init_precisions_cholesky: np.ndarray,
    sigma_bounds: tuple[float, float],
    *,
    lp: np.ndarray,
    loc_prec_handle: Literal["local", "abs"] = "local",
    cx: np.ndarray = np.array([]),
    cy: np.ndarray = np.array([]),
    mag_factor: float = 0.79,
    angles: np.ndarray = np.array([]),
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bool,
    np.ndarray,
]:
    """Fit G5M to the data X using the initial weights, means and
    precisions_cholesky. The function returns the fitted G5M parameters.

    Parameters
    ----------
    X : np.ndarray
        Data points, shape (n_samples, n_dimensions).
    min_locs : int
        Minimum number of localizations per component. Used to filter
        out components with too few localizations that likely represent
        background.
    init_weights : np.ndarray
        Initial weights of the G5M components. Shape (n_init,
        n_components).
    init_means : np.ndarray
        Initial means of the G5M components. Shape (n_init,
        n_components, n_dimensions).
    init_precisions_cholesky : np.ndarray
        Initial cholesky decomposition of precisions. Shape (n_init,
        n_components) for 2D data and (n_init, n_components, 3) for 3D
        data.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma.
    lp : np.ndarray
        Localization precision for each localization. Only used if
        loc_prec_handle is "local". Shape (n_samples,) for 2D and
        (n_samples, 3) for 3D data.
    loc_prec_handle : {"local", "abs"}, optional
        How to handle sigma bounds. If "local", localization precisions
        of points around each component are used to bound sigmas. Else,
        sigma_bounds specifies the absolute bounds on sigmas. Default
        is "local".
    cx, cy : np.ndarray, optional
        X and Y coefficients for astigmatism fitting. Required for 3D
        data only. See https://picassosr.readthedocs.io/en/latest/localize.html#d-calibration.  # noqa: E501
    mag_factor : float, optional
        Magnification factor for astigmatism fitting. Required for 3D
        data only.
    angles : np.ndarray, optional
        Angle of each localization in radians, shape (n_samples,). Used
        only by the rotated 3D model; empty otherwise.

    Returns
    -------
    weights, means, covariances, precisions_cholesky : tuple
        Fitted G5M parameters.
    converged : bool
        True if the G5M converged, False otherwise.
    valid_idx : np.ndarray
        Indices of the valid components (min_locs).
    """
    # The kernels are selected by the ndim of init_precisions_cholesky.
    # numba constant-folds .ndim for a direct argument and prunes the
    # dead branches, so each specialization compiles exactly one of
    # these. This works only while every model maps to a distinct ndim:
    #   2D spherical (n_init, K)       -> 2
    #   3D diagonal  (n_init, K, 3)    -> 3
    #   3D rotated   (n_init, K, 3, 3) -> 4
    # Adding a 2D diagonal (n_init, K, 2) or 2D rotated
    # (n_init, K, 2, 2) model would collide with the 3D entries above
    # and this dispatch would have to be replaced - e.g. by passing the
    # three kernels in as numba Dispatcher arguments chosen from a
    # Python-level table keyed on (n_dimensions, covariance_type). Note
    # that such callees must not use keyword-only arguments, which
    # cannot be bound through a Dispatcher-typed argument.
    if init_precisions_cholesky.ndim == 2:  # 2D data
        e_step = _e_step_2D
        m_step = _m_step_2D
        check_resolution = _check_G5M_resolution_2D
    elif init_precisions_cholesky.ndim == 3:
        e_step = _e_step_3D
        m_step = _m_step_3D
        check_resolution = _check_G5M_resolution_3D
    elif init_precisions_cholesky.ndim == 4:  # 3D, rotated xy block
        e_step = _e_step_3D_rot
        m_step = _m_step_3D_rot
        check_resolution = _check_G5M_resolution_3D_rot
    else:
        raise ValueError(
            "Only 2D and 3D data are supported. Data points suggest "
            f"{X.shape[1]} dimensions. The initial precisions suggest "
            f"{init_precisions_cholesky.ndim} dimensions. 3D data "
            "requires a calibration dictionary with the keys 'X Coefficients',"
            " 'Y Coefficients' and 'Magnification factor'."
        )

    converged = False
    # best log-likelihood for all inits
    max_lower_bound = -np.inf
    # best parameters for all inits
    best_params = (None, None, None, None)
    # valid components (min_locs)
    valid_idx = np.arange(init_means.shape[1]).astype(np.int32)

    # run the procedure n_init times
    for ii in range(len(init_weights)):
        # initialize
        weights = init_weights[ii]
        means = init_means[ii]
        precisions_cholesky = init_precisions_cholesky[ii]

        # fit G5M
        lower_bound = -np.inf
        converged_ = False
        for _ in range(100):  # max_iter=100
            prev_lower_bound = lower_bound
            log_prob_norm, log_resp = e_step(
                X,
                weights,
                means,
                precisions_cholesky,
            )
            (weights, means, covariances, precisions_cholesky) = m_step(
                X,
                log_resp,
                sigma_bounds=sigma_bounds,
                lp=lp,
                loc_prec_handle=loc_prec_handle,
                cx=cx,
                cy=cy,
                mag_factor=mag_factor,
                angles=angles,
            )
            lower_bound = log_prob_norm
            change = lower_bound - prev_lower_bound

            if abs(change) < 1e-3:
                converged_ = True
                break

        # extract the valid components (min_locs)
        n_locs = np.float64(len(X))
        n = np.round(weights * n_locs).astype(np.int32)
        valid_idx_ = (np.where(n >= min_locs)[0]).astype(np.int32)
        # check if FWHM limit is passed
        resolution_pass = check_resolution(
            means[valid_idx_],
            weights[valid_idx_],
            precisions_cholesky[valid_idx_],
        )
        # check if the current result is the best
        if resolution_pass and (
            lower_bound > max_lower_bound or max_lower_bound == -np.inf
        ):
            max_lower_bound = lower_bound
            best_params = (weights, means, covariances, precisions_cholesky)
            converged = converged_
            valid_idx = valid_idx_

    return best_params, converged, valid_idx


def _run_g5m_in_clusters(
    i: int,
    n_groups_task: int,
    locs: pd.DataFrame,
    min_locs: int,
    loc_prec_handle: Literal["local", "abs"],
    sigma_bounds: tuple[float, float],
    pixelsize: float,
    max_rounds_without_best_bic: int,
    bootstrap_check: bool,
    calibration: dict | None,
    max_locs_per_cluster: int,
    mode: Literal["astigmatism", "spline"] = "astigmatism",
    covariance_type: Literal["diagonal", "rotated"] = "diagonal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run G5M for a given group of localizations clusters. See
    :func:`g5m` for a detailed explanation of the parameters.

    Note that the arguments are passed positionally by
    ``_run_g5m_parallel``, so their order must be kept in sync there.

    Parameters
    ----------
    i : int
        Index of the first group to analyze.
    n_groups_task : int
        Number of groups to analyze.
    locs : pd.DataFrame
        Clustered localizations with a "group" column.
    min_locs : int
        Minimum number of localizations per component.
    loc_prec_handle : {"local", "abs"}
        How to handle sigma bounds, see :func:`g5m`.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components, see :func:`g5m`.
    pixelsize : float
        Camera pixel size in nm.
    max_rounds_without_best_bic : int
        Maximum number of rounds without BIC improvement to terminate
        the search for optimal G5M n_components.
    bootstrap_check : bool
        If True, the SEM is calculated using bootstrapping.
    calibration : dict or None
        Astigmatism calibration dictionary. Only used for 3D data with
        ``mode="astigmatism"``.
    max_locs_per_cluster : int
        Maximum number of localizations per cluster accepted for G5M.
    mode : {"astigmatism", "spline"}, optional
        Fitting mode of the input 3D localizations. Default is
        "astigmatism".
    covariance_type : {"diagonal", "rotated"}, optional
        Shape of the 3D G5M components. Default is "diagonal".

    Returns
    -------
    centers : list of pd.DataFrames
        Centers of the G5M components in the format of localizations.
        Each element corresponds to one cluster of localizations.
    clustered_locs : list of pd.DataFrames
        Localizations with assigned cluster labels, based on the G5M
        components.
    """
    centers = []
    clustered_locs = []
    for group in np.unique(locs.group)[i : i + n_groups_task]:
        if "z" in locs.columns:
            centers_, clustered_locs_ = _run_g5m_group_3D(
                locs_group=locs[locs["group"] == group],
                calibration=calibration,
                min_locs=min_locs,
                loc_prec_handle=loc_prec_handle,
                sigma_bounds=sigma_bounds,
                pixelsize=pixelsize,
                mode=mode,
                covariance_type=covariance_type,
                max_rounds_without_best_bic=max_rounds_without_best_bic,
                bootstrap_check=bootstrap_check,
                max_locs_per_cluster=max_locs_per_cluster,
            )
        else:
            centers_, clustered_locs_ = _run_g5m_group_2D(
                locs_group=locs[locs["group"] == group],
                min_locs=min_locs,
                loc_prec_handle=loc_prec_handle,
                sigma_bounds=sigma_bounds,
                pixelsize=pixelsize,
                max_rounds_without_best_bic=max_rounds_without_best_bic,
                bootstrap_check=bootstrap_check,
                max_locs_per_cluster=max_locs_per_cluster,
            )
        if centers_ is not None and len(centers_):
            centers.append(centers_)
            clustered_locs.append(clustered_locs_)
    return centers, clustered_locs


def _run_g5m_parallel(
    locs: pd.DataFrame,
    *,
    min_locs: int = MIN_LOCS,
    loc_prec_handle: Literal["local", "abs"] = "local",
    sigma_bounds: tuple[float, float] = (MIN_SIGMA_FACTOR, MAX_SIGMA_FACTOR),
    pixelsize: float = 130.0,
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
    bootstrap_check: bool = False,
    calibration: dict | None = None,
    max_locs_per_cluster: int = np.inf,
    mode: Literal["astigmatism", "spline"] = "astigmatism",
    covariance_type: Literal["diagonal", "rotated"] = "diagonal",
) -> list:
    """Run G5M in parallel using multiprocessing. See :func:`g5m` for a
    detailed explanation of the parameters.

    Parameters
    ----------
    locs : pd.DataFrame
        Clustered localizations with a "group" column.
    min_locs : int, optional
        Minimum number of localizations per component. Default is
        `MIN_LOCS`.
    loc_prec_handle : {"local", "abs"}, optional
        How to handle sigma bounds, see :func:`g5m`. Default is
        "local".
    sigma_bounds : tuple, optional
        Bounds for the standard deviation (sigma) of the Gaussian
        components, see :func:`g5m`. Default is `(MIN_SIGMA_FACTOR,
        MAX_SIGMA_FACTOR)`.
    pixelsize : float, optional
        Camera pixel size in nm. Default is 130.0.
    max_rounds_without_best_bic : int, optional
        Maximum number of rounds without BIC improvement to terminate
        the search for optimal G5M n_components. Default is
        `MAX_ROUNDS_WITHOUT_BEST_BIC`.
    bootstrap_check : bool, optional
        If True, the SEM is calculated using bootstrapping. Default is
        False.
    calibration : dict or None, optional
        Astigmatism calibration dictionary. Only used for 3D data with
        ``mode="astigmatism"``. Default is None.
    max_locs_per_cluster : int, optional
        Maximum number of localizations per cluster accepted for G5M.
        Default is np.inf.
    mode : {"astigmatism", "spline"}, optional
        Fitting mode of the input 3D localizations. Default is
        "astigmatism".
    covariance_type : {"diagonal", "rotated"}, optional
        Shape of the 3D G5M components. Default is "diagonal".

    Returns
    -------
    fs : list
        List of futures.
    """
    n_groups = len(np.unique(locs["group"]))
    n_workers = lib.n_workers(0.35)
    groups_per_task = [
        (
            int(n_groups / N_TASKS + 1)
            if _ < n_groups % N_TASKS
            else int(n_groups / N_TASKS)
        )
        for _ in range(N_TASKS)
    ]
    start_indices = np.cumsum([0] + groups_per_task[:-1])
    fs = []
    executor = ProcessPoolExecutor(n_workers)
    for i, n_groups_task in zip(start_indices, groups_per_task):
        fs.append(
            executor.submit(
                _run_g5m_in_clusters,
                i,
                n_groups_task,
                locs,
                min_locs,
                loc_prec_handle,
                sigma_bounds,
                pixelsize,
                max_rounds_without_best_bic,
                bootstrap_check,
                calibration,
                max_locs_per_cluster,
                mode,
                covariance_type,
            )
        )
    return fs


def _g5m(
    locs: pd.DataFrame,
    min_locs: int,
    loc_prec_handle: Literal["local", "abs"],
    sigma_bounds: tuple[float, float],
    pixelsize: float,
    max_rounds_without_best_bic: int,
    bootstrap_check: bool,
    calibration: dict | None,
    max_locs_per_cluster: int,
    asynch: bool,
    n_steps: int,
    progress: lib.ProgressType,
    mode: Literal["astigmatism", "spline"] = "astigmatism",
    covariance_type: Literal["diagonal", "rotated"] = "diagonal",
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Run G5M with or without multiprocessing. The function returns the
    centers of the G5M components and localizations with assigned cluster
    labels. See ``g5m`` for parameters explanation."""
    if asynch:  # run G5M using multiprocessing
        fs = _run_g5m_parallel(
            locs,
            min_locs=min_locs,
            loc_prec_handle=loc_prec_handle,
            sigma_bounds=sigma_bounds,
            pixelsize=pixelsize,
            max_rounds_without_best_bic=max_rounds_without_best_bic,
            bootstrap_check=bootstrap_check,
            calibration=calibration,
            max_locs_per_cluster=max_locs_per_cluster,
            mode=mode,
            covariance_type=covariance_type,
        )

        # display progress
        while lib.n_futures_done(fs) < n_steps:
            n_done = lib.n_futures_done(fs)
            progress.set_value(n_done)
            time.sleep(0.2)

        # extract centers from futures
        centers = [_.result()[0] for _ in fs if len(_.result())]
        centers = list(itchain(*centers))
        clustered_locs = [_.result()[1] for _ in fs if len(_.result())]
        clustered_locs = list(itchain(*clustered_locs))

    else:  # run G5M without multiprocessing
        centers = []
        clustered_locs = []
        for i, group in enumerate(np.unique(locs["group"])):
            if "z" in locs.columns:
                centers_, clustered_locs_ = _run_g5m_group_3D(
                    locs[locs["group"] == group],
                    calibration=calibration,
                    min_locs=min_locs,
                    loc_prec_handle=loc_prec_handle,
                    sigma_bounds=sigma_bounds,
                    pixelsize=pixelsize,
                    mode=mode,
                    covariance_type=covariance_type,
                    max_rounds_without_best_bic=max_rounds_without_best_bic,
                    bootstrap_check=bootstrap_check,
                    max_locs_per_cluster=max_locs_per_cluster,
                )
            else:
                centers_, clustered_locs_ = _run_g5m_group_2D(
                    locs[locs["group"] == group],
                    min_locs=min_locs,
                    loc_prec_handle=loc_prec_handle,
                    sigma_bounds=sigma_bounds,
                    pixelsize=pixelsize,
                    max_rounds_without_best_bic=max_rounds_without_best_bic,
                    bootstrap_check=bootstrap_check,
                    max_locs_per_cluster=max_locs_per_cluster,
                )
            if centers_ is not None and len(centers_):
                centers.append(centers_)
                clustered_locs.append(clustered_locs_)

            progress.set_value(i)

    # complete and close the progress tracker
    progress.set_value(n_steps)
    progress.close()

    return centers, clustered_locs


def _resolve_covariance_type(
    covariance_type: Literal["auto", "spherical", "diagonal", "rotated"],
    locs: pd.DataFrame,
    mode: Literal["astigmatism", "spline"],
) -> Literal["spherical", "diagonal", "rotated"]:
    """Resolve the ``"auto"`` sentinel and validate the requested
    covariance type against the data.

    "auto" selects "rotated" for 3D astigmatism localizations that carry
    an ``angle`` column, i.e. those fitted with the rotated elliptical
    Gaussian PSF model. For such data the astigmatic stretch does not run
    along the camera axes, so the axis-aligned model would apply the
    calibration in the wrong frame. Everything else keeps the model it
    has always used.

    Parameters
    ----------
    covariance_type : {"auto", "spherical", "diagonal", "rotated"}
        Requested covariance type.
    locs : pd.DataFrame
        Localizations, used to detect dimensionality and the presence of
        an "angle" column.
    mode : {"astigmatism", "spline"}
        Fitting mode of the input localizations (3D only).

    Returns
    -------
    covariance_type : {"spherical", "diagonal", "rotated"}
        Resolved covariance type.
    """
    is_3d = "z" in locs.columns
    has_angle = "angle" in locs.columns

    if covariance_type not in ("auto",) + COVARIANCE_TYPES:
        raise ValueError(
            f"covariance_type must be 'auto' or one of {COVARIANCE_TYPES}, "
            f"got '{covariance_type}'."
        )

    if covariance_type == "auto":
        if not is_3d:
            return "spherical"
        if mode == "astigmatism" and has_angle:
            return "rotated"
        return "diagonal"

    # explicitly requested - validate it against the data
    if covariance_type == "spherical" and is_3d:
        raise ValueError(
            "covariance_type='spherical' is not available for 3D data: "
            "the axial localization precision is several times worse "
            "than the lateral one, so an isotropic component is not "
            "meaningful. Use 'diagonal' or 'rotated'."
        )
    if covariance_type in ("diagonal", "rotated") and not is_3d:
        raise ValueError(
            f"covariance_type='{covariance_type}' is not available for 2D "
            "data. Use 'spherical'."
        )
    if covariance_type == "rotated":
        if mode != "astigmatism":
            raise ValueError(
                "covariance_type='rotated' requires mode='astigmatism'. "
                "The rotated model couples the two principal axes of the "
                "xy block via the astigmatism calibration, which spline "
                "fitting does not provide. Use mode='astigmatism', or "
                "covariance_type='diagonal'."
            )
        if not has_angle:
            raise ValueError(
                "covariance_type='rotated' requires an 'angle' column in "
                "the localizations. It is written by picasso.localize "
                "when fitting with a rotated elliptical Gaussian model "
                "(e.g. 'gausslq-rotated'). Use covariance_type='diagonal' "
                "for localizations fitted without rotation."
            )
    return covariance_type


def _validate_g5m_args(
    loc_prec_handle: str,
    sigma_bounds: tuple,
    group_column: str,
    mode: str,
    locs: pd.DataFrame,
) -> None:
    """Validate the argument combination :func:`g5m` was called with."""
    assert loc_prec_handle in [
        "local",
        "abs",
    ], "loc_prec_handle must be 'local' or 'abs'."
    assert (
        len(sigma_bounds) == 2
    ), "sigma_bounds must be a tuple of two values."
    assert (
        sigma_bounds[0] <= sigma_bounds[1]
    ), "sigma_bounds[0] must not be larger than sigma_bounds[1]."
    assert group_column in [
        "group",
        "group_input",
    ], "group_column must be 'group' or 'group_input'."
    assert group_column in locs.columns, (
        f"Localizations must be grouped. Column '{group_column}' not "
        "found. Use DBSCAN or similar."
    )
    assert mode in [
        "astigmatism",
        "spline",
    ], "mode must be 'astigmatism' or 'spline'."


def _resolve_group_column(
    locs: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    """Copy ``group_column`` into "group" if a different column was named.

    G5M works on the "group" column internally; if a different column is
    requested (e.g. because "group" was overwritten), copy it over.
    """
    if group_column != "group":
        locs = locs.copy()
        locs["group"] = locs[group_column].to_numpy()
    return locs


def _build_g5m_progress(callback_parent, n_steps: int):
    """Progress tracker for :func:`g5m`, matching ``lib.normalize_progress``.

    A parent widget builds a ``lib.ProgressDialog`` for it; "console" uses
    tqdm; a ready-made tracker (anything with the ``ProgressDialog``
    interface) is driven directly instead of being wrapped.
    """
    if callback_parent is None or callback_parent == "console":
        progress = lib.normalize_progress(
            callback_parent, description="Running G5M..."
        )
        progress.setMaximum(n_steps)
    elif callable(getattr(callback_parent, "set_value", None)):
        progress = callback_parent
        progress.zero_progress("Running G5M...")
        progress.setMaximum(n_steps)
    else:  # a parent widget: build the dialog for it
        progress = lib.ProgressDialog(
            "Running G5M...", 0, n_steps, callback_parent
        )
        progress.set_value(0)
    return progress


def _g5m_info(
    locs: pd.DataFrame,
    centers: pd.DataFrame,
    min_locs: int,
    max_rounds_without_best_bic: int,
    bootstrap_check: bool,
    covariance_type: str,
    loc_prec_handle: str,
    sigma_bounds: tuple,
    pixelsize: float,
    mode: str,
    calibration: dict | None,
) -> dict:
    """The info dictionary :func:`g5m` appends for one run."""
    new_info = {
        "Generated by": f"Picasso v{__version__} G5M",
        "Model determination": "BIC",
        "Number of molecules": int(len(centers)),
        "Min. no. locs per molecule": int(min_locs),
        "Max. rounds w/o BIC improvement": int(max_rounds_without_best_bic),
        "Bootstrap SEM": bool(bootstrap_check),
        "Initialization method": "KMeans++",
        # the resolved type, not the requested one, so the saved file
        # always records the model that actually ran
        "Covariance type": covariance_type,
        "Filtered": False,
    }
    if loc_prec_handle == "local":
        new_info["Sigma bounds (factors)"] = [float(_) for _ in sigma_bounds]
        new_info["Sigma bounds method"] = "Local"
    else:
        new_info["Sigma bounds (nm)"] = [
            float(sigma_bounds[0] * pixelsize),
            float(sigma_bounds[1] * pixelsize),
        ]
        new_info["Sigma bounds method"] = "Abs"
    if "z" in locs.columns:
        new_info["Fit mode"] = mode
        if mode == "astigmatism":
            new_info["X Coefficients"] = [
                float(_) for _ in calibration["X Coefficients"]
            ]
            new_info["Y Coefficients"] = [
                float(_) for _ in calibration["Y Coefficients"]
            ]
            new_info["Magnification factor"] = float(
                calibration["Magnification factor"]
            )
    return new_info


def _postprocess_g5m(
    centers: pd.DataFrame, clustered_locs: pd.DataFrame, info: list[dict]
) -> tuple:
    """Filter G5M centers by mean frame, std frame, p-value and n_events."""
    n_frames = info[0]["Frames"]
    min_std_frame = 0.1 * n_frames
    min_pval = 0.015
    min_n_events = 3
    idx = (
        (centers["std_frame"] > min_std_frame)
        & (centers["p_val"] > min_pval)
        & (centers["n_events"] > min_n_events)
    )
    centers = centers[idx]
    clustered_locs = clustered_locs[
        np.isin(clustered_locs["group"], np.arange(len(idx))[idx])
    ]
    info[-1]["Filtered"] = True
    info[-1]["Filter; min. std frame"] = float(min_std_frame)
    info[-1]["Filter; min. p value"] = float(min_pval)
    info[-1]["Filter; min. n_events"] = int(min_n_events)
    return centers, clustered_locs


def g5m(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    min_locs: int = MIN_LOCS,
    loc_prec_handle: Literal["local", "abs"] = "local",
    sigma_bounds: tuple[float, float] = (MIN_SIGMA_FACTOR, MAX_SIGMA_FACTOR),
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
    bootstrap_check: bool = False,
    calibration: dict | None = None,
    mode: Literal["astigmatism", "spline"] = "astigmatism",
    covariance_type: Literal[
        "auto", "spherical", "diagonal", "rotated"
    ] = "auto",
    postprocess: bool = True,
    max_locs_per_cluster: int = np.inf,
    asynch: bool = True,
    group_column: Literal["group", "group_input"] = "group",
    callback_parent: (
        QtWidgets.QMainWindow | Literal["console"] | None
    ) = "console",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Run G5M with or without multiprocessing. The function returns
    the centers of the G5M components and localizations with assigned
    cluster labels.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list
        Information dictionaries.
    min_locs : int, optional
        Minimum number of localizations per component. Used to filter
        out components with too few localizations that likely represent
        background. Default is `MIN_LOCS`.
    loc_prec_handle : {"local", "abs"}, optional
        How to handle sigma bounds. If "local", localization precisions
        of points around each component are used to bound sigmas. Else,
        sigma_bounds specifies the absolute bounds on sigmas. Default
        is "local".
    sigma_bounds : tuple, optional
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma. Default is `(MIN_SIGMA_FACTOR,
        MAX_SIGMA_FACTOR)`.
    max_rounds_without_best_bic : int, optional
        Maximum number of rounds without BIC improvement to terminate
        the search for optimal G5M n_components. Default is 3.
    bootstrap_check : bool, optional
        If True, the standard error of the means (SEM) is calculated
        using bootstrapping. If False, the standard, single Gaussian SEM
        is used as approximation. Default is False.
    calibration : dict, optional
        Astigmatism calibration dictionary with x and y coefficients and
        magnification factor. Only required for 3D data fit with
        ``mode="astigmatism"``. Ignored for spline. Default is None.
    mode : {"astigmatism", "spline"}, optional
        Fitting mode of the input 3D localizations. "astigmatism"
        couples the x/y covariances via the calibration polynomials and
        requires ``calibration``; "spline" uses a plain diagonal 3D
        model (independent x/y/z covariances) and reads z/lpz directly
        from the localizations, requiring no calibration. Ignored for 2D
        data. Default is "astigmatism".
    covariance_type : {"auto", "spherical", "diagonal", "rotated"}, optional
        Shape of the fitted loc. clouds. "spherical" is the isotropic
        2D model and "diagonal" the axis-aligned 3D one. "rotated"
        orients each component's xy covariance along the angle reported
        by the localizations, for 3D astigmatism data localized with the
        rotated elliptical Gaussian PSF model. It requires 3D localizations,
        ``mode="astigmatism"`` and an "angle" column, and adds the
        columns "fitted_sigma_major", "fitted_sigma_minor",
        "rel_sigma_major", "rel_sigma_minor", "axis_ratio" and "angle"
        to the output. "auto" selects "rotated" for 3D astigmatism data
        with the "angle" column; "diagonal" for 3D astigmatism data
        without the column and "spherical" for 2D data. Default is "auto".
    postprocess : bool, optional
        If True, the G5M components are postprocessed to remove likely
        sticky events (mean frame, std frame, n_events filtering).
        Additionally, filters by p_val to dismiss poorly fitted
        components. Default is True.
    max_locs_per_cluster : int, optional
        Maximum number of localizations per cluster accepted for G5M.
        Used to avoid fitting to fiducial markers. Such clusters are
        ignored. Default is np.inf.
    asynch : bool, optional
        If True, G5M is run in parallel using multiprocessing. Default
        is True.
    group_column : {"group", "group_input"}, optional
        Name of the column used to group localizations into clusters
        prior to G5M. Use "group_input" when the "group" column has been
        overwritten but the original cluster ids are preserved in
        "group_input". Default is "group".
    callback_parent : QMainWindow, ProgressType, "console" or None, optional
        Where progress is reported. A parent widget builds a
        ``lib.ProgressDialog`` for it; "console" uses tqdm; None
        displays nothing. A ready-made progress tracker (anything with
        the ``ProgressDialog`` interface, see
        ``lib.normalize_progress``) can also be passed and is driven
        directly. Default is "console".

    Returns
    -------
    centers : pd.DataFrame
        Centers of the G5M components in the format of localizations.
    clustered_locs : pd.DataFrame
        Localizations with assigned cluster labels, based on the G5M
        components.
    info : list
        Updated information dictionaries.
    """
    _validate_g5m_args(loc_prec_handle, sigma_bounds, group_column, mode, locs)
    locs = _resolve_group_column(locs, group_column)

    # resolve "auto" once, here, so that everything downstream (and the
    # metadata) sees the concrete model that was actually used
    covariance_type = _resolve_covariance_type(covariance_type, locs, mode)

    pixelsize = lib.get_from_metadata(info, "Pixelsize")
    if pixelsize is None:
        raise ValueError("Camera pixel size must be provided in info.")

    # astigmatism 3D data requires a calibration; spline 3D data recovers
    # z (and lpz) directly, so no calibration is needed
    if "z" in locs.columns and mode == "astigmatism" and calibration is None:
        raise ValueError(
            "Calibration dictionary must be provided for astigmatism 3D "
            "data. The dictionary must specify 'X Coefficients' and 'Y "
            "Coefficients' and 'Magnification factor'. See "
            "https://picassosr.readthedocs.io/en/latest/localize.html#d-calibration"  # noqa: E501
        )

    # determine how many steps are displayed in the progress bar
    n_steps = N_TASKS if asynch else len(np.unique(locs["group"]))
    # everything below drives the same ProgressDialog-like interface (see
    # lib.normalize_progress), so a ready-made tracker can be passed
    # instead of a parent window.
    progress = _build_g5m_progress(callback_parent, n_steps)

    centers, clustered_locs = _g5m(
        locs,
        min_locs=min_locs,
        loc_prec_handle=loc_prec_handle,
        sigma_bounds=sigma_bounds,
        pixelsize=pixelsize,
        max_rounds_without_best_bic=max_rounds_without_best_bic,
        bootstrap_check=bootstrap_check,
        calibration=calibration,
        max_locs_per_cluster=max_locs_per_cluster,
        asynch=asynch,
        n_steps=n_steps,
        progress=progress,
        mode=mode,
        covariance_type=covariance_type,
    )
    # stack centers to form a pd.DataFrame in the format of localizations
    if len(centers):
        centers = pd.concat(centers, ignore_index=True)
    else:  # no molecules found, return None
        return None, None, info
    # assing group ids to the clustered localizations
    max_label = 0
    for i, clustered_locs_ in enumerate(clustered_locs):
        clustered_locs_["group"] += max_label
        max_label = clustered_locs_["group"].max() + 1
        clustered_locs[i] = clustered_locs_
    clustered_locs = pd.concat(clustered_locs, ignore_index=True)

    # update info
    new_info = _g5m_info(
        locs,
        centers,
        min_locs,
        max_rounds_without_best_bic,
        bootstrap_check,
        covariance_type,
        loc_prec_handle,
        sigma_bounds,
        pixelsize,
        mode,
        calibration,
    )
    info = info + [new_info]
    if postprocess:
        centers, clustered_locs = _postprocess_g5m(
            centers, clustered_locs, info
        )
    return centers, clustered_locs, info
