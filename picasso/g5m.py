"""
picasso.g5m
~~~~~~~~~~~

Gaussian Mixture Modelling with Modifications for Molecular Mapping
(G5M). Published in: TODO: add DOI

G5M is based on the sklearn implementation of Gaussian Mixture Modeling
(GMM) with numba optimizations for fitting, as well as for kmeans++
initialization. Several modifications for molecular mapping in DNA-PAINT
are added, for example, localization cloud shape modeling.

:authors: Rafal Kowalewski, 2023-2025
:copyright: Copyright (c) 2023-2025 Jungmann Lab, MPI Biochemistry
"""

from __future__ import annotations

import os
import time
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from itertools import chain as itchain
from typing import Literal

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import erf
from sklearn.utils import check_random_state
from tqdm import tqdm
from PyQt5 import QtWidgets

from . import lib, zfit, __version__

# default min. number of localizations per molecule
MIN_LOCS = 15
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


# helper functions for numba operations along axes
fastmath = True


@njit(fastmath=fastmath)
def max_along_axis1(X: np.ndarray, final_shape: tuple[int]) -> np.ndarray:
    output = np.zeros(final_shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        output[i] = np.max(X[i])
    return output


@njit(fastmath=fastmath)
def sum_along_axis0(X: np.ndarray, final_shape: tuple[int]) -> np.ndarray:
    output = np.zeros(final_shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        output += X[i]
    return output


@njit(fastmath=fastmath)
def sum_along_axis1(X: np.ndarray, final_shape: tuple[int]) -> np.ndarray:
    output = np.zeros(final_shape, dtype=X.dtype)
    for i in range(X.shape[1]):
        output += X[:, i]
    return output


@njit(fastmath=fastmath)
def mean_along_axis1(X: np.ndarray, final_shape: tuple[int]) -> np.ndarray:
    output = sum_along_axis1(X, final_shape)
    return output / X.shape[1]


@njit(fastmath=fastmath)
def logsumexp_axis1(X: np.ndarray, final_shape: tuple[int]) -> np.ndarray:
    """njit implementation of ``scipy.special.logsumexp``. Note that we
    cannot use ``np.log(np.sum(np.exp(X), axis=1))`` because it will
    cause overflow for large numbers. Thus, we use the ``logsumexp``
    formula: ``log(sum(exp(X))) = log(sum(exp(X - max(X))) + max(X)``
    where ``max(X)`` is subtracted from X to avoid overflow."""
    max_val = max_along_axis1(X, final_shape)
    exp_values = np.exp(X - max_val[:, np.newaxis])
    exp_sum = sum_along_axis1(exp_values, final_shape)
    output = np.log(exp_sum) + max_val
    return output


@njit(fastmath=fastmath)
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
def square_elements_1d(X: np.ndarray) -> np.ndarray:
    output = np.zeros(X.shape, dtype=X.dtype)
    for i in range(X.shape[0]):
        output[i] = X[i] ** 2
    return output


@njit(fastmath=fastmath)
def square_elements_2d(X: np.ndarray) -> np.ndarray:
    m, n = X.shape
    output = np.zeros((m, n), dtype=X.dtype)
    for i in range(m):
        for j in range(n):
            output[i, j] = X[i, j] ** 2
    return output


# In sklearn's GaussianMixture implementation, the term in the nominator
# of the exponential term ((x - mu)^2 / sigma^2) is calculated as
# (x^2 - 2*x*mu + mu^2). This can cause numerical instability when x and
# mu are large, which is the case for our data. To avoid this, we first
# calculate (x - mu) and then square it (x and mu values are similar,
# thus the instability is avoided). Note: precision = 1/sigma**2
@njit(fastmath=fastmath)
def gauss_exponential_term_2D(
    X: np.ndarray,  # shape (n_samples, 2)
    means: np.ndarray,  # shape (n_components, 2)
    precision: np.ndarray,  # shape (n_components,)
) -> np.ndarray:
    n_samples = X.shape[0]
    n_components = means.shape[0]
    sq_diff = np.zeros((n_samples, n_components), dtype=X.dtype)
    for i in range(n_samples):
        for j in range(n_components):
            for k in range(2):
                sq_diff[i, j] += (X[i, k] - means[j, k]) ** 2 * precision[j]
    return sq_diff


@njit(fastmath=fastmath)
def gauss_exponential_term_3D(
    X: np.ndarray,  # shape (n_samples, 3)
    means: np.ndarray,  # shape (n_components, 3)
    precision: np.ndarray,  # shape (n_components, 3)
) -> np.ndarray:
    """Same as ``gauss_exponential_term_2D`` but precision has shape
    (K, 3), where K is the number of components."""
    n_samples = X.shape[0]
    n_components = means.shape[0]
    sq_diff = np.zeros((n_samples, n_components), dtype=X.dtype)
    for i in range(n_samples):
        for j in range(n_components):
            for k in range(3):
                sq_diff[i, j] += (X[i, k] - means[j, k]) ** 2 * precision[j, k]
    return sq_diff


# kmeans++ init, adopted from sklearn, numba implementation #
@njit
def euclidean_distances(
    X: np.ndarray,
    Y: np.ndarray,
    X_norm_squared: np.ndarray | None = None,
    Y_norm_squared: np.ndarray | None = None,
) -> np.ndarray:
    """njit implementation of
    ``sklearn.metrics.pairwise._euclidean_distances`` with
    ``squared=True``."""
    if X_norm_squared is not None:
        XX = X_norm_squared.reshape(-1, 1)
    else:
        XX = sum_along_axis1(square_elements_2d(X), (X.shape[0],))[
            :, np.newaxis
        ]

    if Y is X:
        YY = None if XX is None else XX.T
    else:
        if Y_norm_squared is not None:
            YY = Y_norm_squared.reshape(1, -1)
        else:
            YY = sum_along_axis1(square_elements_2d(Y), (Y.shape[0],))[
                :, np.newaxis
            ]

    distances = -2 * matmul(X, Y.T)
    distances += XX
    distances += YY
    distances = np.maximum(distances, 0)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances  # squared distances


@njit
def kmeans_plusplus(
    X: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """njit implementation of ``sklearn.cluster._kmeans_plusplus``. Used
    for initializing ``G5M``'s."""
    np.random.seed(random_state)

    n_samples, n_dimensions = X.shape
    centers = np.empty((n_components, n_dimensions), dtype=X.dtype)
    n_local_trials = 2 + int(np.log(n_components))
    x_squared_norms = sum_along_axis1(square_elements_2d(X), (X.shape[0],))

    # Pick first center randomly and track index of point
    center_id = np.random.choice(n_samples)
    indices = np.full(n_components, -1, dtype=np.int64)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
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
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms
        )

        # update closest distances squared and potential for each candidate
        distance_to_candidates = np.minimum(
            closest_dist_sq, distance_to_candidates
        )
        candidates_pot = sum_along_axis1(
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
    mag_factor : float
        Magnification factor for astigmatism fitting. Required for 3D
        data only. Extracted from the 3D calibration file, see
        ``unpack_calibration``.
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
    spot_size : (2,) np.ndarray, optional
        Spot width and height for astigmatism fitting. Required for 3D
        data only. Extracted from the 3D calibration file, see
        ``unpack_calibration``. Default is None.
    valid_idx : np.ndarray
        Indices of valid components (based on min_locs), applied after
        fitting. Its length gives the number of valid components.
    weights_ : np.ndarray
        Weights of the G5M components, shape (n_components,).
    weights : np.ndarray
        Same as weights_ but only valid components (based on min_locs)
        are indexed. Renormalized to sum to 1.
    z_range : np.ndarray, optional
        Z range for astigmatism fitting. Required for 3D data only.
        Extracted from the 3D calibration file, see
        ``unpack_calibration``.
        Default is None.

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
        means_init: np.ndarray | None = None,
    ) -> None:

        assert sigma_bounds[0] >= 0.0 and sigma_bounds[1] >= 0.0
        assert sigma_bounds[1] >= sigma_bounds[0]

        self.n_components = int(n_components)
        self.min_locs = int(min_locs)
        self.sigma_bounds = sigma_bounds
        self.n_init = max(int(n_components), 3)
        self.random_state = 42
        self.converged = False
        self.means_init = means_init
        self.loc_prec_handle = "local"

        # for 3D compatibility
        self.spot_size = None
        self.z_range = None
        self.mag_factor = None

        # indices for valid components (based on min_locs), applied
        # after fitting
        self.valid_idx = np.arange(n_components).astype(int)
        # number of locs per component (applied after fitting)
        self.n_locs = np.zeros(n_components, dtype=int)

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion (BIC) for the G5M."""
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
    def estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Calculate the log probabilities of the data X under the G5M,
        without weights."""
        pass

    def estimate_weighted_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Calculate the log probabilities of the data X under the G5M,
        with weights."""
        return self.estimate_log_prob(X) + np.log(self.weights)

    def fit(
        self,
        X: np.ndarray,
        lp: np.ndarray,
        loc_prec_handle: Literal["local", "abs"] = "local",
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

        if self.n_dimensions == 2:
            initialize_G5M = initialize_G5M_2D
        elif self.n_dimensions == 3:
            initialize_G5M = initialize_G5M_3D
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

        (w, m, c, pc), converged, valid_idx = fit_G5M(
            X,
            min_locs=self.min_locs,
            init_weights=init_weights,
            init_means=init_means,
            init_precisions_cholesky=init_precisions_cholesky,
            sigma_bounds=self.sigma_bounds,
            lp=lp,
            loc_prec_handle=loc_prec_handle,
            spot_size=self.spot_size,
            z_range=self.z_range,
            mag_factor=self.mag_factor,
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the cluster labels for the data X."""
        return self.estimate_weighted_log_prob(X).argmax(axis=1)

    @abstractmethod
    def sample(self, n_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Sample data points from the G5M."""
        pass

    def set_parameters(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        precisions_cholesky: np.ndarray,
        converged: bool,
        valid_idx: np.ndarray | None = None,
    ) -> None:
        """Set the G5M parameters, used after fitting."""
        self.weights_ = weights / weights.sum()
        self.means_ = means
        self.covariances_ = covs
        self.precisions_cholesky_ = precisions_cholesky
        self.converged = converged
        if self.valid_idx is not None:
            self.valid_idx = valid_idx
        else:
            self.valid_idx = np.arange(len(weights))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of the data X under the G5M."""
        weighted_log_prob = self.estimate_weighted_log_prob(X)
        final_shape = (weighted_log_prob.shape[0],)
        return logsumexp_axis1(weighted_log_prob, final_shape)

    @property
    def weights(self) -> np.ndarray:
        """Valid weights."""
        w = self.weights_[self.valid_idx]
        # return w / w.sum()
        return w


# 2D G5M functions and classes #
@njit
def check_G5M_resolution_2D(
    means: np.ndarray,
    weights: np.ndarray,
    precisions_chol: np.ndarray,
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
            ll = estimate_log_gaussian_prob_2D(X, means_, prec_chol_) + np.log(
                weights_
            )
            pdf = sum_along_axis1(np.exp(ll), ll.shape[0])

            # find if there is at least one local minimum (may be more
            # if components in between align)
            if not len(lib.find_local_minima(pdf)):
                return False

    # if all components are well separated
    return True


@njit
def initialize_G5M_2D(
    X: np.ndarray, n_init: int, n_components: int, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        indices = kmeans_plusplus(X, n_components, random_state)  # kmeans++

        for i in range(n_components):
            resp[indices[i], i] = 1

        # initialize G5M parameters (m-step-like)
        weights, means, covariances = estimate_gaussian_parameters_2D(X, resp)
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
def estimate_gaussian_parameters_2D(
    X: np.ndarray,
    resp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nk, means, covariances = estimate_gaussian_parameters_diag_cov(X, resp)
    covariances = mean_along_axis1(covariances, final_shape=(len(nk),))
    return (
        np.asarray(nk, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(covariances, dtype=np.float64),
    )


@njit
def estimate_log_gaussian_prob_2D(
    X: np.ndarray, means: np.ndarray, precisions_chol: np.ndarray
) -> np.ndarray:
    log_det = 2 * np.log(precisions_chol)
    precisions = square_elements_1d(precisions_chol)
    log_prob = gauss_exponential_term_2D(X, means, precisions)
    return -0.5 * (2 * np.log(2 * np.pi) + log_prob) + log_det


@njit
def e_step_2D(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    precisions_cholesky: np.ndarray,
) -> tuple[float, np.ndarray]:
    weighted_log_prob = estimate_log_gaussian_prob_2D(
        X, means, precisions_cholesky
    ) + np.log(weights)
    log_prob_norm = logsumexp_axis1(weighted_log_prob, (X.shape[0],))
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    return np.mean(log_prob_norm), log_resp.astype(np.float64)


@njit
def m_step_2D(
    X: np.ndarray,
    log_resp: np.ndarray,
    sigma_bounds: tuple[float, float],
    lp: np.ndarray,
    loc_prec_handle: Literal["local", "abs"],
    spot_size: np.ndarray | None = None,  # for 3D consistency
    z_range: np.ndarray | None = None,
    mag_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D m step. spot_size, z_range and mag_factor are not used and are
    here for compatibility with the 3D m step."""
    min_cov = sigma_bounds[0] ** 2
    max_cov = sigma_bounds[1] ** 2
    resp = np.exp(log_resp)
    weights, means, covs = estimate_gaussian_parameters_2D(X, resp)
    # clip covariances (numba does not support np.clip or multidim.
    # indexing)
    if loc_prec_handle == "local":  # local sigma bounds
        # take weighted (based on resp) avg. loc. prec. per component
        lp_ = np.reshape(lp, (-1, 1))
        mean_lp_per_component = sum_along_axis0(
            resp * lp_, (resp.shape[1])
        ) / sum_along_axis0(resp, (resp.shape[1],))
        mean_cov_per_component = square_elements_1d(mean_lp_per_component)
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


def find_optimal_G5M_2D(
    X: np.ndarray,
    min_locs: int,
    sigma_bounds: tuple[float, float],
    *,
    lp: np.ndarray,
    loc_prec_handle: Literal["local", "abs"] = "local",
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
) -> G5M_2D:
    """Find the optimal G5M for the given 2D dataset.

    Parameters
    ----------
    X : np.ndarray
        2D array of localizations, shape (n_samples, 2).
    min_locs : int
        Minimum number of localizations per component.
    sigma_bounds : tuple
        Bounds for the standard deviation (sigma) of the Gaussian
        components. If local loc. prec. is used, the bounds specify the
        margin of error in units of localization precision. Else,
        absolute bounds on sigma.
    lp : np.ndarray
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
        if g5m is None or not check_G5M_resolution_2D(
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


def run_g5m_group_2D(
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
        lp = locs_group[["lpx", "lpy"]].mean(axis=1).values
    else:
        lp = np.ones(len(locs_group))  # dummy
    X = locs_group[["x", "y"]].values.astype(np.float64)

    g5m = find_optimal_G5M_2D(
        X,
        min_locs=min_locs,
        sigma_bounds=sigma_bounds,
        lp=lp,
        loc_prec_handle=loc_prec_handle,
        max_rounds_without_best_bic=max_rounds_without_best_bic,
    )
    if g5m is None or len(g5m.valid_idx) == 0:
        return None, None

    return convert_G5M_results(g5m, locs_group, pixelsize, bootstrap_check)


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
        means_init: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            min_locs=min_locs,
            sigma_bounds=sigma_bounds,
            means_init=means_init,
        )
        self.n_dimensions = 2

    def estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Calculate the log probabilities of the data X under the G5M,
        without weights."""
        return estimate_log_gaussian_prob_2D(
            X,
            self.means,
            self.precisions_cholesky,
        )

    def n_parameters(self) -> int:
        """Find the number of parameters in the G5M."""
        n_valid = len(self.valid_idx)
        cov_params = n_valid
        mean_params = 2 * n_valid
        weight_params = n_valid - 1
        return int(cov_params + mean_params + weight_params)

    def sample(self, n_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Sample data points from the G5M."""
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
def check_G5M_resolution_3D(
    means: np.ndarray,
    weights: np.ndarray,
    precisions_chol: np.ndarray,
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
            ll = estimate_log_gaussian_prob_3D(X, means_, prec_chol_) + np.log(
                weights_
            )
            pdf = sum_along_axis1(np.exp(ll), ll.shape[0])

            # find if there is at least one local minimum (may be more
            # if components in between align)
            if not len(lib.find_local_minima(pdf)):
                return False

    # if all components are well separated
    return True


@njit
def initialize_G5M_3D(
    X: np.ndarray, n_init: int, n_components: int, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        indices = kmeans_plusplus(X, n_components, random_state)  # kmeans++

        for i in range(n_components):
            resp[indices[i], i] = 1

        # initialize G5M parameters (m-step-like)
        weights, means, covariances = estimate_gaussian_parameters_3D(X, resp)
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
def estimate_gaussian_parameters_3D(
    X: np.ndarray,
    resp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return estimate_gaussian_parameters_diag_cov(X, resp)


@njit
def estimate_log_gaussian_prob_3D(
    X: np.ndarray,
    means: np.ndarray,
    precisions_chol: np.ndarray,
) -> np.ndarray:
    log_det = sum_along_axis1(
        np.log(precisions_chol),
        (precisions_chol.shape[0],),
    )
    precisions = square_elements_2d(precisions_chol)
    log_prob = gauss_exponential_term_3D(X, means, precisions)
    return -0.5 * (3 * np.log(2 * np.pi) + log_prob) + log_det


@njit
def e_step_3D(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    precisions_cholesky: np.ndarray,
) -> tuple[float, np.ndarray]:
    weighted_log_prob = estimate_log_gaussian_prob_3D(
        X, means, precisions_cholesky
    ) + np.log(weights)
    log_prob_norm = logsumexp_axis1(weighted_log_prob, (X.shape[0],))
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    return np.mean(log_prob_norm), log_resp.astype(np.float64)


@njit
def m_step_3D(
    X: np.ndarray,
    log_resp: np.ndarray,
    sigma_bounds: tuple[float, float],
    lp: np.ndarray,
    loc_prec_handle: Literal["local", "abs"],
    spot_size: np.ndarray,
    z_range: np.ndarray,
    mag_factor: float = 0.79,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Modified m-step to handle astigmatism in 3D G5M.

    The astigmatism modification handles the astigmatism effect in
    3D DNA-PAINT data. As in sklearn's implementation, the weights,
    means and diagonal covariance matrices are estimated first. In the
    next step, sigma bound are imposed and then the ratio of the spot
    width and height is extracted from calibration, based on the z
    position, for each component and imposed on the covariances' x and
    y values."""
    resp = np.exp(log_resp)
    weights, means, covs = estimate_gaussian_parameters_3D(X, resp)

    # find the min. and max. covariances in each dimension
    if loc_prec_handle == "local":
        # extract loc. precisions and convert to covariances
        lpx = np.ascontiguousarray(lp[:, 0]).reshape(-1, 1)
        lpy = np.ascontiguousarray(lp[:, 1]).reshape(-1, 1)
        lpz = np.ascontiguousarray(lp[:, 2]).reshape(-1, 1)

        mean_lpx_per_component = sum_along_axis0(
            resp * lpx, (resp.shape[1])
        ) / sum_along_axis0(resp, (resp.shape[1]))
        mean_lpy_per_component = sum_along_axis0(
            resp * lpy, (resp.shape[1])
        ) / sum_along_axis0(resp, (resp.shape[1]))
        mean_lpz_per_component = sum_along_axis0(
            resp * lpz, (resp.shape[1])
        ) / sum_along_axis0(resp, (resp.shape[1]))

        mean_covx_per_component = square_elements_1d(mean_lpx_per_component)
        mean_covy_per_component = square_elements_1d(mean_lpy_per_component)
        mean_covz_per_component = square_elements_1d(mean_lpz_per_component)

        min_cov_x = sigma_bounds[0] ** 2 * mean_covx_per_component
        max_cov_x = sigma_bounds[1] ** 2 * mean_covx_per_component
        min_cov_y = sigma_bounds[0] ** 2 * mean_covy_per_component
        max_cov_y = sigma_bounds[1] ** 2 * mean_covy_per_component
        min_cov_z = sigma_bounds[0] ** 2 * mean_covz_per_component
        # max_cov_z = sigma_bounds[1] ** 2 * mean_covz_per_component
        max_cov_z = (
            (sigma_bounds[1] - 1.0) * 0.5 + 1.0
        ) ** 2 * mean_covz_per_component  # decrease max z cov because the lpz is already pretty high
    elif loc_prec_handle == "abs":
        min_cov_x = np.full(covs.shape[0], sigma_bounds[0] ** 2)
        max_cov_x = np.full(covs.shape[0], sigma_bounds[1] ** 2)
        min_cov_y = np.full(covs.shape[0], sigma_bounds[0] ** 2)
        max_cov_y = np.full(covs.shape[0], sigma_bounds[1] ** 2)
        # roughly account for worse z precision
        min_cov_z = np.full(covs.shape[0], sigma_bounds[0] ** 2 * 2.0**2)
        max_cov_z = np.full(covs.shape[0], sigma_bounds[1] ** 2 * 2.5**2)

    # apply the bounds to xy covariances
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

    # impose the ratio of x and y covariances based on the spot width
    # and height ratio

    # find the spot width and height for each component
    z_idx = np.abs(z_range[:, np.newaxis] - (means[:, 2] / mag_factor)).argmin(
        0
    )  # find the closest z values to the current means
    spot_width, spot_height = spot_size
    # impose their ratio of x and y covariances
    ratio = spot_width[z_idx] / spot_height[z_idx]
    covs_xy = np.empty((covs.shape[0], 2))
    covs_xy[:, 0] = covs[:, 0]
    covs_xy[:, 1] = covs[:, 1]
    mean_xy_covs = mean_along_axis1(covs_xy, (covs_xy.shape[0],))
    covs[:, 0] = mean_xy_covs * ratio
    covs[:, 1] = mean_xy_covs / ratio
    weights /= weights.sum()
    precisions_cholesky = 1.0 / np.sqrt(covs)
    return weights, means, covs, precisions_cholesky


def find_optimal_G5M_3D(
    X: np.ndarray,
    min_locs: int,
    sigma_bounds: tuple[float, float],
    spot_size: np.ndarray,
    z_range: np.ndarray,
    *,
    lp: np.ndarray,
    loc_prec_handle: Literal["local", "abs"] = "local",
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
    mag_factor: float = 0.79,
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
    spot_size : (2,) np.ndarray
        Spot width and height from the 3D calibration for each z
        position (...)
    z_range : np.ndarray
        (...) and the corresponding z values (in camera pixels).
    lp : np.ndarray
        Localization precision for each localization in x, y and z. Only
        used if loc_prec_handle is "local". Shape (n_samples, 3).
    loc_prec_handle : {"local", "abs"}, optional
        How to handle sigma bounds. If "local", localization precisions
        of points around each component are used to bound sigmas. Else,
        sigma_bounds specifies the absolute bounds on sigmas. Default
        is "local".
    max_rounds_without_best_bic : int, optional
        Maximum number of rounds without BIC improvement to terminate
        the search for optimal G5M n_components. Default is
        `MAX_ROUNDS_WITHOUT_BEST_BIC`.
    mag_factor : float, optional
        Magnification factor used for correcting the refractive index
        mismatch for 3D imaging. Default is 0.79.

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
            spot_size=spot_size,
            z_range=z_range,
            mag_factor=mag_factor,
        ).fit(X, lp=lp, loc_prec_handle=loc_prec_handle)
        if g5m is None or not check_G5M_resolution_3D(
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


def run_g5m_group_3D(
    locs_group: pd.DataFrame,
    calibration: dict,
    *,
    min_locs: int = MIN_LOCS,
    loc_prec_handle: Literal["local", "abs"] = "local",
    sigma_bounds: tuple[float, float] = (MIN_SIGMA_FACTOR, MAX_SIGMA_FACTOR),
    pixelsize: float = 130.0,
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
    calibration : dict
        Calibration dictionary with the following keys:
        "X Coefficients", "Y Coefficients", "Step size in nm",
        "Number of frames" and "Magnification factor", see
        ``unpack_calibration`` for more details.
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
    # make sure lpz is available (assume gauss least-squares used for localization)
    if not hasattr(locs_group, "lpz"):
        locs_group = locs_group.copy()
        locs_group["lpz"] = zfit.axial_localization_precision(
            locs_group, [{"Pixelsize": pixelsize}], calibration, "gausslq"
        )
    # check that the number of localizations is within the limits
    n_locs = len(locs_group)
    if n_locs < min_locs or n_locs > max_locs_per_cluster:
        return None, None

    spot_size, z_range, mag_factor = lib.unpack_calibration(
        calibration, pixelsize
    )

    if loc_prec_handle == "local":
        lp = locs_group[["lpx", "lpy", "lpz"]].values
    else:
        lp = np.ones((len(locs_group), 3))  # dummy
    X = locs_group[["x", "y", "z"]].values
    X[:, 2] /= pixelsize  # convert z to camera pixels
    lp[:, 2] /= pixelsize  # convert lpz to camera pixels

    g5m = find_optimal_G5M_3D(
        X,
        min_locs=min_locs,
        sigma_bounds=sigma_bounds,
        spot_size=spot_size,
        z_range=z_range,
        lp=lp,
        loc_prec_handle=loc_prec_handle,
        max_rounds_without_best_bic=max_rounds_without_best_bic,
        mag_factor=mag_factor,
    )
    if g5m is None or len(g5m.valid_idx) == 0:
        return None, None

    return convert_G5M_results(g5m, locs_group, pixelsize, bootstrap_check)


class G5M_3D(G5M):
    """G5M for 3D data (astigmatism). See ``G5M`` for more details.

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
    spot_size : np.ndarray
        Spot width and height from the 3D calibration for each z
        position.
    z_range : np.ndarray
        Corresponding z values (in camera pixels) for the spot size.
    mag_factor : float, optional
        Magnification factor used for correcting the refractive index
        mismatch for 3D imaging. Default is 0.79.
    means_init : np.ndarray or None, optional
        Initial means (mu) of the Gaussian components. If None, the
        means are initialized using kmeans++. Default is None.
    """

    def __init__(
        self,
        n_components: int,
        min_locs: int,
        sigma_bounds: tuple[float, float],
        spot_size: np.ndarray,
        z_range: np.ndarray,
        *,
        mag_factor: float = 0.79,
        means_init: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            min_locs=min_locs,
            sigma_bounds=sigma_bounds,
            means_init=means_init,
        )
        self.spot_size = spot_size
        self.z_range = z_range
        self.mag_factor = mag_factor
        self.n_dimensions = 3

    def estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Calculate the log probabilities of the data X under the G5M,
        without weights."""
        return estimate_log_gaussian_prob_3D(
            X,
            self.means,
            self.precisions_cholesky,
        )

    def n_parameters(self) -> int:
        """Return the number of free parameters in the model. Note that
        the astigmatism-modification reduces the number of free
        parameters for each component by one."""
        n_valid = len(self.valid_idx)
        cov_params = n_valid * 2  # cov. in y depends on cov. in x
        mean_params = 3 * n_valid
        weight_params = n_valid - 1
        return int(cov_params + mean_params + weight_params)

    def sample(self, n_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Sample data points from the G5M."""
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights)

        X = np.vstack(
            [
                mean
                + rng.standard_normal(size=(sample, 3)) * np.sqrt(covariance)
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
def estimate_gaussian_parameters_diag_cov(
    X: np.ndarray,
    resp: np.ndarray,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        sum_along_axis0(resp, (resp.shape[1],))
        + 10.0 * np.finfo(resp.dtype).eps
    )
    means = matmul(resp.T, X) / nk[:, np.newaxis]
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


def approximate_sem(g5m: G5M, locs: pd.DataFrame) -> np.ndarray:
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

    if not hasattr(locs, "z"):
        covariances = np.repeat(covariances, 2).reshape(-1, 2)
    N = len(locs) * weights.reshape(len(weights), -1)
    sem = np.sqrt(covariances / N)
    return sem


def bootstrap_sem(
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
        if hasattr(locs, "z"):
            g5m_boot = G5M_3D(
                n_components=len(g5m.valid_idx),
                min_locs=g5m.min_locs,
                sigma_bounds=g5m.sigma_bounds,
                spot_size=g5m.spot_size,
                z_range=g5m.z_range,
                means_init=g5m.means,
                mag_factor=g5m.mag_factor,
            )
            lp = locs[["lpx", "lpy"]].values
        else:
            g5m_boot = G5M_2D(
                n_components=len(g5m.valid_idx),
                min_locs=g5m.min_locs,
                sigma_bounds=g5m.sigma_bounds,
                means_init=g5m.means,
            )
            lp = locs[["lpx", "lpy"]].mean(axis=1).values
        g5m_boot.fit(X_boot, lp=lp, loc_prec_handle=g5m.loc_prec_handle)
        if hasattr(g5m_boot, "means_"):  # converged
            boot_means.append(g5m_boot.means_)
    sem = np.std(boot_means, axis=0)
    g5m.random_state = old_random_state
    return sem


def convert_G5M_results(
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
    locs_group = locs_group.copy()
    means = g5m.means
    covariances = g5m.covariances
    weights = g5m.weights
    # find responsibilites which are used for weighted averaging of
    # properties per component
    if hasattr(locs_group, "z"):
        X = locs_group[["x", "y", "z"]].values
        X[:, 2] /= pixelsize  # convert z to camera pixels
        e_step = e_step_3D
    else:
        X = locs_group[["x", "y"]].values
        e_step = e_step_2D
    log_prob = g5m.estimate_weighted_log_prob(X)
    sample_scores = logsumexp_axis1(log_prob, (X.shape[0],))
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
    if X.shape[1] == 2:  # (2D)
        expected = np.log(weights / (2 * np.pi * covariances)) - 1
    else:  # 3D
        expected = (
            np.log(
                weights / ((2 * np.pi) ** 1.5 * np.sqrt(covariances).prod(1))
            )
            - 1.5
        )
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
        sem = bootstrap_sem(g5m, locs_group)
    else:
        sem = approximate_sem(g5m, locs_group)
    lpx = sem[:, 0]
    lpy = sem[:, 1]

    if hasattr(locs_group, "z"):
        z = means[:, 2] * pixelsize
        sigma_x = np.sqrt(covariances[:, 0]) * pixelsize
        sigma_y = np.sqrt(covariances[:, 1]) * pixelsize
        sigma_z = np.sqrt(covariances[:, 2]) * pixelsize
        lpz = sem[:, 2]
        weighted_lpx = (
            (resp * locs_group["lpx"].values.reshape(-1, 1)).sum(0) / rsum
        ).reshape(-1)
        weighted_lpy = (
            (resp * locs_group["lpy"].values.reshape(-1, 1)).sum(0) / rsum
        ).reshape(-1)
        weighted_lpz = (
            (resp * locs_group["lpz"].values.reshape(-1, 1)).sum(0) / rsum
        ).reshape(-1)
        rel_sigma_x = sigma_x / weighted_lpx / pixelsize
        rel_sigma_y = sigma_y / weighted_lpy / pixelsize
        rel_sigma_z = sigma_z / weighted_lpz / pixelsize
    else:
        sigma = np.sqrt(covariances) * pixelsize
        # relative sigma
        lp = locs_group[["lpx", "lpy"]].mean(axis=1).values
        weighted_lp = ((resp * lp.reshape(-1, 1)).sum(0) / rsum).reshape(-1)
        rel_sigma = sigma / weighted_lp / pixelsize

    # extract frame info and group_input
    frames_locs = np.reshape(locs_group["frame"].values, (-1, 1))
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

    # photons, PSF size and background (weighted average)
    photons = (
        (resp * locs_group["photons"].values.reshape(-1, 1)).sum(0) / rsum
    ).reshape(-1)
    sx = (
        (resp * locs_group["sx"].values.reshape(-1, 1)).sum(0) / rsum
    ).reshape(-1)
    sy = (
        (resp * locs_group["sy"].values.reshape(-1, 1)).sum(0) / rsum
    ).reshape(-1)
    bg = (
        (resp * locs_group["bg"].values.reshape(-1, 1)).sum(0) / rsum
    ).reshape(-1)

    # extract the number of binding events, i.e., link localizations
    # and assign them to molecules - sticky events will likely have only
    # one or two such events associated

    # idx to split localizations into binding events, where up to 3
    # frames of no signal are allowed
    split_idx = np.where(np.diff(locs_group["frame"].values) > 3)[0] + 1
    # link localizations into binding events, we only need the center
    # of mass
    x_events = np.split(locs_group["x"].values, split_idx)
    x_events = [np.mean(_) for _ in x_events]
    y_events = np.split(locs_group["y"].values, split_idx)
    y_events = [np.mean(_) for _ in y_events]
    if hasattr(locs_group, "z"):
        z_events = np.split(locs_group["z"].values, split_idx)
        z_events = [np.mean(_) / pixelsize for _ in z_events]
        X_events = np.stack((x_events, y_events, z_events)).T
    else:
        X_events = np.stack((x_events, y_events)).T
    # find the closest G5M component to each binding event and assign
    # the binding event to the component but account for the case when
    # no binding event is assigned to a component
    labels = g5m.predict(X_events)
    expected_labels = np.arange(len(g5m.valid_idx))
    found_labels, counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(found_labels, counts))
    n_events = np.array([count_dict.get(_, 0) for _ in expected_labels])

    # convert to DataFrame
    if hasattr(locs_group, "z"):
        centers = pd.DataFrame(
            {
                "frame": frame.astype(np.float32),
                "std_frame": std_frame.astype(np.float32),
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "z": z.astype(np.float32),
                "photons": photons.astype(np.float32),
                "sx": sx.astype(np.float32),
                "sy": sy.astype(np.float32),
                "bg": bg.astype(np.float32),
                "lpx": lpx.astype(np.float32),
                "lpy": lpy.astype(np.float32),
                "lpz": lpz.astype(np.float32),
                "fitted_sigma_x": sigma_x.astype(np.float32),
                "fitted_sigma_y": sigma_y.astype(np.float32),
                "fitted_sigma_z": sigma_z.astype(np.float32),
                "rel_sigma_x": rel_sigma_x.astype(np.float32),
                "rel_sigma_y": rel_sigma_y.astype(np.float32),
                "rel_sigma_z": rel_sigma_z.astype(np.float32),
                "p_val": p_val.astype(np.float32),
                "mol_log_likelihood": mol_ll.astype(np.float32),
                "group_log_likelihood": group_ll.astype(np.float32),
                "n_locs": g5m.n_locs.astype(np.int32),
                "n_events": n_events.astype(np.int32),
                "group_input": group_input.astype(np.int32),
            }
        )
    else:
        centers = pd.DataFrame(
            {
                "frame": frame.astype(np.float32),
                "std_frame": std_frame.astype(np.float32),
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "photons": photons.astype(np.float32),
                "sx": sx.astype(np.float32),
                "sy": sy.astype(np.float32),
                "bg": bg.astype(np.float32),
                "lpx": lpx.astype(np.float32),
                "lpy": lpy.astype(np.float32),
                "fitted_sigma": sigma.astype(np.float32),
                "rel_sigma": rel_sigma.astype(np.float32),
                "p_val": p_val.astype(np.float32),
                "mol_log_likelihood": mol_ll.astype(np.float32),
                "group_log_likelihood": group_ll.astype(np.float32),
                "n_locs": g5m.n_locs.astype(np.int32),
                "n_events": n_events.astype(np.int32),
                "group_input": group_input.astype(np.int32),
            }
        )
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
    # get means
    means = np.vstack([_.means for _ in g5ms])
    # get covariances, note that the shape of the covs array depends
    # on the dimensionality: 2D -> shape: (n_components, ), 3D -> shape:
    # (n_components, 3)
    if g5ms[0].__class__ == G5M_2D:
        covs = np.hstack([_.covariances for _ in g5ms])
    elif g5ms[0].__class__ == G5M_3D:
        covs = np.stack([_.covariances for _ in g5ms]).reshape(len(weights), 3)
    pc = 1 / np.sqrt(covs)

    sum_g5m = g5ms[0].__class__(
        n_components=len(weights),
        min_locs=g5ms[0].min_locs,
        sigma_bounds=g5ms[0].sigma_bounds,
        spot_size=g5ms[0].spot_size,
        z_range=g5ms[0].z_range,
        mag_factor=g5ms[0].mag_factor,
    )

    # set parameters (just like after fitting)
    valid_idx = np.arange(len(weights))
    sum_g5m.set_parameters(weights, means, covs, pc, True, valid_idx)
    sum_g5m.n_locs = n_locs
    return sum_g5m


@njit
def fit_G5M(
    X: np.ndarray,
    min_locs: int,
    init_weights: np.ndarray,
    init_means: np.ndarray,
    init_precisions_cholesky: np.ndarray,
    sigma_bounds: tuple[float, float],
    *,
    lp: np.ndarray,
    loc_prec_handle: Literal["local", "abs"] = "local",
    spot_size: np.ndarray | None = None,
    z_range: np.ndarray | None = None,
    mag_factor: float | None = None,
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
    spot_size : (2,) np.ndarray, optional
        Spot width and height for astigmatism fitting. Required for 3D
        data only. Extracted from the 3D calibration file, see
        ``unpack_calibration``. Default is None.
    z_range : np.ndarray, optional
        Z range for astigmatism fitting. Required for 3D data only.
        Extracted from the 3D calibration file, see
        ``unpack_calibration``.
        Default is None.
    mag_factor : float, optional
        Magnification factor for astigmatism fitting. Required for 3D
        data only. Extracted from the 3D calibration file, see
        ``unpack_calibration``. Default is None.

    Returns
    -------
    weights, means, covariances, precisions_cholesky : tuple
        Fitted G5M parameters.
    converged : bool
        True if the G5M converged, False otherwise.
    valid_idx : np.ndarray
        Indices of the valid components (min_locs).
    """
    if init_precisions_cholesky.ndim == 2:  # 2D data
        e_step = e_step_2D
        m_step = m_step_2D
        check_resolution = check_G5M_resolution_2D
    elif init_precisions_cholesky.ndim == 3:
        e_step = e_step_3D
        m_step = m_step_3D
        check_resolution = check_G5M_resolution_3D
        if spot_size is None or z_range is None or mag_factor is None:
            raise ValueError(
                "spot_size, z_range and mag_factor are required for "
                "3D data."
            )
    else:
        raise ValueError(
            "Only 2D and 3D data are supported. Data points suggest "
            f"{X.shape[1]} dimensions. The initial precisions suggest "
            f"{init_precisions_cholesky.ndim} dimensions. 3D data "
            "requires spot_size, z_range and mag_factor."
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
                spot_size=spot_size,
                z_range=z_range,
                mag_factor=mag_factor,
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


def run_g5m_in_clusters(
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run G5M for a given group of localizations clusters. See ``g5m``
    for parameters explanation.

    Parameters
    ----------
    i : int
        Index of the first group to analyze.
    n_groups_task : int
        Number of groups to analyze.

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
        if hasattr(locs, "z"):
            centers_, clustered_locs_ = run_g5m_group_3D(
                locs_group=locs[locs["group"] == group],
                calibration=calibration,
                min_locs=min_locs,
                loc_prec_handle=loc_prec_handle,
                sigma_bounds=sigma_bounds,
                pixelsize=pixelsize,
                max_rounds_without_best_bic=max_rounds_without_best_bic,
                bootstrap_check=bootstrap_check,
                max_locs_per_cluster=max_locs_per_cluster,
            )
        else:
            centers_, clustered_locs_ = run_g5m_group_2D(
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


def run_g5m_parallel(
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
) -> list:
    """Run G5M in parallel using multiprocessing. See ``g5m`` for
    parameters explanation.

    Returns
    -------
    fs : list
        List of futures.
    """
    n_groups = len(np.unique(locs["group"]))
    n_workers = min(
        60, max(1, int(0.35 * os.cpu_count()))
    )  # Python crashes when using >64 cores
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
                run_g5m_in_clusters,
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
            )
        )
    return fs


def g5m(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    min_locs: int = MIN_LOCS,
    loc_prec_handle: Literal["local", "abs"] = "local",
    sigma_bounds: tuple[float, float] = (MIN_SIGMA_FACTOR, MAX_SIGMA_FACTOR),
    pixelsize: float = 130.0,
    max_rounds_without_best_bic: int = MAX_ROUNDS_WITHOUT_BEST_BIC,
    bootstrap_check: bool = False,
    calibration: dict | None = None,
    postprocess: bool = True,
    max_locs_per_cluster: int = np.inf,
    asynch: bool = True,
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
    pixelsize : float, optional
        Camera pixel size in nm. Default is 130.0.
    bootstrap_check : bool, optional
        If True, the standard error of the means (SEM) is calculated
        using bootstrapping. If False, the standard, single Gaussian SEM
        is used as approximation. Default is False.
    calibration : dict, optional
        Calibration dictionary with x and y coefficients, z step size
        and the number of frames, see run_g5m_group_3D for more details.
        Only required for 3D data. Default is None.
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
    callback_parent : {QtWidgets.QMainWindow, "console", None}, optional
        Callback function's parent object for displaying progress bar.
        If "console" tqdm is used to display the progress bar in the
        console. If None, no progress is displayed. Default is
        "console".

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
    assert hasattr(
        locs, "group"
    ), "Localizations must be grouped. Use DBSCAN or similar."

    # check that calibration is provided for 3D data
    if hasattr(locs, "z") and calibration is None:
        raise ValueError(
            "Calibration dictionary must be provided for 3D data."
        )

    # determine how many steps are displayed in the progress bar
    n_steps = N_TASKS if asynch else len(np.unique(locs["group"]))

    # initialize the progress bar
    if callback_parent == "console":
        progress = tqdm(total=n_steps, desc="Running G5M...")
    elif callback_parent is None:
        progress = lib.MockProgress()
    else:
        progress = lib.ProgressDialog(
            "Running G5M...", 0, n_steps, callback_parent
        )
        progress.set_value(0)

    if asynch:  # run G5M using multiprocessing
        fs = run_g5m_parallel(
            locs,
            min_locs=min_locs,
            loc_prec_handle=loc_prec_handle,
            sigma_bounds=sigma_bounds,
            pixelsize=pixelsize,
            max_rounds_without_best_bic=max_rounds_without_best_bic,
            bootstrap_check=bootstrap_check,
            calibration=calibration,
            max_locs_per_cluster=max_locs_per_cluster,
        )

        # display progress
        while lib.n_futures_done(fs) < n_steps:
            n_done = lib.n_futures_done(fs)
            if callback_parent != "console":
                progress.set_value(n_done)
            else:
                progress.update(n_done - progress.n)
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
            if hasattr(locs, "z"):
                centers_, clustered_locs_ = run_g5m_group_3D(
                    locs[locs["group"] == group],
                    calibration=calibration,
                    min_locs=min_locs,
                    loc_prec_handle=loc_prec_handle,
                    sigma_bounds=sigma_bounds,
                    pixelsize=pixelsize,
                    max_rounds_without_best_bic=max_rounds_without_best_bic,
                    bootstrap_check=bootstrap_check,
                    max_locs_per_cluster=max_locs_per_cluster,
                )
            else:
                centers_, clustered_locs_ = run_g5m_group_2D(
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

            if callback_parent == "console":
                progress.update(1)
            else:
                progress.set_value(i)

    # close progress widget if present
    if callback_parent != "console":
        progress.close()
    else:
        progress.update(1)

    # stack centers to form a pd.DataFrame in the format of localizations
    centers = pd.concat(centers, ignore_index=True)
    # assing group ids to the clustered localizations
    max_label = 0
    for i, clustered_locs_ in enumerate(clustered_locs):
        clustered_locs_["group"] += max_label
        max_label = clustered_locs_["group"].max() + 1
        clustered_locs[i] = clustered_locs_
    clustered_locs = pd.concat(clustered_locs, ignore_index=True)

    # update info
    new_info = {
        "Generated by": f"Picasso v{__version__} G5M",
        "Model determination": "BIC",
        "Number of molecules": len(centers),
        "Min. no. locs per molecule": min_locs,
        "Max. rounds w/o BIC improvement": max_rounds_without_best_bic,
        "Bootstrap SEM": bootstrap_check,
        "Initialization method": "KMeans++",
        "Filtered": False,
    }
    if loc_prec_handle == "local":
        new_info["Sigma bounds (factors)"] = list(sigma_bounds)
        new_info["Sigma bounds method"] = "Local"
    else:
        new_info["Sigma bounds (nm)"] = [
            sigma_bounds[0] * pixelsize,
            sigma_bounds[1] * pixelsize,
        ]
        new_info["Sigma bounds method"] = "Abs"
    if hasattr(locs, "z"):
        new_info["X Coefficients"] = calibration["X Coefficients"]
        new_info["Y Coefficients"] = calibration["Y Coefficients"]
        new_info["Calibration z Step size in nm"] = calibration[
            "Step size in nm"
        ]
        new_info["Calibration number of frames"] = calibration[
            "Number of frames"
        ]
        new_info["Magnification factor"] = calibration["Magnification factor"]
    info = info + [new_info]
    if postprocess:
        # filter out by mean frame, std frame, p_val and n_events
        n_frames = info[0]["Frames"]
        min_frame = 0.1 * n_frames
        max_frame = 0.9 * n_frames
        min_std_frame = 0.1 * n_frames
        min_pval = 0.015
        min_n_events = 3

        idx = (
            (centers["frame"] > min_frame)
            & (centers["frame"] < max_frame)
            & (centers["std_frame"] > min_std_frame)
            & (centers["p_val"] > min_pval)
            & (centers["n_events"] > min_n_events)
        )
        centers = centers[idx]
        clustered_locs = clustered_locs[
            np.isin(clustered_locs["group"], np.arange(len(idx))[idx])
        ]
        info[-1]["Filtered"] = True
        info[-1]["Filter; min. mean frame"] = min_frame
        info[-1]["Filter; max. mean frame"] = max_frame
        info[-1]["Filter; min. std frame"] = min_std_frame
        info[-1]["Filter; min. p value"] = min_pval
        info[-1]["Filter; min. n_events"] = min_n_events
    return centers, clustered_locs, info
