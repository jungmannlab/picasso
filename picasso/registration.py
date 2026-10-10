"""
picasso.registration
~~~~~~~~~~~~~~~~~~~~

Registering the channels of a multichannel acquisition onto one reference
channel, and the standalone calibration file that stores the result.

A multichannel fit needs to know where a molecule seen at ``x`` in the
reference channel lands in every other channel. That mapping, the *channel
registration*, is one coordinate transform per channel (reference ->
channel), independent of the PSF model fitted afterwards: :mod:`picasso.spline`
bakes one into its multichannel PSF calibration, and the 2D Gaussian
multichannel fit reads one from the standalone calibration built here.

Two ways to measure it, both producing the same file:

``calibrate_channel_registration_from_beads``
    From images of fiducial beads, which appear in every channel at once.

``calibrate_channel_registration_from_signal``
    From the experimental blinking data itself. The channels are
    frame-synchronized, so the same emitter fluoresces in every channel in the
    *same* frame; pairing those detections frame by frame registers the
    channels with no separate bead acquisition.

The matching machinery underneath is shared with :mod:`picasso.spline`, which
registers the channels of its own multichannel PSF calibration the same way.
That shared part is public API rather than module-private:

================================  =============================================
:func:`match_points`              nearest-neighbor pairing of two point clouds
:func:`ransac_match`              robust pairing with no prior estimate
:func:`fit_registration`          one ICP iteration's transform
:func:`register_from_point_sets`  the whole bootstrap + ICP + trim loop
:func:`resolve_model`             the transform model a calibration implies
:func:`frames_in_bounds`          the frame indices a bound allows
================================  =============================================

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import datetime
import warnings
from itertools import combinations
from typing import Callable

import numpy as np
from scipy.spatial import KDTree

from . import io, localize, __version__

# aliased: `wavelet` is the keyword that passes the wavelet identification
# settings through this module
from . import wavelet as wavelets

# aliased: `transforms` is used as a local name for lists of channel
# transforms throughout this module
from . import transforms as tform

#: ``model`` of the standalone channel-registration calibration, so it can be
#: told apart from a spline PSF calibration that also carries channel
#: transforms.
REGISTRATION_MODEL = "channel-registration"

#: How far apart matched beads may start out when registering separate bead
#: movies of the same field, which overlap to begin with. Mirrors the radius
#: ``picasso.localize.calibrate_lateral_transform`` has always paired at.
_BEAD_MATCH_RADIUS_PX = 40.0


def resolve_model(calibration: dict, model: str | None) -> str:
    """The transform model to register with.

    Parameters
    ----------
    calibration : dict
        A calibration carrying ``channel_transforms``; the model its stored
        transforms were fitted with is used when ``model`` is None. Read off
        the transforms themselves rather than a separate key, so it cannot
        disagree with them.
    model : str or None
        The model asked for, as in :mod:`picasso.transforms`. None falls back
        to the calibration's own.

    Returns
    -------
    model : str
        The model asked for, else the calibration's own, else ``"affine"``.
    """
    if model is not None:
        return model
    stored = calibration.get("channel_transforms")
    for entry in (stored or [])[1:] or (stored or [])[:1]:
        return tform.from_dict(entry).model
    return "affine"


def _icp_model(model: str) -> str:
    """The model the ICP machinery works in for a registration finally fitted
    with ``model``.

    An affine, except for a translation: that is the one model *below* an
    affine, so falling back to an affine would loosen the fit rather than
    steady it, and the affine's 3-point minimum would reject pairings a
    translation handles.
    """
    return "translation" if model == "translation" else "affine"


def _icp_min_pairs(model: str) -> int:
    """Fewest correspondences an ICP pass will fit from.

    The ICP model's own minimum, but never a single pair: one pair fixes a
    translation exactly, so a lone coincidental match would converge to a
    zero residual and look like a perfect registration.
    """
    return max(tform.min_points(_icp_model(model)), 2)


def fit_registration(
    src: np.ndarray,
    dst: np.ndarray,
    model: str,
    final: bool = True,
) -> tuple[tform.Transform, str]:
    """Fit one ICP iteration's transform.

    Intermediate iterations (``final=False``) always fit an affine - a
    translation, when that is what was asked for - however flexible ``model``
    is: the early pairing is deliberately loose and contains cross-molecule
    mismatches, and a flexible model bends to accommodate them,
    locking in the wrong correspondence field on the next pass - a failure an
    affine cannot have. Only the final iteration, which pairs at the tightest
    radius, fits the model the user asked for.

    Parameters
    ----------
    src, dst : np.ndarray
        ``(n, 2)`` matched correspondences, in ``[x, y]``. The fitted transform
        maps ``src`` onto ``dst``.
    model : str
        Transform model to fit, as in :mod:`picasso.transforms`.
    final : bool, optional
        Whether this is the last ICP iteration, and therefore the one that
        fits ``model`` rather than an affine. Default True.

    Returns
    -------
    transform : picasso.transforms.Transform
        The fitted transform.
    model : str
        The model that was *actually* fitted. If too few correspondences
        survived for the one asked for, the fallback model is fitted instead
        and named here, so the caller can report it rather than hide it.
    """
    if final and len(src) >= tform.min_points(model):
        return tform.estimate(src, dst, model), model
    fallback = _icp_model(model)
    with warnings.catch_warnings():
        # An intermediate ICP pass is an internal step towards the pairing,
        # not a registration anyone keeps, so its "thin data" warning would
        # only be noise; the final fit above still warns.
        if not final:
            warnings.simplefilter("ignore")
        return tform.estimate(src, dst, fallback), fallback


def _similarity_from_two(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> list[tform.AffineTransform]:
    """Candidate similarity transforms mapping ``a -> b`` from two point pairs.

    A similarity (translation + rotation + isotropic scale, optionally a
    reflection) is fixed by two correspondences up to the reflection
    ambiguity, so both the proper-rotation and the reflected solution are
    returned. Using a *similarity* (4 DOF) as the RANSAC minimal model -
    rather than a full 6-DOF affine, which three points always fit exactly -
    keeps a spare bead to validate the sample, so correct correspondences can
    be told from wrong ones even with only three beads. Empty if the two
    reference points coincide.

    This stays a similarity whatever model the registration is finally fitted
    with: matching only needs a hypothesis good enough to rank correspondences,
    and a higher-DOF minimal model would defeat the consensus vote - a degree-3
    polynomial fits *any* 10 points exactly, so every sample would score the
    maximum."""
    va, vb = a1 - a0, b1 - b0
    na = float(np.hypot(va[0], va[1]))
    if na < 1e-9:
        return []
    s = float(np.hypot(vb[0], vb[1])) / na
    ang_a = np.arctan2(va[1], va[0])
    ang_b = np.arctan2(vb[1], vb[0])
    out = []
    # proper rotation (angle b - angle a) and reflection (across the a/b
    # bisector)
    th = ang_b - ang_a
    r_rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    two_alpha = ang_a + ang_b
    r_ref = np.array(
        [
            [np.cos(two_alpha), np.sin(two_alpha)],
            [np.sin(two_alpha), -np.cos(two_alpha)],
        ]
    )
    for r in (r_rot, r_ref):
        A = s * r
        t = b0 - A @ a0
        matrix = np.eye(3)
        matrix[:2, :2] = A
        matrix[:2, 2] = t
        out.append(tform.AffineTransform(matrix=matrix))
    return out


def _fov_groups(
    ref_fov: np.ndarray | None,
    c_fov: np.ndarray | None,
    n_ref: int,
    n_c: int,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Row blocks of the reference / channel bead clouds that share a field of
    view, as ``[(ref_idx, c_idx), ...]``, or None to treat them as one pooled
    cloud.

    Fields present in only one of the two clouds are dropped: a bead with no
    counterpart to pair against cannot contribute a correspondence, and letting
    it search the other fields is exactly the mis-pairing this prevents. None
    is returned when either label array is missing or does not describe its
    cloud, so callers without FOV information keep the pooled behavior.
    """
    if ref_fov is None or c_fov is None:
        return None
    ref_fov = np.asarray(ref_fov)
    c_fov = np.asarray(c_fov)
    if len(ref_fov) != n_ref or len(c_fov) != n_c:
        return None
    groups = []
    for k in np.unique(ref_fov):
        ri = np.flatnonzero(ref_fov == k)
        ci = np.flatnonzero(c_fov == k)
        if len(ri) and len(ci):
            groups.append((ri, ci))
    return groups or None


def _ransac_candidate_pairs(
    ref_xy: np.ndarray,
    aligned_c: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]] | None,
    radius: float,
) -> list[tuple[int, int]]:
    """Candidate ``(ref_i, c_j)`` pairs: ``c`` beads near ``ref_i`` in the
    coarse overlay, within a field of view when ``groups`` is given."""
    if groups is None:
        overlay_tree = KDTree(aligned_c)
        return [
            (i, j)
            for i in range(len(ref_xy))
            for j in overlay_tree.query_ball_point(ref_xy[i], radius)
        ]
    pairs = []
    for ri, ci in groups:
        overlay_tree = KDTree(aligned_c[ci])
        for i in ri:
            pairs.extend(
                (int(i), int(ci[j]))
                for j in overlay_tree.query_ball_point(ref_xy[i], radius)
            )
    return pairs


def _ransac_msac_cost(
    c_xy: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]] | None,
    tol_sq: float,
) -> Callable[[np.ndarray], float]:
    """The MSAC cost function scoring a mapped reference cloud against
    ``c_xy``: every point's squared distance to its nearest partner, capped
    at ``tol_sq``, within a field of view when ``groups`` is given."""
    if groups is None:
        c_tree = KDTree(c_xy)

        def msac_cost(pred: np.ndarray) -> float:
            dist, _ = c_tree.query(pred, k=1)
            return float(np.sum(np.minimum(dist**2, tol_sq)))

        return msac_cost

    # one tree per field, so a bead can only find partners in its own
    group_trees = [(ri, KDTree(c_xy[ci])) for ri, ci in groups]

    def msac_cost(pred: np.ndarray) -> float:
        total = 0.0
        for ri, tree in group_trees:
            dist, _ = tree.query(pred[ri], k=1)
            total += float(np.sum(np.minimum(dist**2, tol_sq)))
        return total

    return msac_cost


def _ransac_samples(n_pairs: int, max_iter: int) -> np.ndarray:
    """Index-pair samples to try: every combination, or - above ``max_iter``
    - a deterministic random draw, so a calibration stays reproducible."""
    n_samples = n_pairs * (n_pairs - 1) // 2
    if n_samples > max_iter:
        rs = np.random.RandomState(0)  # deterministic for reproducible calib
        return rs.randint(0, n_pairs, size=(max_iter, 2))
    return np.asarray(list(combinations(range(n_pairs), 2)), dtype=int)


def _ransac_best_transform(
    ref_xy: np.ndarray,
    c_xy: np.ndarray,
    pairs: np.ndarray,
    samples: np.ndarray,
    msac_cost: Callable[[np.ndarray], float],
) -> tform.AffineTransform | None:
    """The lowest-MSAC-cost similarity transform among the sampled candidate
    pairs, or None if none was scorable."""
    best_M, best_cost = None, np.inf
    for a, b in samples:
        (i0, j0), (i1, j1) = pairs[a], pairs[b]
        if i0 == i1 or j0 == j1:  # need two distinct ref and channel beads
            continue
        # the two sampled pairs may come from different fields - that is
        # welcome, the transform is global and a longer baseline pins it down
        # better; only the correspondences themselves stay within a field
        for M in _similarity_from_two(
            ref_xy[i0], ref_xy[i1], c_xy[j0], c_xy[j1]
        ):
            cost = msac_cost(M.apply(ref_xy))
            if cost < best_cost:
                best_cost, best_M = cost, M
    return best_M


def _ransac_final_inliers(
    best_M: tform.AffineTransform,
    ref_xy: np.ndarray,
    c_xy: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray]] | None,
    inlier_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The winning transform's inliers (unique nearest-neighbor assignment),
    within a field of view when ``groups`` is given."""
    pred = best_M.apply(ref_xy)
    if groups is None:
        return match_points(pred, c_xy, inlier_tol)
    acc_ref, acc_c = [], []
    for ri, ci in groups:
        a, b = match_points(pred[ri], c_xy[ci], inlier_tol)
        if len(a):
            acc_ref.append(ri[a])
            acc_c.append(ci[b])
    if not acc_ref:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(acc_ref), np.concatenate(acc_c)


def ransac_match(
    ref_xy: np.ndarray,
    c_xy: np.ndarray,
    aligned_c: np.ndarray,
    inlier_tol: float,
    radius: float,
    max_iter: int = 20000,
    ref_fov: np.ndarray | None = None,
    c_fov: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly match beads across channels via RANSAC on a similarity
    transform.

    Correspondence candidates are proposed from ``aligned_c`` (``c_xy``
    coarsely overlaid onto the reference frame - a flip orientation plus an
    approximate shift), but the transform is fit on the original
    **absolute** ``ref_xy`` / ``c_xy``. Two candidate pairs are sampled, the
    similarity transforms they imply (see :func:`_similarity_from_two`) are
    formed, and each is scored by how well it maps the reference cloud onto
    the other one - every bead contributing its squared distance to the
    nearest partner, capped at ``inlier_tol`` for the ones with no partner at
    all (the MSAC cost). The cheapest wins and its inliers (unique
    nearest-neighbor assignment) are returned. Capped-square rather than
    plain inlier counting because a dense cloud, such as the blinking signal
    registration pairs, maps *every* bead within ``inlier_tol`` under
    hundreds of different candidates, which all tie at the maximum count and
    leave the winner to the sampling order; the residuals still separate the
    true transform from the coincidental ones.

    Because only the *candidate proposal* uses the coarse overlay - not the fit
    - an inaccurate overlay (e.g. an imperfectly placed split-FOV ROI) cannot
    mis-pair beads and corrupt the transform, which otherwise makes the
    calibration hypersensitive to ROI placement.

    Parameters
    ----------
    ref_xy : np.ndarray
        ``(n_ref, 2)`` reference-channel positions in ``[x, y]``.
    c_xy : np.ndarray
        ``(n_c, 2)`` the other channel's positions, in absolute coordinates.
        The transform is fitted on these, not on ``aligned_c``.
    aligned_c : np.ndarray
        ``c_xy`` coarsely overlaid onto the reference frame, used **only** to
        propose candidate pairs. Pass ``c_xy`` itself for an identity overlay.
    inlier_tol : float
        Distance (camera pixels) within which a mapped point counts as an
        inlier: the cap on the scoring cost, and the pairing radius of the
        final assignment.
    radius : float
        Radius (camera pixels) around each reference point in which the
        overlay proposes candidate partners. It only has to be generous enough
        to contain the true partner.
    max_iter : int, optional
        Cap on the number of sampled pair combinations. Above it the samples
        are drawn at random from a fixed seed, so a calibration stays
        reproducible. Default 20000.
    ref_fov, c_fov : np.ndarray, optional
        A group label per point. When both are given, a correspondence may only
        pair points of the **same** group, at every stage: candidate proposal,
        the consensus count and the final assignment. For a multi-FOV bead
        stack this is the field of view: every field images onto the same
        sensor coordinates, so pooling them packs the cloud far denser than any
        one field is, and a reference bead can then sit within ``inlier_tol``
        of an unrelated field's bead and be paired to it. For signal
        registration it is the *frame*: a molecule may only pair with one that
        blinked in the same frame (see :func:`register_from_point_sets`).
        Either way the transform itself stays global - one optical mapping -
        and is still fitted by the caller on the pooled inliers, so every
        group's points constrain it. Without the labels the pairing falls back
        to one pooled cloud.

    Returns
    -------
    ref_idx, c_idx : np.ndarray
        Index arrays of the winning consensus's inlier pairs, into ``ref_xy``
        and ``c_xy``. Both are empty when no consensus is found, or when either
        cloud holds fewer than three points.
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    c_xy = np.asarray(c_xy, dtype=np.float64)
    aligned_c = np.asarray(aligned_c, dtype=np.float64)
    empty = (np.array([], dtype=int), np.array([], dtype=int))
    if min(len(ref_xy), len(c_xy)) < 3:
        return empty

    # per-field index blocks (ref rows, channel rows), or None to pool
    groups = _fov_groups(ref_fov, c_fov, len(ref_xy), len(c_xy))

    pairs = _ransac_candidate_pairs(ref_xy, aligned_c, groups, radius)
    if len(pairs) < 2:
        return empty
    pairs = np.asarray(pairs, dtype=int)

    # Candidates are scored by the M-estimator (MSAC) cost
    tol_sq = float(inlier_tol) ** 2
    msac_cost = _ransac_msac_cost(c_xy, groups, tol_sq)
    samples = _ransac_samples(len(pairs), max_iter)
    best_M = _ransac_best_transform(ref_xy, c_xy, pairs, samples, msac_cost)

    if best_M is None:
        return empty
    return _ransac_final_inliers(best_M, ref_xy, c_xy, groups, inlier_tol)


def match_points(
    ref_xy: np.ndarray, other_xy: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor match two point clouds across channels.

    The points are fiducial beads when registering on beads and
    single-molecule detections when registering on signal; the matching is the
    same either way.

    Parameters
    ----------
    ref_xy : np.ndarray
        ``(n_ref, 2)`` reference positions in ``[x, y]``, already mapped into
        the other channel's frame by the current transform estimate.
    other_xy : np.ndarray
        ``(n_other, 2)`` the other channel's own positions.
    max_distance : float
        Pairing radius in camera pixels; a reference point with no partner
        inside it stays unmatched.

    Returns
    -------
    ref_idx, other_idx : np.ndarray
        Index arrays of the matched pairs, into ``ref_xy`` and ``other_xy``.
        Each ``other`` point is used at most once - conflicts are resolved in
        order of increasing distance, so the closest match wins. Both are empty
        if either cloud is.
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    other_xy = np.asarray(other_xy, dtype=np.float64)
    if len(ref_xy) == 0 or len(other_xy) == 0:
        empty = np.array([], dtype=int)
        return empty, empty
    tree = KDTree(other_xy)
    dist, idx = tree.query(ref_xy, k=1)
    keep = np.where(dist <= max_distance)[0]
    # resolve duplicate targets: assign each target to its closest reference
    order = keep[np.argsort(dist[keep])]
    seen: set[int] = set()
    ref_idx, other_idx = [], []
    for r in order:
        o = int(idx[r])
        if o in seen:
            continue
        seen.add(o)
        ref_idx.append(int(r))
        other_idx.append(o)
    return np.array(ref_idx, dtype=int), np.array(other_idx, dtype=int)


def frames_in_bounds(
    n_frames: int, frame_bounds: tuple[int, int] | list | None
) -> np.ndarray:
    """The frame indices a frame-range setting allows.

    Parameters
    ----------
    n_frames : int
        Total number of frames in the movie.
    frame_bounds : tuple, list or None
        The frames to allow, following :func:`picasso.localize.identify`:
        either a single ``(min, max)`` range or a list of such ranges, both
        inclusive and 0-indexed, with ``None`` for an open end. None allows
        every frame.

    Returns
    -------
    frames : np.ndarray
        Sorted, unique frame indices, clipped to ``[0, n_frames - 1]``. Empty
        when the bounds select nothing.
    """
    n_frames = int(n_frames)
    if frame_bounds is None:
        return np.arange(n_frames, dtype=int)
    segs = frame_bounds
    first = segs[0] if len(segs) else None
    if first is None or np.isscalar(first):
        segs = [frame_bounds]  # a single (min, max) range
    mask = np.zeros(n_frames, dtype=bool)
    for lo, hi in segs:
        lo = 0 if lo is None else max(0, int(lo))
        hi = n_frames - 1 if hi is None else min(n_frames - 1, int(hi))
        if hi >= lo:
            mask[lo : hi + 1] = True
    return np.nonzero(mask)[0]


# The four mirror orientations tried when nothing is known about the optical
# path (identity, flip-x, flip-y, flip-xy); (sx, sy) are the mirror signs. A
# splitter that folds one channel about an axis is common, and no amount of
# ICP recovers from starting at the wrong orientation - the pairing has to be
# seeded at each in turn and the winner kept.
_FLIP_SIGNS = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))


def flip_affine(
    sx: float, sy: float, w: float, h: float
) -> tform.AffineTransform:
    """A pure mirror about a ``w`` x ``h`` box, per axis sign.

    Parameters
    ----------
    sx, sy : float
        ``-1`` to mirror that axis, ``+1`` to leave it.
    w, h : float
        Width and height of the box the mirror is taken about.

    Returns
    -------
    transform : picasso.transforms.AffineTransform
    """
    matrix = np.array(
        [
            [sx, 0.0, w if sx < 0 else 0.0],
            [0.0, sy, h if sy < 0 else 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return tform.AffineTransform(matrix=matrix)


def flip_seed_transforms(
    channel: int,
    region_rects: list | None,
    frame_shape: tuple[int, int] | None,
    ref_xy: np.ndarray,
    chan_xy: np.ndarray,
) -> list[tform.AffineTransform]:
    """Coarse reference->channel seed transforms, one per mirror orientation.

    Split-FOV (``region_rects`` given): the mirror is taken about the channel's
    region and the region origins supply the placement. Separate movies
    (``frame_shape`` given): the mirror is taken about the frame and the
    translation comes from aligning the pooled detection centroids.

    Seeds are always affine, whatever the registration is finally fitted with:
    they only have to get the pairing started.

    Parameters
    ----------
    channel : int
        Index of the channel being seeded.
    region_rects : list or None
        Split-FOV region rectangles, reference first. None for separate movies.
    frame_shape : tuple or None
        ``(height, width)`` of the frame, for the separate-movie case.
    ref_xy, chan_xy : np.ndarray
        Pooled detections of the reference and of this channel, used to line up
        the centroids in the separate-movie case.

    Returns
    -------
    seeds : list of picasso.transforms.AffineTransform
        One per mirror orientation, to be tried in turn.
    """
    seeds = []
    identity = tform.identity()
    if region_rects is not None:
        (cy0, cx0), (cy1, cx1) = region_rects[channel]
        h, w = float(cy1 - cy0), float(cx1 - cx0)
        for sx, sy in _FLIP_SIGNS:
            seeds.append(
                localize.compose_region_transforms(
                    [region_rects[0], region_rects[channel]],
                    [identity, flip_affine(sx, sy, w, h)],
                )[1]
            )
        return seeds
    if len(ref_xy) == 0 or len(chan_xy) == 0:
        return [identity]
    h, w = (
        (float(frame_shape[0] - 1), float(frame_shape[1] - 1))
        if frame_shape is not None
        else (0.0, 0.0)
    )
    for sx, sy in _FLIP_SIGNS:
        seed = flip_affine(sx, sy, w, h)
        pred = seed.apply(ref_xy)
        seeds.append(
            seed.compose_translations(
                post=chan_xy.mean(axis=0) - pred.mean(axis=0)
            )
        )
    return seeds


def _pooled(by_frame: dict, frames: list) -> tuple[np.ndarray, np.ndarray]:
    """One ``(xy, frame_label)`` cloud from a per-frame mapping."""
    if not frames:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    xy = np.vstack([by_frame[f] for f in frames])
    labels = np.concatenate(
        [np.full(len(by_frame[f]), f, dtype=int) for f in frames]
    )
    return xy, labels


def _bootstrap_transform(
    ref_by_frame: dict,
    chan_by_frame: dict,
    common: list,
    model: str,
    radius: float,
    inlier_tol: float,
) -> tform.Transform | None:
    """A first reference->channel transform with no prior estimate.

    Pairing normally starts from a seed that is already close. Without one,
    propose correspondences with :func:`ransac_match` on the pooled
    detections, labeling every point with its **frame** so a molecule can only
    pair with one that blinked in the same frame - the constraint that makes
    signal registration work at all, and exactly what the field-of-view
    grouping already implements. The overlay is the identity, which only has to
    be good enough to propose candidates: the winning transform is fitted on
    absolute coordinates, so a channel offset well inside ``radius`` is
    recovered even though the overlay ignores it.

    Returns None when no consensus is found.
    """
    ref_xy, ref_frames = _pooled(ref_by_frame, common)
    c_xy, c_frames = _pooled(chan_by_frame, common)
    ref_idx, c_idx = ransac_match(
        ref_xy,
        c_xy,
        c_xy,  # identity overlay: candidates only, the fit uses absolutes
        inlier_tol,
        radius,
        ref_fov=ref_frames,
        c_fov=c_frames,
    )
    if len(ref_idx) < _icp_min_pairs(model):
        return None
    transform, _ = fit_registration(
        ref_xy[ref_idx], c_xy[c_idx], model, final=False
    )
    return transform


def _icp_from_seed(
    ref_by_frame: dict,
    chan_by_frame: dict,
    common: list,
    seed: tform.Transform,
    model: str,
    tols: np.ndarray,
    tol_lo: float,
) -> tuple:
    """One ICP run from one seed, plus the closing robust trim.

    Returns ``(transform, matched_ref, matched_c, fitted_model)``. Split out of
    :func:`register_from_point_sets` so several candidate seeds - the mirror
    orientations - can be run and compared."""
    transform = seed
    fitted_model = _icp_model(model)
    floor = _icp_min_pairs(model)
    matched_ref = matched_c = np.empty((0, 2))
    for k, tol in enumerate(tols):
        acc_ref, acc_c = [], []
        for f in common:
            rxy = ref_by_frame[f]
            cxy = chan_by_frame[f]
            pred = transform.apply(rxy)
            ri, ci = match_points(pred, cxy, tol)
            if len(ri):
                acc_ref.append(rxy[ri])
                acc_c.append(cxy[ci])
        if not acc_ref:
            break
        matched_ref = np.vstack(acc_ref)
        matched_c = np.vstack(acc_c)
        if len(matched_ref) < floor:
            break
        transform, fitted_model = fit_registration(
            matched_ref, matched_c, model, final=k == len(tols) - 1
        )

    # robust trim: drop coincidental pairs far from the converged transform,
    # then re-fit once on the inliers
    if len(matched_ref) >= floor:
        resid = matched_c - transform.apply(matched_ref)
        dist = np.sqrt(np.sum(resid**2, axis=1))
        keep = dist <= max(tol_lo, 3.0 * np.median(dist))
        if keep.sum() >= floor:
            matched_ref = matched_ref[keep]
            matched_c = matched_c[keep]
            transform, fitted_model = fit_registration(
                matched_ref, matched_c, model
            )
    return transform, matched_ref, matched_c, fitted_model


def _bootstrap_seed(
    ref_by_frame: dict,
    chan_by_frame: dict,
    common: list,
    model: str,
    radius: float,
    tol_hi: float,
) -> tform.Transform:
    """A first reference->channel transform for a seedless
    :func:`register_from_point_sets` call, raising if RANSAC finds no
    consistent set of correspondences to bootstrap from."""
    seed = _bootstrap_transform(
        ref_by_frame, chan_by_frame, common, model, radius, tol_hi
    )
    if seed is None:
        raise ValueError(
            "Could not find a consistent set of correspondences between "
            "the reference and this channel. The channels may not share "
            "signal, or the offset between them may exceed the search "
            "radius."
        )
    return seed


def _best_icp_across_seeds(
    ref_by_frame: dict,
    chan_by_frame: dict,
    common: list,
    seeds: list,
    model: str,
    tols: np.ndarray,
    tol_lo: float,
) -> tuple:
    """Run ICP from every candidate seed and keep the one with the most
    correspondences among the geometrically plausible - how a mirrored
    channel is recovered in :func:`register_from_point_sets`. Falls back to
    the first seed's result when none is plausible, so the caller's own
    pair-count check reports the real problem rather than a bare
    implausibility."""
    with warnings.catch_warnings():
        if len(seeds) > 1:
            # Only one candidate is kept, so a losing orientation's "thin
            # data" warning would be noise about a registration nobody sees.
            # The winner is refitted by the caller, outside this block, so a
            # genuine warning about the *kept* transform still reaches the
            # user.
            warnings.simplefilter("ignore")
        best = None
        for candidate in seeds:
            result = _icp_from_seed(
                ref_by_frame,
                chan_by_frame,
                common,
                candidate,
                model,
                tols,
                tol_lo,
            )
            # A wrong mirror orientation converges onto coincidental pairs,
            # which are few at the tightest radius and usually imply an
            # absurd scale - so the most pairs wins, among the
            # geometrically sane.
            if not tform.is_plausible(result[0]):
                continue
            if best is None or len(result[1]) > len(best[1]):
                best = result
        if best is None:
            best = _icp_from_seed(
                ref_by_frame,
                chan_by_frame,
                common,
                seeds[0],
                model,
                tols,
                tol_lo,
            )
    return best


def register_from_point_sets(
    ref_by_frame: dict,
    chan_by_frame: dict,
    model: str,
    box: int,
    seed: tform.Transform | None = None,
    n_iter: int = 4,
    max_pair_distance: float | None = None,
    min_pairs: int = 20,
    bootstrap_radius: float | None = None,
) -> dict:
    """Fit the reference -> channel transform from per-frame point clouds.

    Correspondences are paired at the current transform, a fresh transform is
    fitted on them, and the pairing radius shrinks over ``n_iter`` ICP passes;
    a robust trim then drops the coincidental pairs and refits once on the
    inliers.

    Parameters
    ----------
    ref_by_frame, chan_by_frame : dict
        Frame index to that frame's ``(n, 2)`` ``[x, y]`` detections, for the
        reference channel and the channel being registered. Only frames present
        in both are used: the channels are frame-synchronized, so a molecule
        may only be paired with one that fluoresced in the *same* frame.
    model : str
        Transform model to fit, as in :mod:`picasso.transforms`.
    box : int
        Box side length (camera pixels). Sets the default pairing radii.
    seed : picasso.transforms.Transform or list, optional
        The transform the first pass pairs at. Pass a stored one to *refine* a
        registration that has drifted. None (the default) builds one from
        scratch, bootstrapping the pairing with a RANSAC consensus. A **list**
        of candidate seeds - e.g. the mirror orientations from
        :func:`flip_seed_transforms` - runs the pairing from each and keeps the
        one that ends with the most correspondences and a geometrically sane
        transform, which is how a mirrored channel is recovered.
    n_iter : int, optional
        Number of ICP passes, over which the pairing radius shrinks from
        ``max_pair_distance`` to ``max(2, 0.3 * box)``. Default 4.
    max_pair_distance : float, optional
        Radius (camera pixels) the first pass pairs at. None (the default) uses
        one ``box``, which absorbs a seed's residual drift without inviting
        coincidental cross-molecule pairs.
    min_pairs : int, optional
        Fewest correspondences the result may rest on. Default 20.
    bootstrap_radius : float, optional
        Radius (camera pixels) the seedless bootstrap proposes candidate pairs
        within, so it bounds the inter-channel offset that can be recovered.
        None (the default) uses ``20 * box``. Ignored when ``seed`` is given.

    Returns
    -------
    info : dict
        ``transform`` (the fitted reference -> channel transform),
        ``n_matches`` (correspondences it was fitted on), ``ref_xy`` / ``c_xy``
        (those correspondences), ``rms`` (their residual, camera pixels),
        ``model`` (the model actually fitted) and ``model_requested`` (the one
        asked for; the two differ when too few pairs survived).

    Raises
    ------
    ValueError
        If no frame carries detections in both channels, if the seedless
        bootstrap finds no consistent set of correspondences, or if fewer than
        ``min_pairs`` survive.
    """
    common = sorted(set(ref_by_frame) & set(chan_by_frame))
    if not common:
        raise ValueError(
            "No frames with detections in both the reference and this "
            "channel; the channels may not share signal, or the movies are "
            "not frame-synchronized."
        )
    if max_pair_distance is None:
        # a seed is already close, so about one box absorbs the residual drift
        # without inviting coincidental cross-molecule pairs
        max_pair_distance = float(box)
    tol_hi = float(max_pair_distance)
    tol_lo = max(2.0, 0.3 * box)

    if seed is None:
        if bootstrap_radius is None:
            # nothing is known about the offset, so candidates are proposed
            # over a generous fraction of the frame
            bootstrap_radius = 20.0 * float(box)
        seed = _bootstrap_seed(
            ref_by_frame,
            chan_by_frame,
            common,
            model,
            bootstrap_radius,
            tol_hi,
        )
    seeds = list(seed) if isinstance(seed, (list, tuple)) else [seed]

    tols = np.linspace(tol_hi, tol_lo, max(1, int(n_iter)))
    transform, matched_ref, matched_c, fitted_model = _best_icp_across_seeds(
        ref_by_frame, chan_by_frame, common, seeds, model, tols, tol_lo
    )
    if len(seeds) > 1 and len(matched_ref) >= _icp_min_pairs(model):
        # refit the winner audibly, so a thin-data warning is raised for the
        # registration that is actually kept
        transform, fitted_model = fit_registration(
            matched_ref, matched_c, model
        )
    n_pairs = int(len(matched_ref))
    if n_pairs < min_pairs:
        # Neutral wording: this is shared by the bead and the signal builders,
        # each of which appends the advice that applies to its own input.
        raise ValueError(
            f"Only {n_pairs} correspondences survived (need >= {min_pairs})."
        )
    resid = matched_c - transform.apply(matched_ref)
    rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
    return {
        "transform": transform,
        "n_matches": n_pairs,
        "ref_xy": matched_ref,
        "c_xy": matched_c,
        "rms": rms,
        # what was actually fitted, and what was asked for: they differ when
        # too few pairs survived for the chosen model
        "model": fitted_model,
        "model_requested": model,
    }


def detections_by_frame(
    movie,
    minimum_ng: float | None,
    box: int,
    frames: np.ndarray,
    wavelet: wavelets.WaveletParameters | None = None,
) -> dict:
    """Detect spots on selected frames, grouped by frame.

    Parameters
    ----------
    movie : AbstractPicassoMovie
        The movie to detect in.
    minimum_ng : float or None
        Minimum net gradient for a spot to be kept. Ignored if
        ``wavelet`` is given.
    box : int
        Box side length (camera pixels) used for the detection.
    frames : np.ndarray
        Indices of the frames to detect on, e.g. from
        :func:`frames_in_bounds`.
    wavelet : wavelet.WaveletParameters, optional
        Settings of the wavelet identification, see
        :func:`picasso.localize.identify`. Default is None, i.e. the net
        gradient identification.

    Returns
    -------
    by_frame : dict
        Frame index to that frame's ``(n, 2)`` ``[x, y]`` detections, in
        absolute camera pixels, in the form :func:`register_from_point_sets`
        takes. Frames with no detection are absent, and an empty dict is
        returned when nothing was detected at all. The frames are read into one
        stack and identified in a single pass, so the keys are the *original*
        frame indices rather than positions within that stack.
    """
    stack = np.stack([np.asarray(movie[int(f)]) for f in frames])
    ids, _ = localize.identify(stack, minimum_ng, box, wavelet=wavelet)
    if len(ids) == 0:
        return {}
    frame = np.asarray(ids["frame"], dtype=np.int64)
    xy = np.column_stack(
        [
            np.asarray(ids["x"], dtype=np.float64),
            np.asarray(ids["y"], dtype=np.float64),
        ]
    )
    frames = np.asarray(frames)
    return {int(frames[f]): xy[frame == f] for f in np.unique(frame)}


def _minimum_ng_for(
    minimum_ng: float | list | None, channel: int
) -> float | None:
    """``minimum_ng`` for one channel, from a scalar or a per-channel list;
    None stays None (wavelet identification, which has no net gradient)."""
    if minimum_ng is None:
        return None
    if isinstance(minimum_ng, (list, tuple, np.ndarray)):
        return float(minimum_ng[channel])
    return float(minimum_ng)


def _region_mask(xy: np.ndarray, region: list) -> np.ndarray:
    """Which rows of ``xy`` (in ``[x, y]``) fall inside a
    ``[[y_min, x_min], [y_max, x_max]]`` rectangle."""
    (y0, x0), (y1, x1) = localize._normalize_rect(region)
    xy = np.asarray(xy, dtype=np.float64)
    if not len(xy):
        return np.zeros(0, dtype=bool)
    return (
        (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
    )


def _by_frame_in_region(by_frame: dict, region: list) -> dict:
    """Per-frame detections restricted to one region, frames with none
    dropped."""
    out = {}
    for f, xy in by_frame.items():
        kept = np.asarray(xy)[_region_mask(xy, region)]
        if len(kept):
            out[f] = kept
    return out


def _split_fov_flip_seeds(regions: list, reference: int, channel: int) -> list:
    """Candidate reference->channel seeds for a split-FOV channel, one per
    mirror orientation.

    The drawn ROIs fix where the channel sits, so no search over *placement*
    is needed - but not how it is oriented, and a splitter that folds one
    channel about an axis is common. :func:`flip_seed_transforms` reads the
    reference as ``region_rects[0]``, so the pair is handed over
    reference-first.
    """
    rects = [
        localize._normalize_rect(regions[reference]),
        localize._normalize_rect(regions[channel]),
    ]
    return flip_seed_transforms(
        1, rects, None, np.empty((0, 2)), np.empty((0, 2))
    )


def _split_fov_calibration_keys(
    regions: list, reference: int, transforms: list
) -> dict:
    """The split-FOV half of a registration calibration.

    The inter-channel registration is stored *region-local* (relative to the
    region origins), so the channels can be re-placed at fit time by re-drawing
    the ROIs; the absolute ``channel_transforms`` are rebuilt from these plus
    the ROIs actually in use. Same layout as a split-FOV spline calibration, so
    :func:`picasso.localize.split_fov_fit_geometry` reads either.
    """
    rects = [localize._normalize_rect(r) for r in regions]
    return {
        "split_fov": True,
        "reference": int(reference),
        "regions": rects,
        "channel_registration": [
            a.to_dict()
            for a in localize.decompose_region_transforms(rects, transforms)
        ],
    }


def _registration_calibration(
    transforms: list,
    reference: int,
    model: str,
    box: int,
    minimum_ng: float | list | None,
    source: str,
    infos: list,
    channel_paths: list[str] | None,
    extra: dict | None = None,
    wavelet: wavelets.WaveletParameters | list | None = None,
) -> dict:
    """Assemble the calibration dict both builders return.

    ``transforms`` is one entry per channel in channel order, the reference's
    being the identity, stored in the same wire format as a multichannel spline
    calibration's ``channel_transforms`` so every consumer of those works
    unchanged. The detection settings are recorded for traceability: the
    minimum net gradient, or the wavelet settings as a plain dict (the file is
    YAML).
    """
    if wavelet is None:
        detection = {
            "identification_method": localize.IDENTIFY_METHOD_NET_GRADIENT,
            "minimum_ng": minimum_ng,
        }
    else:
        detection = {
            "identification_method": localize.IDENTIFY_METHOD_WAVELET,
            # one dict per channel when each has its own settings
            "wavelet": (
                wavelet.to_dict()
                if isinstance(wavelet, wavelets.WaveletParameters)
                else [_.to_dict() for _ in wavelet]
            ),
        }
    calibration = {
        "model": REGISTRATION_MODEL,
        "n_channels": len(transforms),
        "channel_transforms": [t.to_dict() for t in transforms],
        "registration_model": model,
        "reference": int(reference),
        "source": source,
        "box": int(box),
        **detection,
        # per non-reference channel, in channel order
        "n_pairs": [int(i["n_matches"]) for i in infos],
        "rms": [float(i["rms"]) for i in infos],
        "fitted_model": [i["model"] for i in infos],
        "channel_paths": list(channel_paths or []),
        "date": datetime.datetime.now().isoformat(),
        "Generated by": f"Picasso v{__version__} Channel registration",
    }
    if extra:
        calibration.update(extra)
    return calibration


def _validate_channel_setup(
    movies: list,
    regions: list | None,
    reference: int,
    movie_noun: str = "movie",
) -> tuple[bool, int]:
    """Validate the channel setup of a registration calibration.

    The channel-count / split-FOV / reference-index checks shared by
    both calibration builders.

    Parameters
    ----------
    movies : list
        The input movies (or bead movies), one per channel, or a single
        one in split-FOV mode.
    regions : list or None
        Split field of view: one ``[[y_min, x_min], [y_max, x_max]]``
        rectangle per channel, or None for separate movies.
    reference : int
        Index of the reference channel.
    movie_noun : str, optional
        Noun naming the movies in error messages. Default is
        ``"movie"``.

    Returns
    -------
    split_fov : bool
        True if the channels are regions of a single movie.
    n_channels : int
        Number of channels.

    Raises
    ------
    ValueError
        If there are fewer than 2 channels, split-FOV mode is not given
        exactly one movie, or ``reference`` is out of range.
    """
    split_fov = regions is not None
    n_channels = len(regions) if split_fov else len(movies)
    if n_channels < 2:
        raise ValueError(
            f"Channel registration needs at least 2 channels, got "
            f"{n_channels}."
        )
    if split_fov and len(movies) != 1:
        raise ValueError(
            f"Split-FOV registration takes the single {movie_noun} whose "
            f"regions are the channels, got {len(movies)} movies."
        )
    if not (0 <= reference < n_channels):
        raise ValueError(f"reference={reference} out of range.")
    return split_fov, n_channels


def _bead_channel_seed(
    model: str,
    regions: list | None,
    reference: int,
    channel: int,
    split_fov: bool,
) -> tuple:
    """Seed transform(s) and pairing radius for one bead channel.

    Every mirror orientation for split-FOV, since the ROIs fix where the
    channel sits but not how it is oriented; the identity for separate
    overlapping bead movies, which is the assumption the single-pair bead
    calibration has always made - a mirrored or far-displaced channel needs
    the split-FOV form or a signal registration, which search for the
    orientation.
    """
    if split_fov:
        return _split_fov_flip_seeds(regions, reference, channel), None
    return tform.identity(model), _BEAD_MATCH_RADIUS_PX


def _register_bead_channel(
    ref_by_frame: dict,
    chan_by_frame: dict,
    model: str,
    box: int,
    seed,
    radius: float | None,
    min_pairs: int,
    channel: int,
) -> dict:
    """One channel's bead registration, wrapping a thin-data failure with
    bead-specific advice."""
    try:
        info = register_from_point_sets(
            ref_by_frame,
            chan_by_frame,
            model,
            box,
            seed=seed,
            max_pair_distance=radius,
            min_pairs=min_pairs,
        )
    except ValueError as e:
        raise ValueError(
            f"Channel {channel}: {e} Too few matched bead pairs - check the "
            "bead images and the detection parameters, or choose a "
            "simpler transform model."
        ) from e
    info["channel"] = channel
    return info


def calibrate_channel_registration_from_beads(
    movies: list,
    box: int,
    minimum_ng: float | list,
    model: str = "affine",
    reference: int = 0,
    regions: list | None = None,
    multi_fov: bool = False,
    min_pairs: int | None = None,
    channel_paths: list[str] | None = None,
    path: str | None = None,
    wavelet: wavelets.WaveletParameters | list | None = None,
) -> dict:
    """Register channels from images of fiducial beads.

    Beads are detected and refined to sub-pixel accuracy, matched to the
    reference channel's beads, and a transform is fitted per channel.

    Parameters
    ----------
    movies : list
        One bead movie per channel, in channel order. For split field of view
        (``regions`` given) the single bead movie whose regions are the
        channels. Multi-frame movies are averaged unless ``multi_fov``.
    box : int
        Box size used to detect and fit the beads.
    minimum_ng : float, list or None
        Minimum net gradient for a bead candidate, shared or per channel.
        Ignored if ``wavelet`` is given.
    model : str, optional
        Transform model, as in :mod:`picasso.transforms`. Default "affine".
    reference : int, optional
        Index of the reference channel. Default 0.
    regions : list, optional
        Split field of view: one ``[[y_min, x_min], [y_max, x_max]]`` rectangle
        per channel, reference first, marking where each channel sits on the
        single sensor. The beads are detected once and split by region, and
        every mirror orientation is tried so a folded channel is recovered.
        Default None (separate bead movies per channel).
    multi_fov : bool, optional
        Each **frame** of the bead movie images a different field of view. The
        beads are then detected frame by frame and a bead is only ever paired
        with one in the *same* frame - different fields land on the same sensor
        coordinates, so pooling them would pair beads that are nowhere near
        each other - while every field's pairs constrain the one global
        transform. Default False: the frames are repeats of a single field and
        are averaged into one image, which is what a plain bead acquisition is.
    min_pairs : int, optional
        Fewest correspondences a channel may end with. None (the default) uses
        the minimum the transform model needs.
    channel_paths : list of str, optional
        Source paths, recorded in the calibration for traceability.
    path : str, optional
        If given, the calibration is saved there (YAML).
    wavelet : wavelet.WaveletParameters or list of them, optional
        Detect the bead candidates by wavelet segmentation with these
        settings instead of by their net gradient; a list gives one per
        channel, like ``minimum_ng``. Default is None.

    Returns
    -------
    calibration : dict
        See :func:`_registration_calibration`. The transforms map **reference
        channel coordinates into each channel**, the direction
        ``picasso.localize.get_spots_multichannel`` expects. A split-FOV
        registration additionally carries ``split_fov``, ``regions`` and the
        ROI-agnostic ``channel_registration``, so it can be re-placed at
        re-drawn ROIs.
    """
    split_fov, n_channels = _validate_channel_setup(
        movies, regions, reference, movie_noun="bead movie"
    )

    needed = tform.min_points(model)
    if min_pairs is None:
        min_pairs = needed

    def beads_by_frame(index: int) -> dict:
        """Sub-pixel bead positions of one channel, keyed by field of view.

        ``[x, y]``, in the per-frame form :func:`register_from_point_sets`
        takes. With ``multi_fov`` each frame is its own field and keeps its own
        key, so a bead can only ever be paired within it; otherwise the frames
        are averaged into one image under a single key."""
        movie = movies[0] if split_fov else movies[index]
        mng = _minimum_ng_for(minimum_ng, index)
        frames = (
            range(int(np.asarray(movie).shape[0])) if multi_fov else [None]
        )
        out = {}
        for f in frames:
            image = (
                np.asarray(np.asarray(movie)[f], dtype=np.float32)
                if multi_fov
                else localize._movie_to_image(movie)
            )
            coarse = localize._lateral_detect_beads(
                image, box, mng, wavelet=localize.wavelet_for(wavelet, index)
            )
            refined = localize._lateral_refine_bead_positions(
                image, coarse, box
            )
            if split_fov and len(refined):
                # one image holds every channel, so keep this region's beads
                refined = refined[
                    _region_mask(refined[:, ::-1], regions[index])
                ]
            if len(refined):
                # the matching machinery works in [x, y]
                out[0 if f is None else int(f)] = refined[:, ::-1]
        return out

    ref_by_frame = beads_by_frame(reference)
    if not ref_by_frame:
        raise ValueError(
            "No beads detected in the reference channel; lower the minimum "
            "net gradient (or the wavelet threshold) or check the bead image."
        )

    transforms: list = [None] * n_channels
    transforms[reference] = tform.identity(model)
    infos = []
    for c in range(n_channels):
        if c == reference:
            continue
        chan_by_frame = beads_by_frame(c)
        seed, radius = _bead_channel_seed(
            model, regions, reference, c, split_fov
        )
        info = _register_bead_channel(
            ref_by_frame,
            chan_by_frame,
            model,
            box,
            seed,
            radius,
            min_pairs,
            c,
        )
        transforms[c] = info["transform"]
        infos.append(info)

    calibration = _registration_calibration(
        transforms,
        reference,
        model,
        box,
        minimum_ng,
        "beads",
        infos,
        channel_paths,
        extra=(
            _split_fov_calibration_keys(regions, reference, transforms)
            if split_fov
            else None
        ),
        wavelet=wavelet,
    )
    if path:
        io.save_any_calibration(path, calibration)
    return calibration


def _sample_frames_for_signal(
    movies: list,
    frame_bounds: tuple[int, int] | list | None,
    max_frames: int,
) -> np.ndarray:
    """An evenly spaced sample of frames every movie has, within
    ``frame_bounds``, so the per-frame pairing stays aligned across
    channels."""
    n_frames = min(int(m.shape[0]) for m in movies)
    allowed = frames_in_bounds(n_frames, frame_bounds)
    if allowed.size == 0:
        raise ValueError("No frames in the requested frame range.")
    pick = np.unique(
        np.linspace(
            0, allowed.size - 1, min(int(max_frames), allowed.size)
        ).astype(int)
    )
    return allowed[pick]


def _signal_detections(
    movies: list,
    regions: list | None,
    reference: int,
    minimum_ng: float | list | None,
    box: int,
    sample_frames: np.ndarray,
    split_fov: bool,
    wavelet: wavelets.WaveletParameters | list | None = None,
) -> list[dict]:
    """Per-frame detections for every channel, in channel order.

    For split-FOV the single movie is detected once and split by region;
    otherwise each channel's own movie is detected with its own
    ``minimum_ng`` and wavelet settings.
    """
    if split_fov:
        movie_by_frame = detections_by_frame(
            movies[0],
            _minimum_ng_for(minimum_ng, reference),
            box,
            sample_frames,
            wavelet=localize.wavelet_for(wavelet, reference),
        )
        return [_by_frame_in_region(movie_by_frame, r) for r in regions]
    return [
        detections_by_frame(
            m,
            _minimum_ng_for(minimum_ng, c),
            box,
            sample_frames,
            wavelet=localize.wavelet_for(wavelet, c),
        )
        for c, m in enumerate(movies)
    ]


def _signal_channel_seed(
    seed_transforms: list | None,
    regions: list | None,
    reference: int,
    channel: int,
    split_fov: bool,
):
    """Seed transform(s) for one signal channel: the given
    ``seed_transforms`` entry when provided, else every mirror orientation
    for split-FOV (the ROIs fix placement but not orientation), else None to
    bootstrap the pairing from scratch."""
    if seed_transforms is not None:
        entry = seed_transforms[channel]
        return tform.from_dict(entry) if isinstance(entry, dict) else entry
    if split_fov:
        return _split_fov_flip_seeds(regions, reference, channel)
    return None


def _register_signal_channel(
    ref_by_frame: dict,
    chan_by_frame: dict,
    model: str,
    box: int,
    seed,
    n_iter: int,
    min_pairs: int,
    channel: int,
) -> dict:
    """One channel's signal registration, wrapping a thin-data failure with
    signal-specific advice."""
    try:
        info = register_from_point_sets(
            ref_by_frame,
            chan_by_frame,
            model,
            box,
            seed=seed,
            n_iter=n_iter,
            min_pairs=min_pairs,
        )
    except ValueError as e:
        raise ValueError(
            f"Channel {channel}: {e} Use a longer / denser movie, lower the "
            "minimum net gradient, or register on beads instead."
        ) from e
    info["channel"] = channel
    return info


def calibrate_channel_registration_from_signal(
    movies: list,
    box: int,
    minimum_ng: float | list,
    model: str = "affine",
    reference: int = 0,
    frame_bounds: tuple[int, int] | list | None = None,
    max_frames: int = 50,
    seed_transforms: list | None = None,
    n_iter: int = 4,
    min_pairs: int = 20,
    regions: list | None = None,
    channel_paths: list[str] | None = None,
    path: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
    wavelet: wavelets.WaveletParameters | list | None = None,
) -> dict:
    """Register channels from the experimental (blinking) signal.

    The channels are frame-synchronized, so the same emitter fluoresces in
    every channel in the same frame. Single molecules are detected on an evenly
    spaced sample of frames and paired frame by frame, which registers the
    channels without a separate bead acquisition.

    Parameters
    ----------
    movies : list
        One movie per channel, in channel order. For split field of view
        (``regions`` given) the single movie whose regions are the channels.
    box, minimum_ng : int, float or list
        Detection settings, as used for localization. ``minimum_ng`` may be
        per channel, and is ignored (and may be None) if ``wavelet`` is
        given.
    model : str, optional
        Transform model. Default "affine".
    reference : int, optional
        Index of the reference channel. Default 0.
    frame_bounds : optional
        Frames to draw the sample from, as in :func:`localize.identify`. Early
        frames are often too dense to pair unambiguously and late ones too
        sparse, so bounding this matters.
    max_frames : int, optional
        How many frames are evenly sampled from that range. Default 50.
    seed_transforms : list, optional
        One transform per channel to start the pairing from, e.g. an already
        loaded registration being re-aligned. None (the default) builds the
        registration from scratch, bootstrapping the pairing.
    n_iter, min_pairs : int, optional
        ICP passes, and the fewest correspondences a channel may end with.
    regions : list, optional
        Split field of view: one ``[[y_min, x_min], [y_max, x_max]]`` rectangle
        per channel, reference first, marking where each channel sits on the
        single sensor. The movie is detected once and the detections split by
        region. The drawn regions also seed the pairing, so no search is
        needed to get started. Default None (separate movies per channel).
    channel_paths : list of str, optional
        Source paths, recorded for traceability.
    path : str, optional
        If given, the calibration is saved there (YAML).
    progress_callback : callable, optional
        Called with the number of channels registered so far.
    wavelet : wavelet.WaveletParameters or list of them, optional
        Detect the molecules by wavelet segmentation with these settings
        instead of by their net gradient; a list gives one per channel,
        like ``minimum_ng``. Default is None.

    Returns
    -------
    calibration : dict
        As :func:`calibrate_channel_registration_from_beads`.
    """
    split_fov, n_channels = _validate_channel_setup(movies, regions, reference)
    if seed_transforms is not None and len(seed_transforms) != n_channels:
        raise ValueError(
            f"Got {len(seed_transforms)} seed transforms but "
            f"{n_channels} channels."
        )

    sample_frames = _sample_frames_for_signal(movies, frame_bounds, max_frames)
    by_channel = _signal_detections(
        movies,
        regions,
        reference,
        minimum_ng,
        box,
        sample_frames,
        split_fov,
        wavelet=wavelet,
    )
    ref_by_frame = by_channel[reference]
    if not ref_by_frame:
        raise ValueError(
            "No detections in the reference channel; lower the minimum net "
            "gradient or check the frame range."
        )

    transforms: list = [None] * n_channels
    transforms[reference] = tform.identity(model)
    infos = []
    done = 0
    for c in range(n_channels):
        if c == reference:
            continue
        seed = _signal_channel_seed(
            seed_transforms, regions, reference, c, split_fov
        )
        info = _register_signal_channel(
            ref_by_frame,
            by_channel[c],
            model,
            box,
            seed,
            n_iter,
            min_pairs,
            c,
        )
        transforms[c] = info["transform"]
        infos.append(info)
        done += 1
        if callable(progress_callback):
            progress_callback(done)

    extra = {
        "frame_bounds": frame_bounds,
        "max_frames": int(max_frames),
        "n_sampled_frames": int(len(sample_frames)),
    }
    if split_fov:
        extra.update(
            _split_fov_calibration_keys(regions, reference, transforms)
        )
    calibration = _registration_calibration(
        transforms,
        reference,
        model,
        box,
        minimum_ng,
        "signal",
        infos,
        channel_paths,
        extra=extra,
        wavelet=wavelet,
    )
    if path:
        io.save_any_calibration(path, calibration)
    return calibration
