"""
picasso.transforms
~~~~~~~~~~~~~~~~~~

Selectable 2D geometric transform models.

Picasso fits a coordinate transform in two places: the channel registration
of a multichannel / split-FOV cubic-spline PSF calibration (reference channel
-> channel ``c``), and the lateral astigmatism / chromatic corrections
(target image -> reference frame). This module supplies the models they can
be fitted with, in order of increasing flexibility:

``translation``
    2 DOF. A rigid shift in x and y, nothing else. The right model when the
    two frames are known to differ only by an offset - a stage drift, a
    detector read out with a different origin, or a splitter whose two halves
    are optically identical and only misplaced. With one free parameter per
    axis it is the least noise-prone of the models, and a single
    correspondence already fixes it.
``affine``
    6 DOF. Translation, rotation, anisotropic scale and shear. Parallel lines
    stay parallel. The default, and what an aligned image splitter does to
    first order.
``projective``
    8 DOF homography. Adds the keystone/perspective component a tilted
    dichroic or an unequal path length introduces, so straight lines stay
    straight but parallel lines may converge.
``polynomial``
    Degree 2 (12 coefficients) or 3 (20 coefficients). A smooth global warp
    that follows genuine field distortion. Not an optical model, and it
    extrapolates badly outside the correspondences it was fitted on - see
    ``Transform.domain``.

Every model exposes :meth:`Transform.jacobian`, the local linear part. That is
what lets a non-affine model reach the spline fit kernels, which linearize the
channel map around each spot (see ``picasso.localize.channel_roi_geometry``).

The design follows globLoc's registration step and SMAP's ``fitgeotrans``
menu; SMAP's local models (``lwm``, ``pwl``) are not offered.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

__all__ = [
    "MODELS",
    "Transform",
    "TranslationTransform",
    "AffineTransform",
    "ProjectiveTransform",
    "PolynomialTransform",
    "estimate",
    "identity",
    "from_dict",
    "min_points",
    "is_plausible",
    "warp_image",
    "channel_jacobians",
]

MODELS = (
    "translation",
    "affine",
    "projective",
    "polynomial2",
    "polynomial3",
)

Model = Literal[
    "translation", "affine", "projective", "polynomial2", "polynomial3"
]

# Minimum correspondences per model. A model fitted on exactly this many
# points interpolates them exactly and says nothing about the field in
# between, hence the "thin data" warning at _WARN_FACTOR times these.
_MIN_POINTS = {"translation": 1, "affine": 3, "projective": 4}
_WARN_FACTOR = 3


def _polynomial_degree(model: str) -> int | None:
    """The degree of a polynomial model, or None if ``model`` is not one."""
    if model.startswith("polynomial"):
        suffix = model[len("polynomial") :]
        if suffix.isdigit():
            return int(suffix)
    return None


def _monomial_powers(degree: int) -> list[tuple[int, int]]:
    """Graded-lexicographic monomial exponents ``(p, q)`` for ``u**p * v**q``
    up to ``degree``.

    The constant term is always first, which is what makes a translation of
    the *output* a change to a single coefficient (see
    :meth:`PolynomialTransform.compose_translations`).
    """
    return [(i - j, j) for i in range(degree + 1) for j in range(i + 1)]


def min_points(model: Model) -> int:
    """Minimum number of correspondences needed to fit ``model``.

    Parameters
    ----------
    model : str
        The transform model, one of :data:`MODELS`. See the module
        docstring.

    Returns
    -------
    n_points : int
        1 for translation, 3 for affine, 4 for projective, 6 for
        ``polynomial2`` and 10 for ``polynomial3``.

    Raises
    ------
    ValueError
        If ``model`` is not one of :data:`MODELS`.
    """
    _validate_model(model)
    degree = _polynomial_degree(model)
    if degree is not None:
        return len(_monomial_powers(degree))
    return _MIN_POINTS[model]


def _validate_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError(
            f"Unknown transform model '{model}'; expected one of {MODELS}."
        )


def _as_xy(xy) -> np.ndarray:
    """Coerce to a C-contiguous ``(n, 2)`` float64 array of points."""
    out = np.asarray(xy, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(1, 2)
    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(
            f"Expected (n, 2) coordinates, got shape {out.shape}."
        )
    return out


def _domain_of(xy: np.ndarray) -> np.ndarray:
    """``[[xmin, ymin], [xmax, ymax]]`` bounding box of ``xy``."""
    return np.array([xy.min(axis=0), xy.max(axis=0)], dtype=np.float64)


def _domain_to_list(domain: np.ndarray | None) -> list | None:
    return None if domain is None else np.asarray(domain).tolist()


def _check_pairs(src: np.ndarray, dst: np.ndarray, model: Model) -> None:
    """Raise if there are too few correspondences, warn if there are barely
    enough."""
    if len(src) != len(dst):
        raise ValueError(
            f"src_xy and dst_xy must be the same length, got {len(src)} and "
            f"{len(dst)}."
        )
    needed = min_points(model)
    degree = _polynomial_degree(model)
    name = (
        f"A degree-{degree} polynomial"
        if degree is not None
        else f"A{'n' if model == 'affine' else ''} {model} transform"
    )
    if len(src) < needed:
        raise ValueError(
            f"{name} needs at least {needed} "
            f"correspondence{'s' if needed != 1 else ''}; only "
            f"{len(src)} were given. Use more beads, spread them across the "
            "field, or choose a simpler model."
        )
    if model == "translation":
        # The generic warning below advises a simpler model, and there is
        # none; what a thinly fitted translation is short of is averaging.
        if len(src) < _WARN_FACTOR:
            warnings.warn(
                f"A translation is being fitted on only {len(src)} "
                "correspondence"
                f"{'s' if len(src) != 1 else ''}; the shift inherits their "
                "localization noise instead of averaging it out. Use more "
                "beads.",
                stacklevel=3,
            )
        return
    if len(src) < _WARN_FACTOR * needed:
        warnings.warn(
            f"{name} is being fitted on only {len(src)} correspondences "
            f"(the minimum is {needed}). It will interpolate the noise in "
            "them rather than average it out; a simpler model is likely to "
            "register better.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Transform:
    """Base class for the 2D transform models.

    A transform maps ``(n, 2)`` source coordinates to ``(n, 2)`` destination
    coordinates in the same units (camera pixels, throughout Picasso).
    Instances are immutable; every method that would change one returns a new
    transform.

    Attributes
    ----------
    model : str
        One of :data:`MODELS`.
    domain : np.ndarray or None
        ``[[xmin, ymin], [xmax, ymax]]`` bounding box of the correspondences
        the transform was fitted on, in **source** coordinates. Used to pick
        the evaluation point of :meth:`decompose`, to bound
        :func:`is_plausible`, and to tell a caller when it is extrapolating.
    """

    domain: np.ndarray | None

    @property
    def model(self) -> str:
        """The model's name, one of :data:`MODELS`."""
        raise NotImplementedError

    # -- to be implemented by the models ---------------------------------

    def apply(self, xy) -> np.ndarray:
        """Map source coordinates to destination coordinates.

        Parameters
        ----------
        xy : array
            ``(n, 2)`` source coordinates. A single ``(2,)`` point is
            accepted and treated as ``(1, 2)``.

        Returns
        -------
        dst_xy : np.ndarray
            ``(n, 2)`` destination coordinates, in the same units as ``xy``.
        """
        raise NotImplementedError

    def jacobian(self, xy) -> np.ndarray:
        """Local linear part of the transform at each of the given points.

        Parameters
        ----------
        xy : array
            ``(n, 2)`` source coordinates.

        Returns
        -------
        jacobian : np.ndarray
            ``(n, 2, 2)``, with ``J[k] = [[du/dx, du/dy], [dv/dx, dv/dy]]``
            at ``xy[k]``. Constant for an affine, position-dependent
            otherwise.
        """
        raise NotImplementedError

    def inverse(self) -> "Transform":
        """The destination -> source transform.

        Returns
        -------
        inverse : Transform
            A transform of the same model mapping destination coordinates
            back to source coordinates. Exact for affine and projective. For
            a polynomial it is the *independently fitted* reverse map, so the
            round trip is approximate;
            :attr:`PolynomialTransform.roundtrip_rms_px` reports by how much.
        """
        raise NotImplementedError

    def compose_translations(self, pre=(0.0, 0.0), post=(0.0, 0.0)):
        """The transform ``x -> self(x + pre) + post``.

        Exact for all three models. This is the single primitive behind the
        split-FOV region bookkeeping in ``picasso.localize``
        (``decompose_region_transforms`` / ``compose_region_transforms``),
        which factors a region origin in and out of a channel transform.

        Parameters
        ----------
        pre : tuple
            ``(x, y)`` shift applied to the input before the transform.
        post : tuple
            ``(x, y)`` shift applied to the output after the transform.

        Returns
        -------
        transform : Transform
            A new transform of the same model, with :attr:`domain` shifted by
            ``-pre`` if it was set.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """A JSON- and YAML-serializable description.

        Returns
        -------
        data : dict
            The model name, its parameters and its domain, using builtin
            types only: ``picasso.io.save_calibration`` uses ``yaml.dump``
            (not ``safe_dump``), which would otherwise emit
            ``!!python/object/apply:numpy...`` tags for numpy scalars, and the
            spline calibration path JSON-encodes the same dict into an HDF5
            attribute. Round-trips through :func:`from_dict`.
        """
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        """Degrees of freedom of the model."""
        raise NotImplementedError

    # -- shared -----------------------------------------------------------

    def _at(self, at=None) -> np.ndarray:
        """Point to linearize about: ``at`` if given, else the center of
        :attr:`domain`, else the origin."""
        if at is not None:
            return np.asarray(at, dtype=np.float64).reshape(2)
        if self.domain is not None:
            return 0.5 * (self.domain[0] + self.domain[1])
        return np.zeros(2, dtype=np.float64)

    def decompose(self, pixelsize: float | None = None, at=None) -> dict:
        """Human-readable decomposition of the transform at one point.

        The linear part is the local Jacobian at ``at`` (the center of
        :attr:`domain` by default, the origin if there is none), which for an
        affine is simply the constant linear part. It is split two ways,
        because the two are useful for different things:

        * by SVD into ``rotation_deg``, ``scale_major``, ``scale_minor``,
          ``mirror`` and ``flip_axis``. A reflected channel (biplane /
          spectral splitters) has ``mirror=True``, and the reported rotation
          then has the canonical axis flip removed - the axis that minimizes
          the residual rotation is named in ``flip_axis`` - so a pure mirror
          reads as ~0 degrees, not ~180. The registration refinement rebuilds
          its mirror seed from these two fields.
        * by QR into ``scale_x``, ``scale_y`` and ``shear_deg``.

        The translation is the transform's displacement at ``at``, reported in
        pixels as ``tx_px``/``ty_px`` and, when ``pixelsize`` is given, in
        nanometres as ``tx_nm``/``ty_nm``.

        Parameters
        ----------
        pixelsize : float, optional
            Camera pixel size (nm). If given, the translation is additionally
            reported in nanometres.
        at : array, optional
            ``(2,)`` point to linearize about. Defaults to the center of
            :attr:`domain`, or the origin if the transform has no domain.

        Returns
        -------
        decomposition : dict
            With keys:

            * ``rotation_deg`` (float) - rotation of the orthogonal part, the
              canonical axis flip removed if ``mirror``.
            * ``scale_major``, ``scale_minor`` (float) - singular values of
              the linear part.
            * ``mirror`` (bool) - whether the linear part is orientation
              reversing.
            * ``flip_axis`` ({"x", "y"} or None) - the flip axis removed from
              ``rotation_deg``; None unless ``mirror``.
            * ``scale_x``, ``scale_y``, ``shear_deg`` (float) - the QR split.
            * ``tx_px``, ``ty_px`` (float) - displacement at ``at``, in
              pixels.
            * ``tx_nm``, ``ty_nm`` (float) - the same in nanometres; present
              only when ``pixelsize`` is given.
        """
        point = self._at(at)
        A = self.jacobian(point[None, :])[0]
        shift = self.apply(point[None, :])[0] - point

        det = float(np.linalg.det(A))
        mirror = det < 0
        U, S, Vt = np.linalg.svd(A)
        R = U @ Vt  # orthogonal part (determinant +/- 1)
        flip_axis = None
        if mirror:
            best = None
            for axis, D in (
                ("x", np.array([[-1.0, 0.0], [0.0, 1.0]])),
                ("y", np.array([[1.0, 0.0], [0.0, -1.0]])),
            ):
                Rr = R @ D
                ang = float(np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0])))
                if best is None or abs(ang) < abs(best[0]):
                    best = (ang, axis)
            rotation, flip_axis = best
        else:
            rotation = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

        Q, Rq = np.linalg.qr(A)
        signs = np.sign(np.diag(Rq))
        signs[signs == 0] = 1.0
        Rq = Rq * signs[np.newaxis, :]
        out = {
            "rotation_deg": rotation,
            "scale_major": float(S[0]),
            "scale_minor": float(S[1]),
            "mirror": mirror,
            "flip_axis": flip_axis,
            "scale_x": float(Rq[0, 0]),
            "scale_y": float(Rq[1, 1]),
            "shear_deg": float(np.degrees(np.arctan2(Rq[0, 1], Rq[1, 1]))),
            "tx_px": float(shift[0]),
            "ty_px": float(shift[1]),
        }
        if pixelsize is not None:
            out["tx_nm"] = float(shift[0]) * pixelsize
            out["ty_nm"] = float(shift[1]) * pixelsize
        return out

    def is_identity(self, tol: float = 1e-12) -> bool:
        """Whether the transform leaves every coordinate untouched.

        Parameters
        ----------
        tol : float, optional
            Absolute tolerance of the comparison. Default is 1e-12.

        Returns
        -------
        is_identity : bool
            True if the transform is the identity within ``tol``.
        """
        raise NotImplementedError

    def allclose(self, other, tol: float = 1e-9) -> bool:
        """Whether two transforms are the same model with the same
        parameters.

        Compared by parameters, not by identity or source path: the same
        correction saved twice under different names must count as one, since
        applying it twice would correct twice.

        Parameters
        ----------
        other : Transform
            The transform to compare against. Anything that is not a
            ``Transform`` compares as False.
        tol : float
            Absolute tolerance of the comparison.

        Returns
        -------
        allclose : bool
        """
        if not isinstance(other, Transform) or other.model != self.model:
            return False
        return self._params_close(other, tol)

    def _params_close(self, other, tol: float) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.model}>"


def _matrix_3x3(matrix, name: str) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(
            f"{name} must be a (3, 3) homogeneous matrix, got shape "
            f"{m.shape}."
        )
    # C-contiguous unconditionally: `apply` multiplies by a slice of this
    # matrix, and a transposed (F-contiguous) buffer would take a different
    # BLAS path and round differently, so a transform would stop being
    # bit-identical to the same transform loaded from a file.
    return np.ascontiguousarray(m)


@dataclass(frozen=True, eq=False)
class AffineTransform(Transform):
    """A 6-DOF affine, stored as a ``(3, 3)`` homogeneous matrix whose last
    row is exactly ``[0, 0, 1]``.

    Attributes
    ----------
    matrix : np.ndarray
        The ``(3, 3)`` homogeneous matrix, always C-contiguous float64.
    domain : np.ndarray or None
        See :class:`Transform`.
    """

    matrix: np.ndarray
    domain: np.ndarray | None = None

    @property
    def model(self) -> str:
        return "affine"

    def __post_init__(self):
        object.__setattr__(
            self, "matrix", _matrix_3x3(self.matrix, "An affine matrix")
        )
        if self.domain is not None:
            object.__setattr__(
                self, "domain", np.asarray(self.domain, dtype=np.float64)
            )

    @property
    def linear(self) -> np.ndarray:
        """The ``(2, 2)`` linear part."""
        return self.matrix[:2, :2]

    @property
    def translation(self) -> np.ndarray:
        """The ``(2,)`` translation."""
        return self.matrix[:2, 2]

    def apply(self, xy) -> np.ndarray:
        # Deliberately the exact expression the (2, 3) implementation used, so
        # that an affine calibration keeps producing bit-identical numbers.
        xy = _as_xy(xy)
        return xy @ self.matrix[:2, :2].T + self.matrix[:2, 2]

    def jacobian(self, xy) -> np.ndarray:
        xy = _as_xy(xy)
        return np.broadcast_to(self.matrix[:2, :2], (len(xy), 2, 2)).copy()

    def inverse(self) -> "AffineTransform":
        matrix = np.linalg.inv(self.matrix)
        matrix[2] = (0.0, 0.0, 1.0)
        domain = None
        if self.domain is not None:
            corners = _corners(self.domain)
            domain = _domain_of(self.apply(corners))
        return AffineTransform(matrix=matrix, domain=domain)

    def compose_translations(self, pre=(0.0, 0.0), post=(0.0, 0.0)):
        pre = np.asarray(pre, dtype=np.float64).reshape(2)
        post = np.asarray(post, dtype=np.float64).reshape(2)
        linear = self.matrix[:2, :2]
        matrix = self.matrix.copy()
        matrix[:2, 2] = linear @ pre + self.matrix[:2, 2] + post
        domain = None if self.domain is None else self.domain - pre
        return AffineTransform(matrix=matrix, domain=domain)

    def is_identity(self, tol: float = 1e-12) -> bool:
        return bool(np.allclose(self.matrix, np.eye(3), atol=tol, rtol=0.0))

    def _params_close(self, other, tol: float) -> bool:
        return bool(np.allclose(self.matrix, other.matrix, atol=tol, rtol=0.0))

    def to_dict(self) -> dict:
        return {
            "model": "affine",
            "matrix": [[float(v) for v in row] for row in self.matrix],
            "domain": _domain_to_list(self.domain),
        }

    @property
    def n_params(self) -> int:
        return 6


@dataclass(frozen=True, eq=False)
class TranslationTransform(AffineTransform):
    """A 2-DOF pure shift in x and y.

    A translation is an affine whose linear part is the identity, and it is
    stored as one - the same ``(3, 3)`` homogeneous matrix, with the upper
    ``(2, 2)`` block constrained to ``I``. Deriving it from
    :class:`AffineTransform` rather than reimplementing it is what lets every
    affine-only fast path take it unchanged: the constant-Jacobian route
    through the spline fit kernels, and the single ``scipy`` call in
    :func:`warp_image`.

    Constrain a fit to this model when the two frames are known to differ
    only by an offset. It cannot absorb a rotation or a scale, so a
    misalignment that is not a pure shift shows up in the registration
    residual instead of being silently soaked up by parameters the optics do
    not justify.

    Attributes
    ----------
    matrix : np.ndarray
        The ``(3, 3)`` homogeneous matrix, always C-contiguous float64, with
        ``matrix[:2, :2] == np.eye(2)``.
    domain : np.ndarray or None
        See :class:`Transform`.
    """

    @property
    def model(self) -> str:
        return "translation"

    def __post_init__(self):
        super().__post_init__()
        if not np.array_equal(self.matrix[:2, :2], np.eye(2)):
            raise ValueError(
                "A translation matrix must have an identity linear part, got "
                f"{self.matrix[:2, :2].tolist()}. Use AffineTransform for a "
                "transform with rotation, scale or shear."
            )

    @classmethod
    def from_shift(cls, shift, domain=None) -> "TranslationTransform":
        """Build a translation from its ``(x, y)`` displacement.

        Parameters
        ----------
        shift : array
            ``(2,)`` displacement ``(dx, dy)``, in pixels.
        domain : array, optional
            ``[[xmin, ymin], [xmax, ymax]]`` to attach as
            :attr:`Transform.domain`.

        Returns
        -------
        transform : TranslationTransform
        """
        matrix = np.eye(3)
        matrix[:2, 2] = np.asarray(shift, dtype=np.float64).reshape(2)
        return cls(matrix=matrix, domain=domain)

    @property
    def shift(self) -> np.ndarray:
        """The ``(2,)`` displacement ``(dx, dy)``. Same as
        :attr:`AffineTransform.translation`."""
        return self.matrix[:2, 2]

    def inverse(self) -> "TranslationTransform":
        domain = None
        if self.domain is not None:
            domain = self.domain + self.shift
        return TranslationTransform.from_shift(-self.shift, domain=domain)

    def compose_translations(self, pre=(0.0, 0.0), post=(0.0, 0.0)):
        pre = np.asarray(pre, dtype=np.float64).reshape(2)
        post = np.asarray(post, dtype=np.float64).reshape(2)
        # The linear part is the identity, so `pre` passes through unchanged
        # and both shifts simply add.
        domain = None if self.domain is None else self.domain - pre
        return TranslationTransform.from_shift(
            self.shift + pre + post, domain=domain
        )

    def to_dict(self) -> dict:
        return {
            "model": "translation",
            "shift": [float(v) for v in self.shift],
            "domain": _domain_to_list(self.domain),
        }

    @property
    def n_params(self) -> int:
        return 2


@dataclass(frozen=True, eq=False)
class ProjectiveTransform(Transform):
    """An 8-DOF homography, stored as a ``(3, 3)`` matrix normalized so that
    ``matrix[2, 2] == 1`` where that is possible.

    Attributes
    ----------
    matrix : np.ndarray
        The ``(3, 3)`` homogeneous matrix, always C-contiguous float64.
    domain : np.ndarray or None
        See :class:`Transform`.
    """

    matrix: np.ndarray
    domain: np.ndarray | None = None

    @property
    def model(self) -> str:
        return "projective"

    def __post_init__(self):
        object.__setattr__(
            self, "matrix", _matrix_3x3(self.matrix, "A projective matrix")
        )
        if self.domain is not None:
            object.__setattr__(
                self, "domain", np.asarray(self.domain, dtype=np.float64)
            )

    def _homogeneous(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(numerators (n, 2), w (n,))`` before the perspective divide."""
        h = self.matrix
        w = xy @ h[2, :2] + h[2, 2]
        num = xy @ h[:2, :2].T + h[:2, 2]
        return num, w

    def apply(self, xy) -> np.ndarray:
        xy = _as_xy(xy)
        # A homography maps the horizon line to infinity, so w can legitimately
        # vanish; the caller sees inf/nan rather than a warning. (Some BLAS
        # builds also raise spurious FPU flags on a plain matmul, which this
        # keeps out of the way.)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            num, w = self._homogeneous(xy)
            return num / w[:, None]

    def jacobian(self, xy) -> np.ndarray:
        xy = _as_xy(xy)
        h = self.matrix
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            num, w = self._homogeneous(xy)
            # d/dx (num / w) = (dnum/dx * w - num * dw/dx) / w**2
            inv_w = 1.0 / w
            out = np.empty((len(xy), 2, 2), dtype=np.float64)
            for i in range(2):  # output component
                for j in range(2):  # input coordinate
                    out[:, i, j] = (
                        h[i, j] - num[:, i] * inv_w * h[2, j]
                    ) * inv_w
        return out

    def inverse(self) -> "ProjectiveTransform":
        matrix = _normalize_homography(np.linalg.inv(self.matrix))
        domain = None
        if self.domain is not None:
            domain = _domain_of(self.apply(_corners(self.domain)))
        return ProjectiveTransform(matrix=matrix, domain=domain)

    def compose_translations(self, pre=(0.0, 0.0), post=(0.0, 0.0)):
        # H' = T(post) @ H @ T(pre). Left-multiplying by T(post) is legitimate
        # despite the perspective divide: T(post) H x = [u + post_x * w,
        # v + post_y * w, w], which dehomogenizes to (u/w + post_x,
        # v/w + post_y).
        pre = np.asarray(pre, dtype=np.float64).reshape(2)
        post = np.asarray(post, dtype=np.float64).reshape(2)
        matrix = _translation_3x3(post) @ self.matrix @ _translation_3x3(pre)
        domain = None if self.domain is None else self.domain - pre
        return ProjectiveTransform(
            matrix=_normalize_homography(matrix), domain=domain
        )

    def is_identity(self, tol: float = 1e-12) -> bool:
        return bool(
            np.allclose(
                _normalize_homography(self.matrix),
                np.eye(3),
                atol=tol,
                rtol=0.0,
            )
        )

    def _params_close(self, other, tol: float) -> bool:
        return bool(
            np.allclose(
                _normalize_homography(self.matrix),
                _normalize_homography(other.matrix),
                atol=tol,
                rtol=0.0,
            )
        )

    def to_dict(self) -> dict:
        return {
            "model": "projective",
            "matrix": [[float(v) for v in row] for row in self.matrix],
            "domain": _domain_to_list(self.domain),
        }

    @property
    def n_params(self) -> int:
        return 8


@dataclass(frozen=True, eq=False)
class PolynomialTransform(Transform):
    """A bivariate polynomial warp of degree 2 or 3.

    The **input** is normalized before the monomials are evaluated -
    ``u = (xy - center) * scale``, landing in roughly ``[-1, 1]`` - because
    raw camera coordinates on a 2048-pixel chip give ``u**3 ~ 8.6e9`` against
    a constant column of 1, a condition number around 1e10 that burns ten of
    float64's sixteen digits. The **output** is in raw pixels. That asymmetry
    is deliberate: it makes :meth:`compose_translations` an exact two-line
    operation instead of a refit.

    ``reverse`` is an independently fitted destination -> source polynomial
    (as MATLAB's ``fitgeotrans`` does for its non-invertible types), not an
    algebraic inverse, so ``inverse().apply(apply(x))`` returns ``x`` only up
    to ``roundtrip_rms_px``. No Picasso science path inverts a channel
    transform - the reverse map is used for image warping in the quality
    control figures - so that approximation never reaches a saved coordinate.

    Attributes
    ----------
    degree : int
        2 or 3.
    forward : np.ndarray
        ``(2, n_terms)`` source -> destination coefficients, one row per
        output coordinate, ordered as :attr:`powers`.
    center : np.ndarray
        ``(2,)`` center of the forward input normalization.
    scale : float
        Isotropic scale of the forward input normalization.
    reverse : np.ndarray
        ``(2, n_terms)`` destination -> source coefficients, fitted
        independently of ``forward``.
    reverse_center : np.ndarray
        ``(2,)`` center of the reverse input normalization.
    reverse_scale : float
        Isotropic scale of the reverse input normalization.
    roundtrip_rms_px : float
        RMS of ``inverse().apply(apply(x)) - x`` over :attr:`domain` (px).
    domain : np.ndarray or None
        See :class:`Transform`.
    """

    degree: int
    forward: np.ndarray
    center: np.ndarray
    scale: float
    reverse: np.ndarray
    reverse_center: np.ndarray
    reverse_scale: float
    roundtrip_rms_px: float = 0.0
    domain: np.ndarray | None = None

    @property
    def model(self) -> str:
        return f"polynomial{self.degree}"

    def __post_init__(self):
        if f"polynomial{self.degree}" not in MODELS:
            offered = [m for m in MODELS if m.startswith("polynomial")]
            raise ValueError(
                f"A polynomial transform must be one of {offered}, got "
                f"degree {self.degree!r}."
            )
        n_terms = len(_monomial_powers(self.degree))
        for name in ("forward", "reverse"):
            coeff = np.asarray(getattr(self, name), dtype=np.float64)
            if coeff.shape != (2, n_terms):
                raise ValueError(
                    f"'{name}' must have shape {(2, n_terms)} for a "
                    f"degree-{self.degree} polynomial, got {coeff.shape}."
                )
            # C-contiguous unconditionally, so that a transform rounds the
            # same way whether it was just fitted (where the coefficients
            # arrive transposed, i.e. F-contiguous, from `lstsq`) or loaded
            # back from a file.
            object.__setattr__(self, name, np.ascontiguousarray(coeff))
        for name in ("center", "reverse_center"):
            object.__setattr__(
                self,
                name,
                np.asarray(getattr(self, name), dtype=np.float64).reshape(2),
            )
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "reverse_scale", float(self.reverse_scale))
        object.__setattr__(
            self, "roundtrip_rms_px", float(self.roundtrip_rms_px)
        )
        if self.domain is not None:
            object.__setattr__(
                self, "domain", np.asarray(self.domain, dtype=np.float64)
            )

    @property
    def powers(self) -> list[tuple[int, int]]:
        """The monomial exponents ``(p, q)`` of ``u**p * v**q``, in the column
        order of :attr:`forward` and :attr:`reverse`. The constant term is
        first."""
        return _monomial_powers(self.degree)

    def _evaluate(self, xy: np.ndarray, coeff, center, scale) -> np.ndarray:
        u = (xy - center) * scale
        return _monomials(u, self.degree) @ np.asarray(coeff).T

    def apply(self, xy) -> np.ndarray:
        xy = _as_xy(xy)
        return self._evaluate(xy, self.forward, self.center, self.scale)

    def jacobian(self, xy) -> np.ndarray:
        xy = _as_xy(xy)
        u = (xy - self.center) * self.scale
        du, dv = _monomial_gradients(u, self.degree)
        # chain rule: d(out)/d(xy) = d(out)/d(u) * d(u)/d(xy), and
        # d(u)/d(xy) is the isotropic `scale`.
        out = np.empty((len(xy), 2, 2), dtype=np.float64)
        out[:, :, 0] = (du @ self.forward.T) * self.scale
        out[:, :, 1] = (dv @ self.forward.T) * self.scale
        return out

    def inverse(self) -> "PolynomialTransform":
        domain = None
        if self.domain is not None:
            domain = _domain_of(self.apply(_corners(self.domain)))
        return PolynomialTransform(
            degree=self.degree,
            forward=self.reverse,
            center=self.reverse_center,
            scale=self.reverse_scale,
            reverse=self.forward,
            reverse_center=self.center,
            reverse_scale=self.scale,
            roundtrip_rms_px=self.roundtrip_rms_px,
            domain=domain,
        )

    def compose_translations(self, pre=(0.0, 0.0), post=(0.0, 0.0)):
        # q(x) = p(x + pre) + post. The input normalization absorbs `pre`
        # into `center`, and `post` lands on the constant monomial, which
        # _monomial_powers guarantees is the first one. Symmetrically for the
        # reverse branch, since r(y) = p^-1(y - post) - pre.
        pre = np.asarray(pre, dtype=np.float64).reshape(2)
        post = np.asarray(post, dtype=np.float64).reshape(2)
        forward = self.forward.copy()
        forward[:, 0] += post
        reverse = self.reverse.copy()
        reverse[:, 0] -= pre
        return replace(
            self,
            forward=forward,
            center=self.center - pre,
            reverse=reverse,
            reverse_center=self.reverse_center + post,
            domain=None if self.domain is None else self.domain - pre,
        )

    def is_identity(self, tol: float = 1e-12) -> bool:
        # An identity polynomial has zero non-linear coefficients and maps
        # u back onto x, which is easiest to check by evaluating it.
        probe = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
        if self.domain is not None:
            probe = _corners(self.domain)
        return bool(np.allclose(self.apply(probe), probe, atol=tol, rtol=0.0))

    def _params_close(self, other, tol: float) -> bool:
        if self.degree != other.degree:
            return False
        # Compare what the polynomials do, not how they are parameterized:
        # two different (center, scale, coefficients) triples can describe
        # the same map.
        domain = self.domain if self.domain is not None else other.domain
        if domain is None:
            domain = np.array([[-1.0, -1.0], [1.0, 1.0]])
        probe = _grid(domain, 5)
        return bool(
            np.allclose(
                self.apply(probe), other.apply(probe), atol=tol, rtol=0.0
            )
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "forward": [[float(v) for v in row] for row in self.forward],
            "center": [float(v) for v in self.center],
            "scale": float(self.scale),
            "reverse": [[float(v) for v in row] for row in self.reverse],
            "reverse_center": [float(v) for v in self.reverse_center],
            "reverse_scale": float(self.reverse_scale),
            "roundtrip_rms_px": float(self.roundtrip_rms_px),
            "domain": _domain_to_list(self.domain),
        }

    @property
    def n_params(self) -> int:
        return 2 * len(_monomial_powers(self.degree))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translation_3x3(t: np.ndarray) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, t[0]], [0.0, 1.0, t[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _normalize_homography(matrix: np.ndarray) -> np.ndarray:
    """Scale a homography so ``matrix[2, 2] == 1`` (a homography is defined
    only up to scale). Falls back to the Frobenius norm for the degenerate
    case where that entry vanishes."""
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = matrix[2, 2]
    if abs(scale) > 1e-12:
        return matrix / scale
    norm = np.linalg.norm(matrix)
    return matrix / norm if norm > 0 else matrix


def _corners(domain: np.ndarray) -> np.ndarray:
    """The four corners of a ``[[xmin, ymin], [xmax, ymax]]`` box."""
    (x0, y0), (x1, y1) = np.asarray(domain, dtype=np.float64)
    return np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64)


def _grid(domain: np.ndarray, n: int) -> np.ndarray:
    """An ``n x n`` grid of points spanning ``domain``, as ``(n*n, 2)``."""
    (x0, y0), (x1, y1) = np.asarray(domain, dtype=np.float64)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _monomials(u: np.ndarray, degree: int) -> np.ndarray:
    """``(n, n_terms)`` monomial design matrix for normalized coords ``u``."""
    x, y = u[:, 0], u[:, 1]
    return np.column_stack([x**p * y**q for p, q in _monomial_powers(degree)])


def _monomial_gradients(
    u: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(d/du, d/dv)`` of the monomial matrix, each ``(n, n_terms)``."""
    x, y = u[:, 0], u[:, 1]
    du, dv = [], []
    for p, q in _monomial_powers(degree):
        du.append(np.zeros_like(x) if p == 0 else p * x ** (p - 1) * y**q)
        dv.append(np.zeros_like(y) if q == 0 else q * x**p * y ** (q - 1))
    return np.column_stack(du), np.column_stack(dv)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def _estimate_translation(
    src: np.ndarray, dst: np.ndarray, domain: np.ndarray
) -> TranslationTransform:
    # With the linear part fixed, the least-squares shift is just the mean
    # displacement - there is nothing to solve.
    return TranslationTransform.from_shift(
        (dst - src).mean(axis=0), domain=domain
    )


def _estimate_affine(
    src: np.ndarray, dst: np.ndarray, domain: np.ndarray
) -> AffineTransform:
    # The exact operations the (2, 3) implementation used, so an affine
    # calibration keeps producing bit-identical numbers.
    a = np.hstack([src, np.ones((len(src), 1))])
    solution, *_ = np.linalg.lstsq(a, dst, rcond=None)  # (3, 2)
    matrix = np.eye(3)
    matrix[:2, :] = solution.T
    return AffineTransform(matrix=matrix, domain=domain)


def _hartley_normalization(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T, normalized)`` moving the centroid to the origin and scaling the
    mean distance from it to sqrt(2) - the conditioning Hartley's normalized
    DLT needs."""
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    mean_dist = float(np.mean(np.hypot(centered[:, 0], centered[:, 1])))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return T, centered * scale


def _estimate_projective(
    src: np.ndarray, dst: np.ndarray, domain: np.ndarray, refine: bool
) -> ProjectiveTransform:
    T_s, s = _hartley_normalization(src)
    T_d, d = _hartley_normalization(dst)
    n = len(src)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    x, y = s[:, 0], s[:, 1]
    u, v = d[:, 0], d[:, 1]
    ones = np.ones(n)
    A[0::2, 0] = -x
    A[0::2, 1] = -y
    A[0::2, 2] = -ones
    A[0::2, 6] = x * u
    A[0::2, 7] = y * u
    A[0::2, 8] = u
    A[1::2, 3] = -x
    A[1::2, 4] = -y
    A[1::2, 5] = -ones
    A[1::2, 6] = x * v
    A[1::2, 7] = y * v
    A[1::2, 8] = v
    _, _, Vt = np.linalg.svd(A)
    matrix = _normalize_homography(
        np.linalg.inv(T_d) @ Vt[-1].reshape(3, 3) @ T_s
    )

    if refine and n >= 8:
        matrix = _refine_projective(src, dst, matrix)
    return ProjectiveTransform(matrix=matrix, domain=domain)


def _refine_projective(
    src: np.ndarray, dst: np.ndarray, seed: np.ndarray
) -> np.ndarray:
    """Gauss-Newton refinement of a DLT homography against the *geometric*
    residual.

    The DLT minimizes an algebraic residual, so without this the RMS the
    quality control figures report is not the geometric RMS they claim.
    Warm-started from the DLT and discarded if it fails to improve on it.
    """
    from scipy import optimize

    def residual(params: np.ndarray) -> np.ndarray:
        h = np.append(params, 1.0).reshape(3, 3)
        w = src @ h[2, :2] + h[2, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pred = (src @ h[:2, :2].T + h[:2, 2]) / w[:, None]
        out = pred - dst
        return np.where(np.isfinite(out), out, 1e6).ravel()

    seed_params = seed.ravel()[:8]
    before = float(np.sum(residual(seed_params) ** 2))
    try:
        result = optimize.least_squares(residual, seed_params, method="lm")
    except Exception:
        return seed
    if not np.all(np.isfinite(result.x)):
        return seed
    after = float(np.sum(residual(result.x) ** 2))
    if not (after < before):
        return seed
    return _normalize_homography(np.append(result.x, 1.0).reshape(3, 3))


def _fit_polynomial_branch(
    src: np.ndarray, dst: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(coefficients (2, n_terms), center, scale)`` mapping ``src`` to
    ``dst``, with the input normalized and the output in raw pixels."""
    lo, hi = src.min(axis=0), src.max(axis=0)
    center = 0.5 * (lo + hi)
    scale = 1.0 / max(0.5 * float((hi - lo).max()), 1e-9)
    design = _monomials((src - center) * scale, degree)
    n_terms = design.shape[1]
    if np.linalg.matrix_rank(design) < n_terms:
        raise ValueError(
            f"The correspondences are degenerate (e.g. collinear) for a "
            f"degree-{degree} polynomial: the design matrix has rank "
            f"{np.linalg.matrix_rank(design)} of {n_terms}. Spread the "
            "points across the field or choose a simpler model."
        )
    solution, *_ = np.linalg.lstsq(design, dst, rcond=None)  # (n_terms, 2)
    return solution.T, center, scale


def _estimate_polynomial(
    src: np.ndarray, dst: np.ndarray, degree: int, domain: np.ndarray
) -> PolynomialTransform:
    forward, center, scale = _fit_polynomial_branch(src, dst, degree)
    reverse, r_center, r_scale = _fit_polynomial_branch(dst, src, degree)
    transform = PolynomialTransform(
        degree=degree,
        forward=forward,
        center=center,
        scale=scale,
        reverse=reverse,
        reverse_center=r_center,
        reverse_scale=r_scale,
        domain=domain,
    )
    # Measure what the independently fitted reverse map actually costs, so
    # callers (and the QC figures) can report it instead of guessing.
    probe = _grid(domain, 11)
    back = transform.inverse().apply(transform.apply(probe))
    rms = float(np.sqrt(np.mean(np.sum((back - probe) ** 2, axis=1))))
    return replace(transform, roundtrip_rms_px=rms)


def estimate(
    src_xy,
    dst_xy,
    model: Model = "affine",
    refine: bool = True,
) -> Transform:
    """Least-squares transform mapping ``src_xy`` to ``dst_xy``.

    Both inputs are ``(n, 2)`` arrays of matching point correspondences (e.g.
    the same beads seen in two channels).

    Parameters
    ----------
    src_xy, dst_xy : array
        ``(n, 2)`` correspondences, in the same units.
    model : str
        The transform model, one of :data:`MODELS`. See the module
        docstring.
    refine : bool
        Whether to follow the projective DLT with a Gauss-Newton refinement
        against the geometric residual. Ignored by the other models.

    Returns
    -------
    transform : Transform
        The fitted transform, of the class matching ``model``, with
        :attr:`Transform.domain` set to the bounding box of ``src_xy``.

    Raises
    ------
    ValueError
        If ``model`` is invalid, if there are fewer than
        :func:`min_points` correspondences, or if they are degenerate for the
        requested model. A warning (not an error) is issued when there are
        barely enough.
    """
    _validate_model(model)
    src = _as_xy(src_xy)
    dst = _as_xy(dst_xy)
    _check_pairs(src, dst, model)
    domain = _domain_of(src)
    if model == "translation":
        return _estimate_translation(src, dst, domain)
    if model == "affine":
        return _estimate_affine(src, dst, domain)
    if model == "projective":
        return _estimate_projective(src, dst, domain, refine)
    return _estimate_polynomial(src, dst, _polynomial_degree(model), domain)


def identity(model: Model = "affine", domain=None) -> Transform:
    """The identity transform of the requested model.

    Parameters
    ----------
    model : str
        The transform model, one of :data:`MODELS`. See the module
        docstring.
    domain : array, optional
        ``[[xmin, ymin], [xmax, ymax]]`` to attach as
        :attr:`Transform.domain`.

    Returns
    -------
    transform : Transform
        A transform of the requested model that maps every point onto itself.

    Raises
    ------
    ValueError
        If ``model`` is not one of :data:`MODELS`.
    """
    _validate_model(model)
    domain = None if domain is None else np.asarray(domain, dtype=np.float64)
    if model == "translation":
        return TranslationTransform(matrix=np.eye(3), domain=domain)
    if model == "affine":
        return AffineTransform(matrix=np.eye(3), domain=domain)
    if model == "projective":
        return ProjectiveTransform(matrix=np.eye(3), domain=domain)
    degree = _polynomial_degree(model)
    powers = _monomial_powers(degree)
    coeff = np.zeros((2, len(powers)), dtype=np.float64)
    # u = (x - 0) * 1 == x, so the linear monomials reproduce the input
    coeff[0, powers.index((1, 0))] = 1.0
    coeff[1, powers.index((0, 1))] = 1.0
    return PolynomialTransform(
        degree=degree,
        forward=coeff,
        center=np.zeros(2),
        scale=1.0,
        reverse=coeff.copy(),
        reverse_center=np.zeros(2),
        reverse_scale=1.0,
        roundtrip_rms_px=0.0,
        domain=domain,
    )


def from_dict(data) -> Transform:
    """Rebuild a transform from :meth:`Transform.to_dict`.

    Parameters
    ----------
    data : dict or Transform
        The dict produced by :meth:`Transform.to_dict`. A ``Transform`` is
        passed through unchanged, so callers can accept either.

    Returns
    -------
    transform : Transform

    Raises
    ------
    ValueError
        If ``data`` is neither a dict nor a ``Transform``, or if its
        ``"model"`` entry is not one of :data:`MODELS`.
    """
    if isinstance(data, Transform):
        return data
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a transform dict or Transform, got {type(data)}."
        )
    model = data.get("model")
    domain = data.get("domain")
    domain = None if domain is None else np.asarray(domain, dtype=np.float64)
    if model == "translation":
        return TranslationTransform.from_shift(data["shift"], domain=domain)
    if model == "affine":
        return AffineTransform(matrix=data["matrix"], domain=domain)
    if model == "projective":
        return ProjectiveTransform(matrix=data["matrix"], domain=domain)
    degree = _polynomial_degree(model or "")
    if degree is not None:
        return PolynomialTransform(
            degree=degree,
            forward=data["forward"],
            center=data["center"],
            scale=data["scale"],
            reverse=data["reverse"],
            reverse_center=data["reverse_center"],
            reverse_scale=data["reverse_scale"],
            roundtrip_rms_px=data.get("roundtrip_rms_px", 0.0),
            domain=domain,
        )
    raise ValueError(
        f"Unknown transform model '{model}'; expected one of {MODELS}."
    )


def is_plausible(
    transform: Transform,
    domain=None,
    det_range: tuple[float, float] = (0.5, 2.0),
    n_grid: int = 5,
) -> bool:
    """Whether a fitted transform is geometrically sane over ``domain``.

    A wrong mirror orientation converges onto coincidental correspondences and
    usually implies an absurd scale, so the local area change is required to
    stay inside ``det_range`` everywhere. Its *sign* is required to be
    constant as well: a flexible model fitted on sparse points routinely folds
    over near the edge of the field, which the magnitude alone would not
    catch.

    For an affine the Jacobian is constant, so this reduces exactly to the
    single determinant test it replaces.

    Parameters
    ----------
    transform : Transform
        The transform to check.
    domain : array, optional
        ``[[xmin, ymin], [xmax, ymax]]`` region to test over. Defaults to
        ``transform.domain``; if that is None too, only the origin is tested.
    det_range : tuple
        ``(lo, hi)`` bounds on the magnitude of the Jacobian determinant, i.e.
        on the local area change.
    n_grid : int
        The transform is sampled on an ``n_grid x n_grid`` grid spanning
        ``domain``.

    Returns
    -------
    is_plausible : bool
        False if the determinant is non-finite anywhere, leaves
        ``det_range``, or changes sign over the grid; True otherwise.
    """
    if domain is None:
        domain = transform.domain
    points = (
        np.zeros((1, 2))
        if domain is None
        else _grid(np.asarray(domain), n_grid)
    )
    det = np.linalg.det(transform.jacobian(points))
    if not np.all(np.isfinite(det)):
        return False
    lo, hi = det_range
    magnitude = np.abs(det)
    if not (np.all(magnitude >= lo) and np.all(magnitude <= hi)):
        return False
    return bool(np.all(np.sign(det) == np.sign(det[0])))


def channel_jacobians(transform_list, ref_xy) -> np.ndarray:
    """Per-point local Jacobians of several channel transforms.

    Parameters
    ----------
    transform_list : list
        The reference -> channel transforms, each a ``Transform`` or a dict
        accepted by :func:`from_dict`, one per channel.
    ref_xy : array
        ``(n, 2)`` points in the reference frame to linearize about.

    Returns
    -------
    jacobians : np.ndarray
        ``(n, n_channels, 4)`` float64 rows ``[a00, a01, a10, a11]`` - the
        layout the multichannel spline kernels consume. See
        ``picasso.localize.channel_roi_geometry``.
    """
    ref_xy = _as_xy(ref_xy)
    out = np.empty((len(ref_xy), len(transform_list), 4), dtype=np.float64)
    for c, transform in enumerate(transform_list):
        out[:, c, :] = from_dict(transform).jacobian(ref_xy).reshape(-1, 4)
    return out


# ---------------------------------------------------------------------------
# Image warping
# ---------------------------------------------------------------------------

# Above this many output pixels the general warp builds its coordinate map in
# row blocks, so a full-chip warp does not materialize several hundred MB of
# float64 grid at once.
_WARP_BLOCK_PIXELS = 4_000_000


def warp_image(
    image: np.ndarray,
    pull: Transform,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    order: int = 3,
    cval: float = 0.0,
    dtype=None,
) -> np.ndarray:
    """Resample ``image`` onto an output grid.

    Parameters
    ----------
    image : np.ndarray
        The input image, indexed ``[row, col]``.
    pull : Transform
        Maps **output** ``(x, y)`` to **input** ``(x, y)`` - the pull (or
        backward) convention that resampling needs. Note that the reference ->
        channel channel transforms are already pull maps when the output grid
        is the reference frame, so they are passed straight in without
        inversion.
    output_shape : tuple, optional
        ``(height, width)`` of the output; ``image.shape`` if omitted.
    origin : tuple
        ``(row, col)`` offset added to the output indices before ``pull`` is
        applied, so a caller can resample a window without allocating the
        whole frame.
    order : int
        Spline interpolation order.
    cval : float
        Value for samples that fall outside ``image``.
    dtype : dtype, optional
        Output dtype; the input's if omitted.

    Returns
    -------
    warped : np.ndarray
        ``(height, width)`` resampled image, indexed ``[row, col]``.
    """
    from scipy.ndimage import affine_transform, map_coordinates

    image = np.asarray(image)
    if output_shape is None:
        output_shape = image.shape
    height, width = int(output_shape[0]), int(output_shape[1])
    y0, x0 = float(origin[0]), float(origin[1])

    if isinstance(pull, AffineTransform):
        # `affine_transform` evaluates output[o] = input[matrix @ o + offset]
        # in (row, col) order, so the matrix it needs is the (x, y) affine
        # with both axes swapped - no inversion.
        t = pull.matrix
        matrix = t[:2, :2][::-1, ::-1]
        offset = t[:2, 2][::-1]
        return affine_transform(
            image,
            matrix,
            offset=matrix @ np.array([y0, x0]) + offset,
            output_shape=(height, width),
            order=order,
            mode="constant",
            cval=cval,
            output=dtype,
        )

    out = np.empty((height, width), dtype=dtype or image.dtype)
    block = max(1, int(_WARP_BLOCK_PIXELS // max(width, 1)))
    cols = np.arange(width, dtype=np.float64) + x0
    for start in range(0, height, block):
        stop = min(start + block, height)
        rows = np.arange(start, stop, dtype=np.float64) + y0
        xx, yy = np.meshgrid(cols, rows)
        src = pull.apply(np.column_stack([xx.ravel(), yy.ravel()]))
        out[start:stop] = map_coordinates(
            image,
            [
                src[:, 1].reshape(stop - start, width),
                src[:, 0].reshape(stop - start, width),
            ],
            order=order,
            mode="constant",
            cval=cval,
        )
    return out
