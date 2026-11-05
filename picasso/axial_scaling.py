"""
picasso.axial_scaling
~~~~~~~~~~~~~~~~~~~~~

The magnification factor of z under a refractive-index mismatch, after
Loginov et al. (2024).

When the immersion medium of the objective (refractive index ``n1``, e.g.,
oil) and the sample (``n2``, e.g., water) differ, moving the focus by some
distance into the sample does not move it by the same distance inside the
sample. Two depths, both measured from the coverslip (the interface
between the two media), describe this:

- the *nominal* depth ``z_N``: how far the focus was moved, i.e., the
  movement of the stage or the objective (the apparent depth), and
- the *actual* depth ``z_A``: the depth in the sample that is in focus.

Their ratio ``zeta = z_A / z_N`` is the re-scaling factor of the paper and
the ``Magnification factor`` of Picasso's 3D fits: the astigmatism and
spline calibrations record z as the stage movement, with beads on the
coverslip, i.e., nominal z, and the fitted z is multiplied by the factor.
``zeta`` is below 1 when ``n2 < n1`` (oil into water: z is compressed) and
above 1 when ``n2 > n1`` (air into water).

The factor depends on the depth (Eq. 3 of the paper):

1. Near the coverslip it is constant, the *near-surface value*, which
   depends on the numerical aperture and the two refractive indices only.
   There are two estimates of it (``near_surface``):

   - ``"midpoint"`` (default): the ray whose axial wave number in the
     immersion medium is the mean of the paraxial and the marginal ones
     sets the focus (Lyakin et al., 2017, Eq. 19 of the paper). This is
     the value the authors use when they re-scale data (Sec. 3.C of the
     paper) and in their code, e.g., 0.587 for a TIRF objective (NA 1.49)
     from oil (1.518) into water (1.33). It falls steeply as the numerical
     aperture approaches the refractive index of the immersion medium,
     and it does not exist when that ray is totally internally reflected
     at the interface, e.g., for NA 1.4 from oil into air, or NA 1.7 from
     n = 1.78 into water; use ``"full-aperture"`` then.
   - ``"full-aperture"``: the whole aperture interferes constructively
     (Eq. 15 of the paper, the analytical critical value of its Eq. 3),
     e.g., 0.736 for the TIRF objective above.

2. Deeper, it follows the numerical-aperture-independent expression of
   Eq. 17 of the paper, which depends on the depth through
   ``(wavelength / 4) / (n1 * z_N)`` and approaches the paraxial value
   ``n2 / n1`` at large depths. The near-surface value applies up to the
   depth where the two meet (``near_surface_depth``); how deep that is
   depends on the wavelength, e.g., 1.2 um for NA 1.49 from oil (1.518)
   into water (1.33) at 670 nm, or 7 um for NA 0.85 from air into water
   (1.336) at 520 nm.

The paper prints Eq. 19 as its reciprocal, ``z_N / z_A``; this module
follows the authors' code (https://github.com/hoogenboom-group/SF), which
reproduces the values in the text of the paper (e.g., 1.532 for NA 0.85
from air into n = 1.336). Eq. 17 gives the factor as a function of the
nominal depth, which is what is known in an experiment; it is the same
relation as Eq. 13 of the paper (the factor as a function of the actual
depth), rearranged.

Picasso applies one magnification factor to all z. With the focal plane
near the coverslip, within ``near_surface_depth``, that is the near-surface
value. With the focal plane deeper, ``magnification_factor`` at its
nominal depth is the paper's ratio there, which describes the z of the
emitters around it the better the smaller their range is compared with the
depth.

Within a few um of the coverslip, the theory, wave-optics calculations and
measurements differ by up to about 10-20 % (Figs. 5 and 7 of the paper),
and residual spherical aberration (a correction collar not set for the
sample) changes the factor too. A factor measured on the microscope is the
reliable value; the theory is an estimate where none is available.

References
----------
Loginov, S. V., Boltje, D. B., Hensgens, M. N. F., Hoogenboom, J. P. &
van der Wee, E. B. "Depth-dependent scaling of axial distances in light
microscopy." Optica 11, 553-568 (2024).
https://doi.org/10.1364/OPTICA.520595

Lyakin, D. V., Maksimova, L. A., Sdobnov, A. Y. & Ryabukho, V. P. "The
influence of the numerical aperture of a beam probing an object on the
determination of the thickness of a layered object in confocal
microscopy." Optics and Spectroscopy 123, 487-494 (2017).
https://doi.org/10.1134/S0030400X17090235

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import cmath
import math
from typing import Literal

import numpy as np

__all__ = [
    "magnification_factor",
    "actual_depth",
    "near_surface_factor",
    "near_surface_depth",
]

#: Near-surface value from the ray in the middle of the axial wave numbers
#: (Lyakin et al., 2017), as the paper re-scales data; the default.
NEAR_SURFACE_MIDPOINT = "midpoint"
#: Near-surface value from constructive interference over the whole
#: aperture (Eq. 15 of the paper).
NEAR_SURFACE_FULL_APERTURE = "full-aperture"
NEAR_SURFACE_METHODS = (NEAR_SURFACE_MIDPOINT, NEAR_SURFACE_FULL_APERTURE)

NearSurface = Literal["midpoint", "full-aperture"]


def _check_optics(na: float, n_immersion: float, n_sample: float) -> None:
    """Raise a ValueError unless the numerical aperture and the
    refractive indices describe an objective."""
    for name, value in (
        ("na", na),
        ("n_immersion", n_immersion),
        ("n_sample", n_sample),
    ):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive, got {value}.")
    if na >= n_immersion:
        raise ValueError(
            f"The numerical aperture ({na}) must be smaller than the "
            f"refractive index of the immersion medium ({n_immersion})."
        )


def _check_wavelength(wavelength: float) -> None:
    """Raise a ValueError unless ``wavelength`` is positive."""
    if not (math.isfinite(wavelength) and wavelength > 0):
        raise ValueError(f"wavelength must be positive, got {wavelength}.")


def near_surface_factor(
    *,
    na: float,
    n_immersion: float,
    n_sample: float,
    near_surface: NearSurface = NEAR_SURFACE_MIDPOINT,
) -> float:
    """The magnification factor near the coverslip, where it does not
    depend on the depth (the critical value of Loginov et al., 2024).

    Parameters
    ----------
    na : float
        Numerical aperture of the objective, smaller than
        ``n_immersion``.
    n_immersion : float
        Refractive index of the immersion medium, e.g., 1.518 for oil.
    n_sample : float
        Refractive index of the sample, e.g., 1.33 for water.
    near_surface : {"midpoint", "full-aperture"}, optional
        How the value is estimated: from the ray in the middle of the
        axial wave numbers (Lyakin et al., 2017; Eq. 19 of the paper, as
        ``z_A / z_N``), as the paper re-scales data, or from constructive
        interference over the whole aperture (Eq. 15 of the paper). See
        the module docstring. Default is "midpoint".

    Returns
    -------
    factor : float
        The magnification factor ``z_A / z_N`` near the coverslip.

    Raises
    ------
    ValueError
        If ``na`` is not smaller than ``n_immersion``, a value is not
        positive, ``near_surface`` is unknown, or the midpoint estimate
        does not exist because the midpoint ray is totally internally
        reflected at the coverslip (use "full-aperture" then).
    """
    _check_optics(na, n_immersion, n_sample)
    n1, n2 = float(n_immersion), float(n_sample)
    if near_surface == NEAR_SURFACE_MIDPOINT:
        # twice the axial wave number of the midpoint ray in the immersion
        # medium, and twice the one it has in the sample, squared (units of
        # the vacuum wave number)
        k_immersion = n1 + math.sqrt(n1**2 - na**2)
        k_sample_squared = k_immersion**2 + 4 * (n2**2 - n1**2)
        if k_sample_squared <= 0:
            raise ValueError(
                f"The midpoint estimate of the magnification factor does "
                f"not exist for NA {na} from n = {n1} into n = {n2}: its "
                f"ray is totally internally reflected at the coverslip. "
                f"Use the full-aperture estimate instead."
            )
        return math.sqrt(k_sample_squared) / k_immersion
    if near_surface == NEAR_SURFACE_FULL_APERTURE:
        # the real part covers NA > n_sample
        factor = (n1 - math.sqrt(n1**2 - na**2)) / (
            n2 - cmath.sqrt(n2**2 - na**2)
        )
        return factor.real
    raise ValueError(
        f"near_surface must be one of {NEAR_SURFACE_METHODS}, got "
        f"{near_surface!r}."
    )


def _depth_dependent_factor(
    depth: np.ndarray, n1: float, n2: float, wavelength: float
) -> np.ndarray:
    """Eq. 17 of Loginov et al. (2024) at the nominal depths ``depth`` (all
    positive, in the units of ``wavelength``). The paper writes it with
    complex square roots for ``n2 < n1``; here they are real ones with the
    sign ``s = -1``."""
    s = 1.0 if n2 >= n1 else -1.0
    m = math.sqrt(abs(n2**2 - n1**2))
    with np.errstate(over="ignore"):
        delta = wavelength / (4 * n1 * depth)
        # Imaginary beyond delta = 2 (n2 < n1, within a few tens of nm of
        # the coverslip), where the factor is negative and the near-surface
        # value applies
        root = np.sqrt(np.clip(delta * (2 + s * delta), 0, None))
        return n2 / n1 * (1 + s * delta) + s * m / n1 * root


def magnification_factor(
    depth: float | np.ndarray,
    *,
    na: float,
    n_immersion: float,
    n_sample: float,
    wavelength: float,
    near_surface: NearSurface = NEAR_SURFACE_MIDPOINT,
) -> float | np.ndarray:
    """The magnification factor ``z_A / z_N`` of z at a nominal depth
    under a refractive-index mismatch, after Loginov et al. (2024).

    The factor is the near-surface value (``near_surface_factor``) up to
    ``near_surface_depth`` and follows Eq. 17 of the paper deeper (Eq. 3 of
    the paper), approaching ``n_sample / n_immersion``. See the module
    docstring.

    Parameters
    ----------
    depth : float or np.ndarray
        Nominal depth(s) ``z_N`` in nm: how far the focus was moved into
        the sample from the coverslip (the movement of the stage or the
        objective). 0 is the coverslip.
    na : float
        Numerical aperture of the objective, smaller than
        ``n_immersion``.
    n_immersion : float
        Refractive index of the immersion medium, e.g., 1.518 for oil.
    n_sample : float
        Refractive index of the sample, e.g., 1.33 for water.
    wavelength : float
        Emission wavelength (in vacuum) in nm.
    near_surface : {"midpoint", "full-aperture"}, optional
        How the near-surface value is estimated, see
        ``near_surface_factor``. Default is "midpoint".

    Returns
    -------
    factor : float or np.ndarray
        The magnification factor at each depth; a float for a single
        depth.

    Raises
    ------
    ValueError
        If a depth is negative or not finite, ``wavelength`` is not
        positive, or for the reasons ``near_surface_factor`` gives.
    """
    zeta_crit = near_surface_factor(
        na=na,
        n_immersion=n_immersion,
        n_sample=n_sample,
        near_surface=near_surface,
    )
    _check_wavelength(wavelength)
    depths = np.asarray(depth, dtype=np.float64)
    if not np.all(np.isfinite(depths)) or np.any(depths < 0):
        raise ValueError(
            "depth must be finite and not negative (nm into the sample "
            "from the coverslip)."
        )
    n1, n2 = float(n_immersion), float(n_sample)
    flat = depths.ravel()
    factor = np.full(flat.shape, zeta_crit)
    deep = flat > 0
    zeta_univ = _depth_dependent_factor(flat[deep], n1, n2, wavelength)
    # Eq. 3: the depth-dependent factor, but not beyond the near-surface
    # value, where Eq. 17 no longer holds
    if n2 >= n1:
        factor[deep] = np.minimum(zeta_univ, zeta_crit)
    else:
        factor[deep] = np.maximum(zeta_univ, zeta_crit)
    if depths.ndim == 0:
        return float(factor[0])
    return factor.reshape(depths.shape)


def actual_depth(
    depth: float | np.ndarray,
    *,
    na: float,
    n_immersion: float,
    n_sample: float,
    wavelength: float,
    near_surface: NearSurface = NEAR_SURFACE_MIDPOINT,
) -> float | np.ndarray:
    """The actual depth ``z_A`` in the sample that is in focus at a nominal
    depth ``z_N``, i.e., ``z_N`` times the magnification factor, as Loginov
    et al. (2024) re-scale the z of image stacks.

    Parameters
    ----------
    depth : float or np.ndarray
        Nominal depth(s) ``z_N`` in nm: how far the focus was moved into
        the sample from the coverslip. 0 is the coverslip.
    na : float
        Numerical aperture of the objective, smaller than
        ``n_immersion``.
    n_immersion : float
        Refractive index of the immersion medium, e.g., 1.518 for oil.
    n_sample : float
        Refractive index of the sample, e.g., 1.33 for water.
    wavelength : float
        Emission wavelength (in vacuum) in nm.
    near_surface : {"midpoint", "full-aperture"}, optional
        How the near-surface value is estimated, see
        ``near_surface_factor``. Default is "midpoint".

    Returns
    -------
    depth : float or np.ndarray
        The actual depth(s) ``z_A`` in nm; a float for a single depth.

    Raises
    ------
    ValueError
        For the reasons ``magnification_factor`` gives.
    """
    factor = magnification_factor(
        depth,
        na=na,
        n_immersion=n_immersion,
        n_sample=n_sample,
        wavelength=wavelength,
        near_surface=near_surface,
    )
    if np.ndim(factor) == 0:
        return float(depth) * factor
    return np.asarray(depth, dtype=np.float64) * factor


def near_surface_depth(
    *,
    na: float,
    n_immersion: float,
    n_sample: float,
    wavelength: float,
    near_surface: NearSurface = NEAR_SURFACE_MIDPOINT,
) -> float:
    """The nominal depth up to which the magnification factor is the
    near-surface value; deeper, it changes with the depth.

    It is where Eq. 17 of Loginov et al. (2024) reaches the near-surface
    value, solved for the depth.

    Parameters
    ----------
    na : float
        Numerical aperture of the objective, smaller than
        ``n_immersion``.
    n_immersion : float
        Refractive index of the immersion medium, e.g., 1.518 for oil.
    n_sample : float
        Refractive index of the sample, e.g., 1.33 for water.
    wavelength : float
        Emission wavelength (in vacuum) in nm.
    near_surface : {"midpoint", "full-aperture"}, optional
        How the near-surface value is estimated, see
        ``near_surface_factor``. Default is "midpoint".

    Returns
    -------
    depth : float
        The nominal depth ``z_N`` in nm; ``math.inf`` if the factor does
        not change with the depth (equal refractive indices).

    Raises
    ------
    ValueError
        If ``wavelength`` is not positive, or for the reasons
        ``near_surface_factor`` gives.
    """
    zeta_crit = near_surface_factor(
        na=na,
        n_immersion=n_immersion,
        n_sample=n_sample,
        near_surface=near_surface,
    )
    _check_wavelength(wavelength)
    n1, n2 = float(n_immersion), float(n_sample)
    s = 1.0 if n2 >= n1 else -1.0
    m = math.sqrt(abs(n2**2 - n1**2))
    # Eq. 17 with factor = zeta_crit is a quadratic in 1 + s * delta; this
    # root is the one of the unsquared equation
    one_plus_delta = zeta_crit * n2 / n1 - s * m / n1 * math.sqrt(
        abs(zeta_crit**2 - 1)
    )
    delta = s * (one_plus_delta - 1)
    if delta <= 0:
        return math.inf
    return wavelength / (4 * n1 * delta)
