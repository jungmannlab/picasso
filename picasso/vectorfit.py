"""
picasso.vectorfit
~~~~~~~~~~~~~~~~~

Fits spots with a vectorial PSF model, using the VectorFit library through
its Python bindings, ``vectorfit_py``. Its CPU build is ``pip install
picassosr[vectorfit]``; its GPU (CUDA) build is the ``vectorfit-py-cu12``
package, which installs the same ``vectorfit_py`` module. Both exist for
Windows and Linux only.

Despite its name, ``vectorfit-py-cu12`` is built with CUDA 13 (driver 580
or newer). On an older driver its GPU fit crashes the process, so
:func:`gpu_unavailable_reason` checks the binary against the driver before
VectorFit touches the GPU. Its CPU fit runs on an older driver, too.

The vectorial model describes the PSF from the optics (numerical aperture,
emission wavelength, refractive indices) rather than from a measured
calibration, and recovers ``z`` directly for the 3D model.

The spots are fitted in batches; within a batch VectorFit parallelizes over
its own threads, or over the GPU, so no worker processes are started here.

:author: Thomas Bonte, 2026
"""

from __future__ import annotations

import functools
import importlib
import importlib.machinery
import importlib.util
import re
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from . import lib

#: Whether the VectorFit bindings can be imported. Probed without importing
#: them, so that importing Picasso stays fast without VectorFit.
VECTORFIT_AVAILABLE = importlib.util.find_spec("vectorfit_py") is not None

#: VectorFit device of the GPU fit: Fourier transforms in single precision,
#: as VectorFit's authors recommend ("gpu" would run them in double
#: precision).
GPU_DEVICE = "gpu-lowaccuracy"

#: Oldest NVIDIA driver of each CUDA major version (its minor versions run
#: on it). Despite its name, vectorfit-py-cu12 (0.3.0 to 0.3.2) is built
#: with CUDA 13.1, so it needs driver 580: on a CUDA 12 driver it cannot
#: use the GPU, see :func:`gpu_unavailable_reason`.
_MIN_DRIVER = {11: 450, 12: 525, 13: 580}

#: Default convergence schedule, Vermeer's. Tighter than the defaults of
#: ``vectorfit_py.FitParams`` (1e-6, 30), which stop small boxes after a
#: single iteration, far from the optimum.
TOLERANCE = 5e-10
MAX_ITERATIONS = 20

#: Spots handed to VectorFit at once; bounds the memory of one batch and
#: sets how often progress is reported and an abort is checked.
SPOTS_PER_BATCH = 20_000

#: Fit box sizes VectorFit supports and the pupil sampling each needs: the
#: chirp-z transform requires ``box + n_pupil - 1`` to be a power of 2 or
#: three times one.
N_PUPIL = {
    5: 44,
    7: 42,
    9: 40,
    11: 54,
    13: 52,
    15: 50,
    17: 48,
    19: 46,
    21: 44,
    27: 38,
}

#: VectorFit model of each "Fit model" setting.
_MODELS = {"2D": "xy_constaberrations", "3D": "xyz_constaberrations"}

#: Optical and model settings of a VectorFit fit, with their defaults. Keys
#: are written to the metadata as they are, prefixed by "VectorFit".
#:
#: - "Fit model": "2D" (x, y, photons, bg) or "3D" (adds z).
#: - "Wavelength": peak emission wavelength (nm).
#: - "NA": numerical aperture of the objective.
#: - "Refractive index medium", "Refractive index cover slip",
#:   "Refractive index immersion": of the sample medium, the cover slip
#:   and the immersion fluid.
#: - "Vertical astigmatism" (Z_2^2), "Oblique astigmatism" (Z_2^-2),
#:   "Horizontal coma" (Z_3^1), "Vertical coma" (Z_3^-1), "Spherical
#:   aberration" (Z_4^0): fixed primary aberrations (nm).
#: - "Discard outliers": drop the spots VectorFit flags as outliers (not
#:   converged, final merit more than 3 standard deviations below the
#:   batch mean, or fitted position far from the box center), as its own
#:   ``localize`` script does. ``localize.fit`` records how many were
#:   discarded in the metadata, as "VectorFit outliers discarded".
DEFAULT_PARAMETERS = {
    "Fit model": "2D",
    "Wavelength": 680.0,
    "NA": 1.5,
    "Refractive index medium": 1.406,
    "Refractive index cover slip": 1.525,
    "Refractive index immersion": 1.515,
    "Vertical astigmatism": 0.0,
    "Oblique astigmatism": 0.0,
    "Horizontal coma": 0.0,
    "Vertical coma": 0.0,
    "Spherical aberration": 0.0,
    "Discard outliers": True,
}

#: Number of VectorFit's Legendre (nodal aberration theory) coefficients
#: for ``NAT_order=1``.
_N_LEGENDRE = 13

#: Index of each primary aberration among the Legendre coefficients.
_PRIMARY_ABERRATIONS = {
    "Vertical astigmatism": 4,
    "Oblique astigmatism": 5,
    "Horizontal coma": 9,
    "Vertical coma": 10,
    "Spherical aberration": 12,
}


#: Metadata key ``localize.fit`` records the number of discarded outliers
#: under.
OUTLIERS_KEY = "VectorFit outliers discarded"


def outliers_discarded(info: dict | list[dict]) -> int | None:
    """How many outliers a VectorFit fit discarded, from its metadata.

    Unlike ``lib.get_from_metadata``, a count of 0 is returned as such.

    Parameters
    ----------
    info : dict or list of dict
        Fit or localization metadata.

    Returns
    -------
    n_outliers : int or None
        None if the metadata does not come from a VectorFit fit.
    """
    for entry in [info] if isinstance(info, dict) else info[::-1]:
        if OUTLIERS_KEY in entry:
            return entry[OUTLIERS_KEY]
    return None


def _import_vectorfit():
    """Import the VectorFit bindings, with an install hint if missing."""
    try:
        return importlib.import_module("vectorfit_py")
    except ImportError as e:
        raise ImportError(
            "Fitting with VectorFit requires the optional 'vectorfit-py' "
            "package. Install it with 'pip install picassosr[vectorfit]'."
        ) from e


def _extension_path() -> Path | None:
    """The compiled ``vectorfit_py._vectorfit_py`` module on disk, found
    without importing ``vectorfit_py``."""
    spec = importlib.util.find_spec("vectorfit_py")
    if spec is None or not spec.submodule_search_locations:
        return None
    for directory in spec.submodule_search_locations:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = Path(directory, "_vectorfit_py" + suffix)
            if path.is_file():
                return path
    return None


def _cuda_build(path: Path) -> tuple[tuple[int, int], int] | None:
    """The CUDA toolkit version a VectorFit binary was compiled with and
    the lowest GPU architecture (compute capability as an int, e.g. 80)
    it carries code for; None for a CPU build.

    The two builds install the same module, ``vectorfit_py`` (with both
    installed, the one installed last wins) and neither reports which one
    it is, so the binary itself is read. NVCC leaves its version and the
    PTX targets in it as plain text."""
    data = path.read_bytes()
    version = re.search(rb"Cuda compilation tools, release (\d+)\.(\d+)", data)
    if version is None:
        return None
    archs = [int(_) for _ in re.findall(rb"\.target sm_(\d+)", data)]
    return (int(version[1]), int(version[2])), min(archs, default=0)


def _cuda_device() -> tuple[tuple[int, int], tuple[int, int]] | None:
    """The CUDA version the NVIDIA driver supports and the compute
    capability of the first GPU, the one VectorFit fits on, as
    ``numba.cuda`` reports them; None without a CUDA GPU."""
    from numba import cuda

    if not cuda.is_available():
        return None
    return (
        tuple(cuda.cudadrv.driver.driver.get_version()),
        tuple(cuda.list_devices()[0].compute_capability),
    )


@functools.cache
def gpu_unavailable_reason() -> str | None:
    """Why VectorFit cannot fit on the GPU, or None if it can.

    The checks run before VectorFit touches the GPU because its GPU build,
    on a driver too old for the CUDA it was compiled with, crashes the
    process rather than raising an error. They need the GPU build of
    VectorFit (``vectorfit-py-cu12``), a CUDA GPU of at least the lowest
    architecture it was compiled for, and a driver supporting its CUDA
    version. Cached: the answer cannot change while Picasso runs.

    Returns
    -------
    reason : str or None
        A sentence for the user, or None if the GPU fit can run.
    """
    if not VECTORFIT_AVAILABLE:
        return (
            "VectorFit is not installed; install its GPU build with "
            "'pip install vectorfit-py-cu12'."
        )
    path = _extension_path()
    build = None if path is None else _cuda_build(path)
    if build is None:
        # The builds share their files, so uninstalling one deletes the
        # other's too: both go before the GPU build is installed.
        return (
            "The installed VectorFit is its CPU build. Install the GPU "
            "build instead with 'pip uninstall vectorfit-py "
            "vectorfit-py-cu12' followed by 'pip install vectorfit-py-cu12'."
        )
    (cuda_major, _), min_arch = build
    try:
        device = _cuda_device()
    except Exception as e:  # a broken CUDA setup
        return f"The CUDA GPU could not be queried ({e})."
    if device is None:
        return "No CUDA-capable GPU was found."
    driver, capability = device
    if driver[0] < cuda_major:
        # the package name says CUDA 12, which misleads users of CUDA 12
        misnamed = (
            f" Despite its name, vectorfit-py-cu12 is built with CUDA "
            f"{cuda_major}."
            if cuda_major != 12
            else ""
        )
        newer = (
            f"to version {_MIN_DRIVER[cuda_major]} or newer"
            if cuda_major in _MIN_DRIVER
            else f"to one supporting CUDA {cuda_major}"
        )
        return (
            f"The GPU build of VectorFit needs CUDA {cuda_major}, but the "
            f"NVIDIA driver supports CUDA {driver[0]}.{driver[1]} only."
            f"{misnamed} Update the NVIDIA driver {newer}, or fit on the "
            "CPU."
        )
    if 10 * capability[0] + capability[1] < min_arch:
        return (
            "The GPU build of VectorFit needs a GPU of compute capability "
            f"{min_arch // 10}.{min_arch % 10} or higher, but the GPU has "
            f"{capability[0]}.{capability[1]}."
        )
    return None


def gpu_available() -> bool:
    """Whether VectorFit can fit on the GPU, see
    :func:`gpu_unavailable_reason`."""
    return gpu_unavailable_reason() is None


def check_gpu() -> None:
    """Raise a RuntimeError if VectorFit cannot fit on the GPU, saying
    why (see :func:`gpu_unavailable_reason`)."""
    reason = gpu_unavailable_reason()
    if reason is not None:
        raise RuntimeError(f"VectorFit cannot fit on the GPU. {reason}")


def parameters_with_defaults(parameters: dict | None = None) -> dict:
    """Validate VectorFit settings and fill in the missing ones.

    Parameters
    ----------
    parameters : dict or None, optional
        Any subset of ``DEFAULT_PARAMETERS``. Default None (all defaults).

    Returns
    -------
    parameters : dict
        A new dict with every key of ``DEFAULT_PARAMETERS``.

    Raises
    ------
    ValueError
        If a key is unknown or a value is out of range.
    """
    parameters = dict(parameters or {})
    unknown = set(parameters) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(
            f"Unknown VectorFit parameter(s): {', '.join(sorted(unknown))}. "
            f"Known ones are: {', '.join(DEFAULT_PARAMETERS)}."
        )
    parameters = DEFAULT_PARAMETERS | parameters
    if parameters["Fit model"] not in _MODELS:
        raise ValueError(
            f"VectorFit 'Fit model' must be one of {', '.join(_MODELS)}, "
            f"not {parameters['Fit model']!r}."
        )
    for key in (
        "Wavelength",
        "NA",
        "Refractive index medium",
        "Refractive index cover slip",
        "Refractive index immersion",
    ):
        if not parameters[key] > 0:
            raise ValueError(f"VectorFit '{key}' must be positive.")
    return parameters


def check_box(box: int) -> None:
    """Raise a ValueError if VectorFit does not support the fit ``box``."""
    if box not in N_PUPIL:
        raise ValueError(
            f"VectorFit does not support a box size of {box}. Supported "
            f"box sizes are {', '.join(str(_) for _ in N_PUPIL)}."
        )


def fit_params(
    parameters: dict,
    *,
    box: int,
    pixelsize: float,
    image_shape: tuple[int, int],
    tolerance: float | None = None,
    max_iterations: int | None = None,
    n_threads: int = 1,
):
    """Build the ``vectorfit_py.FitParams`` of a fit.

    Parameters
    ----------
    parameters : dict
        VectorFit settings, see ``DEFAULT_PARAMETERS``.
    box : int
        Side length of the fitted spots (camera pixels), one of ``N_PUPIL``.
    pixelsize : float
        Effective camera pixel size (nm).
    image_shape : tuple of int
        ``(height, width)`` of the movie frames (camera pixels).
    tolerance, max_iterations : float, int or None, optional
        Convergence schedule; None picks ``TOLERANCE`` / ``MAX_ITERATIONS``.
    n_threads : int, optional
        Number of CPU threads VectorFit fits on. Default 1.

    Returns
    -------
    params : vectorfit_py.FitParams
    """
    vectorfit_py = _import_vectorfit()
    parameters = parameters_with_defaults(parameters)
    check_box(box)
    params = vectorfit_py.FitParams(
        vectorfit_py.FitParams.get_fitmodel_from_string(
            _MODELS[parameters["Fit model"]]
        )
    )
    params.imgSizeY, params.imgSizeX = (int(_) for _ in image_shape)

    # optics
    params.NA = parameters["NA"]
    params.RefMed = parameters["Refractive index medium"]
    params.RefCov = parameters["Refractive index cover slip"]
    params.RefImm = parameters["Refractive index immersion"]
    params.RefImmNom = params.RefImm
    params.Depth = 0
    params.Lambda = parameters["Wavelength"]
    params.PupilSize = params.NA / params.Lambda

    # sampling
    params.PixelSizeX = params.PixelSizeY = pixelsize
    params.SamplingDistanceX = params.SamplingDistanceY = pixelsize
    params.Mx = params.My = box
    params.Mz = params.K = 1  # single images, not a z-stack
    params.NPupil = N_PUPIL[box]
    params.XRange = params.YRange = pixelsize * box / 2

    # axial range, emitter and excitation model
    params.ZRange = [-800, 800]
    params.ZSpread = [-1000, 1000]
    params.ZType = vectorfit_py.ZTypeEnum.stage
    params.Excitation = vectorfit_py.ExcitationEnum.constant
    params.DipoleType = vectorfit_py.DipoleTypeEnum.free
    params.DoeType = vectorfit_py.DoeTypeEnum.none
    params.Alpha = params.Beta = 0
    params.FlagOTF = False

    # fit
    params.initialization = vectorfit_py.InitializationMethod.phasor
    params.Tollim = TOLERANCE if tolerance is None else tolerance
    params.NiterMax = (
        MAX_ITERATIONS if max_iterations is None else max_iterations
    )
    params.NumThreads = int(n_threads)
    params.Varfit = 0
    params.ReadNoiseStd = params.ReadNoiseVariance = 0
    return params


def _aberrations(vectorfit_py, orders: np.ndarray):
    """``(n, 3)`` (radial order, azimuthal order, coefficient) rows as a
    ``vectorfit_py.aberrationArray1D``."""
    aberrations = vectorfit_py.aberrationArray1D(len(orders))
    for i, (radial, azimuthal, coefficient) in enumerate(orders):
        aberration = vectorfit_py.Aberration()
        aberration.radial_order = int(radial)
        aberration.azimuthal_order = int(azimuthal)
        aberration.coefficient = float(coefficient)
        aberrations.set(aberration, (i,))
    return aberrations


def _fixed_aberrations(
    vectorfit_py, params, parameters: dict
) -> tuple[np.ndarray, np.ndarray]:
    """The constant aberrations of the fit, set by the primary aberrations
    of ``parameters``: the Legendre coefficients (``gammas``, nm) and the
    ``(n, 3)`` Zernike aberrations (radial order, azimuthal order,
    coefficient) of the model."""
    gammas = np.zeros(_N_LEGENDRE)
    for key, index in _PRIMARY_ABERRATIONS.items():
        gammas[index] = parameters[key]
    orders = np.array(params.InitialAberrationOrders, dtype=np.float64)
    # the Zernike coefficients the Legendre ones give
    orders[:, 2] = np.array(
        vectorfit_py.compute_xy_gamma_legendre(
            vectorfit_py.doubleArray1D(1),
            vectorfit_py.doubleArray1D(1),
            vectorfit_py.doubleArray1D(gammas),
            _aberrations(vectorfit_py, orders),
        )
    )[0]
    return gammas, orders


def _log_likelihood(spots: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-spot Poisson log-likelihood as Picasso reports it, i.e. minus
    half the deviance against the saturated model (see
    ``localize.reduced_chi_square``)."""
    spots = spots.reshape(len(spots), -1)
    mu = np.maximum(mu.reshape(len(mu), -1), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(spots > 0, spots * np.log(spots / mu), 0.0)
    return -np.sum(mu - spots + log_ratio, axis=1)


def _fit_batch(
    vectorfit_py,
    spots: np.ndarray,
    roi_yx: np.ndarray,
    params,
    gammas: np.ndarray,
    orders: np.ndarray,
    result: dict,
    device: str = "cpu",
) -> None:
    """Fit one batch of spots with VectorFit on ``device`` ("cpu" or
    ``GPU_DEVICE``) and append its results to the lists in ``result`` (see
    :func:`fit_spots`)."""
    # float64 to match VectorFit's (MATLAB) double precision
    batch = np.maximum(np.asarray(spots, np.float64), 1e-3)
    fitter = vectorfit_py.Fitter(
        vectorfit_py.doubleArray4D(np.ascontiguousarray(batch[..., None])),
        vectorfit_py.doubleArray2D(np.ascontiguousarray(roi_yx)),
        params,
        device,
    )
    fitter.update_zernike_coefficients(
        vectorfit_py.doubleArray1D(gammas),
        _aberrations(vectorfit_py, orders),
    )
    fitter.init()
    fitter.update()
    # theta also holds the fixed aberrations, after the fit parameters
    theta = np.array(fitter.theta())[:, : -len(orders)]
    result["theta"].append(theta)
    result["crlb"].append(np.array(fitter.fisher_crlb()).T)
    result["iterations"].append(np.array(fitter.num_iters()))
    mu = np.array(fitter.mu())
    result["log_likelihood"].append(_log_likelihood(batch, mu))
    outlier = np.zeros(len(batch), dtype=bool)
    outlier[np.asarray(fitter.get_outliers(), dtype=np.int64)] = True
    result["outlier"].append(outlier)


def fit_spots(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    *,
    parameters: dict,
    pixelsize: float,
    image_shape: tuple[int, int],
    tolerance: float | None = None,
    max_iterations: int | None = None,
    n_threads: int | None = None,
    use_gpu: bool = False,
    progress: lib.Progress | Literal["console"] | None = None,
    abort_callback: Callable[[], bool] | None = None,
) -> dict | None:
    """Fit a vectorial PSF model to each spot, on the CPU or the GPU.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts. Values below 1e-3 are
        clipped to it, since the Poisson likelihood needs a positive mean.
    identifications : pd.DataFrame
        The identifications the spots were cut from; their ``x`` / ``y``
        (the box centers) place each spot in the field of view.
    parameters : dict
        VectorFit settings, see ``DEFAULT_PARAMETERS``.
    pixelsize : float
        Effective camera pixel size (nm).
    image_shape : tuple of int
        ``(height, width)`` of the movie frames (camera pixels).
    tolerance, max_iterations : float, int or None, optional
        Convergence schedule; None picks ``TOLERANCE`` / ``MAX_ITERATIONS``.
    n_threads : int or None, optional
        CPU threads to fit on; None uses ``lib.n_workers()``.
    use_gpu : bool, optional
        Fit on the GPU (device ``GPU_DEVICE``), which needs the GPU build
        of VectorFit, see :func:`gpu_unavailable_reason`. Default False.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``. Reports one phase,
        "Fitting spots", over the spots, advanced after every batch.
        Default None.
    abort_callback : callable or None, optional
        Checked before every batch; returning True stops the fit. Default
        None.

    Returns
    -------
    result : dict or None
        None if aborted. Otherwise, with one entry per spot:

        - "theta": ``(n_spots, n_parameters)`` fitted y, x (nm, relative to
          the center of the box), z (nm, 3D only, VectorFit's sign),
          photons and background (photons per pixel);
        - "crlb": ``(n_spots, n_parameters)`` their Cramer-Rao lower
          bounds, as standard deviations;
        - "iterations": iterations of each fit;
        - "log_likelihood": Poisson log-likelihood at the optimum;
        - "outlier": whether VectorFit flags the spot as an outlier.

    Raises
    ------
    RuntimeError
        If ``use_gpu`` but VectorFit cannot fit on the GPU.
    """
    if use_gpu:
        check_gpu()
    vectorfit_py = _import_vectorfit()
    parameters = parameters_with_defaults(parameters)
    box = spots.shape[-1]
    if n_threads is None:
        n_threads = lib.n_workers()
    params = fit_params(
        parameters,
        box=box,
        pixelsize=pixelsize,
        image_shape=image_shape,
        tolerance=tolerance,
        max_iterations=max_iterations,
        n_threads=n_threads,
    )
    gammas, orders = _fixed_aberrations(vectorfit_py, params, parameters)
    # 1-based (y, x) of the box centers, as VectorFit indexes the frame
    roi_yx = np.stack(
        [
            identifications["y"].to_numpy(dtype=np.float64) + 1,
            identifications["x"].to_numpy(dtype=np.float64) + 1,
        ]
    )

    n_spots = len(spots)
    keys = ("theta", "crlb", "iterations", "log_likelihood", "outlier")
    result = {key: [] for key in keys}
    with lib.tracked_progress(progress, "Fitting spots", "spot") as progress:
        lib.start_phase(progress, "Fitting spots", n_spots, unit="spot")
        for start in range(0, n_spots, SPOTS_PER_BATCH):
            if callable(abort_callback) and abort_callback():
                return None
            stop = min(start + SPOTS_PER_BATCH, n_spots)
            _fit_batch(
                vectorfit_py,
                spots[start:stop],
                roi_yx[:, start:stop],
                params,
                gammas,
                orders,
                result,
                GPU_DEVICE if use_gpu else "cpu",
            )
            progress.set_value(stop)
        progress.set_value(n_spots)
    if n_spots == 0:
        n_parameters = 5 if parameters["Fit model"] == "3D" else 4
        return {
            "theta": np.zeros((0, n_parameters)),
            "crlb": np.zeros((0, n_parameters)),
            "iterations": np.zeros(0, dtype=np.int32),
            "log_likelihood": np.zeros(0),
            "outlier": np.zeros(0, dtype=bool),
        }
    return {key: np.concatenate(value) for key, value in result.items()}


def locs_from_fits(
    identifications: pd.DataFrame,
    result: dict,
    *,
    pixelsize: float,
    em: bool,
    discard_outliers: bool = True,
    spots: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert VectorFit results into a data frame of localizations.

    Parameters
    ----------
    identifications : pd.DataFrame
        The identifications the spots were cut from.
    result : dict
        The output of :func:`fit_spots`.
    pixelsize : float
        Effective camera pixel size (nm), placing the box centers, which
        are in camera pixels, in nm.
    em : bool
        Whether EMCCD was used. Its excess noise doubles the variance of
        every estimate, as in the other fits.
    discard_outliers : bool, optional
        Drop the spots VectorFit flags as outliers. Default True.
    spots : np.ndarray or None, optional
        The fitted spots in photons; with them, the ``reduced_chi_square``
        column is added (see ``localize.reduced_chi_square``). Default
        None.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, sorted by frame, with ``frame``, ``x``, ``y``,
        ``photons``, ``bg``, ``lpx``, ``lpy``, ``net_gradient``,
        ``photons_unc``, ``bg_unc``, ``log_likelihood``, ``iterations`` and
        ``reduced_chi_square`` (with ``spots``), plus ``z`` and ``lpz`` for
        the 3D model. Every length is in nm.
    """
    from .localize import _fit_statistic_columns

    pixelsize = float(pixelsize)
    if not np.isfinite(pixelsize) or pixelsize <= 0:
        raise ValueError(
            f"The camera pixel size must be positive, got {pixelsize} nm."
        )
    theta = result["theta"]
    crlb = result["crlb"] * (np.sqrt(2.0) if em else 1.0)
    is_3d = theta.shape[1] == 5
    # VectorFit fits in nm, relative to the center of the box
    x = theta[:, 1] + identifications["x"].to_numpy(np.float64) * pixelsize
    y = theta[:, 0] + identifications["y"].to_numpy(np.float64) * pixelsize
    columns = {
        "frame": identifications["frame"].to_numpy().astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "photons": theta[:, -2].astype(np.float32),
        "bg": theta[:, -1].astype(np.float32),
        "lpx": crlb[:, 1].astype(np.float32),
        "lpy": crlb[:, 0].astype(np.float32),
        **lib.net_gradient_column(identifications),
    }
    if is_3d:
        # VectorFit's z axis points the other way
        columns["z"] = (-theta[:, 2]).astype(np.float32)
        columns["lpz"] = crlb[:, 2].astype(np.float32)
    columns["photons_unc"] = crlb[:, -2].astype(np.float32)
    columns["bg_unc"] = crlb[:, -1].astype(np.float32)
    columns.update(
        _fit_statistic_columns(
            spots,
            theta.shape[1],
            em,
            log_likelihood=result["log_likelihood"],
            iterations=result["iterations"],
            chi_square=None,
            variance=None,
        )
    )
    locs = pd.DataFrame(columns)
    if discard_outliers:
        locs = locs[~result["outlier"]]
    locs = locs.sort_values(by="frame", kind="quicksort")
    return locs.reset_index(drop=True)
