"""
    picasso.zfit
    ~~~~~~~~~~~~

    Fitting z coordinates using astigmatism.

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

import multiprocessing
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor

import numba
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize_scalar

from . import lib


plt.style.use("ggplot")


def nan_index(y: np.ndarray) -> tuple[np.ndarray, callable]:
    """Find indices of NaN values in an array."""
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nan(data: np.ndarray) -> np.ndarray:
    """Linear interpolattion of NaN values in an array ``data``."""
    nans, x = nan_index(data)
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data


def calibrate_z(
    locs: np.recarray,
    info: list[dict],
    d: float,
    magnification_factor: float,
    path: str | None = None,
) -> dict:
    """Given localizations of a calibration sample (e.g., gold beads at
    different z positions), calibrate the z-axis by fitting a polynomial
    to the mean spot width/height of each frame. See Huang et al.
    Science, 2008. DOI: 10.1126/science.1153529.

    Parameters
    ----------
    locs : np.recarray
        Localizations of a calibration sample.
    info : list of dicts
        Information about the calibration sample, including the number
        of frames.
    d : float
        Step size in nm, i.e., the distance between the z positions of
        the calibration sample.
    magnification_factor : float
        Magnification factor of the microscope, i.e., the ratio between
        the actual z position of the calibration sample and the
        estimated z position from the localization data.
    path : str, optional
        Path to save the calibration data as a YAML file. If None, the
        calibration data will not be saved. Default is None.

    Returns
    -------
    calibration : dict
        Dictionary containing the calibration coefficients (i.e.,
        polynomial coefficients), number of frames, step size, and
        magnification factor.
    """
    n_frames = info[0]["Frames"]
    range = (n_frames - 1) * d
    frame_range = np.arange(n_frames)
    z_range = -(frame_range * d - range / 2)  # negative so that the
    # first frames of a bottom-to-up scan are positive z coordinates.

    mean_sx = np.array(
        [np.mean(locs.sx[locs.frame == _]) for _ in frame_range]
    )
    mean_sy = np.array(
        [np.mean(locs.sy[locs.frame == _]) for _ in frame_range]
    )
    var_sx = np.array(
        [np.var(locs.sx[locs.frame == _]) for _ in frame_range]
    )
    var_sy = np.array(
        [np.var(locs.sy[locs.frame == _]) for _ in frame_range]
    )

    keep_x = (locs.sx - mean_sx[locs.frame]) ** 2 < var_sx[locs.frame]
    keep_y = (locs.sy - mean_sy[locs.frame]) ** 2 < var_sy[locs.frame]
    keep = keep_x & keep_y
    locs = locs[keep]

    # Fits calibration curve to the mean of each frame
    mean_sx = np.array(
        [np.mean(locs.sx[locs.frame == _]) for _ in frame_range]
    )
    mean_sy = np.array(
        [np.mean(locs.sy[locs.frame == _]) for _ in frame_range]
    )

    # Fix nan
    mean_sx = interpolate_nan(mean_sx)
    mean_sy = interpolate_nan(mean_sy)

    cx = np.polyfit(z_range, mean_sx, 6, full=False)
    cy = np.polyfit(z_range, mean_sy, 6, full=False)

    # Fits calibration curve to each localization
    # true_z = locs.frame * d - range / 2
    # cx = np.polyfit(true_z, locs.sx, 6, full=False)
    # cy = np.polyfit(true_z, locs.sy, 6, full=False)

    calibration = {
        "X Coefficients": [float(_) for _ in cx],
        "Y Coefficients": [float(_) for _ in cy],
        "Number of frames": int(n_frames),
        "Step size in nm": float(d),
        "Magnification factor": float(magnification_factor),
    }
    if path is not None:
        with open(path, "w") as f:
            yaml.dump(calibration, f, default_flow_style=False)

    locs = fit_z(locs, info, calibration, magnification_factor)
    locs.z /= magnification_factor

    plt.figure(figsize=(18, 10))

    plt.subplot(231)
    # Plot this if calibration curve is fitted to each localization
    # plt.plot(true_z, locs.sx, '.', label='x', alpha=0.2)
    # plt.plot(true_z, locs.sy, '.', label='y', alpha=0.2)
    # plt.plot(true_z, np.polyval(cx, true_z), '0.3', lw=1.5, label='x fit')
    # plt.plot(true_z, np.polyval(cy, true_z), '0.3', lw=1.5, label='y fit')
    plt.plot(z_range, mean_sx, ".-", label="x")
    plt.plot(z_range, mean_sy, ".-", label="y")
    plt.plot(z_range, np.polyval(cx, z_range), "0.3", lw=1.5, label="x fit")
    plt.plot(z_range, np.polyval(cy, z_range), "0.3", lw=1.5, label="y fit")
    plt.xlabel("Stage position")
    plt.ylabel("Mean spot width/height")
    plt.xlim(z_range.min(), z_range.max())
    plt.legend(loc="best")

    ax = plt.subplot(232)
    plt.scatter(locs.sx, locs.sy, c="k", lw=0, alpha=0.1)
    plt.plot(
        np.polyval(cx, z_range),
        np.polyval(cy, z_range),
        lw=1.5,
        label="calibration from fit of mean width/height",
    )
    plt.plot()
    ax.set_aspect("equal")
    plt.xlabel("Spot width")
    plt.ylabel("Spot height")
    plt.legend(loc="best")

    plt.subplot(233)
    plt.plot(locs.z, locs.sx, ".", label="x", alpha=0.2)
    plt.plot(locs.z, locs.sy, ".", label="y", alpha=0.2)
    plt.plot(
        z_range, np.polyval(cx, z_range), "0.3", lw=1.5, label="calibration",
    )
    plt.plot(z_range, np.polyval(cy, z_range), "0.3", lw=1.5)
    plt.xlim(z_range.min(), z_range.max())
    plt.xlabel("Estimated z")
    plt.ylabel("Spot width/height")
    plt.legend(loc="best")

    ax = plt.subplot(234)
    plt.plot(z_range[locs.frame], locs.z, ".k", alpha=0.1)
    plt.plot(
        [z_range.min(), z_range.max()],
        [z_range.min(), z_range.max()],
        lw=1.5,
        label="identity",
    )
    plt.xlim(z_range.min(), z_range.max())
    plt.ylim(z_range.min(), z_range.max())
    ax.set_aspect("equal")
    plt.xlabel("Stage position")
    plt.ylabel("Estimated z")
    plt.legend(loc="best")

    ax = plt.subplot(235)
    deviation = locs.z - z_range[locs.frame]
    bins = lib.calculate_optimal_bins(deviation, max_n_bins=1000)
    plt.hist(deviation, bins)
    plt.xlabel("Deviation to true position")
    plt.ylabel("Occurence")

    ax = plt.subplot(236)
    square_deviation = deviation**2
    mean_square_deviation_frame = [
        np.mean(square_deviation[locs.frame == _]) for _ in frame_range
    ]
    rmsd_frame = np.sqrt(mean_square_deviation_frame)
    plt.plot(z_range, rmsd_frame, ".-", color="0.3")
    plt.xlim(z_range.min(), z_range.max())
    plt.gca().set_ylim(bottom=0)
    plt.xlabel("Stage position")
    plt.ylabel("Mean z precision")

    plt.tight_layout(pad=2)

    if path is not None:
        dirname = path[0:-5]
        plt.savefig(dirname + ".png", format="png", dpi=300)

    plt.show()

    export = False
    # Export
    if export:
        print("Exporting...")
        np.savetxt("mean_sx.txt", mean_sx, delimiter="/t")
        np.savetxt("mean_sy.txt", mean_sy, delimiter="/t")
        np.savetxt("locs_sx.txt", locs.sx, delimiter="/t")
        np.savetxt("locs_sy.txt", locs.sy, delimiter="/t")
        np.savetxt("cx.txt", cx, delimiter="/t")
        np.savetxt("cy.txt", cy, delimiter="/t")
        np.savetxt("z_range.txt", z_range, delimiter="/t")
        np.savetxt("locs_z.txt", locs.z, delimiter="/t")
        np.savetxt(
            "z_range_locs_frame.txt", z_range[locs.frame], delimiter="/t",
        )
        np.savetxt("rmsd_frame.txt", rmsd_frame, delimiter="/t")

    # np.savetxt('test.out', x, delimiter=',')   # X is an array
    # np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    # np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    return calibration


@numba.jit(nopython=True, nogil=True)
def _fit_z_target(
    z: np.array,
    sx: np.array,
    sy: np.array,
    cx: np.array,
    cy: np.array,
) -> np.array:
    """Target function that's to be minimized for fitting the z
    coordinates given the single-emitter image width and height as well
    as the calibration curve coefficients. It calculates the difference
    between the square root of the spot width/height and the polynomial
    fit of the z-axis calibration curve."""
    z2 = z * z
    z3 = z * z2
    z4 = z * z3
    z5 = z * z4
    z6 = z * z5
    wx = (
        cx[0] * z6
        + cx[1] * z5
        + cx[2] * z4
        + cx[3] * z3
        + cx[4] * z2
        + cx[5] * z
        + cx[6]
    )
    wy = (
        cy[0] * z6
        + cy[1] * z5
        + cy[2] * z4
        + cy[3] * z3
        + cy[4] * z2
        + cy[5] * z
        + cy[6]
    )
    return (sx**0.5 - wx**0.5) ** 2 + (
        sy**0.5 - wy**0.5
    ) ** 2  # Apparently this results in slightly more accurate z coordinates
    # (Huang et al. '08)


def fit_z(
    locs: np.recarray,
    info: list[dict],
    calibration: dict,
    magnification_factor: float,
    filter: int = 2
) -> np.recarray:
    """Fit z coordinates to the localizations based on the calibration
    curve coefficients and the single-emitter image width and height.

    Parameters
    ----------
    locs : np.recarray
        Localizations to fit the z-axis calibration curve to.
    info : list of dicts
        Information about the localizations, including the number of
        frames.
    calibration : dict
        Calibration data containing the polynomial coefficients for
        the x and y axes, number of frames, step size, and magnification
        factor.
    magnification_factor : float
        Magnification factor of the microscope, i.e., the ratio between
        the actual z position of the calibration sample and the
        estimated z position from the localization data.
    filter : int, optional
        Filter for the z fits. If set to 0, no filtering is applied.
        If set to 2, the z fits are filtered based on the root mean
        square deviation (RMSD) of the z calibration. Default is 2.

    Returns
    -------
    locs : np.recarray
        Localizations with the fitted z coordinates and their residuals
        (d_zcalib).
    """
    cx = np.array(calibration["X Coefficients"])
    cy = np.array(calibration["Y Coefficients"])
    z = np.zeros_like(locs.x)
    square_d_zcalib = np.zeros_like(z)
    sx = locs.sx
    sy = locs.sy
    for i in range(len(z)):
        # set bounds to avoid potential gaps in the calibration curve,
        # credits to Loek Andriessen
        result = minimize_scalar(
            _fit_z_target,
            bounds=[-1000, 1000],
            args=(sx[i], sy[i], cx, cy)
        )
        z[i] = result.x
        square_d_zcalib[i] = result.fun
    z *= magnification_factor
    locs = lib.append_to_rec(locs, z, "z")
    locs = lib.append_to_rec(locs, np.sqrt(square_d_zcalib), "d_zcalib")
    locs = lib.ensure_sanity(locs, info)
    return filter_z_fits(locs, filter)


def fit_z_parallel(
    locs: np.recarray,
    info: list[dict],
    calibration: dict,
    magnification_factor: float,
    filter: int = 2,
    asynch: bool = False,
) -> np.recarray | list[futures.Future]:
    """Fit z coordinates to the localizations based on the calibration
    curve coefficients and the single-emitter image width and height,
    optionally using multiprocessing.

    Parameters
    ----------
    locs : np.recarray
        Localizations to fit the z-axis calibration curve to.
    info : list of dicts
        Information about the localizations, including the number of
        frames.
    calibration : dict
        Calibration data containing the polynomial coefficients for
        the x and y axes, number of frames, step size, and magnification
        factor.
    magnification_factor : float
        Magnification factor of the microscope, i.e., the ratio between
        the actual z position of the calibration sample and the
        estimated z position from the localization data.
    filter : int, optional
        Filter for the z fits. If set to 0, no filtering is applied.
        If set to 2, the z fits are filtered based on the root mean
        square deviation (RMSD) of the z calibration. Default is 2.
    asynch : bool, optional
        If True, use multiprocessing. Then, a list of futures that can
        be used to retrieve the results asynchronously is returned. If
        False, the function waits for all tasks to complete and returns
        the combined results. Default is False.

    Returns
    -------
    locs : np.recarray or list of futures.Future
        If `asynch` is False, returns a recarray of localizations with
        the fitted z coordinates and their residuals (d_zcalib).
        If `asynch` is True, returns a list of futures that can be
        used to retrieve the results asynchronously.
    """
    n_workers = min(
        60, max(1, int(0.75 * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores
    n_locs = len(locs)
    n_tasks = 100 * n_workers
    spots_per_task = [
        int(n_locs / n_tasks + 1)
        if _ < n_locs % n_tasks
        else int(n_locs / n_tasks)
        for _ in range(n_tasks)
    ]
    start_indices = np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = ProcessPoolExecutor(n_workers)
    for i, n_locs_task in zip(start_indices, spots_per_task):
        fs.append(
            executor.submit(
                fit_z,
                locs[i:i + n_locs_task],
                info,
                calibration,
                magnification_factor,
                filter=0,
            )
        )
    if asynch:
        return fs
    with tqdm(total=n_tasks, unit="task") as progress_bar:
        for f in futures.as_completed(fs):
            progress_bar.update()
    return locs_from_futures(fs, filter=filter)


def locs_from_futures(
    futures: list[futures.Future],
    filter: int = 2
) -> np.recarray:
    """Combine the results from a list of futures (i.e.,
    multiprocessing results) into a single recarray of localizations
    with fitted z coordinates and their residuals (d_zcalib).

    Parameters
    ----------
    futures : list of futures.Future
        List of futures that contain the results of the z fits.
    filter : int, optional
        Filter for the z fits. If set to 0, no filtering is applied.
        If set to 2, the z fits are filtered based on the root mean
        square deviation (RMSD) of the z calibration. Default is 2.

    Returns
    -------
    locs : np.recarray
        Recarray of localizations with the fitted z coordinates and
        their residuals (d_zcalib).
    """
    locs = [_.result() for _ in futures]
    locs = np.hstack(locs).view(np.recarray)
    return filter_z_fits(locs, filter)


def filter_z_fits(locs: np.recarray, range: int) -> np.recarray:
    """Filter the z fits based on the root mean square deviation (RMSD)
    of the z calibration (d_zcalib residual). If `range` is set to 0, no
    filtering is applied. If `range` is greater than 0, the
    localizations with a RMSD greater than `range` are removed.

    Parameters
    ----------
    locs : np.recarray
        Localizations with fitted z coordinates and their residuals
        (d_zcalib).
    range : int
        Range for filtering the z fits. If set to 0, no filtering is
        applied. If set to a positive value, localizations with a
        RMSD greater than `range` times the RMSD of the z calibration
        are removed.

    Returns
    -------
    locs : np.recarray
        Recarray of localizations with the fitted z coordinates and
        their residuals (d_zcalib) after filtering.
    """
    if range > 0:
        rmsd = np.sqrt(np.nanmean(locs.d_zcalib**2))
        locs = locs[locs.d_zcalib <= range * rmsd]
    return locs
