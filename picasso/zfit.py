import numpy as _np
import numba as _numba
import multiprocessing as _multiprocessing
import concurrent.futures as _futures
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from scipy.optimize import minimize_scalar as _minimize_scalar
from tqdm import tqdm as _tqdm
import yaml as _yaml
import matplotlib.pyplot as _plt
from . import lib as _lib


_plt.style.use("ggplot")


def nan_index(y):
    return _np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nan(data):
    nans, x = nan_index(data)
    data[nans] = _np.interp(x(nans), x(~nans), data[~nans])
    return data


def calibrate_z(locs, info, d, magnification_factor, path=None):
    n_frames = info[0]["Frames"]
    range = (n_frames - 1) * d
    frame_range = _np.arange(n_frames)
    z_range = -(frame_range * d - range / 2)  # negative so that the first frames of
    # a bottom-to-up scan are positive z coordinates.

    mean_sx = _np.array([_np.mean(locs.sx[locs.frame == _]) for _ in frame_range])
    mean_sy = _np.array([_np.mean(locs.sy[locs.frame == _]) for _ in frame_range])
    var_sx = _np.array([_np.var(locs.sx[locs.frame == _]) for _ in frame_range])
    var_sy = _np.array([_np.var(locs.sy[locs.frame == _]) for _ in frame_range])

    keep_x = (locs.sx - mean_sx[locs.frame]) ** 2 < var_sx[locs.frame]
    keep_y = (locs.sy - mean_sy[locs.frame]) ** 2 < var_sy[locs.frame]
    keep = keep_x & keep_y
    locs = locs[keep]

    # Fits calibration curve to the mean of each frame
    mean_sx = _np.array([_np.mean(locs.sx[locs.frame == _]) for _ in frame_range])
    mean_sy = _np.array([_np.mean(locs.sy[locs.frame == _]) for _ in frame_range])

    # Fix nan
    mean_sx = interpolate_nan(mean_sx)
    mean_sy = interpolate_nan(mean_sy)

    cx = _np.polyfit(z_range, mean_sx, 6, full=False)
    cy = _np.polyfit(z_range, mean_sy, 6, full=False)

    # Fits calibration curve to each localization
    # true_z = locs.frame * d - range / 2
    # cx = _np.polyfit(true_z, locs.sx, 6, full=False)
    # cy = _np.polyfit(true_z, locs.sy, 6, full=False)

    calibration = {
        "X Coefficients": [float(_) for _ in cx],
        "Y Coefficients": [float(_) for _ in cy],
    }
    if path is not None:
        with open(path, "w") as f:
            _yaml.dump(calibration, f, default_flow_style=False)

    locs = fit_z(locs, info, calibration, magnification_factor)
    locs.z /= magnification_factor

    _plt.figure(figsize=(18, 10))

    _plt.subplot(231)
    # Plot this if calibration curve is fitted to each localization
    # _plt.plot(true_z, locs.sx, '.', label='x', alpha=0.2)
    # _plt.plot(true_z, locs.sy, '.', label='y', alpha=0.2)
    # _plt.plot(true_z, _np.polyval(cx, true_z), '0.3', lw=1.5, label='x fit')
    # _plt.plot(true_z, _np.polyval(cy, true_z), '0.3', lw=1.5, label='y fit')
    _plt.plot(z_range, mean_sx, ".-", label="x")
    _plt.plot(z_range, mean_sy, ".-", label="y")
    _plt.plot(z_range, _np.polyval(cx, z_range), "0.3", lw=1.5, label="x fit")
    _plt.plot(z_range, _np.polyval(cy, z_range), "0.3", lw=1.5, label="y fit")
    _plt.xlabel("Stage position")
    _plt.ylabel("Mean spot width/height")
    _plt.xlim(z_range.min(), z_range.max())
    _plt.legend(loc="best")

    ax = _plt.subplot(232)
    _plt.scatter(locs.sx, locs.sy, c="k", lw=0, alpha=0.1)
    _plt.plot(
        _np.polyval(cx, z_range),
        _np.polyval(cy, z_range),
        lw=1.5,
        label="calibration from fit of mean width/height",
    )
    _plt.plot()
    ax.set_aspect("equal")
    _plt.xlabel("Spot width")
    _plt.ylabel("Spot height")
    _plt.legend(loc="best")

    _plt.subplot(233)
    _plt.plot(locs.z, locs.sx, ".", label="x", alpha=0.2)
    _plt.plot(locs.z, locs.sy, ".", label="y", alpha=0.2)
    _plt.plot(z_range, _np.polyval(cx, z_range), "0.3", lw=1.5, label="calibration")
    _plt.plot(z_range, _np.polyval(cy, z_range), "0.3", lw=1.5)
    _plt.xlim(z_range.min(), z_range.max())
    _plt.xlabel("Estimated z")
    _plt.ylabel("Spot width/height")
    _plt.legend(loc="best")

    ax = _plt.subplot(234)
    _plt.plot(z_range[locs.frame], locs.z, ".k", alpha=0.1)
    _plt.plot(
        [z_range.min(), z_range.max()],
        [z_range.min(), z_range.max()],
        lw=1.5,
        label="identity",
    )
    _plt.xlim(z_range.min(), z_range.max())
    _plt.ylim(z_range.min(), z_range.max())
    ax.set_aspect("equal")
    _plt.xlabel("Stage position")
    _plt.ylabel("Estimated z")
    _plt.legend(loc="best")

    ax = _plt.subplot(235)
    deviation = locs.z - z_range[locs.frame]
    bins = _lib.calculate_optimal_bins(deviation, max_n_bins=1000)
    _plt.hist(deviation, bins)
    _plt.xlabel("Deviation to true position")
    _plt.ylabel("Occurence")

    ax = _plt.subplot(236)
    square_deviation = deviation**2
    mean_square_deviation_frame = [
        _np.mean(square_deviation[locs.frame == _]) for _ in frame_range
    ]
    rmsd_frame = _np.sqrt(mean_square_deviation_frame)
    _plt.plot(z_range, rmsd_frame, ".-", color="0.3")
    _plt.xlim(z_range.min(), z_range.max())
    _plt.gca().set_ylim(bottom=0)
    _plt.xlabel("Stage position")
    _plt.ylabel("Mean z precision")

    _plt.tight_layout(pad=2)

    if path is not None:
        dirname = path[0:-5]
        _plt.savefig(dirname + ".png", format="png", dpi=300)

    _plt.show()

    export = False
    # Export
    if export:
        print("Exporting...")
        _np.savetxt("mean_sx.txt", mean_sx, delimiter="/t")
        _np.savetxt("mean_sy.txt", mean_sy, delimiter="/t")
        _np.savetxt("locs_sx.txt", locs.sx, delimiter="/t")
        _np.savetxt("locs_sy.txt", locs.sy, delimiter="/t")
        _np.savetxt("cx.txt", cx, delimiter="/t")
        _np.savetxt("cy.txt", cy, delimiter="/t")
        _np.savetxt("z_range.txt", z_range, delimiter="/t")
        _np.savetxt("locs_z.txt", locs.z, delimiter="/t")
        _np.savetxt("z_range_locs_frame.txt", z_range[locs.frame], delimiter="/t")
        _np.savetxt("rmsd_frame.txt", rmsd_frame, delimiter="/t")

    # np.savetxt('test.out', x, delimiter=',')   # X is an array
    # np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    # np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation

    return calibration


@_numba.jit(nopython=True, nogil=True)
def _fit_z_target(z, sx, sy, cx, cy):
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
    locs, 
    info, 
    calibration, 
    magnification_factor, 
    filter=2
):
    cx = _np.array(calibration["X Coefficients"])
    cy = _np.array(calibration["Y Coefficients"])
    z = _np.zeros_like(locs.x)
    square_d_zcalib = _np.zeros_like(z)
    sx = locs.sx
    sy = locs.sy
    for i in range(len(z)):
        result = _minimize_scalar(_fit_z_target, args=(sx[i], sy[i], cx, cy))
        z[i] = result.x
        square_d_zcalib[i] = result.fun
    z *= magnification_factor
    locs = _lib.append_to_rec(locs, z, "z")
    locs = _lib.append_to_rec(locs, _np.sqrt(square_d_zcalib), "d_zcalib")
    locs = _lib.ensure_sanity(locs, info)
    return filter_z_fits(locs, filter)


def fit_z_parallel(
    locs, 
    info, 
    calibration, 
    magnification_factor, 
    filter=2, 
    asynch=False,
):
    n_workers = min(
        60, max(1, int(0.75 * _multiprocessing.cpu_count()))
    ) # Python crashes when using >64 cores
    n_locs = len(locs)
    n_tasks = 100 * n_workers
    spots_per_task = [
        int(n_locs / n_tasks + 1) if _ < n_locs % n_tasks else int(n_locs / n_tasks)
        for _ in range(n_tasks)
    ]
    start_indices = _np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = _ProcessPoolExecutor(n_workers)
    for i, n_locs_task in zip(start_indices, spots_per_task):
        fs.append(
            executor.submit(
                fit_z,
                locs[i : i + n_locs_task],
                info,
                calibration,
                magnification_factor,
                filter=0,
            )
        )
    if asynch:
        return fs
    with _tqdm(total=n_tasks, unit="task") as progress_bar:
        for f in _futures.as_completed(fs):
            progress_bar.update()
    return locs_from_futures(fs, filter=filter)


def locs_from_futures(futures, filter=2):
    locs = [_.result() for _ in futures]
    locs = _np.hstack(locs).view(_np.recarray)
    return filter_z_fits(locs, filter)


def filter_z_fits(locs, range):
    if range > 0:
        rmsd = _np.sqrt(_np.nanmean(locs.d_zcalib**2))
        locs = locs[locs.d_zcalib <= range * rmsd]
    return locs
