import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .. import lib, __version__

try:
    from numba import cuda as _cuda

    _CUDA_AVAILABLE = _cuda.is_available()
except Exception:
    _cuda = None
    _CUDA_AVAILABLE = False


def comet(
    locs: pd.DataFrame,
    info: list[dict],
    locs_per_segment: int,
    max_drift_nm: float,
    max_locs_per_segment: int | None = None,
    *,
    initial_sigma_nm: float | None = None,
    target_sigma_nm: float = 10.0,
    boxcar_width: int = 3,
    drift_max_bound_factor: float = 2.0,
    interpolation_method: str = "cubic",
    mode: str = "cuda",
    display: bool = False,
    progress=None,  # currently unused
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Apply COMET undrifting to the localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Localization table. Must contain columns 'x', 'y', 'frame'.
        Optional: 'z'.
        x/y are expected in camera pixels, z in the native z-unit used by Picasso.
    info : list of dict
        Localization metadata.
    locs_per_segment : int
        Target number of localizations per segment for COMET segmentation.
    max_drift_nm : float
        Maximum expected drift in nm.
    max_locs_per_segment : int or None, optional
        Optional cap for localizations per segment. If negative, treated as None.
    initial_sigma_nm : float or None, optional
        Initial Gaussian sigma for COMET. If None, defaults to max_drift_nm / 3.
    target_sigma_nm : float, optional
        Final Gaussian sigma for refinement.
    boxcar_width : int, optional
        Smoothing width over segments.
    drift_max_bound_factor : float, optional
        Bound factor for optimizer.
    interpolation_method : str, optional
        Drift interpolation method.
    mode : str, optional
        Backend mode. Currently only ``"cuda"`` is supported, which requires
        a CUDA-capable NVIDIA GPU and numba installed with CUDA support.
        Raises ``RuntimeError`` when CUDA is not available.
    display : bool, optional
        Whether to show diagnostic plots.
    progress : optional
        Placeholder for API compatibility with AIM.

    Returns
    -------
    locs : pd.DataFrame
        Undrift-corrected localizations.
    new_info : list[dict]
        Updated metadata.
    drift : pd.DataFrame
        Per-frame drift table with columns x, y, and optionally z.
        Drift is returned in the same units as locs:
        x/y in pixels, z in the original z-unit.
    """
    # Accept numpy structured arrays as well as pandas DataFrames.
    if isinstance(locs, np.ndarray) and locs.dtype.names is not None:
        locs = pd.DataFrame({name: locs[name] for name in locs.dtype.names})

    locs = locs.copy()

    if not _CUDA_AVAILABLE:
        raise RuntimeError(
            "COMET requires a CUDA-capable NVIDIA GPU and numba with CUDA support, "
            "which are not available on this system. "
            "Please use a different drift-correction method (e.g. AIM or RCC)."
        )

    if max_locs_per_segment is not None and max_locs_per_segment < 0:
        max_locs_per_segment = None

    pixelsize_nm = lib.get_from_metadata(info, "Pixelsize", raise_error=True)

    required_cols = {"x", "y", "frame"}
    missing = required_cols - set(locs.columns)
    if missing:
        raise KeyError(
            f"Missing required columns for COMET: {sorted(missing)}"
        )

    has_z = "z" in locs.columns

    # Normalize frames so they start at 0 for internal indexing.
    # This is important because COMET later indexes drift by frame number.
    frame = locs["frame"].to_numpy(dtype=np.int64)
    frame0 = frame - frame.min()

    # Build COMET input array in nm
    dataset = np.zeros((len(locs), 4), dtype=np.float64)
    dataset[:, 0] = locs["x"].to_numpy(dtype=np.float64) * pixelsize_nm
    dataset[:, 1] = locs["y"].to_numpy(dtype=np.float64) * pixelsize_nm
    dataset[:, 2] = locs["z"].to_numpy(dtype=np.float64) if has_z else 0.0
    dataset[:, 3] = frame0

    if initial_sigma_nm is None:
        initial_sigma_nm = max_drift_nm / 3

    drift_nm_with_frame = comet_run_kd(
        dataset=dataset,
        segmentation_mode=1,  # fixed locs per segment
        segmentation_var=locs_per_segment,
        max_locs_per_segment=max_locs_per_segment,
        initial_sigma_nm=initial_sigma_nm,
        gt_drift=None,
        display=display,
        return_corrected_locs=False,
        max_drift_nm=max_drift_nm,
        target_sigma_nm=target_sigma_nm,
        boxcar_width=boxcar_width,
        drift_max_bound_factor=drift_max_bound_factor,
        interpolation_method=interpolation_method,
        mode=mode,
        min_max_frames=(int(frame0.min()), int(frame0.max())),
    )

    # drift_nm_with_frame columns: dx_nm, dy_nm, dz_nm, frame0
    drift_nm = drift_nm_with_frame[:, :3]

    # Apply drift back to dataframe in original units
    frame_idx = frame0.astype(np.int64)
    locs["x"] = (
        locs["x"].to_numpy(dtype=np.float64)
        - drift_nm[frame_idx, 0] / pixelsize_nm
    )
    locs["y"] = (
        locs["y"].to_numpy(dtype=np.float64)
        - drift_nm[frame_idx, 1] / pixelsize_nm
    )
    if has_z:
        locs["z"] = (
            locs["z"].to_numpy(dtype=np.float64) - drift_nm[frame_idx, 2]
        )

    # Build drift dataframe in Picasso-style units
    drift_dict = {
        "x": (drift_nm[:, 0] / pixelsize_nm).astype("float32"),
        "y": (drift_nm[:, 1] / pixelsize_nm).astype("float32"),
    }
    if has_z:
        drift_dict["z"] = drift_nm[:, 2].astype("float32")

    drift = pd.DataFrame(drift_dict)

    new_info_entry = {
        "Generated by": f"Picasso v{__version__} COMET",
        "Segmentation mode": "localizations per segment",
        "Localizations per segment": locs_per_segment,
        "Maximum drift (nm)": max_drift_nm,
        "Initial sigma (nm)": float(initial_sigma_nm),
        "Target sigma (nm)": float(target_sigma_nm),
        "Boxcar width": int(boxcar_width),
        "Max localizations per segment": (
            None if max_locs_per_segment is None else int(max_locs_per_segment)
        ),
        "Interpolation method": interpolation_method,
        "Backend": mode,
    }
    new_info = info + [new_info_entry]

    return locs, new_info, drift


def comet_run_kd(
    dataset,
    segmentation_mode,
    segmentation_var,
    max_locs_per_segment=None,
    initial_sigma_nm=None,
    gt_drift=None,
    display=False,
    return_corrected_locs=False,
    max_drift_nm=300,
    target_sigma_nm=1,
    boxcar_width=1,
    drift_max_bound_factor=2,
    save_intermediate_results=False,
    interpolation_method="cubic",
    mode="cuda",
    min_max_frames=None,
    pair_indices_safety_check=False,
    mapped_memory_threshold_bytes=None,
    device_mem_fraction=0.5,
):
    """
    Run COMET drift correction end-to-end.

    Pipeline: temporal segmentation -> KD-tree neighbor pairs -> cost optimization (L-BFGS-B)
    -> optional temporal smoothing -> spline interpolation to per-frame drift -> (optional) subtract drift.

    Parameters
    ----------
    dataset : ndarray of shape (N, 4)
        Localization array with columns [x_nm, y_nm, z_nm, frame]. Units in nm; frame is int.
        For 2D CSVs, insert a zero z column to get (N, 4).
    segmentation_mode : {0, 1, 2}
        Temporal segmentation mode:
        0 = number of windows (choose S directly),
        1 = localizations per window (accumulate frames until >= X locs),
        2 = fixed frame window size (default).
    segmentation_var : int
        Mode-dependent value (S, locs per window, or frames per window).
    initial_sigma_nm : float, default=100
        Initial Gaussian length scale for the overlap kernel (coarse scale).
    target_sigma_nm : float, default=1
        Target (final) Gaussian length scale for fine refinement.
    max_drift_nm : float or None, default=None
        Pair radius in nm used for neighbor search. If None, uses 3 * initial_sigma_nm.
    drift_max_bound_factor : float, default=1.0
        Multiplicative factor for L-BFGS-B box bounds around +-max_drift.
    boxcar_width : int, default=1
        Temporal smoothing width (segments) applied to the estimated drift between optimizer steps.
    interpolation_method : {"cubic", "catmull-rom"}, default="cubic"
        Spline used to convert per-segment drift to per-frame drift.
    max_locs_per_segment : int or None, default=None
        Optional downsampling cap per segment (to control memory/time).
    mode : str, "cuda"
        only Cuda in this version
    return_corrected_locs : bool, default=False
        If True, also return drift-corrected localizations.
    mapped_memory_threshold_bytes : int or None, default=None
        Byte budget for the pair-index arrays on the GPU. If exceeded, the
        arrays are staged in mapped host memory instead of VRAM. None
        auto-detects the budget from free VRAM (see `device_mem_fraction`).
        Set explicitly to force deterministic behavior across machines/GPUs.
    device_mem_fraction : float, default=0.5
        Fraction of free VRAM used as the index-array budget when
        `mapped_memory_threshold_bytes` is None. Lower on small GPUs to avoid
        out-of-memory; raise toward 1.0 to keep more in faster device memory.

    Returns
    -------
    drift_interp_with_frames : ndarray of shape (F, 4)
        Per-frame drift with columns [dx_nm, dy_nm, dz_nm, frame].
    corrected_locs : ndarray of shape (N, 4), optional
        Only if return_corrected_locs=True. Columns are [x_nm, y_nm, z_nm, segment_id].
    """

    loc_frames = dataset[:, -1]
    if min_max_frames is None:
        min_max_frames = (loc_frames.min(), loc_frames.max())

    # Segment the dataset based on frame numbers into time windows

    result, sorted_dataset, idx_i, idx_j = (
        segmentation_and_pair_indices_wrapper(
            dataset,
            segmentation_var,
            segmentation_mode,
            max_drift_nm,
            max_locs_per_segment,
            pair_indices_safety_check=pair_indices_safety_check,
        )
    )

    # Set default initial sigma if not provided
    if initial_sigma_nm is None:
        initial_sigma_nm = max_drift_nm // 3

    # Run drift optimization
    t0 = time.time()
    drift_est = optimize_3d_chunked_better_moving_avg_kd(
        result.n_segments,
        sorted_dataset,
        idx_i,
        idx_j,
        sigma_nm=initial_sigma_nm,
        target_sigma_nm=target_sigma_nm,
        drift_max_nm=max_drift_nm,
        drift_max_bound_factor=drift_max_bound_factor,
        display_steps=display,
        boxcar_width=boxcar_width,
        segmentation_result=result,
        mode=mode,
        mapped_memory_threshold_bytes=mapped_memory_threshold_bytes,
        device_mem_fraction=device_mem_fraction,
    )
    elapsed = time.time() - t0

    # Reshape and interpolate drift across all frames
    drift_est = drift_est.reshape((result.n_segments, 3))
    vld_tp = np.where(~np.isnan(drift_est[:, 0]))

    frame_interp = np.arange(0, min_max_frames[1] + 1, dtype=int)
    drift_interp = interpolate_drift(
        result.center_frames[vld_tp],
        drift_est[vld_tp],
        frame_interp,
        method=interpolation_method,
    )
    drift_interp_with_frames = np.hstack(
        (drift_interp, frame_interp[:, np.newaxis])
    )

    # Apply drift correction to original localizations
    for i in range(3):
        dataset[:, i] = (
            dataset[:, i] - drift_interp[dataset[:, -1].astype(int), i]
        )

    # Optionally show estimated drift curve
    if display:
        print(f"Drift estimation completed in {elapsed:.2f} seconds.")
        plt.figure()
        plt.plot(frame_interp, drift_interp)
        plt.title("Estimated Drift")
        plt.xlabel("Frames")
        plt.ylabel("Drift (nm)")
        plt.legend(["X", "Y", "Z"])
        plt.show()

    # Optional GT comparison plot
    if display and gt_drift is not None:
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(gt_drift[:, 3], gt_drift[:, 0], label="GT Drift X")
        ax[1].plot(gt_drift[:, 3], gt_drift[:, 1], label="GT Drift Y")
        ax[2].plot(gt_drift[:, 3], gt_drift[:, 2], label="GT Drift Z")
        ax[0].plot(
            frame_interp,
            drift_interp[:, 0],
            label="Estimated Drift X",
            linestyle="--",
        )
        ax[1].plot(
            frame_interp,
            drift_interp[:, 1],
            label="Estimated Drift Y",
            linestyle="--",
        )
        ax[2].plot(
            frame_interp,
            drift_interp[:, 2],
            label="Estimated Drift Z",
            linestyle="--",
        )
        ax[1].set_title("Ground Truth vs Estimated Drift")
        ax[1].set_xlabel("Frames")
        ax[0].set_ylabel("Drift (nm)")
        plt.legend()
        plt.show()

    # Return corrected locs + drift
    if return_corrected_locs:
        return drift_interp_with_frames, dataset
    else:
        return drift_interp_with_frames


def optimize_3d_chunked_better_moving_avg_kd(
    n_segments,
    locs_nm,
    idx_i,
    idx_j,
    sigma_nm=30,
    drift_max_nm=300,
    target_sigma_nm=30,
    display_steps=False,
    boxcar_width=3,
    drift_max_bound_factor=2,
    segmentation_result=None,
    mode="cuda",
    return_calc_time=False,
    mapped_memory_threshold_bytes=None,
    device_mem_fraction=0.5,
):
    """
    Estimate per-segment drift (mu) by minimizing the negative Gaussian-overlap cost
    with an L-BFGS-B optimizer and a coarse-to-fine schedule on sigma.

    The routine operates on temporally segmented localizations and reuses a static
    neighbor graph (pairs within drift_max_nm). Between optimizer steps, a moving
    average (boxcar) can be applied to mu as temporal regularization. Sigma is
    reduced iteratively from `sigma_nm` toward `target_sigma_nm` for robust
    convergence.

    Parameters
    ----------
    n_segments : int
        Number of temporal segments S (0..S-1). One 3D drift vector is estimated per segment.
    locs_nm : ndarray of shape (M, 3)
        localizations in nanometers, columns [x, y, z];
    sigma_nm : float, default=30
        Initial Gaussian width (coarse scale) for the overlap kernel.
    drift_max_nm : float, default=300
        Maximum expected drift (nm). Also used as the radius for neighbor pairs and
        as the L-BFGS-B bound scale (see `drift_max_bound_factor`).
    target_sigma_nm : float, default=30
        Target / final Gaussian width (fine scale). The optimizer reduces sigma toward
        this value over iterations.
    display_steps : bool, default=False
        If True, print or log intermediate progress per iteration/scale.

    boxcar_width : int, default=3
        Temporal smoothing width (in segments) for a moving average applied to mu between steps.
        Use 0 or 1 to disable smoothing.
    drift_max_bound_factor : float, default=2
        Multiplier for L-BFGS-B bounds around +- drift_max_nm to keep updates physically reasonable.
    segmentation_result : object or None, default=None
        Segmentation info and metadata. Expected to provide, at minimum:
        - segment IDs per localization
        - center frames per segment
        - any additional structures required by backend (e.g., pair indices)
        If None, pairs/ids may be built internally depending on implementation.
    mode : str, default=cuda
        If True, use the CPU backend; otherwise try GPU (CUDA) and fall back to CPU if unavailable.
    return_calc_time : bool, default=False
        If True, also return the total computation time in seconds.
    mapped_memory_threshold_bytes : int or None, default=None
        Size budget (in bytes) for the pair-index arrays on the GPU. If the two
        int32 index arrays together exceed this, they are staged in mapped host
        memory instead of device VRAM. If None, the budget is auto-detected from
        currently-free VRAM (see `device_mem_fraction`). Provide an explicit value
        to force deterministic behavior regardless of GPU state.
    device_mem_fraction : float, default=0.5
        Fraction of currently-free VRAM used as the index-array budget when
        `mapped_memory_threshold_bytes` is None. Lower it if you hit out-of-memory
        errors (more aggressive fallback to mapped memory); raise it toward 1.0 to
        keep larger arrays in faster device memory. Ignored when an explicit
        threshold is given.

    Returns
    -------
    mu : ndarray of shape (S, 3)
        Estimated per-segment drift (dx, dy, dz) in nanometers.
    calc_time_s : float, optional
        Only when `return_calc_time=True`. Wall-clock time for the optimization.
    """

    if segmentation_result is None:
        segmentation_result = {}
    intermediate_results_filehandle = None
    sigma_factor = 1.0

    # Extract coordinate + time arrays, convert to device if CUDA
    coords = locs_nm[:, :3].astype(np.float32).copy()
    times = locs_nm[:, 3].astype(np.int32).copy()

    chunk_size = int(1e8)  # 1E7

    quality_control = mode == "torch_qc" or mode == "cuda_qc"

    d_coords = _cuda.to_device(coords)
    d_times = _cuda.to_device(times)
    # The pair-index arrays can dwarf available VRAM on large datasets. Decide
    # whether to stage them in device memory (fast) or fall back to mapped host
    # memory (handles arrays larger than VRAM, slower per access). The threshold
    # auto-adapts to the current GPU unless an explicit override is given.
    idx_bytes = len(idx_i) * 4 * 2  # two int32 arrays (idx_i + idx_j)
    if mapped_memory_threshold_bytes is None:
        # Budget a fraction of currently-free VRAM, leaving headroom for d_val,
        # d_coords, d_times and the optimizer working set.
        free_bytes, _ = _cuda.current_context().get_memory_info()
        threshold = free_bytes * device_mem_fraction
        threshold_src = (
            f"{free_bytes / 1e9:.1f} GB free x {device_mem_fraction:g} (auto)"
        )
    else:
        threshold = float(mapped_memory_threshold_bytes)
        threshold_src = f"{threshold / 1e9:.1f} GB (override)"

    if idx_bytes > threshold:
        # Use mapped memory if index arrays are large relative to the budget.
        print(
            f"Large index arrays ({idx_bytes / 1e9:.1f} GB) exceed device budget "
            f"({threshold_src}) — using mapped memory."
        )
        d_idx_i = _cuda.mapped_array_like(idx_i.astype(np.int32), wc=True)
        d_idx_j = _cuda.mapped_array_like(idx_j.astype(np.int32), wc=True)
        d_idx_i[:] = idx_i
        d_idx_j[:] = idx_j
    else:
        d_idx_i = _cuda.to_device(idx_i.astype(np.int32))
        d_idx_j = _cuda.to_device(idx_j.astype(np.int32))
    # Preallocate device arrays
    d_sigma = np.float64(sigma_nm)
    d_val = _cuda.to_device(np.zeros(chunk_size))
    d_deri = _cuda.to_device(np.zeros((n_segments, 3), dtype=np.float64))
    wrapper = cuda_wrapper_chunked

    # Initial drift estimate + bounds
    drift_est = np.zeros(n_segments * 3)
    bounds = [
        (
            -drift_max_nm * drift_max_bound_factor,
            drift_max_nm * drift_max_bound_factor,
        )
    ] * (3 * n_segments)

    drift_est_gradient = np.inf
    fails = 0
    done = False
    itr_counter = 0
    start_time = time.time()

    while not done:
        # Apply boxcar smoothing to current estimate
        tmp = drift_est.reshape((-1, 3))
        for i in range(3):
            tmp[:, i] = convolve(
                tmp[:, i], np.ones(boxcar_width) / boxcar_width
            )
        drift_est = tmp.flatten()

        # Run L-BFGS-B optimization step
        result = minimize(
            wrapper,
            drift_est,
            method="L-BFGS-B",
            args=(
                d_coords,
                d_times,
                d_idx_i,
                d_idx_j,
                d_sigma,
                sigma_factor,
                d_val,
                d_deri,
                chunk_size,
            ),
            jac=True,
            bounds=bounds,
            options={
                "disp": display_steps,
                "gtol": 1e-5,
                "ftol": 1e3 * np.finfo(float).eps,
                "maxls": 40,
            },
        )
        itr_counter += 1
        print(
            f"Iteration {itr_counter}: status = {result.status}, success = {result.success}"
        )
        print(f"  current sigma: {np.round(sigma_nm * sigma_factor, 2)} nm")

        # Update if successful
        if result.success:
            delta = np.median((result.x - drift_est) ** 2)
            print(f"  drift estimate gradient: {delta}")
            print(f"  previous gradient: {drift_est_gradient}")
            # Check convergence
            if (
                delta > drift_est_gradient or sigma_nm * sigma_factor <= 1.0
            ) and sigma_nm * sigma_factor <= target_sigma_nm:
                done = True
                calc_time = time.time() - start_time
                print(f"Optimization completed in {calc_time:.2f} s")
            else:
                sigma_factor /= 1.5
                drift_est_gradient = delta
                drift_est = result.x
        else:
            fails += 1
            if fails > 2:
                sigma_factor *= 2
                print("Restarting with larger sigma_factor")
            if fails > 5:
                raise RuntimeError(
                    "L-BFGS-B Optimization failed after multiple retries"
                )

    if return_calc_time:
        return drift_est, time.time() - start_time, itr_counter
    else:
        return drift_est


def segmentation_and_pair_indices_wrapper(
    dataset,
    segmentation_var,
    segmentation_mode,
    max_drift_nm,
    max_locs_per_segment,
    pair_indices_safety_check=False,
    hard_limit_pairs=None,
):
    if not segmentation_mode == -1:  # -1 is for pre-segmented data
        result = segmentation_wrapper(
            dataset[:, -1],
            segmentation_var,
            segmentation_mode,
            max_locs_per_segment,
            return_param_dict=True,
        )
    else:
        # pre segmented data, anyway we set these values in case auto downsampling is needed
        segmentation_mode = 2  # dummy --> segment per frame ...
        segmentation_var = 1  # dummy --> ... using 1 frame per segment
    if pair_indices_safety_check:
        n_pairs_est = estimate_pairs(
            dataset[result.loc_valid, :3], max_drift_nm
        )
        print(
            f"Estimated number of pairs within {max_drift_nm} nm: {n_pairs_est:,}"
        )
        if hard_limit_pairs is not None and n_pairs_est > hard_limit_pairs:
            raise RuntimeError(
                f"Estimated number of pairs {n_pairs_est} exceeds hard limit of {hard_limit_pairs}. "
                f"Aborting to avoid crash."
            )
        if n_pairs_est > 5e8:
            print(
                f"Estimated number of pairs is very large {n_pairs_est}. "
                f"Automatic down-sampling is usually required above 500 mil."
                f"Billions of pairs can lead to crash."
            )
            ans = input("Continue anyway? (y/n): ")
            if ans.lower() != "y":
                raise RuntimeError(
                    "Aborted by user due to large estimated number of pairs."
                )
    idx_i, idx_j, successful = pair_indices_kdtree(
        dataset[result.loc_valid, :3], max_drift_nm
    )
    if not successful:
        if max_locs_per_segment is None:
            max_locs_per_segment = int(
                result.out_dict["locs_per_segment"].max()
            )
    while not successful:
        max_locs_per_segment = int(max_locs_per_segment * 0.9)
        print(
            f"Segmentation and Pairing attempt failed, automatic down-sampling active..."
        )
        print(
            f"Retrying segmentation with max_locs_per_segment={max_locs_per_segment}..."
        )
        result = segmentation_wrapper(
            dataset[:, -1],
            segmentation_var,
            segmentation_mode,
            max_locs_per_segment,
            return_param_dict=True,
        )
        sorted_dataset = dataset.copy()
        sorted_dataset[:, -1] = result.loc_segments
        sorted_dataset = sorted_dataset[result.loc_valid]
        idx_i, idx_j, successful = pair_indices_kdtree(
            sorted_dataset[:, :3], max_drift_nm
        )
    print(
        f"Segmentation and Pairing successful resulting in {result.n_segments:,} time windows with on average "
        f"{int(np.median(result.out_dict['locs_per_segment']))} locs per time window. "
        f"{len(idx_i):,} Pairs where found."
    )
    sorted_dataset = dataset.copy()
    sorted_dataset[:, -1] = result.loc_segments
    sorted_dataset = sorted_dataset[result.loc_valid]
    return result, sorted_dataset, idx_i, idx_j


def estimate_pairs(coordinates, distance):
    for i in range(len(coordinates[0])):
        coordinates[:, i] -= np.min(coordinates[:, i])
    coordinates = np.array(np.floor(coordinates / distance), dtype=int)
    coordinates = np.array(list(map(tuple, coordinates)))
    sort_indices = np.lexsort(
        coordinates.T
    )  # get the unique tuples and their counts
    unique_tuples, counts = np.unique(
        coordinates[sort_indices], axis=0, return_counts=True
    )
    # get the indices of the similar tuples
    similar_indices = np.split(sort_indices, np.cumsum(counts[:-1]))
    idx_i = []
    idx_j = []
    pair_idc_estimate = 0
    for i in range(len(similar_indices)):
        n_elements = len(similar_indices[i])
        pair_idc_estimate += n_elements * (n_elements - 1)
    rounded = round(pair_idc_estimate, -4)
    return rounded


def interpolate_drift(center_frames, drift_est, frame_range, method="cubic"):
    """
    Interpolates drift estimates to all frames using specified method.
    Parameters:
    - center_frames: np.ndarray of shape (M,), frames corresponding to drift estimates.
    - drift_est: np.ndarray of shape (M, 3), drift estimates at center frames.
    - frame_range: array-like, frames to interpolate drift estimates to.
    - method: str, interpolation method ('cubic' or 'catmull-rom').
    Returns:
    - drift_interp: np.ndarray of shape (len(frame_range), 3), interpolated drift estimates.
    """
    if method == "cubic":
        return _interpolate_cubic(center_frames, drift_est, frame_range)
    elif method == "catmull-rom":
        return _interpolate_catmull_rom(center_frames, drift_est, frame_range)
    elif method == "linear":
        drift_x = np.interp(frame_range, center_frames, drift_est[:, 0])
        drift_y = np.interp(frame_range, center_frames, drift_est[:, 1])
        drift_z = np.interp(frame_range, center_frames, drift_est[:, 2])
        return np.vstack([drift_x, drift_y, drift_z]).T
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def _interpolate_cubic(center_frames, drift_est, frame_range):
    from scipy.interpolate import CubicSpline

    drift_x = CubicSpline(center_frames, drift_est[:, 0])(frame_range)
    drift_y = CubicSpline(center_frames, drift_est[:, 1])(frame_range)
    drift_z = CubicSpline(center_frames, drift_est[:, 2])(frame_range)
    return np.vstack([drift_x, drift_y, drift_z]).T


def _interpolate_catmull_rom(center_frames, drift_est, frame_range):
    def catmull_rom_1d(x, y, x_interp):
        result = np.zeros_like(x_interp)
        for i in range(1, len(x) - 2):
            x0, x1, x2, x3 = x[i - 1], x[i], x[i + 1], x[i + 2]
            y0, y1, y2, y3 = y[i - 1], y[i], y[i + 1], y[i + 2]

            mask = (x_interp >= x1) & (x_interp <= x2)
            t = (x_interp[mask] - x1) / (x2 - x1)

            result[mask] = 0.5 * (
                (2 * y1)
                + (-y0 + y2) * t
                + (2 * y0 - 5 * y1 + 4 * y2 - y3) * t**2
                + (-y0 + 3 * y1 - 3 * y2 + y3) * t**3
            )
        return result

    x_interp = np.asarray(frame_range)
    x = np.asarray(center_frames)

    if len(x) < 4:
        raise ValueError(
            "Catmull-Rom interpolation requires at least 4 points."
        )

    drift_x = catmull_rom_1d(x, drift_est[:, 0], x_interp)
    drift_y = catmull_rom_1d(x, drift_est[:, 1], x_interp)
    drift_z = catmull_rom_1d(x, drift_est[:, 2], x_interp)

    return np.vstack([drift_x, drift_y, drift_z]).T


# CUDA kernel and wrapper — only defined when CUDA is available at import time.
if _CUDA_AVAILABLE:

    @_cuda.jit
    def cost_function_full_3d_chunked(
        d_locs_time,
        start_idx,
        chunk_size,
        d_idx_i,
        d_idx_j,
        d_sigma,
        d_sigma_factor,
        d_val,
        d_val_sum,
        d_deri,
        d_locs_coords,
        mu,
    ):
        """Compute negative-overlap cost and gradient for 3D localizations (CUDA kernel)."""
        tx = _cuda.threadIdx.x
        ty = _cuda.blockIdx.x
        bw = _cuda.blockDim.x
        pos = tx + ty * bw

        if pos < chunk_size:
            i = d_idx_i[pos + start_idx]
            j = d_idx_j[pos + start_idx]

            ti = d_locs_time[i]
            tj = d_locs_time[j]

            dx = (d_locs_coords[i, 0] - mu[ti, 0]) - (
                d_locs_coords[j, 0] - mu[tj, 0]
            )
            dy = (d_locs_coords[i, 1] - mu[ti, 1]) - (
                d_locs_coords[j, 1] - mu[tj, 1]
            )
            dz = (d_locs_coords[i, 2] - mu[ti, 2]) - (
                d_locs_coords[j, 2] - mu[tj, 2]
            )
            sigma_sq = (2 * d_sigma * d_sigma_factor) ** 2

            diff_sq = dx * dx + dy * dy + dz * dz
            val = (
                1 / (d_sigma * d_sigma_factor) * math.exp(-diff_sq / sigma_sq)
            )
            d_val[pos] = val

            _cuda.atomic.add(
                d_deri,
                (tj, 0),
                2
                * val
                * (
                    d_locs_coords[j, 0]
                    - d_locs_coords[i, 0]
                    + mu[ti, 0]
                    - mu[tj, 0]
                )
                / sigma_sq,
            )
            _cuda.atomic.add(
                d_deri,
                (tj, 1),
                2
                * val
                * (
                    d_locs_coords[j, 1]
                    - d_locs_coords[i, 1]
                    + mu[ti, 1]
                    - mu[tj, 1]
                )
                / sigma_sq,
            )
            _cuda.atomic.add(
                d_deri,
                (tj, 2),
                2
                * val
                * (
                    d_locs_coords[j, 2]
                    - d_locs_coords[i, 2]
                    + mu[ti, 2]
                    - mu[tj, 2]
                )
                / sigma_sq,
            )

            _cuda.atomic.add(
                d_deri,
                (ti, 0),
                2
                * val
                * (
                    d_locs_coords[i, 0]
                    - d_locs_coords[j, 0]
                    + mu[tj, 0]
                    - mu[ti, 0]
                )
                / sigma_sq,
            )
            _cuda.atomic.add(
                d_deri,
                (ti, 1),
                2
                * val
                * (
                    d_locs_coords[i, 1]
                    - d_locs_coords[j, 1]
                    + mu[tj, 1]
                    - mu[ti, 1]
                )
                / sigma_sq,
            )
            _cuda.atomic.add(
                d_deri,
                (ti, 2),
                2
                * val
                * (
                    d_locs_coords[i, 2]
                    - d_locs_coords[j, 2]
                    + mu[tj, 2]
                    - mu[ti, 2]
                )
                / sigma_sq,
            )

            _cuda.atomic.add(d_val_sum, 0, val)
            d_val[pos] = 0

    def cuda_wrapper_chunked(
        mu,
        d_locs_coords,
        d_locs_time,
        d_idx_i,
        d_idx_j,
        d_sigma,
        d_sigma_factor,
        d_val,
        d_deri,
        chunk_size,
    ):
        """Interface between Python optimizer and the CUDA kernel (chunked to manage memory)."""
        val_total = 0
        d_val_sum = _cuda.to_device(np.zeros(1, dtype=np.float64))
        mu_dev = _cuda.to_device(
            np.asarray(mu.reshape(int(mu.size / 3), 3), dtype=np.float64)
        )

        n_chunks = int(np.ceil(d_idx_i.size / chunk_size))
        threadsperblock = 128

        for i in range(n_chunks - 1):
            idc_start = i * chunk_size
            blockspergrid = (
                chunk_size + (threadsperblock - 1)
            ) // threadsperblock
            cost_function_full_3d_chunked[blockspergrid, threadsperblock](
                d_locs_time,
                idc_start,
                chunk_size,
                d_idx_i,
                d_idx_j,
                d_sigma,
                d_sigma_factor,
                d_val,
                d_val_sum,
                d_deri,
                d_locs_coords,
                mu_dev,
            )
            val_total += d_val_sum.copy_to_host()

        # Final chunk
        n_remaining = d_idx_i.size - (n_chunks - 1) * chunk_size
        idc_start = (n_chunks - 1) * chunk_size
        blockspergrid = (
            n_remaining + (threadsperblock - 1)
        ) // threadsperblock
        cost_function_full_3d_chunked[blockspergrid, threadsperblock](
            d_locs_time,
            idc_start,
            n_remaining,
            d_idx_i,
            d_idx_j,
            d_sigma,
            d_sigma_factor,
            d_val,
            d_val_sum,
            d_deri,
            d_locs_coords,
            mu_dev,
        )
        val_total += d_val_sum.copy_to_host()
        deri = d_deri.copy_to_host()
        d_deri[:] = 0

        return -np.nansum(val_total), -deri.flatten()


@dataclass
class SegmentationResult:
    """Container for segmentation results."""

    loc_segments: np.ndarray
    loc_valid: np.ndarray
    center_frames: np.ndarray
    n_segments: int
    out_dict: Optional[Dict] = None


def _group_by_frame(loc_frames: np.ndarray):
    """Returns a dict {frame_number: indices_in_loc_frames} efficiently."""
    sort_idx = np.argsort(loc_frames)
    sorted_frames = loc_frames[sort_idx]
    unique_frames, start_idx, counts = np.unique(
        sorted_frames, return_index=True, return_counts=True
    )
    frame_to_indices = {
        frame: sort_idx[start : start + count]
        for frame, start, count in zip(unique_frames, start_idx, counts)
    }
    return unique_frames, frame_to_indices


def segment_by_num_locs_per_window(
    loc_frames: np.ndarray,
    min_n_locs_per_window: int,
    max_locs_per_segment=None,
    return_param_dict: bool = False,
) -> SegmentationResult:
    """
    Segments by collecting a minimum number of localizations per window.
    Once the threshold is met and enough locs remain, a new segment is created.
    This method ensures that each segment has at least `min_n_locs_per_window` localizations,
    while also trying to avoid creating segments that are too small at the end of the dataset.
    If `max_locs_per_segment` is set, a random subset of that size is chosen from each segment.
    This method is particularly useful for datasets with varying localization densities over time.
    Parameters:
    loc_frames (np.ndarray): Array of frame numbers for each localization.
    min_n_locs_per_window (int): Minimum number of localizations per segment.
    max_locs_per_segment (Optional[int]): Maximum number of localizations per segment. If None, all locs are used.
    return_param_dict (bool): Whether to return a dictionary of segmentation parameters.
    Returns:
    SegmentationResult: A dataclass containing segmentation results and parameters.
    """
    loc_frames = np.asarray(loc_frames, dtype=int)
    n_locs = len(loc_frames)

    if (
        max_locs_per_segment is not None and max_locs_per_segment < 1
    ):  # downsampling in percentage
        max_locs_per_segment = int(
            min_n_locs_per_window * max_locs_per_segment
        )

    unique_frames, frame_to_indices = _group_by_frame(loc_frames)
    loc_segments = np.full(n_locs, -1, dtype=int)  # Default to -1 for safety
    segment_counter = 0
    n_locs_in_current_segment = 0
    current_segment_indices = []
    start_frames, end_frames, locs_per_segment = [], [], []

    for i, frame in enumerate(unique_frames):
        indices = frame_to_indices[frame]
        n_locs_this_frame = len(indices)
        remaining_locs = n_locs - (
            len(current_segment_indices)
            + n_locs_this_frame
            + np.sum(locs_per_segment)
        )

        # Add frame to current segment if:
        # - It fills the current segment to threshold
        # - AND there are enough locs left for another segment (or it's the last frame)
        if (
            n_locs_in_current_segment + n_locs_this_frame
            >= min_n_locs_per_window
        ) and (
            remaining_locs >= min_n_locs_per_window
            or i == len(unique_frames) - 1
        ):
            current_segment_indices.extend(indices)
            n_locs_in_current_segment += n_locs_this_frame

            loc_segments[current_segment_indices] = segment_counter
            start_frames.append(loc_frames[current_segment_indices[0]])
            end_frames.append(loc_frames[current_segment_indices[-1]])
            locs_per_segment.append(len(current_segment_indices))

            segment_counter += 1
            current_segment_indices = []
            n_locs_in_current_segment = 0
        else:
            # Defer frame to current segment
            current_segment_indices.extend(indices)
            n_locs_in_current_segment += n_locs_this_frame

    n_segments = segment_counter
    center_frames = np.zeros(n_segments)
    loc_valid = np.zeros(n_locs, dtype=bool)

    for i in range(n_segments):
        segment_indices = np.where(loc_segments == i)[0]
        if (
            max_locs_per_segment
            and len(segment_indices) > max_locs_per_segment
        ):
            selected = np.random.choice(
                segment_indices, max_locs_per_segment, replace=False
            )
        else:
            selected = segment_indices
        loc_valid[selected] = True
        locs_per_segment[i] = len(selected)
        center_frames[i] = np.mean(loc_frames[selected])

    out_dict = None
    if return_param_dict:
        n_locs_valid = loc_valid.sum()
        out_dict = {
            "n_segments": n_segments,
            "min_n_locs_per_window": min_n_locs_per_window,
            "frames_per_window": -1,
            "start_frames": np.array(start_frames),
            "end_frames": np.array(end_frames),
            "locs_per_segment": np.array(locs_per_segment),
            "n_locs": n_locs,
            "n_locs_valid": n_locs_valid,
            "n_locs_invalid": n_locs - n_locs_valid,
            "center_frames": center_frames,
        }

    return SegmentationResult(
        loc_segments, loc_valid, center_frames, n_segments, out_dict
    )


def segment_by_frame_windows(
    loc_frames: np.ndarray,
    n_frames_per_window: int,
    max_locs_per_segment=None,
    return_param_dict: bool = False,
) -> SegmentationResult:
    """
    Splits localization data into fixed-size windows of N frames.
    All localizations in those frames are grouped into one segment.
    """
    loc_frames = np.asarray(loc_frames, dtype=int)
    frames, frame_to_indices = _group_by_frame(loc_frames)
    n_locs = len(loc_frames)
    n_segments = int(np.ceil(len(frames) / n_frames_per_window))

    if (
        max_locs_per_segment is not None and max_locs_per_segment < 1
    ):  # downsampling in percentage
        max_locs_per_segment = n_locs / n_segments * max_locs_per_segment

    loc_segments = np.zeros(n_locs, dtype=int)
    center_frames = np.zeros(n_segments)
    loc_valid = np.ones(n_locs, dtype=bool)
    start_frames, end_frames, locs_per_segment = [], [], []

    for i in range(n_segments):
        frame_window = frames[
            i * n_frames_per_window : (i + 1) * n_frames_per_window
        ]
        indices = np.concatenate(
            [
                frame_to_indices[frame]
                for frame in frame_window
                if frame in frame_to_indices
            ]
        )
        if len(indices) == 0:
            continue
        loc_segments[indices] = i
        start_frames.append(frame_window[0])
        end_frames.append(frame_window[-1])
        center_frames[i] = np.mean(loc_frames[indices])
        locs_per_segment.append(len(indices))
        if max_locs_per_segment and len(indices) > max_locs_per_segment:
            mask = np.ones(len(indices), dtype=bool)
            mask[
                np.random.choice(
                    len(indices),
                    len(indices) - max_locs_per_segment,
                    replace=False,
                )
            ] = False
            loc_valid[indices[~mask]] = False

    out_dict = None
    if return_param_dict:
        n_locs_valid = loc_valid.sum()
        out_dict = {
            "n_segments": n_segments,
            "min_n_locs_per_window": -1,
            "frames_per_window": n_frames_per_window,
            "start_frames": np.array(start_frames),
            "end_frames": np.array(end_frames),
            "locs_per_segment": np.array(locs_per_segment),
            "n_locs": n_locs,
            "n_locs_valid": n_locs_valid,
            "n_locs_invalid": n_locs - n_locs_valid,
            "center_frames": center_frames,
        }

    return SegmentationResult(
        loc_segments, loc_valid, center_frames, n_segments, out_dict
    )


def segment_by_num_windows(
    loc_frames: np.ndarray,
    n_windows: int,
    max_locs_per_segment=None,
    return_param_dict: bool = False,
) -> SegmentationResult:
    """
    Converts number of windows into an equivalent minimum locs per window,
    then calls `segment_by_num_locs_per_window`.
    """
    n_locs = len(loc_frames)
    n_locs_per_window = int(np.ceil(n_locs / n_windows))
    if (
        max_locs_per_segment is not None and max_locs_per_segment < 1
    ):  # downsampling in percentage
        max_locs_per_segment = int(n_locs_per_window * max_locs_per_segment)
    return segment_by_num_locs_per_window(
        loc_frames, n_locs_per_window, max_locs_per_segment, return_param_dict
    )


def segmentation_wrapper(
    loc_frames: np.ndarray,
    segmentation_var: int,
    segmentation_mode: int = 2,
    max_locs_per_segment=None,
    return_param_dict: bool = False,
) -> SegmentationResult:
    """
    Dispatch function that selects segmentation method:
    0 → fixed number of windows
    1 → fixed number of localizations per window
    2 → fixed number of frames per window (default)
    """
    if segmentation_mode == 0:
        return segment_by_num_windows(
            loc_frames,
            segmentation_var,
            max_locs_per_segment,
            return_param_dict,
        )
    elif segmentation_mode == 1:
        return segment_by_num_locs_per_window(
            loc_frames,
            segmentation_var,
            max_locs_per_segment,
            return_param_dict,
        )
    else:
        return segment_by_frame_windows(
            loc_frames,
            segmentation_var,
            max_locs_per_segment,
            return_param_dict,
        )


def pair_indices_kdtree(coordinates, distance):
    """
    Find all pairs of points within a certain distance using a KD-tree.
    Parameters:
    - coordinates: np.ndarray of shape (N, D) where N is the number of points and D is the dimensionality.
    - distance: float, the maximum distance to consider points as a pair.
    Returns:
    - idx1: np.ndarray of shape (M,), indices of the first point in each pair.
    - idx2: np.ndarray of shape (M,), indices of the second point in each pair.
    """
    tree = cKDTree(coordinates)
    try:
        pairs = tree.query_pairs(r=distance, output_type="ndarray")
    except MemoryError:
        print("[pair_indices_kdtree] MemoryError encountered")
        return [], [], False
    return (
        np.ascontiguousarray(pairs[:, 0], dtype=np.int32),
        np.ascontiguousarray(pairs[:, 1], dtype=np.int32),
        True,
    )
