"""
    picasso.aim
    ~~~~~~~~~~~

    Picasso implementation of Adaptive Intersection Maximization (AIM)
    for fast undrifting in 2D and 3D. 

    Adapted from: Ma, H., et al. Science Advances. 2024.

    :author: Hongqiang Ma, Maomao Chen, Phuong Nguyen, Yang Liu,
        Rafal Kowalewski, 2024
    :copyright: Copyright (c) 2016-2024 Jungmann Lab, MPI of Biochemistry
"""

from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

import numpy as _np
from scipy.interpolate import InterpolatedUnivariateSpline as \
    _InterpolatedUnivariateSpline
from tqdm import tqdm as _tqdm


def intersect1d(a, b):
    """Slightly faster implementation of _np.intersect1d without 
    unnecessary checks, etc.
    
    Finds the indices of common elements in two 1D arrays (a and b). 
    Both a and b are assumed to be sorted and contain only unique 
    values.
    
    Parameters
    ----------
    a : _np.array
        1D array of integers.
    b : _np.array
        1D array of integers.
    
    Returns
    -------
    a_indices : _np.array
        Indices of common elements in a.
    b_indices : _np.array
        Indices of common elements in b.
    """

    aux = _np.concatenate((a, b))
    aux_sort_indices = _np.argsort(aux, kind='mergesort')
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]
    a_indices = aux_sort_indices[:-1][mask]
    b_indices = aux_sort_indices[1:][mask] - a.size

    return a_indices, b_indices


def count_intersections(l0_coords, l0_counts, l1_coords, l1_counts):
    """Counts the number of intersected localizations between the two
    datasets. We assume that the intersection distance is 1 and since 
    the coordinates are expressed in the units of intersection distance,
    we require the coordinates to be exactly the same to count as 
    intersection. Also, coordinates are converted to 1D arrays 
    (x + y * width).

    Parameters
    ----------
    l0_coords : _np.array
        Unique coordinates of the reference localizations.
    l0_counts : _np.array
        Counts of the unique values of reference localizations.
    l1_coords : _np.array
        Unique coordinates of the target localizations.
    l1_counts : _np.array
        Counts of the unique values of target localizations.
    
    Returns
    -------
    n_intersections : int
        Number of intersections.
    """

    # indices of common elements
    idx0, idx1 = intersect1d(l0_coords, l1_coords) 
    # extract the counts of these elements
    l0_counts_subset = l0_counts[idx0]
    l1_counts_subset = l1_counts[idx1]
    # for each overlapping coordinate, take the minimum count from l0
    # and l1, sum up across all overlapping coordinates
    n_intersections = _np.sum(
        _np.minimum(l0_counts_subset, l1_counts_subset)
    )
    return n_intersections


def run_intersections(
    l0_coords, l0_counts, l1_coords, l1_counts, shifts_xy, box
):
    """Run intersection counting across the local search region. Returns
    the 2D array with number of intersections across the local search
    region.
    
    Parameters
    ----------
    l0_coords : _np.array
        Unique coordinates of the reference localizations.
    l0_counts : _np.array
        Counts of the reference localizations.
    l1_coords : _np.array
        Unique coordinates of the target localizations.
    l1_counts : _np.array
        Counts of the target localizations.
    shifts_xy : _np.array
        1D array with x and y shifts.
    box : int  
        Side length of the local search region.
    
    Returns
    -------
    roi_cc : _np.2darray
        2D array with number of intersections across the local search 
        region.
    """
    
    # create the 2D array with shifts
    roi_cc = _np.zeros(shifts_xy.shape, dtype=_np.int32)
    # shift target coordinates
    l1_coords_shifted = l1_coords[:, _np.newaxis] + shifts_xy
    # go through each element in the local search region
    for i in range(len(shifts_xy)):
        n_intersections = count_intersections(
            l0_coords, l0_counts, l1_coords_shifted[:, i], l1_counts
        )
        roi_cc[i] = n_intersections
    return roi_cc.reshape(box, box)


def run_intersections_multithread(
    l0_coords, l0_counts, l1_coords, l1_counts, shifts_xy, box
):
    """Run intersection counting across the local search region. Returns
    the 2D array with number of intersections across the local search
    region. Uses multithreading.
    
    Parameters
    ----------
    l0_coords : _np.array
        Unique coordinates of the reference localizations.
    l0_counts : _np.array
        Counts of the reference localizations.
    l1_coords : _np.array
        Unique coordinates of the target localizations.
    l1_counts : _np.array
        Counts of the target localizations.
    shifts_xy : _np.array
        1D array with x and y shifts.
    box : int  
        Side length of the local search region.
    
    Returns
    -------
    roi_cc : _np.2darray
        2D array with number of intersections across the local search 
        region.
    """
    
    # shift target coordinates
    l1_coords_shifted = l1_coords[:, _np.newaxis] + shifts_xy
    # run multiple threads
    n_workers = len(shifts_xy) 
    executor = _ThreadPoolExecutor(n_workers)
    f = [
        executor.submit(
            count_intersections, 
            l0_coords, 
            l0_counts, 
            l1_coords_shifted[:, i], 
            l1_counts,
        )
        for i in range(len(shifts_xy))
    ]
    executor.shutdown(wait=True)
    if box == 1: # z intersection only, for z undrifting
        roi_cc = _np.array([_.result() for _ in f])
    else: # 2D intersection
        roi_cc = _np.array([_.result() for _ in f]).reshape(box, box)
    return roi_cc


def point_intersect_2d(
    l0_coords, l0_counts, x1, y1, intersect_d, width_units, shifts_xy, box
):
    """Converts target coordinates into a 1D array in units of 
    intersect_d and counts the number of intersections in the local
    search region.
    
    Parameters
    ----------
    l0_coords : _np.array
        Unique values of the reference localizations.
    l0_counts : _np.array
        Counts of the unique values of reference localizations.
    x1 : _np.array
        x-coordinates of the target (currently undrifted) localizations.
    y1 : _np.array
        y-coordinates of the target (currently undrifted) localizations.
    intersect_d : float
        Intersect distance in camera pixels.
    width_units : int 
        Width of the camera image in units of intersect_d.
    shifts_xy : _np.array
        1D array with x and y shifts.
    box : int  
        Final side length of the local search region.
    
    Returns
    -------
    roi_cc : _np.2darray
        2D array with numbers of intersections in the local search 
        region.
    """

    # convert target coordinates to a 1D array in intersect_d units
    x1_units = _np.round(x1 / intersect_d)
    y1_units = _np.round(y1 / intersect_d)
    l1 = _np.int32(x1_units + y1_units * width_units) # 1d list
    # get unique values and counts of the target localizations
    l1_coords, l1_counts = _np.unique(l1, return_counts=True)
    # run the intersections counting
    roi_cc = run_intersections_multithread(
        l0_coords, l0_counts, l1_coords, l1_counts, shifts_xy, box
    )
    return roi_cc


def point_intersect_3d(
    l0_coords, l0_counts, x1, y1, z1, 
    intersect_d, width_units, height_units, shifts_z,
): 
    """Converts target coordinates into a 1D array in units of 
    intersect_d and counts the number of intersections in the local
    search region.
    
    Parameters
    ----------
    l0_coords : _np.array
        Unique values of the reference localizations.
    l0_counts : _np.array
        Counts of the unique values of reference localizations.
    x1 : _np.array
        x-coordinates of the target (currently undrifted) localizations.
    y1 : _np.array
        y-coordinates of the target (currently undrifted) localizations.
    z1 : _np.array
        z-coordinates of the target (currently undrifted) localizations.
    intersect_d : float
        Intersect distance in camera pixels.
    width_units : int 
        Width of the camera image in units of intersect_d.
    height_units : int
        Height of the camera image in units of intersect_d.
    shifts_z : _np.array
        1D array with z shifts.
    
    Returns
    -------
    roi_cc : _np.2darray
        2D array with numbers of intersections in the local search 
        region.
    """

    # convert target coordinates to a 1D array in intersect_d units
    x1_units = _np.round(x1 / intersect_d)
    y1_units = _np.round(y1 / intersect_d)
    z1_units = _np.round(z1 / intersect_d)
    l1 = _np.int32(
        x1_units 
        + y1_units * width_units 
        + z1_units * width_units * height_units
    ) # 1d list
    # get unique values and counts of the target localizations
    l1_coords, l1_counts = _np.unique(l1, return_counts=True)
    # run the intersections counting
    roi_cc = run_intersections_multithread(
        l0_coords, l0_counts, l1_coords, l1_counts, shifts_z, 1
    )
    return roi_cc


def get_fft_peak(roi_cc, roi_size):
    """Estimate the precise sub-pixel position of the peak of roi_cc 
    with FFT.
    
    Parameters
    ----------
    roi_cc : _np.2darray
        2D array with numbers of intersections in the local search region.
    roi_size : int
        Size of the local search region.
    
    Returns
    -------
    px : float
        Estimated x-coordinate of the peak.
    py : float
        Estimated y-coordinate of the peak.
    """
    
    fft_values = _np.fft.fft2(roi_cc.T) 
    ang_x = _np.angle(fft_values[0, 1]) 
    ang_x = ang_x - 2 * _np.pi * (ang_x > 0) # normalize
    px = (
        _np.abs(ang_x) / (2 * _np.pi / roi_cc.shape[0])
        - (roi_cc.shape[0] - 1) / 2
    ) # peak in x
    px *= roi_size / roi_cc.shape[0] # convert to intersect_d units
    ang_y = _np.angle(fft_values[1, 0]) 
    ang_y = ang_y - 2 * _np.pi * (ang_y > 0) # normalize 
    py = (
        _np.abs(ang_y) / (2 * _np.pi / roi_cc.shape[1])
        - (roi_cc.shape[1] - 1) / 2
    ) # peak in y
    py *= roi_size / roi_cc.shape[1] # convert to intersect_d units
    return px, py


def get_fft_peak_z(roi_cc, roi_size):
    """Estimate the precise sub-pixel position of the peak of 1D roi_cc.
    
    Parameters
    ----------
    roi_cc : _np.array
        1D array with numbers of intersections in the local search 
        region.
    roi_size : int
        Size of the local search region.

    Returns
    -------
    pz : float
        Estimated z-coordinate of the peak.
    """

    fft_values = _np.fft.fft(roi_cc) 
    ang_z = _np.angle(fft_values[1]) 
    ang_z = ang_z - 2 * _np.pi * (ang_z > 0) # normalize
    pz = (
        _np.abs(ang_z) / (2 * _np.pi / roi_cc.size)
        - (roi_cc.size - 1) / 2
    ) # peak in z
    pz *= roi_size / roi_cc.size # convert to intersect_d units
    return pz


def intersection_max(
    x, y, ref_x, ref_y, 
    frame, seg_bounds, intersect_d, roi_r, width, 
    aim_round=1, progress=None,
):
    """Maximize intersection (undrift) for 2D localizations.
    
    Parameters
    ----------
    x : _np.array
        x-coordinates of the localizations.
    y : _np.array
        y-coordinates of the localizations.
    ref_x_list : _np.array
        x-coordinates of the reference localizations.
    ref_y_list : _np.array
        y-coordinates of the reference localizations.
    frame : _np.array 
        Frame indices of localizations.
    seg_bounds : np.array
        Frame indices of the segmentation bounds. Defines temporal 
        intervals used to estimate drift.
    intersect_d : float
        Intersect distance in camera pixels.
    roi_r : float
        Radius of the local search region in camera pixels. Should be 
        higher than the maximum expected drift within one segment.
    width : int
        Width of the camera image in camera pixels.
    aim_round : {1, 2}
        Round of AIM algorithm. The first round uses the first interval
        as reference, the second round uses the entire dataset as
        reference. The impact is that in the second round, the first
        interval is also undrifted.
    progress : picasso.lib.ProgressDialog (default=None)
        Progress dialog. If None, progress is displayed with tqdm.
    
    Returns
    -------
    x_pdc : _np.array
        Undrifted x-coordinates.
    y_pdc : _np.array
        Undrifted y-coordinates.
    drift_x : _np.array
        Drift in x-direction.
    drift_y : _np.array
        Drift in y-direction.
    """

    assert aim_round in [1, 2], "aim_round must be 1 or 2."

    # number of segments
    n_segments = len(seg_bounds) - 1
    rel_drift_x = 0 # adaptive drift (updated at each interval)
    rel_drift_y = 0

    # drift in x and y
    drift_x = _np.zeros(n_segments) 
    drift_y = _np.zeros(n_segments) 

    # find shifts for the local search region (in units of intersect_d)
    roi_units = int(_np.ceil(roi_r / intersect_d))
    steps = _np.arange(-roi_units, roi_units + 1, 1)
    box = len(steps)
    shifts_xy = _np.zeros((box, box), dtype=_np.int32)
    width_units = width / intersect_d
    for i, shift_x in enumerate(steps):
        for j, shift_y in enumerate(steps):
            shifts_xy[i, j] = shift_x + shift_y * width_units
    shifts_xy = shifts_xy.reshape(box ** 2)

    # convert reference to a 1D array in units of intersect_d and find
    # unique values and counts
    x0_units = _np.round(ref_x / intersect_d)
    y0_units = _np.round(ref_y / intersect_d)
    l0 = _np.int32(x0_units + y0_units * width_units) # 1d list
    l0_coords, l0_counts = _np.unique(l0, return_counts=True)

    # initialize progress such that if GUI is used, tqdm is omitted
    start_idx = 1 if aim_round == 1 else 0
    if progress is not None:
        iterator = range(start_idx, n_segments)
    else:
        iterator = _tqdm(
            range(start_idx, n_segments), 
            desc=f"Undrifting ({aim_round}/2)", 
            unit="segment",
        ) 

    # run across each segment
    for s in iterator:
        # get the target localizations within the current segment
        min_frame_idx = frame > seg_bounds[s]
        max_frame_idx = frame <= seg_bounds[s+1]
        x1 = x[min_frame_idx & max_frame_idx]
        y1 = y[min_frame_idx & max_frame_idx]

        # skip if no reference localizations
        if len(x1) == 0:
            drift_x[s] = drift_x[s-1]
            drift_y[s] = drift_y[s-1]
            continue

        # undrifting from the previous round
        x1 += rel_drift_x
        y1 += rel_drift_y

        # count the number of intersected localizations
        roi_cc = point_intersect_2d(
            l0_coords, l0_counts, x1, y1, 
            intersect_d, width_units, shifts_xy, box,
        )

        # estimate the precise sub-pixel position of the peak of roi_cc 
        # with FFT 
        px, py = get_fft_peak(roi_cc, 2 * roi_r)

        # update the relative drift reference for the subsequent 
        # segmented subset (interval) and save the drifts
        rel_drift_x += px 
        rel_drift_y += py
        drift_x[s] = -rel_drift_x
        drift_y[s] = -rel_drift_y

        # update progress
        if progress is not None:
            progress.set_value(s)
        else:
            iterator.update(s - iterator.n)

    # interpolate the drifts (cubic spline) for all frames
    t = (seg_bounds[1:] + seg_bounds[:-1]) / 2
    drift_x_pol = _InterpolatedUnivariateSpline(t, drift_x, k=3)
    drift_y_pol = _InterpolatedUnivariateSpline(t, drift_y, k=3)
    t_inter = _np.arange(seg_bounds[-1]) + 1
    drift_x = drift_x_pol(t_inter)
    drift_y = drift_y_pol(t_inter)

    # undrift the localizations
    x_pdc = x - drift_x[frame-1]
    y_pdc = y - drift_y[frame-1]

    return x_pdc, y_pdc, drift_x, drift_y


def intersection_max_z(
    x, y, z, ref_x, ref_y, ref_z,
    frame, seg_bounds, intersect_d, roi_r, width, height, pixelsize,
    aim_round=1, progress=None,
):
    """Maximize intersection (undrift) for 3D localizations. Assumes
    that x and y coordinates were already undrifted. x and y are in
    units of camera pixels, z is in nm.
    
    See intersection_max for more details."""

    # convert z to camera pixels
    z = z.copy() / pixelsize
    ref_z = ref_z.copy() / pixelsize #TODO: remember to convert back to nm, also for drift_z

    # number of segments
    n_segments = len(seg_bounds) - 1
    rel_drift_z = 0 # adaptive drift (updated at each interval)

    # drift in z
    drift_z = _np.zeros(n_segments)

    # find shifts for the local search region (in units of intersect_d)
    roi_units = int(_np.ceil(roi_r / intersect_d))
    steps = _np.arange(-roi_units, roi_units + 1, 1)
    box = len(steps)
    width_units = width / intersect_d
    height_units = height / intersect_d
    shifts_z = steps.astype(_np.int32) * width_units * height_units

    # convert reference to a 1D array in units of intersect_d and find
    # unique values and counts
    x0_units = _np.round(ref_x / intersect_d)
    y0_units = _np.round(ref_y / intersect_d)
    z0_units = _np.round(ref_z / intersect_d)
    l0 = _np.int32(
        x0_units 
        + y0_units * width_units 
        + z0_units * width_units * height_units
    ) # 1d list
    l0_coords, l0_counts = _np.unique(l0, return_counts=True) 

    # initialize progress such that if GUI is used, tqdm is omitted
    start_idx = 1 if aim_round == 1 else 0
    if progress is not None:
        iterator = range(start_idx, n_segments)
    else:
        iterator = _tqdm(
            range(start_idx, n_segments), 
            desc=f"Undrifting z ({aim_round}/2)", 
            unit="segment",
        )

    # run across each segment
    for s in iterator:
        # get the target localizations within the current segment
        min_frame_idx = frame > seg_bounds[s]
        max_frame_idx = frame <= seg_bounds[s+1]
        x1 = x[min_frame_idx & max_frame_idx]
        y1 = y[min_frame_idx & max_frame_idx]
        z1 = z[min_frame_idx & max_frame_idx]

        # skip if no reference localizations
        if len(x1) == 0:
            drift_z[s] = drift_z[s-1]
            continue

        # undrifting from the previous round
        z1 += rel_drift_z

        # count the number of intersected localizations
        roi_cc = point_intersect_3d(
            l0_coords, l0_counts, x1, y1, z1,
            intersect_d, width_units, height_units, shifts_z,
        )

        # estimate the precise sub-pixel position of the peak of roi_cc 
        # with FFT 
        pz = get_fft_peak_z(roi_cc, 2 * roi_r)

        # update the relative drift reference for the subsequent 
        # segmented subset (interval) and save the drifts
        rel_drift_z += pz
        drift_z[s] = -rel_drift_z

        # update progress
        if progress is not None:
            progress.set_value(s)
        else:
            iterator.update(s - iterator.n)
        

    # interpolate the drifts (cubic spline) for all frames
    t = (seg_bounds[1:] + seg_bounds[:-1]) / 2
    drift_z_pol = _InterpolatedUnivariateSpline(t, drift_z, k=3)
    t_inter = _np.arange(seg_bounds[-1]) + 1
    drift_z = drift_z_pol(t_inter)

    # undrift the localizations
    z_pdc = z - drift_z[frame-1]

    # convert back to nm
    z_pdc *= pixelsize
    drift_z *= pixelsize

    return z_pdc, drift_z    


def aim(
    locs, 
    info,
    segmentation=100, 
    intersect_d=20/130, 
    roi_r=60/130,
    progress=None,
):
    """Apply AIM undrifting to the localizations.

    Parameters
    ----------
    locs : _np.rec.array
        Localizations list to be undrifted.
    info : list of dicts
        Localizations list's metadata.
    intersect_d : float
        Intersect distance in camera pixels.
    segmentation : int
        Time interval for drift tracking, unit: frames.
    roi_r : float
        Radius of the local search region in camera pixels. Should be 
        larger than the  maximum expected drift within segmentation.
    progress : picasso.lib.ProgressDialog (default=None)
        Progress dialog. If None, progress is displayed with tqdm.
    
    Returns
    -------
    locs : _np.rec.array
        Undrifted localizations.
    new_info : list of 1 dict
        Updated metadata.
    drift : _np.rec.array
        Drift in x and y directions (and z if applicable).
    """

    # extract metadata
    width = _np.nan
    height = _np.nan
    pixelsize = _np.nan
    n_frames = _np.nan
    for inf in info:
        if val := inf.get("Width"):
            width = val
        if val := inf.get("Height"):
            height = val
        if val := inf.get('Frames'):
            n_frames = val
        if val := inf.get("Pixelsize"):
            pixelsize = val
    if _np.isnan(width * height * pixelsize * n_frames):
        raise KeyError("Insufficient metadata available.")

    # frames should start at 1 
    frame = locs["frame"] + 1
    # find the segmentation bounds (temporal intervals)
    seg_bounds = _np.concatenate((
        _np.arange(0, n_frames, segmentation), [n_frames]
    ))

    # get the reference localizations (first interval)
    ref_x = locs["x"][frame <= segmentation]
    ref_y = locs["y"][frame <= segmentation]

    ### RUN AIM TWICE ###
    # the first run is with the first interval as reference
    x_pdc, y_pdc, drift_x1, drift_y1 = intersection_max(
        locs.x, locs.y, ref_x, ref_y,
        frame, seg_bounds, intersect_d, roi_r, width, 
        aim_round=1, progress=progress,
    )
    # the second run is with the entire dataset as reference
    if progress is not None:
        progress.zero_progress(description="Undrifting by AIM (2/2)")
    x_pdc, y_pdc, drift_x2, drift_y2 = intersection_max(
        x_pdc, y_pdc, x_pdc, y_pdc,
        frame, seg_bounds, intersect_d, roi_r, width,
        aim_round=2, progress=progress,
    )

    # add the drifts together from the two rounds
    drift_x = drift_x1 + drift_x2
    drift_y = drift_y1 + drift_y2

    # shift the drifts by the mean value
    shift_x = _np.mean(drift_x)
    shift_y = _np.mean(drift_y)
    drift_x -= shift_x
    drift_y -= shift_y
    x_pdc += shift_x
    y_pdc += shift_y

    # combine to Picasso format
    drift = _np.rec.array((drift_x, drift_y), dtype=[("x", "f"), ("y", "f")])

    # 3D undrifting
    if hasattr(locs, "z"):
        if progress is not None:
           progress.zero_progress(description="Undrifting z (1/2)")
        ref_x = x_pdc[frame <= segmentation]
        ref_y = y_pdc[frame <= segmentation]
        ref_z = locs.z[frame <= segmentation]
        z_pdc, drift_z1 = intersection_max_z(
            x_pdc, y_pdc, locs.z, ref_x, ref_y, ref_z,
            frame, seg_bounds, intersect_d, roi_r, width, height, pixelsize,
            aim_round=1, progress=progress,
        )
        if progress is not None:
            progress.zero_progress(description="Undrifting z (2/2)")
        z_pdc, drift_z2 = intersection_max_z(
            x_pdc, y_pdc, z_pdc, x_pdc, y_pdc, z_pdc,
            frame, seg_bounds, intersect_d, roi_r, width, height, pixelsize,
            aim_round=2, progress=progress,
        )
        drift_z = drift_z1 + drift_z2
        shift_z = _np.mean(drift_z)
        drift_z -= shift_z
        z_pdc += shift_z
        drift = _np.rec.array(
            (drift_x, drift_y, drift_z), 
            dtype=[("x", "f"), ("y", "f"), ("z", "f")]
        )
    
    # apply the drift to localizations
    locs["x"] = x_pdc
    locs["y"] = y_pdc
    if hasattr(locs, "z"):
        locs["z"] = z_pdc

    new_info = {
        "Generated by": "AIM undrift",
        "Intersect distance (nm)": intersect_d * pixelsize,
        "Segmentation": segmentation,
        "Search regions radius (nm)": roi_r * pixelsize,
    }
    new_info = info + [new_info]

    if progress is not None:
        progress.close()

    return locs, new_info, drift