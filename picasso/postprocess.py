import numpy as _np
from numpy.lib.recfunctions import append_fields as _append_fields
import numba as _numba
from sklearn.cluster import DBSCAN as _DBSCAN


def dbscan(locs, radius, min_density):
    X = _np.vstack((locs.x, locs.y)).T
    db = _DBSCAN(eps=radius, min_samples=min_density).fit(X)
    group = db.labels_
    locs = _append_fields(locs, 'group', group, 'i4', usemask=False, asrecarray=True)
    return locs


def get_dark_times(locs):
    last_frame = locs.frame + locs.len - 1
    dark = _get_dark_times(locs, last_frame)
    locs = _append_fields(locs, 'dark', dark, dtypes=dark.dtype, usemask=False, asrecarray=True)
    return locs


@_numba.jit(nopython=True)
def _get_dark_times(locs, last_frame):
    N = len(locs)
    max_frame = locs.frame.max()
    dark = max_frame * _np.ones(len(locs), dtype=_np.int32)
    for i in range(N):
        for j in range(N):
            if (locs.group[i] == locs.group[j]) and (i != j):
                dark_ij = locs.frame[i] - last_frame[j]
                if (dark_ij > 0) and (dark_ij < dark[i]):
                    dark[i] = dark_ij
    for i in range(N):
        if dark[i] == max_frame:
            dark[i] = -1
    return dark


def link(locs, radius, max_dark_time, combine_mode='average'):
    locs.sort(kind='mergesort', order='frame')
    group = get_link_groups(locs, radius, max_dark_time)
    if combine_mode == 'average':
        linked_locs = link_loc_groups(locs, group)
        # TODO: set len to -1 if loc lasts until last frame or starts at first frame
    elif combine_mode == 'refit':
        pass    # TODO
    return linked_locs


@_numba.jit(nopython=True)
def get_link_groups(locs, radius, max_dark_time):
    ''' Assumes that locs are sorted by frame '''
    frame = locs.frame
    x = locs.x
    y = locs.y
    N = len(x)
    group = -_np.ones(N, dtype=_np.int32)
    current_group = -1
    for i in range(N):
        if group[i] == -1:  # loc has no group yet
            current_group += 1
            group[i] = current_group
            current_index = i
            next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, radius, max_dark_time)
            while next_loc_index_in_group != -1:
                group[next_loc_index_in_group] = current_group
                current_index = next_loc_index_in_group
                next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, radius, max_dark_time)
    return group


@_numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, radius, max_dark_time):
    current_frame = frame[current_index]
    current_x = x[current_index]
    current_y = y[current_index]
    min_frame = current_frame + 1
    for min_index in range(current_index + 1, N):
        if frame[min_index] >= min_frame:
            break
    max_frame = current_frame + max_dark_time + 1
    for max_index in range(min_index + 1, N):
        if frame[max_index] > max_frame:
            break
    for j in range(min_index, max_index):
        if group[j] == -1:
            if (abs(current_x - x[j]) < radius) and (abs(current_y - y[j]) < radius):
                distance_to_current = _np.sqrt((current_x - x[j])**2 + (current_y - y[j])**2)
                if distance_to_current < radius:
                    return j
    return -1


def link_loc_groups(locs, group):
    linked_locs_data = _link_loc_groups(locs, group)
    dtype = locs.dtype.descr + [('len', 'u4'), ('n', 'u4')]
    return _np.rec.array(linked_locs_data, dtype=dtype)


@_numba.jit(nopython=True)
def _link_loc_groups(locs, group):
    N_linked = group.max() + 1
    frame_ = locs.frame.max() * _np.ones(N_linked, dtype=_np.uint32)
    x_ = _np.zeros(N_linked, dtype=_np.float32)
    y_ = _np.zeros(N_linked, dtype=_np.float32)
    photons_ = _np.zeros(N_linked, dtype=_np.float32)
    sx_ = _np.zeros(N_linked, dtype=_np.float32)
    sy_ = _np.zeros(N_linked, dtype=_np.float32)
    bg_ = _np.zeros(N_linked, dtype=_np.float32)
    lpx_ = _np.zeros(N_linked, dtype=_np.float32)
    lpy_ = _np.zeros(N_linked, dtype=_np.float32)
    len_ = _np.zeros(N_linked, dtype=_np.uint32)
    n_ = _np.zeros(N_linked, dtype=_np.uint32)
    last_frame_ = _np.zeros(N_linked, dtype=_np.uint32)
    weights_x = 1/locs.lpx**2
    weights_y = 1/locs.lpy**2
    sum_weights_x_ = _np.zeros(N_linked, dtype=_np.float32)
    sum_weights_y_ = _np.zeros(N_linked, dtype=_np.float32)
    N = len(group)
    for i in range(N):
        i_ = group[i]
        n_[i_] += 1
        x_[i_] += weights_x[i] * locs.x[i]
        sum_weights_x_[i_] += weights_x[i]
        y_[i_] += weights_y[i] * locs.y[i]
        sum_weights_y_[i_] += weights_y[i]
        photons_[i_] += locs.photons[i]
        sx_[i_] += locs.sx[i]
        sy_[i_] += locs.sy[i]
        bg_[i_] += locs.bg[i]
        if locs.frame[i] < frame_[i_]:
            frame_[i_] = locs.frame[i]
        if locs.frame[i] > last_frame_[i_]:
            last_frame_[i_] = locs.frame[i]
    x_ = x_ / sum_weights_x_
    y_ = y_ / sum_weights_y_
    sx_ = sx_ / n_
    sy_ = sy_ / n_
    bg_ = bg_ / n_
    lpx_ = _np.sqrt(1/sum_weights_x_)
    lpy_ = _np.sqrt(1/sum_weights_y_)
    len_ = last_frame_ - frame_ + 1
    return frame_, x_, y_, photons_, sx_, sy_, bg_, lpx_, lpy_, len_, n_


@_numba.jit(nopython=True)
def __link_loc_groups(locs, group):
    N_linked = group.max() + 1
    frame_ = locs.frame.max() * _np.ones(N_linked, dtype=_np.uint32)
    x_ = _np.zeros(N_linked, dtype=_np.float32)
    y_ = _np.zeros(N_linked, dtype=_np.float32)
    photons_ = _np.zeros(N_linked, dtype=_np.float32)
    sx_ = _np.zeros(N_linked, dtype=_np.float32)
    sy_ = _np.zeros(N_linked, dtype=_np.float32)
    bg_ = _np.zeros(N_linked, dtype=_np.float32)
    lpx_ = _np.zeros(N_linked, dtype=_np.float32)
    lpy_ = _np.zeros(N_linked, dtype=_np.float32)
    len_ = _np.zeros(N_linked, dtype=_np.uint32)
    n_ = _np.zeros(N_linked, dtype=_np.uint32)
    last_frame_ = _np.zeros(N_linked, dtype=_np.uint32)
    N = len(group)
    for i in range(N):
        i_ = group[i]
        n_[i_] += 1
        x_[i_] += locs.x[i]
        y_[i_] += locs.y[i]
        photons_[i_] += locs.photons[i]
        sx_[i_] += locs.sx[i]
        sy_[i_] += locs.sy[i]
        bg_[i_] += locs.bg[i]
        lpx_[i_] += locs.lpx[i]**2
        lpy_[i_] += locs.lpy[i]**2
        if locs.frame[i] < frame_[i_]:
            frame_[i_] = locs.frame[i]
        if locs.frame[i] > last_frame_[i_]:
            last_frame_[i_] = locs.frame[i]
    x_ = x_ / n_
    y_ = y_ / n_
    sx_ = sx_ / n_
    sy_ = sy_ / n_
    bg_ = bg_ / n_
    lpx_ = _np.sqrt(lpx_) / n_
    lpy_ = _np.sqrt(lpy_) / n_
    len_ = last_frame_ - frame_ + 1
    return frame_, x_, y_, photons_, sx_, sy_, bg_, lpx_, lpy_, len_, n_


def calculate_optimal_bins(data, max_n_bins=None):
    iqr = _np.subtract(*_np.percentile(data, [75, 25]))
    bin_size = 2 * iqr * len(data)**(-1/3)
    if data.dtype.kind in ('u', 'i') and bin_size < 1:
        bin_size = 1
    bin_min = max(data.min() - bin_size / 2, 0)
    n_bins = int(_np.ceil((data.max() - bin_min) / bin_size))
    if max_n_bins and n_bins > max_n_bins:
        n_bins = max_n_bins
    return _np.linspace(bin_min, data.max(), n_bins)
