import numpy as _np
import numba as _numba
from . import localize as _localize

LINKED_LOCS_DTYPE = _localize.LOCS_DTYPE + [('len', 'u4')]


def link(locs, radius, max_dark_time, combine_mode='average'):
    locs.sort(kind='mergesort', order='frame')
    group = _get_link_groups(locs.frame, locs.x, locs.y, radius, max_dark_time)
    if combine_mode == 'average':
        linked_locs_data = _link_locs(group, locs)
        linked_locs = _np.rec.array(linked_locs_data, dtype=LINKED_LOCS_DTYPE)
    elif combine_mode == 'refit':
        pass    # TODO
    return linked_locs


@_numba.jit(nopython=True)
def _link_locs(group, locs):
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
    return frame_, x_, y_, photons_, sx_, sy_, bg_, lpx_, lpy_, len_


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


@_numba.jit(nopython=True)
def _get_link_groups(frame, x, y, radius, max_dark_time):
    ''' Assumes that locs are sorted by frame '''
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
