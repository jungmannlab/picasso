import numpy as _np
import numba as _numba
from localize import LOCS_DTYPE as _LOCS_DTYPE

LINKED_LOCS_DTYPE = _LOCS_DTYPE + [('len', 'u4')]


def link(locs, radius, max_dark_time):
    group = _get_link_groups(locs.x, locs.y, locs.frame, radius, max_dark_time)
    n_groups = group.max() + 1
    linked_locs = _np.recarray((n_groups,), dtype=LINKED_LOCS_DTYPE)
    for i in range(n_groups):
        group_locs = locs[group == i]
        n_group_locs = len(group_locs)
        linked_locs.frame[i] = group_locs.frame.min()
        linked_locs.x[i] = group_locs.x.mean()
        linked_locs.y[i] = group_locs.y.mean()
        linked_locs.photons[i] = group_locs.photons.sum()
        linked_locs.sx[i] = group_locs.sx.mean()
        linked_locs.sy[i] = group_locs.sy.mean()
        linked_locs.bg[i] = group_locs.bg.mean()
        linked_locs.CRLBx[i] = _np.sqrt(group_locs.CRLBx**2) / n_group_locs
        linked_locs.CRLBy[i] = _np.sqrt(group_locs.CRLBy**2) / n_group_locs
        linked_locs.len[i] = group_locs.max() - group_locs.min()
    return linked_locs


@_numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(current_index, group, N, x, y, frame, radius, max_dark_time):
    possible_frames = range(frame[current_index] + 1, frame[current_index] + max_dark_time + 2)
    for j in range(N):
        if group[j] != -1:
            if j != current_index:
                for possible_frame in possible_frames:
                    if frame[j] == possible_frame:
                        distance_to_current = _np.sqrt((x[current_index] - x[j])**2 + (y[current_index] - y[j])**2)
                        if distance_to_current <= radius:
                            return j
    return None


@_numba.jit(nopython=True)
def _get_link_groups(x, y, frame, radius, max_dark_time):
    N = len(x)
    group = -_np.ones(N, dtype=_np.int32)
    current_group = 0
    for i in range(N):
        if group[i] == -1:  # loc has no group yet
            group[i] = current_group
            current_index = i
            next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, group, N, x, y, frame, radius, max_dark_time)
            while next_loc_index_in_group:
                group[next_loc_index_in_group] = current_group
                current_index = next_loc_index_in_group
                next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, group, N, x, y, frame, radius, max_dark_time)
            current_group += 1
    return group
