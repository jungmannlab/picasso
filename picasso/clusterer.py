"""
	picasso.clusterer
	~~~~~~~~~~~~~~~~~

	Clusterer optimized for DNA PAINT in CPU and GPU versions.

	Based on the work of Susanne Reinhardt.
"""

import os as _os

import numpy as _np
import yaml as _yaml
from scipy.spatial import distance_matrix as _dm
from numba import njit as _njit
from numba import cuda as _cuda
from tqdm import tqdm as _tqdm

from icecream import ic

#todo: gpu clustering, check:
'''
threadsperblock = 32
blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
increment_by_one[blockspergrid, threadsperblock](an_array)
'''
#todo: 3D for all cases!
#todo: docstrings
#todo: check if I can put the lm[i] == 1 loop in picked clusterer inside njit

def get_distance(x1, x2, y1, y2):
	return _np.sqrt((x2-x1)**2 + (y2-y1)**2)

def assing_locs_to_boxes(x, y, box_size):

	# assigns each loc to a box
	box_id = _np.zeros(len(x), dtype=_np.int32)
	
	x_start = x.min()
	x_end = x.max()
	y_start = y.min()
	y_end = y.max()

	n_boxes_x = int((x_end - x_start) / box_size) + 1
	n_boxes_y = int((y_end - y_start) / box_size) + 1
	n_boxes = n_boxes_x * n_boxes_y

	for i in range(len(x)):
		box_id[i] = (
			int((x[i] - x_start) / box_size)
			+ int((y[i] - y_start) / box_size)
			* n_boxes_x
		)

	# gives indeces of locs in a given box
	locs_id_box = [[]]
	for i in range(n_boxes):
		locs_id_box.append([])

	# fill values for locs_id_box
	# also add locs that are in the adjacent boxes
	for i in range(len(x)):
		locs_id_box[box_id[i]].append(i)

		if box_id[i] != n_boxes:
			locs_id_box[box_id[i]+1].append(i)
		if box_id[i] != 0:
			locs_id_box[box_id[i]-1].append(i)

		if box_id[i] > n_boxes_x:
			locs_id_box[box_id[i]-n_boxes_x].append(i)
			locs_id_box[box_id[i]-n_boxes_x+1].append(i)
			locs_id_box[box_id[i]-n_boxes_x-1].append(i)

		if box_id[i] < n_boxes - n_boxes_x:
			locs_id_box[box_id[i]+n_boxes_x].append(i)
			locs_id_box[box_id[i]+n_boxes_x+1].append(i)
			locs_id_box[box_id[i]+n_boxes_x-1].append(i)	

	return locs_id_box, box_id	

def count_neighbors_CPU(locs_id_box, box_id, x, y, radius):
	nn = _np.zeros(len(x), dtype=_np.int32) # number of neighbors
	for i in range(len(x)):
		for j_, j in enumerate(locs_id_box[box_id[i]]):
			if i != j_:
				dist = get_distance(x[i], x[j], y[i], y[j])
				if dist <= radius:
					nn[i] += 1
	return nn

def local_maxima_CPU(locs_id_box, box_id, nn, x, y, radius):
	lm = _np.zeros(len(x))
	for i in range(len(x)):
		for j in locs_id_box[box_id[i]]:
			dist = get_distance(x[i], x[j], y[i], y[j])
			if dist <= radius and nn[i] >= nn[j]:
				lm[i] = 1
			if dist <= radius and nn[i] < nn[j]:
				lm[i] = 0
				break
	return lm

def assign_to_cluster_CPU(locs_id_box, box_id, nn, lm, x, y, radius):
	cluster_id = _np.zeros(len(x), dtype=_np.int32)
	for i in range(len(x)):
		if lm[i]:
			for j in locs_id_box[box_id[i]]:
				dist = get_distance(x[i], x[j], y[i], y[j])
				if dist <= radius:
					if cluster_id[i] != 0:
						if cluster_id[j] == 0:
							cluster_id[j] = cluster_id[i]
					else:
						if j == 0:
							cluster_id[i] = i + 1
						cluster_id[j] = i + 1
	return cluster_id

def clusterer_CPU(x, y, frame, eps, min_samples):
	"""
	almost the same as clusterer picked, except locs are allocated
	to boxes and neighbors counting is slightly different.
	"""
	xy = _np.stack((x, y)).T
	locs_id_box, box_id = assing_locs_to_boxes(x, y, eps)
	n_neighbors = count_neighbors_CPU(locs_id_box, box_id, x, y, eps)
	local_max = local_maxima_CPU(locs_id_box, box_id, n_neighbors, x, y, eps)
	cluster_id = assign_to_cluster_CPU(
		locs_id_box, box_id, n_neighbors, local_max, x, y, eps
	)
	cluster_n_locs = _np.bincount(cluster_id)
	cluster_id = check_cluster_size(
		cluster_n_locs, min_samples, cluster_id
	)
	clusters = _np.unique(cluster_id)
	cluster_id = rename_clusters(cluster_id, clusters)
	n_clusters = len(clusters)
	mean_frame, locs_perc = cluster_properties(
		cluster_id, n_clusters, frame
	)
	true_cluster = find_true_clusters(mean_frame, locs_perc, frame)
	labels = -1 * _np.ones(len(x), dtype=_np.int32)
	for i in range(len(x)):
		if cluster_id[i] != 0 and true_cluster[cluster_id[i]] == 1:
			labels[i] = cluster_id[i] - 1
	return labels

@_njit 
def count_neighbors_picked(dist, eps):
	nn = _np.zeros(dist.shape[0], dtype=_np.int32)
	for i in range(len(nn)):
		nn[i] = _np.where(dist[i] <= eps)[0].shape[0] - 1
	return nn

@_njit
def local_maxima_picked(dist, nn, eps):
	"""
	Finds the positions of the local maxima which are the locs 
	with the highest number of neighbors within given distance dist
	"""
	n = dist.shape[0]
	lm = _np.zeros(n, dtype=_np.int32)
	for i in range(n):
		for j in range(n):
			if dist[i][j] <= eps and nn[i] >= nn[j]:
				lm[i] = 1
			if dist[i][j] <= eps and nn[i] < nn[j]:
				lm[i] = 0
				break
	return lm

@_njit
def assign_to_cluster_picked(dist, eps, lm):
	n = dist.shape[0]
	cluster_id = _np.zeros(n, dtype=_np.int32)
	for i in range(n):
		if lm[i]:
			for j in range(n):
				if dist[i][j] <= eps:
					if cluster_id[i] != 0:
						if cluster_id[j] == 0:
							cluster_id[j] = cluster_id[i]
					if cluster_id[i] == 0:
						if j == 0:
							cluster_id[i] = i + 1
						cluster_id[j] = i + 1
	return cluster_id

@_njit
def check_cluster_size(cluster_n_locs, min_samples, cluster_id):
	for i in range(len(cluster_id)):
		if cluster_n_locs[cluster_id[i]] <= min_samples:
			cluster_id[i] = 0
	return cluster_id

@_njit
def rename_clusters(cluster_id, clusters):
	for i in range(len(cluster_id)):
		for j in range(len(clusters)):
			if cluster_id[i] == clusters[j]:
				cluster_id[i] = j
	return cluster_id

@_njit 
def cluster_properties(cluster_id, n_clusters, frame):
	mean_frame = _np.zeros(n_clusters, dtype=_np.float32)
	n_locs_cluster = _np.zeros(n_clusters, dtype=_np.int32)
	locs_in_window = _np.zeros((n_clusters, 21), dtype=_np.int32)
	locs_perc = _np.zeros(n_clusters, dtype=_np.float32)
	window_search = frame[-1] / 20
	for j in range(n_clusters):
		for i in range(len(cluster_id)):
			if j == cluster_id[i]:
				n_locs_cluster[j] += 1
				mean_frame[j] += frame[i]
				locs_in_window[j][int(frame[i] / window_search)] += 1
	mean_frame = mean_frame / n_locs_cluster
	for i in range(n_clusters):
		for j in range(21):
			temp = locs_in_window[i][j] / n_locs_cluster[i]
			if temp > locs_perc[i]:
				locs_perc[i] = temp
	return mean_frame, locs_perc

@_njit
def find_true_clusters(mean_frame, locs_perc, frame):
	n_frame = _np.int32(_np.max(frame))
	true_cluster = _np.zeros(len(mean_frame), dtype=_np.int32)
	for i in range(len(mean_frame)):
		cond1 = locs_perc[i] < 0.8
		cond2 = mean_frame[i] < n_frame * 0.8
		cond3 = mean_frame[i] > n_frame * 0.2
		if cond1 and cond2 and cond3:
			true_cluster[i] = 1
	return true_cluster

def clusterer_picked(x, y, frame, eps, min_samples):
	xy = _np.stack((x, y)).T
	dist = _dm(xy, xy)
	n_neighbors = count_neighbors_picked(dist, eps)
	local_max = local_maxima_picked(dist, n_neighbors, eps)
	cluster_id = assign_to_cluster_picked(dist, eps, local_max)
	cluster_n_locs = _np.bincount(cluster_id)
	cluster_id = check_cluster_size(
		cluster_n_locs, min_samples, cluster_id
	)
	clusters = _np.unique(cluster_id)
	cluster_id = rename_clusters(cluster_id, clusters)
	n_clusters = len(clusters)
	mean_frame, locs_perc = cluster_properties(
		cluster_id, n_clusters, frame
	)
	true_cluster = find_true_clusters(mean_frame, locs_perc, frame)
	labels = -1 * _np.ones(len(x), dtype=_np.int32)
	for i in range(len(x)):
		if cluster_id[i] != 0 and true_cluster[cluster_id[i]] == 1:
			labels[i] = cluster_id[i] - 1
	return labels


@_cuda.jit
def count_neighbors_GPU(nn, radius, x, y):
	i = _cuda.grid(1)
	if i >= len(x):
		return

	temp_sum = 0
	for j in range(len(x)):
		# dist = _np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
		dist = get_distance(x[i], x[j], y[i], y[j])
		if dist <= radius:
			temp_sum += 1

	nn[i] = temp_sum - 1 # for case i == j

@_cuda.jit
def local_maxima_GPU(lm, nn, radius, x, y):
	i = _cuda.grid(1)
	if i >= len(x):
		return

	for j in range(len(x)):
		dist = get_distance(x[i], x[j], y[i], y[j])
		if dist <= radius and nn[i] >= nn[j]:
			lm[i] = 1
		if dist <= radius and nn[i] < nn[j]:
			lm[i] = 0
			break

# todo: if i thing must stay, maybe cpu version is better?
@_cuda.jit
def assign_to_cluster_GPU(i, radius, cluster_id, x, y):
	j = cuda.grid(1)
	if j >= len(x):
		return

	dist = get_distance(x[i], x[j], y[i], y[j])
	if dist <= radius:
		if cluster_id[i] != 0:
			if cluster_id[j] == 0:
				cluster_id[j] = cluster_id[i]
		if cluster_id[i] == 0:
			if j == 0:
				cluster_id[i] = i + 1
			cluster_id[j] = i + 1

# todo: check if check_min_size and true_cluster may be on cpu
@_cuda.jit
def check_cluster_size_GPU(cluster_n_locs, min_samples, cluster_id):
	i = _cuda.grid(1)
	if i >= len(cluster_id):
		return

	if cluster_n_locs[cluster_id[i]] <= min_samples:
		cluster_id[i] = 0

@_cuda.jit
def rename_clusters_GPU(cluster_id, clusters):
	i = _cuda.grid(1)
	if i >= len(cluster_id):
		return

	for j in range(len(clusters)):
		if cluster_id[i] == clusters[j]:
			cluster_id[i] = j

# todo: try this with just one function, similar to picked
@_cuda.jit
def cluster_properties_GPU1(
	cluster_id,
	clusters,
	com_x,
	com_y,
	x,
	y,
	frame,
	occ_locs_window,
	window_search,
	elements_per_cluster,
	mean_frame,
):
	j = _cuda.grid(1)
	if j >= len(clusters):
		return

	for i in range(len(x)):
		if cluster_id[i] == j:
			com_x[j] += x[i]
			com_y[j] += y[i]
			elements_per_cluster[j] += 1
			mean_frame[j] += frame[i]
			occ_locs_window[j][int(frame[i]/window_search)] += 1

@_cuda.jit
def cluster_properties_GPU2(
	com_x,
	com_y,
	occ_locs_window,
	elements_per_cluster,
	mean_frame,
	locs_perc,
):
	i = _cuda.grid(1)
	if i >= len(com_x):
		return

	com_x[i] /= elements_per_cluster[i]
	com_y[i] /= elements_per_cluster[i]
	mean_frame[i] /= elements_per_cluster[i]
	locs_perc[i] = 0

	for j in range(21):
		new_perc = occ_locs_window[i][j] / elements_per_cluster[i]
		if new_perc > locs_perc[i]:
			locs_perc[i] = new_perc

@_cuda.jit
def true_cluster_GPU(mean_frame, locs_perc, true_cluster, frame_nr):
	i = _cuda.grid(1)
	if i >= len(mean_frame):
		return

	cond1 = locs_perc[i] < 0.8
	cond2 = mean_frame[i] < frame_nr * 0.8
	cond3 = mean_frame[i] > frame_nr * 0.2
	if cond1 and cond2 and cond3:
		true_cluster[i] = 1

	if locs_perc[i] >= 0.8:
		true_cluster[i] = 0

def clusterer_GPU(x, y, frame, eps, min_samples):
	block = 32
	grid = len(x) // block + 1
	# todo: remember about moving to and from cpu and gpu + cuda.synchronize()
	### number of neighbors
	# move arrays to device
	d_x = _cuda.to_device(x)
	d_y = _cuda.to_device(y)
	d_n_neighbors = _cuda.to_device(_np.zeros(len(x), dtype=_np.int32))
	count_neighbors_GPU[grid, block](
		d_n_neighbors, radius, d_x, d_y
	)
	_cuda.synchronize()

	### local maxima
	d_local_max = _cuda.to_device(_np.zeros(len(x), dtype=_np.int32))
	local_maxima_GPU[grid, block](
		d_local_max, d_n_neighbors, radius, d_x, d_y
	)
	_cuda.synchronize()
	local_max = d_local_max.copy_to_host()

	### assign clusters
	d_cluster_id = _cuda.to_device(_np.zeros(len(x), dtype=_np.int32))
	for i in range(len(x)):
		if local_max[i]:
			assign_to_cluster_GPU[grid, block](
				i, radius, d_cluster_id, d_x, d_y
			)
			_cuda.synchronize()
	cluster_id = d_cluster_id.copy_to_host()

	### check cluster size
	cluster_n_locs = _np.bincount(cluster_id)
	d_cluster_n_locs = _cuda.to_device(cluster_n_locs)
	check_cluster_size_GPU[grid, block](
		d_cluster_n_locs, min_samples, d_cluster_id
	)
	_cuda.synchronize()

	### renaming cluster ids
	cluster_id = d_cluster_id.copy_to_host()
	clusters = _np.unique(cluster_id)
	d_clusters = _cuda.to_device(clusters)
	rename_clusters_GPU[grid, block](
		d_cluster_id, d_clusters
	)
	_cuda.synchronize()

	### cluster props 1
	n = len(clusters)
	d_frame = _cuda.to_device(frame)
	d_com_x = _cuda.to_device(_np.zeros(n, dtype=_np.float32))
	d_com_y = _cuda.to_device(_np.zeros(n, dtype=_np.float32))
	d_elements_per_cluster = _cuda.to_device(_np.zeros(n, dtype=_np.int32))
	d_mean_frame = _cuda.to_device(_np.zeros(n, dtype=_np.float32))
	window_search = frame[-1] / 20
	d_occ_locs_window = _cuda.to_device(
		_np.zeros((n ,21), dtype=_np.float32)
	)
	cluster_properties_GPU1[grid, block](
		d_cluster_id,
		d_clusters,
		d_com_x,
		d_com_y,
		d_x,
		d_y,
		d_frame,
		d_occ_locs_window,
		window_search,
		d_elements_per_cluster,
		d_mean_frame
	)
	_cuda.synchronize()

	### check cluster props 2
	d_locs_perc = _cuda.to_device(_np.zeros(n), dtype=_np.float32)
	cluster_properties_GPU2[grid, block](
		d_com_x, 
		d_com_y, 
		d_occ_locs_window, 
		d_elements_per_cluster,
		d_mean_frame,
		d_locs_perc,
	)
	_cuda.synchronize()

	### check for true clusters
	d_true_cluser = _cuda.to_device(_np.zeros(n), dtype=_np.int8)
	frame_nr = frame[-1]
	true_cluster_GPU[grid, block](
		d_mean_frame, d_locs_perc, d_true_cluser, frame_nr
	)
	_cuda.synchronize()

	### return labels
	cluster_id = d_cluster_id.copy_to_host()
	true_cluster = d_true_cluser.copy_to_host()
	labels = -1 * _np.ones(len(x), dtype=_np.int32)
	for i in range(len(x)):
		if cluster_id[i] != 0 and true_cluster[cluster_id[i]] == 1:
			labels[i] = cluster_id[i] - 1
	return labels