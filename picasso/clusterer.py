"""
	picasso.clusterer
	~~~~~~~~~~~~~~~~~

	Clusterer optimized for DNA PAINT in CPU and GPU versions.

	Based on the work of Susanne Reinhardt.
	:authors: Susanne Reinhardt, Rafal Kowalewski, 2020-2022
    :copyright: Copyright (c) 2022 Jungmann Lab, MPI of Biochemistry
"""

import os as _os

import numpy as _np
import math as _math
import yaml as _yaml
from scipy.spatial import distance_matrix as _dm
from numba import njit as _njit
from numba import cuda as _cuda
from tqdm import tqdm as _tqdm

@_njit 
def count_neighbors_picked(dist, radius):
	"""
	Calculates number of neighbors for each point within a given 
	radius.

	Used in clustering picked localizations.

	Parameters
	----------
	dist : np.array
		2D distance matrix
	radius : float
		Radius within which neighbors are counted

	Returns
	-------
	np.array
		1D array with number of neighbors for each point within radius
	"""

	nn = _np.zeros(dist.shape[0], dtype=_np.int32)
	for i in range(len(nn)):
		nn[i] = _np.where(dist[i] <= radius)[0].shape[0] - 1
	return nn

@_njit
def local_maxima_picked(dist, nn, radius):
	"""
	Finds which localizations are local maxima, i.e., localizations 
	with the highest number of neighbors within given radius.

	Used in clustering picked localizations.

	Parameters
	----------
	dist : np.array
		2D distance matrix
	nn : np.array
		1D array with number of neighbors for each localization
	radius : float
		Radius within which neighbors are counted	

	Returns
	-------
	np.array
		1D array with 1 if a localization is a local maximum, 
		0 otherwise	
	"""

	n = dist.shape[0]
	lm = _np.zeros(n, dtype=_np.int8)
	for i in range(n):
		for j in range(n):
			if dist[i][j] <= radius and nn[i] >= nn[j]:
				lm[i] = 1
			if dist[i][j] <= radius and nn[i] < nn[j]:
				lm[i] = 0
				break
	return lm

@_njit
def assign_to_cluster_picked(dist, lm, radius):
	"""
	Finds cluster id for each localization.

	If a localization is within radius from a local maximum, it is
	assigned to a cluster. Otherwise, it's id is 0.

	Used in clustering picked localizations.

	Parameters
	----------
	dist : np.array
		2D distance matrix
	lm : np.array
		1D array with local maxima
	radius : float
		Radius within which neighbors are counted	

	Returns
	-------
	np.array
		1D array with cluster id for each localization
	"""

	n = dist.shape[0]
	cluster_id = _np.zeros(n, dtype=_np.int32)
	for i in range(n):
		if lm[i]:
			for j in range(n):
				if dist[i][j] <= radius:
					if cluster_id[i] != 0:
						if cluster_id[j] == 0:
							cluster_id[j] = cluster_id[i]
					if cluster_id[i] == 0:
						if j == 0:
							cluster_id[i] = i + 1
						cluster_id[j] = i + 1
	return cluster_id

@_njit
def check_cluster_size(cluster_n_locs, min_locs, cluster_id):
	"""
	Filters clusters with too few localizations.

	Parameters
	----------
	cluster_n_locs : np.array
		Contains number of localizations for each cluster
	min_locs : int
		Minimum number of localizations to consider a cluster valid
	cluster_id : np.array
		Array with cluster id for each localization

	Returns
	-------
	np.array
		cluster_id after filtering
	"""

	for i in range(len(cluster_id)):
		if cluster_n_locs[cluster_id[i]] <= min_locs: # too few locs
			cluster_id[i] = 0 # id 0 means loc is not assigned to any cluster
	return cluster_id

@_njit
def rename_clusters(cluster_id, clusters):
	"""
	Reassign cluster ids after filtering (to make them consecutive)

	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization (after filtering)
	clusters : np.array
		Unique cluster ids

	Returns
	-------
	np.array
		Cluster ids with consecutive values
	"""

	for i in range(len(cluster_id)):
		for j in range(len(clusters)):
			if cluster_id[i] == clusters[j]:
				cluster_id[i] = j
	return cluster_id

@_njit 
def cluster_properties(cluster_id, n_clusters, frame):
	"""
	Finds cluster properties used in frame analysis.

	Returns mean frame and highest fraction of localizations within
	1/20th of whole acquisition time for each cluster.

	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization
	n_clusters : int
		Total number of clusters
	frame : np.array
		Frame number for each localization

	Returns
	-------
	np.array
		Mean frame for each cluster
	np.array
		Highest fraction of localizations within 1/20th of whole
		acquisition time.
	"""

	# mean frame for each cluster
	mean_frame = _np.zeros(n_clusters, dtype=_np.float32)
	# number of locs in each cluster
	n_locs_cluster = _np.zeros(n_clusters, dtype=_np.int32)
	# number of locs in each cluster in each time window (1/20th 
	# acquisition time)
	locs_in_window = _np.zeros((n_clusters, 21), dtype=_np.int32)
	# highest fraction of localizations within the time windows 
	# for each cluster
	locs_frac = _np.zeros(n_clusters, dtype=_np.float32)
	# length of the time window
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
			if temp > locs_frac[i]:
				locs_frac[i] = temp
	return mean_frame, locs_frac

@_njit
def find_true_clusters(mean_frame, locs_frac, n_frame):
	"""
	Performs basic frame analysis on clusters.

	Checks for "sticky events" by analyzing mean frame and the
	highest fraction of locs in 1/20th interval of acquisition time.

	Parameters
	----------
	mean_frame : np.array
		Contains mean frame for each cluster
	locs_frac : np.array
		Contains highest fraction of locs withing the time window
	n_frame : int
		Acquisition time given in frames

	Returns
	-------
	np.array
		1D array with 1 if a cluster passed the frame analysis, 
		0 otherwise
	"""

	true_cluster = _np.zeros(len(mean_frame), dtype=_np.int8)
	for i in range(len(mean_frame)):
		cond1 = locs_frac[i] < 0.8
		cond2 = mean_frame[i] < n_frame * 0.8
		cond3 = mean_frame[i] > n_frame * 0.2
		if cond1 and cond2 and cond3:
			true_cluster[i] = 1
	return true_cluster

def find_clusters_picked(dist, radius):
	"""
	Counts neighbors, finds local maxima and assigns cluster ids.

	Used in clustering picked localizations.

	Parameters
	----------
	dist : np.array
		2D distance matrix
	radius : float
		Radius within which neighbors are counted

	Returns
	-------
	np.array
		Cluster ids for each localization
	"""

	n_neighbors = count_neighbors_picked(dist, radius)
	local_max = local_maxima_picked(dist, n_neighbors, radius)
	cluster_id = assign_to_cluster_picked(dist, local_max, radius)
	return cluster_id	

def postprocess_clusters(cluster_id, min_locs, frame):
	"""
	Filters clusters for minimum number of localizations and performs 
	basic frame analysis to filter out "sticky events".

	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization (before filtering)
	min_locs : int
		Minimum number of localizations in a cluster
	frame : np.array
		Frame number for each localization

	Returns
	-------
	np.array
		Contains cluster id for each localization
	np.array
		Specifies if a given cluster passed the frame analysis

	"""
	cluster_n_locs = _np.bincount(cluster_id) # number of locs in each cluster
	cluster_id = check_cluster_size(cluster_n_locs, min_locs, cluster_id)
	clusters = _np.unique(cluster_id)
	cluster_id = rename_clusters(cluster_id, clusters)
	n_clusters = len(clusters)
	mean_frame, locs_frac = cluster_properties(
		cluster_id, n_clusters, frame
	)
	n_frame = _np.int32(_np.max(frame))
	true_cluster = find_true_clusters(mean_frame, locs_frac, n_frame)
	return cluster_id, true_cluster

def get_labels(cluster_id, true_cluster):
	"""
	Gives labels compatible with scikit-learn style, i.e., -1 means
	a point (localization) was not assigned to any cluster

	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization
	true_cluster : np.array
		Specifies if a given cluster passed the frame analysis

	Returns
	-------
	np.array
		Contains label for each localization
	"""

	labels = -1 * _np.ones(len(cluster_id), dtype=_np.int32)
	for i in range(len(cluster_id)):
		if cluster_id[i] != 0 and true_cluster[cluster_id[i]] == 1:
			labels[i] = cluster_id[i] - 1
	return labels

def clusterer_picked_2D(x, y, frame, radius, min_locs):
	"""
	Clusters picked localizations while storing distance matrix and 
	returns labels for each localization (2D).

	Works most efficiently if less than 700 locs are provided.

	Parameters
	----------
	x : np.array
		x coordinates of picked localizations
	y : np.array
		y coordinates of picked localizations
	frame : np.array
		Frame number for each localization
	radius : float
		Clustering radius
	min_locs : int
		Minimum number of localizations in a cluster

	Returns
	-------
	np.array
		Labels for each localization
	"""

	xy = _np.stack((x, y)).T
	dist = _dm(xy, xy) # calculate distance matrix
	cluster_id = find_clusters_picked(dist, radius)
	cluster_id, true_cluster = postprocess_clusters(
		cluster_id, min_locs, frame
	)
	return get_labels(cluster_id, true_cluster)

def clusterer_picked_3D(x, y, z, frame, radius_xy, radius_z, min_locs):
	"""
	Clusters picked localizations while storing distance matrix and 
	returns labels for each localization (3D).

	z coordinate is scaled to account for different clustering radius
	in z.

	Works most efficiently if less than 600 locs are provided.

	Parameters
	----------
	x : np.array
		x coordinates of picked localizations
	y : np.array
		y coordinates of picked localizations
	frame : np.array
		Frame number for each localization
	z : np.array
		z coordinates of picked localizations
	radius_xy : float
		Clustering radius in x and y directions
	radius_z : float
		Clustering radius in z direction
	min_locs : int
		Minimum number of localizations in a cluster

	Returns
	-------
	np.array
		Labels for each localization
	"""

	xyz = _np.stack((x, y, z * (radius_xy/radius_z))).T # scale z
	dist = _dm(xyz, xyz)
	cluster_id = find_clusters_picked(dist, radius_xy)
	cluster_id, true_cluster = postprocess_clusters(
		cluster_id, min_locs, frame
	)
	return get_labels(cluster_id, true_cluster)

#_____________________________________________________________________#
# Functions used for clustering all locs (not picked) using CPU:
#_____________________________________________________________________#

def get_d2_2D(x1, x2, y1, y2):
	"""
	Calculates squared distance between two points in 2D.

	Squaring is more time-efficient than taking a square root.

	Parameters
	----------
	x1, x2 : floats
		x coordinates of the two points
	y1, y2 : floats
		y coordinates of the two points

	Returns
	-------
	float
		Square distance between two points in 2D
	"""

	return (x2-x1) ** 2 + (y2-y1) ** 2

def get_d2_3D(x1, x2, y1, y2, z1, z2, r_rel):
	"""
	Calculates squared distance between two points in 3D.

	Scales z coordinates to account for different clustering radii in 
	xy and z dimensions.

	Squaring is more time-efficient than taking a square root.

	Parameters
	----------
	x1, x2 : floats
		x coordinates of the two points
	y1, y2 : floats
		y coordinates of the two points
	z1, z2 : floats
		z coordinates of the two points
	r_rel : float
		Clustering radius in xy divided by clustering radius in z

	Returns
	-------
	float
		Square distance between two points in 3D
	"""

	return (x2-x1) ** 2 + (y2-y1) ** 2 + (r_rel * (z2-z1)) ** 2

def assing_locs_to_boxes_2D(x, y, box_size):
	"""
	Splits FOV into boxes and assigns localizations to their
	corresponding boxes (2D).

	Localizations in boxes are overlapping, i.e., locs that are
	located in the neighboring boxes are also assigned to the given
	box.

	Parameters
	----------
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	box_size : float
		Size of box in the grid. It is recommended to be equal to, or 
		slightly larger than clustering radius

	Returns
	-------
	list
		Contains localizations assigned to each box (including 
		neighboring boxes)
	list
		Contains box id for each localization
	"""

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
		locs_id_box.append([]) # [[]] * (n_boxes+1) does not work

	# fill values for locs_id_box
	for i in range(len(x)):
		locs_id_box[box_id[i]].append(i)

		# add locs that are in the adjacent boxes
		if box_id[i] != n_boxes:
			locs_id_box[box_id[i]+1].append(i)
		if box_id[i] != 0:
			locs_id_box[box_id[i]-1].append(i)

		if box_id[i] >= n_boxes_x:
			locs_id_box[box_id[i]-n_boxes_x].append(i)
			locs_id_box[box_id[i]-n_boxes_x+1].append(i)
			locs_id_box[box_id[i]-n_boxes_x-1].append(i)

		if box_id[i] <= n_boxes - n_boxes_x:
			locs_id_box[box_id[i]+n_boxes_x].append(i)
			locs_id_box[box_id[i]+n_boxes_x+1].append(i)
			locs_id_box[box_id[i]+n_boxes_x-1].append(i)	

	return locs_id_box, box_id	

def assing_locs_to_boxes_3D(x, y, z, box_size_xy, box_size_z):
	"""
	Splits FOV into boxes and assigns localizations to their
	corresponding boxes (3D).

	Localizations in boxes are overlapping, i.e., locs that are
	located in the neighboring boxes are also assigned to the given
	box.

	Note that box sizes in xy and z can be different.

	Parameters
	----------
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	z : np.array
		z coordinates of localizations
	box_size_xy : float
		Size of box in the grid in xy
	box_size_z : float
		Size of box in the grid in z. It is recommended to be equal to,
		or slightly larger than clustering radius.

	Returns
	-------
	list
		Contains localizations' indeces assigned to each box (including
		neighboring boxes)
	list
		Contains box id for each localization
	"""

	# assigns each loc to a box
	box_id = _np.zeros(len(x), dtype=_np.int32)
	
	x_start = x.min()
	x_end = x.max()
	y_start = y.min()
	y_end = y.max()
	z_start = z.min()
	z_end = z.max()

	n_boxes_x = int((x_end - x_start) / box_size_xy) + 1
	n_boxes_y = int((y_end - y_start) / box_size_xy) + 1
	n_boxes_z = int((z_end - z_start) / box_size_z) + 1
	n_boxes = n_boxes_x * n_boxes_y * n_boxes_z

	for i in range(len(x)):
		box_id[i] = (
			int((x[i] - x_start) / box_size_xy)
			+ int((y[i] - y_start) / box_size_xy)
			* n_boxes_x
			+ int((z[i] - z_start) / box_size_z)
			* n_boxes_x * n_boxes_y
		)

	# gives indeces of locs in a given box
	locs_id_box = [[]]
	for i in range(n_boxes):
		locs_id_box.append([])

	# fill values for locs_id_box
	for i in range(len(x)):
		locs_id_box[box_id[i]].append(i)

		# add locs that are in the adjacent boxes
		if box_id[i] != n_boxes:
			locs_id_box[box_id[i]+1].append(i)
		if box_id[i] != 0:
			locs_id_box[box_id[i]-1].append(i)

		if box_id[i] >= n_boxes_x:
			locs_id_box[box_id[i]-n_boxes_x].append(i)
			locs_id_box[box_id[i]-n_boxes_x+1].append(i)
			locs_id_box[box_id[i]-n_boxes_x-1].append(i)

		if box_id[i] <= n_boxes - n_boxes_x:
			locs_id_box[box_id[i]+n_boxes_x].append(i)
			locs_id_box[box_id[i]+n_boxes_x+1].append(i)
			locs_id_box[box_id[i]+n_boxes_x-1].append(i)	

		if box_id[i] >= n_boxes_x * n_boxes_y:
			locs_id_box[box_id[i]-n_boxes_x*n_boxes_y].append(i)
			locs_id_box[box_id[i]-n_boxes_x*n_boxes_y+1].append(i)
			locs_id_box[box_id[i]-n_boxes_x*n_boxes_y-1].append(i)

			locs_id_box[box_id[i]-(n_boxes_x-1)*n_boxes_y].append(i)
			locs_id_box[box_id[i]-(n_boxes_x-1)*n_boxes_y+1].append(i)
			locs_id_box[box_id[i]-(n_boxes_x-1)*n_boxes_y-1].append(i)

			locs_id_box[box_id[i]-(n_boxes_x+1)*n_boxes_y].append(i)
			locs_id_box[box_id[i]-(n_boxes_x+1)*n_boxes_y+1].append(i)
			locs_id_box[box_id[i]-(n_boxes_x+1)*n_boxes_y-1].append(i)

		if box_id[i] <= n_boxes - n_boxes_x * n_boxes_y:
			locs_id_box[box_id[i]+n_boxes_x*n_boxes_y].append(i)
			locs_id_box[box_id[i]+n_boxes_x*n_boxes_y+1].append(i)
			locs_id_box[box_id[i]+n_boxes_x*n_boxes_y-1].append(i)

			locs_id_box[box_id[i]+(n_boxes_x-1)*n_boxes_y].append(i)
			locs_id_box[box_id[i]+(n_boxes_x-1)*n_boxes_y+1].append(i)
			locs_id_box[box_id[i]+(n_boxes_x-1)*n_boxes_y-1].append(i)

			locs_id_box[box_id[i]+(n_boxes_x+1)*n_boxes_y].append(i)
			locs_id_box[box_id[i]+(n_boxes_x+1)*n_boxes_y+1].append(i)
			locs_id_box[box_id[i]+(n_boxes_x+1)*n_boxes_y-1].append(i)

	return locs_id_box, box_id		

def count_neighbors_CPU_2D(locs_id_box, box_id, x, y, r2):
	"""
	Calculates number of neighbors for each point within a given
	radius (2D).

	Used in clustering all localizations with CPU.
	Calculates distance between points on-demand.

	Parameters
	----------
	locs_id_box : list
		Localizations' indeces assigned to a given box
	box_id : list
		Box id for a given localization
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	r2 : float
		Squared clustering radius

	Returns
	-------
	np.array
		Contains number of neighbors for each localization
	"""

	nn = _np.zeros(len(x), dtype=_np.int32) # number of neighbors
	for i in range(len(x)): # for each loc
		for j in locs_id_box[box_id[i]]: # for each other loc in the box
			d2 = get_d2_2D(x[i], x[j], y[i], y[j]) # squared distance
			if d2 <= r2:
				nn[i] += 1
		nn[i] -= 1 # subtract case i == j
	return nn

def count_neighbors_CPU_3D(locs_id_box, box_id, x, y, z, r2, r_rel):
	"""
	Calculates number of neighbors for each point within a given
	radius (3D).

	Used in clustering all localizations with CPU.
	Calculates distance between points on-demand.
	Z coordinate is scaled to account for different radius in z 
	direction.

	Parameters
	----------
	locs_id_box : list
		Localizations' indeces assigned to a given box
	box_id : list
		Box id for a given localization
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	z : np.array
		z coordinates of localizations
	r2 : float
		Squared clustering radius
	r_rel : float
		Clustering radius in xy divided by clustering radius in z

	Returns
	-------
	np.array
		Contains number of neighbors for each localization
	"""	

	nn = _np.zeros(len(x), dtype=_np.int32) # number of neighbors
	for i in range(len(x)): # for each loc
		for j in locs_id_box[box_id[i]]: # for each other loc in the box
			# squared distance (with scaled z)
			d2 = get_d2_3D(x[i], x[j], y[i], y[j], z[i], z[j], r_rel)
			if d2 <= r2:
				nn[i] += 1
		nn[i] -= 1 # subtract case i == j
	return nn	

def local_maxima_CPU_2D(locs_id_box, box_id, nn, x, y, r2):
	"""
	Finds which localizations are local maxima, i.e., localizations 
	with the highest number of neighbors within given radius (2D).

	Used in clustering all localizations with a CPU.
	Calculates distance between points on-demand.

	Parameters
	----------
	locs_id_box : list
		Localizations' indeces assigned to a given box
	box_id : list
		Box id for a given localization
	nn : np.array
		Number of neighbors for each localization
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	r2 : float
		Squared clustering radius

	Returns
	-------
	np.array
		1D array with 1 if a localization is a local maximum, 
		0 otherwise	
	"""

	lm = _np.zeros(len(x), dtype=_np.int8)
	for i in range(len(x)):
		for j in locs_id_box[box_id[i]]:
			d2 = get_d2_2D(x[i], x[j], y[i], y[j])
			if d2 <= r2 and nn[i] >= nn[j]:
				lm[i] = 1
			if d2 <= r2 and nn[i] < nn[j]:
				lm[i] = 0
				break
	return lm

def local_maxima_CPU_3D(locs_id_box, box_id, nn, x, y, z, r2, r_rel):
	"""
	Finds which localizations are local maxima, i.e., localizations 
	with the highest number of neighbors within given radius (3D).

	Used in clustering all localizations with a CPU.
	Calculates distance between points on-demand.
	Z coordinate is scaled to account for different radius in z 
	direction.

	Parameters
	----------
	locs_id_box : list
		Localizations' indeces assigned to a given box
	box_id : list
		Box id for a given localization
	nn : np.array
		Number of neighbors for each localization
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	z : np.array
		z coordinates of localizations
	r2 : float
		Squared clustering radius
	r_rel : float
		Clustering radius in xy divided by clustering radius in z

	Returns
	-------
	np.array
		1D array with 1 if a localization is a local maximum, 
		0 otherwise	
	"""

	lm = _np.zeros(len(x), dtype=_np.int8)
	for i in range(len(x)):
		for j in locs_id_box[box_id[i]]:
			d2 = get_d2_3D(x[i], x[j], y[i], y[j], z[i], z[j], r_rel)
			if d2 <= r2 and nn[i] >= nn[j]:
				lm[i] = 1
			if d2 <= r2 and nn[i] < nn[j]:
				lm[i] = 0
				break
	return lm	

def assign_to_cluster_CPU_2D(locs_id_box, box_id, nn, lm, x, y, r2):
	"""
	Finds cluster id for each localization (2D).

	If a localization is within radius from a local maximum, it is
	assigned to a cluster. Otherwise, it's id is 0.

	Used in clustering all localizations with a CPU.
	Calculates distance between points on-demand.

	Parameters
	----------
	locs_id_box : list
		Localizations' indeces assigned to a given box
	box_id : list
		Box id for a given localization
	nn : np.array
		Number of neighbors for each localization
	lm : np.array
		Specifies if a given localization is a local maximum
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	r2 : float
		Squared clustering radius

	Returns
	-------
	np.array
		1D array with cluster id for each localization
	"""

	cluster_id = _np.zeros(len(x), dtype=_np.int32)
	for i in range(len(x)):
		if lm[i]:
			for j in locs_id_box[box_id[i]]:
				d2 = get_d2_2D(x[i], x[j], y[i], y[j])
				if d2 <= r2:
					if cluster_id[i] != 0:
						if cluster_id[j] == 0:
							cluster_id[j] = cluster_id[i]
					else:
						if j == 0:
							cluster_id[i] = i + 1
						cluster_id[j] = i + 1
	return cluster_id

def assign_to_cluster_CPU_3D(
	locs_id_box, box_id, nn, lm, x, y, z, r2, r_rel
):
	"""
	Finds cluster id for each localization (3D).

	If a localization is within radius from a local maximum, it is
	assigned to a cluster. Otherwise, it's id is 0.

	Used in clustering all localizations with a CPU.
	Calculates distance between points on-demand.
	Z coordinate is scaled to account for different radius in z 
	direction.

	Parameters
	----------
	locs_id_box : list
		Localizations' indeces assigned to a given box
	box_id : list
		Box id for a given localization
	nn : np.array
		Number of neighbors for each localization
	lm : np.array
		Specifies if a given localization is a local maximum
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	z : np.array
		z coordinates of localizations
	r2 : float
		Squared clustering radius
	r_rel : float
		Clustering radius in xy divided by clustering radius in z

	Returns
	-------
	np.array
		1D array with cluster id for each localization
	"""

	cluster_id = _np.zeros(len(x), dtype=_np.int32)
	for i in range(len(x)):
		if lm[i]:
			for j in locs_id_box[box_id[i]]:
				d2 = get_d2_3D(x[i], x[j], y[i], y[j], z[i], z[j], r_rel)
				if d2 <= r2:
					if cluster_id[i] != 0:
						if cluster_id[j] == 0:
							cluster_id[j] = cluster_id[i]
					else:
						if j == 0:
							cluster_id[i] = i + 1
						cluster_id[j] = i + 1
	return cluster_id

def clusterer_CPU_2D(x, y, frame, radius, min_locs):
	"""
	Clusters all localizations using CPU (2D). Calculates distance 
	between points on-demand.

	Very similar to clusterer_picked_2D, except distance matrix is not
	stored and localizations are assigned to boxes.

	Parameters
	----------
	x : np.array
		x coordinates of picked localizations
	y : np.array
		y coordinates of picked localizations
	frame : np.array
		Frame number for each localization
	radius : float
		Clustering radius
	min_locs : int
		Minimum number of localizations in a cluster

	Returns
	-------
	np.array
		Labels for each localization
	"""

	locs_id_box, box_id = assing_locs_to_boxes_2D(x, y, radius*1.1)
	r2 = radius ** 2
	n_neighbors = count_neighbors_CPU_2D(locs_id_box, box_id, x, y, r2)
	local_max = local_maxima_CPU_2D(locs_id_box, box_id, n_neighbors, x, y, r2)
	cluster_id = assign_to_cluster_CPU_2D(
		locs_id_box, box_id, n_neighbors, local_max, x, y, r2
	)
	cluster_id, true_cluster = postprocess_clusters(
		cluster_id, min_locs, frame
	)
	return get_labels(cluster_id, true_cluster)

def clusterer_CPU_3D(x, y, z, frame, radius_xy, radius_z, min_locs):
	"""
	Clusters all localizations using CPU (3D). Calculates distance 
	between points on-demand.

	Very similar to clusterer_picked_3D, except distance matrix is not
	stored and localizations are assigned to boxes.

	When calculating distance, z coordinates are scaled, see get_d2_3D.

	Parameters
	----------
	x : np.array
		x coordinates of picked localizations
	y : np.array
		y coordinates of picked localizations
	frame : np.array
		Frame number for each localization
	z : np.array
		z coordinates of picked localizations
	radius_xy : float
		Clustering radius in x and y directions
	radius_z : float
		Clustering radius in z direction
	min_locs : int
		Minimum number of localizations in a cluster

	Returns
	-------
	np.array
		Labels for each localization
	"""

	locs_id_box, box_id = assing_locs_to_boxes_3D(
		x, y, z, radius_xy*1.1, radius_z*1.1
	)
	r2 = radius_xy ** 2
	r_rel = radius_xy / radius_z
	n_neighbors = count_neighbors_CPU_3D(
		locs_id_box, box_id, x, y, z, r2, r_rel
	)
	local_max = local_maxima_CPU_3D(
		locs_id_box, box_id, n_neighbors, x, y, z, r2, r_rel
	)
	cluster_id = assign_to_cluster_CPU_3D(
		locs_id_box, box_id, n_neighbors, local_max, x, y, z, r2, r_rel
	)
	cluster_id, true_cluster = postprocess_clusters(
		cluster_id, min_locs, frame
	)
	return get_labels(cluster_id, true_cluster)

#_____________________________________________________________________#
# Functions used for clustering all locs (not picked) using GPU:
#_____________________________________________________________________#

@_cuda.jit
def count_neighbors_GPU_2D(nn, r2, x, y):
	"""
	Calculates number of neighbors for each point within a given
	radius (2D).

	Used in clustering all localizations with a GPU.
	Calculates distance between points on-demand.

	Parameters
	----------
	nn : np.array
		Array to store number of neighbors for each localization
	r2 : float
		Squared clustering radius
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	"""

	i = _cuda.grid(1) # this works like iterating through all locs
	if i >= len(x):
		return

	temp_sum = 0 # used for storing number of neighbors
	for j in range(len(x)): # iterate over all other localizations
		d2 = (x[i]-x[j]) ** 2 + (y[i]-y[j]) ** 2 # squared distance
		if d2 <= r2:
			temp_sum += 1

	nn[i] = temp_sum - 1 # subtract case i == j

@_cuda.jit
def count_neighbors_GPU_3D(nn, r2, r_rel, x, y, z):
	"""
	Calculates number of neighbors for each point within a given
	radius (3D).

	Used in clustering all localizations with a GPU.
	Calculates distance between points on-demand.
	Distance in z is scaled according to clustering radii in xy and z.

	Parameters
	----------
	nn : np.array
		Array to store number of neighbors for each localization
	r2 : float
		Squared clustering radius
	r_rel : float
		Clustering radius in xy divided by clustering radius in z
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	z : np.array
		z coordinates of localizations
	"""

	i = _cuda.grid(1)
	if i >= len(x):
		return

	temp_sum = 0
	for j in range(len(x)):
		# squared distance scaled in z
		d2 = (x[i]-x[j]) ** 2 + (y[i]-y[j]) ** 2 + (r_rel * (z[i]-z[j])) ** 2
		if d2 <= r2:
			temp_sum += 1

	nn[i] = temp_sum - 1 # subtract case i == j

@_cuda.jit
def local_maxima_GPU_2D(lm, nn, r2, x, y):
	"""
	Finds which localizations are local maxima, i.e., localizations 
	with the highest number of neighbors within given radius (2D).

	Used in clustering all localizations with a GPU.
	Calculates distance between points on-demand.

	Parameters
	----------
	lm : np.array
		Array to store local maxima
	nn : np.array
		Number of neighbors for each localization
	r2 : float
		Squared clustering radius
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	"""

	i = _cuda.grid(1)
	if i >= len(x):
		return

	for j in range(len(x)):
		d2 = (x[i]-x[j]) ** 2 + (y[i]-y[j]) ** 2
		if d2 <= r2 and nn[i] >= nn[j]:
			lm[i] = 1
		if d2 <= r2 and nn[i] < nn[j]:
			lm[i] = 0
			break

@_cuda.jit
def local_maxima_GPU_3D(lm, nn, r2, r_rel, x, y, z):
	"""
	Finds which localizations are local maxima, i.e., localizations 
	with the highest number of neighbors within given radius (3D).

	Used in clustering all localizations with a GPU.
	Calculates distance between points on-demand.

	Parameters
	----------
	lm : np.array
		Array to store local maxima
	nn : np.array
		Number of neighbors for each localization
	r2 : float
		Squared clustering radius
	r_rel : float
		Clustering radius in xy divided by clustering radius in z
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	z : np.array
		z coordinates of localizations
	"""

	i = _cuda.grid(1)
	if i >= len(x):
		return

	for j in range(len(x)):
		d2 = (x[i]-x[j]) ** 2 + (y[i]-y[j]) ** 2 + (r_rel * (z[i]-z[j])) ** 2
		if d2 <= r2 and nn[i] >= nn[j]:
			lm[i] = 1
		if d2 <= r2 and nn[i] < nn[j]:
			lm[i] = 0
			break

@_cuda.jit
def assign_to_cluster_GPU_2D(cluster_id, i, r2, x, y):
	"""
	Finds cluster id for each localization (2D).

	Called only when comparing to a local maximum, see 
	clusterer_GPU_2D.

	Used in clustering all localizations with a GPU.
	Calculates distance between points on-demand.

	Parameters
	----------
	cluster_id : np.array
		Array to store cluster ids for each localization
	i : int
		Index of a localization that is a local maximum
	r2 : float
		Squared clustering radius
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	"""

	j = _cuda.grid(1)
	if j >= len(x):
		return

	d2 = (x[i]-x[j]) ** 2 + (y[i]-y[j]) ** 2
	if d2 <= r2:
		if cluster_id[i] != 0:
			if cluster_id[j] == 0:
				cluster_id[j] = cluster_id[i]
		if cluster_id[i] == 0:
			if j == 0:
				cluster_id[i] = i + 1
			cluster_id[j] = i + 1

@_cuda.jit
def assign_to_cluster_GPU_3D(cluster_id, i, r2, r_rel, x, y, z):
	"""
	Finds cluster id for each localization (3D).

	Called only when comparing to a local maximum, see 
	clusterer_GPU_3D.

	Used in clustering all localizations with a GPU.
	Calculates distance between points on-demand.

	Parameters
	----------
	cluster_id : np.array
		Array to store cluster ids for each localization
	i : int
		Index of a localization that is a local maximum
	r2 : float
		Squared clustering radius
	r_rel : float
		Clustering radius in xy divided by clustering radius in z
	x : np.array
		x coordinates of localizations
	y : np.array
		y coordinates of localizations
	z : np.array
		z coordinates of localizations
	"""

	j = _cuda.grid(1)
	if j >= len(x):
		return

	d2 = (x[i]-x[j]) ** 2 + (y[i]-y[j]) ** 2 + (r_rel * (z[i]-z[j])) ** 2
	if d2 <= r2:
		if cluster_id[i] != 0:
			if cluster_id[j] == 0:
				cluster_id[j] = cluster_id[i]
		if cluster_id[i] == 0:
			if j == 0:
				cluster_id[i] = i + 1
			cluster_id[j] = i + 1

@_cuda.jit
def rename_clusters_GPU(cluster_id, clusters):
	"""
	Reassign cluster ids after filtering (to make them consecutive).

	Used in clustering all localizations with a GPU.	

	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization (after filtering)
	clusters : np.array
		Unique cluster ids
	"""

	i = _cuda.grid(1)
	if i >= len(cluster_id):
		return

	for j in range(len(clusters)):
		if cluster_id[i] == clusters[j]:
			cluster_id[i] = j

@_cuda.jit
def cluster_properties_GPU1(
	cluster_id,
	n_clusters,
	frame,
	n_locs_cluster,
	mean_frame,
	locs_in_window,
):
	"""
	Calculates cluster properties used in frame analysis: 
		* number of localizations in cluster
		* mean frame of localizations in cluster
		* number of localizations divided into 20 equal time windows

	Used in clustering all localizations with a GPU.	

	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization
	n_clusters : np.array
		Total number of clusters
	frame : np.array
		Frame number for each localization
	n_locs_cluster : np.array
		Array to store number of localizations in each cluster
	mean_frame : np.array
		Array to store mean frame of localizations in each cluster
	locs_in_window : np.array
		Array to store number of locs split into 20 equal time windows
		in each cluster
	"""

	window_search = frame[-1] / 20 # length of 1/20th of acquisition time
	j = _cuda.grid(1)
	if j >= n_clusters:
		return

	for i in range(len(cluster_id)):
		if cluster_id[i] == j:
			n_locs_cluster[j] += 1
			mean_frame[j] += frame[i]
			locs_in_window[j][int(frame[i]/window_search)] += 1

	mean_frame[j] /= n_locs_cluster[j]

@_cuda.jit
def cluster_properties_GPU2(
	locs_in_window,
	n_locs_cluster,
	locs_frac,
):
	"""
	Calculates highest fraction of localizations within a single time 
	window for each cluster.

	To be used in frame analysis.
	Used in clustering all localizations with a GPU.	

	Parameters
	----------
	locs_in_window : np.array
		Number of locs in each cluster split into 20 equal time windows
	n_locs_cluster : np.array
		Number of localizations in each cluster
	locs_frac : np.array
		Array to store the highest fraction in each cluster
	"""

	i = _cuda.grid(1)
	if i >= len(locs_in_window):
		return

	for j in range(21):
		temp = locs_in_window[i][j] / n_locs_cluster[i]
		if temp > locs_frac[i]:
			locs_frac[i] = temp

def postprocess_clusters_GPU(cluster_id, min_locs, frame):
	"""
	Filters clusters for minimum number of localizations and performs 
	basic frame analysis to filter out "sticky events".

	Used in clustering all localizations with a GPU.

	Parameters
	----------
	cluster_id : np.array
		Contains cluster id for each localization (before filtering)
	min_locs : int
		Minimum number of localizations in a cluster
	frame : np.array
		Frame number for each localization

	Returns
	-------
	np.array
		Contains cluster id for each localization
	np.array
		Specifies if a given cluster passed the frame analysis	
	"""

	# values for initiating GPU functions
	block = 32
	grid_x = len(cluster_id) // block + 1

	### check cluster size
	cluster_n_locs = _np.bincount(cluster_id)
	cluster_id = check_cluster_size(cluster_n_locs, min_locs, cluster_id)

	### renaming cluster ids
	# move arrays to a gpu
	d_cluster_id = _cuda.to_device(cluster_id)
	clusters = _np.unique(cluster_id)
	d_clusters = _cuda.to_device(clusters)
	rename_clusters_GPU[grid_x, block](
		d_cluster_id, d_clusters
	)
	_cuda.synchronize()

	### cluster props 1
	# move arrays to a gpu
	n = len(clusters)
	d_frame = _cuda.to_device(frame)
	d_n_locs_cluster = _cuda.to_device(_np.zeros(n, dtype=_np.int32))
	d_mean_frame = _cuda.to_device(_np.zeros(n, dtype=_np.float32))
	d_locs_in_window = _cuda.to_device(
		_np.zeros((n, 21), dtype=_np.float32)
	)
	grid_n = len(clusters) // block + 1
	cluster_properties_GPU1[grid_n, block](
		d_cluster_id,
		n,
		d_frame,
		d_n_locs_cluster,
		d_mean_frame,
		d_locs_in_window,
	)
	_cuda.synchronize()

	### check cluster props 2
	# move array to a gpu
	d_locs_frac = _cuda.to_device(_np.zeros(n, dtype=_np.float32))
	cluster_properties_GPU2[grid_n, block](
		d_locs_in_window, 
		d_n_locs_cluster,
		d_locs_frac,
	)
	_cuda.synchronize()

	### check for true clusters
	# move arrays back to a cpu
	mean_frame = d_mean_frame.copy_to_host()
	locs_frac = d_locs_frac.copy_to_host()
	n_frame = _np.int32(_np.max(frame))
	true_cluster = find_true_clusters(mean_frame, locs_frac, n_frame)

	### return labels
	cluster_id = d_cluster_id.copy_to_host()
	return cluster_id, true_cluster	

def clusterer_GPU_2D(x, y, frame, radius, min_locs):
	"""
	Clusters all localizations using GPU (2D). Calculates distance 
	between points on-demand.

	Very similar to clusterer_CPU_2D, except most functions are 
	modified to run on a GPU.

	Parameters
	----------
	x : np.array
		x coordinates of picked localizations
	y : np.array
		y coordinates of picked localizations
	frame : np.array
		Frame number for each localization
	radius : float
		Clustering radius
	min_locs : int
		Minimum number of localizations in a cluster

	Returns
	-------
	np.array
		Labels for each localization
	"""

	# cuda does not accept noncontiguous arrays
	x = _np.ascontiguousarray(x, dtype=_np.float32)
	y = _np.ascontiguousarray(y, dtype=_np.float32)
	frame = _np.ascontiguousarray(frame, dtype=_np.float32)
	r2 = radius ** 2

	# values for initiating GPU functions
	block = 32
	grid_x = len(x) // block + 1

	### number of neighbors
	# move arrays to a gpu
	d_x = _cuda.to_device(x)
	d_y = _cuda.to_device(y)
	d_n_neighbors = _cuda.to_device(_np.zeros(len(x), dtype=_np.int32))
	count_neighbors_GPU_2D[grid_x, block](
		d_n_neighbors, r2, d_x, d_y
	)
	_cuda.synchronize()

	### local maxima
	# move array to a gpu
	d_local_max = _cuda.to_device(_np.zeros(len(x), dtype=_np.int32))
	local_maxima_GPU_2D[grid_x, block](
		d_local_max, d_n_neighbors, r2, d_x, d_y
	)
	_cuda.synchronize()
	# move array back to a cpu
	local_max = d_local_max.copy_to_host()

	### assign clusters
	# move array to a gpu
	d_cluster_id = _cuda.to_device(_np.zeros(len(x), dtype=_np.int32))
	# the loop below must be outside GPU to not allow for cross-talk 
	# between threads
	for i in range(len(x)): 
		if local_max[i]:
			assign_to_cluster_GPU_2D[grid_x, block](
				d_cluster_id, i, r2, d_x, d_y
			)
			_cuda.synchronize()

	# move array back to a cpu
	cluster_id = d_cluster_id.copy_to_host()
	### postprocess clusters
	cluster_id, true_cluster = postprocess_clusters_GPU(
		cluster_id, min_locs, frame
	)
	return get_labels(cluster_id, true_cluster)

def clusterer_GPU_3D(x, y, z, frame, radius_xy, radius_z, min_locs):
	"""
	Clusters all localizations using GPU (3D). Calculates distance 
	between points on-demand.

	Very similar to clusterer_CPU_3D, except most functions are 
	modified to run on a GPU.

	Parameters
	----------
	x : np.array
		x coordinates of picked localizations
	y : np.array
		y coordinates of picked localizations
	frame : np.array
		Frame number for each localization
	radius : float
		Clustering radius
	min_locs : int
		Minimum number of localizations in a cluster

	Returns
	-------
	np.array
		Labels for each localization
	"""

	# cuda does not accept noncontiguous arrays
	x = _np.ascontiguousarray(x)
	y = _np.ascontiguousarray(y)
	z = _np.ascontiguousarray(z)
	frame = _np.ascontiguousarray(frame)
	r2 = radius_xy ** 2
	r_rel = radius_xy / radius_z

	# values for initiating GPU functions
	block = 32
	grid_x = len(x) // block + 1

	### number of neighbors
	# move arrays to a gpu
	d_x = _cuda.to_device(x)
	d_y = _cuda.to_device(y)
	d_z = _cuda.to_device(z)
	d_n_neighbors = _cuda.to_device(_np.zeros(len(x), dtype=_np.int32))
	count_neighbors_GPU_3D[grid_x, block](
		d_n_neighbors, r2, r_rel, d_x, d_y, d_z
	)
	_cuda.synchronize()

	### local maxima
	# move array to a gpu
	d_local_max = _cuda.to_device(_np.zeros(len(x), dtype=_np.int8))
	local_maxima_GPU_3D[grid_x, block](
		d_local_max, d_n_neighbors, r2, r_rel, d_x, d_y, d_z
	)
	_cuda.synchronize()
	# move array back to a cpu
	local_max = d_local_max.copy_to_host()

	### assign clusters
	# move array to a gpu
	d_cluster_id = _cuda.to_device(_np.zeros(len(x), dtype=_np.int32))
	# the loop below must be outside GPU to not allow for cross-talk 
	# between threads
	for i in range(len(x)):
		if local_max[i]:
			assign_to_cluster_GPU_3D[grid_x, block](
				d_cluster_id, i, r2, r_rel, d_x, d_y, d_z
			)
			_cuda.synchronize()
	# move array back to a cpu
	cluster_id = d_cluster_id.copy_to_host()
	### postprocess clusters
	cluster_id, true_cluster = postprocess_clusters_GPU(
		cluster_id, min_locs, frame
	)
	return get_labels(cluster_id, true_cluster)
