from scipy import optimize
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba
import multiprocessing
from concurrent import futures
import time


@numba.jit(nopython=True, nogil=True)
def gaussian(mu, sigma, grid):
    norm = 0.3989422804014327 / sigma
    return norm * np.exp(-0.5 * ((grid - mu) / sigma)**2)


'''
def integrated_gaussian(mu, sigma, grid):
    norm = 0.70710678118654757 / sigma   # sq_norm = sqrt(0.5/sigma**2)
    return 0.5 * (erf((grid - mu + 0.5) * norm) - erf((grid - mu - 0.5) * norm))
'''


@numba.jit(nopython=True, nogil=True)
def outer(a, b, size, model, n, bg):
    for i in range(size):
        for j in range(size):
            model[i, j] = n * a[i] * b[j] + bg


@numba.jit(nopython=True, nogil=True)
def compute_model(theta, grid, size, model_x, model_y, model):
    model_x[:] = gaussian(theta[0], theta[4], grid)    # sx and sy are wrong with integrated gaussian
    model_y[:] = gaussian(theta[1], theta[5], grid)
    outer(model_y, model_x, size, model, theta[2], theta[3])
    return model


@numba.jit(nopython=True, nogil=True)
def compute_residuals(theta, spot, grid, size, model_x, model_y, model, residuals):
    compute_model(theta, grid, size, model_x, model_y, model)
    residuals[:, :] = spot - model
    return residuals.flatten()


def fit_spot(spot):
    size = spot.shape[0]
    size_half = int(size / 2)
    grid = np.arange(-size_half, size_half + 1, dtype=np.float32)
    model_x = np.empty(size, dtype=np.float32)
    model_y = np.empty(size, dtype=np.float32)
    model = np.empty((size, size), dtype=np.float32)
    residuals = np.empty((size, size), dtype=np.float32)
    # theta is [x, y, photons, bg, sx, sy]
    theta0 = np.array([0, 0, np.sum(spot-spot.min()), spot.min(), 1, 1], dtype=np.float32)  # make it smarter
    args = (spot, grid, size, model_x, model_y, model, residuals)
    result = optimize.leastsq(compute_residuals, theta0, args=args, ftol=1e-2, xtol=1e-2)   # leastsq is much faster than least_squares
    '''
    model = compute_model(result[0], grid, size, model_x, model_y, model)
    plt.figure()
    plt.subplot(121)
    plt.imshow(spot, interpolation='none')
    plt.subplot(122)
    plt.imshow(model, interpolation='none')
    plt.colorbar()
    plt.show()
    '''
    return result[0]


def fit_spots(spots):
    theta = np.empty((len(spots), 6), dtype=np.float32)
    theta.fill(np.nan)
    for i, spot in enumerate(spots):
        theta[i] = fit_spot(spot)
    return theta


def fit_spots_parallel(spots, async=False):
    n_workers = int(0.75 * multiprocessing.cpu_count())
    n_spots = len(spots)
    n_tasks = 100 * n_workers
    spots_per_task = [int(n_spots / n_tasks + 1) if _ < n_spots % n_tasks else int(n_spots / n_tasks) for _ in range(n_tasks)]
    start_indices = np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = futures.ProcessPoolExecutor(n_workers)
    for i, n_spots_task in zip(start_indices, spots_per_task):
        fs.append(executor.submit(fit_spots, spots[i:i+n_spots_task]))
    if async:
        return fs
    with tqdm(total=n_tasks, unit='task') as progress_bar:
        for f in futures.as_completed(fs):
            progress_bar.update()
    return fits_from_parallel_results(fs)


def fits_from_futures(futures):
    theta = [_.result() for _ in futures]
    return np.vstack(theta)


def locs_from_fits(identifications, theta, box):
    # box_offset = int(box/2)
    x = theta[:, 0] + identifications.x     # - box_offset
    y = theta[:, 1] + identifications.y     # - box_offset
    lpy = theta[:, 4] / np.sqrt(theta[:, 2])
    lpx = theta[:, 5] / np.sqrt(theta[:, 2])
    locs = np.rec.array((identifications.frame, x, y,
                         theta[:, 2], theta[:, 4], theta[:, 5],
                         theta[:, 3], lpx, lpy,
                         identifications.net_gradient),
                        dtype=[('frame', 'u4'), ('x', 'f4'), ('y', 'f4'),
                               ('photons', 'f4'), ('sx', 'f4'), ('sy', 'f4'),
                               ('bg', 'f4'), ('lpx', 'f4'), ('lpy', 'f4'),
                               ('net_gradient', 'f4')])
    locs.sort(kind='mergesort', order='frame')
    return locs
