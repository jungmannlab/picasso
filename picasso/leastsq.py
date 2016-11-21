from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba


#@numba.jit(nopython=True)
def gaussian(mu, amp, sigma, bg, size):
    size_half = int(size / 2)
    grid = np.arange(-size_half, size_half + 1, dtype=np.float32)
    return amp * np.exp(-0.5*((grid-mu)/sigma)**2) + bg

#@numba.jit(nopython=True)
def outer(a, b):
    out = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            out[i, j] = a[i] * b[j]
    return out

#@numba.jit(nopython=False)
def compute_model(theta, size):
    model_x = gaussian(theta[0], theta[2], theta[3], theta[5], size)
    model_y = gaussian(theta[1], theta[2], theta[4], theta[5], size)
    return outer(np.sqrt(model_y), np.sqrt(model_x))

#@numba.jit(nopython=True)
def compute_residuals(theta, spot):
    print(theta[0:2])
    size = spot.shape[0]
    model = compute_model(theta, size)
    residuals = spot - model
    return residuals.flatten()


def leastsq(spots):
    sx = np.zeros(len(spots))
    sy = np.zeros(len(spots))
    for i, spot in enumerate(tqdm(spots)):
        # theta is [x, y, amp, sx, sy, bg]
        theta = np.array([0, 0, spot.ptp(), 1, 1, spot.min()])
        args = (spot,)
        result = optimize.least_squares(compute_residuals, theta, method='lm', args=args)
        plt.figure()
        plt.subplot(121)
        plt.imshow(spot, interpolation='none')
        plt.subplot(122)
        model = compute_model(result.x, spot.shape[0])
        plt.imshow(model, interpolation='none')
        plt.colorbar()
        plt.show()
        sx[i] = result.x[3]
        sy[i] = result.x[4]
    return sx, sy
