from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba


#@numba.jit(nopython=True)
def gaussian(mu, amp, sigma, bg, grid, out):
    out[:] = amp * np.exp(-0.5*((grid-mu)/sigma)**2) + bg

#@numba.jit(nopython=True)
def outer(a, b, out):
    len_a = len(a)
    len_b = len(b)
    for i in range(len_a):
        for j in range(len_b):
            out[i, j] = a[i] * b[j]

#@numba.jit(nopython=False)
def compute_model(theta, model_x, model_y, model, grid):
    gaussian(theta[0], theta[2], theta[3], theta[5], grid, model_x)
    gaussian(theta[1], theta[2], theta[4], theta[5], grid, model_y)
    outer(np.sqrt(model_y), np.sqrt(model_x), model)

#@numba.jit(nopython=True)
def compute_residuals(theta, spot, residuals, model_x, model_y, model, grid):
    compute_model(theta, model_x, model_y, model, grid)
    residuals[:, :] = spot - model
    return residuals.flatten()


class LeastSquares:

    def __init__(self, spots):
        self.spots = spots
        # theta is [x, y, amp, sx, sy, bg]
        self.size = spots.shape[1]
        self.size_half = int(self.size / 2)
        self.grid = np.arange(-self.size_half, self.size_half + 1, dtype=np.float32)
        self.model_x = np.zeros(self.size, dtype=np.float32)
        self.model_y = np.zeros(self.size, dtype=np.float32)
        self.model = np.zeros((self.size, self.size), dtype=np.float32)
        self.residuals = np.zeros((self.size, self.size), dtype=np.float32)

    def fit(self):
        sx = np.zeros(len(self.spots))
        sy = np.zeros(len(self.spots))
        for i, spot in enumerate(tqdm(self.spots)):
            initial_guess = np.array([0, 0, spot.ptp(), 1, 1, spot.min()])
            args = (spot, self.residuals, self.model_x, self.model_y, self.model, self.grid)
            result = optimize.least_squares(compute_residuals, initial_guess, method='lm', args=args)
            plt.figure()
            plt.subplot(121)
            plt.imshow(spot, interpolation='none')
            plt.subplot(122)
            compute_model(result.x, self.model_x, self.model_y, self.model, self.grid)
            plt.imshow(self.model, interpolation='none')
            plt.show()
            sx[i] = result.x[3]
            sy[i] = result.x[4]
        return sx, sy
