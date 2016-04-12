import numpy as _np
import numba as _numba
import math as _math


ITERATIONS = 20
MAX_STEP = _np.array([1e0, 1e0, 1e2, 2e0, 1e-1, 1e-1])
GAMMA = _np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
print('local!')


@_numba.jit(nopython=True, nogil=True)
def _center_of_mass(spot, size):
    x = 0.0
    y = 0.0
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            x += spot[i, j] * i
            y += spot[i, j] * j
            _sum_ += spot[i, j]
    x /= _sum_
    y /= _sum_
    return y, x


@_numba.jit(nopython=True, nogil=True)
def _filtered_pixel(spot, size, k, l, sigma):
    norm = 0.5 / sigma**2
    pixel = _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            exp = _np.exp(-norm * ((i - k - 2)**2 + (l - j - 2)**2))
            pixel += exp * spot[i, j]
            _sum_ += exp
    return pixel / _sum_


@_numba.jit(nopython=True, nogil=True)
def _filtered_min_max(spot, size, sigma):
    _min_ = _max_ = _filtered_pixel(spot, size, 0, 0, sigma)
    for k in range(size):
        for l in range(size):
            pixel = _filtered_pixel(spot, size, k, l, sigma)
            _max_ = _np.maximum(_max_, pixel)
            _min_ = _np.minimum(_min_, pixel)
    return _min_, _max_


@_numba.jit(nopython=True, nogil=True)
def centroid(spot, size, sigma):
    y, x = _center_of_mass(spot, size)
    bg, spot_max = _filtered_min_max(spot, size, sigma)
    photons = _np.maximum(0.0, (spot_max - bg) * 2 * _np.pi)
    return x, y, photons, bg


@_numba.vectorize(nopython=True)
def _erf(x):
    ax = _np.abs(x)
    if ax < 0.5:
        t = x*x
        top = ((((.771058495001320e-04*t-.133733772997339e-02)*t+.323076579225834e-01)*t+.479137145607681e-01)*t+.128379167095513e+00) + 1.0
        bot = ((.301048631703895e-02*t+.538971687740286e-01)*t+.375795757275549e+00)*t + 1.0
        return x * (top / bot)
    if ax < 4.0:
        top = ((((((-1.36864857382717e-07*ax+5.64195517478974e-01)*ax+7.21175825088309e+00)*ax+4.31622272220567e+01)*ax+1.52989285046940e+02)*ax+3.39320816734344e+02)*ax+4.51918953711873e+02)*ax + 3.00459261020162e+02
        bot = ((((((1.0*ax+1.27827273196294e+01)*ax+7.70001529352295e+01)*ax+2.77585444743988e+02)*ax+6.38980264465631e+02)*ax+9.31354094850610e+02)*ax+7.90950925327898e+02)*ax + 3.00459260956983e+02
        erf = 0.5 + (0.5 - _np.exp(-x * x) * top / bot)
        if x < 0.0:
            erf = -erf
        return erf
    if ax < 5.8:
        x2 = x*x
        t = 1.0 / x2
        top = (((2.10144126479064e+00*t+2.62370141675169e+01)*t+2.13688200555087e+01)*t+4.65807828718470e+00)*t + 2.82094791773523e-01
        bot = (((9.41537750555460e+01*t+1.87114811799590e+02)*t+9.90191814623914e+01)*t+1.80124575948747e+01)*t + 1.0
        erf = (.564189583547756e0 - top / (x2 * bot)) / ax
        erf = 0.5 + (0.5 - _np.exp(-x2) * erf)
        if x < 0.0:
            erf = -erf
        return erf
    return _np.sign(x)


@_numba.jit(nopython=True, nogil=True)
def _gaussian_integral(x, mu, sigma):
    sq_norm = 0.70710678118654757 / sigma       # sq_norm = sqrt(0.5/sigma**2)
    d = x - mu
    return 0.5 * (_math.erf((d + 0.5) * sq_norm) - _math.erf((d - 0.5) * sq_norm))


@_numba.jit(nopython=True, nogil=True)
def _derivative_gaussian_integral(x, mu, sigma, photons, PSFc):
    d = x - mu
    a = _np.exp(-0.5 * ((d + 0.5) / sigma)**2)
    b = _np.exp(-0.5 * ((d - 0.5) / sigma)**2)
    dudt = -photons * PSFc * (a - b) / (_np.sqrt(2.0 * _np.pi) * sigma)
    d2udt2 = -photons * ((d + 0.5) * a - (d - 0.5) * b) * PSFc / (_np.sqrt(2.0 * _np.pi) * sigma**3)
    return dudt, d2udt2


@_numba.jit(nopython=True, nogil=True)
def _derivative_gaussian_integral_sigma(x, mu, sigma, photons, PSFc):
    ax = _np.exp(-0.5 * ((x + 0.5 - mu) / sigma)**2)
    bx = _np.exp(-0.5 * ((x - 0.5 - mu) / sigma)**2)
    dudt = -photons * (ax * (x + 0.5 - mu) - bx * (x - 0.5 - mu)) * PSFc / (_np.sqrt(2.0 * _np.pi) * sigma**2)
    d2udt2 = -2.0 * dudt / sigma - photons * (ax * (x + 0.5 - mu)**3 - bx * (x - 0.5 - mu)**3) * PSFc / (_np.sqrt(2.0 * _np.pi) * sigma**5)
    return dudt, d2udt2


def gaussmle_cpu(spots):
    ''' A wrappper around the gufunc to hide a workaround for the fixed size return array. '''
    return _fit_gaussians_gufunc(spots, _np.zeros(6, dtype=_np.float32))


@_numba.guvectorize("void(float32[:,:],float32[:],float32[:])", "(n,n),(p)->(p)", nopython=True, target='parallel')
def _fit_gaussians_gufunc(spot, dummy, theta):
    '''
    Theta is [x, y, N, bg, Sx, Sy]
    Hence: x = theta[0]
           y = theta[1]
           N = theta[2]
           bg = theta[3]
           sx = theta[4]
           sy = theta[5]
    '''
    n_params = 6
    initial_sigma = 1.0
    size, _ = spot.shape

    # Initial values
    theta[0], theta[1], theta[2], theta[3] = centroid(spot, size, initial_sigma)
    theta[4] = theta[5] = initial_sigma

    # Memory allocation (we do that outside of the loops to avoid huge delays in threaded code):
    dudt = _np.zeros(n_params, dtype=_np.float32)
    d2udt2 = _np.zeros(n_params, dtype=_np.float32)
    numerator = _np.zeros(n_params, dtype=_np.float32)
    denominator = _np.zeros(n_params, dtype=_np.float32)

    for kk in range(ITERATIONS):

        numerator[:] = 0.0
        denominator[:] = 0.0

        for ii in range(size):
            for jj in range(size):
                PSFx = _gaussian_integral(ii, theta[0], theta[4])
                PSFy = _gaussian_integral(jj, theta[1], theta[5])

                # Derivatives
                dudt[0], d2udt2[0] = _derivative_gaussian_integral(ii, theta[0], theta[4], theta[2], PSFy)
                dudt[1], d2udt2[1] = _derivative_gaussian_integral(jj, theta[1], theta[5], theta[2], PSFx)
                dudt[2] = PSFx * PSFy
                d2udt2[2] = 0.0
                dudt[3] = 1.0
                d2udt2[3] = 0.0
                dudt[4], d2udt2[4] = _derivative_gaussian_integral_sigma(ii, theta[0], theta[4], theta[2], PSFy)
                dudt[5], d2udt2[5] = _derivative_gaussian_integral_sigma(jj, theta[1], theta[5], theta[2], PSFx)

                model = theta[2] * dudt[2] + theta[3]
                cf = df = 0.0
                pixel_value = spot[ii, jj]
                if model > 10e-3:
                    cf = pixel_value / model - 1
                    df = pixel_value / model**2
                cf = _np.minimum(cf, 10e4)
                df = _np.minimum(df, 10e4)

                for ll in range(n_params):
                    numerator[ll] = numerator[ll] + cf * dudt[ll]
                    denominator[ll] = denominator[ll] + cf * d2udt2[ll] - (dudt[ll]**2) * df

        # The update
        for ll in range(n_params):
            theta[ll] = theta[ll] - GAMMA[ll] * _np.minimum(_np.maximum(numerator[ll] / denominator[ll],
                                                                        -MAX_STEP[ll]),
                                                            MAX_STEP[ll])

        # Other constraints
        theta[2] = _np.maximum(theta[2], 1.0)
        theta[3] = _np.maximum(theta[3], 0.1)
        theta[4] = _np.maximum(theta[4], 0.1)
        theta[5] = _np.maximum(theta[5], 0.1)
