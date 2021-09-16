from enum import IntEnum
import numpy as np
import scipy.optimize



class Params2D(IntEnum):
    X = 0
    Y = 1
    I = 2
    BG = 3
    SIGMA = 4



def fit_sigma_2d(img, initial_sigma=1):
    W = img.shape[0]
    X, Y = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    img_sum = np.sum(img)
    momentX = np.sum(X * img)
    momentY = np.sum(Y * img)

    bg = np.min(img)
    I = img_sum - bg * W * W

    initialValue = [momentX / img_sum, momentY / img_sum, I, bg, initial_sigma]

    def logl(p, plot=False):
        # [x,y,I,bg,sigma]
        p = np.clip(p, [2, 2, 1, 1e-6, 0.1], [W - 2, W - 2, 1e9, 1e5, 10])

        t_x = p[Params2D.X]
        t_y = p[Params2D.Y]
        t_I = p[Params2D.I]
        t_bg = p[Params2D.BG]
        t_sigma = p[Params2D.SIGMA]

        Xexp0 = (X - t_x + 0.5) / (np.sqrt(2) * t_sigma)
        Xexp1 = (X - t_x - 0.5) / (np.sqrt(2) * t_sigma)
        Yexp0 = (Y - t_y + 0.5) / (np.sqrt(2) * t_sigma)
        Yexp1 = (Y - t_y - 0.5) / (np.sqrt(2) * t_sigma)

        DeltaX = 0.5 * (scipy.special.erf(Xexp0) - scipy.special.erf(Xexp1))
        DeltaY = 0.5 * (scipy.special.erf(Yexp0) - scipy.special.erf(Yexp1))

        mu = t_I * DeltaX * DeltaY + t_bg

        return 2.0 * np.sum(mu - img) - 2.0 * np.sum(img * np.log(mu / img))

    result = scipy.optimize.minimize(logl, initialValue, method="Nelder-Mead", options={"maxiter": 1000, "disp": False})

    return result.x