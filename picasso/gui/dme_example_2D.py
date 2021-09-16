# -*- coding: utf-8 -*-
"""
3D drift estimation example
Units for X,Y,Z are pixels, pixels, and microns resp.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(sys.path[0] + "\\dme")

from dme import dme_estimate
from dme import rcc3D
from dme import NativeAPI


# Need to have CUDA installed
use_cuda=False


# Simulate an SMLM dataset in 3D with blinking molecules
def smlm_simulation(
        drift_trace,
        fov_width, # field of view size in pixels
        loc_error, # localization error XYZ 
        n_sites, # number of locations where molecules blink on and off
        n_frames,
        on_prob = 0.1, # probability of a binding site generating a localization in a frame
        ): 
    
    """
    localization error is set to 20nm XY and 50nm Z precision 
    (assumping Z coordinates are in um and XY are in pixels)
    """

    # typical 2D acquisition with small Z range and large XY range        
    binding_sites = np.random.uniform([0,0], [fov_width,fov_width], size=(n_sites,2))
    
    localizations = []
    framenum = []
    
    for i in range(n_frames):
        on = np.random.binomial(1, on_prob, size=n_sites).astype(np.bool)
        locs = binding_sites[on]*1
        # add localization error
        locs += drift_trace[i] + np.random.normal(0, loc_error, size=locs.shape)
        framenum.append(np.ones(len(locs),dtype=np.int32)*i)
        localizations.append(locs)
        
    return np.concatenate(localizations), np.concatenate(framenum)

n_frames = 2000
fov_width = 200
drift_mean = (0.001,0,0)
drift_stdev = (0.02,0.02,0.02)
loc_error = np.array((0.1,0.1,0.03)) # pixel, pixel, um
# Ground truth drift trace
drift_trace = np.cumsum(np.random.normal(drift_mean, drift_stdev, size=(n_frames,2)), 0)
drift_trace -= drift_trace.mean(0)


localizations, framenum = smlm_simulation(drift_trace, fov_width, loc_error, 
                                          n_sites=200,
                                          n_frames=n_frames)
print(f"Total localizations: {len(localizations)}")

crlb = np.ones(localizations.shape) * np.array(loc_error)[None]

estimated_drift,_ = dme_estimate(localizations, framenum, 
             crlb, 
             framesperbin = 1,  # note that small frames per bin use many more iterations
             imgshape=[fov_width, fov_width], 
             coarseFramesPerBin=200,
             coarseSigma=[0.2,0.2,0.2],  # run a coarse drift correction with large Z sigma
             useCuda=use_cuda,
             useDebugLibrary=False)

estimated_drift_rcc = rcc3D(localizations, framenum, timebins=10, zoom=1)


rmsd = np.sqrt(np.mean((estimated_drift-drift_trace)**2, 0))
print(f"RMSD of drift estimate compared to true drift: {rmsd}")

fig,ax=plt.subplots(3, figsize=(7,6))
for i in range(3):
    ax[i].plot(drift_trace[:,i],label='True drift')
    ax[i].plot(estimated_drift[:,i]+0.2,label='Estimated drift (DME)')
    ax[i].plot(estimated_drift_rcc[:,i]-0.2,label='Estimated drift (RCC)')
    ax[i].set_title(['x', 'y', 'z'][i])

    unit = ['px', 'px', 'um'][i]
    ax[i].set_ylabel(f'Drift [{unit}]')
ax[0].legend()
plt.tight_layout()
plt.show()
