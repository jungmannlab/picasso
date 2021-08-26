# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline


from fit_gauss_2D import fit_sigma_2d
from native_api import NativeAPI



def crosscorrelation(A, B):
    A_fft = np.fft.fft2(A)
    B_fft = np.fft.fft2(B)
    return np.fft.ifft2(A_fft * np.conj(B_fft))


def findshift(cc, plot=False):
    # look for the peak in a small subsection
    r = 6
    hw = 20
    cc_middle = cc[cc.shape[0] // 2 - hw : cc.shape[0] // 2 + hw, cc.shape[1] // 2 - hw : cc.shape[1] // 2 + hw]
    peak = np.array(np.unravel_index(np.argmax(cc_middle), cc_middle.shape))
    peak += [cc.shape[0] // 2 - hw, cc.shape[1] // 2 - hw]
    
    peak = np.clip(peak, r, np.array(cc.shape) - r)
    roi = cc[peak[0] - r + 1 : peak[0] + r, peak[1] - r + 1 : peak[1] + r]
    if plot:
        plt.figure()
        plt.imshow(cc_middle)
        plt.figure()
        plt.imshow(roi)

    px,py = fit_sigma_2d(roi, initial_sigma=2)[[0, 1]]
    return (peak[1] + px - r + 1 - cc.shape[1] / 2), (peak[0] + py - r + 1 - cc.shape[0] / 2) 


def findshift_pairs(images, pairs):
    print(f"RCC: Computing image cross correlations. Image stack shape: {images.shape}. Size: {images.size*4//1024//1024} MB",flush=True)
    w = images.shape[-1]
    if False:
        fft_images = np.fft.fft2(images)
        fft_conv = np.zeros((len(pairs), w, w),dtype=np.complex64)
        for i, (a,b) in enumerate(pairs):
            fft_conv[i] = np.conj(fft_images[a]) * fft_images[b]
            
        cc =  np.fft.ifft2(fft_conv)
        cc = np.abs(np.fft.fftshift(cc, (-2, -1)))

        shift = np.zeros((len(pairs),2))
        for i in tqdm.trange(len(pairs)):
            shift[i] = findshift(cc[i])
    else:
        fft_images = np.fft.fft2(images)
        shift = np.zeros((len(pairs),2))
        # low memory use version
        for i, (a,b) in tqdm.tqdm(enumerate(pairs),total=len(pairs)):
            fft_conv = np.conj(fft_images[a]) * fft_images[b]
            
            cc =  np.fft.ifft2(fft_conv)
            cc = np.abs(np.fft.fftshift(cc))
            shift[i] = findshift(cc)
    
    return shift

def rcc(xy, framenum, timebins, dll: NativeAPI, zoom=1, 
        sigma=1, maxpairs=1000):
    
    rendersize = int(np.max(xy))
    area = np.array([rendersize,rendersize])
    nframes = np.max(framenum)+1
    framesperbin = nframes/timebins
    
    imgshape = area*zoom
    images = np.zeros((timebins, *imgshape))

    for k in range(timebins):
        img = np.zeros(imgshape,dtype=np.float32)
        
        indices = np.nonzero((0.5 + framenum/framesperbin).astype(int)==k)[0]

        spots = np.zeros((len(indices), 5), dtype=np.float32)
        spots[:, 0] = xy[indices,0] * zoom
        spots[:, 1] = xy[indices,1] * zoom
        spots[:, 2] = sigma
        spots[:, 3] = sigma
        spots[:, 4] = 1
        
        if len(spots) == 0:
            raise ValueError(f'no spots in bin {k}')

        images[k] = dll.DrawGaussians(img, spots)       
    
    #print(f"RCC pairs: {timebins*(timebins-1)//2}. Bins={timebins}")
    pairs = np.array(np.triu_indices(timebins,1)).T
    if len(pairs)>maxpairs:
        pairs = pairs[np.random.choice(len(pairs),maxpairs)]
    pair_shifts = findshift_pairs(images, pairs)
    
    A = np.zeros((len(pairs),timebins))
    A[np.arange(len(pairs)),pairs[:,0]] = 1
    A[np.arange(len(pairs)),pairs[:,1]] = -1
    
    inv = np.linalg.pinv(A)
    shift_x = inv @ pair_shifts[:,0]
    shift_y = inv @ pair_shifts[:,1]
    shift_y -= shift_y[0]
    shift_x -= shift_x[0]
    shift = -np.vstack((shift_x,shift_y)).T / zoom
        
    t = (0.5+np.arange(timebins))*framesperbin
    
    shift -= np.mean(shift,0)

    shift_estim = np.zeros((len(shift),3))
    shift_estim[:,[0,1]] = shift
    shift_estim[:,2] = t

    if timebins != nframes:
        spl_x = InterpolatedUnivariateSpline(t, shift[:,0], k=2)
        spl_y = InterpolatedUnivariateSpline(t, shift[:,1], k=2)
    
        shift_interp = np.zeros((nframes,2))
        shift_interp[:,0] = spl_x(np.arange(nframes))
        shift_interp[:,1] = spl_y(np.arange(nframes))
    else:
        shift_interp = shift
            
    return shift_interp, shift_estim, images


def rcc3D(xyz, framenum, timebins, zoom, dll:NativeAPI=None, sigma=1):

    def run(dll):
        print("Computing XY drift")
        drift_xy = rcc(xyz[:,:2], framenum, timebins, zoom=zoom,sigma=sigma, dll=dll)[0]
    
        sheared = xyz[:,:2]*1
        sheared[:,:2] -= drift_xy[framenum]
        sheared[:,1] += xyz[:,2]
        print("Computing Z drift")
        drift_sheared = rcc(sheared, framenum, timebins, dll=dll, zoom=zoom, sigma=sigma)[0]
    
        drift_xyz = np.zeros((len(drift_xy),3))
        drift_xyz[:,:2] = drift_xy
        drift_xyz[:,2] = drift_sheared[:,1]
        drift_xyz -= drift_xyz.mean(0)
        return drift_xyz
    
    if dll is None:
        with NativeAPI(False) as dll:
            return run(dll)
    else:
        return run(dll)
    