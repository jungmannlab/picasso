# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from icecream import ic
from native_api import NativeAPI

from rcc import rcc, rcc3D
        
def dme_estimate(positions, framenum, crlb, framesperbin, imgshape, 
          coarseFramesPerBin=None,
          coarseSigma=None, 
          perSpotCRLB=False,
          useCuda=False,
          display=True, # make a plot
          pixelsize=None,
          maxspots=None, 
          initializeWithRCC=True, 
          initialEstimate=None, 
          rccZoom=2,
          estimatePrecision=True,
          maxNeighbors=1000,
          useDebugLibrary=False,
          custom_format=False):
    """
    Estimate drift using minimum entropy method. Parameters:

    Required parameters: 
        
    positions: a N by K sized numpy array with all the positions, with N being the number of localizations and K the number of dimensions
    framenum: a integer numpy array of frame numbers corresponding to each localization in positions
    crlb: an N by K sized numpy array with uncertainty of positions (cramer rao lower bound from the localization code)
    framesperbin: Number of frames per spline point. Either 1 to disable spline interpolation, or >= 4 to enable cubic splines.
    imgshape: Field of view size [height, width]. Only used for computing the initial 2D estimate using RCC 
        
    Optional parameters:

    estimatePrecision: Split the dataset in two, and estimate drift on both. The difference gives an indication of estimation precision.
    display: Generate a matplotlib plot with results
    coarseFramesPerBin / coarseSigma: If not None, do a coarse initialization to prevent a local minimum. 
            coarseSigma sets an alternative 'CRLB' to smooth the optimziation landscape (typically make it 4x larger).
    pixelsize: Size of pixels in nm. If display=True, it will convert the units in the plot to nm
    maxspots: If not None, it will select the brightess spots to use and ignore the rest. Useful for large datasets > 1M spots
    initialEstimate: Initial drift estimate, replaces RCC initialization
    maxNeighbors: Limit the number of neighbors a single spot can have. 
    
    Return value:
        
    If estimatePrecision is True:
        The estimated drift of full dataset, a tuple with drifts of split dataset
    Else
        The estimated drift as numpy array
    
    """
    ndims = positions.shape[1]
    numframes = np.max(framenum)+1

    initial_drift = np.zeros((numframes,ndims))
    
    with NativeAPI(useCuda, debugMode=useDebugLibrary) as dll:

        if initialEstimate is not None:
            initial_drift = np.ascontiguousarray(initialEstimate,dtype=np.float32)
            assert initial_drift.shape[1] == ndims
            
        elif initializeWithRCC:
            if type(initializeWithRCC) == bool:
                initializeWithRCC = 10
    
            posI = np.ones((len(positions),positions.shape[1]+1)) 
            posI[:,:-1] = positions
    
            if positions.shape[1] == 3:
                initial_drift = rcc3D(posI, framenum, initializeWithRCC, dll=dll, zoom=rccZoom)
            else:
                initial_drift = rcc(posI, framenum ,initializeWithRCC, dll=dll, zoom=rccZoom)[0]
                np.save("drift_RCC_from_DME.npy", initial_drift)
            
    
        if maxspots is not None and maxspots < len(positions):
            print(f"Drift correction: Limiting spot count to {maxspots}/{len(positions)} spots.")
            bestspots = np.argsort(np.prod(crlb,1))
            indices = bestspots[-maxspots:]
            crlb = crlb[indices]
            positions = positions[indices]
            framenum = framenum[indices]
        
        if not perSpotCRLB:
            crlb = np.mean(crlb,0)[:ndims]
            
        numIterations = 10000
        step = 0.000001

        splitAxis = np.argmax( np.var(positions[:,:2],0) ) # only in X or Y
        splitValue = np.median(positions[:,splitAxis])
        
        set1 = positions[:,splitAxis] > splitValue
        set2 = np.logical_not(set1)
        
        if perSpotCRLB:
            print("Using drift correction with per-spot CRLB")
            crlb_set1 = crlb[set1]
            crlb_set2 = crlb[set2]
        else:
            crlb_set1 = crlb
            crlb_set2 = crlb
                            
        maxdrift=0 # ignored at the moment
        if coarseFramesPerBin is not None:
            print(f"Computing initial coarse drift estimate... ({coarseFramesPerBin} frames/bin)",flush=True)
            with tqdm.tqdm() as pbar:
                def update_pbar(i,info): 
                    pbar.set_description(info.decode("utf-8")); pbar.update(1)
                    return 1
    
                initial_drift,score = dll.MinEntropyDriftEstimate(
                    positions, framenum, initial_drift*1, coarseSigma, numIterations, step, maxdrift, 
                    framesPerBin=coarseFramesPerBin, cuda=useCuda,progcb=update_pbar)
                
        print(f"\nEstimating drift... ({framesperbin} frames/bin)",flush=True)
        with tqdm.tqdm() as pbar:
            def update_pbar(i,info): 
                pbar.set_description(info.decode("utf-8"));pbar.update(1)
                return 1
            drift,score = dll.MinEntropyDriftEstimate(
                positions, framenum, initial_drift*1, crlb, numIterations, step, maxdrift, framesPerBin=framesperbin, maxneighbors=maxNeighbors,
                cuda=useCuda, progcb=update_pbar)
                
        if estimatePrecision:
            print(f"\nComputing drift estimation precision... (Splitting axis={splitAxis})",flush=True)
            with tqdm.tqdm() as pbar:
                def update_pbar(i,info): 
                    pbar.set_description(info.decode("utf-8"));pbar.update(1)
                    return 1
                drift_set1,score_set1 = dll.MinEntropyDriftEstimate(
                    positions[set1], framenum[set1], initial_drift*1, crlb_set1, numIterations, step, maxdrift, 
                    framesPerBin=framesperbin,cuda=useCuda, progcb=update_pbar)
    
                drift_set2,score_set2 = dll.MinEntropyDriftEstimate(
                    positions[set2], framenum[set2], initial_drift*1, crlb_set2, numIterations, step, maxdrift, 
                    framesPerBin=framesperbin,cuda=useCuda,progcb=update_pbar)
    
        drift -= np.mean(drift,0)

        if estimatePrecision:
            drift_set1 -= np.mean(drift_set1,0)
            drift_set2 -= np.mean(drift_set2,0)

            L = min(len(drift_set1),len(drift_set2))
            diff = drift_set1[:L] - drift_set2[:L]
            rmsd = np.sqrt( (diff**2).mean(0) )
            print(f"\nRMSD of drift traces on split dataset: {rmsd}",flush=True)

        if display:
            L=len(drift)
            fig,ax=plt.subplots(ndims,1,sharex=True,figsize=(10,8),dpi=100)
            
            axnames = ['X', 'Y', 'Z']
            axunits = ['px', 'px', 'um']
            for i in range(ndims):
                axname=axnames[i]
                axunit = axunits[i]
                if estimatePrecision:
                    ax[i].plot(drift_set1[:L,i], '--', label=f'{axname} - set1')
                    ax[i].plot(drift_set2[:L,i], '--', label=f'{axname} - set2')
                ax[i].plot(drift[:L,i], label=f'{axname} - full')
                ax[i].plot(initial_drift[:L,i], label=f'Initial value {axname}')
                ax[i].set_ylabel(f'Drift {axname} [{axunit}]')
                ax[i].set_xlabel('Frame number')
                if i==0: ax[i].legend(fontsize=12)
            
            if estimatePrecision:
                if pixelsize is not None:
                    p=rmsd
                    scale = [pixelsize, pixelsize, 1000]
                    info = ';'.join([ f'{axnames[i]}: {p[i]*scale[i]:.1f} nm ({p[i]:.3f} {axunits[i]})' for i in range(ndims)])
                    
                    plt.suptitle(f'Drift trace. RMSD: {info}')
                else:
                    plt.suptitle(f'Drift trace. RMSD: X/Y={rmsd[1]:.3f}/{rmsd[1]:.3f} pixels')

        if custom_format:
            return (drift[:,0], drift[:,1])

        if estimatePrecision:
            return drift, (drift_set1, drift_set2)
                                
        return drift


