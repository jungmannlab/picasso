# -*- coding: utf-8 -*-

import ctypes
import os
import math
import numpy as np
import numpy.ctypeslib as ctl
import matplotlib.pyplot as plt
import sys
import tqdm


def debugPrint(msg):
    sys.stdout.write(msg.decode("utf-8"))
    return 1 # also print using OutputDebugString on C++ side


class NativeAPI:
    def __init__(self, useCuda=False, debugMode=False):
        thispath = os.path.dirname(os.path.abspath(__file__))

        if ctypes.sizeof(ctypes.c_voidp) == 4:
            raise RuntimeError(f"The DME drift estimation code can only be used on 64-bit systems.")

        if useCuda:
            dllpath = "dme-cuda"
        else:
            dllpath = "dme-cpu"
            
        if debugMode:
            dllpath = "Debug/" + dllpath
        else:
            dllpath = "Release/" + dllpath
            
        dllpath = f"/x64/{dllpath}.dll"
        abs_dllpath = os.path.abspath(thispath + dllpath)
        
        if debugMode:
            print("Using " + abs_dllpath)
        self.debugMode = debugMode
        
        currentDir = os.getcwd()
        os.chdir(os.path.dirname(abs_dllpath))

        lib = ctypes.CDLL(abs_dllpath)
        os.chdir(currentDir)
        
        self.lib_path = abs_dllpath
        self.lib = lib
        
        self.DebugPrintCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_char_p)
        self._SetDebugPrintCallback = lib.SetDebugPrintCallback
        self._SetDebugPrintCallback.argtypes = [self.DebugPrintCallback]

#void(*cb)(int width,int height, int numImg, const float* data, const char* title));
                
#        self._GetDeviceMemoryAllocation = smlmlib.GetDeviceMemoryAllocation

        self.SetDebugPrintCallback(debugPrint)
        
        self.ProgressCallback = ctypes.CFUNCTYPE(
            ctypes.c_int32,  # continue
            ctypes.c_int32,  # iteration
            ctypes.c_char_p
        )
        
        
        self._MinEntropyDriftEstimate = self.lib.MinEntropyDriftEstimate
        self._MinEntropyDriftEstimate.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xy: float[numspots, dims]
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # crlb: float[numspots, dims] or float[dims]
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # framenum
            ctypes.c_int32,  # numspots
            ctypes.c_int32, #maxit
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # drift XY
            ctypes.c_int32, # framesperbin
            ctypes.c_float, # gradientstep
            ctypes.c_float, # maxdrift
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # scores
            ctypes.c_int32, # flags
            ctypes.c_int32, # maxneighbors
            self.ProgressCallback] # flags
        self._MinEntropyDriftEstimate.restype = ctypes.c_int32
        
        
        # (float * image, int imgw, int imgh, float * spotList, int nspots)
        self._Gauss2D_Draw = lib.Gauss2D_Draw
        self._Gauss2D_Draw.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # mu
            ctypes.c_int32,
            ctypes.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # mu
            ctypes.c_int32
        ]
        
        
    # Spots is an array with rows: [ x,y, sigmaX, sigmaY, intensity ]
    def DrawGaussians(self, img, spots):
        spots = np.ascontiguousarray(spots, dtype=np.float32)
        nspots = spots.shape[0]
        assert spots.shape[1] == 5
        img = np.ascontiguousarray(img, dtype=np.float32)
        self._Gauss2D_Draw(img, img.shape[1], img.shape[0], spots, nspots)
        return img


    def SetDebugPrintCallback(self, fn):
        self.dbgPrintCallback = self.DebugPrintCallback(fn)  # make sure the callback doesnt get garbage collected
        self._SetDebugPrintCallback(self.dbgPrintCallback)


    def Close(self):
        if self.lib is not None:
            # Free DLL so we can overwrite the file when we recompile
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
            ctypes.windll.kernel32.FreeLibrary(self.lib._handle)
            self.lib = None
        
        
    def MinEntropyDriftEstimate(self, positions, framenum, drift, crlb, iterations, 
                        stepsize, maxdrift, framesPerBin=1, cuda=False, progcb=None,flags=0, 
                        maxneighbors=10000):
        
        positions = np.ascontiguousarray(positions,dtype=np.float32)
        framenum = np.ascontiguousarray(framenum,dtype=np.int32)
        drift = np.ascontiguousarray(drift,dtype=np.float32)
        
        nframes = np.max(framenum)+1
        
        assert len(drift)>=nframes and drift.shape[1]==positions.shape[1]

        if len(drift)>nframes:
            drift = drift[:nframes]
            drift = np.ascontiguousarray(drift,dtype=np.float32)

        if cuda:
            flags |= 2
                    
        scores = np.zeros(iterations,dtype=np.float32)
        
        if positions.shape[1] == 3:
            flags |= 1 # 3D

        if np.isscalar(crlb):
            crlb=np.ones(positions.shape[1])*crlb

        crlb = np.array(crlb,dtype=np.float32)
        if len(crlb.shape) == 1: # constant CRLB values, all points have the same CRLB
            flags |= 4
            assert len(crlb) == positions.shape[1]
            #print(f"DME: Using constant crlb")
        else:
            assert np.array_equal(crlb.shape,positions.shape)
            #print(f"DME: Using variable crlb")
            
        crlb=np.ascontiguousarray(crlb,dtype=np.float32)
                
        if progcb is None:
            progcb = lambda i,txt: 1

        nIterations = self._MinEntropyDriftEstimate(
            positions, crlb, framenum, len(positions), iterations, drift, framesPerBin,
            stepsize, maxdrift, scores, flags, maxneighbors, self.ProgressCallback(progcb))

        return drift, scores[:nIterations]


    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Close()
        