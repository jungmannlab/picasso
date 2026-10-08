.. _localize-gpu:

GPU Fitting
===========

Picasso can run all of its Gaussian and cubic-spline fitting on a CUDA-capable NVIDIA GPU. The fitting kernels are written in Python and compiled for the GPU at run time by numba. This needs Picasso installed with CUDA support, see :doc:`/getting-started/installation`.

The fitting algorithm — the Levenberg-Marquardt driver, its damping rule, its estimators and its PSF models — is a port of `Gpufit <https://github.com/gpufit/Gpufit>`_ (Przybylski et al., *Scientific Reports* **7**, 15722, 2017).

Gpufit's license is reproduced in `LICENSES/Gpufit-LICENSE.txt <https://github.com/jungmannlab/picasso/blob/master/LICENSES/Gpufit-LICENSE.txt>`__.

.. _localize-gpu-usage:

Using it
--------

When a CUDA GPU is detected, the **Use GPU** checkbox becomes available in the ``Parameters`` dialog. Otherwise the checkbox stays hidden and the CPU implementations are used.

GPU fitting is entirely optional; it is typically one to two orders of magnitude faster than a serial CPU fit.

.. _localize-cpu-fitting:

CPU fitting
-----------

CPU fitting spreads spot identification and fitting over a pool of worker processes. The fraction of CPU cores it may use is set by ``cpu_utilization`` under ``Localize`` in ``~/.picasso/settings.yaml`` (see :ref:`user-settings-file`):

- a number between 0 and 1 (exclusive), default 0.8;
- an invalid value (e.g. 1 or above) is reset to 0.8 the next time it is read.
