.. _localize-gpu:

GPU Fitting
===========

Picasso can run all of its Gaussian and cubic-spline fitting on a CUDA-capable NVIDIA GPU. The fitting kernels are written in Python and compiled for the GPU at run time by numba.

The fitting algorithm — the Levenberg-Marquardt driver, its damping rule, its estimators and its PSF models — is a port of `Gpufit <https://github.com/gpufit/Gpufit>`_ (Przybylski et al., *Scientific Reports* **7**, 15722, 2017), which earlier versions of Picasso used as a compiled dependency.

Picasso no longer ships or links against the Gpufit binary; its license is reproduced in ``LICENSES/Gpufit-LICENSE.txt``.

.. _localize-gpu-installation:

Installation
------------

The GPU kernels need the CUDA runtime, which is pulled in as an optional dependency:

.. code-block:: bash

   pip install picassosr[gpu]

.. _localize-gpu-usage:

Using it
--------

When a CUDA GPU is detected, the **Use GPU** checkbox becomes available in the ``Parameters`` dialog for both optimizers, since Picasso implements a least-squares and a maximum-likelihood estimator on the GPU. Otherwise the checkbox stays hidden and the CPU implementations are used.

GPU fitting is entirely optional; it is typically one to two orders of magnitude faster than a serial CPU fit.

.. _localize-cpu-fitting:

CPU fitting
-----------

CPU fitting spreads spot identification and fitting over a pool of worker processes. The fraction of CPU cores it may use is set by ``cpu_utilization`` under ``Localize`` in ``~/.picasso/settings.yaml`` (see :ref:`user-settings-file`):

- a number between 0 and 1 (exclusive), default 0.8;
- an invalid value (e.g. 1 or above) is reset to 0.8 the next time it is read.
