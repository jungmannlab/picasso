.. _localize-vectorfit:

Vectorial PSF (VectorFit)
=========================

Picasso can fit a **vectorial PSF model** to every spot with the VectorFit library. The model is computed from the optics of the microscope, so no bead calibration is needed, and its 3D variant recovers ``z`` directly.

VectorFit is described in `Droste et al., Optica <https://doi.org/10.1364/OPTICA.560721>`_. That work also estimates field-dependent aberrations from the single-molecule data itself; Picasso does not do this yet and only uses the aberrations you set (see below).

Theory
------

**Vectorial model.** At high numerical aperture, the scalar (and Gaussian) approximations of the PSF break down: the polarization of the emitted light and the refraction at the interfaces between sample medium, cover slip and immersion fluid shape the spot. The vectorial model propagates the field of the emitter, a freely rotating dipole, through the objective's pupil and images it onto the camera. It depends on:

- the numerical aperture and the emission wavelength,
- the refractive indices of the medium, the cover slip and the immersion fluid,
- the camera pixel size and the fit box size,
- the aberrations of the system, described in the pupil by Zernike modes.

Because the refractive-index mismatch is part of the model, the fitted ``z`` needs no magnification factor.

**Aberrations.** Five primary aberrations can be added to the pupil, all in nm and 0 by default:

.. list-table::
   :header-rows: 1
   :widths: 40 20

   * - Aberration
     - Zernike mode
   * - Vertical astigmatism
     - Z\ :sub:`2`\ :sup:`2`
   * - Oblique astigmatism
     - Z\ :sub:`2`\ :sup:`−2`
   * - Horizontal coma
     - Z\ :sub:`3`\ :sup:`1`
   * - Vertical coma
     - Z\ :sub:`3`\ :sup:`−1`
   * - Spherical aberration
     - Z\ :sub:`4`\ :sup:`0`

They are **fixed**: the same for every spot across the field of view, and not fitted.

**Fit.** Each spot is fitted by maximum likelihood under Poisson noise, with Levenberg-Marquardt iterations started from a phasor-based estimate of the position. The *2D* model fits ``x``, ``y``, photons and background; the *3D* model fits ``z`` as well, within ±800 nm of the focal plane. The localization precisions (``lpx``, ``lpy``, ``lpz``, ``photons_unc``, ``bg_unc``) are the Cramér-Rao lower bounds at the fitted parameters.

Fitting with VectorFit
----------------------

VectorFit runs on the CPU and is an optional dependency:

.. code-block:: bash

   pip install picassosr[vectorfit]

1. Identify the spots as usual (see :ref:`localize-identification`). The **box size** must be one VectorFit supports: 5, 7, 9, 11, 13, 15, 17, 19, 21 or 27.
2. Make sure the **pixel size** of the camera is right: the model is computed in nm.
3. Open ``Analyze`` > ``Parameters`` and set **Model** to ``Vectorial PSF (VectorFit)``. There is no optimizer to choose.
4. In the **Vectorial PSF (VectorFit)** box, set the **Fit model** (``2D`` or ``3D``), the emission wavelength, the NA and the three refractive indices of your setup.
5. Set the aberrations you know. For 3D imaging with a cylindrical lens, enter its astigmatism: an aberration-free PSF is symmetric about focus, so ``z`` above and below focus cannot be told apart.
6. Run ``Analyze`` > ``Localize (Identify & Fit)`` (or ``Fit`` for already-identified spots).

**Outliers.** VectorFit flags the spots whose fit did not converge, whose likelihood is far below the others' or whose position lies far from the box center. They are discarded unless **Discard outliers** is unticked. Their number is shown in the status bar and saved in the metadata as ``VectorFit outliers discarded``.

In addition to the usual columns, VectorFit fits report ``lpx``, ``lpy`` (and ``z``, ``lpz`` for 3D), ``photons_unc``, ``bg_unc``, ``log_likelihood``, ``reduced_chi_square`` and ``iterations``. The settings are recorded in the ``_locs.yaml`` metadata.

.. tab-set::

   .. tab-item:: Command line

      .. code-block:: bash

         picasso localize movie.tif --fit-method vectorfit -b 9 \
             --vf-model 3D --vf-wavelength 670 --vf-na 1.49 \
             --vf-vertical-astigmatism 80

      The ``--vf-*`` options set the optics (``--vf-wavelength``, ``--vf-na``, ``--vf-n-medium``, ``--vf-n-coverslip``, ``--vf-n-immersion``) and the aberrations (``--vf-vertical-astigmatism``, ``--vf-oblique-astigmatism``, ``--vf-horizontal-coma``, ``--vf-vertical-coma``, ``--vf-spherical-aberration``). ``--vf-keep-outliers`` keeps the outliers.

   .. tab-item:: Python

      .. code-block:: python

         from picasso import localize

         locs, info = localize.localize(
             movie,
             camera_info=camera_info,
             identification_parameters=localize.IdentificationParameters(
                 method=localize.NetGradient(5000), box=9
             ),
             movie_info=info,
             fitting_method="vectorfit",
             vectorfit_parameters={
                 "Fit model": "3D",
                 "Wavelength": 670,          # nm
                 "NA": 1.49,
                 "Vertical astigmatism": 80, # nm
             },
         )

      Settings left out take the defaults of ``picasso.vectorfit.DEFAULT_PARAMETERS``.
