Camera Configuration
====================

Converting camera counts to photons correctly is needed for accurate photon counts, background and localization precisions, and for the noise model of the maximum-likelihood fit (see step 8 of :ref:`localize-identification`). This page describes how Picasso can remember camera parameters in a config file, and how to measure and use a per-pixel calibration of an sCMOS camera.

.. _localize-camera-config:

Camera config file
------------------

Picasso can remember default cameras and will use saved camera parameters. To use camera configs, create a file named ``config.yaml`` in your Picasso user folder ``~/.picasso``:

- ``C:\Users\<you>\.picasso`` on Windows,
- ``/Users/<you>/.picasso`` on macOS,
- ``/home/<you>/.picasso`` on Linux.

This is the same folder that already holds ``settings.yaml``, so the config no longer hides inside the installed package and is identical for every install type (one-click installer, PyPI, source).

.. important::

   **The config file is never created for you — you have to create it manually.**

   To locate the folder quickly, open ``Picasso: Localize`` and select ``File`` > ``Open camera config file location...``; this opens ``~/.picasso`` in your file browser (creating the folder if needed), where you place your ``config.yaml``. If a config is already in use, the same menu entry reveals wherever it actually lives.

To start with a template, copy ``config_template.yaml`` into ``~/.picasso``, rename it to ``config.yaml``, and edit it. The template is bundled inside the ``picasso`` package folder (see :ref:`localize-camera-config-legacy` for where to find it for each type of installation) and is also available `on GitHub <https://github.com/jungmannlab/picasso/blob/master/picasso/config_template.yaml>`__.

Indentations are used for definitions; see `config_template.yaml <https://github.com/jungmannlab/picasso/blob/master/picasso/config_template.yaml>`__ for the expected structure.

.. _localize-camera-config-matching:

How cameras and settings are matched
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a movie is opened, Picasso selects the camera and its settings automatically from the movie's Micro-Manager metadata. Micro-Manager does not need any calibration for this; the config only has to use the same names that Micro-Manager writes into every movie, **character for character**:

- **Camera:** the keys under ``Cameras`` are compared with the camera's device label from your Micro-Manager hardware configuration (e.g. ``HamamatsuHam_DCAM``), saved as ``Camera`` in the metadata.
- **Settings:** Micro-Manager saves every device property as ``<camera>-<property>``. The entries of ``Sensitivity Categories``, ``Gain Property Name`` and ``EM Switch Property`` are property names (e.g. ``PixelReadoutRate``), and the keys under ``Sensitivity`` are the property values exactly as saved (e.g. ``540 MHz - fastest readout``).
- **Emission wavelength:** ``Channel Device`` > ``Name`` is the full metadata key, device and property (e.g. ``FilterTurret1-Label``), and its saved value (e.g. ``3-TIRF 640``) is looked up in ``Emission Wavelengths``.

To find the exact names, open a movie recorded with Micro-Manager and select ``File`` > ``Show metadata...``, or look at the Micro-Manager metadata in the ``.yaml`` file saved next to the localizations.

``.nd2`` files from Nikon NIS-Elements are matched the same way, using the camera name stored in the file. For other files, nothing is matched; the config then only fills the dropdown menus, and the camera and its settings are selected by hand.

.. _localize-camera-config-legacy:

Backward compatibility (legacy in-package config)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Older Picasso versions read ``config.yaml`` from inside the installed ``picasso`` package folder. That still works:

- If no ``~/.picasso/config.yaml`` exists, Picasso falls back to a ``config.yaml`` in the package folder and reads it **in place** (it is never moved or copied, so an existing setup keeps working unchanged).
- ``~/.picasso/config.yaml`` takes precedence when both are present.

For reference, the legacy in-package location per install type is:

- **One-click installer (Windows):** the installation folder (by default ``C:/Picasso``), then ``_internal/picasso``.
- **One-click installer (macOS):** right-click the picasso app in Applications, "Show Package Contents", then ``Contents/Frameworks/picasso``.
- **PyPI:** run ``pip show picassosr`` and look at the ``Location:`` line; the folder is ``<Location>/picasso``.
- **GitHub:** ``picasso/picasso/`` inside your cloned repository.

.. _localize-camera-config-default:

Example: default camera
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   Cameras:
     Camera1:
       Baseline: 100
       Sensitivity: 0.5

If there is only one camera entry, Picasso will create a dropdown menu that has always selected this camera.

.. _localize-camera-config-gain:

Gain
~~~~

If the string ``Gain Property Name`` can be found in the config, Picasso will search for a value for this key in the Micro-Manager metadata and match if found.

.. _localize-camera-config-sensitivity:

Sensitivity
~~~~~~~~~~~

If the string ``Sensitivity Categories`` can be found in the config, Picasso will create a dropdown menu for each entry, and if the property can be located in the Micro-Manager Metadata, it will be automatically set.

.. code-block:: yaml

   Cameras:
     Camera1:
       Baseline: 100
       Sensitivity Categories:
         - PixelReadoutRate
         - Sensitivity/DynamicRange
       Sensitivity:
         540 MHz - fastest readout:
           12-bit (high well capacity): 7.18
           12-bit (low noise): 0.29
           16-bit (low noise & high well capacity): 0.46
         200 MHz - lowest noise:
           12-bit (high well capacity): 7.0
           12-bit (low noise): 0.26
           16-bit (low noise & high well capacity): 0.45

Here, two Sensitivity Categories are given, ``PixelReadoutRate`` and ``Sensitivity/DynamicRange``:

- In the upper dropdown menu, one now will be able to choose from ``540 MHz - fastest readout`` and ``200 MHz - lowest noise``.
- Within 540 MHz it will be ``12-bit (high well capacity): 7.18``, ``12-bit (low noise): 0.29`` and ``16-bit (low noise & high well capacity): 0.46``. Accordingly for the 200 MHz entry.

The dropdown menus can be further nested, e.g., when considering Gain modes:

.. code-block:: yaml

   Sensitivity:
     Electron Multiplying:
       17.000 MHz:
         Gain 1: 15.9
         Gain 2: 9.34
         Gain 3: 5.32

.. _localize-camera-config-qe:

Quantum Efficiency
~~~~~~~~~~~~~~~~~~

This feature is not used since Picasso 0.6.0. It is kept for backward compatibility only.

.. _localize-camera-config-several:

Several cameras
~~~~~~~~~~~~~~~

.. code-block:: yaml

   Cameras:
     Camera1:
     Camera2:
     Camera3:

Once there are several cameras present, Picasso will select the camera whose name matches the Micro-Manager Metadata. If no camera is found, the first one is automatically selected. In the dropdown menu, the configured cameras are displayed in alphabetical order by default. To show selected cameras first, see below.

.. _localize-camera-config-priorities:

Camera priorities
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   CameraPriority:
     - Camera3
     - Camera1

If many cameras are configured, the dropdown can become cluttered. For that reason, the config can additionally include a ``CameraPriority`` field:

- It describes a list of camera names which must match names in the ``Cameras`` field.
- The listed cameras are then displayed on top of the dropdown menu while the non-listed cameras are shown below in alphabetical order.

.. _localize-camera-config-calibrations:

Incorporating calibrations in config file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The astigmatism calibration depends on the microscope, camera, and emission wavelength used. It can become tedious to navigate to and select the correct calibration yaml file. Therefore, the config file can include a field to map camera and emission wavelength to the path of the z calibration yaml file (see :doc:`3d-calibration`):

.. code-block:: yaml

   z-calibrations:
     Camera1:
       525: /path/to/Camera1-GFP-zcalibration.yaml
       595: /path/to/Camera1-Cy3B-zcalibration.yaml

If the camera names and emission wavelengths match the settings in Micromanager, the correct z-calibration is automatically loaded. In any case an alternative calibration yaml file can be loaded by button.

The same mechanism is available for the experimental PSF (cubic spline) calibration (see :doc:`spline`), using a ``spline-calibrations`` field that maps camera and emission wavelength to the path of the spline calibration ``.hdf5`` file:

.. code-block:: yaml

   spline-calibrations:
     Camera1:
       525: /path/to/Camera1-GFP-spline-calibration.hdf5
       595: /path/to/Camera1-Cy3B-spline-calibration.hdf5

As with the z-calibration, the matching spline calibration is loaded automatically when the camera and emission wavelength match the Micromanager settings, and an alternative calibration file can always be loaded via the ``Load calibration`` button in the ``Experimental PSF (spline)`` box.

sCMOS camera calibrations can be selected the same way, through a ``camera-calibrations`` field; see :ref:`localize-scmos-using-maps`.

.. _localize-scmos-calibration:

sCMOS camera calibration
------------------------

An sCMOS sensor has no single readout characteristic. Every pixel has its own offset, amplification gain and readout noise variance, and those variances range from a few to several thousand ADU² on the same chip. Fitting such data with one scalar ``Baseline`` and one scalar ``Sensitivity`` loses both precision and accuracy.

Picasso implements the pixel-dependent noise model of Huang et al. (`Nat. Methods 10, 653-658, 2013 <https://doi.org/10.1038/nmeth.2488>`_).

Given per-pixel maps, the readout variance ``var_k`` (converted to photoelectrons²) is added to both the measured value and the model mean, which makes the sum approximately Poisson again and lets the established maximum-likelihood machinery carry over unchanged.

.. tip::

   **For sCMOS data, use the maximum-likelihood optimizer (MLE) .** The per-pixel noise model only enters the maximum-likelihood fit. Its Cramér-Rao bound is evaluated pixel by pixel and is exact under the model, whereas the closed-form precision used by the least-squares methods assumes a spatially uniform background and can only take the *mean* readout variance over the fitting box. See :ref:`localize-scmos-per-method` for details.

.. _localize-scmos-measuring:

Measuring the maps
~~~~~~~~~~~~~~~~~~

Select ``Calibration`` > ``Characterize sCMOS camera (dark movie)``, which opens a dialog collecting the dark movie, any bright movies and the output file in one place, or run in a command window/terminal:

.. code-block:: bash

   picasso camera-calibrate dark.raw -l light_01.raw -l light_02.raw ... -o mycam_scmos_calib.hdf5

Two acquisitions feed it:

**Dark movie** (required)
   Frames recorded with no light reaching the sensor (cap on the camera head, or a dark room). Its temporal mean per pixel is the offset map, its temporal variance the readout-variance map. This is the only required input.

   - Huang et al. used 60,000 frames.
   - The relative uncertainty of a variance estimate is ``sqrt(2 / (M - 1))``, so 1,000 frames give about 4.5 %, 10,000 about 1.4 % and 60,000 about 0.6 %.
   - Picasso warns below 10,000 frames and refuses below 100.

**Bright series** (optional)
   Several movies of floating fluorophores at different quasi-uniform illumination levels with stable signal across the acquisitions (i.e., without photobleaching effects), taken with  the same camera settings. They are used to measure the per-pixel gain (sensitivity) map, which replaces the scalar ``Sensitivity``.

   - The illumination level can be varied with either the laser power or the exposure time. Varying the exposure time tends to be more stable.
   - A pixel's output mean is ``g·u + o`` and its output variance ``g²·u + var``, so the pair ``(mean − o, variance − var)`` traces a photon-transfer curve whose slope is that pixel's gain.
   - Huang et al. used 15 levels of 20,000 frames spanning roughly 20 to 200 photons per pixel.
   - Without a bright series there is no gain map and the scalar ``Sensitivity`` keeps being used.

The maps are stored raw and camera-native — offset in ADU, variance in ADU², gain in ADU per photoelectron — in a single HDF5 file, so a calibration does not depend on any Picasso setting.

.. important::

   Make sure the camera's offset is high enough that readout noise never drives a pixel below zero ADU. That is what the offset is engineered for, but with an unusually noisy pixel and a low offset the raw counts can clip or wrap, and the measured variance for that pixel then becomes meaningless.

Optionally, record the illumination each bright movie was taken at — the laser power, the exposure time, whatever was varied — in the dialog's ``Illumination`` column, or with one ``-p`` per ``-l`` on the command line:

.. code-block:: bash

   picasso camera-calibrate dark.raw -l light_01.raw -p 0.1 -l light_02.raw -p 1 -l light_03.raw -p 10 --power-unit mW -o mycam_scmos_calib.hdf5

Nothing in the gain fit uses these numbers; they only label the x-axis of the linearity plot described below, which is what makes that plot readable when the levels are not evenly spaced. Give one per bright movie or none at all.

.. _localize-scmos-plot:

Reading the sCMOS calibration plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alongside the ``.hdf5`` Picasso writes a ``*_maps.png``, from both the GUI and the command line. Each map appears next to its own histogram showing the distribution, as in Supplementary Fig. 1 of Huang et al. Readout variance, offset and gain are shown.

A bright series of more than one level adds a fourth row, asking whether the sensor responded linearly over the range the gain was fitted on. Both panels are chip medians, one point per illumination level.

**Linearity**
   The median signal (per-pixel temporal mean, minus the offset map) against the illumination it was recorded at, or against the level index when none was given.

   - The points should lie on a straight line; the fitted line is drawn and the title reports the worst point's deviation as a percentage of the span.
   - Levels that flatten off at the top are saturating, and every level above that knee biases the gain fit without making it fail.
   - A nonzero intercept is not a nonlinearity — it usually means stray light or background — and the fit absorbs it.
   - When no illumination was recorded, the axis says so, the points are only joined by a guide line, and no deviation is quoted, because equal steps on the axis then stand for unknown steps in illumination.

**Photon transfer curve**
   The median excess variance (per-pixel temporal variance, minus the readout-variance map) against that same median signal, with the median of the fitted gain map drawn through the origin as a line.

   - Since the signal is ``g·u`` and the excess variance ``g²·u``, the points lie on a line of slope ``g`` whatever the illumination levels were, which makes this the panel to read when they were uneven or unknown.
   - Curving *down* at the bright end is saturation.
   - Curving *up* can be an unstable laser, whose frame-to-frame intensity fluctuation adds a variance term growing as ``u²``.
   - Missing the origin means the dark movie does not describe these movies, through a changed camera setting, a temperature drift, or light leaking into the dark acquisition.

The two fail separately: a linear camera behind a nonlinear laser gives a straight transfer curve and a bent linearity panel, which says to fix the power calibration rather than the camera.

.. _localize-scmos-using-maps:

Using the maps
~~~~~~~~~~~~~~

In the ``Photon conversion`` group of the ``Parameters`` dialog, load the file next to ``sCMOS noise maps``, or pass it on the command line:

.. code-block:: bash

   picasso localize movie.raw -a mle -cm mycam_scmos_calib.hdf5

While a calibration is loaded, the scalars it supersedes are set to the maps' own medians and disabled:

- ``Baseline`` to the median offset,
- ``Sensitivity`` to the reciprocal of the median gain, if the calibration carries a gain map,
- ``EM gain`` to 1.

Clearing the calibration restores the previous values. Only the maps are used in the fit; the medians are shown because those numbers still go into the localization metadata. The calibration path and a summary of the maps are recorded there too.

A calibration can also be selected automatically through a ``camera-calibrations`` section in ``config.yaml``, keyed by camera and then by emission wavelength exactly like ``z-calibrations`` and ``spline-calibrations`` (see :ref:`localize-camera-config-calibrations`):

.. code-block:: yaml

   camera-calibrations:
     HamamatsuHam_DCAM:
       525: C:/path/to/your_scmos_calib_525.hdf5
       595: C:/path/to/your_scmos_calib_595.hdf5

.. _localize-scmos-per-method:

What it changes, per fitting method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Maximum likelihood** (``MLE`` with any PSF model) is where the noise model does its work. The likelihood becomes the one Huang et al. derive, the reported ``lpx`` / ``lpy`` become the sCMOS Cramér-Rao bound, and the goodness-of-fit statistic becomes their ``LLR_sCMOS``.
- **Least squares** is mathematically *unaffected* by the variance map.

  - Its reported uncertainty does grow, because readout noise is a genuine part of the residual scatter and pretending otherwise makes the error bars optimistic.
  - If the calibration carries offset or gain maps, the least-squares fit does move slightly — but through the improved counts-to-photons conversion, not through the noise model.

Two further caveats:

- Set ``EM Gain`` to 1. This settings should be used only for the EMCCD cameras. An sCMOS sensor does not multiply, and combining an EM gain with a camera calibration applies the EMCCD excess-noise factor on top of the readout variance, double-counting the noise. Picasso warns if you do.
- The reported ``log_likelihood`` is evaluated on the shifted data, so its values are not comparable between runs with and without a calibration.

.. _localize-scmos-checking:

Checking a calibration
~~~~~~~~~~~~~~~~~~~~~~

The maps drift with the sensor: Huang et al. report that switching their camera from fan to liquid cooling, a change of about 30 K, was enough to invalidate a calibration. Bit depth, readout rate and any selectable gain setting change them outright.

``Calibration`` > ``Check sCMOS calibration (fresh dark movie)`` tests a stored calibration against a short fresh dark movie — about 1,000 frames is plenty. The same check runs from the command line:

.. code-block:: bash

   picasso camera-validate mycam_scmos_calib.hdf5 fresh_dark.raw

If the camera still behaves as characterized, the per-pixel p-values are uniform and their mean sits at 0.5. A mean outside 0.5 ± 0.1 means the camera has drifted and should be re-characterized.

.. _localize-scmos-multichannel:

Multichannel and split field of view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multichannel spline workflows (see :ref:`localize-multichannel-spline`) take one calibration **per channel**, through the ``camera_calibrations`` argument of ``localize.fit_spline_multichannel``, ``fit_spline_multichannel_ratiometric`` and ``get_spots_multichannel``.

Entries may individually be ``None`` when only some channels sit on a characterized camera; such a channel keeps the plain Poisson model.

Each channel's maps are cut at *that channel's* mapped and rounded box origin, the same origin its spot is cut at, so a calibration follows its channel through the affine registration. This matters as soon as the channels are registered more than a pixel apart: reading a non-reference channel's noise at the reference position would sample the wrong pixels.

Split field of view is one physical sensor whose sub-regions are the channels, so ``localize.fit_spline_split_fov`` takes a single ``camera_calibration`` and applies the same full-frame maps to every region. The maps are indexed by absolute frame coordinates, so each region reads its own pixels without further bookkeeping.

.. _localize-scmos-scalars:

Still using the scalars
~~~~~~~~~~~~~~~~~~~~~~~

Spline PSF calibration from a bead z-stack converts its bead spots with the scalar camera parameters. This is harmless in practice, because calibration beads are bright enough that readout noise is negligible against their shot noise.
