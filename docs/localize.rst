Localize
========

.. figure:: /images/localize.png
   :width: 560px
   :alt: Picasso Localize main window showing a single-molecule movie frame with identified spots marked by boxes

   Picasso Localize with identified spots in a movie frame.

Localize performs the super-resolution reconstruction of image stacks in two steps:

- **Identification** finds the single-molecule spots in every frame and places a box around each, at a whole-pixel position. By default, spots are found by their net gradient; a B-spline wavelet segmentation is available as an alternative (see :ref:`localize-wavelet`).
- **Fitting** fits a **PSF model** to the pixels in each box, with an independently chosen **optimizer**: least squares (LQ) or maximum likelihood (MLE, Poisson). Every PSF model can be fitted with either optimizer, on the CPU or on the GPU (see :doc:`localize/gpu`). This gives the sub-pixel position of each molecule, its photons, background and localization precision.

Together, the two steps are referred to here as **localization** (``Analyze`` > ``Localize (Identify & Fit)``). The steps can also be run on their own (``Analyze`` > ``Identify`` and ``Analyze`` > ``Fit``), for example to fit loaded identifications.

To get started, open a movie and follow the steps in :ref:`localize-identification`.

PSF models
----------

The following PSF models are implemented:

- **Elliptical Gaussian.** Fits a 2D Gaussian distribution independent widths ``sx`` and ``sy``.
- **Spherical (isotropic) Gaussian.** Fits a single shared width, so ``sx`` and ``sy`` are always equal. The ``ellipticity`` column is not saved for this model. Supports multichannel fitting as well, see :ref:`localize-multichannel-gaussian`.
- **Rotated elliptical Gaussian.** Same as *Elliptical Gaussian*, however, an in-plane rotation angle is also fitted and saved in the ``angle`` column, in degrees.
- **Experimental PSF (cubic spline).** Fits an experimentally measured PSF; a 3D calibration recovers ``z`` directly; see :doc:`localize/spline`. Supports multichannel fitting as well, see :ref:`localize-multichannel-spline`.
- **Overlapping emitters (3D-DAOSTORM).** Fits all emitters of a frame at once, so that emitters closer than about 2 PSF σ are localized separately, with a Gaussian PSF whose σ is fixed for the movie or fitted per emitter, in x and y for astigmatic 3D, or with an experimental PSF from a spline calibration. It finds the spots itself, without the identification step; see :doc:`localize/multi-emitter`.

In addition, ``Average of ROI`` is available as a non-fitting option that simply sums the intensity of each spot.

Fitting can run on a CUDA-capable GPU (see :doc:`localize/gpu`). The kernels are compiled at run time by Numba, so there is no library to build or install beyond the CUDA runtime (``pip install picassosr[gpu]``), on Windows and Linux alike. When no CUDA GPU is available, the GPU fitting option simply does not appear and Picasso uses the CPU algorithms.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   Identify and fit a movie in one call and save the localizations.
   ``identification_parameters`` holds the identification settings of the
   ``Parameters`` dialog (see :ref:`localize-identification`);
   ``fitting_method`` names the PSF model and the optimizer
   (``gausslq``, ``gaussmle``, ``gausslq-spherical``, ``gaussmle-rotated``,
   ``spline``, ``spline-mle``, ...), with ``-gpu`` appended to fit on the
   GPU.

   .. code-block:: python

      from picasso import io, localize

      movie, info = io.load_movie("movie.tif")
      camera_info = {   # Pixelsize (nm) is required: the localizations are returned in nm
          "Baseline": 100, "Sensitivity": 0.53, "Gain": 1, "Pixelsize": 130
      }

      locs, info = localize.localize(
          movie,
          camera_info=camera_info,
          identification_parameters=localize.IdentificationParameters(
              method=localize.NetGradient(5000), box=7
          ),
          movie_info=info,
          fitting_method="gaussmle",
      )
      io.save_locs("movie_locs.hdf5", locs, info)


.. _localize-file-formats:

Supported file formats
----------------------

Picasso Localize reads the following movie formats:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Format
     - Extension
     - Notes
   * - OME-TIFF and plain TIFF image stacks
     - ``.ome.tif``, ``.tif``, ``.tiff``
     - If a movie is split into multiple μManager ``.tif`` files, open only the first one; Picasso detects the remaining files from their file names.
   * - MicroManager "separate image files"
     - folder of ``img_*.tif``
     - One ``img_*.tif`` per frame in a folder; see :ref:`localize-micromanager-folder`.
   * - NDTiffStack
     - ``.tif``
     -
   * - BigTIFF
     - ``.tif``, ``.btf``, ``.tf8``, ``.tf2``
     -
   * - Zeiss LSM
     - ``.lsm``
     -
   * - Zeiss CZI
     - ``.czi``
     - Requires ``pip install picassosr[czi]``, Python ≥ 3.12; available in the one-click installer.
   * - Leica LIF
     - ``.lif``
     - Requires ``pip install picassosr[lif]``, Python ≥ 3.12; available in the one-click installer.
   * - PicoQuant PTU
     - ``.ptu``
     - Confocal image scans recorded with time-correlated single photon counting (T3 mode); see the note below. Requires ``pip install picassosr[ptu]``, Python ≥ 3.12; available in the one-click installer.
   * - Raw binary
     - ``.raw``
     - A dialog asks for the file specifications when the file is opened.
   * - Imaris
     - ``.ims``
     - For files with several channels, a dialog asks which channel to load.
   * - Nikon ND2
     - ``.nd2``
     - Either a time series (e.g., SMLM measurement) or a z-stack (e.g., calibration) (``T`` or ``Z`` axis).
   * - MetaMorph STK
     - ``.stk``
     - For consecutive files (e.g. ``name_001.stk``, ``name_002.stk``, …), open the first file of the desired range; all subsequent files with a higher numeric suffix are included automatically.

.. note::

   **TIFF-family files** (``.tif``, ``.tiff``, ``.ome.tif``, ``.btf``, ``.tf8``, ``.tf2``, ``.lsm``) are read via the `tifffile <https://github.com/cgohlke/tifffile>`_ library.

   - **Picasso expects grayscale image stacks with one frame per TIFF page; multi-channel, RGB or tiled whole-slide TIFF variants are not supported.**
   - ImageJ "contiguous stack" files are also read correctly, with every plane detected as a frame. In these files ImageJ stores the whole stack as a single TIFF page followed by all planes' pixel data, as its "Save As > Tiff" does for large stacks (e.g. when re-saving a folder of separate images as one ``.tiff``).

.. note::

   **Zeiss** ``.czi`` **and Leica** ``.lif`` movies are read via the optional `czifile <https://github.com/cgohlke/czifile>`_ and `liffile <https://github.com/cgohlke/liffile>`_ libraries, installed with the ``czi`` / ``lif`` extras (e.g. ``pip install picassosr[czi,lif]``; both require Python ≥ 3.12).

   These files are reduced to a single-channel ``(frames, height, width)`` movie:

   - when a file contains more than one channel, a dialog asks which channel to load;
   - a ``.lif`` file may also contain several acquisitions, in which case the one with the most frames is used.

   To load every channel of a multichannel file at once, see :ref:`localize-opening-channels`.

.. _localize-ptu:

.. note::

   **PicoQuant** ``.ptu`` files of confocal image scans are read via the optional `ptufile <https://github.com/cgohlke/ptufile>`_ library, installed with the ``ptu`` extra (``pip install picassosr[ptu]``; requires Python ≥ 3.12).

   A ``.ptu`` file stores every detected photon. Picasso turns the photons of one detector into a movie: a pixel counts the photons detected in it during one scan frame, whatever their arrival time after the laser pulse (the microtime). Photons are assigned to pixels with the pixel dwell time of the file header, as in PicoQuant's software; photons detected while the scanner returns to the start of a line are left out.

   - **The pixel values are photons**, so the camera parameters are set accordingly when the file is opened: baseline 0, sensitivity 1 and EM gain 1 (sCMOS noise maps are cleared). The pixel size is read from the file.
   - **Scan frames can be summed.** A single confocal scan frame can be too dim to localize, so a dialog asks how many consecutive scan frames (*Frames to sum*) are added up into one frame, and how many scan frames lie between the starts of two sums (*Stride*). By default the stride equals the number of frames to sum: the sums do not overlap and every photon is used once. A stride of 1 is a sliding window, where every scan frame starts a new sum, so a binding event is localized in several overlapping frames. Summing applies to photon counts only; camera movies cannot be summed, since every camera frame carries an offset (the baseline).
   - **Microtime gates split the photons by arrival time.** The same dialog shows the decay histogram of every detector: the photons counted against their arrival time after the laser pulse. A *microtime gate* keeps only the photons arriving within a range of times, e.g., one gate per laser of pulsed interleaved excitation (PIE). Drag over the histogram or click *Add gate*; without gates every photon counts.
   - **Each detector is a channel**, or, with microtime gates, each detector and gate (*Detector 1 gate 1*, *Detector 1 gate 2*, ...). When a file has several channels, a dialog asks which one to load; to load all of them, see :ref:`localize-opening-channels`.
   - Only T3 image scans are read, not point or line scans, z-stacks or T2 data.

   The metadata of the movie record the summing (``Frame Sum Window``, ``Frame Sum Stride``, ``Scan Frames``), the gate of the channel (``Microtime Gate (ns)``), ``Photon Counting: true`` and the main header entries of the file (``PTU Metadata``).

We are open to feature requests regarding support for other file formats, please visit our `GitHub page <https://github.com/jungmannlab/picasso>`_.

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   ``io.load_movie`` opens every format in the table, chosen by the file
   extension, and returns the movie with its metadata. Frames are read on
   demand, so large movies open quickly. A ``.raw`` movie needs its
   ``.yaml`` next to it.

   .. code-block:: python

      from picasso import io

      movie, info = io.load_movie("movie.nd2")
      frame = movie[100]      # one frame as a NumPy array
      print(info[0]["Frames"], info[0]["Height"], info[0]["Width"])

   For a photon-counting ``.ptu`` scan, ``frame_sum_window`` sums that many
   consecutive scan frames into one frame and ``frame_sum_stride`` sets
   the number of scan frames between the starts of two sums (default: the
   window, so the sums do not overlap; 1 gives a sliding window).
   ``microtime_gates`` takes ``(start, end)`` gates in ns, giving one channel
   per detector and gate. Any other format raises a ``ValueError`` when
   frames are to be summed or gated.

   .. code-block:: python

      movie, info = io.load_movie("scan.ptu", frame_sum_window=20)
      # two PIE gates, every channel at once
      movies, infos = io.load_movie_all(
          "scan.ptu", frame_sum_window=20, microtime_gates=[(0, 25), (25, 50)]
      )


Topics
------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`search;1.5em;sd-mr-1` Identification and fitting
      :link: localize/identification
      :link-type: doc

      Spot identification and fitting, temporal median and Gaussian filters, wavelet identification, ROIs and extra file actions.

   .. grid-item-card:: :octicon:`zap;1.5em;sd-mr-1` GPU fitting
      :link: localize/gpu
      :link-type: doc

      Installing and using GPU fitting, and the CPU worker pool.

   .. grid-item-card:: :octicon:`device-camera;1.5em;sd-mr-1` Camera configuration
      :link: localize/camera
      :link-type: doc

      The camera config file and per-pixel sCMOS camera calibration.

   .. grid-item-card:: :octicon:`stack;1.5em;sd-mr-1` 3D calibration (astigmatism)
      :link: localize/3d-calibration
      :link-type: doc

      Astigmatic z calibration and fitting.

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` Experimental PSF (cubic spline)
      :link: localize/spline
      :link-type: doc

      Building and checking a spline PSF calibration, fitting with it, and multichannel (e.g. biplane) spline fitting.

   .. grid-item-card:: :octicon:`copy;1.5em;sd-mr-1` Overlapping emitters
      :link: localize/multi-emitter
      :link-type: doc

      Multi-emitter fitting of dense frames: when to use it, its settings and its output.

   .. grid-item-card:: :octicon:`git-compare;1.5em;sd-mr-1` Lateral corrections
      :link: localize/lateral-correction
      :link-type: doc

      Correcting ``x`` and ``y`` for the cylindrical lens and for chromatic aberration, in 2D and 3D.

   .. grid-item-card:: :octicon:`columns;1.5em;sd-mr-1` Multichannel fitting
      :link: localize/multichannel
      :link-type: doc

      Joint 2D Gaussian fitting of registered channels, and analyzing each channel on its own.

.. toctree::
   :hidden:

   localize/identification
   localize/gpu
   localize/camera
   localize/3d-calibration
   localize/spline
   localize/multi-emitter
   localize/lateral-correction
   localize/multichannel
