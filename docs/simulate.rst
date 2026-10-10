Simulate
========

Picasso's simulation module (``Picasso: Simulate``) is a tool for evaluating
experimental conditions for DNA-PAINT and generating ground-truth data for test
purposes. This allows systematic analysis of how different experimental
parameters such as imager concentration, target density or integration time
influence the imaging quality and whether the target structure can be resolved
with DNA-PAINT.

By default, ``Picasso: Simulate`` starts with preset parameters that are
typical for a DNA-PAINT experiment. The simulation output is a movie file in .raw
format, similar to one generated during a DNA-PAINT experiment on a
microscope. Sample drift is not simulated, so the structures stay in place
for the whole movie.

.. figure:: /images/simulate.png
   :width: 360px
   :class: screenshot
   :alt: Picasso Simulate window with the Structure, PAINT, Imager and Camera parameter groups and the structure and positions previews

   The ``Picasso: Simulate`` window.

Simulate DNA-PAINT image acquisitions
-------------------------------------

1. Start ``Picasso: Simulate``.
2. Define the number and type of structures that should be simulated in the
   group ``Structure`` (see :ref:`simulate-structure` below).
3. The group ``PAINT parameters`` allows adjustment of the duty cycle of the
   DNA-PAINT imaging system. The mean dark time is calculated by
   :math:`\tau_d = 1/(k_\mathrm{on} \cdot c)`, where :math:`k_\mathrm{on}`
   is the association rate and :math:`c` the imager concentration. The mean
   ON time in a DNA-PAINT system is dependent on the
   DNA duplex properties. For typical 7-bp imager strands, the ON time is
   ~200-300 ms.
4. In ``Imager parameters``, fluorophore characteristics such as PSF width and
   photon budget can be set. Adjusting the ``Power density`` field affects the
   simulation analogously to changing the laser power in an experiment.
5. The ``Camera parameters`` group allows the user to set the number of
   acquisition frames and integration time. Values calculated from the
   settings, such as the total acquisition time, the mean dark time or the
   photons per frame, are shown in a muted color.

   The default image size is set to 32 pixels. As the computation time
   increases considerably with image size, it is recommended to simulate only
   a subset of the actual camera field of view.
6. Select ``Simulate data`` (bottom right, or ``Simulation > Simulate data...``,
   :kbd:`Ctrl+R`) to start the simulation; a progress bar is shown while it
   runs (see :ref:`simulate-run` below).
7. (Optional step for multiplexing) Multiplexed Exchange-PAINT data can be
   simulated by adjusting the ``Exchange labels`` setting (see
   :ref:`simulate-multiplexing` below).

.. _simulate-structure:

Defining the structures
~~~~~~~~~~~~~~~~~~~~~~~

Predefined grid- and circle-like structures can be readily defined by their
number of columns and rows, or their diameter and the number of handles,
respectively.

Alternatively, a custom structure can be defined in an arbitrary coordinate
system:

- Enter comma-separated coordinates into ``Structure X`` and ``Structure Y``
  (and ``Structure Z`` for 3D), or edit the docking strands in a table with
  ``Edit structure...``, which switches the type to ``Custom``.
- The unit of length of the respective axes can be changed by setting the
  spacing in ``Spacing X, Y``.
- For each coordinate point, an identifier for the docking site sequence needs
  to be set in ``Exchange labels`` as a comma-separated list.
- Correctly defined points will be updated live in the ``Structure preview``.
  Note that entries with missing x coordinate, y coordinate or exchange label
  will be disregarded.

Structures can also be imported:

- When a structure has been previously designed with :doc:`design`, it can be
  imported with ``File > Import structure from Picasso: Design...``.
- Docking strand positions of a .yaml or .hdf5 file can be used with
  ``File > Import handles...``.

A probability for the presence of a handle can be set with ``Incorporation``.

By default, all structures are arranged on a grid with boundaries defined by
``Image size`` in ``Camera parameters`` and the ``Frame`` parameter in the
``Structure`` group. ``Random arrangement`` distributes the structures randomly
within that area, whereas ``Random orientation`` rotates the structures
randomly.

Selecting the button ``Generate positions`` (``Simulation > Generate
positions``, :kbd:`Ctrl+G`) will generate a list of positions with the current
settings and update the two preview panels:

- ``Positions`` shows the whole simulated field of view in camera pixels.
  Each cross is one handle (binding site) that will be simulated, so a
  structure appears as a small cluster of crosses. Handles dropped by
  ``Incorporation`` are not shown. The dashed square marks the ``Frame``
  margin: structures are placed inside it, and handles outside it are not
  simulated.
- ``Structure preview`` zooms in on the first structure, in nm. Each circle
  is one of its handles, colored by imaging round when multiplexing (see
  :ref:`simulate-multiplexing` below).

.. _simulate-run:

Running the simulation
~~~~~~~~~~~~~~~~~~~~~~

The simulation will begin by calculating the photons for each handle site of
every structure and then converting it to a movie that will be saved as a .raw
file, ready for subsequent localization.

- All simulation settings are saved and can be loaded at a later time with
  ``File > Load settings from previous simulation...``.
- For 3D data, check ``Simulate 3D`` and load the 3D calibration with
  ``File > Load 3D calibration...`` (see :ref:`localize-3d-calibration` for
  how to create one).

.. dropdown:: Python
   :icon: code
   :class-container: api-example

   The same steps as the GUI: define a structure (handle coordinates in
   nm), place copies of it, draw the blinking of every handle, render the
   frames and save the ``.raw`` movie with its ``.yaml``.

   .. code-block:: python

      import numpy as np
      from picasso import simulate

      pixelsize, imagesize, frames, itime = 130, 32, 5000, 100   # nm, px, -, ms

      # 3 x 2 grid of handles, 20 nm apart; one exchange round, z = 0
      structure = simulate.defineStructure(
          np.array([0, 20, 40, 0, 20, 40]), np.array([0, 0, 0, 20, 20, 20]),
          np.ones(6), np.zeros(6), pixelsize,
      )
      positions = simulate.generatePositions(9, imagesize, 6, 0)     # 9 copies, 6 px margin, grid
      handles = simulate.prepareStructures(
          structure, positions, orientation=0, number=9, incorporation=0.85, exchange=0
      )

      # Blinking of every handle: photons per frame
      n_handles = handles.shape[1]
      photons = np.zeros((n_handles, frames), dtype=int)
      for i in range(n_handles):
          photons[i], _, _ = simulate.distphotons(
              handles, itime, frames, taud=54054, taub=280,          # ms
              photonrate=53, photonratestd=29, photonbudget=1.5e6,
          )

      # Frames, background and camera noise
      movie = np.zeros((frames, imagesize, imagesize))
      for f in range(frames):
          movie[f] = simulate.convertMovie(
              f, photons, handles, imagesize, frames, psf=0.82,
              photonrate=53, background=4, noise=2, mode3Dstate=False, cx=[], cy=[],
          )
      movie = simulate.check_type(simulate.noisy_p(movie, 4))

      info = {
          "Byte Order": "<", "Data Type": "uint16", "Frames": frames,
          "Height": imagesize, "Width": imagesize, "Camera": "Simulation",
          "Camera.Pixelsize": pixelsize,
      }
      simulate.saveMovie("simulated.raw", movie, info)


.. _simulate-multiplexing:

Multiplexing
~~~~~~~~~~~~

For each handle in the custom coordinate system (``Structure X``,
``Structure Y``), an Exchange round can be specified in ``Exchange labels``.
The different imaging rounds can be visually identified by color in the
``Structure preview``. For each round, a new movie file will be generated.

By default, the simulation software detects the number of exchange rounds
based on the structure definition and will simulate all multiplexing rounds
with the same imaging parameters.
