Simulate
========

Picasso's simulation module (``Picasso: Simulate``) is a tool for evaluating
experimental conditions for DNA-PAINT and generating ground-truth data for test
purposes. This allows systematic analysis of how different experimental
parameters such as imager concentration, target density or integration time
influence the imaging quality and whether the target structure can be resolved
with DNA-PAINT.

By default, ``Picasso: Simulate`` starts with preset parameters that are
typical for a DNA-PAINT experiment. Thus, meaningful raw DNA-PAINT data can be
readily simulated for a given input structure without the need of a
super-resolution microscope. The simulation output is a movie file in .raw
format, as it would be generated during an in vitro DNA-PAINT experiment on a
microscope.

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
   τd = 1/(kon·c). The mean ON time in a DNA-PAINT system is dependent on the
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
settings and update the preview panels. A preview of the arrangement of all
structures is shown in ``Positions``, whereas an individual structure is shown
in ``Structure preview``.

.. _simulate-run:

Running the simulation
~~~~~~~~~~~~~~~~~~~~~~

The simulation will begin by calculating the photons for each handle site of
every structure and then converting it to a movie that will be saved as a .raw
file, ready for subsequent localization.

- All simulation settings are saved and can be loaded at a later time with
  ``File > Load settings from previous simulation...``.
- For 3D data, check ``Simulate 3D`` and load the 3D calibration with
  ``File > Load 3D calibration...``.

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

It is possible to have different imaging parameters for each round, e.g., when
using imagers with different ON-times. To do so, simulate the multiplexing
rounds individually:

1. In the ``Exchange rounds`` field of the ``Simulation`` group, enter only the
   rounds that should be simulated with the current set of parameters.
2. Simulate the data.
3. Change the set parameters and the multiplexing round and simulate the next
   data sets.
4. Repeat until all multiplexing rounds are simulated.
