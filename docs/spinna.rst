SPINNA
======

SPINNA is a module for analyzing the oligomerization of proteins using
super-resolution microscopy data. For more information, please refer to the
publication
`L. A. Masullo, R. Kowalewski, et al. Nature Comm, 2025 <https://doi.org/10.1038/s41467-025-59500-z>`_.

SPINNA supports labeling efficiency fitting as described in
`J. Hellmeier, S. Strauss, et al. Nature Methods, 2024 <https://doi.org/10.1038/s41592-024-02242-5>`_.
See :ref:`spinna-le-fitting` below for instructions.

Overview of the GUI
-------------------

The GUI consists of three tabs that can be navigated at the top of the screen:

1. :ref:`Structures <spinna-structures-tab>`: to define the structures (oligomers) used in
   simulations.
2. :ref:`Simulate <spinna-simulate-tab>`: to simulate any combination of
   structures with user-defined parameters as well as to fit the proportions of
   structures to experimental data.
3. :ref:`Mask generation <spinna-mask-generation-tab>`: to generate masks for
   simulations with heterogeneous densities of the molecular targets.

.. _spinna-structures-tab:

Structures tab
--------------

.. figure:: /images/spinna_structures_tab.png
   :width: 720px
   :class: screenshot
   :alt: Structures tab of Picasso SPINNA with the preview of a tetramer structure, the structures summary and the table of molecular targets with their coordinates

   The *Structures* tab with the example EGFR structures.

This tab allows the user to define the model structures for SPINNA. The
outline of the tab is shown above. Follow these steps to create new
structures:

1. Click *Add a new structure* in the *Structures summary* box (top right
   corner).
2. Enter the name of the structure in the new dialog and confirm by clicking
   *OK*.
3. The structure is now loaded and its name is displayed in the *Preview* box
   (left panel).
4. To add molecular targets, navigate to the box *Molecular targets* (bottom
   right corner).
5. Click *Add a molecular target*. This creates a new row in the *Molecular
   targets* box. Please specify the following: name of the molecular target
   (e.g., EGFR), x, y and z coordinates (in nm). Please note that the structure
   will be rotated around the origin (i.e., x = y = z = 0 nm) during
   simulations.
6. It is possible to delete each molecular target by clicking its
   corresponding delete button (the trash icon in the *Molecular targets*
   box).
7. The user can navigate between structures by clicking on their names in the
   *Structures summary* box.
8. The *Preview* box allows the user to see the currently loaded structure,
   rotate it in 3D (drag with the left mouse button; with :kbd:`Ctrl`, around
   the z axis; *Reset rotation* undoes it), show/hide legend and scale bar
   (whose length is adjustable) as well as save the current view as a .png or
   .tif file.
9. Once at least two molecular targets are defined for the given structure, it
   is possible to add a new molecular target by clicking with the right mouse
   button on the structure view.

The image above illustrates the example structures generated for simulations
of EGFR described in the `SPINNA publication <https://doi.org/10.1038/s41467-025-59500-z>`_. Once the structures are ready to use, save
them by clicking *Save all structures* in the *Structures summary* box.
*Load structures* adds the structures of a saved .yaml file to the current
list, and each structure has a *Delete* button.

.. important::

   The user must ensure that no typos are introduced in the names of the
   molecular targets, since SPINNA will interpret these as separate molecular
   target species.

.. _spinna-simulate-tab:

Simulate tab
------------

.. figure:: /images/spinna_simulate_tab_before_load.png
   :width: 720px
   :class: screenshot
   :alt: Simulate tab of Picasso SPINNA before loading data, with only the Load structures button active

   The *Simulate* tab before loading data.

This tab is used for fitting the SPINNA model (see
:ref:`spinna-structures-tab` above) to experimental data, displaying nearest
neighbor distances (NND) and saving simulated molecules that can later be
loaded into :doc:`render`. The image above shows the outline of the tab before
loading data.

.. _spinna-load-data:

Load data and parameters
~~~~~~~~~~~~~~~~~~~~~~~~

1. Click the *Load structures* button in the top left corner of the window.
   Upon loading, new widgets will appear in the GUI.
2. For each detected molecular target species, load the experimental data,
   which must be saved in .hdf5 format that is compatible with localization
   files in other Picasso modules, see :ref:`files-hdf5`.
3. Furthermore, input label uncertainty, labeling efficiency and observed
   density in the *Load data* box. Alternatively, load the mask to simulate a
   heterogeneous distribution by clicking on *Masks* in the bottom left corner
   of the box. For more information about the mask, see
   :ref:`spinna-mask-generation-tab`.
4. Moreover, in the *Load data* box, the user can change the dimensionality of
   the simulation. If 3D simulation is chosen without a mask, the user needs to
   input the range of z coordinates of molecular targets simulated by clicking
   *Z range*.
5. In the "Optional settings" dialog, the user can change:

   - the mode of rotations of the simulated structures: *random 2D
     rotations* (around the z axis, the default), *random 3D rotations*
     (around all three axes) or *No rotations*;
   - the fitting mode: *Bayesian* (the default), *Coarse to fine* or *Brute
     force*. The chosen fitting mode applies to all fitting workflows (*Find
     best fitting combination*, *Compare models* and *Fit LE*). For more
     information about the fitting modes, see :ref:`spinna-fitting-modes`
     below;
   - *Use multiprocessing* (on by default), which runs the simulations on
     several CPU cores. The Bayesian fitting mode always runs on a single
     core;
   - *Auto set # of NNs for fitting* (on by default): how many nearest
     neighbors are compared between each pair of molecular targets when
     fitting. Automatically, this is the largest number of neighbors of that
     pair within any of the loaded structures (e.g., 3 for tetramers of one
     target). Untick it to set the number for each pair (``NN A → B``)
     yourself.

   The fitting mode chosen here is remembered across sessions, under
   ``Fitting mode`` in the ``SPINNA`` section of ``~/.picasso/settings.yaml``
   (see :ref:`user-settings-file`).

The defaults in the *Load data* box are a label uncertainty of 5 nm, an LE of
50% and an observed density of 100 μm⁻² (μm⁻³ in 3D). If the experimental
data were saved from picks, the density is filled in from the pick area in
their metadata. *Homogeneous distribution* (the default) and *Masks* switch
between the two ways of placing the structures: evenly at the observed
density, or following a density mask (see :ref:`spinna-mask-generation-tab`).

.. _spinna-fitting:

Fitting
~~~~~~~

Within the *Fitting* box:

1. To generate the search space, i.e., the set of stoichiometries tested in
   SPINNA, click the button *Generate parameter search space* and define the
   number of simulation repeats (``# simulations``, 10 by default) and
   ``Granularity`` (21 by default); *Save as .csv* saves the search space,
   which *Load parameter search space* loads again later. For more information, see
   Supplementary Figure 2 in the
   `SPINNA publication <https://doi.org/10.1038/s41467-025-59500-z>`_.
2. To save the fitting scores (Kolmogorov-Smirnov test statistics) for each
   tested stoichiometry, tick *Save fitting scores*. The user will be asked to input the name of the resulting
   .csv file.
3. To obtain the result's uncertainty, check the *Bootstrap* box, which will
   resample from the best fitting model 20 times and rerun SPINNA on the
   resampled datasets. Note that this will increase the computation time.
4. To test different SPINNA models, click *Compare models* (see
   :ref:`spinna-compare-models` below).
5. To run SPINNA, click *Find best fitting stoichiometry*. Once a search
   space is generated or loaded, the button shows the number of tested
   combinations and the estimated time. The progress dialog will be
   displayed. Changing the data, the masks or the densities resets the search
   space, which then has to be generated again.
6. After the fitting is finished, specify the name for saving a fit summary
   file (.txt). This file includes all the information about the fitting, the
   parameters and the results. The user may also choose not to save the file
   by clicking *Cancel* in the dialog.

   Additionally, the fitted stoichiometry is displayed in the *Single
   simulation* box and the NND histograms are shown in the *Plotting* box, see
   the image in :ref:`spinna-fitting-modes` below.

.. _spinna-compare-models:

Compare models
^^^^^^^^^^^^^^

*Compare models* opens a dialog asking the user to input the range of tested
label uncertainties (the user can choose to fit label uncertainty or not) and
the candidate SPINNA models. For example, the user may want to explore the
models with different spacings between the structures or different shape.

- We recommend choosing a lower granularity when comparing models, since the
  fitting may take a long time.
- A single progress dialog is displayed throughout the comparison; its title
  shows the current round number (``[Round X/Y]``) so the user knows how many
  SPINNA rounds remain.
- The fitting mode selected in *Optional settings* is honored.
- Tick *Label uncertainties* to test a range per target (*From* / *To* /
  *Step*, 3 / 8 / 1 nm by default); otherwise the values from the *Load data*
  box are used. *Add a model* adds .yaml structure files (each with the same
  targets as the loaded data); click a model to remove it.
- It needs a generated search space (a loaded one is not accepted). The best
  model is loaded into the tab afterwards.

.. _spinna-le-fitting:

Fitting labeling efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fitting the labeling efficiency (LE), as described in `Hellmeier, Strauss,
et al., Nature Methods, 2024 <https://doi.org/10.1038/s41592-024-02242-5>`__,
has its own workflow. When the
loaded structures contain exactly two molecular targets, a *Fit labeling
efficiency* button is shown in the *Fitting* box. The user does not need to
build the "monomer A / monomer B / heterodimer AB" structures: SPINNA
constructs them internally from the two target names. Before fitting, load
the experimental data of both targets and *generate* the search space (a
loaded search space is not accepted).

Clicking *Fit labeling efficiency* opens a small dialog with three sections:

1. **Fit label uncertainty** (checkbox) - when checked, the dialog exposes a
   *From / To / Step* row per target so SPINNA can search for the best label
   uncertainty. When unchecked, the current value from the *Load data* box is
   used as a fixed input for that target.
2. **Fit heterodimer distance** (checkbox) - when checked, the dialog exposes
   a *From / To / Step* row in nm. When unchecked, a single fixed distance is
   used (entered in the *Distance (nm)* field; by default the distance in a
   loaded heterodimer structure, otherwise 10 nm).
3. **Save fit scores** (checkbox) - when checked, the user selects a folder
   where SPINNA saves the fit scores for every candidate.

The dialog also displays a live "Estimated SPINNA rounds" preview that updates
as the spin boxes change, so the user can gauge how long the fit will take
before starting it.

After the fit, the fitted LEs, distance and label uncertainties are shown
below the button and the label uncertainties are set in the *Load data* box.
The LEs in the *Load data* box are set to 100%. A summary can be saved as
a .txt file.

.. _spinna-fitting-modes:

Fitting modes
~~~~~~~~~~~~~

Since v0.10.0, in the "Optional settings", the user can choose between three
fitting modes. The chosen mode is honored by *Find best fitting
stoichiometry*, *Compare models* and *Fit labeling efficiency*. Previously, only brute force mode was
available.

bayesian
   The search space is explored using Bayesian optimization with Gaussian
   process regression. This is a more efficient way to explore the search
   space, especially when it is large, and it is recommended as the default
   fitting mode.
coarse to fine
   A coarse grid of structure combinations is tested, which consists of 10% of
   evenly distributed structure combinations. Then, a finer grid is tested
   around the best combination from the coarse grid. The "coarse to fine" mode
   is recommended for faster fitting, especially when the search space is
   large.
brute force
   All combinations of structures are tested sequentially.

.. figure:: /images/spinna_simulate_tab_after_fit.png
   :width: 720px
   :class: screenshot
   :alt: Simulate tab of Picasso SPINNA after fitting, with the fitted proportions of monomers, dimers and tetramers and the simulated and experimental nearest neighbor distance histograms

   The *Simulate* tab after fitting.

.. _spinna-single-simulation:

Single simulation
~~~~~~~~~~~~~~~~~

SPINNA allows the user to run a single simulation to visually inspect NNDs
for a specific set of proportions of structures as well as to save the
positions of the simulated molecular targets in an .hdf5 format. Once the
model structures and simulation parameters in the *Load data* box are defined:

1. Enter the proportions of the structures (%) in the *Input proportions of
   structures* box; they must add up to 100%. The simulated area is set by the
   observed density and the number of molecules of the experimental data
   (10,000 molecules if no data are loaded). With masks, experimental data are
   required.
2. To save the positions of molecules from a simulation, tick *Save positions
   of simulated molecules*. The user will be asked to enter the name of the
   resulting file.
3. Click *Run single simulation*. This will generate and display NND
   histogram(s) of the simulated molecular targets (solid lines) and (if
   loaded) of the experimental data (histogram bars).
4. If fitting was completed before, the user can retrieve the best fitting
   combination of proportions of structures by clicking *Best fitting
   combination* in the bottom of the *Input proportions of structures* box.

.. _spinna-plotting:

Plotting
~~~~~~~~

.. figure:: /images/spinna_nnd_plot_settings.png
   :width: 260px
   :class: screenshot
   :align: right
   :alt: Nearest neighbors plots dialog of Picasso SPINNA with the legend, bin size, distance range, labels and colors of the nearest neighbor histograms

   The *Nearest neighbors plots* dialog.

The *Plotting* box, located in the top right corner of the GUI, displays the
NND histograms for simulated (solid lines) and experimental data (histogram
bars).

- The NND plots can be saved by clicking *Save plots* and the plotted values
  (bins and frequencies) by *Save values*.
- *# simulations* controls how many simulation results are accumulated to draw
  NND histograms. The higher the value, the smoother the histograms will be
  obtained.
- *Plot settings* opens the *Nearest neighbors plots* dialog: legend, bin
  sizes for the simulated and experimental data (4 nm by default), the plotted
  distance range (0 to 200 nm), the y-axis maximum (0 for automatic), title,
  axis labels, transparency, the colors of the 1st to 10th nearest neighbors
  and how many neighbors are plotted per pair of targets. *Update plot(s)*
  applies the changes.

  The font family and size chosen there for the title, axis labels and ticks are remembered across sessions,
  under ``NND fonts`` in the ``SPINNA`` section of ``~/.picasso/settings.yaml``
  (see :ref:`user-settings-file`).

If the loaded structures include several molecular target species, several NND
histograms are plotted, one for each pair of molecular target species, which
can be explored by clicking left and right arrows in the *Plotting* box.

.. _spinna-mask-generation-tab:

Mask generation tab
-------------------

.. figure:: /images/spinna_mask_generation_tab.png
   :width: 720px
   :class: screenshot
   :alt: Mask generation tab of Picasso SPINNA with the preview of a 2D density mask, the mask parameters, navigation buttons and mask information

   The *Mask generation* tab with a 2D density mask.

This tab allows the user to create a density/binary mask capable of recovering
the heterogeneous density distribution present in the experimental data.

1. Click *Load molecules* to open the .hdf5 file with molecules/localizations
   that will be used to generate the mask.
2. Adjust ``Pixel/voxel size (nm)`` (50 nm by default) and ``Gaussian blur
   (nm)`` (500 nm by default) to be applied to the mask. The user can choose anisotropic bin size and Gaussian blur with one value in
   the xy plane and another value in the z direction.
3. The mask can be generated in 3D (if the molecules have z coordinates;
   *Isotropic mask* keeps the z values equal to the xy ones) and/or converted
   to a binary mask (*Mask type*).
4. Click *Generate mask*. This may take a while, especially for a 3D mask. The
   mask will be displayed automatically. The legend in the *Display* box
   displays the probability of finding a molecular target per pixel/voxel.
5. Tick *Apply threshold* to threshold the density mask at a probability
   value (off by default). The value is pre-filled with the Otsu threshold
   (`Otsu, IEEE Transactions on Systems, Man, and Cybernetics, 1979
   <https://doi.org/10.1109/TSMC.1979.4310076>`__) after *Generate mask*. With the mask type
   *Binary*, the thresholded mask contains only 0 and 1.
6. To explore the mask:

   - scroll (or pinch on a trackpad) over the preview to zoom at the cursor,
     drag to pan and double-click to show the whole mask again;
   - with the preview focused, the arrow keys pan, :kbd:`+`/:kbd:`-` zoom and
     :kbd:`0` fits the whole mask.

   The position and probability of the pixel under the cursor are shown below
   the preview, and *Fit* shows the whole mask. For 3D masks, tick *Show
   z-slice* in the *Display* box to slice through individual z planes with the
   slider. A scale bar can be shown (*Scale bar (nm)*, 1000 nm by default),
   and *Save view* saves the image. The *Mask information* box shows the area
   (volume in 3D) and the dimensions of the mask.
7. Once the mask is ready, click *Save mask*. This saves a numpy array in the
   .npy format (the thresholded mask if *Apply threshold* is ticked) and a
   .yaml file with its metadata next to it. For 3D masks with *Show z-slice*
   ticked, a .png of every z-slice can be saved as well.

.. _spinna-batch-analysis:

Command window - batch analysis
-------------------------------

SPINNA can be run directly from the command window to allow fast and
efficient batch analysis - either to analyze many datasets or to analyze the
same datasets with many user settings, or both. The entire list thereof is
summarized in a .csv file. For more information on Picasso direct command
window usage, see :doc:`cmd`. SPINNA functions can also be run in a Python
script directly.

.. tab-set::

   .. tab-item:: Command line

      To run SPINNA batch analysis, run:

      .. code-block:: bash

         python -m picasso spinna -p NAME_OF_CSV_FILE

      The following arguments are available:

      .. list-table::
         :header-rows: 1
         :widths: 25 75

         * - Argument
           - Effect
         * - ``-a``, ``--asynch``
           - Switches off the multiprocessing mode. If not specified,
             multiprocessing is used.
         * - ``-v``, ``--verbose``
           - Switches on the verbose mode, i.e., a progress bar for each row is
             displayed. If not specified, the verbose mode is off.
         * - ``-b``, ``--bootstrap``
           - Switches on the bootstrap mode, i.e., the best fitting model is
             resampled 20 times and SPINNA is rerun on the resampled datasets.
             If not specified, the bootstrap mode is off.

      The full column reference of the .csv file (see
      :ref:`spinna-batch-columns` below) can also be printed from the command
      line:

      .. code-block:: bash

         python -m picasso spinna --columns

   .. tab-item:: Python

      SPINNA functions can also be run in a Python script directly. Examples
      are presented in ``samples/sample_notebook_4_spinna.ipynb`` (see
      :ref:`notebook-spinna`).

.. _spinna-batch-columns:

Parameters file
~~~~~~~~~~~~~~~

Each row in the .csv file will specify parameters for which SPINNA is run. In
the file, define the following column names (i.e., the values typed into the
first row) as follows:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - *structures_filename*
     - Path to the file with structures saved (.yaml), see
       :ref:`spinna-structures-tab` above. Required unless ``le_fitting=1``,
       in which case the monomer/heterodimer structures are built internally
       from the two ``exp_data_TARGET`` columns.
   * - *exp_data_TARGET*
     - Path to the file with experimental data (.hdf5) for each molecular
       target species. Each target in the structures must have a
       corresponding column, for example, *exp_data_EGFR*.
   * - *le_TARGET*
     - Labeling efficiency (%) for each molecular target species. Ignored
       when ``le_fitting=1``.
   * - *label_unc_TARGET*
     - Label uncertainty (nm) for each molecular target species. When
       ``le_fitting=1``, this may be a comma-separated list of candidates
       (e.g., ``"3,4,5,6"``); a single value disables the per-target search.
   * - *granularity*
     - Granularity used in parameters search space generation. The higher the
       value the more combinations of structure counts will be tested.
   * - *save_filename*
     - Name of the .txt file where the results will be saved.
   * - *NND_bin*
     - Bin size (nm) for plotting the NND histogram(s).
   * - *NND_maxdist*
     - Maximum distance (nm) for plotting the NND histogram(s).
   * - *sim_repeats*
     - Number of simulation repeats.

Depending on whether a homo- or heterogeneous distribution is used, the
following columns must be present.

For a homogeneous distribution:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - *area* or *volume*
     - Area (2D simulation) or volume (3D simulation) of the simulated ROI
       (um^2 or um^3). For 2D rows, *area* is optional: if omitted, the area
       is read from the experimental data metadata key ``Area (um^2)``
       (written by Picasso when picks/areas are saved).
   * - *z_range*
     - Applicable only when *volume* is provided. Defines the range of z
       coordinates (nm) of simulated molecular targets.

For a heterogeneous distribution:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - *mask_filename_TARGET*
     - Name of the .npy file with the mask saved for each molecular target
       species.

Optional columns are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - *rotation_mode*
     - Random rotations mode used in analysis. Values must be one of {*3D*,
       *2D*, *None*}. Default: *2D*.
   * - *nn_plotted*
     - Number of nearest neighbors plotted, default: 4.
   * - *fitting_mode*
     - Optimization method used to fit the structure counts. Values must be
       one of {*coarse-to-fine*, *bayesian*, *brute-force*}. Default:
       *bayesian*.
   * - *le_fitting*
     - 0 if standard SPINNA is run, 1 if labeling efficiency fitting is to be
       performed. If the column is not provided, standard SPINNA is run. When
       set to 1:

       - monomer A, monomer B and heterodimer structures are built internally
         for each candidate ``distances`` value;
       - label uncertainty is fit per target from the comma-separated
         candidates in ``label_unc_TARGET``;
       - the per-target LE is recovered from the fitted structure proportions;
       - exactly two ``exp_data_*`` columns must be present; the first maps to
         ``target_a``;
       - ``-b/--bootstrap`` is ignored on LE-fitting rows.

       For more details, see
       `Hellmeier, Strauss, et al. Nature Methods, 2024 <https://doi.org/10.1038/s41592-024-02242-5>`_.
   * - *distances*
     - Comma-separated list of candidate heterodimer distances in nm (e.g.,
       ``"5,10,15,20"``). A single value fixes the distance. Required when
       ``le_fitting=1``; ignored otherwise.
