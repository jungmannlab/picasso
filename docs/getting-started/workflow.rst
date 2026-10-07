First Steps
===========

Picasso is a set of modules, each in its own window, that pass files from one
step of a DNA-PAINT experiment to the next. Start them from their shortcuts
or from a terminal with ``picasso <module>`` (for example ``picasso
render``), depending on how you installed Picasso (see
:doc:`installation`).

A typical analysis
------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: :octicon:`pencil;1.2em;sd-mr-1` 1 · Plan a DNA origami experiment (optional)
      :link: /design
      :link-type: doc
      :class-card: sd-card-hover

      Design a DNA origami in :doc:`/design` and test whether your imaging
      conditions can resolve it with :doc:`/simulate`, to get a movie
      file like from a microscope. *Note: only applicable if you are
      interested in imaging DNA origamis*.

   .. grid-item-card:: :octicon:`location;1.2em;sd-mr-1` 2 · Localize
      :link: /localize
      :link-type: doc
      :class-card: sd-card-hover

      Open the raw movie (``.raw``, ``.tif``, ``.nd2``, ``.czi``, ...) in
      :doc:`/localize`, identify the single-molecule spots and fit them.

      **Output:** ``<movie>_locs.hdf5`` with one row per localization and
      its metadata.

   .. grid-item-card:: :octicon:`filter;1.2em;sd-mr-1` 3 · Filter (optional)
      :link: /filter
      :link-type: doc
      :class-card: sd-card-hover

      Inspect the localization properties in :doc:`/filter` and remove
      outliers, for example localizations with too few photons or a poor
      precision.

   .. grid-item-card:: :octicon:`image;1.2em;sd-mr-1` 4 · Render and analyze
      :link: /render
      :link-type: doc
      :class-card: sd-card-hover

      Open the localizations in :doc:`/render` to see the super-resolution
      image. Correct the drift, pick structures of interest and analyze
      them, for example by clustering, RESI or G5M.

      **Output:** for example ``<locs>_render.hdf5`` (localizations saved
      after drift correction), ``<locs>_picked.hdf5`` (picked
      localizations), exported images.

   .. grid-item-card:: :octicon:`stack;1.2em;sd-mr-1` 5 · Further analysis (optional)
      :link: /average
      :link-type: doc
      :class-card: sd-card-hover

      Average picked structures in :doc:`/average`, analyze protein
      oligomerization with :doc:`/spinna` or classify structures with
      :doc:`/nanotron`.

In most windows, the :octicon:`book` buttons and ``File > Help`` open the
matching section of this documentation. The meaning of the localization
columns (``x``, ``photons``, ``lpx``, ...) is explained in
:ref:`files-localization-hdf5`, and the information saved with each file
(pixel size, processing history, ...) in :ref:`files-metadata-settings`.

Picasso as a package
--------------------

Everything the GUI does is available in Python as well. For example,
link localizations into binding events and compute their dark times:

.. code-block:: python

   from picasso import io, postprocess

   locs, info = io.load_locs("testdata_locs.hdf5")

   # Link localizations and calculate dark times
   linked_locs = postprocess.link(locs, info, r_max=0.05, max_dark_time=1)
   linked_locs = postprocess.compute_dark_times(linked_locs)

   print(f"Average bright time {linked_locs['n'].mean():.2f} frames")
   print(f"Average dark time {linked_locs['dark'].mean():.2f} frames")

More examples are in the :doc:`/postprocessing` notebooks and the
:doc:`/api/index`. Some steps also run in batch from the :doc:`/cmd`.
