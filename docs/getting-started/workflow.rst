First Steps
===========

Picasso is a set of modules, each in its own window, that pass files from one
step of a DNA-PAINT experiment to the next. Start them from their shortcuts
(created by the one-click installer), or from a terminal with ``picasso
<module>`` (for example ``picasso render``).

A typical analysis
------------------

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: :octicon:`pencil;1.2em;sd-mr-1` 1 · Plan the experiment (optional)
      :link: /design
      :link-type: doc
      :class-card: sd-card-hover

      Design a DNA origami in :doc:`/design` and test whether your imaging
      conditions can resolve it with :doc:`/simulate`, which writes a
      ``.raw`` movie just like a microscope would.

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

      **Output:** for example ``_undrift.hdf5`` (drift-corrected),
      ``_picked.hdf5`` (picked localizations), exported images.

   .. grid-item-card:: :octicon:`stack;1.2em;sd-mr-1` 5 · Go further (optional)
      :link: /average
      :link-type: doc
      :class-card: sd-card-hover

      Average picked structures in :doc:`/average`, analyze protein
      oligomerization with :doc:`/spinna` or classify structures with
      :doc:`/nanotron`.

Each localization file carries metadata on the steps that produced it, see
:doc:`/files`.

Tips for getting around
-----------------------

- **Drag and drop** files into any Picasso window to open them.
- **Help** (the :octicon:`question` buttons and the *Help* menu) opens the
  matching section of this documentation.
- Picasso remembers your settings, for example the last folder you opened,
  in ``~/.picasso/settings.yaml``, see :ref:`user-settings-file`.
- Picasso's windows follow the light or dark mode of your system. Change the
  look in ``File > Appearance...``, see :ref:`appearance`.

Prefer scripting?
-----------------

Everything the windows do is available in Python as well. For example, link
localizations into binding events and compute their dark times:

.. code-block:: python

   from picasso import io, postprocess

   locs, info = io.load_locs("testdata_locs.hdf5")

   # Link localizations and calculate dark times
   linked_locs = postprocess.link(locs, info, r_max=0.05, max_dark_time=1)
   linked_locs = postprocess.compute_dark_times(linked_locs)

   print(f"Average bright time {linked_locs['n'].mean():.2f} frames")
   print(f"Average dark time {linked_locs['dark'].mean():.2f} frames")

More examples are in the :doc:`/postprocessing` notebooks and the
:doc:`/api/index`. Many steps also run in batch from the :doc:`/cmd`.
