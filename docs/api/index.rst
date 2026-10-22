Python API
==========

Everything the Picasso windows do is built on the ``picasso`` package, which
you can use in your own scripts and notebooks once Picasso is installed with
``pip install picassosr`` (see :doc:`/getting-started/installation`).

.. code-block:: python

   from picasso import io, postprocess

   locs, info = io.load_locs("my_locs.hdf5")
   linked_locs, linked_info = postprocess.link(locs, info, r_max=10.0)
   io.save_locs("my_locs_link.hdf5", linked_locs, linked_info)

The :doc:`/notebooks` walk through complete analyses. The functions on
localizations follow the pattern of this example, see
:ref:`api-locs-info`.

Picasso reports what it does through Python's :mod:`logging`, with one
logger per module (``picasso.io``, ``picasso.localize``, ...). Warnings,
e.g., about localizations dropped on loading or metadata that could not be
read, are shown without any setup; to see the information messages too, or
to send them to a file, configure logging:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)
   logging.getLogger("picasso").setLevel(logging.INFO)

Functions that take a while report their progress through one ``progress``
argument: ``"console"`` shows a progress bar for each step in the console,
None (the default) reports nothing, and an object with the
:class:`picasso.lib.Progress` interface (e.g., a subclass of
:class:`picasso.lib.MockProgress`) receives the label and size of each step
and the number of items done:

.. code-block:: python

   from picasso import lib, postprocess

   picked = postprocess.picked_locs(
       locs,
       info,
       picks=picks,
       pick_shape="Circle",
       pick_size=100,
       progress="console",
   )

   class PrintProgress(lib.MockProgress):
       def set_value(self, value, *args, **kwargs):
           print(f"{self.description_base}: {value} / {self.maximum()}")

   groups = postprocess.groupprops(locs, progress=PrintProgress())

.. note::

   These pages do not list every function in the package, only those that
   are useful in your own scripts. Helpers that serve the Picasso windows or
   other modules internally are left out. Each module lists its documented
   functions and classes in its ``__all__``. These names are the public API,
   which stays compatible within a major version, see :doc:`/versioning`.

.. _api-locs-info:

Localizations and metadata
--------------------------

The functions on localizations share one pattern:

- The localizations (a ``pandas.DataFrame``) and their metadata (the list of
  dictionaries ``io.load_locs`` returns) are the first two arguments,
  ``f(locs, info, ...)``; functions on several channels take lists,
  ``f(locs, infos, ...)``. All other arguments are passed by name, e.g.,
  ``postprocess.link(locs, info, r_max=10.0)``. Functions that write files
  take the path first, ``io.save_locs(path, locs, info)``.
- Functions that do not need the metadata accept ``info=None``.
- Functions that return new localizations, e.g., ``postprocess.link``,
  ``postprocess.undrift`` or ``clusterer.dbscan``, also return their
  complete metadata: ``info`` with an entry recording the step and its
  parameters appended. Further results follow, e.g., ``locs, info, drift =
  postprocess.undrift(...)``. The returned metadata can be saved as it is;
  ``info`` itself is not modified.
- Identifying and fitting a movie work the same way, with the metadata of
  the movie: ``ids, ids_info = localize.identify(movie, movie_info, ...)``,
  then ``locs, locs_info = localize.fit(movie, ids_info, ...)``.

To record a step of your own in the same way, append an entry with
:func:`picasso.lib.append_metadata` and :func:`picasso.lib.step_info`:

.. code-block:: python

   from picasso import io, lib

   locs, info = io.load_locs("my_locs.hdf5")
   bright = locs[locs["photons"] > 1000]
   bright_info = lib.append_metadata(
       info, lib.step_info("Photon cut", **{"Min. photons": 1000})
   )
   io.save_locs("my_locs_bright.hdf5", bright, bright_info)

Input and output
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.io
   picasso.legacy

Localization
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.localize
   picasso.zfit
   picasso.spline
   picasso.registration
   picasso.transforms
   picasso.wavelet
   picasso.scmos
   picasso.imageprocess

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:

   picasso.fitting

Postprocessing
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.postprocess
   picasso.aim
   picasso.clusterer
   picasso.g5m
   picasso.spt
   picasso.masking
   picasso.expression

Rendering
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.render
   picasso.stitch

Simulation and analysis
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.simulate
   picasso.average
   picasso.spinna
   picasso.nanotron

Utilities
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.lib
   picasso.plugins
