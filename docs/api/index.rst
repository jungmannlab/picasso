Python API
==========

Everything the Picasso windows do is built on the ``picasso`` package, which
you can use in your own scripts and notebooks once Picasso is installed with
``pip install picassosr`` (see :doc:`/getting-started/installation`).

.. code-block:: python

   from picasso import io, postprocess

   locs, info = io.load_locs("my_locs.hdf5")
   linked = postprocess.link(locs, info)

The :doc:`/notebooks` walk through complete analyses.

Picasso reports what it does through Python's :mod:`logging`, with one
logger per module (``picasso.io``, ``picasso.localize``, ...). Warnings,
e.g., about localizations dropped on loading or metadata that could not be
read, are shown without any setup; to see the information messages too, or
to send them to a file, configure logging:

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)
   logging.getLogger("picasso").setLevel(logging.INFO)

.. note::

   These pages do not list every function in the package, only those that
   are useful in your own scripts. Helpers that serve the Picasso windows or
   other modules internally are left out. Each module lists its documented
   functions and classes in its ``__all__``. These names are the public API,
   which stays compatible within a major version, see :doc:`/versioning`.

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
   picasso.registration
   picasso.stitch
   picasso.spt
   picasso.transforms
   picasso.masking
   picasso.expression

Rendering
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.render

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
