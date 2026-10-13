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

.. note::

   These pages do not list every function in the package, only those that
   are useful in your own scripts. Helpers that serve the Picasso windows or
   other modules internally are left out. Each module lists its documented
   functions and classes in its ``__all__``.

Input and output
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.io

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
