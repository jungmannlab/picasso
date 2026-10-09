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

Input and output
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.io
   picasso.spatial_index

Localization
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.localize
   picasso.gausslq
   picasso.gaussmle
   picasso.avgroi
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
   picasso.transforms
   picasso.masking

Rendering
---------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :recursive:

   picasso.render

Simulation, design and analysis
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   picasso.simulate
   picasso.design
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
