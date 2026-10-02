Postprocessing
==============

Jupyter Notebooks
-----------------
For additional postprocessing steps, one might want to use Picasso as a Python
package together with other libraries. For this, check out the
`samples folder <https://github.com/jungmannlab/picasso/tree/master/samples>`_
of the repository. Here you can find sample Jupyter notebooks that show how the
Picasso package can be used. The sample data in ``samples/data`` was created
using :doc:`simulate`.

.. _postprocessing-notebook-localize:

Localizing raw movies
~~~~~~~~~~~~~~~~~~~~~
``sample_notebook_1_localize.ipynb`` shows some basic interaction with the
``picasso`` library for localizing raw movies, e.g., how to directly call its
functions from a Jupyter notebook, and how to find the function behind a
button or menu action of a Picasso GUI.

.. _postprocessing-notebook-basic-analysis:

Basic analysis
~~~~~~~~~~~~~~
``sample_notebook_2_basic_analysis.ipynb`` shows some basic interaction with
the ``picasso`` library for analyzing localizations: loading them, picking
regions of interest and inspecting the picked localizations.

.. _postprocessing-notebook-clustering:

Clustering
~~~~~~~~~~
``sample_notebook_3_clustering.ipynb`` shows how to perform clustering with
picasso: DBSCAN and the SMLM clusterer, frame analysis for sticking events,
RESI and molecular mapping.

.. _postprocessing-notebook-spinna:

SPINNA
~~~~~~
``sample_notebook_4_spinna.ipynb`` shows how to use the Python API of
:doc:`spinna` on the exemplary EGFR data in ``samples/data``.
