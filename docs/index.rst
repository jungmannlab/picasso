:html_theme.sidebar_secondary.remove: true

Picasso
=======

.. raw:: html

   <div class="hero">
     <img class="only-light" src="_static/picasso-logo.png" alt="Picasso"
          onerror="this.outerHTML='&lt;p class=&quot;hero-name&quot;&gt;Picasso&lt;/p&gt;'">
     <img class="only-dark" src="_static/picasso-logo-dark.png" alt="Picasso">
     <p class="tagline">
       A collection of tools for painting super-resolution images, covering
       single-molecule localization microscopy (SMLM) analysis from raw
       movies to localization, rendering and quantification.
     </p>
   </div>

.. div:: sd-text-center sd-text-muted

   Documentation for Picasso |release|

.. div:: sd-text-center sd-mb-5 hero-buttons

   .. button-ref:: getting-started/installation
      :ref-type: doc
      :color: primary
      :class: sd-px-4 sd-fs-5 sd-mx-1

      Install

   .. button-ref:: getting-started/workflow
      :ref-type: doc
      :color: primary
      :outline:
      :class: sd-px-4 sd-fs-5 sd-mx-1

      First steps

   .. button-ref:: api/index
      :ref-type: doc
      :color: primary
      :outline:
      :class: sd-px-4 sd-fs-5 sd-mx-1

      Python API

Picasso is complemented by our `Nature Protocols publication
<https://doi.org/10.1038/nprot.2017.024>`__. This documentation covers all
updates since then.

Modules
-------

Each module opens as its own window, either from its shortcut or with
``picasso <module>`` on the command line.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card::
      :link: design
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/design.svg
         :class: module-icon dark-light
         :alt:

      **Design**
      ^^^
      Design rectangular DNA origami with DNA-PAINT docking sites and get
      the staple plates and pipetting schemes.

   .. grid-item-card::
      :link: simulate
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/simulate.svg
         :class: module-icon dark-light
         :alt:

      **Simulate**
      ^^^
      Simulate DNA-PAINT acquisitions to evaluate experimental conditions
      and generate ground-truth data.

   .. grid-item-card::
      :link: localize
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/localize.svg
         :class: module-icon dark-light
         :alt:

      **Localize**
      ^^^
      Identify and fit single-molecule spots in raw movies, in 2D or 3D,
      with Gaussian or experimental PSF models, on the CPU or GPU.

   .. grid-item-card::
      :link: filter
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/filter.svg
         :class: module-icon dark-light
         :alt:

      **Filter**
      ^^^
      Inspect localization properties in tables and histograms and filter
      out unwanted localizations.

   .. grid-item-card::
      :link: render
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/render.svg
         :class: module-icon dark-light
         :alt:

      **Render**
      ^^^
      Render, explore and analyze super-resolution images: drift
      correction, picking, 3D views, clustering, RESI and more.

   .. grid-item-card::
      :link: average
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/average.svg
         :class: module-icon dark-light
         :alt:

      **Average**
      ^^^
      Align picked structures by cross-correlation and average them into
      one particle.

   .. grid-item-card::
      :link: spinna
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/spinna.svg
         :class: module-icon dark-light
         :alt:

      **SPINNA**
      ^^^
      Analyze protein oligomerization by comparing nearest-neighbor
      distances with simulations.

   .. grid-item-card::
      :link: nanotron
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/nanotron.png
         :class: module-icon dark-light
         :alt:

      **nanoTRON**
      ^^^
      Classify nanostructures in localization data with a trained neural
      network.

   .. grid-item-card::
      :link: server
      :link-type: doc
      :class-card: sd-card-hover

      .. image:: _static/icons/server.svg
         :class: module-icon dark-light
         :alt:

      **Server**
      ^^^
      Track the quality metrics of your experiments in a local database and
      process new files automatically.

Beyond the GUI
--------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: :octicon:`code;1.5em;sd-mr-1` Python API
      :link: api/index
      :link-type: doc
      :class-card: sd-card-hover

      Use Picasso's routines in your own scripts and notebooks.

   .. grid-item-card:: :octicon:`terminal;1.5em;sd-mr-1` Command line
      :link: cmd
      :link-type: doc
      :class-card: sd-card-hover

      Batch-process files with ``picasso <command>``.

   .. grid-item-card:: :octicon:`plug;1.5em;sd-mr-1` Plugins
      :link: plugins
      :link-type: doc
      :class-card: sd-card-hover

      Install community plugins or write your own.

   .. grid-item-card:: :octicon:`file;1.5em;sd-mr-1` File formats
      :link: files
      :link-type: doc
      :class-card: sd-card-hover

      What is stored in Picasso's ``.hdf5`` and ``.yaml`` files.

Citing Picasso
--------------

.. card::
   :class-card: sd-mb-4

   If you use Picasso in your research, please cite:

   J. Schnitzbauer\*, M.T. Strauss\*, T. Schlichthaerle, F. Schueder,
   R. Jungmann. **Super-Resolution Microscopy with DNA-PAINT.** *Nature
   Protocols* 12, 1198–1228 (2017). DOI: `10.1038/nprot.2017.024
   <https://doi.org/10.1038/nprot.2017.024>`__

   +++
   Many features are based on further publications, see
   :doc:`getting-started/citing` for the full list.

.. toctree::
   :hidden:

   Getting started <getting-started>
   User guide <user-guide>
   Reference <reference>
   Release notes <changelog>
   Contributing <development>
