Design
======

``Picasso: Design`` lets you draw a pattern of DNA-PAINT binding sites on a
rectangular DNA origami and generates the sequences, the pipetting scheme and
the folding scheme needed to make it. A saved design can be loaded into
:doc:`simulate`.

.. figure:: /images/design.png
   :width: 600px
   :class: screenshot
   :alt: Picasso Design window with the hexagon canvas of a rectangular DNA origami and the color palette on the right

   The hexagon canvas of ``Picasso: Design``; each hexagon is one staple of
   the flat origami sheet.

Design rectangular DNA origami
------------------------------

1. Start the design module. A canvas with hexagons, corresponding to the flat
   origami sheet, will appear.
2. Select a color by clicking on the color palette on the right side. Each
   color corresponds to a different extension that can be defined later.
3. Design a pattern of DNA-PAINT binding sites by clicking on the hexagons.

   - The default state of a hexagon (gray color) indicates that this staple
     will not be extended. Click on the ``Clear`` button to reset all
     hexagons.
   - Some hexagons are marked with a ``V`` after coloring. This indicates that
     this staple, in contrast to all the others, is pointing downwards at its
     3' end.

4. Click on ``Extensions`` to define the sequence for each extension.
5. Use ``Get plates`` to get an Excel sheet with all needed sequences in a
   plate format.
6. Use ``Pipetting scheme`` and select the folder with your plate list to
   generate a pipetting scheme.
7. Use ``Folding scheme`` to get a table for pipetting folding mixes.

A design can be saved and loaded using the ``Save`` and ``Load`` buttons. A
saved design can also be loaded into ``Picasso: Simulate``.
