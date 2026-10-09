.. _render-tools:

Tools
=====

The tools decide what the mouse does on the rendered image. Select one in the
``Tools`` menu or with its shortcut; ``Tools settings`` sets how each tool
behaves and looks.

.. _render-zoom:

Zoom
----

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+Z`

Selects the zoom tool. Dragging a rectangle with the left mouse button zooms
into it; dragging with the right mouse button pans. Panning is also available
in every tool by dragging with the left mouse button while holding
:kbd:`Ctrl` (:kbd:`Cmd` on macOS). See :ref:`render-navigation` for all
controls.

.. _render-pick:

Pick
----

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+P`

Selects the pick tool. The mouse can now be used for picking localizations.
The user can set the pick shape in the ``Tools settings`` (:kbd:`Ctrl+T`)
dialog; see :ref:`render-pick-shapes` for all shapes.

- The default shape is Circle with the diameter to be set.
- For rectangles, the user draws the length, while the width is controlled via
  a parameter for all drawn rectangles, similar to the diameter for circular
  picks.
- For a polygonal pick, the user clicks with the left button to draw the
  desired polygon. The right button deletes the last selected vertex. The
  polygon can be closed by clicking with the left button on the starting
  vertex.

.. _render-measure:

Measure
-------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+M`

Selects the measure tool, which is used to measure distances on the rendered
image.

While the tool is active, the cursor is shown as a crosshair that follows the
mouse. **Left click** drops a measurement point: each new point is connected
to the previous one by a line, and the running total distance (in nm) is
displayed live next to the line as you move the mouse, before the next point
is even placed. Chaining several left clicks measures a multi-segment path.

**Right click** has two functions:

* The **first** right click *freezes* the current measurement set: the
  crosshair stops following the mouse and the measured path stays drawn on the
  image. A new, independent set of measurements can then be started simply by
  left-clicking again.
* While in this frozen state, a **further** right click *deletes* the most
  recently finalized set. Repeating it removes the previous sets one by one.

Distances and lines are only drawn within a set, never across sets, so
multiple independent measurements can be displayed at the same time.

.. _render-move:

Move
----

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+G`

Selects the move tool, which changes the x and y coordinates of localizations
by dragging them with the left mouse button, e.g., to register channels by
eye.

- The channels that are dragged together are selected in the
  ``Tools settings`` (:kbd:`Ctrl+T`) dialog: *Select...* opens a list of the
  loaded channels with a checkbox each. By default, the first channel is
  dragged.
- *Undo last move* in the ``Tools settings`` dialog reverses the moves one by
  one.

Normally, localizations outside the image (x or y at or beyond ``Width`` or
``Height`` in the metadata, or negative) would be removed when saving.
Instead, the image (the canvas) is fitted to the localizations after every
move:

* Dragging beyond the right or bottom edge increases ``Width`` or ``Height``.
* Dragging beyond the left or top edge translates all channels, picks and
  measured points by the same whole number of camera pixels, so that no
  coordinate is negative and the channels stay registered. ``Width`` and
  ``Height`` grow by the same amount, so the camera field of view stays inside
  the canvas. The translation is saved in the metadata as
  ``Canvas offset x (cam. px)`` and ``Canvas offset y (cam. px)``; subtract it
  to return to the camera coordinates, e.g., for picks saved before the
  translation.
* The canvas shrinks again when the localizations are moved back, but it is
  never smaller than the camera image, whose size is saved as
  ``Camera Width`` and ``Camera Height``. For example, dragging a channel
  beyond the left edge and back to where it was restores the original canvas
  and coordinates.

A channel loaded later, or saved with a different canvas offset, is brought
into the same frame as the loaded channels.

The shift of each channel done with the move tool is saved in its metadata as
``Manual shift x (cam. px)`` and ``Manual shift y (cam. px)``.

.. _render-tools-settings:

Tools settings
--------------

.. rst-class:: shortcut

Shortcut: :kbd:`Ctrl+T`

Define the settings of the tools:

- The **Pick** section sets:

  - the pick shape and its size (``Diameter``, ``Width``, ``Side length`` or
    ``Stroke width``, see :ref:`render-pick-shapes`);
  - ``Pick similar +/- range (std)`` (2 by default), the tolerance of
    :ref:`Pick similar <render-pick-similar>`. That tool computes the number of
    localizations and their RMSD from the center of mass in each of your
    picks, and accepts a new region only if both lie within the mean ± this
    many standard deviations of your picks. Larger values find more, less
    similar structures. For rectangular picks, the RMSD along and across the
    pick's axis are checked separately;
  - ``Annotate picks``, which shows the index of each pick next to it;
  - ``Display circular picks as points``.
- The **Move** section selects the channels dragged with the move tool and
  undoes the last move, see :ref:`render-move`.

.. _render-tool-appearance:

Appearance
~~~~~~~~~~

How the tools are drawn is set in the **Appearance** section at the bottom,
hidden by default; click its title to show it. It has one tab per tool:
**Pick**, **Measure** and **Move** (the label showing the shift while
dragging). The settings apply to the main window, to exported images and, for
the Measure tool, to the 3D window. If the dialog does not fit on the screen,
it scrolls.

*Color*
   *Auto* is yellow on a black and red on a white background. You can choose a
   preset color, or *Custom...* to pick any color.
*Line*
   Solid, dashed, dotted or dash-dot lines. The crosses of the Measure tool
   are always solid.
*Width*
   Line width in screen pixels. Lines wider than one pixel are smoothed
   (antialiased).
*Opacity*
   Opacity of the lines and labels.
*Fill* (picks only)
   Opacity of the fill of closed picks, in the line color; 0% draws outlines
   only. *Default* fills only brush picks. A polygon is filled once it is
   closed.
*Label size*
   Size of the pick indices (see *Annotate picks*), the measured distances and
   the shift label, in screen pixels.
*While drawing* (picks only)
   Color of a rectangle, box or brush stroke that is still being dragged.
*Center line* (picks only)
   Draws the line along the center of rectangular picks, from the start to the
   end point. On by default.
*Marker size* (Measure only)
   Size of the crosses marking the measured points.

*Reset* restores the default appearance. The appearance is saved when Render
is closed and restored at the next start (``ToolStyles`` in the ``Render``
section of the :ref:`user-settings-file`).
