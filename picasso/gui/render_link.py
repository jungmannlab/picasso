"""
picasso.gui.render_link
~~~~~~~~~~~~~~~~~~~~~~~

Linked windows in Picasso: Render.

View > New linked window... opens another, complete Render window that
holds its own channels. You choose which of the current window's
channels it starts with; they are taken as they are, including unsaved
changes such as filtering or drift correction. More files can be opened
in it later. Every window has its own dialogs, menus and
postprocessing; the windows of one link group mirror only the
attributes enabled in View > Link settings...:

Localizations
    Channels taken into a linked window share one set of localizations
    (in memory) with the window they came from, so filtering, drift
    correction, the Move tool, etc. in either window update both.
    Switching this off gives each window its own copy; channels taken
    while it is off are copies from the start. Only the localizations,
    metadata and drift of a shared channel are shared: adding or
    closing channels, their colors and visibility stay per window.

Navigation
    Pan (viewport center), zoom (camera pixels per screen pixel), a
    crosshair showing the cursor position of the window the mouse is
    in, and the active tool (Zoom, Pick, Measure, Move). Pan and zoom
    are linked separately, and windows of different sizes show the
    same center at the same scale.
Rendering
    Render settings (blur method and its parameters, display pixel
    size), contrast (including automatic contrast adjustments),
    colormap and render by property.
Display
    Scale bar, minimap, background color and legend, camera pixel
    size.
Picks
    Pick shape and size, or the picks themselves (which include their
    shape and size).
Slicing
    Slicing on/off, slice thickness and slice position; the position is
    matched in nm, since the slice bins depend on each window's data.

Channel colors, visibility and intensities are never linked.

A change is mirrored by applying it to the other windows' widgets as if
the user had edited them there, so each window runs its own update
logic. A shared channel is one DataFrame held by several windows: a
window that replaces it (``View.update_scene`` compares the channels
with the last seen ones) or changes it in place
(``View.invalidate_locs_index``) makes the other windows adopt it and
drop what they derived from it. Windows without loaded localizations
are skipped; when a window loads its first files, it adopts the linked
attributes of the group.
The enabled attributes are saved in the user settings.

Closing a linked window leaves the others open; closing the first
window closes all of them.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import numpy as np
from PyQt6 import QtCore, QtWidgets

from .. import io, lib, render


@dataclass(frozen=True)
class LinkCategory:
    """An attribute that can be linked between Render windows.

    Attributes
    ----------
    key : str
        Identifier, also stored in the user settings.
    label : str
        Checkbox text in the link settings dialog.
    section : str
        Group box the checkbox is placed in.
    tooltip : str
        Checkbox tooltip.
    default : bool
        Whether the category is linked when no saved settings exist.
    """

    key: str
    label: str
    section: str
    tooltip: str
    default: bool = False


CATEGORIES = (
    LinkCategory(
        "localizations",
        "Localizations of shared channels",
        "Localizations",
        "Channels taken into a linked window share one set of\n"
        "localizations with the window they came from: filtering, drift\n"
        "correction, the Move tool, etc. in either window update both.\n"
        "Switching this off gives each window its own copy; channels\n"
        "taken while it is off are copies from the start.",
        default=True,
    ),
    LinkCategory(
        "viewport_center",
        "Pan",
        "Navigation",
        "Show the same viewport center in all windows.",
        default=True,
    ),
    LinkCategory(
        "viewport_zoom",
        "Zoom",
        "Navigation",
        "Show the same scale (camera pixels per screen pixel) in all\n"
        "windows.",
        default=True,
    ),
    LinkCategory(
        "crosshair",
        "Cursor crosshair",
        "Navigation",
        "Mark the cursor position of the window the mouse is in with a\n"
        "crosshair in the other windows.",
    ),
    LinkCategory(
        "tool_mode",
        "Active tool",
        "Navigation",
        "Use the same tool (Zoom, Pick, Measure, Move) in all windows.",
    ),
    LinkCategory(
        "render_settings",
        "Render settings",
        "Rendering",
        "Blur method and its parameters, display pixel size.",
    ),
    LinkCategory(
        "contrast",
        "Contrast",
        "Rendering",
        "Minimum and maximum density, including automatic contrast\n"
        "adjustments.",
    ),
    LinkCategory(
        "colormap",
        "Colormap",
        "Rendering",
        "Colormap of single-channel data.",
    ),
    LinkCategory(
        "render_property",
        "Render by property",
        "Rendering",
        "Rendered property, its range, color steps and colormap. Only\n"
        "applied to windows whose data contain the property.",
    ),
    LinkCategory(
        "scalebar",
        "Scale bar",
        "Display",
        "Scale bar visibility, length and label.",
    ),
    LinkCategory(
        "minimap",
        "Minimap",
        "Display",
        "Minimap visibility.",
    ),
    LinkCategory(
        "background_legend",
        "Background and legend",
        "Display",
        "Background color and legend visibility.",
    ),
    LinkCategory(
        "pixelsize",
        "Camera pixel size",
        "Display",
        "Camera pixel size (nm).",
    ),
    LinkCategory(
        "pick_geometry",
        "Pick shape and size",
        "Picks",
        "Pick shape and dimensions.",
    ),
    LinkCategory(
        "picks",
        "Picks",
        "Picks",
        "The picks themselves, including their shape and size.",
    ),
    LinkCategory(
        "slicer",
        "Slicing",
        "Slicing",
        "Slicing on/off, slice thickness and slice position (matched in\n"
        "nm).",
    ),
)
CATEGORY_KEYS = tuple(category.key for category in CATEGORIES)

# categories that are one widget per attribute, mirrored in this order:
# key -> (dialog attribute of the window, widget attribute names)
WIDGET_CATEGORIES = {
    "minimap": ("display_settings_dlg", ("minimap",)),
    "pixelsize": ("display_settings_dlg", ("pixelsize",)),
    "pick_geometry": (
        "tools_settings_dialog",
        ("pick_shape", "pick_diameter", "pick_width", "pick_side_length"),
    ),
}

# widgets of the other categories whose changes are mirrored
WATCHED_WIDGETS = {
    "render_settings": (
        "display_settings_dlg",
        (
            "blur_buttongroup",
            "min_blur_width",
            "quadtree_capacity",
            "triangulation_passes",
            "triangulation_jitter",
            "triangulation_max_locs",
            "disp_px_size",
            "dynamic_disp_px",
        ),
    ),
    "contrast": ("display_settings_dlg", ("minimum", "maximum")),
    "colormap": ("display_settings_dlg", ("colormap",)),
    "render_property": (
        "display_settings_dlg",
        (
            "parameter",
            "minimum_render",
            "maximum_render",
            "color_step",
            "colormap_prop",
            "render_check",
        ),
    ),
    "scalebar": (
        "display_settings_dlg",
        (
            "scalebar_groupbox",
            "scalebar",
            "scalebar_text",
            "optimal_scalebar_check",
        ),
    ),
    "background_legend": ("dataset_dialog", ("wbackground", "legend")),
    "slicer": (
        "slicer_dialog",
        ("slicer_radio_button", "pick_slice", "sl"),
    ),
    **WIDGET_CATEGORIES,
}

# a change of shared channels that makes another window change them
# (fitting its canvas) goes back; this bounds the rounds
MAX_SHARING_ROUNDS = 10

# combo boxes whose handlers run on user interaction only
# (``activated``), such that a mirrored change has to emit it
ACTIVATED_COMBOS = ("parameter", "colormap_prop")


def widget_signal(widget: QtCore.QObject):
    """Return the signal emitted when the user changes ``widget``.

    Parameters
    ----------
    widget : QObject
        A checkbox, group box, spin box, combo box, slider or button
        group.

    Returns
    -------
    signal : pyqtBoundSignal
        The widget's change signal.
    """
    if isinstance(widget, QtWidgets.QButtonGroup):
        return widget.buttonToggled
    if isinstance(widget, QtWidgets.QGroupBox):
        return widget.toggled
    if isinstance(widget, QtWidgets.QAbstractButton):
        return widget.toggled
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentIndexChanged
    # spin boxes and sliders
    return widget.valueChanged


def mirror_widget(
    src: QtCore.QObject,
    dst: QtCore.QObject,
    activated: bool = False,
) -> None:
    """Set ``dst`` to the state of ``src`` (widgets of the same type).

    Signals are not blocked, such that ``dst`` runs its handlers as if
    the user had changed it. Unchanged widgets emit nothing.

    Parameters
    ----------
    src, dst : QObject
        Widgets of the same type, see ``widget_signal``.
    activated : bool, optional
        For combo boxes: also emit ``activated``, which handlers
        connected to user interaction only wait for. Default is False.
    """
    if isinstance(src, QtWidgets.QButtonGroup):
        # the buttons are matched by their position in the group; the
        # blur buttons' handlers run on release, hence the click
        src_buttons = src.buttons()
        dst_buttons = dst.buttons()
        checked = src.checkedButton()
        if checked is None or len(src_buttons) != len(dst_buttons):
            return
        button = dst_buttons[src_buttons.index(checked)]
        if not button.isChecked():
            button.click()
    elif isinstance(src, (QtWidgets.QGroupBox, QtWidgets.QAbstractButton)):
        dst.setChecked(src.isChecked())
    elif isinstance(src, QtWidgets.QComboBox):
        text = src.currentText()
        index = dst.findText(text)
        if index < 0 or index == dst.currentIndex():
            return
        dst.setCurrentIndex(index)
        if activated:
            dst.activated.emit(index)
    else:
        dst.setValue(src.value())


def picks_signature(view) -> tuple:
    """Cheap fingerprint of a view's picks that changes whenever picks
    are added, removed, replaced or (polygons) extended in place.

    Parameters
    ----------
    view : picasso.gui.render.View
        The view whose picks are fingerprinted.

    Returns
    -------
    signature : tuple
        Pick shape, number of picks and each pick's identity and, for
        picks stored as lists, length.
    """
    return (
        view._pick_shape,
        len(view._picks),
        tuple(
            (id(pick), len(pick) if isinstance(pick, list) else -1)
            for pick in view._picks
        ),
    )


def copy_picks(picks: list) -> list:
    """Copy picks such that in-place edits (e.g. extending a polygon)
    in one window do not leak into another.

    Parameters
    ----------
    picks : list
        Picks of a view, see ``View._picks``.

    Returns
    -------
    picks : list
        Copy of the picks; list-valued picks are copied, tuples shared.
    """
    return [list(pick) if isinstance(pick, list) else pick for pick in picks]


def has_data(window) -> bool:
    """Whether a window has localizations loaded and rendered."""
    view = window.view
    return bool(len(view.locs)) and hasattr(view, "viewport")


class LinkGroup(QtCore.QObject):
    """Render windows that mirror a chosen set of attributes.

    Parameters
    ----------
    window : picasso.gui.render.Window
        The window creating the group; it becomes the first window.

    Attributes
    ----------
    enabled : set of str
        Keys of the linked categories, see ``CATEGORIES``.
    settings_dialog : LinkSettingsDialog
        Dialog for choosing the linked categories.
    windows : list
        Linked windows; the first one owns the application, i.e.
        closing it closes all windows.
    """

    def __init__(self, window) -> None:
        super().__init__()
        self.windows = []
        self.enabled = self._load_enabled()
        self._applying = False  # stops mirrored changes from echoing
        self._pick_signatures = {}  # window -> last mirrored picks
        self._seen_locs = {}  # window -> channels at its last redraw
        self._pending = []  # changes of shared channels to pass on
        self._reference = window  # source of newly enabled categories
        self._next_number = 1
        self.settings_dialog = LinkSettingsDialog(self, window)
        self.add(window)

    @staticmethod
    def _load_enabled() -> set[str]:
        """Linked categories from the user settings; categories missing
        there (e.g. added in a later version) take their defaults."""
        settings = io.load_user_settings()
        saved = settings["Render"].get("LinkCategories")
        if not isinstance(saved, dict):
            saved = {}
        return {c.key for c in CATEGORIES if saved.get(c.key, c.default)}

    def _save_enabled(self) -> None:
        """Store the linked categories in the user settings."""
        settings = io.load_user_settings()
        settings["Render"]["LinkCategories"] = {
            c.key: c.key in self.enabled for c in CATEGORIES
        }
        io.save_user_settings(settings)

    # --- membership ---------------------------------------------------

    def add(self, window) -> None:
        """Add a window to the group and connect its widgets."""
        if window in self.windows:
            return
        window.link_group = self
        window._link_number = self._next_number
        self._next_number += 1
        self.windows.append(window)
        self.attach(window)
        self._refresh_titles()
        self.settings_dialog.update_windows()

    def remove(self, window, detach_data: bool = True) -> None:
        """Remove a window from the group (unlink or close it).

        Parameters
        ----------
        window : picasso.gui.render.Window
            Window to remove.
        detach_data : bool, optional
            Give the window its own copies of the channels it shares
            with the group, such that it stays independent. Not needed
            when the window closes. Default is True.
        """
        if window not in self.windows:
            return
        self.windows.remove(window)
        if detach_data:
            self._detach(window, self.windows)
        self._pick_signatures.pop(window, None)
        self._seen_locs.pop(window, None)
        if self._reference is window:
            self._reference = self.windows[0] if self.windows else None
        window.link_group = None
        window.view.set_link_crosshair(None)
        window.setWindowTitle(window._base_title)
        self._refresh_titles()
        self.settings_dialog.update_windows()

    def title_suffix(self, window) -> str:
        """Suffix of a window title that tells the linked windows
        apart; empty while the window is not linked to another one."""
        if len(self.windows) < 2 or window not in self.windows:
            return ""
        return f" (linked view {window._link_number})"

    def _refresh_titles(self) -> None:
        for window in self.windows:
            window.setWindowTitle(window._base_title)

    def attach(self, window) -> None:
        """Connect the change signals of a window's view and dialogs.

        Called when a window joins the group and whenever it rebuilds
        its view and dialogs (``Window.remove_locs``); connecting the
        same view twice is a no-op.
        """
        view = window.view
        if getattr(window, "_link_attached_view", None) is view:
            return
        window._link_attached_view = view
        view.viewport_changed.connect(
            partial(self._on_viewport_changed, window)
        )
        view.picks_drawn.connect(partial(self._on_picks_drawn, window))
        view.cursor_moved.connect(partial(self._on_cursor_moved, window))
        view.scene_requested.connect(partial(self._on_scene_requested, window))
        window.tools_actiongroup.triggered.connect(
            partial(self._on_changed, window, "tool_mode")
        )
        for key, (dialog_name, names) in WATCHED_WIDGETS.items():
            dialog = getattr(window, dialog_name)
            for name in names:
                widget_signal(getattr(dialog, name)).connect(
                    partial(self._on_changed, window, key)
                )
        # pick size and shape changes are part of the linked picks
        _, names = WIDGET_CATEGORIES["pick_geometry"]
        for name in names:
            widget = getattr(window.tools_settings_dialog, name)
            widget_signal(widget).connect(
                partial(self._on_changed, window, "picks")
            )

    # --- propagation --------------------------------------------------

    @contextmanager
    def muted(self):
        """Context in which no change is mirrored."""
        previous = self._applying
        self._applying = True
        try:
            yield
        finally:
            self._applying = previous

    def _targets(self, src) -> list:
        """The windows a change in ``src`` is mirrored to."""
        if self._applying or src not in self.windows or not has_data(src):
            return []
        return [w for w in self.windows if w is not src and has_data(w)]

    def notify(self, src, key: str) -> None:
        """Mirror category ``key`` from ``src`` to the other windows if
        it is linked.

        Parameters
        ----------
        src : picasso.gui.render.Window
            The window in which the change happened.
        key : str
            Category key, see ``CATEGORIES``.
        """
        if key not in self.enabled:
            return
        targets = self._targets(src)
        if not targets:
            return
        with self.muted():
            for dst in targets:
                self.apply(key, src, dst)

    def _on_changed(self, window, key: str, *args) -> None:
        self.notify(window, key)

    def _on_viewport_changed(self, window, interactive: bool) -> None:
        if {"viewport_center", "viewport_zoom"} & self.enabled:
            targets = self._targets(window)
            with self.muted():
                for dst in targets:
                    self._apply_viewport(window, dst, interactive)

    def _on_picks_drawn(self, window) -> None:
        # every pick change ends in a redraw, which is cheap to test
        if (
            "picks" not in self.enabled
            or self._applying
            or window not in self.windows
        ):
            return
        signature = picks_signature(window.view)
        if self._pick_signatures.get(window) == signature:
            return
        self._pick_signatures[window] = signature
        self.notify(window, "picks")

    def _on_cursor_moved(self, window, position) -> None:
        if "crosshair" not in self.enabled:
            return
        targets = self._targets(window)
        with self.muted():
            for dst in targets:
                dst.view.set_link_crosshair(position)

    def apply(self, key: str, src, dst) -> None:
        """Copy category ``key`` from window ``src`` to window ``dst``.

        Parameters
        ----------
        key : str
            Category key, see ``CATEGORIES``.
        src, dst : picasso.gui.render.Window
            Source and destination windows.
        """
        if key in ("viewport_center", "viewport_zoom"):
            self._apply_viewport(src, dst, interactive=False)
        elif key in ("crosshair", "localizations"):
            pass  # follow the cursor / the shared channels' changes
        elif key == "tool_mode":
            self._apply_tool_mode(src, dst)
        elif key == "render_settings":
            self._apply_render_settings(src, dst)
        elif key == "contrast":
            self._apply_contrast(src, dst)
        elif key == "colormap":
            self._apply_colormap(src, dst)
        elif key == "render_property":
            self._apply_render_property(src, dst)
        elif key == "scalebar":
            self._apply_scalebar(src, dst)
        elif key == "background_legend":
            self._apply_background_legend(src, dst)
        elif key == "picks":
            self._apply_picks(src, dst)
        elif key == "slicer":
            self._apply_slicer(src, dst)
        else:
            dialog_name, names = WIDGET_CATEGORIES[key]
            src_dialog = getattr(src, dialog_name)
            dst_dialog = getattr(dst, dialog_name)
            for name in names:
                mirror_widget(
                    getattr(src_dialog, name), getattr(dst_dialog, name)
                )

    def _apply_viewport(
        self,
        src,
        dst,
        interactive: bool,
        autoscale: bool = False,
    ) -> None:
        viewport = src.view.viewport
        center = render.viewport_center(viewport)
        scale = render.viewport_height(viewport) / max(src.view.height(), 1)
        dst.view.apply_link_viewport(
            center,
            scale,
            use_center="viewport_center" in self.enabled,
            use_scale="viewport_zoom" in self.enabled,
            interactive=interactive,
            autoscale=autoscale,
        )

    @staticmethod
    def _apply_tool_mode(src, dst) -> None:
        mode = src.view._mode
        for action in dst.tools_actiongroup.actions():
            if action.text() == mode and not action.isChecked():
                action.trigger()

    @staticmethod
    def _apply_render_settings(src, dst) -> None:
        s = src.display_settings_dlg
        d = dst.display_settings_dlg
        for name in (
            "blur_buttongroup",
            "min_blur_width",
            "quadtree_capacity",
            "triangulation_passes",
            "triangulation_jitter",
            "triangulation_max_locs",
        ):
            mirror_widget(getattr(s, name), getattr(d, name))
        # a dynamic display pixel size is computed by each window; a
        # fixed one unchecks "dynamic" on its own
        if s.dynamic_disp_px.isChecked():
            d.dynamic_disp_px.setChecked(True)
        else:
            d.disp_px_size.setValue(s.disp_px_size.value())
            d.dynamic_disp_px.setChecked(False)

    @staticmethod
    def _apply_contrast(src, dst) -> None:
        s = src.display_settings_dlg
        d = dst.display_settings_dlg
        # keep minimum < maximum at every step
        if s.minimum.value() >= d.maximum.value():
            d.maximum.setValue(s.maximum.value())
            d.minimum.setValue(s.minimum.value())
        else:
            d.minimum.setValue(s.minimum.value())
            d.maximum.setValue(s.maximum.value())

    @staticmethod
    def _apply_colormap(src, dst) -> None:
        s = src.display_settings_dlg.colormap
        d = dst.display_settings_dlg.colormap
        if s.currentText() == "Custom":
            # "Custom" asks for a file when chosen; hand over the array
            cmap = getattr(src.view, "custom_cmap", None)
            if cmap is None:
                return
            dst.view.custom_cmap = cmap
            d.blockSignals(True)
            d.setCurrentText("Custom")
            d.blockSignals(False)
            dst.view.update_scene()
        else:
            mirror_widget(s, d)

    @staticmethod
    def _apply_render_property(src, dst) -> None:
        s = src.display_settings_dlg
        d = dst.display_settings_dlg
        if d.parameter.findText(s.parameter.currentText()) < 0:
            return  # property missing from the other window's data
        # choosing a property resets its range, so it goes first
        for name in (
            "parameter",
            "minimum_render",
            "maximum_render",
            "color_step",
            "colormap_prop",
            "render_check",
        ):
            mirror_widget(
                getattr(s, name),
                getattr(d, name),
                activated=name in ACTIVATED_COMBOS,
            )

    @staticmethod
    def _apply_scalebar(src, dst) -> None:
        s = src.display_settings_dlg
        d = dst.display_settings_dlg
        mirror_widget(s.scalebar_groupbox, d.scalebar_groupbox)
        mirror_widget(s.scalebar_text, d.scalebar_text)
        # a manual length unchecks "Automatic length", so it goes first
        if not s.optimal_scalebar_check.isChecked():
            mirror_widget(s.scalebar, d.scalebar)
        mirror_widget(s.optimal_scalebar_check, d.optimal_scalebar_check)

    @staticmethod
    def _apply_background_legend(src, dst) -> None:
        s = src.dataset_dialog
        d = dst.dataset_dialog
        mirror_widget(s.wbackground, d.wbackground)
        mirror_widget(s.legend, d.legend)
        if d.background_color != s.background_color:
            d.background_color = s.background_color
            d._update_background_swatch()
            d.update_viewport()

    def _apply_picks(self, src, dst) -> None:
        # set the shape first, such that the shape combo box does not
        # ask to delete the other window's picks
        dst.view._pick_shape = src.view._pick_shape
        dst.view._picks = copy_picks(src.view._picks)
        self.apply("pick_geometry", src, dst)
        self._pick_signatures[dst] = picks_signature(dst.view)
        dst.view.update_pick_info_short()
        dst.view.update_scene(picks_only=True)

    @staticmethod
    def _apply_slicer(src, dst) -> None:
        s = src.slicer_dialog
        d = dst.slicer_dialog
        active = s.slicer_radio_button.isChecked() and hasattr(s, "bins")
        if not active:
            d.slicer_radio_button.setChecked(False)
            return
        if not any(len(z) for z in d.zcoord):
            return  # 2D data cannot be sliced
        thickness = s.pick_slice.value()
        if not hasattr(d, "bins") or d.pick_slice.value() != thickness:
            d.pick_slice.blockSignals(True)
            d.pick_slice.setValue(thickness)
            d.pick_slice.blockSignals(False)
            d.calculate_histogram()
        if len(d.bins) < 2:
            return
        # the bins depend on the data, so the slice is matched in nm
        z_min = getattr(s, "slicermin", s.bins[s.sl.value()])
        position = int(np.argmin(np.abs(d.bins[:-1] - z_min)))
        d.sl.setValue(min(position, d.sl.maximum()))
        d.slicer_radio_button.setChecked(True)

    # --- shared channels ----------------------------------------------

    def _on_scene_requested(self, window) -> None:
        # a channel replaced since the last redraw (filtering, linking,
        # undrifting, etc. assign a new DataFrame) is passed on
        current = list(window.view.locs)
        seen = self._seen_locs.get(window)
        self._seen_locs[window] = current
        if seen is None or len(seen) != len(current):
            return  # channels added or closed: nothing is replaced
        changes = [
            (old, i)
            for i, (old, new) in enumerate(zip(seen, current))
            if old is not new
        ]
        if changes:
            self.channels_changed(window, changes)

    def channels_changed(self, src, changes: list[tuple]) -> None:
        """Make the other windows adopt changed channels they share
        with ``src``.

        Parameters
        ----------
        src : picasso.gui.render.Window
            Window in which the channels changed.
        changes : list of tuples
            ``(held, i)`` per changed channel: ``held`` is the DataFrame
            the other windows hold for it (the replaced one, or the
            current one if it changed in place) and ``i`` the channel
            index in ``src``.
        """
        if "localizations" not in self.enabled or src not in self.windows:
            return
        if self._applying:
            # a window adopting a change fitted its canvas, which moved
            # shared channels again: passed on in the loop below
            self._pending.append((src, changes))
            return
        self._pending = [(src, changes)]
        with self.muted():
            for _ in range(MAX_SHARING_ROUNDS):
                if not self._pending:
                    break
                source, items = self._pending.pop(0)
                for dst in self.windows:
                    if dst is source:
                        continue
                    pairs = [
                        (j, i)
                        for held, i in items
                        for j, locs in enumerate(dst.view.locs)
                        if locs is held
                    ]
                    if pairs:
                        dst.view.adopt_shared_channels(source.view, pairs)
                        self._seen_locs[dst] = list(dst.view.locs)
        self._pending = []

    def shared_render_index(self, locs):
        """The render index a linked window built for ``locs``, or None.

        Parameters
        ----------
        locs : pd.DataFrame
            Localizations of a channel.

        Returns
        -------
        render_index : spatial_index.RenderIndexPyramid or None
            The index, which only depends on the localizations, so it
            is valid for every window holding the same DataFrame.
        """
        for window in self.windows:
            view = window.view
            for held, index in zip(view.locs, view.render_index):
                if held is locs and index is not None:
                    return index
        return None

    def _detach(self, window, others: list) -> None:
        """Give ``window`` its own copies of the channels it shares with
        ``others``."""
        held = {id(locs) for w in others for locs in w.view.locs}
        for j, locs in enumerate(window.view.locs):
            if id(locs) in held:
                window.view.detach_channel(j)
        if window in self._seen_locs:
            self._seen_locs[window] = list(window.view.locs)

    # --- enabling categories and adopting windows ---------------------

    def set_enabled(self, key: str, enabled: bool) -> None:
        """Link or unlink a category; a newly linked category is
        applied once from the reference window (the one the settings
        were opened from), such that all windows start consistent.

        Parameters
        ----------
        key : str
            Category key, see ``CATEGORIES``.
        enabled : bool
            True to link the category.
        """
        if enabled == (key in self.enabled):
            return
        if enabled:
            self.enabled.add(key)
            src = self._reference
            if src is not None:
                self.notify(src, key)
                if key == "picks":
                    self._pick_signatures[src] = picks_signature(src.view)
        else:
            self.enabled.discard(key)
            if key == "crosshair":
                for window in self.windows:
                    window.view.set_link_crosshair(None)
            elif key == "localizations":
                for i, window in enumerate(self.windows):
                    self._detach(window, self.windows[:i])
        self._save_enabled()

    def adopt(self, window) -> None:
        """Bring a window that has just loaded its first files in line
        with the group, instead of mirroring its fit-to-window view.

        Parameters
        ----------
        window : picasso.gui.render.Window
            Window whose first files finished loading.
        """
        view = window.view
        refs = [w for w in self.windows if w is not window and has_data(w)]
        with self.muted():
            if not refs:
                view.fit_in_view(autoscale=True)
                return
            ref = refs[0]
            # a viewport for the handlers the categories trigger
            view.fit_in_view()
            for key in CATEGORY_KEYS:
                if key in self.enabled and key not in (
                    "viewport_center",
                    "viewport_zoom",
                    "crosshair",
                    "localizations",
                ):
                    self.apply(key, ref, window)
            # the last render request wins, so it carries the automatic
            # contrast unless the contrast is linked
            autoscale = "contrast" not in self.enabled
            if {"viewport_center", "viewport_zoom"} & self.enabled:
                self._apply_viewport(
                    ref, window, interactive=False, autoscale=autoscale
                )
            else:
                view.fit_in_view(autoscale=autoscale)

    def show_settings(self, window) -> None:
        """Open the link settings dialog from ``window``, which becomes
        the source of newly linked categories."""
        self._reference = window
        self.settings_dialog.show()
        self.settings_dialog.raise_()


class LinkSettingsDialog(lib.Dialog):
    """Choose the attributes linked between Render windows and manage
    the linked windows.

    Parameters
    ----------
    group : LinkGroup
        The link group edited.
    window : picasso.gui.render.Window
        Parent window.
    """

    def __init__(self, group: LinkGroup, window) -> None:
        super().__init__(window)
        self.group = group
        self.setWindowTitle("Link settings")
        layout = QtWidgets.QVBoxLayout(self)

        self.checks = {}
        sections = {}
        for category in CATEGORIES:
            if category.section not in sections:
                box = QtWidgets.QGroupBox(category.section)
                QtWidgets.QVBoxLayout(box)
                layout.addWidget(box)
                sections[category.section] = box
            check = QtWidgets.QCheckBox(category.label)
            check.setToolTip(category.tooltip)
            check.setChecked(category.key in group.enabled)
            check.toggled.connect(partial(group.set_enabled, category.key))
            sections[category.section].layout().addWidget(check)
            self.checks[category.key] = check

        buttons = QtWidgets.QHBoxLayout()
        all_button = QtWidgets.QPushButton("All")
        all_button.clicked.connect(partial(self._check_all, True))
        buttons.addWidget(all_button)
        none_button = QtWidgets.QPushButton("None")
        none_button.clicked.connect(partial(self._check_all, False))
        buttons.addWidget(none_button)
        layout.addLayout(buttons)

        windows_box = QtWidgets.QGroupBox("Linked windows")
        windows_layout = QtWidgets.QVBoxLayout(windows_box)
        self.window_list = QtWidgets.QListWidget()
        self.window_list.itemDoubleClicked.connect(self._show_selected)
        windows_layout.addWidget(self.window_list)
        window_buttons = QtWidgets.QHBoxLayout()
        show_button = QtWidgets.QPushButton("Show")
        show_button.setToolTip("Bring the selected window to the front.")
        show_button.clicked.connect(self._show_selected)
        window_buttons.addWidget(show_button)
        self.unlink_button = QtWidgets.QPushButton("Unlink")
        self.unlink_button.setToolTip(
            "Stop linking the selected window; it stays open. The first\n"
            "window cannot be unlinked."
        )
        self.unlink_button.clicked.connect(self._unlink_selected)
        window_buttons.addWidget(self.unlink_button)
        windows_layout.addLayout(window_buttons)
        layout.addWidget(windows_box)

    def _check_all(self, checked: bool) -> None:
        for check in self.checks.values():
            check.setChecked(checked)

    def update_windows(self) -> None:
        """Refresh the list of linked windows."""
        self.window_list.clear()
        for window in self.group.windows:
            paths = window.view.locs_paths
            files = ", ".join(os.path.basename(p) for p in paths)
            text = f"Linked view {window._link_number}: {files or 'empty'}"
            self.window_list.addItem(text)

    def _selected_window(self):
        row = self.window_list.currentRow()
        if 0 <= row < len(self.group.windows):
            return self.group.windows[row]
        return None

    def _show_selected(self, *args) -> None:
        window = self._selected_window()
        if window is not None:
            window.show()
            window.raise_()
            window.activateWindow()

    def _unlink_selected(self) -> None:
        window = self._selected_window()
        if window is not None and window is not self.group.windows[0]:
            self.group.remove(window)

    def showEvent(self, event) -> None:
        # file names change as windows load and close channels
        self.update_windows()
        super().showEvent(event)


class NewLinkedWindowDialog(lib.Dialog):
    """Choose the channels a new linked window starts with.

    Parameters
    ----------
    paths : list of str
        Paths of the channels loaded in the current window.
    parent : QWidget
        Parent widget.
    """

    def __init__(self, paths: list[str], parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("New linked window")
        self.paths = paths
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Channels shown in the new window, as they are, including\n"
            "unsaved changes. Both windows share them, so edits in either\n"
            "window update both, unless 'Localizations of shared\n"
            "channels' is off in View > Link settings (then they are\n"
            "copies). More files can be opened in the new window later."
        )
        layout.addWidget(note)
        self.checks = []
        for path in paths:
            check = QtWidgets.QCheckBox(os.path.basename(path))
            check.setToolTip(path)
            layout.addWidget(check)
            self.checks.append(check)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_channels(self) -> list[int]:
        """Indices of the checked channels."""
        return [i for i, c in enumerate(self.checks) if c.isChecked()]
