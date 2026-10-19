"""
picasso.gui.render_spt
~~~~~~~~~~~~~~~~~~~~~~

Single-particle tracking in Picasso: Render, see ``picasso.spt``.

Postprocess > Single-particle tracking > Track particles...
    Links the localizations of a channel into tracks
    (``TrackDialog``). The track id of every localization is stored in
    the ``group`` column, so a single tracked channel is colored by
    track.
Postprocess > Single-particle tracking > Analyze tracks...
    Computes the mean squared displacement (MSD), the jump distances
    and a summary of every track of a tracked channel
    (``TrackAnalysisDialog``) and shows them in ``TrackAnalysisWindow``,
    which saves the results.
Display settings > Tracks
    Draws every track as a line through its localizations
    (``TracksGroupBox``, ``picasso.render.draw_tracks``).

``SPTController`` connects these to the ``View`` of Render; it is
created by the ``View`` as ``view.spt``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import weakref
from dataclasses import replace

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .. import io, lib, render, spt, docs_url
from .overlay_style import AUTO_COLOR, OverlayStyleWidget

#: Key of the SPT settings in the Render section of the user settings.
SETTINGS_KEY = "SPT"
#: Fields of the line appearance, see ``OverlayStyleWidget``.
STYLE_FIELDS = ("color", "line_style", "line_width", "opacity")
#: Frame time (ms) suggested if neither the metadata nor the user
#: settings give one.
DEFAULT_FRAME_TIME = 10.0
#: Maximum jump distance (nm) suggested for tracking.
DEFAULT_MAX_JUMP = 500.0
#: Localization error options of the jump distance fit, see
#: ``TrackAnalysisDialog``.
SIGMA_NONE = "None (apparent D)"
SIGMA_PRECISION = "Median localization precision"
SIGMA_CUSTOM = "Custom"


def _load_spt_settings() -> dict:
    """The SPT section of the Render user settings (empty if
    missing)."""
    render_settings = io.load_user_settings()["Render"]
    if not isinstance(render_settings, dict):
        return {}
    stored = render_settings.get(SETTINGS_KEY)
    return dict(stored) if isinstance(stored, dict) else {}


def _save_spt_settings(values: dict) -> None:
    """Update the SPT section of the Render user settings."""
    settings = io.load_user_settings()
    if not isinstance(settings["Render"], dict):
        return
    stored = settings["Render"].get(SETTINGS_KEY)
    stored = dict(stored) if isinstance(stored, dict) else {}
    stored.update(values)
    settings["Render"][SETTINGS_KEY] = stored
    io.save_user_settings(settings)


def _number(value, default: float) -> float:
    """``value`` if it is a positive number, else ``default``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isfinite(value) and value > 0:
            return float(value)
    return default


def _button_box(dialog: QtWidgets.QDialog) -> QtWidgets.QDialogButtonBox:
    """OK and Cancel buttons connected to ``dialog``."""
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        QtCore.Qt.Orientation.Horizontal,
        dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    return buttons


class TrackDialog(lib.Dialog):
    """Choose the parameters for linking localizations into tracks,
    see ``picasso.spt.link_tracks``.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    frame_time : float or None, optional
        Frame time (ms) from the metadata of the channel; None suggests
        the last one used. Default None.

    Attributes
    ----------
    max_jump : QDoubleSpinBox
        Maximum distance between consecutive localizations of a track
        (nm).
    max_gap : QSpinBox
        Maximum number of frames a track may skip.
    min_length : QSpinBox
        Minimum number of localizations of a track.
    frame_time : QDoubleSpinBox
        Time between two consecutive frames (ms), stored in the
        metadata for the analysis of the tracks.
    """

    DOCS_URL = docs_url("render/tracking.html#render-spt-track")

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        frame_time: float | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Track particles")
        stored = _load_spt_settings()
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(lib.HelpButton(self.DOCS_URL), 0, 0)

        jump_label = QtWidgets.QLabel("Max. jump (nm):")
        jump_label.setToolTip(
            "Maximum distance between consecutive localizations of a\n"
            "track. Should be larger than the jumps of the fastest\n"
            "particles (about 3 * sqrt(4 * D * frame time)) and smaller\n"
            "than the distance between particles."
        )
        grid.addWidget(jump_label, 1, 0)
        self.max_jump = QtWidgets.QDoubleSpinBox()
        self.max_jump.setRange(0.1, 1e6)
        self.max_jump.setDecimals(1)
        self.max_jump.setSingleStep(50)
        self.max_jump.setValue(
            _number(stored.get("Max. jump (nm)"), DEFAULT_MAX_JUMP)
        )
        self.max_jump.setToolTip(jump_label.toolTip())
        grid.addWidget(self.max_jump, 1, 1)

        gap_label = QtWidgets.QLabel("Max. gap (frames):")
        gap_label.setToolTip(
            "Maximum number of frames a track may skip, e.g., when the\n"
            "fluorophore blinks or the particle is not detected.\n"
            "0 links consecutive frames only."
        )
        grid.addWidget(gap_label, 2, 0)
        self.max_gap = QtWidgets.QSpinBox()
        self.max_gap.setRange(0, 1000)
        gap = stored.get("Max. gap (frames)")
        self.max_gap.setValue(
            gap if isinstance(gap, int) and gap >= 0 else spt.MAX_GAP
        )
        self.max_gap.setToolTip(gap_label.toolTip())
        grid.addWidget(self.max_gap, 2, 1)

        length_label = QtWidgets.QLabel("Min. track length:")
        length_label.setToolTip(
            "Minimum number of localizations of a track; the\n"
            "localizations of shorter tracks are removed. Fast\n"
            "particles leave the focus quickly, so a large minimum\n"
            "favors slow particles."
        )
        grid.addWidget(length_label, 3, 0)
        self.min_length = QtWidgets.QSpinBox()
        self.min_length.setRange(1, 1_000_000)
        length = stored.get("Min. track length")
        self.min_length.setValue(
            length
            if isinstance(length, int) and length >= 1
            else spt.MIN_LENGTH
        )
        self.min_length.setToolTip(length_label.toolTip())
        grid.addWidget(self.min_length, 3, 1)

        time_label = QtWidgets.QLabel("Frame time (ms):")
        time_label.setToolTip(
            "Time between the starts of two consecutive frames (not the\n"
            "exposure time). Not used for linking, but stored with the\n"
            "tracks for their analysis."
        )
        grid.addWidget(time_label, 4, 0)
        self.frame_time = QtWidgets.QDoubleSpinBox()
        self.frame_time.setRange(0.001, 1e6)
        self.frame_time.setDecimals(3)
        self.frame_time.setValue(
            _number(
                frame_time,
                _number(stored.get(spt.FRAME_TIME_KEY), DEFAULT_FRAME_TIME),
            )
        )
        self.frame_time.setToolTip(time_label.toolTip())
        grid.addWidget(self.frame_time, 4, 1)
        vbox.addLayout(grid)
        vbox.addWidget(_button_box(self))

    def params(self) -> dict:
        """The chosen parameters."""
        return {
            "max_jump": self.max_jump.value(),
            "max_gap": self.max_gap.value(),
            "min_length": self.min_length.value(),
            "frame_time": self.frame_time.value(),
        }

    @staticmethod
    def getParams(
        parent: QtWidgets.QWidget | None = None,
        frame_time: float | None = None,
    ) -> tuple[dict, bool]:
        """Open the dialog and return the chosen parameters.

        Parameters
        ----------
        parent : QWidget or None, optional
            Parent widget. Default None.
        frame_time : float or None, optional
            Frame time (ms) from the metadata. Default None.

        Returns
        -------
        params : dict
            ``max_jump`` (nm), ``max_gap``, ``min_length`` and
            ``frame_time`` (ms).
        ok : bool
            Whether the dialog was accepted. The parameters are then
            remembered in the user settings.
        """
        dialog = TrackDialog(parent, frame_time=frame_time)
        ok = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        params = dialog.params()
        if ok:
            _save_spt_settings(
                {
                    "Max. jump (nm)": params["max_jump"],
                    "Max. gap (frames)": params["max_gap"],
                    "Min. track length": params["min_length"],
                    spt.FRAME_TIME_KEY: params["frame_time"],
                }
            )
        return params, ok


class TrackAnalysisDialog(lib.Dialog):
    """Choose the parameters of the diffusion analysis of tracks, see
    ``picasso.spt.analyze_tracks``.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    frame_time : float or None, optional
        Frame time (ms) from the metadata of the channel; None suggests
        the last one used. Default None.
    precision : float or None, optional
        Median localization precision (nm) of the channel; None if it
        has none. Default None.
    max_jump : float or None, optional
        Maximum jump distance (nm) of the tracking; None if unknown.
        Default None.

    Attributes
    ----------
    frame_time : QDoubleSpinBox
        Time between two consecutive frames (ms).
    max_lag : QSpinBox
        Largest time lag of the MSD (frames).
    n_fit : QSpinBox
        Number of time lags fitted in the MSD analysis.
    sigma_mode : QComboBox
        Localization error of the jump distance fit: none, the median
        localization precision or a custom value.
    sigma : QDoubleSpinBox
        Custom localization error (nm), shown for "Custom" only.
    truncate : QCheckBox
        Whether the jump distance fit corrects for the maximum jump
        distance of the tracking.
    """

    DOCS_URL = docs_url("render/tracking.html#render-spt-analysis")

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        frame_time: float | None = None,
        precision: float | None = None,
        max_jump: float | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analyze tracks")
        stored = _load_spt_settings()
        self._precision = precision
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(lib.HelpButton(self.DOCS_URL), 0, 0)
        time_label = QtWidgets.QLabel("Frame time (ms):")
        time_label.setToolTip(
            "Time between the starts of two consecutive frames (not the\n"
            "exposure time)."
        )
        grid.addWidget(time_label, 1, 0)
        self.frame_time = QtWidgets.QDoubleSpinBox()
        self.frame_time.setRange(0.001, 1e6)
        self.frame_time.setDecimals(3)
        self.frame_time.setValue(
            _number(
                frame_time,
                _number(stored.get(spt.FRAME_TIME_KEY), DEFAULT_FRAME_TIME),
            )
        )
        self.frame_time.setToolTip(time_label.toolTip())
        grid.addWidget(self.frame_time, 1, 1)
        vbox.addLayout(grid)

        msd_box = lib.CollapsibleGroupBox("Mean squared displacement")
        msd_grid = QtWidgets.QGridLayout(msd_box.content)
        lag_label = QtWidgets.QLabel("Max. lag (frames):")
        lag_label.setToolTip("Largest time lag of the MSD curve.")
        msd_grid.addWidget(lag_label, 0, 0)
        self.max_lag = QtWidgets.QSpinBox()
        self.max_lag.setRange(2, 10_000)
        lag = stored.get("Max. lag (frames)")
        self.max_lag.setValue(
            lag if isinstance(lag, int) and lag >= 2 else spt.MAX_LAG
        )
        self.max_lag.setToolTip(lag_label.toolTip())
        msd_grid.addWidget(self.max_lag, 0, 1)
        fit_label = QtWidgets.QLabel("Fit points:")
        fit_label.setToolTip(
            "Number of time lags fitted with MSD = 4 D tau + 4 sigma^2.\n"
            "The first few lags are the most precise. A track needs a\n"
            "pair of localizations at each of them for its own D, i.e.,\n"
            "at least this number plus one localizations."
        )
        msd_grid.addWidget(fit_label, 1, 0)
        self.n_fit = QtWidgets.QSpinBox()
        self.n_fit.setRange(2, self.max_lag.value())
        n_fit = stored.get("MSD fit points")
        self.n_fit.setValue(
            n_fit if isinstance(n_fit, int) and n_fit >= 2 else spt.N_FIT
        )
        self.n_fit.setToolTip(fit_label.toolTip())
        self.max_lag.valueChanged.connect(self.n_fit.setMaximum)
        msd_grid.addWidget(self.n_fit, 1, 1)
        vbox.addWidget(msd_box)

        jd_box = lib.CollapsibleGroupBox("Jump distances")
        jd_grid = QtWidgets.QGridLayout(jd_box.content)
        sigma_label = QtWidgets.QLabel("Localization error:")
        sigma_label.setToolTip(
            "Localization error in the fit of the jump distances. One\n"
            "time lag cannot tell it from diffusion, so it is fixed:\n"
            "None gives apparent diffusion coefficients (D plus\n"
            "error^2 / frame time). The median localization precision\n"
            "is a lower bound of the real error of moving particles."
        )
        jd_grid.addWidget(sigma_label, 0, 0)
        self.sigma_mode = QtWidgets.QComboBox()
        self.sigma_mode.addItem(SIGMA_NONE, SIGMA_NONE)
        if precision is not None and np.isfinite(precision):
            self.sigma_mode.addItem(
                f"{SIGMA_PRECISION} ({precision:.1f} nm)", SIGMA_PRECISION
            )
        self.sigma_mode.addItem(SIGMA_CUSTOM, SIGMA_CUSTOM)
        self.sigma_mode.setToolTip(sigma_label.toolTip())
        jd_grid.addWidget(self.sigma_mode, 0, 1)
        self.sigma_label = QtWidgets.QLabel("Error (nm):")
        jd_grid.addWidget(self.sigma_label, 1, 0)
        self.sigma = QtWidgets.QDoubleSpinBox()
        self.sigma.setRange(0, 1e4)
        self.sigma.setDecimals(1)
        self.sigma.setValue(
            _number(stored.get("Localization error (nm)"), 20.0)
        )
        jd_grid.addWidget(self.sigma, 1, 1)
        mode = stored.get("Localization error")
        index = self.sigma_mode.findData(mode)
        if index >= 0:
            self.sigma_mode.setCurrentIndex(index)
        self.sigma_mode.currentIndexChanged.connect(self._update_sigma)
        self._update_sigma()

        jump_text = (
            "Correct for the max. jump distance"
            if max_jump is None
            else f"Correct for the max. jump distance ({max_jump:.0f} nm)"
        )
        self.truncate = QtWidgets.QCheckBox(jump_text)
        self.truncate.setToolTip(
            "Fast particles may jump further than the max. jump distance\n"
            "of the tracking; those jumps are missing, which biases\n"
            "their D low. If checked, the fit accounts for the cut-off,\n"
            "and the fractions refer to all jumps, including the\n"
            "missing ones. Not part of the original protocol."
        )
        self.truncate.setChecked(
            max_jump is not None and stored.get("Max. jump correction") is True
        )
        self.truncate.setEnabled(max_jump is not None)
        jd_grid.addWidget(self.truncate, 2, 0, 1, 2)
        vbox.addWidget(jd_box)
        lib.remember_expanded_sections(self, "Render", "Track analysis")
        vbox.addWidget(_button_box(self))

    def _update_sigma(self, *args) -> None:
        """Show the custom localization error only if it is used."""
        custom = self.sigma_mode.currentData() == SIGMA_CUSTOM
        self.sigma_label.setVisible(custom)
        self.sigma.setVisible(custom)

    def params(self) -> dict:
        """The chosen parameters, as taken by
        ``picasso.spt.analyze_tracks``."""
        mode = self.sigma_mode.currentData()
        if mode == SIGMA_PRECISION:
            sigma = "precision"
        elif mode == SIGMA_CUSTOM:
            sigma = self.sigma.value()
        else:
            sigma = 0.0
        return {
            "frame_time": self.frame_time.value(),
            "max_lag": self.max_lag.value(),
            "n_fit": self.n_fit.value(),
            "sigma": sigma,
            "truncate": self.truncate.isEnabled()
            and self.truncate.isChecked(),
        }

    @staticmethod
    def getParams(
        parent: QtWidgets.QWidget | None = None,
        frame_time: float | None = None,
        precision: float | None = None,
        max_jump: float | None = None,
    ) -> tuple[dict, bool]:
        """Open the dialog and return the chosen parameters.

        Parameters
        ----------
        parent : QWidget or None, optional
            Parent widget. Default None.
        frame_time : float or None, optional
            Frame time (ms) from the metadata. Default None.
        precision : float or None, optional
            Median localization precision (nm). Default None.
        max_jump : float or None, optional
            Maximum jump distance (nm) of the tracking. Default None.

        Returns
        -------
        params : dict
            Keyword arguments of ``picasso.spt.analyze_tracks``:
            ``frame_time``, ``max_lag``, ``n_fit``, ``sigma`` and
            ``truncate``.
        ok : bool
            Whether the dialog was accepted. The parameters are then
            remembered in the user settings.
        """
        dialog = TrackAnalysisDialog(
            parent,
            frame_time=frame_time,
            precision=precision,
            max_jump=max_jump,
        )
        ok = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        params = dialog.params()
        if ok:
            values = {
                spt.FRAME_TIME_KEY: params["frame_time"],
                "Max. lag (frames)": params["max_lag"],
                "MSD fit points": params["n_fit"],
                "Localization error": dialog.sigma_mode.currentData(),
            }
            if dialog.sigma_mode.currentData() == SIGMA_CUSTOM:
                values["Localization error (nm)"] = params["sigma"]
            if dialog.truncate.isEnabled():
                values["Max. jump correction"] = params["truncate"]
            _save_spt_settings(values)
        return params, ok


class TrackAnalysisWindow(QtWidgets.QWidget):
    """Results of the diffusion analysis of one channel's tracks:
    a summary, plots of the MSD, the jump distances and the track
    lengths, and a button that saves the results.

    Parameters
    ----------
    result : dict
        Result of ``picasso.spt.analyze_tracks``.
    path : str
        Path of the analyzed channel, used for the title and the
        suggested file names.
    icon : QIcon or None, optional
        Window icon. Default None.

    Attributes
    ----------
    tabs : QTabWidget
        One ``lib.GenericPlotWindow`` per plot.
    msd_plot, jd_plot, length_plot : lib.GenericPlotWindow
        The plots of the MSD, the jump distances and the track lengths.
    populations : QComboBox
        Number of populations of the jump distance fit shown.
    summary : QLabel
        Numbers of tracks and jumps and the fitted diffusion
        coefficients.
    """

    def __init__(
        self,
        result: dict,
        path: str,
        icon: QtGui.QIcon | None = None,
    ) -> None:
        super().__init__()
        self.result = result
        self.path = path
        self.setWindowTitle(f"Track analysis - {os.path.basename(path)}")
        if icon is not None:
            self.setWindowIcon(icon)
        self.resize(900, 650)
        vbox = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel()
        self.summary.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.summary.setWordWrap(True)
        vbox.addWidget(self.summary)

        self.tabs = QtWidgets.QTabWidget()
        self.msd_plot = lib.GenericPlotWindow("MSD", "render")
        self.jd_plot = lib.GenericPlotWindow("Jump distances", "render")
        self.length_plot = lib.GenericPlotWindow("Track lengths", "render")
        self.tabs.addTab(self.msd_plot, "MSD")
        self.tabs.addTab(self.jd_plot, "Jump distances")
        self.tabs.addTab(self.length_plot, "Track lengths")
        vbox.addWidget(self.tabs)

        # number of populations of the jump distance fit shown
        self.populations = QtWidgets.QComboBox()
        self.populations.setToolTip(
            "Number of diffusing populations fitted to the jump\n"
            "distances. A lower BIC (Bayesian information criterion)\n"
            "is better; the lowest is selected first."
        )
        fits = {k: f for k, f in result["jd_fits"].items() if f is not None}
        for k, fit in fits.items():
            label = f"{k} population{'s' if k > 1 else ''}"
            self.populations.addItem(f"{label} (BIC {fit['bic']:.1f})", k)
        if fits:
            best = min(fits, key=lambda k: fits[k]["bic"])
            self.populations.setCurrentIndex(self.populations.findData(best))
        self.populations.currentIndexChanged.connect(self._on_populations)
        self.jd_plot.toolbar.addSeparator()
        self.jd_plot.toolbar.addWidget(QtWidgets.QLabel("Fit: "))
        self.jd_plot.toolbar.addWidget(self.populations)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        save_button = QtWidgets.QPushButton("Save results...")
        save_button.setToolTip(
            "Save the summary of every track (CSV), the ensemble MSD\n"
            "(CSV) and the parameters and fits (YAML)."
        )
        save_button.clicked.connect(self.save)
        buttons.addWidget(save_button)
        vbox.addLayout(buttons)

        self._plot_msd()
        self._plot_jump_distances()
        self._plot_track_lengths()
        self._update_summary()

    def jd_fit(self) -> dict | None:
        """The jump distance fit shown, or None if none could be
        fitted."""
        k = self.populations.currentData()
        return None if k is None else self.result["jd_fits"][k]

    def _plot_msd(self) -> None:
        page = self.msd_plot

        def draw() -> None:
            with page.plot_context():
                spt.plot_msd(
                    self.result["msd"], self.result["msd_fit"], page.figure
                )

        draw()
        page.redraw = draw
        page.canvas.draw()

    def _plot_jump_distances(self) -> None:
        page = self.jd_plot

        def draw() -> None:
            style = page.plot_style
            with page.plot_context():
                spt.plot_jump_distances(
                    self.result["jumps"],
                    self.jd_fit(),
                    page.figure,
                    fill=style.hist_fill,
                    outline=style.hist_outline,
                )

        draw()
        page.redraw = draw
        page.canvas.draw()

    def _plot_track_lengths(self) -> None:
        page = self.length_plot

        def draw() -> None:
            style = page.plot_style
            with page.plot_context():
                spt.plot_track_lengths(
                    self.result["table"],
                    page.figure,
                    fill=style.hist_fill,
                    outline=style.hist_outline,
                )

        draw()
        page.redraw = draw
        page.canvas.draw()

    def _on_populations(self, *args) -> None:
        self.jd_plot.redraw()
        self.jd_plot.canvas.draw_idle()
        self._update_summary()

    def _update_summary(self) -> None:
        """Show the numbers of tracks and jumps and the fits."""
        result = self.result
        lines = [
            f"{result['n_tracks']:,} tracks, {result['n_locs']:,} "
            f"localizations, {len(result['jumps']):,} jumps between "
            "consecutive frames."
        ]
        msd_fit = result["msd_fit"]
        if np.isfinite(msd_fit["D"]):
            text = f"MSD: D = {msd_fit['D']:.4g} µm²/s"
            if np.isfinite(msd_fit["sigma"]):
                text += (
                    ", apparent localization error "
                    f"{msd_fit['sigma']:.1f} nm"
                )
            lines.append(text + ".")
        else:
            lines.append("MSD: too few pairs of localizations to fit.")
        fit = self.jd_fit()
        if fit is not None:
            parts = [
                f"D = {d:.4g} µm²/s ({100 * a:.1f}%)"
                for d, a in zip(fit["D"], fit["fractions"])
            ]
            kind = "apparent " if fit["sigma"] == 0 else ""
            lines.append(
                f"Jump distances, {kind}diffusion coefficients: "
                + ", ".join(parts)
                + "."
            )
            if not fit["success"]:
                lines.append(f"The fit did not converge: {fit['message']}")
        elif result["jd_errors"]:
            lines.append(
                "Jump distances: " + next(iter(result["jd_errors"].values()))
            )
        self.summary.setText("\n".join(lines))

    def save(self) -> None:
        """Ask for a file name and save the results, see
        ``picasso.spt.save_analysis``."""
        suffix = "_track_stats.csv"
        default = os.path.splitext(self.path)[0] + suffix
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save track analysis", default, filter="*.csv"
        )
        if not path:
            return
        if path.endswith(suffix):
            base = path[: -len(suffix)]
        else:
            base = os.path.splitext(path)[0]
        paths = spt.save_analysis(base, self.result)
        QtWidgets.QMessageBox.information(
            self,
            "Track analysis saved",
            "Saved:\n" + "\n".join(paths),
        )


class TracksGroupBox(lib.CollapsibleGroupBox):
    """Checkable, collapsible group box that shows the tracks of tracked
    channels as lines and sets their look.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget. Default None.

    Attributes
    ----------
    changed : pyqtSignal
        Emitted when the tracks are shown or hidden or any setting
        changes.
    max_tracks : QSpinBox
        Maximum number of tracks drawn in one view.
    status : QLabel
        Number of tracks in view and drawn.
    style_widget : OverlayStyleWidget
        Color, line style, line width and opacity of the lines.
    """

    changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Tracks", parent=parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.setToolTip(
            "Draw every track of the tracked channels (Postprocess >\n"
            "Single-particle tracking) as a line through its\n"
            "localizations."
        )
        grid = QtWidgets.QGridLayout(self.content)
        max_label = QtWidgets.QLabel("Max. tracks drawn:")
        max_label.setToolTip(
            "If more tracks are in view, a fixed subset of them is\n"
            "drawn (every k-th track), so that panning stays smooth."
        )
        grid.addWidget(max_label, 0, 0)
        self.max_tracks = QtWidgets.QSpinBox()
        self.max_tracks.setRange(100, 10_000_000)
        self.max_tracks.setSingleStep(1000)
        self.max_tracks.setValue(render.MAX_TRACKS_DEFAULT)
        self.max_tracks.setKeyboardTracking(False)
        self.max_tracks.setToolTip(max_label.toolTip())
        self.max_tracks.valueChanged.connect(self.changed)
        grid.addWidget(self.max_tracks, 0, 1)
        self.status = QtWidgets.QLabel("")
        grid.addWidget(self.status, 1, 0, 1, 2)
        self.style_widget = OverlayStyleWidget(
            STYLE_FIELDS, auto_colors=("#FF0000", "#00FFFF")
        )
        # "Auto" colors the tracks like their localizations
        self.style_widget.color.setItemData(
            0,
            "The colors of the localizations: one per track for a\n"
            "single channel, else the channel colors",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.style_widget.changed.connect(self.changed)
        grid.addWidget(self.style_widget, 2, 0, 1, 2)
        self.toggled.connect(self.changed)

    def style(self) -> render.OverlayStyle:
        """The look of the lines; the color is None for "Auto"."""
        style = self.style_widget.style()
        if self.style_widget.value("color") == AUTO_COLOR:
            style = replace(style, color=None)
        return style

    def set_status(self, n_visible: int, n_drawn: int) -> None:
        """Show how many tracks are in view and drawn."""
        if n_drawn < n_visible:
            text = f"{n_drawn:,} of {n_visible:,} tracks in view drawn"
        else:
            text = f"{n_visible:,} tracks in view"
        self.status.setText(text)

    def settings(self) -> dict:
        """All settings except whether the tracks are shown, keyed as
        in the user settings."""
        return {
            "Max. tracks drawn": self.max_tracks.value(),
            **self.style_widget.settings(),
        }

    def load_settings(self, settings: dict | None) -> None:
        """Set the settings from the user settings; missing or invalid
        values keep the current ones. Emits ``changed`` once."""
        if not isinstance(settings, dict):
            return
        value = settings.get("Max. tracks drawn")
        if isinstance(value, int) and not isinstance(value, bool):
            self.max_tracks.blockSignals(True)
            self.max_tracks.setValue(value)
            self.max_tracks.blockSignals(False)
        self.style_widget.blockSignals(True)
        self.style_widget.load_settings(settings)
        self.style_widget.blockSignals(False)
        self.changed.emit()


class SPTController:
    """Single-particle tracking in a Render ``View``: tracking and
    analysis of channels, and drawing their tracks.

    Parameters
    ----------
    view : picasso.gui.render.View
        The view whose channels are tracked.
    """

    def __init__(self, view: QtWidgets.QWidget) -> None:
        self.view = view
        # channel -> (weak reference to its localizations, their
        # ``spt.track_segments`` or None if they are not tracked)
        self._segments = {}
        self.analysis_windows = []

    def invalidate(self, channel: int | None = None) -> None:
        """Forget the track segments of one or all channels. Does not
        read the localizations, so it is safe while idle.

        Parameters
        ----------
        channel : int, optional
            Channel whose segments are dropped. Default None (all).
        """
        if channel is None:
            self._segments.clear()
        else:
            self._segments.pop(channel, None)

    def _channel_segments(self, channel: int) -> tuple | None:
        """Track segments, track ids and start rows of a tracked
        channel (see ``picasso.spt.track_segments``), or None if the
        channel is not tracked. Cached until the channel's
        localizations change."""
        locs = self.view.locs[channel]
        cached = self._segments.get(channel)
        if cached is not None and cached[0]() is locs:
            return cached[1]
        if spt.is_tracked(locs, self.view.infos[channel]):
            data = spt.track_segments(locs)
        else:
            data = None
        self._segments[channel] = (weakref.ref(locs), data)
        return data

    def draw_tracks(
        self,
        image: QtGui.QImage,
        viewport: tuple | None = None,
        channels: list[int] | None = None,
    ) -> QtGui.QImage:
        """Draw the tracks of the shown, tracked channels, if the
        Tracks box in the Display Settings is checked.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations.
        viewport : tuple, optional
            Field of view shown in ``image``. Default is the current
            viewport.
        channels : list of int, optional
            Channels whose tracks are drawn. Default is all shown ones.

        Returns
        -------
        image : QImage
            Image with the drawn tracks.
        """
        view = self.view
        box = view.window.display_settings_dlg.tracks_box
        if not box.isChecked() or not view.n_channels:
            return image
        viewport = view.viewport if viewport is None else viewport
        dataset_dialog = view.window.dataset_dialog
        n_channels = len(view.locs_paths)
        if channels is None:
            channels = range(n_channels)
        invert = dataset_dialog.wbackground.isChecked()
        max_tracks = box.max_tracks.value()
        segments, color_index, palette = [], [], []
        n_visible = n_drawn = 0
        for channel in channels:
            if not dataset_dialog.checks[channel].isChecked():
                continue
            data = self._channel_segments(channel)
            if data is None:
                continue
            channel_segments, track_id, start_row = data
            mask, n_vis, n_drw = render.select_track_segments(
                channel_segments, track_id, viewport, max_tracks
            )
            n_visible += n_vis
            n_drawn += n_drw
            segments.append(channel_segments[mask])
            if n_channels == 1 and len(view.group_color) == len(
                view.locs[channel]
            ):
                # one color per track, as the localizations
                palette = [
                    tuple(int(round(255 * v)) for v in rgb)
                    for rgb in lib.get_colors(render.N_GROUP_COLORS)
                ]
                color_index.append(
                    np.asarray(view.group_color)[start_row[mask]]
                )
            else:
                color_index.append(np.full(mask.sum(), len(palette)))
                palette.append(dataset_dialog.legend_color_8bit(channel))
        box.set_status(n_visible, n_drawn)
        if not segments:
            return image
        if invert:  # the rendered localizations are inverted, too
            palette = [tuple(255 - c for c in rgb) for rgb in palette]
        return render.draw_tracks(
            image,
            viewport,
            np.concatenate(segments),
            np.concatenate(color_index),
            palette,
            box.style(),
        )

    def track(self) -> None:
        """Link the localizations of a channel (or all) into tracks,
        see ``picasso.spt.link_tracks``. Linked localizations (with a
        ``len`` column) and channels that are already tracked are
        skipped."""
        view = self.view
        channel = view.get_channel_all_seq("Track particles")
        if channel is None:
            return
        n_channels = len(view.locs_paths)
        channels = (
            list(range(n_channels)) if channel == n_channels else [channel]
        )
        linked = [c for c in channels if "len" in view.locs[c].columns]
        tracked = [
            c
            for c in channels
            if c not in linked and spt.is_tracked(view.locs[c], view.infos[c])
        ]
        channels = [c for c in channels if c not in linked + tracked]
        skipped = []
        if linked:
            skipped.append(
                "Linked localizations (binding events) cannot be "
                "tracked:\n" + self._names(linked)
            )
        if tracked:
            skipped.append("Already tracked:\n" + self._names(tracked))
        if not channels:
            QtWidgets.QMessageBox.information(
                view, "Track particles", "\n\n".join(skipped)
            )
            return
        frame_time = lib.get_from_metadata(
            view.infos[channels[0]], spt.FRAME_TIME_KEY
        )
        params, ok = TrackDialog.getParams(view.window, frame_time=frame_time)
        if not ok:
            return
        max_jump = params["max_jump"]  # nm
        jobs = [(c, view.locs[c], view.infos[c]) for c in channels]

        def compute(progress: lib.TaskProgress) -> list[tuple]:
            results = []
            for i, (c, locs, info) in enumerate(jobs):
                description = "Tracking particles..."
                if len(jobs) > 1:
                    description = (
                        f"Tracking particles (channel {i + 1}/{len(jobs)})..."
                    )
                progress.phase(description, 0)  # cancellation point
                tracked_locs, new_info = spt.link_tracks(
                    locs,
                    info,
                    max_jump,
                    max_gap=params["max_gap"],
                    min_length=params["min_length"],
                    frame_time=params["frame_time"],
                )
                results.append((c, tracked_locs, new_info))
            return results

        def apply(results: list[tuple]) -> None:
            empty = []
            applied = False
            for c, tracked_locs, new_info in results:
                if len(tracked_locs) == 0:
                    empty.append(c)
                    continue
                entry = dict(new_info[-1])
                entry.update(lib.step_info("Render : Track"))
                entry["Max. jump (nm)"] = float(params["max_jump"])
                view.locs[c] = tracked_locs
                # new list, the old one may be shared with linked windows
                view.infos[c] = new_info[:-1] + [entry]
                applied = True
            if applied:
                if len(view.locs_paths) == 1:
                    view.group_color = render.get_group_color(
                        view.locs[0], shuffle=True
                    )
                self.invalidate()
                # show the tracks, drawn by the redraw below
                box = view.window.display_settings_dlg.tracks_box
                box.blockSignals(True)
                box.setChecked(True)
                box.blockSignals(False)
                view.update_scene(resample_locs=True)
            if empty:
                skipped.append(
                    "No tracks found, the localizations were left "
                    "unchanged:\n" + self._names(empty)
                )
            if skipped:
                QtWidgets.QMessageBox.information(
                    view, "Track particles", "\n\n".join(skipped)
                )

        lib.run_task(
            compute,
            "Tracking particles...",
            view,
            apply,
            title="Track particles",
        )

    def analyze(self) -> None:
        """Analyze the tracks of a channel (MSD, jump distances and a
        summary of every track) and show the results, see
        ``picasso.spt.analyze_tracks``."""
        view = self.view
        channel = view.get_channel("Analyze tracks")
        if channel is None:
            return
        locs = view.locs[channel]
        info = view.infos[channel]
        if not spt.is_tracked(locs, info):
            QtWidgets.QMessageBox.information(
                view,
                "Analyze tracks",
                "The localizations are not tracked. Track them first:\n"
                "Postprocess > Single-particle tracking > Track "
                "particles.",
            )
            return
        precision = None
        if "lpx" in locs.columns and "lpy" in locs.columns:
            precision = spt.median_precision(locs)  # nm
        max_jump = spt.max_jump_nm(info)
        params, ok = TrackAnalysisDialog.getParams(
            view.window,
            frame_time=lib.get_from_metadata(info, spt.FRAME_TIME_KEY),
            precision=precision,
            max_jump=max_jump,
        )
        if not ok:
            return
        path = view.locs_paths[channel]

        def compute(progress: lib.TaskProgress) -> dict:
            return spt.analyze_tracks(
                locs,
                info,
                progress=progress,
                **params,
            )

        def show(result: dict) -> None:
            window = TrackAnalysisWindow(
                result, path, icon=getattr(view.window, "icon", None)
            )
            # forget windows closed meanwhile, keep the new one alive
            self.analysis_windows = [
                w for w in self.analysis_windows if w.isVisible()
            ] + [window]
            window.show()

        lib.run_task(
            compute,
            "Analyzing tracks...",
            view,
            show,
            title="Analyze tracks",
        )

    def _names(self, channels: list[int]) -> str:
        """File names of channels, one per line."""
        return "\n".join(
            os.path.basename(self.view.locs_paths[c]) for c in channels
        )
