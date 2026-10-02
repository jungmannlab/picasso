"""
picasso.gui.average
~~~~~~~~~~~~~~~~~~~

Graphical user interface for averaging particles.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

from .. import io, lib, average, render, __version__, docs_url
from .app import run_gui
from . import theme


class PoolWorker(QtCore.QThread):
    """Worker thread starting the pool of averaging processes.

    Starting the processes takes several seconds, so it is done in the
    background, right after a file has been loaded. The pool is then
    reused by every averaging run on that file.

    Attributes
    ----------
    locs : pd.DataFrame
        Localizations with group indices (``group`` column).
    pool : average.AveragePool
        The pool that was started, available once the thread finished.
    """

    def __init__(self, locs: pd.DataFrame) -> None:
        super().__init__()
        self.locs = locs
        self.pool = None

    def run(self) -> None:
        """Start the pool of worker processes."""
        self.pool = average.AveragePool(self.locs)


class Worker(QtCore.QThread):
    """Worker thread for processing image alignment.

    ...

    Attributes
    ----------
    info : list[dict]
        Metadata for localizations.
    iterations : int
        Number of iterations to average over.
    locs : pd.DataFrame
        Localizations with group indices (``group`` column).
    display_px_size : float
        Display pixel size in nm used in averaging.
    pool : average.AveragePool or None
        Pool of worker processes to average with. If None, a pool is
        started and shut down by the averaging function itself.
    """

    progressMade = QtCore.pyqtSignal(int, int, pd.DataFrame, bool, int, int)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        display_px_size: float,
        iterations: int,
        pool: average.AveragePool | None = None,
    ) -> None:
        super().__init__()
        self.locs = locs.copy()
        self.info = info
        self.display_px_size = display_px_size
        self.iterations = iterations
        self.pool = pool
        self.was_aborted = False

    def on_progress(
        self,
        it: int,
        total_it: int,
        locs_current: pd.DataFrame,
        group: int,
        n_groups: int,
    ) -> None:
        """Callback for progress updates from the averaging process.

        Parameters
        ----------
        it, total_it : int
            Iteration just finished and the total number of iterations.
        locs_current : pd.DataFrame
            The localizations as they stand after this iteration.
        group, n_groups : int
            Group just finished and the total number of groups.
        """
        self.locs = locs_current.copy()
        self.progressMade.emit(it, total_it, self.locs, True, group, n_groups)

    def run(self) -> None:
        """Run averaging across a number of iterations."""
        result = average.average(
            self.locs,
            self.info,
            display_pixel_size=self.display_px_size,
            iterations=self.iterations,
            progress_callback=self.on_progress,
            abort_callback=self.isInterruptionRequested,
            pool=self.pool,
        )
        if result is None:
            self.was_aborted = True
            self.aborted.emit()
        else:
            self.locs = result


class ParametersDialog(lib.Dialog):
    """Dialog for setting parameters - display pixel size and iterations.

    ...

    Attributes
    ----------
    disp_px_size : QtWidgets.QDoubleSpinBox
        Spin box for setting the display pixel size in nm. Determines
        oversampling, see below.
    iterations : QtWidgets.QSpinBox
        Spin box for setting the number of averaging iterations.
    oversampling : float
        Number of display pixels per camera pixel, calculated from the
        display pixel size and the camera pixel size from metadata.
    window : QtWidgets.QMainWindow
        Main window instance.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.oversampling = 10.0  # just some value when starting the module
        self.setWindowTitle("Parameters")
        self.setModal(False)
        grid = QtWidgets.QGridLayout(self)

        disp_px_size_label = QtWidgets.QLabel("Display pixel size (nm):")
        disp_px_size_label.setToolTip(
            "Display pixel size in nm used in averaging."
        )
        grid.addWidget(disp_px_size_label, 0, 0)
        self.disp_px_size = QtWidgets.QDoubleSpinBox()
        self.disp_px_size.setRange(0.01, 1e4)
        self.disp_px_size.setValue(10)
        self.disp_px_size.setDecimals(2)
        self.disp_px_size.setSingleStep(0.01)
        self.disp_px_size.setKeyboardTracking(False)
        self.disp_px_size.valueChanged.connect(self.on_disp_px_size_changed)
        grid.addWidget(self.disp_px_size, 0, 1)

        iter_label = QtWidgets.QLabel("Iterations:")
        iter_label.setToolTip("Number of averaging iterations.")
        grid.addWidget(iter_label, 1, 0)
        self.iterations = QtWidgets.QSpinBox()
        self.iterations.setRange(1, int(1e7))
        self.iterations.setValue(3)
        grid.addWidget(self.iterations, 1, 1)

    def on_disp_px_size_changed(self) -> None:
        """Update oversampling (number of display pixels per camera
        pixel) when display pixel size is changed."""
        if not hasattr(self.window.view, "locs"):  # no file loaded yet
            return
        camera_px = lib.get_from_metadata(
            self.window.view.info, "Pixelsize", raise_error=True
        )
        self.oversampling = camera_px / self.disp_px_size.value()
        self.window.view.update_image()


class View(QtWidgets.QLabel):
    """QLabel for displaying the averaged image.

    ...

    Attributes
    ----------
    avg_history : list of dicts
        Stores the used display pixel size and iterations across
        multiple rounds of averaging.
    _pixmap : QtGui.QPixmap
        Pixmap for displaying the averaged image.
    running : bool
        Flag indicating whether the averaging process is running.
    thread : Worker
        Worker thread for performing the averaging.
    pool : average.AveragePool or None
        Pool of worker processes, started once per loaded file and
        reused by every averaging run.
    pool_thread : PoolWorker or None
        Worker thread starting the pool in the background.
    window : QtWidgets.QMainWindow
        Main window instance.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__()
        self.window = window
        self.setMinimumSize(1, 1)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self._pixmap = None
        self.running = False
        self.thread = None
        self.pool = None
        self.pool_thread = None
        self.avg_history = []

    def average(self):
        """Start the averaging in a worker thread, reading the parameters
        from the parameters dialog."""
        if not self.running:
            self.running = True
            display_px_size = (
                self.window.parameters_dialog.disp_px_size.value()
            )
            iterations = self.window.parameters_dialog.iterations.value()
            self.window.statusBar().showMessage("Preparing for averaging...")
            self.thread = Worker(
                self.locs,
                self.info,
                display_px_size,
                iterations,
                self.wait_for_pool(),
            )
            self.thread.progressMade.connect(self.on_progress)
            self.thread.aborted.connect(self.on_aborted)
            self.thread.finished.connect(self.on_finished)
            self.window.abort_action.setEnabled(True)
            self.thread.start()

    def abort(self) -> None:
        """Request interruption of the running averaging thread."""
        if self.running and self.thread is not None:
            self.thread.requestInterruption()
            self.window.statusBar().showMessage("Aborting...")
            self.window.abort_action.setEnabled(False)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept a drag that carries file URLs.

        Parameters
        ----------
        event : QtGui.QDragEnterEvent
            The Qt drag event.
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Open the first dropped file, if it is an ``.hdf5``.

        Parameters
        ----------
        event : QtGui.QDropEvent
            The Qt drop event.
        """
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile()
        ext = os.path.splitext(path)[1].lower()
        if ext == ".hdf5":
            self.open(path)

    def on_finished(self) -> None:
        """Record the finished run in the history and re-enable the UI."""
        if self.thread is not None and self.thread.was_aborted:
            self.window.statusBar().showMessage("Aborted.")
        else:
            if self.thread is not None:
                self.avg_history.append(
                    {
                        "disp_px_size": self.thread.display_px_size,
                        "it": self.thread.iterations,
                    }
                )
            self.window.statusBar().showMessage("Done!")
        self.running = False
        self.window.abort_action.setEnabled(False)
        if self.pool is not None and self.pool.closed:
            # aborting terminates the pool, so start a new one
            self.pool = None
            self.start_pool()

    def on_aborted(self) -> None:
        """Handle abortion of the averaging thread."""
        self.window.statusBar().showMessage("Aborted.")
        self.window.abort_action.setEnabled(False)

    def on_progress(
        self,
        it: int,
        total_it: int,
        locs: pd.DataFrame,
        update_image: bool,
        group: int,
        n_groups: int,
    ) -> None:
        """Show the intermediate result of one averaging iteration.

        Parameters
        ----------
        it, total_it : int
            Iteration just finished and the total number of iterations.
        locs : pd.DataFrame
            The localizations as they stand after this iteration.
        update_image : bool
            Whether to redraw the rendered average.
        group, n_groups : int
            Group just finished and the total number of groups.
        """
        self.locs = locs.copy()
        if update_image:
            self.update_image()
        self.window.statusBar().showMessage(
            f"Iteration {it}/{total_it} — group {group}/{n_groups}"
        )

    def open(self, path: str) -> None:
        """Load a localization file and preset the pool process.

        Parameters
        ----------
        path : str
            Path to the localization file.
        """
        self.path = path
        try:
            self.locs, self.info = io.load_locs(path, qt_parent=self)
        except io.NoMetadataFileError:
            return
        self.avg_history = []
        if "group" not in self.locs.columns:
            message = (
                "Loaded file contains no group information. Please load"
                " localizations that were picked."
            )
            QtWidgets.QMessageBox.warning(self, "Warning", message)
            return
        group_index = average.build_group_index(self.locs)
        self.locs = average.com_align(self.locs, group_index)
        self.r = 2 * np.sqrt(
            (self.locs["x"] ** 2 + self.locs["y"] ** 2).mean()
        )
        self.window.parameters_dialog.on_disp_px_size_changed()
        self.update_image()
        self.start_pool()

        self.window.statusBar().showMessage("Ready for processing!")

    def start_pool(self) -> None:
        """Start the pool of averaging processes in the background.

        Any pool that is still around (e.g., from a previously loaded
        file) is shut down first.
        """
        self.shutdown_pool()
        self.pool_thread = PoolWorker(self.locs)
        self.pool_thread.start()

    def wait_for_pool(self) -> average.AveragePool | None:
        """Return the pool of averaging processes, waiting for it to be
        started if necessary.

        Returns
        -------
        pool : average.AveragePool or None
            The pool to average with, None if none was started.
        """
        if self.pool_thread is not None:
            self.pool_thread.wait()
            self.pool = self.pool_thread.pool
            self.pool_thread = None
        return self.pool

    def shutdown_pool(self) -> None:
        """Shut down the pool of averaging processes, if there is one."""
        if self.pool_thread is not None:
            self.pool_thread.wait()
            self.pool = self.pool_thread.pool
            self.pool_thread = None
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Rescale the displayed image to the new widget size.

        Parameters
        ----------
        event : QtGui.QResizeEvent
            The Qt resize event.
        """
        if self._pixmap is not None:
            self.set_pixmap(self._pixmap)

    def save(self, path: str) -> None:
        """Save averaged localizations.

        Parameters
        ----------
        path : str
            Path to save localizations.
        """
        display_pixel_size = self.window.parameters_dialog.disp_px_size.value()
        iterations = self.window.parameters_dialog.iterations.value()
        params = {"disp_px_size": display_pixel_size, "it": iterations}
        out_locs, info = average.prepare_locs_for_save(
            self.locs, self.info, params
        )
        if self.avg_history:
            info[-1]["Rounds"] = [
                {
                    "Display pixel size (nm)": r["disp_px_size"],
                    "Iterations": r["it"],
                }
                for r in self.avg_history
            ]
        io.save_locs(path, out_locs, info)
        self.window.statusBar().showMessage(f"File saved to {path}.")

    def set_image(self, image: lib.FloatArray2D) -> None:
        """Sets the new image to be displayed.

        Parameters
        ----------
        image : np.ndarray
            The image to be displayed. Shape (height, width).
        """
        cmap = np.uint8(np.round(255 * plt.get_cmap("magma")(np.arange(256))))
        image /= image.max()
        image = np.minimum(image, 1.0)
        image = np.round(255 * image).astype("uint8")
        Y, X = image.shape
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
        self._bgra[..., 0] = cmap[:, 2][image]
        self._bgra[..., 1] = cmap[:, 1][image]
        self._bgra[..., 2] = cmap[:, 0][image]
        self._bgra[..., 3] = 255
        qimage = QtGui.QImage(
            self._bgra.data, X, Y, QtGui.QImage.Format.Format_RGB32
        )
        self._pixmap = QtGui.QPixmap.fromImage(qimage)
        self.set_pixmap(self._pixmap)

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        """Display a pixmap, scaled into the widget and keeping its aspect
        ratio.

        Parameters
        ----------
        pixmap : QtGui.QPixmap
            The image to show.
        """
        self.setPixmap(
            pixmap.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation,
            )
        )

    def update_image(self, *args) -> None:
        """Update the displayed image based on the changed display
        parameters.

        Parameters
        ----------
        *args
            Ignored; accepted so the method can be wired directly to Qt
            signals that pass a value.
        """
        oversampling = self.window.parameters_dialog.oversampling
        t_min = -self.r
        t_max = self.r
        N_avg, image_avg = render.render_hist_numba(
            self.locs["x"].to_numpy(),
            self.locs["y"].to_numpy(),
            oversampling,
            t_min,
            t_max,
        )
        self.set_image(image_avg)


class Window(QtWidgets.QMainWindow):
    """Main window.

    ...

    Attributes
    ----------
    view : View
        The main view widget for displayed averaged image.
    parameters_dialog : ParametersDialog
        The dialog for adjusting processing parameters.
    """

    DOCS_URL = docs_url("average.html")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Picasso v{__version__}: Average")
        self.resize(512, 512)
        self.user_settings_dialog = lib.UserSettingsDialog(self)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "average.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.view = View(self)
        self.setCentralWidget(self.view)
        self.parameters_dialog = ParametersDialog(self)
        self.metadata_dialog = lib.MetadataDialog(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open...")
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction("Save...")
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save)
        file_menu.addAction(save_action)
        metadata_action = file_menu.addAction("Show metadata...")
        metadata_action.setIcon(theme.icon("metadata"))
        metadata_action.setShortcut("Ctrl+M")
        metadata_action.triggered.connect(self.show_metadata)
        picasso_settings_action = file_menu.addAction("Picasso settings...")
        picasso_settings_action.setIcon(theme.icon("picasso-settings"))
        picasso_settings_action.triggered.connect(
            self.user_settings_dialog.show
        )
        theme.add_menu_action(file_menu)
        help_action = file_menu.addAction("Help")
        help_action.setIcon(theme.icon("help"))
        help_action.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.DOCS_URL))
        )
        process_menu = menu_bar.addMenu("Process")
        parameters_action = process_menu.addAction("Parameters...")
        parameters_action.setShortcut("Ctrl+P")
        parameters_action.triggered.connect(self.parameters_dialog.show)
        average_action = process_menu.addAction("Average")
        average_action.setShortcut("Ctrl+A")
        average_action.triggered.connect(self.view.average)
        self.abort_action = process_menu.addAction("Abort")
        self.abort_action.setShortcut("Ctrl+.")
        self.abort_action.triggered.connect(self.view.abort)
        self.abort_action.setEnabled(False)
        self.plugin_menu = menu_bar.addMenu("Plugins")  # do not delete

        # toolbar of the most used actions, shared with the menus
        self.toolbar = theme.add_toolbar(
            self,
            "Average toolbar",
            [
                (open_action, "open", "Open"),
                (save_action, "save", "Save"),
                None,
                (parameters_action, "parameters", "Parameters"),
                (average_action, "average"),
                (self.abort_action, "abort"),
            ],
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Shut down the averaging processes before closing.

        Parameters
        ----------
        event : QtGui.QCloseEvent
            The Qt close event.
        """
        self.view.shutdown_pool()
        super().closeEvent(event)

    def show_metadata(self) -> None:
        """Open the metadata dialog."""
        if not hasattr(self.view, "info"):
            QtWidgets.QMessageBox.information(
                self, "Metadata", "No file loaded."
            )
            return
        label = os.path.basename(self.view.path)
        self.metadata_dialog.set_infos(self.view.info, labels=label)
        self.metadata_dialog.show()
        self.metadata_dialog.raise_()

    def open(self) -> None:
        """Open the dialog for opening a file to load."""
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open localizations", filter="*.hdf5"
        )
        if path:
            self.view.open(path)

    def save(self) -> None:
        """Open the dialog for saving averaged localizations."""
        out_path = os.path.splitext(self.view.path)[0] + "_avg.hdf5"
        path, ext = lib.get_save_filename_ext_dialog(
            self,
            "Save localizations",
            out_path,
            filter="*.hdf5",
            check_ext=".yaml",
        )
        if path:
            self.view.save(path)


def main() -> None:
    """Start Picasso: Average - see ``picasso.gui.app.run_gui``."""
    sys.exit(run_gui(Window, "average"))


if __name__ == "__main__":
    main()
