"""
picasso.gui.average
~~~~~~~~~~~~~~~~~~~

Graphical user interface for averaging particles.

:author: Joerg Schnitzbauer, 2015
:copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib
import os.path
import pkgutil
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

from .. import io, lib, average, __version__


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
    oversampling : float
        Number of display pixels per camera pixel.
    """

    progressMade = QtCore.pyqtSignal(int, int, pd.DataFrame, bool, int, int)

    def __init__(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        oversampling: float,
        iterations: int,
    ) -> None:
        super().__init__()
        self.locs = locs.copy()
        self.info = info
        self.oversampling = oversampling
        self.iterations = iterations

    def on_progress(
        self,
        it: int,
        total_it: int,
        locs_current: pd.DataFrame,
        group: int,
        n_groups: int,
    ) -> None:
        """Callback for progress updates from averaging process."""
        self.locs = locs_current.copy()
        self.progressMade.emit(it, total_it, self.locs, True, group, n_groups)

    def run(self) -> None:
        """Run averaging across a number of iterations."""
        self.locs = average.average(
            self.locs,
            self.info,
            self.oversampling,
            self.iterations,
            progress_callback=self.on_progress,
        )


class ParametersDialog(lib.Dialog):
    """Dialog for setting parameters - oversampling and iterations.

    ...

    Attributes
    ----------
    disp_px_size : QtWidgets.QDoubleSpinBox
        Spin box for setting the display pixel size in nm. Determines
        oversampling, see below.
    iterations : QtWidgets.QSpinBox
        Spin box for setting the number of iterations.
    oversampling : float
        Number of display pixels per camera pixel.
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
    _pixmap : QtGui.QPixmap
        Pixmap for displaying the averaged image.
    running : bool
        Flag indicating whether the averaging process is running.
    thread : Worker
        Worker thread for performing the averaging.
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

    def average(self):
        if not self.running:
            self.running = True
            oversampling = self.window.parameters_dialog.oversampling
            iterations = self.window.parameters_dialog.iterations.value()
            self.thread = Worker(
                self.locs,
                self.info,
                oversampling,
                iterations,
            )
            self.thread.progressMade.connect(self.on_progress)
            self.thread.finished.connect(self.on_finished)
            self.thread.start()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile()
        ext = os.path.splitext(path)[1].lower()
        if ext == ".hdf5":
            self.open(path)

    def on_finished(self) -> None:
        self.window.statusBar().showMessage("Done!")
        self.running = False

    def on_progress(
        self,
        it: int,
        total_it: int,
        locs: pd.DataFrame,
        update_image: bool,
        group: int,
        n_groups: int,
    ) -> None:
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

        self.window.statusBar().showMessage("Ready for processing!")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self._pixmap is not None:
            self.set_pixmap(self._pixmap)

    def save(self, path: str) -> None:
        """Save averaged localizations.

        Parameters
        ----------
        path : str
            Path to save localizations.
        """
        out_locs, info = average.prepare_locs_for_save(self.locs, self.info)
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
        parameters."""
        oversampling = self.window.parameters_dialog.oversampling
        t_min = -self.r
        t_max = self.r
        N_avg, image_avg = average.render_hist(
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

    DOCS_URL = "https://picassosr.readthedocs.io/en/latest/average.html"

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
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction("Save")
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save)
        file_menu.addAction(save_action)
        metadata_action = file_menu.addAction("Show metadata")
        metadata_action.setShortcut("Ctrl+M")
        metadata_action.triggered.connect(self.show_metadata)
        picasso_settings_action = file_menu.addAction("Picasso settings")
        picasso_settings_action.triggered.connect(
            self.user_settings_dialog.show
        )
        help_action = file_menu.addAction("Help")
        help_action.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.DOCS_URL))
        )
        process_menu = menu_bar.addMenu("Process")
        parameters_action = process_menu.addAction("Parameters")
        parameters_action.setShortcut("Ctrl+P")
        parameters_action.triggered.connect(self.parameters_dialog.show)
        average_action = process_menu.addAction("Average")
        average_action.setShortcut("Ctrl+A")
        average_action.triggered.connect(self.view.average)
        self.plugin_menu = menu_bar.addMenu("Plugins")  # do not delete

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

    app = QtWidgets.QApplication(sys.argv)
    window = Window()

    from . import plugins

    def iter_namespace(pkg):
        return pkgutil.iter_modules(pkg.__path__, pkg.__name__ + ".")

    plugins = [
        importlib.import_module(name)
        for finder, name, ispkg in iter_namespace(plugins)
    ]

    for plugin in plugins:
        p = plugin.Plugin(window)
        if p.name == "average":
            p.execute()

    window.show()

    from ..updater import setup_gui_update_check

    setup_gui_update_check(window)

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(
            window,
            "An error occured",
            message,
        )
        errorbox.exec()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
