"""
    picasso.gui.average
    ~~~~~~~~~~~~~~~~~~~

    Graphical user interface for averaging particles.

    :author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import functools
import multiprocessing
import os.path
import sys
import time
import traceback
import importlib
import pkgutil
from multiprocessing import sharedctypes

import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import io, lib, render, __version__


@numba.jit(nopython=True, nogil=True)
def render_hist(
    x: np.ndarray,
    y: np.ndarray,
    oversampling: float,
    t_min: float,
    t_max: float,
) -> tuple[int, np.ndarray]:
    """Calculate 2D histogram of xy coordinates.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of xy coordinates.
    oversampling : float
        Number of histogram pixels per camera pixel.
    t_min, t_max : float
        Minimum and maximum bounds of the histogram.

    Returns
    n : int
        Number of localizations in the histogram.
    image : np.ndarray
        2D histogram of xy coordinates.
    """
    n_pixel = int(np.ceil(oversampling * (t_max - t_min)))
    in_view = (x > t_min) & (y > t_min) & (x < t_max) & (y < t_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - t_min)
    y = oversampling * (y - t_min)
    image = np.zeros((n_pixel, n_pixel), dtype=np.float32)
    render._fill(image, x, y)
    return len(x), image


def compute_xcorr(CF_image_avg: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Compute cross-correlation between two images.

    Parameters
    ----------
    CF_image_avg : np.ndarray
        Conjugate Fourier transform of the average image.
    image : np.ndarray
        Image to correlate with the average image.

    Returns
    -------
    xcorr : np.ndarray
        Cross-correlation of the two images.
    """
    F_image = np.fft.fft2(image)
    xcorr = np.fft.fftshift(np.real(np.fft.ifft2((F_image * CF_image_avg))))
    return xcorr


def align_group(
    angles: np.ndarray,
    oversampling: float,
    t_min: float,
    t_max: float,
    CF_image_avg: np.ndarray,
    image_half: float,
    counter: None,
    lock: None,
    group: int,
) -> None:
    """Align (shift and rotate) images.

    Parameters
    ----------
    angles : np.ndarray
        Array of rotation angles.
    oversampling : float
        Number of display pixels per camera pixel.
    t_min, t_max : float
        Minimum and maximum bounds for the histogram.
    CF_image_avg : np.ndarray
        Conjugate Fourier transform of the average image.
    image_half : float
        Half the size of the rendered image.
    counter : multiprocessing.Manager.Value
        Counter for the number of processed groups.
    lock : multiprocessing.Manager.Lock
        Lock for synchronizing access to shared resources.
    group : int
        Index of the group to align.
    """
    with lock:
        counter.value += 1
    index = group_index[group].nonzero()[1]
    x_rot = x[index]
    y_rot = y[index]
    x_original = x_rot.copy()
    y_original = y_rot.copy()
    xcorr_max = 0.0
    for angle in angles:
        # rotate locs
        x_rot = np.cos(angle) * x_original - np.sin(angle) * y_original
        y_rot = np.sin(angle) * x_original + np.cos(angle) * y_original
        # render group image
        N, image = render_hist(x_rot, y_rot, oversampling, t_min, t_max)
        # calculate cross-correlation
        xcorr = compute_xcorr(CF_image_avg, image)
        # find the brightest pixel
        y_max, x_max = np.unravel_index(xcorr.argmax(), xcorr.shape)
        # store the transformation if the correlation is larger than before
        if xcorr[y_max, x_max] > xcorr_max:
            xcorr_max = xcorr[y_max, x_max]
            rot = angle
            dy = np.ceil(y_max - image_half) / oversampling
            dx = np.ceil(x_max - image_half) / oversampling
    # rotate and shift image group locs
    x[index] = np.cos(rot) * x_original - np.sin(rot) * y_original - dx
    y[index] = np.sin(rot) * x_original + np.cos(rot) * y_original - dy


def init_pool(
    x_: np.ndarray,
    y_: np.ndarray,
    group_index_: np.ndarray,
) -> None:
    """Initialize pool process variables."""
    global x, y, group_index
    x = np.ctypeslib.as_array(x_)
    y = np.ctypeslib.as_array(y_)
    group_index = group_index_


class Worker(QtCore.QThread):
    """Worker thread for processing image alignment.

    ...

    Attributes
    ----------
    group_index : np.ndarray
        Indexes of the groups.
    iterations : int
        Number of iterations to average over.
    locs : np.recarray
        Localizations with group indeces.
    oversampling : float
        Number of display pixels per camera pixel.
    r : float
        Radius for rendering. See View.open() for details.
    t_min, t_max : float
        Minimum and maximum bounds for the histogram. Set to -r and r.
    """

    progressMade = QtCore.pyqtSignal(int, int, int, int, np.recarray, bool)

    def __init__(
        self,
        locs: np.recarray,
        r: float,
        group_index: np.ndarray,
        oversampling: float,
        iterations: int
    ) -> None:
        super().__init__()
        self.locs = locs.copy()
        self.r = r
        self.t_min = -r
        self.t_max = r
        self.group_index = group_index
        self.oversampling = oversampling
        self.iterations = iterations

    def run(self) -> None:
        """Run averaging across a number of iterations given the average
        image."""
        n_groups = self.group_index.shape[0]
        a_step = np.arcsin(1 / (self.oversampling * self.r))
        angles = np.arange(0, 2 * np.pi, a_step)
        n_workers = min(
            60, max(1, int(0.75 * multiprocessing.cpu_count()))
        )  # Python crashes when using >64 cores
        manager = multiprocessing.Manager()
        counter = manager.Value("d", 0)
        lock = manager.Lock()
        groups_per_worker = max(1, int(n_groups / n_workers))
        for it in range(self.iterations):
            counter.value = 0
            # render average image
            N_avg, image_avg = render.render_hist(
                self.locs,
                self.oversampling,
                self.t_min,
                self.t_min,
                self.t_max,
                self.t_max,
            )
            n_pixel, _ = image_avg.shape
            image_half = n_pixel / 2
            CF_image_avg = np.conj(np.fft.fft2(image_avg))
            # TODO: blur average
            fc = functools.partial(
                align_group,
                angles,
                self.oversampling,
                self.t_min,
                self.t_max,
                CF_image_avg,
                image_half,
                counter,
                lock,
            )
            result = pool.map_async(fc, range(n_groups), groups_per_worker)
            while not result.ready():
                self.progressMade.emit(
                    it + 1,
                    self.iterations,
                    counter.value,
                    n_groups,
                    self.locs,
                    False,
                )
                time.sleep(0.5)
            self.locs.x = np.ctypeslib.as_array(x)
            self.locs.y = np.ctypeslib.as_array(y)
            self.locs.x -= np.mean(self.locs.x)
            self.locs.y -= np.mean(self.locs.y)
            self.progressMade.emit(
                it + 1,
                self.iterations,
                counter.value,
                n_groups,
                self.locs,
                True,
            )


class ParametersDialog(QtWidgets.QDialog):
    """Dialog for setting parameters - oversampling and iterations.

    ...

    Attributes
    ----------
    iterations : QtWidgets.QSpinBox
        Spin box for setting the number of iterations.
    oversampling : QtWidgets.QDoubleSpinBox
        Spin box for setting the number of display pixels per camera
        pixel.
    window : QtWidgets.QMainWindow
        Main window instance.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Parameters")
        self.setModal(False)
        grid = QtWidgets.QGridLayout(self)

        grid.addWidget(QtWidgets.QLabel("Oversampling:"), 0, 0)
        self.oversampling = QtWidgets.QDoubleSpinBox()
        self.oversampling.setRange(1, 1e7)
        self.oversampling.setValue(10)
        self.oversampling.setDecimals(1)
        self.oversampling.setKeyboardTracking(False)
        self.oversampling.valueChanged.connect(self.window.view.update_image)
        grid.addWidget(self.oversampling, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Iterations:"), 1, 0)
        self.iterations = QtWidgets.QSpinBox()
        self.iterations.setRange(0, int(1e7))
        self.iterations.setValue(10)
        grid.addWidget(self.iterations, 1, 1)


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
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setAcceptDrops(True)
        self._pixmap = None
        self.running = False

    def average(self):
        if not self.running:
            self.running = True
            oversampling = self.window.parameters_dialog.oversampling.value()
            iterations = self.window.parameters_dialog.iterations.value()
            self.thread = Worker(
                self.locs, self.r, self.group_index, oversampling, iterations
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
        g: int,
        n_groups: int,
        locs: np.recarray,
        update_image: bool,
    ) -> None:
        self.locs = locs.copy()
        if update_image:
            self.update_image()
        self.window.statusBar().showMessage(
            f"Iteration {it}/{total_it}, Group {g}/{n_groups}"
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
        if not hasattr(self.locs, "group"):
            message = (
                 "Loaded file contains no group information. Please load"
                 " localizations that were picked."
            )
            QtWidgets.QMessageBox.warning(self, "Warning", message)
            return
        groups = np.unique(self.locs.group)
        n_groups = len(groups)
        n_locs = len(self.locs)
        self.group_index = scipy.sparse.lil_matrix(
            (n_groups, n_locs), dtype=bool,
        )
        progress = lib.ProgressDialog(
            "Creating group index", 0, len(groups), self,
        )
        progress.set_value(0)
        for i, group in enumerate(groups):
            index = np.where(self.locs.group == group)[0]
            self.group_index[i, index] = True
            progress.set_value(i + 1)
        progress = lib.ProgressDialog(
            "Aligning by center of mass", 0, len(groups), self
        )
        progress.set_value(0)
        for i in range(n_groups):
            index = self.group_index[i, :].nonzero()[1]
            self.locs.x[index] -= np.mean(self.locs.x[index])
            self.locs.y[index] -= np.mean(self.locs.y[index])
            progress.set_value(i + 1)
        self.r = 2 * np.sqrt(np.mean(self.locs.x**2 + self.locs.y**2))
        self.update_image()
        status = lib.StatusDialog("Starting parallel pool...", self.window)
        global pool, x, y
        try:
            pool.close()
        except NameError:
            pass
        x = sharedctypes.RawArray("f", self.locs.x)
        y = sharedctypes.RawArray("f", self.locs.y)
        n_workers = min(
            60, max(1, int(0.75 * multiprocessing.cpu_count()))
        )  # Python crashes when using >64 cores
        pool = multiprocessing.Pool(
            n_workers, init_pool, (x, y, self.group_index),
        )
        self.window.statusBar().showMessage("Ready for processing!")
        status.close()

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
        cx = self.info[0]["Width"] / 2
        cy = self.info[0]["Height"] / 2
        self.locs.x += cx
        self.locs.y += cy
        info = self.info + [{
            "Generated by": f"Picasso v{__version__} Average"
        }]
        out_locs = self.locs
        io.save_locs(path, out_locs, info)
        self.window.statusBar().showMessage("File saved to {}.".format(path))

    def set_image(self, image: np.ndarray) -> None:
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
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        self._pixmap = QtGui.QPixmap.fromImage(qimage)
        self.set_pixmap(self._pixmap)

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self.setPixmap(
            pixmap.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
        )

    def update_image(self, *args) -> None:
        """Update the displayed image based on the changed display
        parameters."""
        oversampling = self.window.parameters_dialog.oversampling.value()
        t_min = -self.r
        t_max = self.r
        N_avg, image_avg = render.render_hist(
            self.locs, oversampling, t_min, t_min, t_max, t_max
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

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Picasso v{__version__}: Average")
        self.resize(512, 512)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "average.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.view = View(self)
        self.setCentralWidget(self.view)
        self.parameters_dialog = ParametersDialog(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction("Save")
        save_action.setShortcut(QtGui.QKeySequence.Save)
        save_action.triggered.connect(self.save)
        file_menu.addAction(save_action)
        process_menu = menu_bar.addMenu("Process")
        parameters_action = process_menu.addAction("Parameters")
        parameters_action.setShortcut("Ctrl+P")
        parameters_action.triggered.connect(self.parameters_dialog.show)
        average_action = process_menu.addAction("Average")
        average_action.setShortcut("Ctrl+A")
        average_action.triggered.connect(self.view.average)

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
        path, exe = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save localizations", out_path, filter="*.hdf5"
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
        for finder, name, ispkg
        in iter_namespace(plugins)
    ]

    for plugin in plugins:
        p = plugin.Plugin(window)
        if p.name == "average":
            p.execute()

    window.show()

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(
            window, "An error occured", message,
        )
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
