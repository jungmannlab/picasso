"""
picasso.gui.convert_units
~~~~~~~~~~~~~~~~~~~~~~~~~

Dialog of Picasso: Render's ``File > Convert files to nm...``: converts
files written before Picasso 1.0, whose lengths are in camera pixels,
to nm with ``picasso.legacy.convert_files``. The user lists the files
(localizations .hdf5, picks and calibration .yaml, drift .txt), gives
the camera pixel size for the files that do not record it and chooses
whether the originals are replaced or ``_nm`` copies are written.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

from PyQt6 import QtCore, QtWidgets

from .. import io, legacy, lib, lib_qt, docs_url

#: file filter of the Add files dialog
FILE_FILTER = (
    "Picasso files (*.hdf5 *.yaml *.txt);;"
    "Localizations (*.hdf5);;Picks and calibrations (*.yaml);;"
    "Drift (*.txt)"
)


def metadata_pixelsize(paths: list[str]) -> float | None:
    """The camera pixel size (nm) recorded by the first localization file
    among ``paths`` whose metadata has one, None if there is none."""
    for path in paths:
        if not path.endswith(".hdf5"):
            continue
        try:
            info = io.load_info(path)
        except Exception:
            continue
        pixelsize = lib.get_from_metadata(info, "Pixelsize")
        if pixelsize is not None:
            return float(pixelsize)
    return None


class ConvertUnitsDialog(lib.Dialog):
    """Convert files written before Picasso 1.0 from camera pixels to
    nm, see ``picasso.legacy``.

    Attributes
    ----------
    files : QListWidget
        The files to convert.
    pixelsize : QDoubleSpinBox
        Camera pixel size (nm) for files that do not record it. It is
        prefilled from the first localization file's metadata, else
        from the last channel loaded in Render.
    replace : QCheckBox
        If checked, the files are overwritten; otherwise ``_nm`` copies
        are written next to them.
    results : QTableWidget
        The converted file or the message per file of the last run.
    """

    DOCS_URL = docs_url("files.html#files-legacy")

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Convert files to nm")
        self.setModal(False)
        self.resize(640, 480)
        self._pixelsize_edited = False
        self._task = None

        vbox = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Files written before Picasso 1.0 store lengths in camera "
            "pixels; since 1.0 every length is in nm. Picasso converts "
            "such files in memory whenever it loads them. Convert them "
            "here once to skip the camera pixel size question and to "
            "make them readable as nm by other software. Files already "
            "in nm are skipped."
        )
        intro.setWordWrap(True)
        vbox.addWidget(intro)

        files_box = QtWidgets.QGroupBox("Files")
        vbox.addWidget(files_box, 1)
        files_grid = QtWidgets.QGridLayout(files_box)
        self.files = QtWidgets.QListWidget()
        self.files.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.files.setToolTip(
            "Localizations (.hdf5, also cluster centers and pick\n"
            "properties), pick regions, 3D and lateral calibrations,\n"
            "SPINNA masks and nanoTRON models (.yaml), drift (.txt)."
        )
        files_grid.addWidget(self.files, 0, 0, 3, 1)
        self.add_button = QtWidgets.QPushButton("Add files...")
        self.add_button.clicked.connect(self.add_files)
        files_grid.addWidget(self.add_button, 0, 1)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.setToolTip("Remove the selected files.")
        self.remove_button.clicked.connect(self.remove_files)
        files_grid.addWidget(self.remove_button, 1, 1)
        files_grid.setRowStretch(2, 1)

        settings = QtWidgets.QGridLayout()
        vbox.addLayout(settings)
        settings.addWidget(lib.HelpButton(self.DOCS_URL), 0, 0)
        pixelsize_label = QtWidgets.QLabel("Camera pixel size (nm):")
        pixelsize_label.setToolTip(
            "Used for files that do not record the camera pixel size\n"
            "(picks, drift, calibrations, localization files without\n"
            "Pixelsize in their metadata). Prefilled from the first\n"
            "localization file's metadata, else from the last channel\n"
            "loaded in Render."
        )
        settings.addWidget(pixelsize_label, 0, 1)
        self.pixelsize = QtWidgets.QDoubleSpinBox()
        self.pixelsize.setRange(0.01, 100000)
        self.pixelsize.setDecimals(2)
        self.pixelsize.setValue(window.view.default_pixelsize())
        self.pixelsize.setKeyboardTracking(False)
        self.pixelsize.valueChanged.connect(self._on_pixelsize_edited)
        settings.addWidget(self.pixelsize, 0, 2)
        self.replace = QtWidgets.QCheckBox("Replace the original files")
        self.replace.setToolTip(
            "Overwrite the files. Otherwise copies with '_nm' appended\n"
            "to the name are written next to them, e.g.,\n"
            "movie_locs_nm.hdf5 (and movie_locs_nm.yaml)."
        )
        settings.addWidget(self.replace, 1, 1, 1, 2)
        settings.setColumnStretch(3, 1)
        self.convert_button = QtWidgets.QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert)
        settings.addWidget(self.convert_button, 1, 3)

        results_box = QtWidgets.QGroupBox("Results")
        vbox.addWidget(results_box, 1)
        results_layout = QtWidgets.QVBoxLayout(results_box)
        self.results = QtWidgets.QTableWidget(0, 2)
        self.results.setHorizontalHeaderLabels(["File", "Result"])
        self.results.horizontalHeader().setStretchLastSection(True)
        self.results.verticalHeader().setVisible(False)
        self.results.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        results_layout.addWidget(self.results)

    # --- files -----------------------------------------------------------

    def paths(self) -> list[str]:
        """The listed files."""
        return [
            self.files.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            for i in range(self.files.count())
        ]

    def add_paths(self, paths: list[str]) -> None:
        """List ``paths`` (new ones only) and prefill the camera pixel
        size from the first localization file that records it, unless
        the user edited it."""
        listed = set(self.paths())
        for path in paths:
            if path in listed:
                continue
            listed.add(path)
            item = QtWidgets.QListWidgetItem(os.path.basename(path))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.files.addItem(item)
        if not self._pixelsize_edited:
            pixelsize = metadata_pixelsize(self.paths())
            if pixelsize is not None:
                self.pixelsize.blockSignals(True)
                self.pixelsize.setValue(pixelsize)
                self.pixelsize.blockSignals(False)

    def add_files(self) -> None:
        """Ask for files to convert."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add files to convert", filter=FILE_FILTER
        )
        if paths:
            self.add_paths(paths)

    def remove_files(self) -> None:
        """Remove the selected files from the list."""
        for item in self.files.selectedItems():
            self.files.takeItem(self.files.row(item))

    def _on_pixelsize_edited(self, *args) -> None:
        self._pixelsize_edited = True

    # --- conversion ------------------------------------------------------

    def convert(self) -> None:
        """Convert the listed files on a worker thread and show the
        results."""
        paths = self.paths()
        if not paths:
            QtWidgets.QMessageBox.information(
                self, "Convert files to nm", "Add files to convert first."
            )
            return
        if self._task is not None:
            return  # a conversion is running
        pixelsize = self.pixelsize.value()
        in_place = self.replace.isChecked()
        if in_place:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Convert files to nm",
                f"Replace {len(paths)} file(s) with their nm version? "
                "This cannot be undone.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        def compute(progress: lib.TaskProgress) -> list[tuple]:
            return legacy.convert_files(
                paths, pixelsize, in_place=in_place, progress=progress
            )

        def finished(results: list[tuple]) -> None:
            self._task = None
            self.show_results(results)

        def failed(error: BaseException) -> None:
            self._task = None
            QtWidgets.QMessageBox.critical(
                self, "Convert files to nm", str(error)
            )

        def canceled() -> None:
            self._task = None

        self._task = lib_qt.run_task(
            compute,
            "Converting files to nm...",
            self,
            finished,
            on_failed=failed,
            on_canceled=canceled,
            title="Convert files to nm",
        )

    def show_results(self, results: list[tuple]) -> None:
        """Fill the results table with the output path or the message of
        each file, see ``legacy.convert_files``."""
        self.results.setRowCount(len(results))
        for row, (path, output, message) in enumerate(results):
            if output is not None:
                text = f"converted to {os.path.basename(output)}"
                if message:
                    text += f" ({message})"
            else:
                text = f"skipped: {message}"
            tooltips = (path, output or path)
            for column, value in enumerate((os.path.basename(path), text)):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(tooltips[column])
                self.results.setItem(row, column, item)
        self.results.resizeColumnToContents(0)
