"""
    picasso.simulate-gui
    ~~~~~~~~~~~~~~~~

    GUI for Simulate :
    Simulate single molcule fluorescence data

    :author: Maximilian Thomas Strauss, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

import csv
import glob as _glob
import os
import sys
import time
import importlib, pkgutil

import yaml

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as _np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5 import QtCore, QtGui, QtWidgets

from scipy.optimize import curve_fit
from scipy.stats import norm
import os.path as _ospath

from .. import io as _io
from .. import lib, simulate, __version__


def fitFuncBg(x, a, b):
    return (a + b * x[0]) * x[1] * x[2]


def fitFuncStd(x, a, b, c):
    return a * x[0] * x[1] + b * x[2] + c


plt.style.use("ggplot")

"DEFAULT PARAMETERS"
CURRENTROUND = 0

ADVANCEDMODE = 0  # 1 is with calibration of noise model
# CAMERA
IMAGESIZE_DEFAULT = 32
ITIME_DEFAULT = 300
FRAMES_DEFAULT = 7500
PIXELSIZE_DEFAULT = 160
# PAINT
KON_DEFAULT = 1600000
IMAGERCONCENTRATION_DEFAULT = 5
MEANBRIGHT_DEFAULT = 500
# IMAGER
LASERPOWER_DEFAULT = 1.5  # POWER DENSITY
POWERDENSITY_CONVERSION = 20
STDFACTOR = 1.82
if ADVANCEDMODE:
    LASERPOWER_DEFAULT = 30
PSF_DEFAULT = 0.82
PHOTONRATE_DEFAULT = 53
PHOTONRATESTD_DEFAULT = 29
PHOTONBUDGET_DEFAULT = 1500000
PHOTONSLOPE_DEFAULT = 35
PHOTONSLOPESTD_DEFAULT = 19
if ADVANCEDMODE:
    PHOTONSLOPE_DEFAULT = 1.77
    PHOTONSLOPESTD_DEFAULT = 0.97
# NOISE MODEL
LASERC_DEFAULT = 0.012063
IMAGERC_DEFAULT = 0.003195
EQA_DEFAULT = -0.002866
EQB_DEFAULT = 0.259038
EQC_DEFAULT = 13.085473
BGOFFSET_DEFAULT = 0
BGSTDOFFSET_DEFAULT = 0
# STRUCTURE
STRUCTURE1_DEFAULT = 3
STRUCTURE2_DEFAULT = 4
STRUCTURE3_DEFAULT = "20,20"
STRUCTUREYY_DEFAULT = "0,20,40,60,0,20,40,60,0,20,40,60"
STRUCTUREXX_DEFAULT = "0,20,40,0,20,40,0,20,40,0,20,40"
STRUCTUREEX_DEFAULT = "1,1,1,1,1,1,1,1,1,1,1,1"
STRUCTURE3D_DEFAULT = "0,0,0,0,0,0,0,0,0,0,0,0"
STRUCTURENO_DEFAULT = 9
STRUCTUREFRAME_DEFAULT = 6
INCORPORATION_DEFAULT = 85
# Default 3D calibration
CX_DEFAULT = [
    3.1638306844743706e-17,
    -2.2103661248660896e-14,
    -9.775815406044296e-12,
    8.2178622893072e-09,
    4.91181990105529e-06,
    -0.0028759382006135654,
    1.1756537760039398,
]
CY_DEFAULT = [
    1.710907877866197e-17,
    -2.4986657766862576e-15,
    -8.405284979510355e-12,
    1.1548322314075128e-11,
    5.4270591055277476e-06,
    0.0018155881468011011,
    1.011468185618154,
]


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Picasso v{__version__}: Simulate")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "simulate.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.initUI()

    def initUI(self):
        self.currentround = CURRENTROUND
        self.structureMode = True

        self.grid = QtWidgets.QGridLayout()
        self.grid.setSpacing(5)

        # CAMERA PARAMETERS
        camera_groupbox = QtWidgets.QGroupBox("Camera parameters")
        cgrid = QtWidgets.QGridLayout(camera_groupbox)

        camerasize = QtWidgets.QLabel("Image size")
        integrationtime = QtWidgets.QLabel("Integration time")
        totaltime = QtWidgets.QLabel("Total acquisition time")
        frames = QtWidgets.QLabel("Frames")
        pixelsize = QtWidgets.QLabel("Pixelsize")

        self.camerasizeEdit = QtWidgets.QSpinBox()
        self.camerasizeEdit.setRange(1, 512)
        self.integrationtimeEdit = QtWidgets.QSpinBox()
        self.integrationtimeEdit.setRange(1, 10000)  # 1-10.000ms
        self.framesEdit = QtWidgets.QSpinBox()
        self.framesEdit.setRange(10, 100000000)  # 10-100.000.000 frames
        self.framesEdit.setSingleStep(1000)
        self.pixelsizeEdit = QtWidgets.QSpinBox()
        self.pixelsizeEdit.setRange(1, 1000)  # 1 to 1000 nm frame size
        self.totaltimeEdit = QtWidgets.QLabel()

        # Deactivate keyboard tracking

        self.camerasizeEdit.setKeyboardTracking(False)
        self.pixelsizeEdit.setKeyboardTracking(False)

        self.camerasizeEdit.setValue(IMAGESIZE_DEFAULT)
        self.integrationtimeEdit.setValue(ITIME_DEFAULT)
        self.framesEdit.setValue(FRAMES_DEFAULT)
        self.pixelsizeEdit.setValue(PIXELSIZE_DEFAULT)

        self.integrationtimeEdit.valueChanged.connect(self.changeTime)
        self.framesEdit.valueChanged.connect(self.changeTime)
        self.camerasizeEdit.valueChanged.connect(self.generatePositions)

        self.pixelsizeEdit.valueChanged.connect(self.changeStructDefinition)

        cgrid.addWidget(camerasize, 1, 0)
        cgrid.addWidget(self.camerasizeEdit, 1, 1)
        cgrid.addWidget(QtWidgets.QLabel("Px"), 1, 2)
        cgrid.addWidget(integrationtime, 2, 0)
        cgrid.addWidget(self.integrationtimeEdit, 2, 1)
        cgrid.addWidget(QtWidgets.QLabel("ms"), 2, 2)
        cgrid.addWidget(frames, 3, 0)
        cgrid.addWidget(self.framesEdit, 3, 1)
        cgrid.addWidget(totaltime, 4, 0)
        cgrid.addWidget(self.totaltimeEdit, 4, 1)
        cgrid.addWidget(QtWidgets.QLabel("min"), 4, 2)
        cgrid.addWidget(pixelsize, 5, 0)
        cgrid.addWidget(self.pixelsizeEdit, 5, 1)
        cgrid.addWidget(QtWidgets.QLabel("nm"), 5, 2)

        cgrid.addItem(
            QtWidgets.QSpacerItem(
                1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
            )
        )

        # PAINT PARAMETERS
        paint_groupbox = QtWidgets.QGroupBox("PAINT parameters")
        pgrid = QtWidgets.QGridLayout(paint_groupbox)

        kon = QtWidgets.QLabel("k<sub>On</sub>")
        imagerconcentration = QtWidgets.QLabel("Imager concentration")
        taud = QtWidgets.QLabel("Dark time")
        taub = QtWidgets.QLabel("Bright time")

        self.konEdit = QtWidgets.QDoubleSpinBox()
        self.konEdit.setRange(1, 10000000000)
        self.konEdit.setDecimals(0)
        self.konEdit.setSingleStep(100000)
        self.imagerconcentrationEdit = QtWidgets.QDoubleSpinBox()
        self.imagerconcentrationEdit.setRange(0.01, 1000)
        self.taudEdit = QtWidgets.QLabel()
        self.taubEdit = QtWidgets.QDoubleSpinBox()
        self.taubEdit.setRange(1, 10000)
        self.taubEdit.setDecimals(0)
        self.taubEdit.setSingleStep(10)

        self.konEdit.setValue(KON_DEFAULT)
        self.imagerconcentrationEdit.setValue(IMAGERCONCENTRATION_DEFAULT)
        self.taubEdit.setValue(MEANBRIGHT_DEFAULT)

        self.imagerconcentrationEdit.valueChanged.connect(self.changePaint)
        self.konEdit.valueChanged.connect(self.changePaint)

        pgrid.addWidget(kon, 1, 0)
        pgrid.addWidget(self.konEdit, 1, 1)
        pgrid.addWidget(QtWidgets.QLabel("M<sup>−1</sup>s<sup>−1</sup>"), 1, 2)
        pgrid.addWidget(imagerconcentration, 2, 0)
        pgrid.addWidget(self.imagerconcentrationEdit, 2, 1)
        pgrid.addWidget(QtWidgets.QLabel("nM"), 2, 2)
        pgrid.addWidget(taud, 3, 0)
        pgrid.addWidget(self.taudEdit, 3, 1)
        pgrid.addWidget(QtWidgets.QLabel("ms"), 3, 2)
        pgrid.addWidget(taub, 4, 0)
        pgrid.addWidget(self.taubEdit, 4, 1)
        pgrid.addWidget(QtWidgets.QLabel("ms"), 4, 2)
        pgrid.addItem(
            QtWidgets.QSpacerItem(
                1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
            )
        )

        # IMAGER Parameters
        imager_groupbox = QtWidgets.QGroupBox("Imager parameters")
        igrid = QtWidgets.QGridLayout(imager_groupbox)

        laserpower = QtWidgets.QLabel("Power density")
        if ADVANCEDMODE:
            laserpower = QtWidgets.QLabel("Laserpower")
        psf = QtWidgets.QLabel("PSF")
        psf_fwhm = QtWidgets.QLabel("PSF(FWHM)")
        photonrate = QtWidgets.QLabel("Photonrate")
        photonsframe = QtWidgets.QLabel("Photons (frame)")
        photonratestd = QtWidgets.QLabel("Photonrate Std")
        photonstdframe = QtWidgets.QLabel("Photons Std (frame)")
        photonbudget = QtWidgets.QLabel("Photonbudget")
        photonslope = QtWidgets.QLabel("Photon detection rate")
        photonslopeStd = QtWidgets.QLabel("Photonrate Std ")

        self.laserpowerEdit = QtWidgets.QDoubleSpinBox()
        self.laserpowerEdit.setRange(0, 10)
        self.laserpowerEdit.setSingleStep(0.1)
        self.psfEdit = QtWidgets.QDoubleSpinBox()
        self.psfEdit.setRange(0, 3)
        self.psfEdit.setSingleStep(0.01)
        self.psf_fwhmEdit = QtWidgets.QLabel()
        self.photonrateEdit = QtWidgets.QDoubleSpinBox()
        self.photonrateEdit.setRange(0, 1000)
        self.photonrateEdit.setDecimals(0)
        self.photonsframeEdit = QtWidgets.QLabel()
        self.photonratestdEdit = QtWidgets.QDoubleSpinBox()
        self.photonratestdEdit.setRange(0, 1000)
        self.photonratestdEdit.setDecimals(0)
        self.photonstdframeEdit = QtWidgets.QLabel()
        self.photonbudgetEdit = QtWidgets.QDoubleSpinBox()
        self.photonbudgetEdit.setRange(0, 100000000)
        self.photonbudgetEdit.setSingleStep(100000)
        self.photonbudgetEdit.setDecimals(0)

        self.photonslopeEdit = QtWidgets.QSpinBox()
        self.photonslopeStdEdit = QtWidgets.QDoubleSpinBox()

        self.laserpowerEdit.setValue(LASERPOWER_DEFAULT)
        self.psfEdit.setValue(PSF_DEFAULT)
        self.photonrateEdit.setValue(PHOTONRATE_DEFAULT)
        self.photonratestdEdit.setValue(PHOTONRATESTD_DEFAULT)
        self.photonbudgetEdit.setValue(PHOTONBUDGET_DEFAULT)

        self.photonslopeEdit.setValue(PHOTONSLOPE_DEFAULT)
        self.photonslopeStdEdit.setValue(PHOTONSLOPESTD_DEFAULT)

        self.psfEdit.valueChanged.connect(self.changePSF)
        self.photonrateEdit.valueChanged.connect(self.changeImager)
        self.photonratestdEdit.valueChanged.connect(self.changeImager)
        self.laserpowerEdit.valueChanged.connect(self.changeImager)
        self.photonslopeEdit.valueChanged.connect(self.changeImager)
        self.photonslopeStdEdit.valueChanged.connect(self.changeImager)

        self.cx = CX_DEFAULT
        self.cy = CY_DEFAULT

        self.photonslopemodeEdit = QtWidgets.QCheckBox()

        igrid.addWidget(psf, 0, 0)
        igrid.addWidget(self.psfEdit, 0, 1)
        igrid.addWidget(QtWidgets.QLabel("Px"), 0, 2)
        igrid.addWidget(psf_fwhm, 1, 0)
        igrid.addWidget(self.psf_fwhmEdit, 1, 1)
        igrid.addWidget(QtWidgets.QLabel("nm"), 1, 2)

        igrid.addWidget(laserpower, 2, 0)
        igrid.addWidget(self.laserpowerEdit, 2, 1)
        igrid.addWidget(QtWidgets.QLabel("kW cm<sup>-2<sup>"), 2, 2)
        if ADVANCEDMODE:
            igrid.addWidget(QtWidgets.QLabel("mW"), 2, 2)

        igridindex = 1
        if ADVANCEDMODE:
            igrid.addWidget(photonrate, 3, 0)
            igrid.addWidget(self.photonrateEdit, 3, 1)
            igrid.addWidget(QtWidgets.QLabel("Photons ms<sup>-1<sup>"), 3, 2)

            igridindex = 0

        igrid.addWidget(photonsframe, 4 - igridindex, 0)
        igrid.addWidget(self.photonsframeEdit, 4 - igridindex, 1)
        igrid.addWidget(QtWidgets.QLabel("Photons"), 4 - igridindex, 2)
        igridindex = 2

        if ADVANCEDMODE:
            igrid.addWidget(photonratestd, 5, 0)
            igrid.addWidget(self.photonratestdEdit, 5, 1)
            igrid.addWidget(QtWidgets.QLabel("Photons ms<sup>-1<sup"), 5, 2)
            igridindex = 0

        igrid.addWidget(photonstdframe, 6 - igridindex, 0)
        igrid.addWidget(self.photonstdframeEdit, 6 - igridindex, 1)
        igrid.addWidget(QtWidgets.QLabel("Photons"), 6 - igridindex, 2)
        igrid.addWidget(photonbudget, 7 - igridindex, 0)
        igrid.addWidget(self.photonbudgetEdit, 7 - igridindex, 1)
        igrid.addWidget(QtWidgets.QLabel("Photons"), 7 - igridindex, 2)
        igrid.addWidget(photonslope, 8 - igridindex, 0)
        igrid.addWidget(self.photonslopeEdit, 8 - igridindex, 1)

        photonslopeUnit = QtWidgets.QLabel(
            "Photons  ms<sup>-1</sup> kW<sup>-1</sup> cm<sup>2</sup>"
        )
        photonslopeUnit.setWordWrap(True)
        igrid.addWidget(photonslopeUnit, 8 - igridindex, 2)

        igrid.addWidget(self.photonslopemodeEdit, 9 - igridindex, 1)
        igrid.addWidget(QtWidgets.QLabel("Constant detection rate"), 9 - igridindex, 0)

        if ADVANCEDMODE:
            igrid.addWidget(photonslopeStd, 10 - igridindex, 0)
            igrid.addWidget(self.photonslopeStdEdit, 10 - igridindex, 1)
            igrid.addWidget(
                QtWidgets.QLabel(
                    "Photons  ms<sup>-1</sup> kW<sup>-1</sup> cm<sup>2</sup>"
                ),
                10 - igridindex,
                2,
            )

        if not ADVANCEDMODE:
            backgroundframesimple = QtWidgets.QLabel("Background (Frame)")
            self.backgroundframesimpleEdit = QtWidgets.QLabel()
            igrid.addWidget(backgroundframesimple, 12 - igridindex, 0)
            igrid.addWidget(self.backgroundframesimpleEdit, 12 - igridindex, 1)

        # Make a spinbox for adjusting the background level
        backgroundlevel = QtWidgets.QLabel("Background level")

        self.backgroundlevelEdit = QtWidgets.QSpinBox()
        self.backgroundlevelEdit.setRange(1, 100)

        igrid.addWidget(backgroundlevel, 11 - igridindex, 0)
        igrid.addWidget(self.backgroundlevelEdit, 11 - igridindex, 1)
        self.backgroundlevelEdit.valueChanged.connect(self.changeNoise)

        # NOISE MODEL
        noise_groupbox = QtWidgets.QGroupBox("Noise Model")
        ngrid = QtWidgets.QGridLayout(noise_groupbox)

        laserc = QtWidgets.QLabel("Lasercoefficient")
        imagerc = QtWidgets.QLabel("Imagercoefficient")

        EquationA = QtWidgets.QLabel("Equation A")
        EquationB = QtWidgets.QLabel("Equation B")
        EquationC = QtWidgets.QLabel("Equation C")

        Bgoffset = QtWidgets.QLabel("Background Offset")
        BgStdoffset = QtWidgets.QLabel("Background Std Offset")

        backgroundframe = QtWidgets.QLabel("Background (Frame)")
        noiseLabel = QtWidgets.QLabel("Noise (Frame)")

        self.lasercEdit = QtWidgets.QDoubleSpinBox()
        self.lasercEdit.setRange(0, 100000)
        self.lasercEdit.setDecimals(6)

        self.imagercEdit = QtWidgets.QDoubleSpinBox()
        self.imagercEdit.setRange(0, 100000)
        self.imagercEdit.setDecimals(6)

        self.EquationBEdit = QtWidgets.QDoubleSpinBox()
        self.EquationBEdit.setRange(-100000, 100000)
        self.EquationBEdit.setDecimals(6)

        self.EquationAEdit = QtWidgets.QDoubleSpinBox()
        self.EquationAEdit.setRange(-100000, 100000)
        self.EquationAEdit.setDecimals(6)

        self.EquationCEdit = QtWidgets.QDoubleSpinBox()
        self.EquationCEdit.setRange(-100000, 100000)
        self.EquationCEdit.setDecimals(6)

        self.lasercEdit.setValue(LASERC_DEFAULT)
        self.imagercEdit.setValue(IMAGERC_DEFAULT)

        self.EquationAEdit.setValue(EQA_DEFAULT)
        self.EquationBEdit.setValue(EQB_DEFAULT)
        self.EquationCEdit.setValue(EQC_DEFAULT)

        self.BgoffsetEdit = QtWidgets.QDoubleSpinBox()
        self.BgoffsetEdit.setRange(-100000, 100000)
        self.BgoffsetEdit.setDecimals(6)

        self.BgStdoffsetEdit = QtWidgets.QDoubleSpinBox()
        self.BgStdoffsetEdit.setRange(-100000, 100000)
        self.BgStdoffsetEdit.setDecimals(6)

        for button in [
            self.lasercEdit,
            self.imagercEdit,
            self.EquationAEdit,
            self.EquationBEdit,
            self.EquationCEdit,
        ]:
            button.valueChanged.connect(self.changeNoise)

        backgroundframe = QtWidgets.QLabel("Background (Frame)")
        noiseLabel = QtWidgets.QLabel("Noise (Frame)")

        self.backgroundframeEdit = QtWidgets.QLabel()
        self.noiseEdit = QtWidgets.QLabel()

        tags = [
            laserc,
            imagerc,
            EquationA,
            EquationB,
            EquationC,
            Bgoffset,
            BgStdoffset,
            backgroundframe,
            noiseLabel,
        ]
        buttons = [
            self.lasercEdit,
            self.imagercEdit,
            self.EquationAEdit,
            self.EquationBEdit,
            self.EquationCEdit,
            self.BgoffsetEdit,
            self.BgStdoffsetEdit,
            self.backgroundframeEdit,
            self.noiseEdit,
        ]

        for i, tag in enumerate(tags):
            ngrid.addWidget(tag, i, 0)
            ngrid.addWidget(buttons[i], i, 1)

        calibrateNoiseButton = QtWidgets.QPushButton("Calibrate Noise Model")
        calibrateNoiseButton.clicked.connect(self.calibrateNoise)
        importButton = QtWidgets.QPushButton("Import from Experiment (hdf5)")
        importButton.clicked.connect(self.importhdf5)

        ngrid.addWidget(calibrateNoiseButton, 10, 0, 1, 3)
        ngrid.addWidget(importButton, 11, 0, 1, 3)

        # HANDLE DEFINTIIONS
        structureIncorporation = QtWidgets.QLabel("Incorporation")
        self.structureIncorporationEdit = QtWidgets.QDoubleSpinBox()
        self.structureIncorporationEdit.setKeyboardTracking(False)
        self.structureIncorporationEdit.setRange(1, 100)
        self.structureIncorporationEdit.setDecimals(0)
        self.structureIncorporationEdit.setValue(INCORPORATION_DEFAULT)

        handles_groupbox = QtWidgets.QGroupBox("Handles")
        hgrid = QtWidgets.QGridLayout(handles_groupbox)

        hgrid.addWidget(structureIncorporation, 0, 0)
        hgrid.addWidget(self.structureIncorporationEdit, 0, 1)
        hgrid.addWidget(QtWidgets.QLabel("%"), 0, 2)

        importHandlesButton = QtWidgets.QPushButton("Import handles")
        importHandlesButton.clicked.connect(self.importHandles)
        hgrid.addWidget(importHandlesButton, 1, 0, 1, 3)

        # 3D Settings
        self.mode3DEdit = QtWidgets.QCheckBox()
        threed_groupbox = QtWidgets.QGroupBox("3D")
        tgrid = QtWidgets.QGridLayout(threed_groupbox)
        tgrid.addWidget(self.mode3DEdit, 0, 0)
        tgrid.addWidget(QtWidgets.QLabel("3D"), 0, 1)

        load3dCalibrationButton = QtWidgets.QPushButton("Load 3D Calibration")
        load3dCalibrationButton.clicked.connect(self.load3dCalibration)
        tgrid.addWidget(load3dCalibrationButton, 0, 2)

        # STRUCTURE DEFINITIONS
        structure_groupbox = QtWidgets.QGroupBox("Structure")
        sgrid = QtWidgets.QGridLayout(structure_groupbox)

        structureno = QtWidgets.QLabel("Number of structures")
        structureframe = QtWidgets.QLabel("Frame")

        self.structure1 = QtWidgets.QLabel("Columns")
        self.structure2 = QtWidgets.QLabel("Rows")
        self.structure3 = QtWidgets.QLabel("Spacing X,Y")
        self.structure3Label = QtWidgets.QLabel("nm")

        structurexx = QtWidgets.QLabel("Stucture X")
        structureyy = QtWidgets.QLabel("Structure Y")
        structure3d = QtWidgets.QLabel("Structure 3D")
        structureex = QtWidgets.QLabel("Exchange labels")

        structurecomboLabel = QtWidgets.QLabel("Type")

        self.structurenoEdit = QtWidgets.QSpinBox()
        self.structurenoEdit.setRange(1, 1000)
        self.structureframeEdit = QtWidgets.QSpinBox()
        self.structureframeEdit.setRange(4, 16)
        self.structurexxEdit = QtWidgets.QLineEdit(STRUCTUREXX_DEFAULT)
        self.structureyyEdit = QtWidgets.QLineEdit(STRUCTUREYY_DEFAULT)
        self.structureexEdit = QtWidgets.QLineEdit(STRUCTUREEX_DEFAULT)
        self.structure3DEdit = QtWidgets.QLineEdit(STRUCTURE3D_DEFAULT)

        self.structurecombo = QtWidgets.QComboBox()
        for entry in ["Grid", "Circle", "Custom"]:
            self.structurecombo.addItem(entry)

        self.structure1Edit = QtWidgets.QSpinBox()
        self.structure1Edit.setKeyboardTracking(False)
        self.structure1Edit.setRange(1, 1000)
        self.structure1Edit.setValue(STRUCTURE1_DEFAULT)
        self.structure2Edit = QtWidgets.QSpinBox()
        self.structure2Edit.setKeyboardTracking(False)
        self.structure2Edit.setRange(1, 1000)
        self.structure2Edit.setValue(STRUCTURE2_DEFAULT)
        self.structure3Edit = QtWidgets.QLineEdit(STRUCTURE3_DEFAULT)

        self.structure1Edit.valueChanged.connect(self.changeStructDefinition)
        self.structure2Edit.valueChanged.connect(self.changeStructDefinition)
        self.structure3Edit.returnPressed.connect(self.changeStructDefinition)

        self.structurenoEdit.setValue(STRUCTURENO_DEFAULT)
        self.structureframeEdit.setValue(STRUCTUREFRAME_DEFAULT)

        self.structurenoEdit.setKeyboardTracking(False)
        self.structureframeEdit.setKeyboardTracking(False)

        self.structurexxEdit.returnPressed.connect(self.generatePositions)
        self.structureyyEdit.returnPressed.connect(self.generatePositions)
        self.structureexEdit.returnPressed.connect(self.generatePositions)
        self.structure3DEdit.returnPressed.connect(self.generatePositions)

        self.structurenoEdit.valueChanged.connect(self.generatePositions)
        self.structureframeEdit.valueChanged.connect(self.generatePositions)

        self.structurerandomOrientationEdit = QtWidgets.QCheckBox()
        self.structurerandomEdit = QtWidgets.QCheckBox()

        structurerandom = QtWidgets.QLabel("Random arrangement")
        structurerandomOrientation = QtWidgets.QLabel("Random orientation")

        self.structurerandomEdit.stateChanged.connect(self.generatePositions)
        self.structurerandomOrientationEdit.stateChanged.connect(self.generatePositions)
        self.structureIncorporationEdit.valueChanged.connect(self.generatePositions)

        self.structurecombo.currentIndexChanged.connect(self.changeStructureType)

        sgrid.addWidget(structureno, 1, 0)
        sgrid.addWidget(self.structurenoEdit, 1, 1)
        sgrid.addWidget(structureframe, 2, 0)
        sgrid.addWidget(self.structureframeEdit, 2, 1)
        sgrid.addWidget(QtWidgets.QLabel("Px"), 2, 2)
        sgrid.addWidget(structurecomboLabel)
        sgrid.addWidget(self.structurecombo, 3, 1)

        sgrid.addWidget(self.structure1, 4, 0)
        sgrid.addWidget(self.structure1Edit, 4, 1)
        sgrid.addWidget(self.structure2, 5, 0)
        sgrid.addWidget(self.structure2Edit, 5, 1)
        sgrid.addWidget(self.structure3, 6, 0)
        sgrid.addWidget(self.structure3Edit, 6, 1)
        sgrid.addWidget(self.structure3Label, 6, 2)

        sgrid.addWidget(structurexx, 7, 0)
        sgrid.addWidget(self.structurexxEdit, 7, 1)
        sgrid.addWidget(QtWidgets.QLabel("nm"), 7, 2)
        sgrid.addWidget(structureyy, 8, 0)
        sgrid.addWidget(self.structureyyEdit, 8, 1)
        sgrid.addWidget(QtWidgets.QLabel("nm"), 8, 2)
        sindex = 0

        sgrid.addWidget(structure3d, 9, 0)
        sgrid.addWidget(self.structure3DEdit, 9, 1)
        sindex = 1

        sgrid.addWidget(structureex, 9 + sindex, 0)
        sgrid.addWidget(self.structureexEdit, 9 + sindex, 1)

        sindex += -1
        sgrid.addWidget(structurerandom, 11 + sindex, 1)
        sgrid.addWidget(self.structurerandomEdit, 11 + sindex, 0)
        sgrid.addWidget(structurerandomOrientation, 12 + sindex, 1)
        sgrid.addWidget(self.structurerandomOrientationEdit, 12 + sindex, 0)

        sindex += -2

        importDesignButton = QtWidgets.QPushButton(
            "Import structure from Picasso: Design"
        )
        importDesignButton.clicked.connect(self.importDesign)
        sgrid.addWidget(importDesignButton, 15 + sindex, 0, 1, 3)

        generateButton = QtWidgets.QPushButton("Generate positions")
        generateButton.clicked.connect(self.generatePositions)
        sgrid.addWidget(generateButton, 17 + sindex, 0, 1, 3)
        cgrid.addItem(
            QtWidgets.QSpacerItem(
                1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
            )
        )

        simulateButton = QtWidgets.QPushButton("Simulate data")
        self.exchangeroundsEdit = QtWidgets.QLineEdit("1")

        self.conroundsEdit = QtWidgets.QSpinBox()
        self.conroundsEdit.setRange(1, 1000)

        quitButton = QtWidgets.QPushButton("Quit", self)
        quitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        quitButton.resize(quitButton.sizeHint())

        loadButton = QtWidgets.QPushButton("Load settings from previous simulation")

        btngridR = QtWidgets.QGridLayout()

        self.concatExchangeEdit = QtWidgets.QCheckBox()
        self.exportkinetics = QtWidgets.QCheckBox()

        btngridR.addWidget(loadButton, 0, 0, 1, 2)
        btngridR.addWidget(QtWidgets.QLabel("Exchange rounds to be simulated:"), 1, 0)
        btngridR.addWidget(self.exchangeroundsEdit, 1, 1)
        btngridR.addWidget(QtWidgets.QLabel("Concatenate several rounds:"), 2, 0)
        btngridR.addWidget(self.conroundsEdit, 2, 1)
        btngridR.addWidget(QtWidgets.QLabel("Concatenate Exchange"))
        btngridR.addWidget(self.concatExchangeEdit, 3, 1)
        btngridR.addWidget(QtWidgets.QLabel("Export kinetic data"))
        btngridR.addWidget(self.exportkinetics, 4, 1)
        btngridR.addWidget(simulateButton, 5, 0, 1, 2)
        btngridR.addWidget(quitButton, 6, 0, 1, 2)

        simulateButton.clicked.connect(self.simulate)
        loadButton.clicked.connect(self.loadSettings)

        self.show()
        self.changeTime()
        self.changePSF()
        self.changeNoise()
        self.changePaint()

        pos_groupbox = QtWidgets.QGroupBox("Positions [Px]")
        str_groupbox = QtWidgets.QGroupBox("Structure [nm]")

        posgrid = QtWidgets.QGridLayout(pos_groupbox)
        strgrid = QtWidgets.QGridLayout(str_groupbox)

        self.figure1 = plt.figure()
        self.figure2 = plt.figure()
        self.canvas1 = FigureCanvas(self.figure1)
        csize = 180
        self.canvas1.setMinimumSize(csize, csize)

        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.setMinimumSize(csize, csize)

        posgrid.addWidget(self.canvas1)
        strgrid.addWidget(self.canvas2)

        self.mainpbar = QtWidgets.QProgressBar(self)
        # Arrange Buttons
        if ADVANCEDMODE:
            self.grid.addWidget(pos_groupbox, 1, 0)
            self.grid.addWidget(str_groupbox, 1, 1)
            self.grid.addWidget(structure_groupbox, 2, 0, 2, 1)
            self.grid.addWidget(camera_groupbox, 1, 2)
            self.grid.addWidget(paint_groupbox, 3, 1)
            self.grid.addWidget(imager_groupbox, 2, 1)
            self.grid.addWidget(noise_groupbox, 2, 2)
            self.grid.addLayout(btngridR, 3, 2)
            self.grid.addWidget(self.mainpbar, 5, 0, 1, 4)
            self.grid.addWidget(threed_groupbox, 4, 0)
            self.grid.addWidget(handles_groupbox, 4, 1)
        else:
            # Left side
            self.grid.addWidget(pos_groupbox, 1, 0)
            self.grid.addWidget(str_groupbox, 1, 1)
            self.grid.addWidget(structure_groupbox, 2, 0)
            self.grid.addWidget(paint_groupbox, 3, 0)
            self.grid.addWidget(handles_groupbox, 4, 0)
            self.grid.addWidget(threed_groupbox, 5, 0)

            # Right side
            self.grid.addWidget(imager_groupbox, 2, 1)
            self.grid.addWidget(camera_groupbox, 3, 1)
            self.grid.addLayout(btngridR, 4, 1, 2, 1)
            self.grid.addWidget(self.mainpbar, 8, 0, 1, 4)

        mainWidget = QtWidgets.QWidget()
        mainWidget.setLayout(self.grid)
        self.setCentralWidget(mainWidget)
        self.setGeometry(300, 300, 300, 150)
        # CALL FUNCTIONS
        self.generatePositions()

        self.mainpbar.setValue(0)
        self.statusBar().showMessage("Simulate ready.")

    def load3dCalibration(self):
        # if hasattr(self.window, 'movie_path'):
        #    dir = os.path.dirname(self.window.movie_path)
        # else:
        dir = None
        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load 3d calibration", directory=dir, filter="*.yaml"
        )
        if path:
            with open(path, "r") as f:
                z_calibration = yaml.full_load(f)
                self.cx = [_ for _ in z_calibration["X Coefficients"]]
                self.cy = [_ for _ in z_calibration["Y Coefficients"]]
                self.statusBar().showMessage("Caliration loaded from: " + path)

    def changeTime(self):
        laserpower = self.laserpowerEdit.value()
        itime = self.integrationtimeEdit.value()
        frames = self.framesEdit.value()
        totaltime = itime * frames / 1000 / 60
        totaltime = round(totaltime * 100) / 100
        self.totaltimeEdit.setText(str(totaltime))

        photonslope = self.photonslopeEdit.value()
        photonslopestd = photonslope / STDFACTOR
        if ADVANCEDMODE:
            photonslopestd = self.photonslopeStdEdit.value()
        photonrate = photonslope * laserpower
        photonratestd = photonslopestd * laserpower

        photonsframe = round(photonrate * itime)
        photonsframestd = round(photonratestd * itime)

        self.photonsframeEdit.setText(str(photonsframe))
        self.photonstdframeEdit.setText(str(photonsframestd))

        self.changeNoise()

    def changePaint(self):
        kon = self.konEdit.value()
        imagerconcentration = self.imagerconcentrationEdit.value()
        taud = round(1 / (kon * imagerconcentration * 1 / 10**9) * 1000)
        self.taudEdit.setText(str(taud))
        self.changeNoise()

    def changePSF(self):
        psf = self.psfEdit.value()
        pixelsize = self.pixelsizeEdit.value()
        psf_fwhm = round(psf * pixelsize * 2.355)
        self.psf_fwhmEdit.setText(str(psf_fwhm))

    def changeImager(self):
        laserpower = self.laserpowerEdit.value()

        itime = self.integrationtimeEdit.value()
        photonslope = self.photonslopeEdit.value()
        photonslopestd = photonslope / STDFACTOR
        if ADVANCEDMODE:
            photonslopestd = self.photonslopeStdEdit.value()
        photonrate = photonslope * laserpower
        photonratestd = photonslopestd * laserpower

        photonsframe = round(photonrate * itime)
        photonsframestd = round(photonratestd * itime)

        self.photonsframeEdit.setText(str(photonsframe))
        self.photonstdframeEdit.setText(str(photonsframestd))

        self.photonrateEdit.setValue((photonrate))
        self.photonratestdEdit.setValue((photonratestd))
        self.changeNoise()

    def changeNoise(self):
        itime = self.integrationtimeEdit.value()
        imagerconcentration = self.imagerconcentrationEdit.value()
        laserpower = self.laserpowerEdit.value() * POWERDENSITY_CONVERSION
        bglevel = self.backgroundlevelEdit.value()
        if ADVANCEDMODE:
            # NEW NOISE MODEL
            laserc = self.lasercEdit.value()
            imagerc = self.imagercEdit.value()
            bgoffset = self.BgoffsetEdit.value()
            bgmodel = (
                laserc + imagerc * imagerconcentration
            ) * laserpower * itime + bgoffset
            equationA = self.EquationAEdit.value()
            equationB = self.EquationBEdit.value()
            equationC = self.EquationCEdit.value()
            bgstdoffset = self.BgStdoffsetEdit.value()
            bgmodelstd = (
                equationA * laserpower * itime
                + equationB * bgmodel
                + equationC
                + bgstdoffset * bglevel
            )
            self.backgroundframeEdit.setText(str(int(bgmodel)))
            self.noiseEdit.setText(str(int(bgmodelstd)))
        else:
            bgmodel = (
                (LASERC_DEFAULT + IMAGERC_DEFAULT * imagerconcentration)
                * laserpower
                * itime
                * bglevel
            )
            self.backgroundframesimpleEdit.setText(str(int(bgmodel)))

    def changeStructureType(self):
        typeindex = self.structurecombo.currentIndex()
        # TYPEINDEX: 0 = GRID, 1 = CIRCLE, 2 = CUSTOM, 3 = Handles

        if typeindex == 0:
            self.structure1.show()
            self.structure2.show()
            self.structure3.show()
            self.structure1Edit.show()
            self.structure2Edit.show()
            self.structure3Edit.show()
            self.structure3Label.show()
            self.structure1.setText("Columns")
            self.structure2.setText("Rows")
            self.structure3.setText("Spacing X,Y")
            self.structure1Edit.setValue(3)
            self.structure2Edit.setValue(4)
            self.structure3Edit.setText("20,20")
        elif typeindex == 1:
            self.structure1.show()
            self.structure2.show()
            self.structure3.show()
            self.structure1Edit.show()
            self.structure2Edit.show()
            self.structure3Edit.show()
            self.structure3Label.show()
            self.structure1.hide()
            self.structure2.setText("Number of Labels")
            self.structure3.setText("Diameter")
            self.structure1Edit.hide()
            self.structure2Edit.setValue(12)
            self.structure3Edit.setText("100")
        elif typeindex == 2:
            self.structure1.hide()
            self.structure2.hide()
            self.structure3.hide()
            self.structure1Edit.hide()
            self.structure2Edit.hide()
            self.structure3Edit.hide()
            self.structure3Label.hide()
        elif typeindex == 3:
            self.structure1.hide()
            self.structure2.hide()
            self.structure3.hide()
            self.structure1Edit.hide()
            self.structure2Edit.hide()
            self.structure3Edit.hide()
            self.structure3Label.hide()
            self.structure3Label.hide()

        self.changeStructDefinition()

    def changeStructDefinition(self):

        typeindex = self.structurecombo.currentIndex()

        if typeindex == 0:  # grid

            rows = self.structure1Edit.value()
            cols = self.structure2Edit.value()

            spacingtxt = _np.asarray((self.structure3Edit.text()).split(","))
            try:
                spacingx = float(spacingtxt[0])
            except ValueError:
                spacingx = 1
            if spacingtxt.size > 1:
                try:
                    spacingy = float(spacingtxt[1])
                except ValueError:
                    spacingy = 1
            else:
                spacingy = 1

            structurexx = ""
            structureyy = ""
            structureex = ""
            structure3d = ""

            for i in range(0, rows):
                for j in range(0, cols):
                    structurexx = structurexx + str(i * spacingx) + ","
                    structureyy = structureyy + str(j * spacingy) + ","
                    structureex = structureex + "1,"
                    structure3d = structure3d + "0,"

            structurexx = structurexx[:-1]
            structureyy = structureyy[:-1]
            structureex = structureex[:-1]

            self.structurexxEdit.setText(structurexx)
            self.structureyyEdit.setText(structureyy)
            self.structureexEdit.setText(structureex)
            self.structure3DEdit.setText(structure3d)
            self.generatePositions()

        elif typeindex == 1:  # CIRCLE
            labels = self.structure2Edit.value()
            diametertxt = _np.asarray((self.structure3Edit.text()).split(","))
            try:
                diameter = float(diametertxt[0])
            except ValueError:
                diameter = 100

            twopi = 2 * 3.1415926535

            circdata = _np.arange(0, twopi, twopi / labels)

            xxval = _np.round(_np.cos(circdata) * diameter / 2 * 100) / 100
            yyval = _np.round(_np.sin(circdata) * diameter / 2 * 100) / 100

            structurexx = ""
            structureyy = ""
            structureex = ""
            structure3d = ""

            for i in range(0, xxval.size):
                structurexx = structurexx + str(xxval[i]) + ","
                structureyy = structureyy + str(yyval[i]) + ","
                structureex = structureex + "1,"
                structure3d = structure3d + "0,"

            structurexx = structurexx[:-1]
            structureyy = structureyy[:-1]
            structureex = structureex[:-1]

            self.structurexxEdit.setText(structurexx)
            self.structureyyEdit.setText(structureyy)
            self.structureexEdit.setText(structureex)
            self.structure3DEdit.setText(structure3d)
            self.generatePositions()

        elif typeindex == 2:  # Custom
            self.generatePositions()

        elif typeindex == 3:  # Handles
            print("Handles will be displayed..")

    def keyPressEvent(self, e):

        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    def vectorToString(self, x):
        x_arrstr = _np.char.mod("%f", x)
        x_str = ",".join(x_arrstr)
        return x_str

    def simulate(self):
        exchangeroundstoSim = _np.asarray((self.exchangeroundsEdit.text()).split(","))
        exchangeroundstoSim = exchangeroundstoSim.astype(int)

        noexchangecolors = len(set(exchangeroundstoSim))
        exchangecolors = list(set(exchangeroundstoSim))

        if self.concatExchangeEdit.checkState():
            conrounds = noexchangecolors
        else:
            conrounds = self.conroundsEdit.value()

        self.currentround += 1

        if self.currentround == 1:
            fileNameOld, exe = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save movie to..", filter="*.raw"
            )
            if fileNameOld:
                self.fileName = fileNameOld
            else:
                self.currentround -= 1
        else:
            fileNameOld = self.fileName

        if fileNameOld:

            self.statusBar().showMessage(
                "Set round " + str(self.currentround) + " of " + str(conrounds)
            )
            # READ IN PARAMETERS
            # STRUCTURE
            structureNo = self.structurenoEdit.value()
            structureFrame = self.structureframeEdit.value()
            structureIncorporation = self.structureIncorporationEdit.value()
            structureArrangement = int(self.structurerandomEdit.checkState())
            structureOrientation = int(self.structurerandomOrientationEdit.checkState())
            structurex = self.structurexxEdit.text()
            structurey = self.structureyyEdit.text()
            structureextxt = self.structureexEdit.text()

            structure3dtxt = self.structure3DEdit.text()

            # PAINT
            kon = self.konEdit.value()
            imagerconcentration = self.imagerconcentrationEdit.value()
            taub = self.taubEdit.value()
            taud = int(self.taudEdit.text())

            # IMAGER PARAMETERS
            psf = self.psfEdit.value()
            photonrate = self.photonrateEdit.value()
            photonratestd = self.photonratestdEdit.value()
            photonbudget = self.photonbudgetEdit.value()
            laserpower = self.laserpowerEdit.value()
            photonslope = self.photonslopeEdit.value()
            photonslopeStd = photonslope / STDFACTOR
            if ADVANCEDMODE:
                photonslopeStd = self.photonslopeStdEdit.value()

            if self.photonslopemodeEdit.checkState():
                photonratestd = 0

            # CAMERA PARAMETERS
            imagesize = self.camerasizeEdit.value()
            itime = self.integrationtimeEdit.value()
            frames = self.framesEdit.value()
            pixelsize = self.pixelsizeEdit.value()

            # NOISE MODEL
            if ADVANCEDMODE:
                background = int(self.backgroundframeEdit.text())
                noise = int(self.noiseEdit.text())
                laserc = self.lasercEdit.value()
                imagerc = self.imagercEdit.value()
                bgoffset = self.BgoffsetEdit.value()
                equationA = self.EquationAEdit.value()
                equationB = self.EquationBEdit.value()
                equationC = self.EquationCEdit.value()
                bgstdoffset = self.BgStdoffsetEdit.value()
            else:
                background = int(self.backgroundframesimpleEdit.text())
                noise = _np.sqrt(background)
                laserc = LASERC_DEFAULT
                imagerc = IMAGERC_DEFAULT
                bgoffset = BGOFFSET_DEFAULT
                equationA = EQA_DEFAULT
                equationB = EQB_DEFAULT
                equationC = EQC_DEFAULT
                bgstdoffset = BGSTDOFFSET_DEFAULT

            structurexx, structureyy, structureex, structure3d = self.readStructure()

            self.statusBar().showMessage("Simulation started")
            struct = self.newstruct

            handlex = self.vectorToString(struct[0, :])
            handley = self.vectorToString(struct[1, :])
            handleex = self.vectorToString(struct[2, :])
            handless = self.vectorToString(struct[3, :])
            handle3d = self.vectorToString(struct[4, :])

            mode3Dstate = int(self.mode3DEdit.checkState())

            t0 = time.time()

            if self.concatExchangeEdit.checkState():
                noexchangecolors = 1  # Overwrite the number to not trigger the for loop

            for i in range(0, noexchangecolors):

                if noexchangecolors > 1:
                    fileName = _io.multiple_filenames(fileNameOld, i)
                    partstruct = struct[:, struct[2, :] == exchangecolors[i]]
                elif self.concatExchangeEdit.checkState():
                    fileName = fileNameOld
                    partstruct = struct[
                        :,
                        struct[2, :] == exchangecolors[self.currentround - 1],
                    ]
                else:
                    fileName = fileNameOld
                    partstruct = struct[:, struct[2, :] == exchangecolors[0]]

                self.statusBar().showMessage("Distributing photons ...")

                bindingsitesx = partstruct[0, :]

                nosites = len(bindingsitesx)  # number of binding sites in image
                photondist = _np.zeros((nosites, frames), dtype=int)
                spotkinetics = _np.zeros((nosites, 4), dtype=float)

                timetrace = {}

                for i in range(0, nosites):
                    p_temp, t_temp, k_temp = simulate.distphotons(
                        partstruct,
                        itime,
                        frames,
                        taud,
                        taub,
                        photonrate,
                        photonratestd,
                        photonbudget,
                    )
                    photondist[i, :] = p_temp
                    spotkinetics[i, :] = k_temp
                    timetrace[i] = self.vectorToString(t_temp)
                    outputmsg = (
                        "Distributing photons ... "
                        + str(_np.round(i / nosites * 1000) / 10)
                        + " %"
                    )
                    self.statusBar().showMessage(outputmsg)
                    self.mainpbar.setValue(int(_np.round(i / nosites * 1000) / 10))

                self.statusBar().showMessage("Converting to image ... ")
                onevents = self.vectorToString(spotkinetics[:, 0])
                localizations = self.vectorToString(spotkinetics[:, 1])
                meandarksim = self.vectorToString(spotkinetics[:, 2])
                meanbrightsim = self.vectorToString(spotkinetics[:, 3])

                movie = _np.zeros(shape=(frames, imagesize, imagesize))

                info = {
                    "Generated by": "Picasso simulate",
                    "Byte Order": "<",
                    "Camera": "Simulation",
                    "Data Type": "uint16",
                    "Frames": frames,
                    "Structure.Frame": structureFrame,
                    "Structure.Number": structureNo,
                    "Structure.StructureX": structurex,
                    "Structure.StructureY": structurey,
                    "Structure.StructureEx": structureextxt,
                    "Structure.Structure3D": structure3dtxt,
                    "Structure.HandleX": handlex,
                    "Structure.HandleY": handley,
                    "Structure.HandleEx": handleex,
                    "Structure.Handle3d": handle3d,
                    "Structure.HandleStruct": handless,
                    "Structure.Incorporation": structureIncorporation,
                    "Structure.Arrangement": structureArrangement,
                    "Structure.Orientation": structureOrientation,
                    "Structure.3D": mode3Dstate,
                    "Structure.CX": self.cx,
                    "Structure.CY": self.cy,
                    "PAINT.k_on": kon,
                    "PAINT.imager": imagerconcentration,
                    "PAINT.taub": taub,
                    "Imager.PSF": psf,
                    "Imager.Photonrate": photonrate,
                    "Imager.Photonrate Std": photonratestd,
                    "Imager.Constant Photonrate Std": int(
                        self.photonslopemodeEdit.checkState()
                    ),
                    "Imager.Photonbudget": photonbudget,
                    "Imager.Laserpower": laserpower,
                    "Imager.Photonslope": photonslope,
                    "Imager.PhotonslopeStd": photonslopeStd,
                    "Imager.BackgroundLevel": self.backgroundlevelEdit.value(),
                    "Camera.Image Size": imagesize,
                    "Camera.Integration Time": itime,
                    "Camera.Frames": frames,
                    "Camera.Pixelsize": pixelsize,
                    "Noise.Lasercoefficient": laserc,
                    "Noise.Imagercoefficient": imagerc,
                    "Noise.EquationA": equationA,
                    "Noise.EquationB": equationB,
                    "Noise.EquationC": equationC,
                    "Noise.BackgroundOff": bgoffset,
                    "Noise.BackgroundStdOff": bgstdoffset,
                    "Spotkinetics.ON_Events": onevents,
                    "Spotkinetics.Localizations": localizations,
                    "Spotkinetics.MEAN_DARK": meandarksim,
                    "Spotkinetics.MEAN_BRIGHT": meanbrightsim,
                    "Height": imagesize,
                    "Width": imagesize,
                }

                if conrounds != 1:
                    app = QtCore.QCoreApplication.instance()
                    for runner in range(0, frames):
                        movie[runner, :, :] = simulate.convertMovie(
                            runner,
                            photondist,
                            partstruct,
                            imagesize,
                            frames,
                            psf,
                            photonrate,
                            background,
                            noise,
                            mode3Dstate,
                            self.cx,
                            self.cy,
                        )
                        outputmsg = (
                            "Converting to Image ... "
                            + str(_np.round(runner / frames * 1000) / 10)
                            + " %"
                        )

                        self.statusBar().showMessage(outputmsg)
                        self.mainpbar.setValue(_np.round(runner / frames * 1000) / 10)
                        app.processEvents()

                    if self.currentround == 1:
                        self.movie = movie
                    else:
                        movie = movie + self.movie
                        self.movie = movie

                    self.statusBar().showMessage(
                        "Converting to image ... complete. Current round: "
                        + str(self.currentround)
                        + " of "
                        + str(conrounds)
                        + ". Please set and start next round."
                    )
                    if self.currentround == conrounds:
                        self.statusBar().showMessage("Adding noise to movie ...")
                        movie = simulate.noisy_p(movie, background)
                        movie = simulate.check_type(movie)
                        self.statusBar().showMessage("Saving movie ...")

                        simulate.saveMovie(fileName, movie, info)
                        self.statusBar().showMessage("Movie saved to: " + fileName)
                        dt = time.time() - t0
                        self.statusBar().showMessage(
                            "All computations finished. Last file saved to: "
                            + fileName
                            + ". Time elapsed: {:.2f} Seconds.".format(dt)
                        )
                        self.currentround = 0
                    else:  # just save info file
                        # self.statusBar().showMessage('Saving yaml ...')
                        info_path = (
                            _ospath.splitext(fileName)[0]
                            + "_"
                            + str(self.currentround)
                            + ".yaml"
                        )
                        _io.save_info(info_path, [info])

                        if self.exportkinetics.isChecked():
                            # Export the kinetic data if this is checked
                            kinfo_path = (
                                _ospath.splitext(fileName)[0]
                                + "_"
                                + str(self.currentround)
                                + "_kinetics.yaml"
                            )
                            _io.save_info(kinfo_path, [timetrace])

                        self.statusBar().showMessage("Movie saved to: " + fileName)

                else:
                    app = QtCore.QCoreApplication.instance()
                    for runner in range(0, frames):
                        movie[runner, :, :] = simulate.convertMovie(
                            runner,
                            photondist,
                            partstruct,
                            imagesize,
                            frames,
                            psf,
                            photonrate,
                            background,
                            noise,
                            mode3Dstate,
                            self.cx,
                            self.cy,
                        )
                        outputmsg = (
                            "Converting to Image ... "
                            + str(_np.round(runner / frames * 1000) / 10)
                            + " %"
                        )

                        self.statusBar().showMessage(outputmsg)
                        self.mainpbar.setValue(int(_np.round(runner / frames * 1000) / 10))
                        app.processEvents()

                    movie = simulate.noisy_p(movie, background)
                    movie = simulate.check_type(movie)
                    self.mainpbar.setValue(100)
                    self.statusBar().showMessage("Converting to image ... complete.")
                    self.statusBar().showMessage("Saving movie ...")

                    simulate.saveMovie(fileName, movie, info)
                    if self.exportkinetics.isChecked():
                        # Export the kinetic data if this is checked
                        kinfo_path = _ospath.splitext(fileName)[0] + "_kinetics.yaml"
                        _io.save_info(kinfo_path, [timetrace])
                    self.statusBar().showMessage("Movie saved to: " + fileName)
                    dt = time.time() - t0
                    self.statusBar().showMessage(
                        "All computations finished. Last file saved to: "
                        + fileName
                        + ". Time elapsed: {:.2f} Seconds.".format(dt)
                    )
                    self.currentround = 0

    def loadSettings(self):  # TODO: re-write exceptions, check key
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open yaml", filter="*.yaml"
        )
        if path:
            info = _io.load_info(path)
            self.framesEdit.setValue(info[0]["Frames"])

            self.structureframeEdit.setValue(info[0]["Structure.Frame"])
            self.structurenoEdit.setValue(info[0]["Structure.Number"])
            self.structurexxEdit.setText(info[0]["Structure.StructureX"])
            self.structureyyEdit.setText(info[0]["Structure.StructureY"])
            self.structureexEdit.setText(info[0]["Structure.StructureEx"])
            try:
                self.structure3DEdit.setText(info[0]["Structure.Structure3D"])
                self.mode3DEdit.setCheckState(info[0]["Structure.3D"])
                self.cx(info[0]["Structure.CX"])
                self.cy(info[0]["Structure.CY"])
            except Exception as e:
                print(e)
                pass
            try:
                self.photonslopemodeEdit.setCheckState(
                    info[0]["Imager.Constant Photonrate Std"]
                )
            except Exception as e:
                print(e)
                pass

            try:
                self.backgroundlevelEdit.setValue(info[0]["Imager.BackgroundLevel"])
            except Exception as e:
                print(e)
                pass
            self.structureIncorporationEdit.setValue(info[0]["Structure.Incorporation"])

            self.structurerandomEdit.setCheckState(info[0]["Structure.Arrangement"])
            self.structurerandomOrientationEdit.setCheckState(
                info[0]["Structure.Orientation"]
            )

            self.konEdit.setValue(info[0]["PAINT.k_on"])
            self.imagerconcentrationEdit.setValue(info[0]["PAINT.imager"])
            self.taubEdit.setValue(info[0]["PAINT.taub"])

            self.psfEdit.setValue(info[0]["Imager.PSF"])
            self.photonrateEdit.setValue(info[0]["Imager.Photonrate"])
            self.photonratestdEdit.setValue(info[0]["Imager.Photonrate Std"])
            self.photonbudgetEdit.setValue(info[0]["Imager.Photonbudget"])
            self.laserpowerEdit.setValue(info[0]["Imager.Laserpower"])
            self.photonslopeEdit.setValue(info[0]["Imager.Photonslope"])
            self.photonslopeStdEdit.setValue(info[0]["Imager.PhotonslopeStd"])

            self.camerasizeEdit.setValue(info[0]["Camera.Image Size"])
            self.integrationtimeEdit.setValue(info[0]["Camera.Integration Time"])
            self.framesEdit.setValue(info[0]["Camera.Frames"])
            self.pixelsizeEdit.setValue(info[0]["Camera.Pixelsize"])

            if ADVANCEDMODE:
                self.lasercEdit.setValue(info[0]["Noise.Lasercoefficient"])
                self.imagercEdit.setValue(info[0]["Noise.Imagercoefficient"])
                self.BgoffsetEdit.setValue(info[0]["Noise.BackgroundOff"])

                self.EquationAEdit.setValue(info[0]["Noise.EquationA"])
                self.EquationBEdit.setValue(info[0]["Noise.EquationB"])
                self.EquationCEdit.setValue(info[0]["Noise.EquationC"])
                self.BgStdoffsetEdit.setValue(info[0]["Noise.BackgroundStdOff"])

            # SET POSITIONS
            handlexx = _np.asarray((info[0]["Structure.HandleX"]).split(","))
            handleyy = _np.asarray((info[0]["Structure.HandleY"]).split(","))
            handleex = _np.asarray((info[0]["Structure.HandleEx"]).split(","))
            handless = _np.asarray((info[0]["Structure.HandleStruct"]).split(","))

            handlexx = handlexx.astype(float)
            handleyy = handleyy.astype(float)
            handleex = handleex.astype(float)
            handless = handless.astype(float)

            handleex = handleex.astype(int)
            handless = handless.astype(int)

            handle3d = _np.asarray((info[0]["Structure.Handle3d"]).split(","))
            handle3d = handle3d.astype(float)
            structure = _np.array([handlexx, handleyy, handleex, handless, handle3d])

            self.structurecombo.setCurrentIndex(2)
            self.newstruct = structure
            self.plotPositions()
            self.statusBar().showMessage("Settings loaded from: " + path)

    def importDesign(self):
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open yaml", filter="*.yaml"
        )
        if path:
            info = _io.load_info(path)
            self.structurexxEdit.setText(info[0]["Structure.StructureX"])
            self.structureyyEdit.setText(info[0]["Structure.StructureY"])
            self.structureexEdit.setText(info[0]["Structure.StructureEx"])
            structure3d = ""
            for i in range(0, len(self.structurexxEdit.text())):
                structure3d = structure3d + "0,"

            self.structure3DEdit.setText(structure3d)
            self.structurecombo.setCurrentIndex(2)

    def readLine(self, linetxt, type="float", textmode=True):
        if textmode:
            line = _np.asarray((linetxt.text()).split(","))
        else:
            line = _np.asarray((linetxt.split(",")))

        values = []
        for element in line:
            try:
                if type == "int":
                    values.append(int(element))
                elif type == "float":
                    values.append(float(element))

            except ValueError:
                pass
        return values

    def importHandles(self):
        # Import structure <>
        self.handles = {}
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open yaml", filter="*.yaml *.hdf5"
        )
        if path:
            splitpath = _ospath.splitext(path)
            if splitpath[-1] == ".yaml":

                info = _io.load_info(path)

                x = self.readLine(info[0]["Structure.StructureX"], textmode=False)
                y = self.readLine(info[0]["Structure.StructureY"], textmode=False)
                try:
                    ex = self.readLine(
                        info[0]["Structure.StructureEx"],
                        type="int",
                        textmode=False,
                    )
                except Exception as e:
                    print(e)
                    ex = _np.ones_like(x)

                try:
                    z = self.readLine(info[0]["Structure.Structure3D"])
                except Exception as e:
                    print(e)
                    z = _np.zeros_like(x)

                minlen = min(len(x), len(y), len(ex), len(z))

                x = x[0:minlen]
                y = y[0:minlen]
                ex = ex[0:minlen]
                z = z[0:minlen]

            else:

                clusters = _io.load_clusters(path)

                pixelsize = self.pixelsizeEdit.value()
                imagesize = self.camerasizeEdit.value()

                x = clusters["com_x"]
                y = clusters["com_y"]

                # Align in the center of window:
                x = x - _np.mean(x) + imagesize / 2
                y = -(y - _np.mean(y)) + imagesize / 2

                x = x * pixelsize
                y = y * pixelsize

                try:
                    z = clusters["com_z"]
                except Exception as e:
                    print(e)
                    z = _np.zeros_like(x)

                ex = _np.ones_like(x)
                minlen = len(x)

            self.handles["x"] = x
            self.handles["y"] = y
            self.handles["z"] = z
            self.handles["ex"] = ex

            # TODO: Check axis orientation
            exchangecolors = list(set(self.handles["ex"]))
            exchangecolorsList = ",".join(map(str, exchangecolors))
            # UPDATE THE EXCHANGE COLORS IN BUTTON TO BE simulated
            self.exchangeroundsEdit.setText(str(exchangecolorsList))

            self.structurenoEdit.setValue(1)
            self.structureMode = False
            self.generatePositions()

            self.statusBar().showMessage("A total of {} points loaded.".format(minlen))

    def readStructure(self):
        structurexx = self.readLine(self.structurexxEdit)
        structureyy = self.readLine(self.structureyyEdit)
        structureex = self.readLine(self.structureexEdit, "int")
        structure3d = self.readLine(self.structure3DEdit)

        minlen = min(
            len(structureex),
            len(structurexx),
            len(structureyy),
            len(structure3d),
        )

        structurexx = structurexx[0:minlen]
        structureyy = structureyy[0:minlen]
        structureex = structureex[0:minlen]
        structure3d = structure3d[0:minlen]

        return structurexx, structureyy, structureex, structure3d

    def plotStructure(self):

        structurexx, structureyy, structureex, structure3d = self.readStructure()
        noexchangecolors = len(set(structureex))
        exchangecolors = list(set(structureex))
        # self.figure2.suptitle('Structure [nm]')
        ax1 = self.figure2.add_subplot(111)
        ax1.cla()
        # ax1.hold(True)
        ax1.axis("equal")

        for i in range(0, noexchangecolors):
            plotxx = []
            plotyy = []
            for j in range(0, len(structureex)):
                if structureex[j] == exchangecolors[i]:
                    plotxx.append(structurexx[j])
                    plotyy.append(structureyy[j])
            ax1.plot(plotxx, plotyy, "o")

        distx = round(1 / 10 * (max(structurexx) - min(structurexx)))
        disty = round(1 / 10 * (max(structureyy) - min(structureyy)))

        ax1.axes.set_xlim((min(structurexx) - distx, max(structurexx) + distx))
        ax1.axes.set_ylim((min(structureyy) - disty, max(structureyy) + disty))
        self.canvas2.draw()

        exchangecolorsList = ",".join(map(str, exchangecolors))
        # UPDATE THE EXCHANGE COLORS IN BUTTON TO BE simulated
        self.exchangeroundsEdit.setText(str(exchangecolorsList))

    def generatePositions(self):
        self.plotStructure()
        pixelsize = self.pixelsizeEdit.value()
        if self.structureMode:
            structurexx, structureyy, structureex, structure3d = self.readStructure()
            structure = simulate.defineStructure(
                structurexx, structureyy, structureex, structure3d, pixelsize
            )
        else:
            structurexx = self.handles["x"]
            structureyy = self.handles["y"]
            structureex = self.handles["ex"]
            structure3d = self.handles["z"]
            structure = simulate.defineStructure(
                structurexx,
                structureyy,
                structureex,
                structure3d,
                pixelsize,
                mean=False,
            )

        number = self.structurenoEdit.value()
        imageSize = self.camerasizeEdit.value()
        frame = self.structureframeEdit.value()
        arrangement = int(self.structurerandomEdit.checkState())
        gridpos = simulate.generatePositions(number, imageSize, frame, arrangement)

        orientation = int(self.structurerandomOrientationEdit.checkState())
        incorporation = self.structureIncorporationEdit.value() / 100
        exchange = 0

        if self.structureMode:
            self.newstruct = simulate.prepareStructures(
                structure,
                gridpos,
                orientation,
                number,
                incorporation,
                exchange,
            )
        else:
            self.newstruct = simulate.prepareStructures(
                structure,
                _np.array([[0, 0]]),
                orientation,
                number,
                incorporation,
                exchange,
            )

            in_x = _np.logical_and(
                self.newstruct[0, :] < (imageSize - frame),
                self.newstruct[0, :] > frame,
            )
            in_y = _np.logical_and(
                self.newstruct[1, :] < (imageSize - frame),
                self.newstruct[1, :] > frame,
            )
            in_frame = _np.logical_and(in_x, in_y)
            self.newstruct = self.newstruct[:, in_frame]

        # self.figure1.suptitle('Positions [Px]')
        ax1 = self.figure1.add_subplot(111)
        ax1.cla()
        # ax1.hold(True)
        ax1.axis("equal")
        ax1.plot(self.newstruct[0, :], self.newstruct[1, :], "+")
        # PLOT FRAME
        ax1.add_patch(
            patches.Rectangle(
                (frame, frame),
                imageSize - 2 * frame,
                imageSize - 2 * frame,
                linestyle="dashed",
                edgecolor="#000000",
                fill=False,  # remove background
            )
        )

        ax1.axes.set_xlim(0, imageSize)
        ax1.axes.set_ylim(0, imageSize)

        self.canvas1.draw()

        # PLOT first structure
        struct1 = self.newstruct[:, self.newstruct[3, :] == 0]

        noexchangecolors = len(set(struct1[2, :]))
        exchangecolors = list(set(struct1[2, :]))
        self.noexchangecolors = exchangecolors
        # self.figure2.suptitle('Structure [nm]')
        ax1 = self.figure2.add_subplot(111)
        ax1.cla()
        # ax1.hold(True)
        ax1.axis("equal")

        structurexx = struct1[0, :]
        structureyy = struct1[1, :]
        structureex = struct1[2, :]
        structurexx_nm = _np.multiply(structurexx - min(structurexx), pixelsize)
        structureyy_nm = _np.multiply(structureyy - min(structureyy), pixelsize)

        for i in range(0, noexchangecolors):
            plotxx = []
            plotyy = []
            for j in range(0, len(structureex)):
                if structureex[j] == exchangecolors[i]:
                    plotxx.append(structurexx_nm[j])
                    plotyy.append(structureyy_nm[j])
            ax1.plot(plotxx, plotyy, "o")

            distx = round(1 / 10 * (max(structurexx_nm) - min(structurexx_nm)))
            disty = round(1 / 10 * (max(structureyy_nm) - min(structureyy_nm)))

            ax1.axes.set_xlim(
                (min(structurexx_nm) - distx, max(structurexx_nm) + distx)
            )
            ax1.axes.set_ylim(
                (min(structureyy_nm) - disty, max(structureyy_nm) + disty)
            )
        self.canvas2.draw()

    def plotPositions(self):
        structurexx, structureyy, structureex, structure3d = self.readStructure()
        pixelsize = self.pixelsizeEdit.value()
        structure = simulate.defineStructure(
            structurexx, structureyy, structureex, structure3d, pixelsize
        )

        number = self.structurenoEdit.value()
        imageSize = self.camerasizeEdit.value()
        frame = self.structureframeEdit.value()
        arrangement = int(self.structurerandomEdit.checkState())
        gridpos = simulate.generatePositions(number, imageSize, frame, arrangement)

        orientation = int(self.structurerandomOrientationEdit.checkState())
        incorporation = self.structureIncorporationEdit.value() / 100
        exchange = 0

        # self.figure1.suptitle('Positions [Px]')
        ax1 = self.figure1.add_subplot(111)
        ax1.cla()
        # ax1.hold(True)
        ax1.axis("equal")
        ax1.plot(self.newstruct[0, :], self.newstruct[1, :], "+")
        # PLOT FRAME
        ax1.add_patch(
            patches.Rectangle(
                (frame, frame),
                imageSize - 2 * frame,
                imageSize - 2 * frame,
                linestyle="dashed",
                edgecolor="#000000",
                fill=False,  # remove background
            )
        )

        ax1.axes.set_xlim(0, imageSize)
        ax1.axes.set_ylim(0, imageSize)

        self.canvas1.draw()

        # PLOT first structure
        struct1 = self.newstruct[:, self.newstruct[3, :] == 0]

        noexchangecolors = len(set(struct1[2, :]))
        exchangecolors = list(set(struct1[2, :]))
        self.noexchangecolors = exchangecolors
        # self.figure2.suptitle('Structure [nm]')
        ax1 = self.figure2.add_subplot(111)
        ax1.cla()
        # ax1.hold(True)

        structurexx = struct1[0, :]
        structureyy = struct1[1, :]
        structureex = struct1[2, :]
        structurexx_nm = _np.multiply(structurexx - min(structurexx), pixelsize)
        structureyy_nm = _np.multiply(structureyy - min(structureyy), pixelsize)

        for i in range(0, noexchangecolors):
            plotxx = []
            plotyy = []
            for j in range(0, len(structureex)):
                if structureex[j] == exchangecolors[i]:
                    plotxx.append(structurexx_nm[j])
                    plotyy.append(structureyy_nm[j])
            ax1.plot(plotxx, plotyy, "o")

            distx = round(1 / 10 * (max(structurexx_nm) - min(structurexx_nm)))
            disty = round(1 / 10 * (max(structureyy_nm) - min(structureyy_nm)))

            ax1.axes.set_xlim(
                (min(structurexx_nm) - distx, max(structurexx_nm) + distx)
            )
            ax1.axes.set_ylim(
                (min(structureyy_nm) - disty, max(structureyy_nm) + disty)
            )
        self.canvas2.draw()

    def openDialog(self):
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open design", filter="*.yaml"
        )
        if path:
            self.mainscene.loadCanvas(path)
            self.statusBar().showMessage("File loaded from: " + path)

    def importhdf5(self):
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open localizations", filter="*.hdf5"
        )
        if path:
            self.readhdf5(path)

    def calibrateNoise(self):

        bg, bgstd, las, time, conc, ok = CalibrationDialog.setExt()

        _np.asarray(bg)
        _np.asarray(bgstd)
        _np.asarray(las)
        _np.asarray(time)
        _np.asarray(conc)

        x_3d = _np.array([conc, las, time])
        p0 = [1, 1]
        fitParamsBg, fitCovariances = curve_fit(fitFuncBg, x_3d, bg, p0)
        print(" fit coefficients :\n", fitParamsBg)

        # SET VALUES TO PARAMETER
        self.lasercEdit.setValue(fitParamsBg[0])
        self.imagercEdit.setValue(fitParamsBg[1])

        x_3dStd = _np.array([las, time, bg])
        p0S = [1, 1, 1]
        fitParamsStd, fitCovariances = curve_fit(fitFuncStd, x_3dStd, bgstd, p0S)

        print(" fit coefficients2:\n", fitParamsStd)

        self.EquationAEdit.setValue(fitParamsStd[0])
        self.EquationBEdit.setValue(fitParamsStd[1])
        self.EquationCEdit.setValue(fitParamsStd[2])

        # Noise model working point

        figure4 = plt.figure()

        # Background
        bgmodel = fitFuncBg(x_3d, fitParamsBg[0], fitParamsBg[1])
        ax1 = figure4.add_subplot(121)
        ax1.cla()
        ax1.plot(bg, bgmodel, "o")
        x = _np.linspace(*ax1.get_xlim())
        ax1.plot(x, x)
        title = "Background Model:"
        ax1.set_title(title)

        # Std
        bgmodelstd = fitFuncStd(
            x_3dStd, fitParamsStd[0], fitParamsStd[1], fitParamsStd[2]
        )
        ax2 = figure4.add_subplot(122)
        ax2.cla()
        ax2.plot(bgstd, bgmodelstd, "o")
        x = _np.linspace(*ax2.get_xlim())
        ax2.plot(x, x)
        title = "Background Model Std:"
        ax2.set_title(title)

        figure4.show()

    def sigmafilter(self, data, sigmas):
        # Filter data to be withing +- sigma
        sigma = _np.std(data)
        mean = _np.mean(data)

        datanew = data[data < (mean + sigmas * sigma)]
        datanew = datanew[datanew > (mean - sigmas * sigma)]
        return datanew

    def readhdf5(self, path):
        try:
            locs, self.info = _io.load_locs(path, qt_parent=self)
        except _io.NoMetadataFileError:
            return
        integrationtime, ok1 = QtWidgets.QInputDialog.getText(
            self, "Input Dialog", "Enter integration time in ms:"
        )
        integrationtime = int(integrationtime)
        if ok1:
            imagerconcentration, ok2 = QtWidgets.QInputDialog.getText(
                self, "Input Dialog", "Enter imager concentration in nM:"
            )
            imagerconcentration = float(imagerconcentration)

            if ok2:
                laserpower, ok3 = QtWidgets.QInputDialog.getText(
                    self, "Input Dialog", "Enter Laserpower in mW:"
                )
                laserpower = float(laserpower)
                if ok3:
                    cbaseline, ok4 = QtWidgets.QInputDialog.getText(
                        self, "Input Dialog", "Enter camera baseline"
                    )
                    cbaseline = float(cbaseline)
                    # self.le.setText(str(text))

                    photons = locs["photons"]
                    sigmax = locs["sx"]
                    sigmay = locs["sy"]
                    bg = locs["bg"]
                    bg = bg - cbaseline

                    nosigmas = 3
                    photons = self.sigmafilter(photons, nosigmas)
                    sigmax = self.sigmafilter(sigmax, nosigmas)
                    sigmay = self.sigmafilter(sigmay, nosigmas)
                    bg = self.sigmafilter(bg, nosigmas)

                    figure3 = plt.figure()

                    # Photons
                    photonsmu, photonsstd = norm.fit(photons)
                    ax1 = figure3.add_subplot(131)
                    ax1.cla()
                    # ax1.hold(True) # TODO: Investigate again what this causes
                    ax1.hist(photons, bins=25, normed=True, alpha=0.6)
                    xmin, xmax = plt.xlim()
                    x = _np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, photonsmu, photonsstd)
                    ax1.plot(x, p)
                    title = "Photons:\n mu = %.2f\n  std = %.2f" % (
                        photonsmu,
                        photonsstd,
                    )
                    ax1.set_title(title)

                    # Sigma X & Sigma Y
                    sigma = _np.concatenate((sigmax, sigmay), axis=0)
                    sigmamu, sigmastd = norm.fit(sigma)
                    ax2 = figure3.add_subplot(132)
                    ax2.cla()
                    # ax2.hold(True)
                    ax2.hist(sigma, bins=25, normed=True, alpha=0.6)
                    xmin, xmax = plt.xlim()
                    x = _np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, sigmamu, sigmastd)
                    ax2.plot(x, p)
                    title = "PSF:\n mu = %.2f\n  std = %.2f" % (
                        sigmamu,
                        sigmastd,
                    )
                    ax2.set_title(title)

                    # Background
                    bgmu, bgstd = norm.fit(bg)
                    ax3 = figure3.add_subplot(133)
                    ax3.cla()
                    # ax3.hold(True)
                    # Plot the histogram.
                    ax3.hist(bg, bins=25, normed=True, alpha=0.6)
                    xmin, xmax = plt.xlim()
                    x = _np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, bgmu, bgstd)
                    ax3.plot(x, p)
                    title = "Background:\n mu = %.2f\n  std = %.2f" % (
                        bgmu,
                        bgstd,
                    )
                    ax3.set_title(title)
                    figure3.tight_layout()
                    figure3.show()

                    # Calculate Rates
                    # Photonrate, Photonrate Std, PSF

                    photonrate = int(photonsmu / integrationtime)
                    photonratestd = int(photonsstd / integrationtime)
                    psf = int(sigmamu * 100) / 100
                    photonrate = int(photonsmu / integrationtime)

                    # CALCULATE BG AND BG_STD FROM MODEL AND ADJUST OFFSET
                    laserc = self.lasercEdit.value()
                    imagerc = self.imagercEdit.value()

                    bgmodel = (
                        (laserc + imagerc * imagerconcentration)
                        * laserpower
                        * integrationtime
                    )

                    equationA = self.EquationAEdit.value()
                    equationB = self.EquationBEdit.value()
                    equationC = self.EquationCEdit.value()

                    bgmodelstd = (
                        equationA * laserpower * integrationtime
                        + equationB * bgmu
                        + equationC
                    )

                    # SET VALUES TO FIELDS AND CALL DEPENDENCIES
                    self.psfEdit.setValue(psf)

                    self.integrationtimeEdit.setValue(integrationtime)
                    self.photonrateEdit.setValue(photonrate)
                    self.photonratestdEdit.setValue(photonratestd)
                    self.photonslopeEdit.setValue(photonrate / laserpower)
                    self.photonslopeStdEdit.setValue(photonratestd / laserpower)

                    # SET NOISE AND FRAME
                    self.BgoffsetEdit.setValue(bgmu - bgmodel)
                    self.BgStdoffsetEdit.setValue(bgstd - bgmodelstd)

                    self.imagerconcentrationEdit.setValue(imagerconcentration)
                    self.laserpowerEdit.setValue(laserpower)


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(CalibrationDialog, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget()
        self.table.setWindowTitle("Noise Model Calibration")
        self.setWindowTitle("Noise Model Calibration")
        # self.resize(800, 400)

        layout.addWidget(self.table)

        # ADD BUTTONS:
        self.loadTifButton = QtWidgets.QPushButton("Load Tifs")
        layout.addWidget(self.loadTifButton)

        self.evalTifButton = QtWidgets.QPushButton("Evaluate Tifs")
        layout.addWidget(self.evalTifButton)

        self.pbar = QtWidgets.QProgressBar(self)
        layout.addWidget(self.pbar)

        self.loadTifButton.clicked.connect(self.loadTif)
        self.evalTifButton.clicked.connect(self.evalTif)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.ActionRole
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )

        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def exportTable(self):

        table = dict()
        tablecontent = []
        tablecontent.append(
            [
                "FileName",
                "Imager concentration[nM]",
                "Integration time [ms]",
                "Laserpower",
                "Mean [Photons]",
                "Std [Photons]",
            ]
        )
        for row in range(self.table.rowCount()):
            rowdata = []
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                if item is not None:
                    rowdata.append(item.text())
                else:
                    rowdata.append("")
            tablecontent.append(rowdata)

        table[0] = tablecontent
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export calibration table to.", filter="*.csv"
        )
        if path:
            self.savePlate(path, table)

    def savePlate(self, filename, data):
        with open(filename, "w", newline="") as csvfile:
            Writer = csv.writer(
                csvfile,
                delimiter=",",
                quotechar="|",
                quoting=csv.QUOTE_MINIMAL,
            )
            for j in range(0, len(data)):
                exportdata = data[j]
                for i in range(0, len(exportdata)):
                    Writer.writerow(exportdata[i])

    def evalTif(self):

        baseline, ok1 = QtWidgets.QInputDialog.getText(
            self, "Input Dialog", "Enter Camera Baseline:"
        )
        if ok1:
            baseline = int(baseline)
        else:
            baseline = 200  # default

        sensitvity, ok2 = QtWidgets.QInputDialog.getText(
            self, "Input Dialog", "Enter Camera Sensitivity:"
        )
        if ok2:
            sensitvity = float(sensitvity)
        else:
            sensitvity = 1.47

        counter = 0
        for element in self.tifFiles:
            counter = counter + 1
            self.pbar.setValue((counter - 1) / self.tifCounter * 100)
            print("Current Dataset: " + str(counter) + " of " + str(self.tifCounter))
            QtWidgets.qApp.processEvents()
            movie, info = _io.load_movie(element)

            movie = movie[0:100, :, :]

            movie = (movie - baseline) * sensitvity
            self.table.setItem(
                counter - 1, 4, QtWidgets.QTableWidgetItem(str((_np.mean(movie))))
            )
            self.table.setItem(
                counter - 1, 5, QtWidgets.QTableWidgetItem(str((_np.std(movie))))
            )

            self.table.setItem(
                counter - 1,
                1,
                QtWidgets.QTableWidgetItem(str((self.ValueFind(element, "nM_")))),
            )
            self.table.setItem(
                counter - 1,
                2,
                QtWidgets.QTableWidgetItem(str((self.ValueFind(element, "ms_")))),
            )
            self.table.setItem(
                counter - 1,
                3,
                QtWidgets.QTableWidgetItem(str((self.ValueFind(element, "mW_")))),
            )

        self.pbar.setValue(100)

    def ValueFind(self, filename, unit):
        index = filename.index(unit)

        value = 0

        for i in range(4):
            try:
                value += int(filename[index - 1 - i]) * (10**i)
            except ValueError:
                pass

        return value

    def loadTif(self):

        self.path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.path:

            self.tifCounter = len(_glob.glob1(self.path, "*.tif"))
            self.tifFiles = _glob.glob(os.path.join(self.path, "*.tif"))

            self.table.setRowCount(int(self.tifCounter))
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(
                [
                    "FileName",
                    "Imager concentration[nM]",
                    "Integration time [ms]",
                    "Laserpower",
                    "Mean [Photons]",
                    "Std [Photons]",
                ]
            )

            for i in range(0, self.tifCounter):
                self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(self.tifFiles[i]))

    def changeComb(self, indexval):

        sender = self.sender()
        comboval = sender.currentIndex()
        if comboval == 0:
            self.table.setItem(indexval, 2, QtWidgets.QTableWidgetItem(""))
            self.table.setItem(indexval, 3, QtWidgets.QTableWidgetItem(""))
        else:
            self.table.setItem(
                indexval,
                2,
                QtWidgets.QTableWidgetItem(self.ImagersShort[comboval]),
            )
            self.table.setItem(
                indexval, 3, QtWidgets.QTableWidgetItem(self.ImagersLong[comboval])
            )

    def readoutTable(self):
        tableshort = dict()
        tablelong = dict()
        maxcolor = 15
        for i in range(0, maxcolor - 1):
            try:
                tableshort[i] = self.table.item(i, 2).text()
                if tableshort[i] == "":
                    tableshort[i] = "None"
            except AttributeError:
                tableshort[i] = "None"

            try:
                tablelong[i] = self.table.item(i, 3).text()
                if tablelong[i] == "":
                    tablelong[i] = "None"
            except AttributeError:
                tablelong[i] = "None"
        return tablelong, tableshort

    # get current date and time from the dialog
    def evalTable(self):
        conc = []
        time = []
        las = []
        bg = []
        bgstd = []
        for i in range(0, self.tifCounter):
            conc.append(float(self.table.item(i, 1).text()))
            time.append(float(self.table.item(i, 2).text()))
            las.append(float(self.table.item(i, 3).text()))
            bg.append(float(self.table.item(i, 4).text()))
            bgstd.append(float(self.table.item(i, 5).text()))

        # self.exportTable()
        return bg, bgstd, las, time, conc

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def setExt(parent=None):
        dialog = CalibrationDialog(parent)
        result = dialog.exec_()
        bg, bgstd, las, time, conc = dialog.evalTable()
        return (bg, bgstd, las, time, conc, result == QtWidgets.QDialog.Accepted)


def main():

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
        if p.name == "simulate":
            p.execute()

    window.show()
    sys.exit(app.exec_())

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        message = "".join(tback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(window, "An error occured", message)
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook


if __name__ == "__main__":
    main()
