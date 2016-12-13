"""
    picasso.simulate-gui
    ~~~~~~~~~~~~~~~~

    GUI for Simulate :
    Simulate single molcule fluorescence data

    :author: Maximilian Thomas Strauss, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, Max Planck Institute of Biochemistry
"""

import sys
from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .. import io as _io, simulate, lib
import numpy as _np
from scipy.stats import norm
from scipy.optimize import curve_fit
import glob as _glob
import os
from PyQt4.QtGui import QDialog, QVBoxLayout, QDialogButtonBox, QApplication
from PyQt4.QtCore import Qt
import time
import csv


def fitFuncBg(x, a, b):
    return (a + b * x[0]) * x[1] * x[2]


def fitFuncStd(x, a, b, c):
    return (a * x[0] * x[1] + b * x[2] + c)

plt.style.use('ggplot')

"DEFAULT PARAMETERS"

ADVANCEDMODE = 0    # 1 is with calibration of noise model
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
LASERPOWER_DEFAULT = 1.5    # POWER DENSITY
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
BGOFFSET_DEFAULT = 100
BGSTDOFFSET_DEFAULT = 0
# STRUCTURE
STRUCTURE1_DEFAULT = 3
STRUCTURE2_DEFAULT = 4
STRUCTURE3_DEFAULT = '20,20'
STRUCTUREYY_DEFAULT = '0,20,40,60,0,20,40,60,0,20,40,60'
STRUCTUREXX_DEFAULT = '0,20,40,0,20,40,0,20,40,0,20,40'
STRUCTUREEX_DEFAULT = '1,1,1,1,1,1,1,1,1,1,1,1'
STRUCTURENO_DEFAULT = 9
STRUCTUREFRAME_DEFAULT = 6
INCORPORATION_DEFAULT = 85


class Window(QtGui.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Picasso: Simulate')
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'icons', 'simulate.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.initUI()

    def initUI(self):

        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(5)

        # CAMERA PARAMETERS
        camera_groupbox = QtGui.QGroupBox('Camera parameters')
        cgrid = QtGui.QGridLayout(camera_groupbox)

        camerasize = QtGui.QLabel('Image size')
        integrationtime = QtGui.QLabel('Integration time')
        totaltime = QtGui.QLabel('Total acquisition time')
        frames = QtGui.QLabel('Frames')
        pixelsize = QtGui.QLabel('Pixelsize')

        self.camerasizeEdit = QtGui.QSpinBox()
        self.camerasizeEdit.setRange(1, 512)
        self.integrationtimeEdit = QtGui.QSpinBox()
        self.integrationtimeEdit.setRange(1, 10000)  # 1-10.000ms
        self.framesEdit = QtGui.QSpinBox()
        self.framesEdit.setRange(10, 1000000)  # 10-1.000.000 frames
        self.framesEdit.setSingleStep(1000)
        self.pixelsizeEdit = QtGui.QSpinBox()
        self.pixelsizeEdit.setRange(1, 1000)  # 1 to 1000 nm frame size
        self.totaltimeEdit = QtGui.QLabel()

        self.camerasizeEdit.setValue(IMAGESIZE_DEFAULT)
        self.integrationtimeEdit.setValue(ITIME_DEFAULT)
        self.framesEdit.setValue(FRAMES_DEFAULT)
        self.pixelsizeEdit.setValue(PIXELSIZE_DEFAULT)

        self.integrationtimeEdit.valueChanged.connect(self.changeTime)
        self.framesEdit.valueChanged.connect(self.changeTime)
        self.camerasizeEdit.valueChanged.connect(self.generatePositions)

        cgrid.addWidget(camerasize, 1, 0)
        cgrid.addWidget(self.camerasizeEdit, 1, 1)
        cgrid.addWidget(QtGui.QLabel('Px'), 1, 2)
        cgrid.addWidget(integrationtime, 2, 0)
        cgrid.addWidget(self.integrationtimeEdit, 2, 1)
        cgrid.addWidget(QtGui.QLabel('ms'), 2, 2)
        cgrid.addWidget(frames, 3, 0)
        cgrid.addWidget(self.framesEdit, 3, 1)
        cgrid.addWidget(totaltime, 4, 0)
        cgrid.addWidget(self.totaltimeEdit, 4, 1)
        cgrid.addWidget(QtGui.QLabel('min'), 4, 2)
        cgrid.addWidget(pixelsize, 5, 0)
        cgrid.addWidget(self.pixelsizeEdit, 5, 1)
        cgrid.addWidget(QtGui.QLabel('nm'), 5, 2)

        cgrid.addItem(QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding))

        # PAINT PARAMETERS
        paint_groupbox = QtGui.QGroupBox('PAINT parameters')
        pgrid = QtGui.QGridLayout(paint_groupbox)

        kon = QtGui.QLabel('k<sub>On</sub>')
        imagerconcentration = QtGui.QLabel('Imager concentration')
        taud = QtGui.QLabel('Dark time')
        taub = QtGui.QLabel('Bright time')

        self.konEdit = QtGui.QDoubleSpinBox()
        self.konEdit.setRange(1, 10000000)
        self.konEdit.setDecimals(0)
        self.konEdit.setSingleStep(100000)
        self.imagerconcentrationEdit = QtGui.QDoubleSpinBox()
        self.imagerconcentrationEdit.setRange(0.01, 1000)
        self.taudEdit = QtGui.QLabel()
        self.taubEdit = QtGui.QDoubleSpinBox()
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
        pgrid.addWidget(QtGui.QLabel('M<sup>−1</sup>s<sup>−1</sup>'), 1, 2)
        pgrid.addWidget(imagerconcentration, 2, 0)
        pgrid.addWidget(self.imagerconcentrationEdit, 2, 1)
        pgrid.addWidget(QtGui.QLabel('nM'), 2, 2)
        pgrid.addWidget(taud, 3, 0)
        pgrid.addWidget(self.taudEdit, 3, 1)
        pgrid.addWidget(QtGui.QLabel('ms'), 3, 2)
        pgrid.addWidget(taub, 4, 0)
        pgrid.addWidget(self.taubEdit, 4, 1)
        pgrid.addWidget(QtGui.QLabel('ms'), 4, 2)
        pgrid.addItem(QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding))

        # IMAGER Parameters
        imager_groupbox = QtGui.QGroupBox('Imager parameters')
        igrid = QtGui.QGridLayout(imager_groupbox)

        laserpower = QtGui.QLabel('Power density')
        if ADVANCEDMODE:
            laserpower = QtGui.QLabel('Laserpower')
        psf = QtGui.QLabel('PSF')
        psf_fwhm = QtGui.QLabel('PSF(FWHM)')
        photonrate = QtGui.QLabel('Photonrate')
        photonsframe = QtGui.QLabel('Photons (frame)')
        photonratestd = QtGui.QLabel('Photonrate Std')
        photonstdframe = QtGui.QLabel('Photons Std (frame)')
        photonbudget = QtGui.QLabel('Photonbudget')
        photonslope = QtGui.QLabel('Photon detection rate')
        photonslopeStd = QtGui.QLabel('Photonrate Std ')

        self.laserpowerEdit = QtGui.QDoubleSpinBox()
        self.laserpowerEdit.setRange(0, 10)
        self.laserpowerEdit.setSingleStep(0.1)
        self.psfEdit = QtGui.QDoubleSpinBox()
        self.psfEdit.setRange(0, 3)
        self.psfEdit.setSingleStep(0.01)
        self.psf_fwhmEdit = QtGui.QLabel()
        self.photonrateEdit = QtGui.QDoubleSpinBox()
        self.photonrateEdit.setRange(0, 1000)
        self.photonrateEdit.setDecimals(0)
        self.photonsframeEdit = QtGui.QLabel()
        self.photonratestdEdit = QtGui.QDoubleSpinBox()
        self.photonratestdEdit.setRange(0, 1000)
        self.photonratestdEdit.setDecimals(0)
        self.photonstdframeEdit = QtGui.QLabel()
        self.photonbudgetEdit = QtGui.QDoubleSpinBox()
        self.photonbudgetEdit.setRange(0, 100000000)
        self.photonbudgetEdit.setSingleStep(100000)
        self.photonbudgetEdit.setDecimals(0)

        self.photonslopeEdit = QtGui.QSpinBox()
        self.photonslopeStdEdit = QtGui.QDoubleSpinBox()

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

        igrid.addWidget(psf, 0, 0)
        igrid.addWidget(self.psfEdit, 0, 1)
        igrid.addWidget(QtGui.QLabel('Px'), 0, 2)
        igrid.addWidget(psf_fwhm, 1, 0)
        igrid.addWidget(self.psf_fwhmEdit, 1, 1)
        igrid.addWidget(QtGui.QLabel('nm'), 1, 2)

        igrid.addWidget(laserpower, 2, 0)
        igrid.addWidget(self.laserpowerEdit, 2, 1)
        igrid.addWidget(QtGui.QLabel('kW cm<sup>-2<sup>'), 2, 2)
        if ADVANCEDMODE:
            igrid.addWidget(QtGui.QLabel('mW'), 2, 2)

        igridindex = 1
        if ADVANCEDMODE:
            igrid.addWidget(photonrate, 3, 0)
            igrid.addWidget(self.photonrateEdit, 3, 1)
            igrid.addWidget(QtGui.QLabel('Photons ms<sup>-1<sup>'), 3, 2)

            igridindex = 0

        igrid.addWidget(photonsframe, 4 - igridindex, 0)
        igrid.addWidget(self.photonsframeEdit, 4 - igridindex, 1)
        igrid.addWidget(QtGui.QLabel('Photons'), 4 - igridindex, 2)
        igridindex = 2

        if ADVANCEDMODE:
            igrid.addWidget(photonratestd, 5, 0)
            igrid.addWidget(self.photonratestdEdit, 5, 1)
            igrid.addWidget(QtGui.QLabel('Photons ms<sup>-1<sup'), 5, 2)
            igridindex = 0

        igrid.addWidget(photonstdframe, 6 - igridindex, 0)
        igrid.addWidget(self.photonstdframeEdit, 6 - igridindex, 1)
        igrid.addWidget(QtGui.QLabel('Photons'), 6 - igridindex, 2)
        igrid.addWidget(photonbudget, 7 - igridindex, 0)
        igrid.addWidget(self.photonbudgetEdit, 7 - igridindex, 1)
        igrid.addWidget(QtGui.QLabel('Photons'), 7 - igridindex, 2)
        igrid.addWidget(photonslope, 8 - igridindex, 0)
        igrid.addWidget(self.photonslopeEdit, 8 - igridindex, 1)
        igrid.addWidget(QtGui.QLabel('Photons  ms<sup>-1</sup> kW<sup>-1</sup> cm<sup>2</sup>'), 8 - igridindex, 2)

        if ADVANCEDMODE:
            igrid.addWidget(photonslopeStd, 9 - igridindex, 0)
            igrid.addWidget(self.photonslopeStdEdit, 9 - igridindex, 1)
            igrid.addWidget(QtGui.QLabel('Photons  ms<sup>-1</sup> kW<sup>-1</sup> cm<sup>2</sup>'), 9 - igridindex, 2)

        if not ADVANCEDMODE:
            backgroundframesimple = QtGui.QLabel('Background (Frame)')
            self.backgroundframesimpleEdit = QtGui.QLabel()
            igrid.addWidget(backgroundframesimple, 10 - igridindex, 0)
            igrid.addWidget(self.backgroundframesimpleEdit, 10 - igridindex, 1)

        # NOISE MODEL
        noise_groupbox = QtGui.QGroupBox('Noise Model')
        ngrid = QtGui.QGridLayout(noise_groupbox)

        laserc = QtGui.QLabel('Lasercoefficient')
        imagerc = QtGui.QLabel('Imagercoefficient')

        EquationA = QtGui.QLabel('Equation A')
        EquationB = QtGui.QLabel('Equation B')
        EquationC = QtGui.QLabel('Equation C')

        Bgoffset = QtGui.QLabel('Background Offset')
        BgStdoffset = QtGui.QLabel('Background Std Offset')

        backgroundframe = QtGui.QLabel('Background (Frame)')
        noiseLabel = QtGui.QLabel('Noise (Frame)')

        self.lasercEdit = QtGui.QDoubleSpinBox()
        self.lasercEdit.setRange(0, 100000)
        self.lasercEdit.setDecimals(6)

        self.imagercEdit = QtGui.QDoubleSpinBox()
        self.imagercEdit.setRange(0, 100000)
        self.imagercEdit.setDecimals(6)

        self.EquationBEdit = QtGui.QDoubleSpinBox()
        self.EquationBEdit.setRange(-100000, 100000)
        self.EquationBEdit.setDecimals(6)

        self.EquationAEdit = QtGui.QDoubleSpinBox()
        self.EquationAEdit.setRange(-100000, 100000)
        self.EquationAEdit.setDecimals(6)

        self.EquationCEdit = QtGui.QDoubleSpinBox()
        self.EquationCEdit.setRange(-100000, 100000)
        self.EquationCEdit.setDecimals(6)

        self.lasercEdit.setValue(LASERC_DEFAULT)
        self.imagercEdit.setValue(IMAGERC_DEFAULT)

        self.EquationAEdit.setValue(EQA_DEFAULT)
        self.EquationBEdit.setValue(EQB_DEFAULT)
        self.EquationCEdit.setValue(EQC_DEFAULT)

        self.BgoffsetEdit = QtGui.QDoubleSpinBox()
        self.BgoffsetEdit.setRange(-100000, 100000)
        self.BgoffsetEdit.setDecimals(6)

        self.BgStdoffsetEdit = QtGui.QDoubleSpinBox()
        self.BgStdoffsetEdit.setRange(-100000, 100000)
        self.BgStdoffsetEdit.setDecimals(6)

        self.lasercEdit.valueChanged.connect(self.changeNoise)
        self.imagercEdit.valueChanged.connect(self.changeNoise)
        self.EquationAEdit.valueChanged.connect(self.changeNoise)
        self.EquationBEdit.valueChanged.connect(self.changeNoise)
        self.EquationCEdit.valueChanged.connect(self.changeNoise)

        backgroundframe = QtGui.QLabel('Background (Frame)')
        noiseLabel = QtGui.QLabel('Noise (Frame)')

        self.backgroundframeEdit = QtGui.QLabel()
        self.noiseEdit = QtGui.QLabel()

        ngrid.addWidget(laserc, 0, 0)
        ngrid.addWidget(self.lasercEdit, 0, 1)
        ngrid.addWidget(imagerc, 1, 0)
        ngrid.addWidget(self.imagercEdit, 1, 1)
        ngrid.addWidget(EquationA, 2, 0)
        ngrid.addWidget(self.EquationAEdit, 2, 1)
        ngrid.addWidget(EquationB, 3, 0)
        ngrid.addWidget(self.EquationBEdit, 3, 1)
        ngrid.addWidget(EquationC, 4, 0)
        ngrid.addWidget(self.EquationCEdit, 4, 1)
        ngrid.addWidget(Bgoffset, 5, 0)
        ngrid.addWidget(self.BgoffsetEdit, 5, 1)
        ngrid.addWidget(BgStdoffset, 6, 0)
        ngrid.addWidget(self.BgStdoffsetEdit, 6, 1)
        ngrid.addWidget(backgroundframe, 7, 0)
        ngrid.addWidget(self.backgroundframeEdit, 7, 1)
        ngrid.addWidget(noiseLabel, 8, 0)
        ngrid.addWidget(self.noiseEdit, 8, 1)

        calibrateNoiseButton = QtGui.QPushButton("Calibrate Noise Model")
        calibrateNoiseButton.clicked.connect(self.calibrateNoise)
        importButton = QtGui.QPushButton("Import from Experiment (hdf5)")
        importButton.clicked.connect(self.importhdf5)

        ngrid.addWidget(calibrateNoiseButton, 10, 0, 1, 3)
        ngrid.addWidget(importButton, 11, 0, 1, 3)

        # STRUCTURE DEFINITIONS
        structure_groupbox = QtGui.QGroupBox('Structure')
        sgrid = QtGui.QGridLayout(structure_groupbox)

        structureno = QtGui.QLabel('Number of structures')
        structureframe = QtGui.QLabel('Frame')

        self.structure1 = QtGui.QLabel('Columns')
        self.structure2 = QtGui.QLabel('Rows')
        self.structure3 = QtGui.QLabel('Spacing X,Y')
        self.structure3Label = QtGui.QLabel('nm')

        structurexx = QtGui.QLabel('Stucture X')
        structureyy = QtGui.QLabel('Structure Y')
        structureex = QtGui.QLabel('Exchange labels')

        structurecomboLabel = QtGui.QLabel('Type')

        structureIncorporation = QtGui.QLabel('Incorporation')

        self.structurenoEdit = QtGui.QSpinBox()
        self.structurenoEdit.setRange(1, 1000)
        self.structureframeEdit = QtGui.QSpinBox()
        self.structureframeEdit.setRange(4, 16)
        self.structurexxEdit = QtGui.QLineEdit(STRUCTUREXX_DEFAULT)
        self.structureyyEdit = QtGui.QLineEdit(STRUCTUREYY_DEFAULT)
        self.structureexEdit = QtGui.QLineEdit(STRUCTUREEX_DEFAULT)
        self.structureIncorporationEdit = QtGui.QDoubleSpinBox()
        self.structureIncorporationEdit.setRange(1, 100)
        self.structureIncorporationEdit.setDecimals(0)
        self.structureIncorporationEdit.setValue(INCORPORATION_DEFAULT)

        self.structurecombo = QtGui.QComboBox()
        self.structurecombo.addItem("Grid")
        self.structurecombo.addItem("Circle")
        self.structurecombo.addItem("Custom")

        self.structure1Edit = QtGui.QSpinBox()
        self.structure1Edit.setRange(1, 1000)
        self.structure1Edit.setValue(STRUCTURE1_DEFAULT)
        self.structure2Edit = QtGui.QSpinBox()
        self.structure2Edit.setRange(1, 1000)
        self.structure2Edit.setValue(STRUCTURE2_DEFAULT)
        self.structure3Edit = QtGui.QLineEdit(STRUCTURE3_DEFAULT)

        self.structure1Edit.valueChanged.connect(self.changeStructDefinition)
        self.structure2Edit.valueChanged.connect(self.changeStructDefinition)
        self.structure3Edit.textEdited.connect(self.changeStructDefinition)

        self.structurenoEdit.setValue(STRUCTURENO_DEFAULT)
        self.structureframeEdit.setValue(STRUCTUREFRAME_DEFAULT)

        self.structurexxEdit.textChanged.connect(self.generatePositions)
        self.structureyyEdit.textChanged.connect(self.generatePositions)
        self.structureexEdit.textChanged.connect(self.generatePositions)

        self.structurenoEdit.valueChanged.connect(self.generatePositions)
        self.structureframeEdit.valueChanged.connect(self.generatePositions)

        self.structurerandomOrientationEdit = QtGui.QCheckBox()
        self.structurerandomEdit = QtGui.QCheckBox()

        structurerandom = QtGui.QLabel('Random arrangement')
        structurerandomOrientation = QtGui.QLabel('Random orientation')

        self.structurerandomEdit.stateChanged.connect(self.generatePositions)
        self.structurerandomOrientationEdit.stateChanged.connect(self.generatePositions)
        self.structureIncorporationEdit.valueChanged.connect(self.generatePositions)

        self.structurecombo.currentIndexChanged.connect(self.changeStructureType)

        sgrid.addWidget(structureno, 1, 0)
        sgrid.addWidget(self.structurenoEdit, 1, 1)
        sgrid.addWidget(structureframe, 2, 0)
        sgrid.addWidget(self.structureframeEdit, 2, 1)
        sgrid.addWidget(QtGui.QLabel('Px'), 2, 2)
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
        sgrid.addWidget(QtGui.QLabel('nm'), 7, 2)
        sgrid.addWidget(structureyy, 8, 0)
        sgrid.addWidget(self.structureyyEdit, 8, 1)
        sgrid.addWidget(QtGui.QLabel('nm'), 8, 2)
        sgrid.addWidget(structureex, 9, 0)
        sgrid.addWidget(self.structureexEdit, 9, 1)
        sgrid.addWidget(structureIncorporation, 10, 0)
        sgrid.addWidget(self.structureIncorporationEdit, 10, 1)
        sgrid.addWidget(QtGui.QLabel('%'), 10, 2)
        sgrid.addWidget(structurerandom, 11, 1)
        sgrid.addWidget(self.structurerandomEdit, 11, 0)
        sgrid.addWidget(structurerandomOrientation, 12, 1)
        sgrid.addWidget(self.structurerandomOrientationEdit, 12, 0)

        importDesignButton = QtGui.QPushButton("Import structure from design")
        importDesignButton.clicked.connect(self.importDesign)
        sgrid.addWidget(importDesignButton, 13, 0, 1, 3)

        generateButton = QtGui.QPushButton("Generate positions")
        generateButton.clicked.connect(self.generatePositions)
        sgrid.addWidget(generateButton, 14, 0, 1, 3)
        cgrid.addItem(QtGui.QSpacerItem(1, 1, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding))

        simulateButton = QtGui.QPushButton("Simulate data")
        self.exchangeroundsEdit = QtGui.QLineEdit('1')

        quitButton = QtGui.QPushButton('Quit', self)
        quitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        quitButton.resize(quitButton.sizeHint())

        loadButton = QtGui.QPushButton("Load settings from previous simulation")

        btngridR = QtGui.QGridLayout()

        btngridR.addWidget(loadButton)

        btngridR.addWidget(QtGui.QLabel('Exchange rounds to be simulated:'))
        btngridR.addWidget(self.exchangeroundsEdit)
        btngridR.addWidget(simulateButton)
        btngridR.addWidget(quitButton)

        simulateButton.clicked.connect(self.saveDialog)

        loadButton.clicked.connect(self.loadSettings)

        self.show()
        self.changeTime()
        self.changePSF()
        self.changeNoise()
        self.changePaint()

        pos_groupbox = QtGui.QGroupBox('Positions [Px]')
        str_groupbox = QtGui.QGroupBox('Structure [nm]')

        posgrid = QtGui.QGridLayout(pos_groupbox)
        strgrid = QtGui.QGridLayout(str_groupbox)

        self.figure1 = plt.figure()
        self.figure2 = plt.figure()
        self.canvas1 = FigureCanvas(self.figure1)
        csize = 280
        self.canvas1.setMinimumSize(csize, csize)

        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.setMinimumSize(csize, csize)

        posgrid.addWidget(self.canvas1)
        strgrid.addWidget(self.canvas2)

        self.mainpbar = QtGui.QProgressBar(self)
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
            self.grid.addWidget(self.mainpbar, 4, 0, 1, 4)
        else:
            self.grid.addWidget(pos_groupbox, 1, 0)
            self.grid.addWidget(str_groupbox, 1, 1)
            self.grid.addWidget(structure_groupbox, 2, 0, 2, 1)
            self.grid.addWidget(camera_groupbox, 3, 1)
            self.grid.addWidget(paint_groupbox, 4, 0)
            self.grid.addWidget(imager_groupbox, 2, 1)
            self.grid.addLayout(btngridR, 4, 1)
            self.grid.addWidget(self.mainpbar, 5, 0, 1, 4)
        mainWidget = QtGui.QWidget()
        mainWidget.setLayout(self.grid)
        self.setCentralWidget(mainWidget)
        self.setGeometry(300, 300, 300, 150)
        # CALL FUNCTIONS
        self.generatePositions()

        self.mainpbar.setValue(0)
        self.statusBar().showMessage('Simulate ready.')

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
        if ADVANCEDMODE:
            # NEW NOISE MODEL
            laserc = self.lasercEdit.value()
            imagerc = self.imagercEdit.value()
            bgoffset = self.BgoffsetEdit.value()
            bgmodel = (laserc + imagerc * imagerconcentration) * laserpower * itime + bgoffset
            equationA = self.EquationAEdit.value()
            equationB = self.EquationBEdit.value()
            equationC = self.EquationCEdit.value()
            bgstdoffset = self.BgStdoffsetEdit.value()
            bgmodelstd = equationA * laserpower * itime + equationB * bgmodel + equationC + bgstdoffset
            self.backgroundframeEdit.setText(str(int(bgmodel)))
            self.noiseEdit.setText(str(int(bgmodelstd)))
        else:
            bgmodel = (LASERC_DEFAULT + IMAGERC_DEFAULT * imagerconcentration) * laserpower * itime + BGOFFSET_DEFAULT #100 for NP
            self.backgroundframesimpleEdit.setText(str(int(bgmodel)))

    def changeStructureType(self):
        typeindex = self.structurecombo.currentIndex()
        # TYPEINDEX: 0 = GRID, 1 = CIRCLE, 2 = CUSTOM

        if typeindex == 0:
            self.structure1.show()
            self.structure2.show()
            self.structure3.show()
            self.structure1Edit.show()
            self.structure2Edit.show()
            self.structure3Edit.show()
            self.structure3Label.show()
            self.structure1.setText('Columns')
            self.structure2.setText('Rows')
            self.structure3.setText('Spacing X,Y')
            self.structure1Edit.setValue(3)
            self.structure2Edit.setValue(4)
            self.structure3Edit.setText('20,20')
        elif typeindex == 1:
            self.structure1.show()
            self.structure2.show()
            self.structure3.show()
            self.structure1Edit.show()
            self.structure2Edit.show()
            self.structure3Edit.show()
            self.structure3Label.show()
            self.structure1.hide()
            self.structure2.setText('Number of Labels')
            self.structure3.setText('Diameter')
            self.structure1Edit.hide()
            self.structure2Edit.setValue(12)
            self.structure3Edit.setText('100')
        elif typeindex == 2:
            self.structure1.hide()
            self.structure2.hide()
            self.structure3.hide()
            self.structure1Edit.hide()
            self.structure2Edit.hide()
            self.structure3Edit.hide()
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

            structurexx = ''
            structureyy = ''
            structureex = ''

            for i in range(0, rows):
                for j in range(0, cols):
                    structurexx = structurexx + str(i * spacingx) + ','
                    structureyy = structureyy + str(j * spacingy) + ','
                    structureex = structureex + '1,'

            structurexx = structurexx[:-1]
            structureyy = structureyy[:-1]
            structureex = structureex[:-1]

            self.structurexxEdit.setText(structurexx)
            self.structureyyEdit.setText(structureyy)
            self.structureexEdit.setText(structureex)
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

            xxval = _np.round(_np.cos(circdata) * diameter/2 * 100) / 100
            yyval = _np.round(_np.sin(circdata) * diameter/2 * 100) / 100

            structurexx = ''
            structureyy = ''
            structureex = ''

            for i in range(0, xxval.size):
                structurexx = structurexx + str(xxval[i]) + ','
                structureyy = structureyy + str(yyval[i]) + ','
                structureex = structureex + '1,'

            structurexx = structurexx[:-1]
            structureyy = structureyy[:-1]
            structureex = structureex[:-1]

            self.structurexxEdit.setText(structurexx)
            self.structureyyEdit.setText(structureyy)
            self.structureexEdit.setText(structureex)
            self.generatePositions()

        elif typeindex == 2:  # Custom

            self.generatePositions()

    def keyPressEvent(self, e):

        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    def vectorToString(self, x):
        x_arrstr = _np.char.mod('%f', x)
        x_str = ",".join(x_arrstr)
        return x_str

    def simulate(self, fileNameOld):

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

        structurexx, structureyy, structureex = self.readStructure()

        self.statusBar().showMessage('Simulation started')
        struct = self.newstruct
        handlex = self.vectorToString(struct[0, :])
        handley = self.vectorToString(struct[1, :])
        handleex = self.vectorToString(struct[2, :])
        handless = self.vectorToString(struct[3, :])

        exchangeroundstoSim = _np.asarray((self.exchangeroundsEdit.text()).split(","))
        exchangeroundstoSim = exchangeroundstoSim.astype(_np.int)

        noexchangecolors = len(set(exchangeroundstoSim))
        exchangecolors = list(set(exchangeroundstoSim))

        t0 = time.time()

        for i in range(0, noexchangecolors):
            if noexchangecolors > 1:
                fileName = _io.multiple_filenames(fileNameOld, i)
                partstruct = struct[:, struct[2, :] == exchangecolors[i]]

            else:
                fileName = fileNameOld
                partstruct = struct[:, struct[2, :] == exchangecolors[0]]

            self.statusBar().showMessage('Distributing photons ...')

            bindingsitesx = partstruct[0, :]
            nosites = len(bindingsitesx)  # number of binding sites in image
            photondist = _np.zeros((nosites, frames), dtype=_np.int)

            for i in range(0, nosites):
                photondisttemp = simulate.distphotons(partstruct, itime, frames, taud, taub, photonrate, photonratestd, photonbudget)

                photondist[i, :] = photondisttemp
                outputmsg = 'Distributing photons ... ' + str(_np.round(i / nosites * 1000) / 10) + ' %'
                self.statusBar().showMessage(outputmsg)
                self.mainpbar.setValue(_np.round(i / nosites * 1000) / 10)

            self.statusBar().showMessage('Converting to image ... ')

            movie = _np.zeros(shape=(frames, imagesize, imagesize), dtype='<u2')
            app = QtCore.QCoreApplication.instance()
            for runner in range(0, frames):
                movie[runner, :, :] = simulate.convertMovie(runner, photondist, partstruct, imagesize, frames, psf, photonrate, background, noise)
                outputmsg = 'Converting to Image ... ' + str(_np.round(runner / frames * 1000) / 10) + ' %'

                self.statusBar().showMessage(outputmsg)
                self.mainpbar.setValue(_np.round(runner / frames * 1000) / 10)
                app.processEvents()
            self.statusBar().showMessage('Converting to image ... complete.')
            self.statusBar().showMessage('Saving movie ...')

            info = {'Generated by': 'Picasso simulate',
                    'Byte Order': '<',
                    'Camera': 'Simulation',
                    'Data Type': movie.dtype.name,
                    'Frames': frames,
                    'Structure.Frame': structureFrame,
                    'Structure.Number': structureNo,
                    'Structure.StructureX': structurex,
                    'Structure.StructureY': structurey,
                    'Structure.StructureEx': structureextxt,
                    'Structure.HandleX': handlex,
                    'Structure.HandleY': handley,
                    'Structure.HandleEx': handleex,
                    'Structure.HandleStruct': handless,
                    'Structure.Incorporation': structureIncorporation,
                    'Structure.Arrangement': structureArrangement,
                    'Structure.Orientation': structureOrientation,
                    'PAINT.k_on': kon,
                    'PAINT.imager': imagerconcentration,
                    'PAINT.taub': taub,
                    'Imager.PSF': psf,
                    'Imager.Photonrate': photonrate,
                    'Imager.Photonrate Std': photonratestd,
                    'Imager.Photonbudget': photonbudget,
                    'Imager.Laserpower': laserpower,
                    'Imager.Photonslope': photonslope,
                    'Imager.PhotonslopeStd': photonslopeStd,
                    'Camera.Image Size': imagesize,
                    'Camera.Integration Time': itime,
                    'Camera.Frames': frames,
                    'Camera.Pixelsize': pixelsize,
                    'Noise.Lasercoefficient': laserc,
                    'Noise.Imagercoefficient': imagerc,
                    'Noise.EquationA': equationA,
                    'Noise.EquationB': equationB,
                    'Noise.EquationC': equationC,
                    'Noise.BackgroundOff': bgoffset,
                    'Noise.BackgroundStdOff': bgstdoffset,
                    'Height': imagesize,
                    'Width': imagesize}

            simulate.saveMovie(fileName, movie, info)
            self.statusBar().showMessage('Movie saved to: ' + fileName)
        dt = time.time() - t0
        self.statusBar().showMessage('All computations finished. Last file saved to: ' + fileName + '. Time elapsed: {:.2f} Seconds.'.format(dt))

    def loadSettings(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open yaml', filter='*.yaml')
        if path:
            info = _io.load_info(path)
            self.framesEdit.setValue(info[0]['Frames'])

            self.structureframeEdit.setValue(info[0]['Structure.Frame'])
            self.structurenoEdit.setValue(info[0]['Structure.Number'])
            self.structurexxEdit.setText(info[0]['Structure.StructureX'])
            self.structureyyEdit.setText(info[0]['Structure.StructureY'])
            self.structureexEdit.setText(info[0]['Structure.StructureEx'])
            self.structureIncorporationEdit.setValue(info[0]['Structure.Incorporation'])

            self.structurerandomEdit.setCheckState(info[0]['Structure.Arrangement'])
            self.structurerandomOrientationEdit.setCheckState(info[0]['Structure.Orientation'])

            self.konEdit.setValue(info[0]['PAINT.k_on'])
            self.imagerconcentrationEdit.setValue(info[0]['PAINT.imager'])
            self.taubEdit.setValue(info[0]['PAINT.taub'])

            self.psfEdit.setValue(info[0]['Imager.PSF'])
            self.photonrateEdit.setValue(info[0]['Imager.Photonrate'])
            self.photonratestdEdit.setValue(info[0]['Imager.Photonrate Std'])
            self.photonbudgetEdit.setValue(info[0]['Imager.Photonbudget'])
            self.laserpowerEdit.setValue(info[0]['Imager.Laserpower'])
            self.photonslopeEdit.setValue(info[0]['Imager.Photonslope'])
            self.photonslopeStdEdit.setValue(info[0]['Imager.PhotonslopeStd'])

            self.camerasizeEdit.setValue(info[0]['Camera.Image Size'])
            self.integrationtimeEdit.setValue(info[0]['Camera.Integration Time'])
            self.framesEdit.setValue(info[0]['Camera.Frames'])
            self.pixelsizeEdit.setValue(info[0]['Camera.Pixelsize'])

            if ADVANCEDMODE:
                self.lasercEdit.setValue(info[0]['Noise.Lasercoefficient'])
                self.imagercEdit.setValue(info[0]['Noise.Imagercoefficient'])
                self.BgoffsetEdit.setValue(info[0]['Noise.BackgroundOff'])

                self.EquationAEdit.setValue(info[0]['Noise.EquationA'])
                self.EquationBEdit.setValue(info[0]['Noise.EquationB'])
                self.EquationCEdit.setValue(info[0]['Noise.EquationC'])
                self.BgStdoffsetEdit.setValue(info[0]['Noise.BackgroundStdOff'])

            # SET POSITIONS
            handlexx = _np.asarray((info[0]['Structure.HandleX']).split(","))
            handleyy = _np.asarray((info[0]['Structure.HandleY']).split(","))
            handleex = _np.asarray((info[0]['Structure.HandleEx']).split(","))
            handless = _np.asarray((info[0]['Structure.HandleStruct']).split(","))

            handlexx = handlexx.astype(_np.float)
            handleyy = handleyy.astype(_np.float)
            handleex = handleex.astype(_np.float)
            handless = handless.astype(_np.float)

            handleex = handleex.astype(_np.int)
            handless = handless.astype(_np.int)

            structure = _np.array([handlexx, handleyy, handleex, handless])

            self.structurecombo.setCurrentIndex(2)
            self.newstruct = structure
            self.plotPositions()
            self.statusBar().showMessage('Settings loaded from: ' + path)

    def importDesign(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open yaml', filter='*.yaml')
        if path:
            info = _io.load_info(path)
            self.structurexxEdit.setText(info[0]['Structure.StructureX'])
            self.structureyyEdit.setText(info[0]['Structure.StructureY'])
            self.structureexEdit.setText(info[0]['Structure.StructureEx'])
            self.structurecombo.setCurrentIndex(2)

    def readStructure(self):
        structurexxtxt = _np.asarray((self.structurexxEdit.text()).split(","))
        structureyytxt = _np.asarray((self.structureyyEdit.text()).split(","))
        structureextxt = _np.asarray((self.structureexEdit.text()).split(","))

        structurexx = []
        structureyy = []
        structureex = []

        for element in structurexxtxt:
            try:
                structurexx.append(float(element))
            except ValueError:
                pass
        for element in structureyytxt:
            try:
                structureyy.append(float(element))
            except ValueError:
                pass
        for element in structureextxt:
            try:
                structureex.append(int(element))
            except ValueError:
                pass

        minlen = min(len(structureex), len(structurexx), len(structureyy))

        structurexx = structurexx[0:minlen]
        structureyy = structureyy[0:minlen]
        structureex = structureex[0:minlen]

        return structurexx, structureyy, structureex

    def plotStructure(self):

        structurexx, structureyy, structureex = self.readStructure()
        noexchangecolors = len(set(structureex))
        exchangecolors = list(set(structureex))
        # self.figure2.suptitle('Structure [nm]')
        ax1 = self.figure2.add_subplot(111)
        ax1.cla()
        ax1.hold(True)
        ax1.axis('equal')

        for i in range(0, noexchangecolors):
            plotxx = []
            plotyy = []
            for j in range(0, len(structureex)):
                if structureex[j] == exchangecolors[i]:
                    plotxx.append(structurexx[j])
                    plotyy.append(structureyy[j])
            ax1.plot(plotxx, plotyy, 'o')

        distx = round(1 / 10 * (max(structurexx) - min(structurexx)))
        disty = round(1 / 10 * (max(structureyy) - min(structureyy)))

        ax1.axes.set_xlim((min(structurexx) - distx, max(structurexx) + distx))
        ax1.axes.set_ylim((min(structureyy) - disty, max(structureyy) + disty))
        self.canvas2.draw()

        exchangecolorsList = ','.join(map(str, exchangecolors))
        # UPDATE THE EXCHANGE COLORS IN BUTTON TO BE simulated
        self.exchangeroundsEdit.setText(str(exchangecolorsList))

    def generatePositions(self):
        self.plotStructure()
        structurexx, structureyy, structureex = self.readStructure()
        pixelsize = self.pixelsizeEdit.value()
        structure = simulate.defineStructure(structurexx, structureyy, structureex, pixelsize)

        number = self.structurenoEdit.value()
        imageSize = self.camerasizeEdit.value()
        frame = self.structureframeEdit.value()
        arrangement = int(self.structurerandomEdit.checkState())
        gridpos = simulate.generatePositions(number, imageSize, frame, arrangement)

        orientation = int(self.structurerandomOrientationEdit.checkState())
        incorporation = self.structureIncorporationEdit.value() / 100
        exchange = 0
        self.newstruct = simulate.prepareStructures(structure, gridpos, orientation, number, incorporation, exchange)

        # self.figure1.suptitle('Positions [Px]')
        ax1 = self.figure1.add_subplot(111)
        ax1.cla()
        ax1.hold(True)
        ax1.axis('equal')
        ax1.plot(self.newstruct[0, :], self.newstruct[1, :], '+')
        # PLOT FRAME
        ax1.add_patch(
            patches.Rectangle(
                (frame, frame),
                imageSize - 2 * frame,
                imageSize - 2 * frame,
                linestyle='dashed',
                edgecolor="#000000",
                fill=False      # remove background
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
        #self.figure2.suptitle('Structure [nm]')
        ax1 = self.figure2.add_subplot(111)
        ax1.cla()
        ax1.hold(True)
        ax1.axis('equal')

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
            ax1.plot(plotxx, plotyy, 'o')

            distx = round(1 / 10 * (max(structurexx_nm) - min(structurexx_nm)))
            disty = round(1 / 10 * (max(structureyy_nm) - min(structureyy_nm)))

            ax1.axes.set_xlim((min(structurexx_nm) - distx, max(structurexx_nm) + distx))
            ax1.axes.set_ylim((min(structureyy_nm) - disty, max(structureyy_nm) + disty))
        self.canvas2.draw()

    def plotPositions(self):
        structurexx, structureyy, structureex = self.readStructure()
        pixelsize = self.pixelsizeEdit.value()
        structure = simulate.defineStructure(structurexx, structureyy, structureex, pixelsize)

        number = self.structurenoEdit.value()
        imageSize = self.camerasizeEdit.value()
        frame = self.structureframeEdit.value()
        arrangement = int(self.structurerandomEdit.checkState())


        # self.figure1.suptitle('Positions [Px]')
        ax1 = self.figure1.add_subplot(111)
        ax1.cla()
        ax1.hold(True)
        ax1.axis('equal')
        ax1.plot(self.newstruct[0, :], self.newstruct[1, :], '+')
        # PLOT FRAME
        ax1.add_patch(
            patches.Rectangle(
                (frame, frame),
                imageSize - 2 * frame,
                imageSize - 2 * frame,
                linestyle='dashed',
                edgecolor="#000000",
                fill=False      # remove background
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
        ax1.hold(True)

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
            ax1.plot(plotxx, plotyy, 'o')

            distx = round(1 / 10 * (max(structurexx_nm) - min(structurexx_nm)))
            disty = round(1 / 10 * (max(structureyy_nm) - min(structureyy_nm)))

            ax1.axes.set_xlim((min(structurexx_nm) - distx, max(structurexx_nm) + distx))
            ax1.axes.set_ylim((min(structureyy_nm) - disty, max(structureyy_nm) + disty))
        self.canvas2.draw()

    def openDialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open design', filter='*.yaml')
        if path:
            self.mainscene.loadCanvas(path)
            print(path)
            self.statusBar().showMessage('File loaded from: ' + path)

    def saveDialog(self):
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save movie to..', filter='*.raw')
        if path:
            self.simulate(path)

    def importhdf5(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open localizations', filter='*.hdf5')
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
        print(' fit coefficients :\n', fitParamsBg)

        # SET VALUES TO PARAMETER
        self.lasercEdit.setValue(fitParamsBg[0])
        self.imagercEdit.setValue(fitParamsBg[1])

        x_3dStd = _np.array([las, time, bg])
        p0S = [1, 1, 1]
        fitParamsStd, fitCovariances = curve_fit(fitFuncStd, x_3dStd, bgstd, p0S)

        print(' fit coefficients2:\n', fitParamsStd)

        self.EquationAEdit.setValue(fitParamsStd[0])
        self.EquationBEdit.setValue(fitParamsStd[1])
        self.EquationCEdit.setValue(fitParamsStd[2])

        # Noise model working point

        figure4 = plt.figure()

        # Background
        bgmodel = fitFuncBg(x_3d, fitParamsBg[0], fitParamsBg[1])
        ax1 = figure4.add_subplot(121)
        ax1.cla()
        ax1.plot(bg, bgmodel, 'o')
        x = _np.linspace(*ax1.get_xlim())
        ax1.plot(x, x)
        title = "Background Model:"
        ax1.set_title(title)

        # Std
        bgmodelstd = fitFuncStd(x_3dStd, fitParamsStd[0], fitParamsStd[1], fitParamsStd[2])
        ax2 = figure4.add_subplot(122)
        ax2.cla()
        ax2.plot(bgstd, bgmodelstd, 'o')
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
        integrationtime, ok1 = QtGui.QInputDialog.getText(self, 'Input Dialog',
                                                          'Enter integration time in ms:')
        integrationtime = int(integrationtime)
        if ok1:
            imagerconcentration, ok2 = QtGui.QInputDialog.getText(self, 'Input Dialog',
                                                                  'Enter imager concentration in nM:')
            imagerconcentration = float(imagerconcentration)

            if ok2:
                laserpower, ok3 = QtGui.QInputDialog.getText(self, 'Input Dialog',
                                                             'Enter Laserpower in mW:')
                laserpower = float(laserpower)
                if ok3:
                    cbaseline, ok4 = QtGui.QInputDialog.getText(self, 'Input Dialog',
                                                                'Enter camera baseline')
                    cbaseline = float(cbaseline)
                    # self.le.setText(str(text))

                    photons = locs['photons']
                    sigmax = locs['sx']
                    sigmay = locs['sy']
                    bg = locs['bg']
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
                    ax1.hold(True)
                    ax1.hist(photons, bins=25, normed=True, alpha=0.6)
                    xmin, xmax = plt.xlim()
                    x = _np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, photonsmu, photonsstd)
                    ax1.plot(x, p)
                    title = "Photons:\n mu = %.2f\n  std = %.2f" % (photonsmu, photonsstd)
                    ax1.set_title(title)

                    # Sigma X & Sigma Y
                    sigma = _np.concatenate((sigmax, sigmay), axis=0)
                    sigmamu, sigmastd = norm.fit(sigma)
                    ax2 = figure3.add_subplot(132)
                    ax2.cla()
                    ax2.hold(True)
                    ax2.hist(sigma, bins=25, normed=True, alpha=0.6)
                    xmin, xmax = plt.xlim()
                    x = _np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, sigmamu, sigmastd)
                    ax2.plot(x, p,)
                    title = "PSF:\n mu = %.2f\n  std = %.2f" % (sigmamu, sigmastd)
                    ax2.set_title(title)

                    # Background
                    bgmu, bgstd = norm.fit(bg)
                    ax3 = figure3.add_subplot(133)
                    ax3.cla()
                    ax3.hold(True)
                    # Plot the histogram.
                    ax3.hist(bg, bins=25, normed=True, alpha=0.6)
                    xmin, xmax = plt.xlim()
                    x = _np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, bgmu, bgstd)
                    ax3.plot(x, p)
                    title = "Background:\n mu = %.2f\n  std = %.2f" % (bgmu, bgstd)
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

                    bgmodel = (laserc + imagerc * imagerconcentration) * laserpower * integrationtime

                    equationA = self.EquationAEdit.value()
                    equationB = self.EquationBEdit.value()
                    equationC = self.EquationCEdit.value()

                    bgmodelstd = equationA * laserpower * integrationtime + equationB * bgmu + equationC

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


class CalibrationDialog(QtGui.QDialog):

    def __init__(self, parent=None):
        super(CalibrationDialog, self).__init__(parent)

        layout = QtGui.QVBoxLayout(self)

        self.table = QtGui.QTableWidget()
        self.table.setWindowTitle('Noise Model Calibration')
        self.setWindowTitle('Noise Model Calibration')
        self.resize(800, 400)

        layout.addWidget(self.table)

        # ADD BUTTONS:
        self.loadTifButton = QtGui.QPushButton("Load Tifs")
        layout.addWidget(self.loadTifButton)

        self.evalTifButton = QtGui.QPushButton("Evaluate Tifs")
        layout.addWidget(self.evalTifButton)

        self.pbar = QtGui.QProgressBar(self)
        layout.addWidget(self.pbar)

        self.loadTifButton.clicked.connect(self.loadTif)
        self.evalTifButton.clicked.connect(self.evalTif)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.ActionRole | QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def exportTable(self):

        table = dict()
        tablecontent = []
        print(self.table.rowCount())
        tablecontent.append(['FileName', 'Imager concentration[nM]', 'Integration time [ms]', 'Laserpower', 'Mean [Photons]', 'Std [Photons]'])
        for row in range(self.table.rowCount()):
            rowdata = []
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                if item is not None:
                    rowdata.append(item.text())
                else:
                    rowdata.append('')
            tablecontent.append(rowdata)

        table[0] = tablecontent
        print(tablecontent)
        print(table)
        path = QtGui.QFileDialog.getSaveFileName(self,  'Export calibration table to.',  filter='*.csv')
        if path:
            self.savePlate(path, table)

    def savePlate(self, filename, data):
        with open(filename,  'w',  newline='') as csvfile:
            Writer = csv.writer(csvfile,  delimiter=',',
                                quotechar='|',  quoting=csv.QUOTE_MINIMAL)
            for j in range(0, len(data)):
                exportdata = data[j]
                for i in range(0, len(exportdata)):
                    Writer.writerow(exportdata[i])

    def evalTif(self):

        baseline, ok1 = QtGui.QInputDialog.getText(self, 'Input Dialog',
                                                   'Enter Camera Baseline:')
        if ok1:
            baseline = int(baseline)
        else:
            baseline = 200  # default

        sensitvity, ok2 = QtGui.QInputDialog.getText(self, 'Input Dialog',
                                                     'Enter Camera Sensitivity:')
        if ok2:
            sensitvity = float(sensitvity)
        else:
            sensitvity = 1.47

        print(baseline)
        print(sensitvity)

        counter = 0
        for element in self.tifFiles:
            counter = counter + 1
            self.pbar.setValue((counter - 1) / self.tifCounter * 100)
            print('Current Dataset: ' + str(counter) + ' of ' + str(self.tifCounter))
            QtGui.qApp.processEvents()
            movie, info = _io.load_movie(element)

            movie = movie[0:100, :, :]

            movie = (movie - baseline) * sensitvity
            self.table.setItem(counter - 1, 4, QtGui.QTableWidgetItem(str((_np.mean(movie)))))
            self.table.setItem(counter - 1, 5, QtGui.QTableWidgetItem(str((_np.std(movie)))))

            self.table.setItem(counter - 1, 1, QtGui.QTableWidgetItem(str((self.ValueFind(element, 'nM_')))))
            self.table.setItem(counter - 1, 2, QtGui.QTableWidgetItem(str((self.ValueFind(element, 'ms_')))))
            self.table.setItem(counter - 1, 3, QtGui.QTableWidgetItem(str((self.ValueFind(element, 'mW_')))))

            print(_np.mean(movie))
            print(_np.std(movie))
        self.pbar.setValue(100)

    def ValueFind(self, filename, unit):
        index = filename.index(unit)

        value = 0

        for i in range(4):
            try:
                value += int(filename[index - 1 - i]) * (10**i)
            except ValueError:
                pass

        return(value)

    def loadTif(self):

        self.path = QtGui.QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.path:

            self.tifCounter = len(_glob.glob1(self.path, "*.tif"))
            self.tifFiles = _glob.glob(os.path.join(self.path, "*.tif"))

            self.table.setRowCount(int(self.tifCounter))
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(('FileName,Imager concentration[nM],Integration time [ms],Laserpower,Mean [Photons],Std [Photons]').split(','))

            for i in range(0, self.tifCounter):
                self.table.setItem(i, 0, QtGui.QTableWidgetItem(self.tifFiles[i]))

    def changeComb(self, indexval):

        sender = self.sender()
        comboval = sender.currentIndex()
        if comboval == 0:
            self.table.setItem(indexval, 2, QtGui.QTableWidgetItem(''))
            self.table.setItem(indexval, 3, QtGui.QTableWidgetItem(''))
        else:
            self.table.setItem(indexval, 2, QtGui.QTableWidgetItem(self.ImagersShort[comboval]))
            self.table.setItem(indexval, 3, QtGui.QTableWidgetItem(self.ImagersLong[comboval]))

    def readoutTable(self):
        tableshort = dict()
        tablelong = dict()
        maxcolor = 15
        for i in range(0, maxcolor - 1):
            try:
                tableshort[i] = self.table.item(i, 2).text()
                if tableshort[i] == '':
                    tableshort[i] = 'None'
            except AttributeError:
                tableshort[i] = 'None'

            try:
                tablelong[i] = self.table.item(i, 3).text()
                if tablelong[i] == '':
                    tablelong[i] = 'None'
            except AttributeError:
                tablelong[i] = 'None'
        return tablelong, tableshort

        # print(tablelong)
        # print(tableshort)

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
        return (bg, bgstd, las, time, conc, result == QDialog.Accepted)


def main():

    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        message = ''.join(traceback.format_exception(type, value, tback))
        errorbox = QtGui.QMessageBox.critical(window, 'An error occured', message)
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)
    sys.excepthook = excepthook


if __name__ == '__main__':
    main()
