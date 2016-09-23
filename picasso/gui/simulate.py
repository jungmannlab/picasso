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
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .. import io as _io, simulate
import random
import numpy as _np
from scipy.stats import norm
from scipy.optimize import curve_fit
import glob as _glob
import os
from PyQt4.QtGui import QDialog, QVBoxLayout, QDialogButtonBox, QDateTimeEdit, QApplication
from PyQt4.QtCore import Qt, QDateTime
import time

plt.style.use('ggplot')

"DEFAULT PARAMETERS"
# CAMERA
IMAGESIZE_DEFAULT = 64
ITIME_DEFAULT = 300
FRAMES_DEFAULT = 7500
PIXELSIZE_DEFAULT = 160
#PAINT
KON_DEFAULT = 1600000
IMAGERCONCENTRATION_DEFAULT = 5
MEANBRIGHT_DEFAULT = 700
#IMAGER
LASERPOWER_DEFAULT = 80
PSF_DEFAULT = 0.84
PHOTONRATE_DEFAULT = 56
PHOTONRATESTD_DEFAULT = 32
PHOTONBUDGET_DEFAULT = 1500000
PHOTONSLOPE_DEFAULT = 0.8
PHOTONSLOPESTD_DEFAULT = 0.4
#NOISE MODEL
IMAGERC_DEFAULT = 0.002130
LASERC_DEFAULT = 0.007152
CAMERAC_DEFAULT = 238.230221
EQA_DEFAULT = -0.001950
EQB_DEFAULT = 0.259030
EQC_DEFAULT = -42.905998

#STRUCTURE
STRUCTURE1_DEFAULT = 3
STRUCTURE2_DEFAULT = 4
STRUCTURE3_DEFAULT = '20,20'
STRUCTUREXX_DEFAULT = '0,20,40,60,0,20,40,60,0,20,40,60'
STRUCTUREYY_DEFAULT = '0,20,40,0,20,40,0,20,40,0,20,40'
STRUCTUREEX_DEFAULT = '1,1,1,1,1,1,1,1,1,1,1,1'
STRUCTURENO_DEFAULT = 9
STRUCTUREFRAME_DEFAULT = 6


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
        camera_groupbox = QtGui.QGroupBox('Camera Parameters')
        cgrid = QtGui.QGridLayout(camera_groupbox)

        camerasize = QtGui.QLabel('Image Size')
        integrationtime = QtGui.QLabel('Integration Time')
        totaltime = QtGui.QLabel('Total Integration Time')
        frames = QtGui.QLabel('Frames')
        pixelsize = QtGui.QLabel('Pixelsize')

        self.camerasizeEdit = QtGui.QSpinBox()
        self.camerasizeEdit.setRange(1,512)
        self.integrationtimeEdit = QtGui.QSpinBox()
        self.integrationtimeEdit.setRange(1,10000) #1-10.000ms
        self.framesEdit = QtGui.QSpinBox()
        self.framesEdit.setRange(10,1000000) #10-1.000.000 frames
        self.framesEdit.setSingleStep(1000)
        self.pixelsizeEdit = QtGui.QSpinBox()
        self.pixelsizeEdit.setRange(1,1000) #1 to 1000 nm frame size
        self.totaltimeEdit = QtGui.QLabel()

        self.camerasizeEdit.setValue(IMAGESIZE_DEFAULT)
        self.integrationtimeEdit.setValue(ITIME_DEFAULT)
        self.framesEdit.setValue(FRAMES_DEFAULT)
        self.pixelsizeEdit.setValue(PIXELSIZE_DEFAULT)

        self.integrationtimeEdit.valueChanged.connect(self.changeTime)
        self.framesEdit.valueChanged.connect(self.changeTime)

        cgrid.addWidget(camerasize,1,0)
        cgrid.addWidget(self.camerasizeEdit,1,1)
        cgrid.addWidget(QtGui.QLabel('Px'),1,2)
        cgrid.addWidget(integrationtime,2,0)
        cgrid.addWidget(self.integrationtimeEdit,2,1)
        cgrid.addWidget(QtGui.QLabel('ms'),2,2)
        cgrid.addWidget(frames,3,0)
        cgrid.addWidget(self.framesEdit,3,1)
        cgrid.addWidget(totaltime,4,0)
        cgrid.addWidget(self.totaltimeEdit,4,1)
        cgrid.addWidget(QtGui.QLabel('min'),4,2)
        cgrid.addWidget(pixelsize,5,0)
        cgrid.addWidget(self.pixelsizeEdit,5,1)
        cgrid.addWidget(QtGui.QLabel('nm'),5,2)

        #PAINT PARAMETERS
        paint_groupbox = QtGui.QGroupBox('PAINT Parameters')
        pgrid = QtGui.QGridLayout(paint_groupbox)

        kon = QtGui.QLabel('K_On')
        imagerconcentration = QtGui.QLabel('Imager Concentration')
        taud = QtGui.QLabel('Tau D')
        taub = QtGui.QLabel('Tau B')

        self.konEdit = QtGui.QDoubleSpinBox()
        self.konEdit.setRange(1,10000000)
        self.konEdit.setDecimals(0)
        self.konEdit.setSingleStep(100000)
        self.imagerconcentrationEdit = QtGui.QDoubleSpinBox()
        self.imagerconcentrationEdit.setRange(0.01,1000)
        self.taudEdit = QtGui.QLabel()
        self.taubEdit = QtGui.QDoubleSpinBox()
        self.taubEdit.setRange(1,10000)
        self.taubEdit.setDecimals(0)
        self.taubEdit.setSingleStep(10)

        self.konEdit.setValue(KON_DEFAULT)
        self.imagerconcentrationEdit.setValue(IMAGERCONCENTRATION_DEFAULT)
        self.taubEdit.setValue(MEANBRIGHT_DEFAULT)

        self.imagerconcentrationEdit.valueChanged.connect(self.changePaint)
        self.konEdit.valueChanged.connect(self.changePaint)

        pgrid.addWidget(kon,1,0)
        pgrid.addWidget(self.konEdit,1,1)
        pgrid.addWidget(imagerconcentration,2,0)
        pgrid.addWidget(self.imagerconcentrationEdit,2,1)
        pgrid.addWidget(QtGui.QLabel('nM'),2,2)
        pgrid.addWidget(taud,3,0)
        pgrid.addWidget(self.taudEdit,3,1)
        pgrid.addWidget(QtGui.QLabel('ms'),3,2)
        pgrid.addWidget(taub,4,0)
        pgrid.addWidget(self.taubEdit,4,1)
        pgrid.addWidget(QtGui.QLabel('ms'),4,2)

        #IMAGER Parameters
        imager_groupbox = QtGui.QGroupBox('Imager Parameters')
        igrid = QtGui.QGridLayout(imager_groupbox)

        laserpower = QtGui.QLabel('Laserpower')
        psf = QtGui.QLabel('PSF')
        psf_fwhm = QtGui.QLabel('PSF(FWHM)')
        photonrate = QtGui.QLabel('Photonrate')
        photonsframe = QtGui.QLabel('Photons (Frame)')
        photonratestd = QtGui.QLabel('Photonrate Std')
        photonstdframe = QtGui.QLabel('Photonrate Std (Frame)')
        photonbudget = QtGui.QLabel('Photonbudget')
        photonslope = QtGui.QLabel('Photonrate / Laser')
        photonslopeStd = QtGui.QLabel('Photonrate Std / Laser')

        self.laserpowerEdit = QtGui.QSpinBox()
        self.laserpowerEdit.setRange(0,1000)
        self.laserpowerEdit.setSingleStep(1)
        self.psfEdit = QtGui.QDoubleSpinBox()
        self.psfEdit.setRange(0,3)
        self.psfEdit.setSingleStep(0.01)
        self.psf_fwhmEdit = QtGui.QLabel()
        self.photonrateEdit = QtGui.QDoubleSpinBox()
        self.photonrateEdit.setRange(0,1000)
        self.photonrateEdit.setDecimals(0)
        self.photonsframeEdit = QtGui.QLabel()
        self.photonratestdEdit = QtGui.QDoubleSpinBox()
        self.photonratestdEdit.setRange(0,1000)
        self.photonratestdEdit.setDecimals(0)
        self.photonstdframeEdit = QtGui.QLabel()
        self.photonbudgetEdit = QtGui.QDoubleSpinBox()
        self.photonbudgetEdit.setRange(0,100000000)
        self.photonbudgetEdit.setSingleStep(100000)
        self.photonbudgetEdit.setDecimals(0)

        self.photonslopeEdit = QtGui.QDoubleSpinBox()
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

        igrid.addWidget(laserpower,0,0)
        igrid.addWidget(self.laserpowerEdit,0,1)
        igrid.addWidget(psf,1,0)
        igrid.addWidget(self.psfEdit,1,1)
        igrid.addWidget(QtGui.QLabel('Px'),1,2)
        igrid.addWidget(psf_fwhm,2,0)
        igrid.addWidget(self.psf_fwhmEdit,2,1)
        igrid.addWidget(QtGui.QLabel('nm'),2,2)
        igrid.addWidget(photonrate,3,0)
        igrid.addWidget(self.photonrateEdit,3,1)
        igrid.addWidget(QtGui.QLabel('Photons/ms'),3,2)
        igrid.addWidget(photonsframe,4,0)
        igrid.addWidget(self.photonsframeEdit,4,1)
        igrid.addWidget(QtGui.QLabel('Photons'),4,2)
        igrid.addWidget(photonratestd,5,0)
        igrid.addWidget(self.photonratestdEdit,5,1)
        igrid.addWidget(QtGui.QLabel('Photons/ms'),5,2)
        igrid.addWidget(photonstdframe,6,0)
        igrid.addWidget(self.photonstdframeEdit,6,1)
        igrid.addWidget(QtGui.QLabel('Photons'),6,2)
        igrid.addWidget(photonbudget,7,0)
        igrid.addWidget(self.photonbudgetEdit,7,1)
        igrid.addWidget(QtGui.QLabel('Photons'),7,2)
        igrid.addWidget(photonslope,8,0)
        igrid.addWidget(self.photonslopeEdit,8,1)
        igrid.addWidget(photonslopeStd,9,0)
        igrid.addWidget(self.photonslopeStdEdit,9,1)

         #NOISE MODEL
        noise_groupbox = QtGui.QGroupBox('Noise Model')
        ngrid = QtGui.QGridLayout(noise_groupbox)

        laserc = QtGui.QLabel('Lasercoefficient')
        imagerc = QtGui.QLabel('Imagercoefficient')
        camerac = QtGui.QLabel('Cameracoefficient')

        EquationA = QtGui.QLabel('Equation A')
        EquationB = QtGui.QLabel('Equation B')
        EquationC = QtGui.QLabel('Equation C')

        Bgoffset = QtGui.QLabel('Background Offset')
        BgStdoffset = QtGui.QLabel('Background Std Offset')

        backgroundframe = QtGui.QLabel('Background (Frame)')
        noiseLabel = QtGui.QLabel('Noise (Frame)')

        self.lasercEdit = QtGui.QDoubleSpinBox()
        self.lasercEdit.setRange(0,100000)
        self.lasercEdit.setDecimals(6)

        self.imagercEdit = QtGui.QDoubleSpinBox()
        self.imagercEdit.setRange(0,100000)
        self.imagercEdit.setDecimals(6)

        self.cameracEdit = QtGui.QDoubleSpinBox()
        self.cameracEdit.setRange(0,100000)
        self.cameracEdit.setDecimals(6)

        self.EquationBEdit = QtGui.QDoubleSpinBox()
        self.EquationBEdit.setRange(-100000,100000)
        self.EquationBEdit.setDecimals(6)

        self.EquationAEdit = QtGui.QDoubleSpinBox()
        self.EquationAEdit.setRange(-100000,100000)
        self.EquationAEdit.setDecimals(6)

        self.EquationCEdit = QtGui.QDoubleSpinBox()
        self.EquationCEdit.setRange(-100000,100000)
        self.EquationCEdit.setDecimals(6)

        self.lasercEdit.setValue(LASERC_DEFAULT)
        self.imagercEdit.setValue(IMAGERC_DEFAULT)
        self.cameracEdit.setValue(CAMERAC_DEFAULT)

        self.EquationAEdit.setValue(EQA_DEFAULT)
        self.EquationBEdit.setValue(EQB_DEFAULT)
        self.EquationCEdit.setValue(EQC_DEFAULT)

        self.BgoffsetEdit = QtGui.QDoubleSpinBox()
        self.BgoffsetEdit.setRange(-100000,100000)
        self.BgoffsetEdit.setDecimals(6)

        self.BgStdoffsetEdit = QtGui.QDoubleSpinBox()
        self.BgStdoffsetEdit.setRange(-100000,100000)
        self.BgStdoffsetEdit.setDecimals(6)

        self.lasercEdit.valueChanged.connect(self.changeNoise)
        self.imagercEdit.valueChanged.connect(self.changeNoise)
        self.cameracEdit.valueChanged.connect(self.changeNoise)
        self.EquationAEdit.valueChanged.connect(self.changeNoise)
        self.EquationBEdit.valueChanged.connect(self.changeNoise)
        self.EquationCEdit.valueChanged.connect(self.changeNoise)

        backgroundframe = QtGui.QLabel('Background (Frame)')
        noiseLabel = QtGui.QLabel('Noise (Frame)')

        self.backgroundframeEdit = QtGui.QLabel()
        self.noiseEdit = QtGui.QLabel()

        ngrid.addWidget(laserc,0,0)
        ngrid.addWidget(self.lasercEdit,0,1)
        ngrid.addWidget(imagerc,1,0)
        ngrid.addWidget(self.imagercEdit,1,1)
        ngrid.addWidget(camerac,2,0)
        ngrid.addWidget(self.cameracEdit,2,1)
        ngrid.addWidget(EquationA,3,0)
        ngrid.addWidget(self.EquationAEdit,3,1)
        ngrid.addWidget(EquationB,4,0)
        ngrid.addWidget(self.EquationBEdit,4,1)
        ngrid.addWidget(EquationC,5,0)
        ngrid.addWidget(self.EquationCEdit,5,1)
        ngrid.addWidget(Bgoffset,6,0)
        ngrid.addWidget(self.BgoffsetEdit,6,1)
        ngrid.addWidget(BgStdoffset,7,0)
        ngrid.addWidget(self.BgStdoffsetEdit,7,1)
        ngrid.addWidget(backgroundframe,8,0)
        ngrid.addWidget(self.backgroundframeEdit,8,1)
        ngrid.addWidget(noiseLabel,9,0)
        ngrid.addWidget(self.noiseEdit,9,1)

        calibrateNoiseButton = QtGui.QPushButton("Calibrate Noise Model")
        calibrateNoiseButton.clicked.connect(self.calibrateNoise)
        ngrid.addWidget(calibrateNoiseButton,10,0,1,3)

        # STRUCTURE DEFINITIONS
        structure_groupbox = QtGui.QGroupBox('Structure')
        sgrid = QtGui.QGridLayout(structure_groupbox)

        structureno = QtGui.QLabel('Number of Structures')
        structureframe = QtGui.QLabel('Frame')

        self.structure1 = QtGui.QLabel('Rows')
        self.structure2 = QtGui.QLabel('Columns')
        self.structure3 = QtGui.QLabel('Spacing X,Y')
        self.structure3Label = QtGui.QLabel('nm')

        structurexx = QtGui.QLabel('Stucture X')
        structureyy = QtGui.QLabel('Structure Y')
        structureex = QtGui.QLabel('Exchange Labels')

        structurecomboLabel = QtGui.QLabel('Type')

        structureIncorporation = QtGui.QLabel('Incorporation')

        self.structurenoEdit = QtGui.QSpinBox()
        self.structurenoEdit.setRange(1,1000)
        self.structureframeEdit = QtGui.QSpinBox()
        self.structureframeEdit.setRange(4,16)
        self.structurexxEdit = QtGui.QLineEdit(STRUCTUREXX_DEFAULT)
        self.structureyyEdit = QtGui.QLineEdit(STRUCTUREYY_DEFAULT)
        self.structureexEdit = QtGui.QLineEdit(STRUCTUREEX_DEFAULT)
        self.structureIncorporationEdit = QtGui.QDoubleSpinBox()
        self.structureIncorporationEdit.setRange(1,100)
        self.structureIncorporationEdit.setDecimals(0)
        self.structureIncorporationEdit.setValue(100)

        self.structurecombo = QtGui.QComboBox()
        self.structurecombo.addItem("Grid")
        self.structurecombo.addItem("Circle")
        self.structurecombo.addItem("Custom")

        self.structure1Edit = QtGui.QSpinBox()
        self.structure1Edit.setRange(1,1000)
        self.structure1Edit.setValue(STRUCTURE1_DEFAULT)
        self.structure2Edit = QtGui.QSpinBox()
        self.structure2Edit.setRange(1,1000)
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

        structurerandom = QtGui.QLabel('Random Arrangement')
        structurerandomOrientation = QtGui.QLabel('Random Orientation')

        self.structurerandomEdit.stateChanged.connect(self.generatePositions)
        self.structurerandomOrientationEdit.stateChanged.connect(self.generatePositions)
        self.structureIncorporationEdit.valueChanged.connect(self.generatePositions)

        self.structurecombo.currentIndexChanged.connect(self.changeStructureType)

        sgrid.addWidget(structureno,1,0)
        sgrid.addWidget(self.structurenoEdit,1,1)
        sgrid.addWidget(structureframe,2,0)
        sgrid.addWidget(self.structureframeEdit,2,1)
        sgrid.addWidget(QtGui.QLabel('Px'),2,2)
        sgrid.addWidget(structurecomboLabel)
        sgrid.addWidget(self.structurecombo,3,1)

        sgrid.addWidget(self.structure1,4,0)
        sgrid.addWidget(self.structure1Edit,4,1)
        sgrid.addWidget(self.structure2,5,0)
        sgrid.addWidget(self.structure2Edit,5,1)
        sgrid.addWidget(self.structure3,6,0)
        sgrid.addWidget(self.structure3Edit,6,1)
        sgrid.addWidget(self.structure3Label,6,2)

        sgrid.addWidget(structurexx,7,0)
        sgrid.addWidget(self.structurexxEdit,7,1)
        sgrid.addWidget(QtGui.QLabel('nm'),7,2)
        sgrid.addWidget(structureyy,8,0)
        sgrid.addWidget(self.structureyyEdit,8,1)
        sgrid.addWidget(QtGui.QLabel('nm'),8,2)
        sgrid.addWidget(structureex,9,0)
        sgrid.addWidget(self.structureexEdit,9,1)
        sgrid.addWidget(structureIncorporation,10,0)
        sgrid.addWidget(self.structureIncorporationEdit,10,1)
        sgrid.addWidget(QtGui.QLabel('%'),10,2)
        sgrid.addWidget(structurerandom,11,1)
        sgrid.addWidget(self.structurerandomEdit,11,0)
        sgrid.addWidget(structurerandomOrientation,12,1)
        sgrid.addWidget(self.structurerandomOrientationEdit,12,0)

        importDesignButton = QtGui.QPushButton("Import Structure from Design")
        importDesignButton.clicked.connect(self.importDesign)
        sgrid.addWidget(importDesignButton,13,0,1,3)

        generateButton = QtGui.QPushButton("Generate Positions")
        generateButton.clicked.connect(self.generatePositions)
        sgrid.addWidget(generateButton,14,0,1,3)

        # POSITION PREVIEW WINDOW
        posgrid = QtGui.QGridLayout()#
        # STRUCTURE PREVIEW WINDOW
        strgrid = QtGui.QGridLayout()

        simulateButton = QtGui.QPushButton("Simulate Data")
        #simulateButton.setStyleSheet("background-color: white")
        self.exchangeroundsEdit = QtGui.QLineEdit('1')

        quitButton = QtGui.QPushButton('Quit', self)
        quitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        quitButton.resize(quitButton.sizeHint())

        loadButton = QtGui.QPushButton("Load Settings from Previous Simulation")
        importButton = QtGui.QPushButton("Import from Experiment (hdf5)")

        btngridR = QtGui.QGridLayout()
        btngridR.addWidget(importButton)
        btngridR.addWidget(loadButton)

        btngridR.addWidget(QtGui.QLabel('Exchange Rounds to be simulated:'))
        btngridR.addWidget(self.exchangeroundsEdit)
        btngridR.addWidget(simulateButton)
        btngridR.addWidget(quitButton)

        simulateButton.clicked.connect(self.saveDialog)
        importButton.clicked.connect(self.importhdf5)
        loadButton.clicked.connect(self.loadSettings)

        self.show()
        self.changeTime()
        self.changePSF()
        self.changeNoise()
        self.changePaint()

        self.figure1 = plt.figure()
        self.figure2 = plt.figure()
        self.canvas1 = FigureCanvas(self.figure1)
        self.canvas1.setMinimumSize(200,220)

        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.setMinimumSize(200,220)

        posgrid.addWidget(self.canvas1)
        strgrid.addWidget(self.canvas2)
        #strgrid.addWidget(self.button)

        # DEFINE ARRANGEMENT
        self.grid.addLayout(posgrid,1,0)
        self.grid.addLayout(strgrid,1,1)
        self.grid.addWidget(structure_groupbox,2,0,2,1)
        self.grid.addWidget(camera_groupbox,1,2)
        self.grid.addWidget(paint_groupbox,3,1)
        self.grid.addWidget(imager_groupbox,2,1)
        self.grid.addWidget(noise_groupbox,2,2)
        self.grid.addLayout(btngridR,3,2)

        mainWidget = QtGui.QWidget()
        mainWidget.setLayout(self.grid)
        self.setCentralWidget(mainWidget)
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('simulate')
        #CALL FUNCTIONS
        self.generatePositions()
        self.mainpbar = QtGui.QProgressBar(self)
        self.grid.addWidget(self.mainpbar,4,0,1,4)
        self.mainpbar.setValue(0)
        self.statusBar().showMessage('Simulate Ready.')

    def changeTime(self):
        itime = self.integrationtimeEdit.value()
        frames = self.framesEdit.value()
        totaltime = itime*frames/1000/60
        totaltime = round(totaltime*100)/100
        self.totaltimeEdit.setText(str(totaltime))

        photonrate = self.photonrateEdit.value()
        photonratestd = self.photonratestdEdit.value()

        photonsframe = round(photonrate*itime)
        photonsframestd = round(photonratestd*itime)

        self.photonsframeEdit.setText(str(photonsframe))
        self.photonstdframeEdit.setText(str(photonsframestd))

        self.changeNoise()

    def changePaint(self):
        kon = self.konEdit.value()
        imagerconcentration = self.imagerconcentrationEdit.value()
        taud = round(1/(kon*imagerconcentration*1/10**9)*1000)
        self.taudEdit.setText(str(taud))
        self.changeNoise()

    def changePSF(self):
        psf = self.psfEdit.value()
        pixelsize = self.pixelsizeEdit.value()
        psf_fwhm = round(psf*pixelsize*2.355)
        self.psf_fwhmEdit.setText(str(psf_fwhm))

    def changeImager(self):
        laserpower = self.laserpowerEdit.value()

        itime = self.integrationtimeEdit.value()
        photonslope = self.photonslopeEdit.value()
        photonslopestd = self.photonslopeStdEdit.value()
        photonrate = photonslope*laserpower
        photonratestd = photonslopestd*laserpower

        photonsframe = round(photonrate*itime)
        photonsframestd = round(photonratestd*itime)

        self.photonsframeEdit.setText(str(photonsframe))
        self.photonstdframeEdit.setText(str(photonsframestd))

        self.photonrateEdit.setValue((photonrate))
        self.photonratestdEdit.setValue((photonratestd))
        self.changeNoise()

    def changeNoise(self):
        itime = self.integrationtimeEdit.value()

        #NEW NOISE MODEL
        laserc = self.lasercEdit.value()
        imagerc = self.imagercEdit.value()
        camerac = self.cameracEdit.value()
        bgoffset = self.BgoffsetEdit.value()

        laserpower = self.laserpowerEdit.value()
        imagerconcentration = self.imagerconcentrationEdit.value()

        bgmodel = (laserc + imagerc*imagerconcentration)*laserpower*itime+camerac+bgoffset

        equationA = self.EquationAEdit.value()
        equationB = self.EquationBEdit.value()
        equationC = self.EquationCEdit.value()
        bgstdoffset = self.BgStdoffsetEdit.value()

        bgmodelstd = equationA*laserpower*itime+equationB*bgmodel+equationC+bgstdoffset
        #bgmodelstd = self.fitFuncStd(x_3d, fitParamsBg[0],fitParamsBg[1],fitParamsBg[2])

        self.backgroundframeEdit.setText(str(int(bgmodel)))
        self.noiseEdit.setText(str(int(bgmodelstd)))

    def changeStructureType(self):
        typeindex = self.structurecombo.currentIndex()
        #TYPEINDEX: 0 = GRID, 1 = CIRCLE, 2 = CUSTOM

        if typeindex == 0:
            self.structure1.show()
            self.structure2.show()
            self.structure3.show()
            self.structure1Edit.show()
            self.structure2Edit.show()
            self.structure3Edit.show()
            self.structure3Label.show()
            self.structure1.setText('Rows')
            self.structure2.setText('Columns')
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

        if typeindex == 0: # grid

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

            for i in range(0,rows):
                for j in range(0,cols):
                    structurexx = structurexx +str(i*spacingx)+','
                    structureyy = structureyy +str(j*spacingy)+','
                    structureex = structureex +'1,'

            structurexx = structurexx[:-1]
            structureyy = structureyy[:-1]
            structureex = structureex[:-1]

            self.structurexxEdit.setText(structurexx)
            self.structureyyEdit.setText(structureyy)
            self.structureexEdit.setText(structureex)
            self.generatePositions()

        elif typeindex == 1: # CIRCLE
            labels = self.structure2Edit.value()
            diametertxt = _np.asarray((self.structure3Edit.text()).split(","))
            try:
                diameter = float(diametertxt[0])
            except ValueError:
                diameter= 100

            twopi = 2*3.1415926535

            circdata = _np.arange(0,twopi,twopi/labels)


            xxval = _np.round(_np.cos(circdata)*diameter*100)/100
            yyval = _np.round(_np.sin(circdata)*diameter*100)/100


            structurexx = ''
            structureyy = ''
            structureex = ''

            for i in range(0,xxval.size):
                    structurexx = structurexx +str(xxval[i])+','
                    structureyy = structureyy +str(yyval[i])+','
                    structureex = structureex +'1,'


            structurexx = structurexx[:-1]
            structureyy = structureyy[:-1]
            structureex = structureex[:-1]

            self.structurexxEdit.setText(structurexx)
            self.structureyyEdit.setText(structureyy)
            self.structureexEdit.setText(structureex)
            self.generatePositions()

        elif typeindex == 2: # Custom

                self.generatePositions()

    def keyPressEvent(self, e):

        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    def vectorToString(self,x):
        x_arrstr = _np.char.mod('%f', x)
        x_str = ",".join(x_arrstr)
        return x_str

    def simulate(self,fileNameOld):

        #READ IN PARAMETERS
        #STRUCTURE
        structureNo = self.structurenoEdit.value()
        structureFrame = self.structureframeEdit.value()
        structureIncorporation = self.structureIncorporationEdit.value()
        structureArrangement = int(self.structurerandomEdit.checkState())
        structureOrientation = int(self.structurerandomOrientationEdit.checkState())
        structurex =self.structurexxEdit.text()
        structurey = self.structureyyEdit.text()
        structureextxt = self.structureexEdit.text()

        #PAINT
        kon = self.konEdit.value()
        imagerconcentration = self.imagerconcentrationEdit.value()
        taub = self.taubEdit.value()
        taud =int(self.taudEdit.text())

        #IMAGER PARAMETERS
        psf = self.psfEdit.value()
        photonrate = self.photonrateEdit.value()
        photonratestd = self.photonratestdEdit.value()
        photonbudget = self.photonbudgetEdit.value()
        laserpower = self.laserpowerEdit.value()
        photonslope = self.photonslopeEdit.value()
        photonslopeStd = self.photonslopeStdEdit.value()

        #CAMERA PARAMETERS
        imagesize = self.camerasizeEdit.value()
        itime = self.integrationtimeEdit.value()
        frames = self.framesEdit.value()
        pixelsize = self.pixelsizeEdit.value()

        #NOISE MODEL
        background = int(self.backgroundframeEdit.text())
        noise = int(self.noiseEdit.text())

        laserc = self.lasercEdit.value()
        imagerc = self.imagercEdit.value()
        camerac = self.cameracEdit.value()
        bgoffset = self.BgoffsetEdit.value()

        equationA = self.EquationAEdit.value()
        equationB = self.EquationBEdit.value()
        equationC = self.EquationCEdit.value()
        bgstdoffset = self.BgStdoffsetEdit.value()

        structurexx,structureyy,structureex = self.readStructure()

        self.statusBar().showMessage('Simulation started')
        struct = self.newstruct
        handlex = self.vectorToString(struct[0,:])
        handley = self.vectorToString(struct[1,:])
        handleex = self.vectorToString(struct[2,:])
        handless = self.vectorToString(struct[3,:])

        exchangeroundstoSim = _np.asarray((self.exchangeroundsEdit.text()).split(","))
        exchangeroundstoSim = exchangeroundstoSim.astype(_np.int)

        noexchangecolors = len(set(exchangeroundstoSim))
        exchangecolors = list(set(exchangeroundstoSim))

        print(noexchangecolors)
        print(exchangecolors)

        t0 = time.time()


        for i in range(0,noexchangecolors):
            if noexchangecolors > 1:
                fileName = _io.multiple_filenames(fileNameOld, i)
                partstruct = struct[:,struct[2,:]==exchangecolors[i]]

            else:
                fileName = fileNameOld
                partstruct = struct[:,struct[2,:]==exchangecolors[0]]

            self.statusBar().showMessage('Distributing Photons ...')


            bindingsitesx = partstruct[0,:]
            bindingsitesy = partstruct[1,:]
            nosites  = len(bindingsitesx) # number of binding sites in image
            photondist = _np.zeros((nosites,frames),dtype = _np.int)
            meandark = int(taud)
            meanbright = int(taub)
            for i in range(0,nosites):
                photondisttemp = simulate.distphotons(partstruct,itime,frames,taud,taub,photonrate,photonratestd,photonbudget)

                photondist[i,:] = photondisttemp
                outputmsg = 'Distributing Photons ... ' + str(_np.round(i/nosites*1000)/10) +' %'
                self.statusBar().showMessage(outputmsg)
                self.mainpbar.setValue(_np.round(i/nosites*1000)/10)

            self.statusBar().showMessage('Converting to Image ... ')

            movie = _np.zeros(shape=(frames,imagesize,imagesize), dtype='<u2')
            for runner in range(0,frames):
                movie[runner,:,:]=simulate.convertMovie(runner,photondist,partstruct,imagesize,frames,psf,photonrate,background, noise)
                outputmsg = 'Converting to Image ... ' + str(_np.round(runner/frames*1000)/10) +' %'

                self.statusBar().showMessage(outputmsg)
                self.mainpbar.setValue(_np.round(runner/frames*1000)/10)
            self.statusBar().showMessage('Converting to Image ... Complete.')
            self.statusBar().showMessage('Saving Movie ...')

            info = {'Generated by':'Picasso simulate',
                    'Byte Order': '<',
                    'Camera': 'Simulation',
                    'Data Type': movie.dtype.name,
                    'Frames': frames,
                    'Structure.Frame':structureFrame,
                    'Structure.Number':structureNo,
                    'Structure.StructureX':structurex,
                    'Structure.StructureY':structurey,
                    'Structure.StructureEx':structureextxt,
                    'Structure.HandleX':handlex,
                    'Structure.HandleY':handley,
                    'Structure.HandleEx':handleex,
                    'Structure.HandleStruct':handless,
                    'Structure.Incorporation':structureIncorporation,
                    'Structure.Arrangement':structureArrangement,
                    'Structure.Orientation':structureOrientation,
                    'PAINT.k_on':kon,
                    'PAINT.imager':imagerconcentration,
                    'PAINT.taub':taub,
                    'Imager.PSF':psf,
                    'Imager.Photonrate':photonrate,
                    'Imager.Photonrate Std':photonratestd,
                    'Imager.Photonbudget':photonbudget,
                    'Imager.Laserpower':laserpower,
                    'Imager.Photonslope':photonslope,
                    'Imager.PhotonslopeStd':photonslope,
                    'Camera.Image Size':imagesize,
                    'Camera.Integration Time':itime,
                    'Camera.Frames':frames,
                    'Camera.Pixelsize':pixelsize,
                    'Noise.Lasercoefficient':laserc,
                    'Noise.Imagercoefficient':imagerc,
                    'Noise.Cameracoefficient':camerac,
                    'Noise.EquationA':equationA,
                    'Noise.EquationB':equationB,
                    'Noise.EquationC':equationC,
                    'Noise.BackgroundOff':bgoffset,
                    'Noise.BackgroundStdOff':bgstdoffset,
                    'Height': imagesize,
                    'Width': imagesize}

            simulate.saveMovie(fileName,movie,info)
            self.statusBar().showMessage('Movie saved to: '+fileName)
        dt = time.time() - t0
        self.statusBar().showMessage('All computations finished. Last file saved to: '+fileName+'. Total Simulation time: {:.2f} Seconds.'.format(dt))



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

            self.lasercEdit.setValue(info[0]['Noise.Lasercoefficient'])
            self.imagercEdit.setValue(info[0]['Noise.Imagercoefficient'])
            self.cameracEdit.setValue(info[0]['Noise.Cameracoefficient'])
            self.BgoffsetEdit.setValue(info[0]['Noise.BackgroundOff'])

            self.EquationAEdit.setValue(info[0]['Noise.EquationA'])
            self.EquationBEdit.setValue(info[0]['Noise.EquationB'])
            self.EquationCEdit.setValue(info[0]['Noise.EquationC'])
            self.BgStdoffsetEdit.setValue(info[0]['Noise.BackgroundStdOff'])

            #SET POSITIONS

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

            structure = _np.array([handlexx,handleyy,handleex,handless])

            self.newstruct = structure

            self.plotPositions()
            self.statusBar().showMessage('Settings loaded from: '+path)

    def importDesign(self):
            path = QtGui.QFileDialog.getOpenFileName(self, 'Open yaml', filter='*.yaml')
            if path:
                info = _io.load_info(path)
                self.structurexxEdit.setText(info[0]['Structure.StructureX'])
                self.structureyyEdit.setText(info[0]['Structure.StructureY'])
                self.structureexEdit.setText(info[0]['Structure.StructureEx'])



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



        minlen = min(len(structureex),len(structurexx),len(structureyy))

        structurexx = structurexx[0:minlen]
        structureyy = structureyy[0:minlen]
        structureex = structureex[0:minlen]

        return structurexx,structureyy,structureex

    def plotStructure(self):

            structurexx,structureyy,structureex = self.readStructure()
            noexchangecolors = len(set(structureex))
            exchangecolors = list(set(structureex))
            self.figure2.suptitle('Structure [nm]')
            ax1 = self.figure2.add_subplot(111)
            ax1.cla()
            ax1.hold(True)
            ax1.axis('equal')

            for i in range(0,noexchangecolors):
                plotxx = []
                plotyy = []
                for j in range(0,len(structureex)):
                    if structureex[j] == exchangecolors[i]:
                        plotxx.append(structurexx[j])
                        plotyy.append(structureyy[j])
                ax1.plot(plotxx,plotyy,'o')

            distx = round(1/10*(max(structurexx)-min(structurexx)))
            disty = round(1/10*(max(structureyy)-min(structureyy)))

            ax1.axes.set_xlim((min(structurexx)-distx,max(structurexx)+distx))
            ax1.axes.set_ylim((min(structureyy)-disty,max(structureyy)+disty))
            self.canvas2.draw()

            exchangecolorsList= ','.join(map(str, exchangecolors))
            #UPDATE THE EXCHANGE COLORS IN BUTTON TO BE simulated
            self.exchangeroundsEdit.setText(str(exchangecolorsList))



    def generatePositions(self):
        self.plotStructure()
        structurexx,structureyy,structureex = self.readStructure()
        pixelsize = self.pixelsizeEdit.value()
        structure = simulate.defineStructure(structurexx,structureyy,structureex,pixelsize)

        number = self.structurenoEdit.value()
        imageSize =self.camerasizeEdit.value()
        frame = self.structureframeEdit.value()
        arrangement = int(self.structurerandomEdit.checkState())
        gridpos = simulate.generatePositions(number,imageSize,frame,arrangement)

        orientation = int(self.structurerandomOrientationEdit.checkState())
        incorporation = self.structureIncorporationEdit.value()/100
        exchange = 0
        self.newstruct = simulate.prepareStructures(structure,gridpos,orientation,number,incorporation,exchange)

        self.figure1.suptitle('Positions [Px]')
        ax1 = self.figure1.add_subplot(111)
        ax1.cla()
        ax1.hold(True)
        ax1.axis('equal')
        ax1.plot(self.newstruct[0,:],self.newstruct[1,:],'+')
        # PLOT FRAME
        ax1.add_patch(
            patches.Rectangle(
                (frame, frame),
                imageSize-2*frame,
                imageSize-2*frame,
                linestyle='dashed',
                edgecolor="#000000",
                fill=False      # remove background
            )
        )

        ax1.axes.set_xlim(0,imageSize)
        ax1.axes.set_ylim(0,imageSize)

        self.canvas1.draw()

        #PLOT first structure
        struct1 = self.newstruct[:,self.newstruct[3,:]==0]

        noexchangecolors = len(set(struct1[2,:]))
        exchangecolors = list(set(struct1[2,:]))
        self.noexchangecolors = exchangecolors
        self.figure2.suptitle('Structure [nm]')
        ax1 = self.figure2.add_subplot(111)
        ax1.cla()
        ax1.hold(True)
        ax1.axis('equal')

        structurexx = struct1[0,:]
        structureyy = struct1[1,:]
        structureex = struct1[2,:]
        structurexx_nm = _np.multiply(structurexx-min(structurexx),pixelsize)
        structureyy_nm = _np.multiply(structureyy-min(structureyy),pixelsize)

        for i in range(0,noexchangecolors):
            plotxx = []
            plotyy = []
            for j in range(0,len(structureex)):
                if structureex[j] == exchangecolors[i]:
                    plotxx.append(structurexx_nm[j])
                    plotyy.append(structureyy_nm[j])
            ax1.plot(plotxx,plotyy,'o')

            #plotxx = structurexx(structureex == exchangecolors[i])
            #print(plotxx)

        #ax1.plot(structurexx,structureyy,'o')
        #ax1.axis('equal')
            distx = round(1/10*(max(structurexx_nm)-min(structurexx_nm)))
            disty = round(1/10*(max(structureyy_nm)-min(structureyy_nm)))

            ax1.axes.set_xlim((min(structurexx_nm)-distx,max(structurexx_nm)+distx))
            ax1.axes.set_ylim((min(structureyy_nm)-disty,max(structureyy_nm)+disty))
        self.canvas2.draw()

    def fitFunc(x, a, b, c):
        return (a + b*x[0])*x[1]*x[2]+c

    def fitFuncStd(x, a, b, c):
        return (a*x[0]*x[1]+b*x[2]+c)



    def plotPositions(self):
        structurexx,structureyy,structureex = self.readStructure()
        pixelsize = self.pixelsizeEdit.value()
        structure = simulate.defineStructure(structurexx,structureyy,structureex,pixelsize)

        number = self.structurenoEdit.value()
        imageSize =self.camerasizeEdit.value()
        frame = self.structureframeEdit.value()
        arrangement = int(self.structurerandomEdit.checkState())
        gridpos = simulate.generatePositions(number,imageSize,frame,arrangement)

        orientation = int(self.structurerandomOrientationEdit.checkState())
        incorporation = self.structureIncorporationEdit.value()/100
        exchange = 0

        self.figure1.suptitle('Positions [Px]')
        ax1 = self.figure1.add_subplot(111)
        ax1.cla()
        ax1.hold(True)
        ax1.axis('equal')
        ax1.plot(self.newstruct[0,:],self.newstruct[1,:],'+')
        # PLOT FRAME
        ax1.add_patch(
            patches.Rectangle(
                (frame, frame),
                imageSize-2*frame,
                imageSize-2*frame,
                linestyle='dashed',
                edgecolor="#000000",
                fill=False      # remove background
            )
        )

        ax1.axes.set_xlim(0,imageSize)
        ax1.axes.set_ylim(0,imageSize)

        self.canvas1.draw()

        #PLOT first structure
        struct1 = self.newstruct[:,self.newstruct[3,:]==0]

        noexchangecolors = len(set(struct1[2,:]))
        exchangecolors = list(set(struct1[2,:]))
        self.noexchangecolors = exchangecolors
        self.figure2.suptitle('Structure [nm]')
        ax1 = self.figure2.add_subplot(111)
        ax1.cla()
        ax1.hold(True)

        structurexx = struct1[0,:]
        structureyy = struct1[1,:]
        structureex = struct1[2,:]
        structurexx_nm = _np.multiply(structurexx-min(structurexx),pixelsize)
        structureyy_nm = _np.multiply(structureyy-min(structureyy),pixelsize)

        for i in range(0,noexchangecolors):
            plotxx = []
            plotyy = []
            for j in range(0,len(structureex)):
                if structureex[j] == exchangecolors[i]:
                    plotxx.append(structurexx_nm[j])
                    plotyy.append(structureyy_nm[j])
            ax1.plot(plotxx,plotyy,'o')

            #plotxx = structurexx(structureex == exchangecolors[i])
            #print(plotxx)

        #ax1.plot(structurexx,structureyy,'o')
        #ax1.axis('equal')
            distx = round(1/10*(max(structurexx_nm)-min(structurexx_nm)))
            disty = round(1/10*(max(structureyy_nm)-min(structureyy_nm)))

            ax1.axes.set_xlim((min(structurexx_nm)-distx,max(structurexx_nm)+distx))
            ax1.axes.set_ylim((min(structureyy_nm)-disty,max(structureyy_nm)+disty))
        self.canvas2.draw()

    def openDialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open design', filter='*.yaml')
        if path:
            self.mainscene.loadCanvas(path)
            print(path)
            self.statusBar().showMessage('File loaded from: '+path)

    def saveDialog(self):
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save movie to..', filter='*.raw')
        if path:
            self.simulate(path)

    def importhdf5(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open localizations', filter='*.hdf5')
        if path:
            self.readhdf5(path)

    def calibrateNoise(self):

        bg,bgstd, las, time, conc, ok = CalibrationDialog.setExt()

        _np.asarray(bg)
        _np.asarray(bgstd)
        _np.asarray(las)
        _np.asarray(time)
        _np.asarray(conc)

        #bg = _np.array([295.946426,233.0569123,274.4858148,492.9237402,350.7499474,427.8461072,686.6002138,450.1768474,578.2886671,317.4836201,263.7906357,295.183758,556.4378002,393.6139337,489.2362959,790.0168324,519.3466806,675.4153349,390.6218063,311.8076188,362.3901329,770.4890059,534.3767198,686.1671561,1139.555401,751.2064186,1001.537454,346.4805993,633.0231806,912.0119824])
        #las = _np.array([110,50,80,110,50,80,110,50,80,110,50,80,110,50,80,110,50,80,110,50,80,110,50,80,110,50,80,50,50,50])
        #time = _np.array([100,100,100,300,300,300,500,500,500,100,100,100,300,300,300,500,500,500,100,100,100,300,300,300,500,500,500,100,300,500])
        #conc = _np.array([0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5,5,8,8,8])

        x_3d = _np.array([conc,las,time])

        p0 = [1,1,100]

        fitParamsBg, fitCovariances = curve_fit(self.fitFunc, x_3d, bg, p0)
        print(' fit coefficients :\n', fitParamsBg)

        # SET VALUES TO PARAMETER


        self.imagercEdit.setValue(fitParamsBg[0])
        self.lasercEdit.setValue(fitParamsBg[1])
        self.cameracEdit.setValue(fitParamsBg[2])

        x_3dStd = _np.array([las,time,bg])
        p0S = [1,1,1]
        fitParamsStd, fitCovariances = curve_fit(self.fitFuncStd, x_3dStd, bgstd, p0S)

        print(' fit coefficients2:\n', fitParamsStd)

        self.EquationAEdit.setValue(fitParamsStd[0])
        self.EquationBEdit.setValue(fitParamsStd[1])
        self.EquationCEdit.setValue(fitParamsStd[2])

        #Noise model working point

        figure4 = plt.figure()
        #figure3.suptitle('hdf5 import')

        #Background
        bgmodel = self.fitFunc(x_3d, fitParamsBg[0],fitParamsBg[1],fitParamsBg[2])
        ax1 = figure4.add_subplot(121)
        ax1.cla()
        ax1.plot(bg, bgmodel,'o')
        title = "Background Model:"
        ax1.set_title(title)

        #Std
        ax2 = figure4.add_subplot(122)
        ax2.cla()
        ax2.plot(bgstd, bgmodelstd,'o')
        title = "Background Model Std:"
        ax2.set_title(title)

        figure4.show()









    def sigmafilter(self,data,sigmas):
        #Filter data to be withing +- sigma
        sigma = _np.std(data)
        mean = _np.mean(data)

        datanew = data[data<(mean+sigmas*sigma)]
        datanew = datanew[datanew>(mean-sigmas*sigma)]
        return datanew

    def readhdf5(self, path):
        locs, self.info = _io.load_locs(path)
        integrationtime, ok1 = QtGui.QInputDialog.getText(self, 'Input Dialog',
            'Enter integration time in ms:')
        integrationtime = int(integrationtime)
        if ok1:
            imagerconcentration, ok2 = QtGui.QInputDialog.getText(self, 'Input Dialog',
                'Enter imager concentration in nM:')
            imagerconcentration = float(imagerconcentration)

            if ok2:
                laserpower, ok3 = QtGui.QInputDialog.getText(self, 'Input Dialog',
                    'Enter Laserpower in a.u.:')
                laserpower = float(laserpower)
                if ok3:
                    #self.le.setText(str(text))

                    photons = locs['photons']
                    sigmax = locs['sx']
                    sigmay = locs['sy']
                    bg = locs['bg']

                    nosigmas = 3
                    photons = self.sigmafilter(photons,nosigmas)
                    sigmax = self.sigmafilter(sigmax,nosigmas)
                    sigmay = self.sigmafilter(sigmay,nosigmas)
                    bg = self.sigmafilter(bg,nosigmas)

                    #Clean up data:

                    figure3 = plt.figure()
                    #figure3.suptitle('hdf5 import')

                    #Photons
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

                    #Sigma X & Sigma Y
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

                    figure3.show()

                    #Calculate Rates
                    #Photonrate, Photonrate Std, PSF

                    photonrate = int(photonsmu/integrationtime)
                    photonratestd = int(photonsstd/integrationtime)
                    psf = int(sigmamu*100)/100
                    photonrate = int(photonsmu/integrationtime)

                    #Calculate backgroundrate
                    bgbase = self.backgroundbaseEdit.value()
                    bgrate = int((bgmu-bgbase)/(integrationtime/1000))

                    # CALCULATE BG AND BG_STD FROM MODEL AND ADJUST OFFSET

                    laserc = self.lasercEdit.value()
                    imagerc = self.imagercEdit.value()
                    camerac = self.cameracEdit.value()


                    bgmodel = (laserc + imagerc*imagerconcentration)*laserpower*integrationtime+camerac

                    equationA = self.EquationAEdit.value()
                    equationB = self.EquationBEdit.value()
                    equationC = self.EquationCEdit.value()



                    bgmodelstd = equationA*laserpower*integrationtime+equationB*bgmu+equationC
                    #bgmodelstd = self.fitFuncStd(x_3d, fitParamsBg[0],fitParamsBg[1],fitParamsBg[2])
                    #return (a*x[0]*x[1]+b*x[2]+c)

                    #SET VALUES TO FIELDS AND CALL DEPENDENCIES
                    self.psfEdit.setValue(psf)

                    self.integrationtimeEdit.setValue(integrationtime)
                    self.photonrateEdit.setValue(photonrate)
                    self.photonratestdEdit.setValue(photonratestd)
                    self.photonslopeEdit.setValue(photonrate/laserpower)
                    self.photonslopeStdEdit.setValue(photonratestd/laserpower)

                    #SET NOISE AND FRAME
                    #self.backgroundrateEdit.setValue(bgrate)
                    self.BgoffsetEdit.setValue(bgmu-bgmodel)
                    self.BgStdoffsetEdit.setValue(bgstd-bgmodelstd)

                    self.imagerconcentrationEdit.setValue(imagerconcentration)
                    self.laserpowerEdit.setValue(laserpower)





class CalibrationDialog(QtGui.QDialog):
    def __init__(self, parent = None):
        super(CalibrationDialog, self).__init__(parent)

        layout = QtGui.QVBoxLayout(self)

        self.table = QtGui.QTableWidget()
        tableitem = QtGui.QTableWidgetItem()
        self.table.setWindowTitle('Noise Model Calibration')
        self.setWindowTitle('Noise Model Calibration')
        self.resize(800, 400)


        layout.addWidget(self.table)

        #ADD BUTTONS:
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


    def evalTif(self):
        counter = 0
        for element in self.tifFiles:
            counter = counter+1
            self.pbar.setValue((counter-1)/self.tifCounter*100)
            print('Current Dataset: '+str(counter)+' of ' +str(self.tifCounter))
            movie, info = _io.load_movie(element)
            print(movie.shape)
            movie = movie[0:100,:,:]
            print(movie.shape)
            self.table.setItem(counter-1,4, QtGui.QTableWidgetItem(str(_np.mean(movie))))
            self.table.setItem(counter-1,5, QtGui.QTableWidgetItem(str(_np.std(movie))))
            print(_np.mean(movie))
            print(_np.std(movie))
        self.pbar.setValue(100)


    def loadTif(self):

        self.path = QtGui.QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.path:

            self.tifCounter = len(_glob.glob1(self.path,"*.tif"))
            self.tifFiles = _glob.glob(os.path.join(self.path, "*.tif"))

            self.table.setRowCount(int(self.tifCounter))
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(('FileName,Imager concentration[nM],Integration time [ms],Laserpower,Mean,Std').split(','))

            for i in range(0,self.tifCounter):
                self.table.setItem(i,0, QtGui.QTableWidgetItem(self.tifFiles[i]))


    def changeComb(self, indexval):

        sender = self.sender()
        comboval = sender.currentIndex()
        if comboval == 0:
            self.table.setItem(indexval,2, QtGui.QTableWidgetItem(''))
            self.table.setItem(indexval,3, QtGui.QTableWidgetItem(''))
        else:
            self.table.setItem(indexval,2, QtGui.QTableWidgetItem(self.ImagersShort[comboval]))
            self.table.setItem(indexval,3, QtGui.QTableWidgetItem(self.ImagersLong[comboval]))

    def readoutTable(self):
        tableshort = dict()
        tablelong = dict()
        maxcolor = 15
        for i in range(0,maxcolor-1):
            try:
                tableshort[i] = self.table.item(i,2).text()
                if tableshort[i] == '':
                    tableshort[i] = 'None'
            except AttributeError:
                tableshort[i] = 'None'

            try:
                tablelong[i] = self.table.item(i,3).text()
                if tablelong[i] == '':
                    tablelong[i] = 'None'
            except AttributeError:
                tablelong[i] = 'None'
        return tablelong, tableshort

        #print(tablelong)
        #print(tableshort)

    # get current date and time from the dialog
    def evalTable(self):
        conc = []
        time = []
        las = []
        bg = []
        bgstd = []
        for i in range(0,self.tifCounter):
            conc.append(float(self.table.item(i,1).text()))
            time.append(float(self.table.item(i,2).text()))
            las.append(float(self.table.item(i,3).text()))
            bg.append(float(self.table.item(i,4).text()))
            bgstd.append(float(self.table.item(i,5).text()))


        return bg,bgstd, las, time, conc

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def setExt(parent = None):
        dialog = CalibrationDialog(parent)
        result = dialog.exec_()
        bg,bgstd, las, time, conc = dialog.evalTable()
        return (bg,bgstd, las, time, conc, result == QDialog.Accepted)


def main():

    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

    def excepthook(type, value, tback):
        message = ''.join(traceback.format_exception(type, value, tback))
        errorbox = QtGui.QMessageBox.critical(window, 'An error occured', message)
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)
    sys.excepthook = excepthook


if __name__ == '__main__':
    main()
