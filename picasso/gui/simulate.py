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
from .. import io, simulate
import random
import numpy as _np


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
PSF_DEFAULT = 0.84
PHOTONRATE_DEFAULT = 56
PHOTONRATESTD_DEFAULT = 32
PHOTONBUDGET_DEFAULT = 1500000
#NOISE MODEL
BACKGROUNDRATE_DEFAULT = 1050
BACKGROUNDBASE_DEFAULT = 286
BACKGROUNDRATESTD_DEFAULT = 80
BACKGROUNDBASESTD_DEFAULT = 7
#STRUCTURE
STRUCTURE1_DEFAULT = 3
STRUCTURE2_DEFAULT = 4
STRUCTURE3_DEFAULT = '20,20'
STRUCTUREXX_DEFAULT = '0,20,40,60,0,20,40,60,0,20,40,60'
STRUCTUREYY_DEFAULT = '0,20,40,0,20,40,0,20,40,0,20,40'
STRUCTUREEX_DEFAULT = '1,1,1,1,1,1,1,1,1,1,1,1'
STRUCTURENO_DEFAULT = 9
STRUCTUREFRAME_DEFAULT = 6




class Example(QtGui.QMainWindow):

    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):



        # DEFINE LABELS
        self.spacing = 5

        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(self.spacing)

        # CAMERA PARAMETERS index = 1, cgrid

        camera_groupbox = QtGui.QGroupBox('Camera Parameters')
        cgrid = QtGui.QGridLayout(camera_groupbox)


        self.cindex = 1;
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


        cgrid.addWidget(camerasize,self.cindex,0)
        cgrid.addWidget(self.camerasizeEdit,self.cindex,1)
        cgrid.addWidget(QtGui.QLabel('Px'),self.cindex,2)
        cgrid.addWidget(integrationtime,self.cindex+1,0)
        cgrid.addWidget(self.integrationtimeEdit,self.cindex+1,1)
        cgrid.addWidget(QtGui.QLabel('ms'),self.cindex+1,2)
        cgrid.addWidget(frames,self.cindex+2,0)
        cgrid.addWidget(self.framesEdit,self.cindex+2,1)
        cgrid.addWidget(totaltime,self.cindex+3,0)
        cgrid.addWidget(self.totaltimeEdit,self.cindex+3,1)
        cgrid.addWidget(QtGui.QLabel('min'),self.cindex+3,2)
        cgrid.addWidget(pixelsize,self.cindex+4,0)
        cgrid.addWidget(self.pixelsizeEdit,self.cindex+4,1)
        cgrid.addWidget(QtGui.QLabel('nm'),self.cindex+4,2)

        #PAINT PARAMETERS, pgrid
        paint_groupbox = QtGui.QGroupBox('PAINT Parameters')
        pgrid = QtGui.QGridLayout(paint_groupbox)

        self.pindex = 1
        kon = QtGui.QLabel('K_On')
        imagerconcentration = QtGui.QLabel('Imager Concentration')
        taud = QtGui.QLabel('Tau D')
        taub = QtGui.QLabel('Tau B')

        self.konEdit = QtGui.QDoubleSpinBox()
        self.konEdit.setRange(1,10000000)
        self.konEdit.setDecimals(0)
        self.konEdit.setSingleStep(100000)
        self.imagerconcentrationEdit = QtGui.QDoubleSpinBox()
        self.imagerconcentrationEdit.setRange(1,1000)
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

        pgrid.addWidget(kon,self.pindex,0)
        pgrid.addWidget(self.konEdit,self.pindex,1)
        pgrid.addWidget(imagerconcentration,self.pindex+1,0)
        pgrid.addWidget(self.imagerconcentrationEdit,self.pindex+1,1)
        pgrid.addWidget(QtGui.QLabel('nM'),self.pindex+1,2)
        pgrid.addWidget(taud,self.pindex+2,0)
        pgrid.addWidget(self.taudEdit,self.pindex+2,1)
        pgrid.addWidget(QtGui.QLabel('ms'),self.pindex+2,2)
        pgrid.addWidget(taub,self.pindex+3,0)
        pgrid.addWidget(self.taubEdit,self.pindex+3,1)
        pgrid.addWidget(QtGui.QLabel('ms'),self.pindex+3,2)

        #IMAGER Parameters, igrid
        imager_groupbox = QtGui.QGroupBox('Imager Parameters')
        igrid = QtGui.QGridLayout(imager_groupbox)

        iindex = 1
        psf = QtGui.QLabel('PSF')
        psf_fwhm = QtGui.QLabel('PSF(FWHM)')
        photonrate = QtGui.QLabel('Photonrate')
        photonsframe = QtGui.QLabel('Photons (Frame)')
        photonratestd = QtGui.QLabel('Photonrate Std')
        photonstdframe = QtGui.QLabel('Photonrate Std (Frame)')
        photonbudget = QtGui.QLabel('Photonbudget')

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

        self.psfEdit.setValue(PSF_DEFAULT)
        self.photonrateEdit.setValue(PHOTONRATE_DEFAULT)
        self.photonratestdEdit.setValue(PHOTONRATESTD_DEFAULT)
        self.photonbudgetEdit.setValue(PHOTONBUDGET_DEFAULT)

        self.psfEdit.valueChanged.connect(self.changePSF)

        self.photonrateEdit.valueChanged.connect(self.changeImager)
        self.photonratestdEdit.valueChanged.connect(self.changeImager)

        igrid.addWidget(psf,iindex,0)
        igrid.addWidget(self.psfEdit,iindex,1)
        igrid.addWidget(QtGui.QLabel('Px'),iindex,2)
        igrid.addWidget(psf_fwhm,iindex+1,0)
        igrid.addWidget(self.psf_fwhmEdit,iindex+1,1)
        igrid.addWidget(QtGui.QLabel('nm'),iindex+1,2)
        igrid.addWidget(photonrate,iindex+2,0)
        igrid.addWidget(self.photonrateEdit,iindex+2,1)
        igrid.addWidget(QtGui.QLabel('Photons/ms'),iindex+2,2)
        igrid.addWidget(photonsframe,iindex+3,0)
        igrid.addWidget(self.photonsframeEdit,iindex+3,1)
        igrid.addWidget(QtGui.QLabel('Photons'),iindex+3,2)
        igrid.addWidget(photonratestd,iindex+4,0)
        igrid.addWidget(self.photonratestdEdit,iindex+4,1)
        igrid.addWidget(QtGui.QLabel('Photons/ms'),iindex+4,2)
        igrid.addWidget(photonstdframe,iindex+5,0)
        igrid.addWidget(self.photonstdframeEdit,iindex+5,1)
        igrid.addWidget(QtGui.QLabel('Photons'),iindex+5,2)
        igrid.addWidget(photonbudget,iindex+6,0)
        igrid.addWidget(self.photonbudgetEdit,iindex+6,1)
        igrid.addWidget(QtGui.QLabel('Photons'),iindex+6,2)

         #NOISE MODEL, ngrid

        noise_groupbox = QtGui.QGroupBox('Noise Model')
        ngrid = QtGui.QGridLayout(noise_groupbox)

        self.nindex = 1
        backgroundrate = QtGui.QLabel('Backgroundrate')
        backgroundbase = QtGui.QLabel('Background Base')
        backgroundframe = QtGui.QLabel('Background (Frame)')
        backgroundratestd = QtGui.QLabel('Backgroundrate Std')
        backgroundratestdbase = QtGui.QLabel('Backgroundrate Std Base')
        noiseLabel = QtGui.QLabel('Noise (Frame)')

        self.backgroundrateEdit = QtGui.QDoubleSpinBox()
        self.backgroundrateEdit.setRange(0,100000)
        self.backgroundrateEdit.setDecimals(0)
        self.backgroundbaseEdit = QtGui.QDoubleSpinBox()
        self.backgroundbaseEdit.setRange(0,100000)
        self.backgroundbaseEdit.setDecimals(0)
        self.backgroundframeEdit = QtGui.QLabel()
        self.backgroundratestdEdit = QtGui.QDoubleSpinBox()
        self.backgroundratestdEdit.setRange(0,10000)
        self.backgroundratestdEdit.setDecimals(0)
        self.backgroundratestdbaseEdit = QtGui.QDoubleSpinBox()
        self.backgroundratestdbaseEdit.setRange(0,1000)
        self.backgroundratestdbaseEdit.setDecimals(0)
        self.noiseEdit = QtGui.QLabel()

        self.backgroundrateEdit.setValue(BACKGROUNDRATE_DEFAULT)
        self.backgroundbaseEdit.setValue(BACKGROUNDBASE_DEFAULT)
        self.backgroundratestdEdit.setValue(BACKGROUNDRATESTD_DEFAULT)
        self.backgroundratestdbaseEdit.setValue(BACKGROUNDBASESTD_DEFAULT)

        self.backgroundrateEdit.valueChanged.connect(self.changeNoise)
        self.backgroundbaseEdit.valueChanged.connect(self.changeNoise)
        self.backgroundratestdEdit.valueChanged.connect(self.changeNoise)
        self.backgroundratestdbaseEdit.valueChanged.connect(self.changeNoise)


        ngrid.addWidget(backgroundrate,self.nindex,0)
        ngrid.addWidget(self.backgroundrateEdit,self.nindex,1)
        ngrid.addWidget(QtGui.QLabel('Photons/s'),self.nindex,2)
        ngrid.addWidget(backgroundbase,self.nindex+1,0)
        ngrid.addWidget(self.backgroundbaseEdit,self.nindex+1,1)
        ngrid.addWidget(QtGui.QLabel('Photons'),self.nindex+1,2)
        ngrid.addWidget(backgroundframe,self.nindex+2,0)
        ngrid.addWidget(self.backgroundframeEdit,self.nindex+2,1)
        ngrid.addWidget(QtGui.QLabel('Photons'),self.nindex+2,2)
        ngrid.addWidget(backgroundratestd,self.nindex+3,0)
        ngrid.addWidget(self.backgroundratestdEdit,self.nindex+3,1)
        ngrid.addWidget(QtGui.QLabel('Photons/s'),self.nindex+3,2)
        ngrid.addWidget(backgroundratestdbase,self.nindex+4,0)
        ngrid.addWidget(self.backgroundratestdbaseEdit,self.nindex+4,1)
        ngrid.addWidget(QtGui.QLabel('Photons'),self.nindex+4,2)
        ngrid.addWidget(noiseLabel,self.nindex+5,0)
        ngrid.addWidget(self.noiseEdit,self.nindex+5,1)
        ngrid.addWidget(QtGui.QLabel('Photons'),self.nindex+5,2)

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

        self.structurexxEdit.textChanged.connect(self.plotStructure)
        self.structureyyEdit.textChanged.connect(self.plotStructure)
        self.structureexEdit.textChanged.connect(self.plotStructure)

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


        # POSITION PREVIEW WINDOW

        posgrid = QtGui.QGridLayout()

#
        # STRUCTURE PREVIEW WINDOW

        strgrid = QtGui.QGridLayout()

        'BUTTONS L'
        designButton = QtGui.QPushButton("Design Structure")
        generateButton = QtGui.QPushButton("Generate Positions")
        loadPositions = QtGui.QPushButton("Load Positions")

        generateButton.clicked.connect(self.generatePositions)
        btngridL = QtGui.QGridLayout()

        btngridL.addWidget(designButton)
        btngridL.addWidget(generateButton)
        btngridL.addWidget(loadPositions)


        'BUTTONS R'
        simulateButton = QtGui.QPushButton("Simulate Data")
        quitButton = QtGui.QPushButton('Quit', self)
        quitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        quitButton.resize(quitButton.sizeHint())

        loadButton = QtGui.QPushButton("Load Settings")
        btngridR = QtGui.QGridLayout()
        btngridR.addWidget(loadButton)
        btngridR.addWidget(simulateButton)
        btngridR.addWidget(quitButton)

        simulateButton.clicked.connect(self.saveDialog)

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
        self.button = QtGui.QPushButton('Plot')
        self.button.clicked.connect(self.plot)
        posgrid.addWidget(self.canvas1)
        strgrid.addWidget(self.canvas2)
        #strgrid.addWidget(self.button)

        # DEFINE ARRANGEMENT
        self.grid.addLayout(posgrid,1,0)
        self.grid.addLayout(strgrid,1,1)
        self.grid.addWidget(structure_groupbox,2,0,2,1)
        self.grid.addWidget(camera_groupbox,1,2)
        self.grid.addWidget(paint_groupbox,2,1)
        self.grid.addWidget(imager_groupbox,3,1)
        self.grid.addWidget(noise_groupbox,2,2)
        self.grid.addLayout(btngridL,4,0)
        self.grid.addLayout(btngridR,4,2)



        #STATUSBAR etc

        self.statusBar().showMessage('Ready')
        #self.setCentralWidget(self.grid)

        #self.setLayout(self.grid)

        mainWidget = QtGui.QWidget()
        mainWidget.setLayout(self.grid)
        self.setCentralWidget(mainWidget)
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('simulate')
        #CALL FUNCTIONS
        self.plotStructure()
        self.generatePositions()

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

        backgroundrate = self.backgroundrateEdit.value()
        backgroundbase = self.backgroundbaseEdit.value()
        backgroundframe = round(backgroundrate*itime/1000+backgroundbase)

        noiserate = self.backgroundratestdEdit.value()
        noisebase = self.backgroundratestdbaseEdit.value()
        noiseframe = round(noiserate*itime/1000+noisebase)

        self.backgroundframeEdit.setText(str(backgroundframe))
        self.noiseEdit.setText(str(noiseframe))

    def changePaint(self):
        kon = self.konEdit.value()
        imagerconcentration = self.imagerconcentrationEdit.value()
        taud = round(1/(kon*imagerconcentration*1/10**9)*1000)
        self.taudEdit.setText(str(taud))

    def changePSF(self):
        psf = self.psfEdit.value()
        pixelsize = self.pixelsizeEdit.value()
        psf_fwhm = round(psf*pixelsize*2.355)
        self.psf_fwhmEdit.setText(str(psf_fwhm))

    def changeImager(self):
        itime = self.integrationtimeEdit.value()
        photonrate = self.photonrateEdit.value()
        photonratestd = self.photonratestdEdit.value()

        photonsframe = round(photonrate*itime)
        photonsframestd = round(photonratestd*itime)

        self.photonsframeEdit.setText(str(photonsframe))
        self.photonstdframeEdit.setText(str(photonsframestd))

    def changeNoise(self):
        itime = self.integrationtimeEdit.value()
        backgroundrate = self.backgroundrateEdit.value()
        backgroundbase = self.backgroundbaseEdit.value()
        backgroundframe = round(backgroundrate*itime/1000+backgroundbase)

        noiserate = self.backgroundratestdEdit.value()
        noisebase = self.backgroundratestdbaseEdit.value()
        noiseframe = round(noiserate*itime/1000+noisebase)

        self.backgroundframeEdit.setText(str(backgroundframe))
        self.noiseEdit.setText(str(noiseframe))


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
            self.plotStructure()
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

    def simulate(self,fileNameOld):

        #READ IN PARAMETERS

        #STRUCTURE
        structureNo = self.structurenoEdit.value()
        structureFrame = self.structureframeEdit.value()
        structureIncorporation = self.structureIncorporationEdit.value()
        structureArrangement = self.structurerandomEdit.checkState()
        structureOrientation = self.structurerandomOrientationEdit.checkState()

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

        #CAMERA PARAMETERS
        imagesize = self.camerasizeEdit.value()
        itime = self.integrationtimeEdit.value()
        frames = self.framesEdit.value()
        pixelsize = self.pixelsizeEdit.value()

        #NOISE MODEL
        background = int(self.backgroundframeEdit.text())
        noise = int(self.noiseEdit.text())

        backgroundrate = self.backgroundrateEdit.value()
        backgroundbase = self.backgroundbaseEdit.value()
        backgroundratestd = self.backgroundratestdEdit.value()
        backgroundbasestd = self.backgroundratestdbaseEdit.value()

        structurexx,structureyy,structureex = self.readStructure()
        noexchangecolors = len(set(structureex))
        exchangecolors = list(set(structureex))


        #self.progress_dialog = QtGui.QProgressDialog(text, 'Cancel', 0, n_movies, self)
        #progress_bar = QtGui.QProgressBar(self.progress_dialog)
        #progress_bar.setTextVisible(False)
        #self.progress_dialog.setBar(progress_bar)
        #self.progress_dialog.setMaximum(n_movies)
        #self.progress_dialog.setWindowTitle('Picasso: ToRaw')
        #self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        #self.progress_dialog.canceled.connect(self.cancel)
        #self.progress_dialog.closeEvent = self.cancel
        #self.worker = Worker(movie_groups)
        #self.worker.progressMade.connect(self.update_progress)
        #self.worker.finished.connect(self.on_finished)
        #self.worker.start()
        #self.progress_dialog.show()

        self.statusBar().showMessage('Simulation started')

        struct = self.newstruct

        for i in range(0,noexchangecolors):
            if noexchangecolors > 1:
                fileName = io.multiple_filenames(fileNameOld, i)
                partstruct = struct[:,struct[2,:]==exchangecolors[i]]
            else:
                fileName = fileNameOld
                partstruct = struct

            self.statusBar().showMessage('Distributing Photons ...')
            photondist = simulate.distphotons(partstruct,itime,frames,taud,taub,photonrate,photonratestd,photonbudget)
            self.statusBar().showMessage('Distributing Photons ... Complete.')

            self.statusBar().showMessage('Converting to Image ... ')

            movie = _np.zeros(shape=(frames,imagesize,imagesize), dtype='<u2')
            for runner in range(0,frames):
                movie[runner,:,:]=simulate.convertMovie(runner,photondist,partstruct,imagesize,frames,psf,photonrate,background, noise)
                outputmsg = 'Converting to Image ... ' + str(_np.round(runner/frames*1000)/10) +' %'
                self.statusBar().showMessage(outputmsg)
            self.statusBar().showMessage('Converting to Image ... Complete.')
            self.statusBar().showMessage('Saving Image ...')
            placeholder = 0

            info = {'Generated by':'Picasso simulate',
                    'Byte Order': '<',
                    'Camera': 'Simulation',
                    'Data Type': movie.dtype.name,
                    'Frames': frames,
                    'Structure.Frame':structureNo,
                    'Structure.Number':structureFrame,
                    'Structure.Incorporation':structureIncorporation,
                    'Structure.Arrangement':structureArrangement,
                    'Structure.Orientation':structureOrientation,
                    'PAINT.k_on':kon,
                    'PAINT.imager':imagerconcentration,
                    'PAINT.taub':taub,
                    'Imager.PSF':psf,
                    'Imager.Photonrate':photonrate,
                    'Imager.Photonrate Std':photonratestd,
                    'Imager.Photonbudget ':photonbudget,
                    'Camera.Image Size':imagesize,
                    'Camera.Integration Time':itime,
                    'Camera.Frames':frames,
                    'Camera.Pixelsize':pixelsize,
                    'Noise.Backgroundrate':backgroundrate,
                    'Noise.Background Base':backgroundbase,
                    'Noise.Backgroundrate Std':backgroundratestd,
                    'Noise.Backgroundrate Std Base':backgroundbasestd,
                    'Height': imagesize,
                    'Width': imagesize}

            simulate.saveMovie(fileName,movie,info)

            self.statusBar().showMessage('All computations finished.')



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

            for i in range(0,noexchangecolors):
                plotxx = []
                plotyy = []
                for j in range(0,len(structureex)):
                    if structureex[j] == exchangecolors[i]:
                        plotxx.append(structurexx[j])
                        plotyy.append(structureyy[j])
                ax1.plot(plotxx,plotyy,'o')
                #plotxx = structurexx(structureex == exchangecolors[i])
                #print(plotxx)


            #ax1.plot(structurexx,structureyy,'o')
            #ax1.axis('equal')
            #ax1.set_axis_bgcolor((1, 1, 1))
            #ax1.set_xlabel('[Nm]')

            distx = round(1/10*(max(structurexx)-min(structurexx)))
            disty = round(1/10*(max(structureyy)-min(structureyy)))

            ax1.axes.set_xlim((min(structurexx)-distx,max(structurexx)+distx))
            ax1.axes.set_ylim((min(structureyy)-disty,max(structureyy)+disty))
            self.canvas2.draw()

    def generatePositions(self):
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

        #for i in range(0,number):
        #    struct1 = self.newstruct[:,self.newstruct[3,:]==i]
        #    plt.plot(struct1[0,:],struct1[1,:],'o')

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
        #ax1.plot([(frame,frame,imageSize-frame,imageSize-frame)],[(frame,imageSize-frame,imageSize-frame,frame)])
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



    def plot(self):
        data = [random.random() for i in range(10)]
        ax1 = self.figure1.add_subplot(111)
        ax1.hold(False)
        ax1.plot(data, '*-')
        self.canvas1.draw()

        ax2 = self.figure2.add_subplot(111)
        ax2.hold(False)
        ax2.plot(data, '*-')
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

            self.statusBar().showMessage('All Simulations complete.')

class Worker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        for i, (basename, paths) in enumerate(self.movie_groups.items()):
            io.to_raw_combined(basename, paths)
            self.progressMade.emit(i+1)
        self.finished.emit(i)


def main():

    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
