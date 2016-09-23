"""
    picasso.design-gui
    ~~~~~~~~~~~~~~~~

    GUI for design :
    Design rectangular rothemund origami

    :author: Maximilian Thomas Strauss, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, Max Planck Institute of Biochemistry
"""

import sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import *
from PyQt4.QtGui import QDialog, QVBoxLayout, QDialogButtonBox, QDateTimeEdit, QApplication
from PyQt4.QtCore import Qt, QDateTime
from math import sqrt
import numpy as _np
from .. import io as _io, design
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import csv


_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
BaseSequencesFile = os.path.join(_this_directory, '..', 'base_sequences.csv')

def readPlate(filename):
    File = open(filename)
    Reader = csv.reader(File)
    data = list(Reader)
    return data

def savePlate(filename,data):
    with open(filename, 'w', newline='') as csvfile:
        Writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for j in range(0,len(data)):
            exportdata = data[j]
            for i in range(0,len(exportdata)):
                Writer.writerow(exportdata[i])
        #Writer.writerow(['Spam'] * 5 + ['Baked Beans'])
        #Writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

def plotPlate(selection,platename):
    #DIMENSIONS OF 96 WELL PLATES ARE 9mm
    radius = 4.5
    radiusc = 4
    circles = dict()
    rows = 8
    cols = 12
    colsStr = ['1','2','3','4','5','6','7','8','9','10','11','12']
    rowsStr = ['A','B','C','D','E','F','G','H']
    rowsStr = rowsStr[::-1]


    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    ax.cla()
    plt.axis('equal')
    for xcord in range(0,cols):
        for ycord in range(0,rows):
            string = rowsStr[ycord]+colsStr[xcord]
            xpos = xcord*radius*2+radius
            ypos = ycord*radius*2+radius
            if string in selection:
                circle = plt.Circle((xpos,ypos),radiusc, facecolor='black',edgecolor='black')
                ax.text(xpos, ypos, string, fontsize=10, color = 'white',horizontalalignment='center',
                        verticalalignment='center')
            else:
                circle = plt.Circle((xpos,ypos),radiusc, facecolor='white',edgecolor='black')
            ax.add_artist(circle)
        # INNER RECTANLGE
    ax.add_patch(patches.Rectangle((0, 0),cols*2*radius,rows*2*radius,fill=False))
    # OUTER RECTANGLE
    ax.add_patch(patches.Rectangle((0-2*radius, 0),cols*2*radius+2*radius,rows*2*radius+2*radius,fill=False))

    #ADD ROWS AND COLUMNS
    for xcord in range(0,cols):
        ax.text(xcord*2*radius+radius, rows*2*radius+radius, colsStr[xcord], fontsize=10, color = 'black',horizontalalignment='center',
                        verticalalignment='center')
    for ycord in range(0,rows):
        ax.text(-radius, ycord*2*radius+radius, rowsStr[ycord], fontsize=10, color = 'black',horizontalalignment='center',
                        verticalalignment='center')
    ax.set_xlim([-2*radius,cols*2*radius])
    ax.set_ylim([0,rows*2*radius+2*radius])
    print(fig.get_size_inches())
    plt.title(platename+' - '+str(len(selection))+' Staples')
    ax.set_xticks([])
    ax.set_yticks([])
    print(fig.get_size_inches())
    inch = 25.4
    xsize = 13*2*radius/inch #Dont ask why
    ysize = 9*2*radius/inch

    fig.set_size_inches(xsize,ysize)
    print(fig.get_size_inches())
    #plt.show()
    return fig

BasePlate = readPlate(BaseSequencesFile)


#LIST OF STANDARD setSequences

allSeqShort = ('None,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10')
allSeqLong = (' ,TTATACATCTA,TTATCTACATA,TTTCTTCATTA,TTATGAATCTA,TTTCAATGTAT,TTTTAGGTAAA,TTAATTGAGTA,TTATGTTAATG,TTAATTAGGAT,TTATAATGGAT')
TABLESHORT_DEFAULT = ['None','None','None','None','None','None','None']
TABLELONG_DEFAULT = ['None','None','None','None','None','None','None']

HEX_SIDE_HALF = 20
HEX_SCALE = 1
HEX_PEN = QtGui.QPen(QtGui.QBrush(QtGui.QColor('black')), 2)

# ORIGAMI DEFINITION
rows = 12
rowIndex = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
columns = 16
columnIndex= range(1,columns+1)
ORIGAMI_SITES = [(x, y) for x in range(rows) for y in range(columns)]

ind2remove = [(1,2),(2,2),(8,2),(9,2),(1,6),(2,6),(8,6),(9,6),(1,10),(2,10),(8,10),(9,10),(1,14),(2,14),(8,14),(9,14)]
for element1 in ind2remove:
    for element2 in ORIGAMI_SITES:
        if element1 == element2:
            ORIGAMI_SITES.remove(element2)

# ADD COLOR PALETTE
maxbinding = max(ORIGAMI_SITES)
COLOR_SITES = [(0,maxbinding[1]+3),(2,maxbinding[1]+3),(3,maxbinding[1]+3),(4,maxbinding[1]+3),(5,maxbinding[1]+3),(6,maxbinding[1]+3),(7,maxbinding[1]+3),(8,maxbinding[1]+3),(10,maxbinding[1]+3)]


BINDING_SITES = ORIGAMI_SITES+COLOR_SITES

# WRITE ALL COLORS IN A DICTIONARY
allcolors = dict()
allcolors[0] = QtGui.QColor(205, 205, 205, 255) #DEFAULT, GREY
allcolors[1] = QtGui.QColor(166,206,227, 255)
allcolors[2] = QtGui.QColor(31,120,180, 255)
allcolors[3] = QtGui.QColor(178,223,138, 255)
allcolors[4] = QtGui.QColor(51,160,44, 255)
allcolors[5] = QtGui.QColor(251,154,153, 255)
allcolors[6] = QtGui.QColor(227,26,28, 255)
allcolors[7] = QtGui.QColor(0,0,0, 255) # BLACK
allcolors[8] = whitecolor = QtGui.QColor('white') # WHITE
defaultcolor = allcolors[0]
maxcolor = 8


def indextoHex(y,x):
        hex_center_x = x*1.5*HEX_SIDE_HALF
        if _np.mod(x,2)==0:
            hex_center_y = -y*sqrt(3)/2*HEX_SIDE_HALF*2
        else:
            hex_center_y = -(y+0.5)*sqrt(3)/2*HEX_SIDE_HALF*2
        return hex_center_x,hex_center_y

def indextoStr(x,y):

    rowStr = rowIndex[y]
    colStr = columnIndex[x]
    strIndex = (rowStr,colStr)
    return strIndex

class PipettingDialog(QtGui.QDialog):
    def __init__(self, parent = None):
        super(PipettingDialog, self).__init__(parent)
        layout = QtGui.QVBoxLayout(self)
        self.setWindowTitle('Pipetting Dialog')

        self.loadButton = QtGui.QPushButton("Select Folder")
        self.folderEdit = QtGui.QLabel('')
        self.csvCounter = QtGui.QLabel('')
        self.plateCounter = QtGui.QLabel('')
        self.uniqueCounter = QtGui.QLabel('')

        self.fulllist = []


        self.loadButton.clicked.connect(self.loadFolder)

        layout.addWidget(self.loadButton)
        layout.addWidget(self.folderEdit)
        layout.addWidget(self.csvCounter)
        layout.addWidget(self.plateCounter)
        layout.addWidget(self.uniqueCounter)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def loadFolder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        oldpath = os.getcwd()
        if path:
            self.folderEdit.setText(path)
            csvFiles = glob.glob1   (path,"*.csv")
            csvCounter = len(csvFiles)
            self.csvCounter.setText('A total of '+str(csvCounter)+ ' *.csv files detected.')
            platelist = []
            sequencelist = []
            for element in csvFiles:
                #print(element)
                os.chdir(path)
                data = readPlate(element)
                for i in range(0,len(data)):
                    platelist.append(data[i][0])
                    sequencelist.append(data[i][3])
                    self.fulllist.append(data[i][0:4])

            #print(sorted(set(sequencelist)))
            #print(sorted(set(platelist)))
            self.plateCounter.setText('A total of '+str(len(set(platelist))-1)+ '  plates detected.')
            self.uniqueCounter.setText('A total of '+str(len(set(sequencelist))-2)+ '  unique sequences detected.')
            #print(sorted(set(fulllist)))
            os.chdir(oldpath)
            # SEQUENCE AND ' '(EMPTY)



    # get current date and time from the dialog
    def getfulllist(self):

        fulllist = self.fulllist
        return fulllist

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getSchemes(parent = None):
        dialog = PipettingDialog(parent)
        result = dialog.exec_()
        fulllist= dialog.getfulllist()

        return (fulllist, result == QDialog.Accepted)



class SeqDialog(QtGui.QDialog):
    def __init__(self, parent = None):
        super(SeqDialog, self).__init__(parent)

        layout = QtGui.QVBoxLayout(self)

        self.table = QtGui.QTableWidget()
        tableitem = QtGui.QTableWidgetItem()
        self.table.setWindowTitle('Extension Table')
        self.setWindowTitle('Extenstion Table')
        self.resize(500, 285)
        self.table.setRowCount(maxcolor-1)
        self.table.setColumnCount(4)

        #SET LABELS ETC
        self.table.setHorizontalHeaderLabels(('Color,Pre,Shortname,Sequence').split(','))

        for i in range(0,maxcolor-1):
            self.table.setItem(i,0, QtGui.QTableWidgetItem("Ext "+str(i+1)))
            self.table.item(i, 0).setBackground(allcolors[i+1])

        self.ImagersShort = allSeqShort.split(",")
        self.ImagersLong = allSeqLong.split(",")

        comb = dict()

        for i in range(0,maxcolor-1):
            comb[i] = QtGui.QComboBox()

        for element in self.ImagersShort:
            for i in range(0,maxcolor-1):
                comb[i].addItem(element)

        for i in range(0,maxcolor-1):
            self.table.setCellWidget(i, 1, comb[i])
            comb[i].currentIndexChanged.connect(lambda state, indexval=i: self.changeComb(indexval))
            #comb[i].currentIndexChanged.connect(lambda: self.changeComb(1))

        #comb1.currentIndexChanged.connect(self.changeComb(1))

        layout.addWidget(self.table)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

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

        tablelong, tableshort = self.readoutTable()
        return tablelong, tableshort

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def setExt(parent = None):
        dialog = SeqDialog(parent)
        result = dialog.exec_()
        tablelong, tableshort = dialog.evalTable()
        return (tablelong, tableshort, result == QDialog.Accepted)

class PlateDialog(QtGui.QDialog):
    def __init__(self, parent = None):
        super(PlateDialog, self).__init__(parent)
        layout = QtGui.QVBoxLayout(self)
        self.info = QtGui.QLabel('Please make selection:  ')
        self.radio1 = QtGui.QRadioButton('Export only sequences in current design. (176 staples in 2 plates)')
        self.radio2 = QtGui.QRadioButton('Export full plates for all sequences used (176 * Number of Sequences)')

        self.setWindowTitle('Plate Export')
        layout.addWidget(self.info)
        layout.addWidget(self.radio1)
        layout.addWidget(self.radio2)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)


    # get current date and time from the dialog
    def evalSelection(self):
        if self.radio1.isChecked():
            selection = 1
        elif self.radio2.isChecked():
            selection = 2
        else:
            selection = 0
        return selection

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getPlates(parent = None):
        dialog = PlateDialog(parent)
        result = dialog.exec_()
        selection = dialog.evalSelection()
        return (selection, result == QDialog.Accepted)




class BindingSiteItem(QtGui.QGraphicsPolygonItem):

    def __init__(self, y, x):

        hex_center_x, hex_center_y = indextoHex(y,x)

        center = QtCore.QPointF(hex_center_x, hex_center_y)
        points = [HEX_SCALE*HEX_SIDE_HALF * QtCore.QPointF(-1, 0) + center,
                    HEX_SCALE*HEX_SIDE_HALF * QtCore.QPointF(-0.5,sqrt(3)/2)  + center,
                    HEX_SCALE*HEX_SIDE_HALF * QtCore.QPointF(0.5,sqrt(3)/2) + center,
                    HEX_SCALE*HEX_SIDE_HALF * QtCore.QPointF(1,0) + center,
                    HEX_SCALE*HEX_SIDE_HALF * QtCore.QPointF(0.5,-sqrt(3)/2) + center,
                    HEX_SCALE*HEX_SIDE_HALF * QtCore.QPointF(-0.5,-sqrt(3)/2) + center]

        hexagonPointsF = QtGui.QPolygonF(points)
        super().__init__(hexagonPointsF)
        self.setPen(HEX_PEN)
        self.setBrush(defaultcolor) #INITIALIZE ALL HEXAGONS IN GREY


class Scene(QtGui.QGraphicsScene):

    def __init__(self, window):
        super().__init__()
        self.window = window
        for coords in BINDING_SITES:
            self.addItem(BindingSiteItem(*coords))
        self.allcords = []
        self.indices = []

        self.tableshort = TABLESHORT_DEFAULT
        self.tablelong = TABLELONG_DEFAULT

        for coords in BINDING_SITES:
            y = coords[0]
            x = coords[1]
            hex_center_x, hex_center_y = indextoHex(y,x)
            self.allcords.append((hex_center_x*0.125*4/3,2.5-hex_center_y*0.125*2/sqrt(3))) #5nm ORIGAMI GRID, half is 20, should be 2.5 % RETHINK THIS

        #PREPARE FILE FORMAT FOR ORIGAMI DEFINITION
        self.origamicoords = []
        self.origamiindices = []
        for coords in ORIGAMI_SITES:
            y = coords[0]
            x = coords[1]

            self.origamiindices.append((indextoStr(y,x)))
            hex_center_x, hex_center_y = indextoHex(y,x)
            self.origamicoords.append((hex_center_x*0.125,hex_center_y*0.125))

        #print(self.origamiindices)
        # INITIALIZE COLOR PALETTE
        allitems = self.items()
        lenitems = len(allitems)
        bindingitems = len(BINDING_SITES)
        paletteindex = lenitems-bindingitems

        allitems[paletteindex].setBrush(allcolors[1])
        allitems[paletteindex+maxcolor].setBrush(allcolors[0])
        for i in range(1,maxcolor):
            allitems[paletteindex+i].setBrush(allcolors[i])

        self.alllbl = dict() # LABELS FOR COUNTING
        self.alllblseq = dict() # LABELS FOR EXTENSIONS

        labelspacer = 0.5
        yoff = 0.2
        xoff = 0.2
        xofflbl = 1

        for i in range(0,maxcolor):
            self.alllbl[i] = QtGui.QGraphicsTextItem('   ')
            self.alllbl[i].setPos(*(1.5*HEX_SIDE_HALF * (labelspacer+COLOR_SITES[7-paletteindex-i][1]+xoff),-labelspacer-sqrt(3)*HEX_SIDE_HALF * (COLOR_SITES[7-paletteindex-i][0]+yoff)))
            self.addItem(self.alllbl[i])

            self.alllblseq[i] = QtGui.QGraphicsTextItem('   ')
            self.alllblseq[i].setPos(*(1.5*HEX_SIDE_HALF * (labelspacer+COLOR_SITES[7-paletteindex-i][1]+xofflbl),-labelspacer-sqrt(3)*HEX_SIDE_HALF * (COLOR_SITES[7-paletteindex-i][0]+yoff)))
            self.addItem(self.alllblseq[i])

        #MAKE A LABEL FOR THE CURRENTCOLOR
        self.cclabel = QtGui.QGraphicsTextItem('Current Color')
        self.cclabel.setPos(*(1.5*HEX_SIDE_HALF * (labelspacer+COLOR_SITES[8-paletteindex][1]+xofflbl),-labelspacer-sqrt(3)*HEX_SIDE_HALF * (COLOR_SITES[8-paletteindex][0]+yoff)))
        self.addItem(self.cclabel)
        self.evaluateCanvas()

    def mousePressEvent(self, event):
        clicked_item = self.itemAt(event.scenePos(), self.window.view.transform())
        if clicked_item:
            if clicked_item.type() == 5:
                allitems = self.items()
                lenitems = len(allitems)
                bindingitems = len(BINDING_SITES)
                paletteindex = lenitems-bindingitems
                selectedcolor  = allitems[paletteindex].brush().color()

                if clicked_item == allitems[paletteindex]: # DO NOTHING
                    pass
                elif clicked_item == allitems[paletteindex+1]: #
                    allitems[paletteindex].setBrush(allcolors[1])
                    selectedcolor = allcolors[1]
                elif clicked_item == allitems[paletteindex+2]: #
                    allitems[paletteindex].setBrush(allcolors[2])
                    selectedcolor = allcolors[2]
                elif clicked_item == allitems[paletteindex+3]: #
                    allitems[paletteindex].setBrush(allcolors[3])
                    selectedcolor = allcolors[3]
                elif clicked_item == allitems[paletteindex+4]: #
                    allitems[paletteindex].setBrush(allcolors[4])
                    selectedcolor = allcolors[4]
                elif clicked_item == allitems[paletteindex+5]: #
                    selectedcolor = allcolors[5]
                    allitems[paletteindex].setBrush(allcolors[5])
                elif clicked_item == allitems[paletteindex+6]: #
                    allitems[paletteindex].setBrush(allcolors[6])
                    selectedcolor = allcolors[6]
                elif clicked_item == allitems[paletteindex+7]: #
                    allitems[paletteindex].setBrush(allcolors[7])
                    selectedcolor = allcolors[7]
                elif clicked_item == allitems[paletteindex+8]: #
                    allitems[paletteindex].setBrush(QtGui.QBrush(defaultcolor))
                    selectedcolor = defaultcolor;
                else:
                    currentcolor  = clicked_item.brush().color()
                    if currentcolor == selectedcolor:
                        clicked_item.setBrush(defaultcolor) # TURN WHITE AGAIN IF NOT USED
                    else:
                        clicked_item.setBrush(QtGui.QBrush(selectedcolor))
                self.evaluateCanvas()

    def evaluateCanvas(self):
    #READS OUT COLOR VALUES AND MAKES A LIST WITH CORRESPONDING COLORS
        allitems = self.items()
        lenitems = len(allitems)
        bindingitems = len(BINDING_SITES)
        origamiitems = len(ORIGAMI_SITES)
        paletteindex = lenitems-bindingitems+9
        canvascolors = []
        for i in range(0,origamiitems):
            currentcolor = allitems[paletteindex+i].brush().color()
            for j in range(0,maxcolor):
                if currentcolor == allcolors[j]:
                    canvascolors.append(j)
        for i in range(0,maxcolor):
            tocount = i+1
            if i == maxcolor-1:
                tocount = 0
            count = canvascolors.count(tocount)
            if count == 0:
                self.alllbl[i].setPlainText('   ')
            else:
                self.alllbl[i].setPlainText(str(canvascolors.count(tocount)))

        return canvascolors

    def updateExtensions(self,tableshort):
     # Takes a list of tableshort and updates the display
         allitems = self.items()
         lenitems = len(allitems)
         bindingitems = len(BINDING_SITES)
         origamiitems = len(ORIGAMI_SITES)
         paletteindex = lenitems-bindingitems+9
         canvascolors = []

         for i in range(0,maxcolor-1):
             if tableshort[i] == 'None':
                 self.alllblseq[i].setPlainText('   ')
             else:
                 self.alllblseq[i].setPlainText(tableshort[i])

    def saveExtensions(self,tableshort,tablelong):
        self.tableshort = tableshort
        self.tablelong = tablelong

    def clearCanvas(self):
            allitems = self.items()
            lenitems = len(allitems)
            bindingitems = len(BINDING_SITES)
            origamiitems = len(ORIGAMI_SITES)
            paletteindex = lenitems-bindingitems+9

            for i in range(0,origamiitems):
                allitems[paletteindex+i].setBrush(QtGui.QBrush(defaultcolor))

            self.tableshort = TABLESHORT_DEFAULT
            self.tablelong = TABLELONG_DEFAULT
            self.evaluateCanvas()
            self.updateExtensions(self.tableshort)

    def vectorToString(self,x):
        x_arrstr = _np.char.mod('%f', x)
        x_str = ",".join(x_arrstr)
        return x_str

    def vectorToStringInt(self,x):
        x_arrstr = _np.char.mod('%i', x)
        x_str = ",".join(x_arrstr)
        return x_str



    def saveCanvas(self,path):
        canvascolors = self.evaluateCanvas()
        canvascolors = canvascolors[::-1]
        #print(canvascolors)
        structurec = []
        structure = []
        structureInd = []
        for x in range(0,len(canvascolors)):
            structure.append((self.allcords[x][0],self.allcords[x][1],canvascolors[x]))

        for x in range(0,len(canvascolors)):
                structurec.append((self.allcords[x][0],self.allcords[x][1],canvascolors[x]))

        for x in range(0,len(canvascolors)):
                structureInd.append([self.origamiindices[x][0],self.origamiindices[x][1],canvascolors[x]])
        #CALCULATE COORDINATeS FOR SIMULATE
        #1 -Convert to String etc.
        structurex = []
        structurey = []
        structureex = []
        for element in structurec:
            if element[2] != 0:
                structurex.append(element[0])
                structurey.append(element[1])
                structureex.append(element[2])
        structurex = self.vectorToString(structurex)
        structurey = self.vectorToString(structurey)
        structureex = self.vectorToStringInt(structureex)


        info = {'Generated by':'Picasso design',
                'Structure':structureInd,
                'Extensions Short':self.tableshort,
                'Extensions Long':self.tablelong,
                'Structure.StructureX':structurex,
                'Structure.StructureY':structurey,
                'Structure.StructureEx':structureex}
        design.saveInfo(path, info)
        print('Data saved.')

    def loadCanvas(self,path):
        info = _io.load_info(path)
        structure = info[0]['Structure']
        structure = structure[::-1]
        allitems = self.items()
        lenitems = len(allitems)
        bindingitems = len(BINDING_SITES)
        origamiitems = len(ORIGAMI_SITES)
        paletteindex = lenitems-bindingitems+9

        for i in range(0,origamiitems):
            colorindex = structure[i][2]
            allitems[paletteindex+i].setBrush(QtGui.QBrush(allcolors[colorindex]))

        self.evaluateCanvas()
        self.tableshort = info[0]['Extensions Short']
        self.tablelong = info[0]['Extensions Long']
        self.updateExtensions(self.tableshort)

    def preparePlates(self, mode):
        #read the colors of the canvas
        canvascolors = self.evaluateCanvas()
        canvascolors = canvascolors[::-1]

        #get number of plates
        noplates = len(set(canvascolors))
        colors = list(set(canvascolors))
        allplates = dict()

        if mode == 2: # ake full plates for each extension
            for j in range(0,noplates):
                if colors[j] == 0:
                    allplates[j] = design.convertToPlate(readPlate(BaseSequencesFile),'BLK')
                else:
                    ExportPlate = readPlate(BaseSequencesFile)
                    for i in range(0,len(canvascolors)):
                        ExportPlate[1+i][2] = ExportPlate[1+i][2]+' '+self.tablelong[colors[j]-1]
                        ExportPlate[1+i][1] = ExportPlate[1+i][1][:-3]+self.tableshort[colors[j]-1]
                    allplates[j] = design.convertToPlate(ExportPlate,self.tableshort[colors[j]-1])

        elif mode == 1: # only one plate with the modifications
            ExportPlate = readPlate(BaseSequencesFile)
            for i in range(0,len(canvascolors)):
                if canvascolors[i] == 0:
                    pass
                else:
                    ExportPlate[1+i][2] = ExportPlate[1+i][2]+' '+self.tablelong[canvascolors[i]-1]
                    ExportPlate[1+i][1] = ExportPlate[1+i][1][:-3]+self.tableshort[canvascolors[i]-1]
            allplates[0] = design.convertToPlate(ExportPlate,'CUSTOM')

        return allplates


class Window(QtGui.QMainWindow):

    def __init__(self):
        super().__init__()
        self.mainscene = Scene(self)
        self.view = QtGui.QGraphicsView(self.mainscene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setCentralWidget(self.view)
        self.statusBar().showMessage('Ready. Sequences loaded from '+BaseSequencesFile+'.')

    def openDialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open design', filter='*.yaml')
        if path:
            self.mainscene.loadCanvas(path)
            self.statusBar().showMessage('File loaded from: '+path)
        else:
            self.statusBar().showMessage('Filename not specified. File not loaded.')

    def saveDialog(self):
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save design to..', filter='*.yaml')
        if path:
            self.mainscene.saveCanvas(path)
            self.statusBar().showMessage('File saved as: '+path)
        else:
            self.statusBar().showMessage('Filename not specified. Design not saved.')

    def clearDialog(self):
        self.mainscene.clearCanvas()
        self.statusBar().showMessage('Canvas clearead.')

    def takeScreenshot(self):
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save Screenshot to..', filter='*.png')
        if path:
            p = QPixmap.grabWidget(self.view)
            p.save(path, 'png')
            self.statusBar().showMessage('Screenshot saved to: '+path)
        else:
            self.statusBar().showMessage('Filename not specified. Screenshot not saved.')

    def setSeq(self):
        tablelong, tableshort, ok = SeqDialog.setExt()
        if ok:
            self.mainscene.updateExtensions(tableshort)
            self.mainscene.saveExtensions(tableshort,tablelong)
            self.statusBar().showMessage('Extensions set.')

    def generatePlates(self):
        selection, ok = PlateDialog.getPlates()
        if ok:
            if selection == 0:
                pass
            else:
                allplates = self.mainscene.preparePlates(selection)
                self.statusBar().showMessage('A total of '+str(len(allplates)*2)+' Plates generated.')
                path = QtGui.QFileDialog.getSaveFileName(self, 'Save csv files to.', filter='*.csv')
                if path:
                    savePlate(path,allplates)
                    self.statusBar().showMessage('Plates saved to : '+path)
                else:
                    self.statusBar().showMessage('Filename not specified. Plates not saved.')

    def pipettingScheme(self):
        structureData = self.mainscene.preparePlates(1)[0]
        fulllist, ok = PipettingDialog.getSchemes()
        if fulllist == []:
            self.statusBar().showMessage('No *.csv found. Scheme not created.')
        else:
            pipettlist = []
            platelist = []
            for i in range(1,len(structureData)):
                sequencerow = structureData[i]
                sequence = sequencerow[3]
                if sequence == ' ':
                    pass
                else:
                    for j in range(0,len(fulllist)):
                        fulllistrow = fulllist[j]
                        fulllistseq = fulllistrow[3]
                        if sequence == fulllistseq:
                            pipettlist.append(fulllist[j])
                            platelist.append(fulllist[j][0])
            noplates = len(set(platelist))
            platenames = list(set(platelist))
            platenames.sort()
            if (len(structureData)-1-16)==(len(pipettlist)):
                self.statusBar().showMessage('All sequences found in '+str(noplates)+' Plates. Pipetting Scheme complete.')
            else:
                self.statusBar().showMessage('Error: Some sequences missing. Please check file..')

            allfig = dict()
            for x in range(0,len(platenames)):
                platename = platenames[x]
                print(platename)
                selection = []
                for y in range(0,len(platelist)):
                    if pipettlist[y][0]==platename:
                        selection.append(pipettlist[y][1])
                print(selection)
                allfig[x] = plotPlate(selection,platename)

            path = QtGui.QFileDialog.getSaveFileName(self, 'Save Pipetting Schemes to.', filter='*.pdf')
            if path:
                with PdfPages(path) as pdf:
                    for x in range(0,len(platenames)):
                        pdf.savefig(allfig[x],  bbox_inches='tight', pad_inches=0.1)
                self.statusBar().showMessage('Pippetting Scheme saved to: '+path)


class MainWindow(QtGui.QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):

        #create window with canvas
        self.window = Window()
        #define buttons
        loadbtn = QtGui.QPushButton("Load")
        savebtn = QtGui.QPushButton("Save")
        clearbtn =  QtGui.QPushButton("Clear Canvas")
        sshotbtn = QtGui.QPushButton("Screenshot")
        seqbtn = QtGui.QPushButton("Extensions")
        platebtn = QtGui.QPushButton("Get Plates")
        pipettbtn = QtGui.QPushButton("Pipetting Scheme")

        loadbtn.clicked.connect(self.window.openDialog)
        savebtn.clicked.connect(self.window.saveDialog)
        clearbtn.clicked.connect(self.window.clearDialog)
        sshotbtn.clicked.connect(self.window.takeScreenshot)
        seqbtn.clicked.connect(self.window.setSeq)
        platebtn.clicked.connect(self.window.generatePlates)
        pipettbtn.clicked.connect(self.window.pipettingScheme)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(loadbtn)
        hbox.addWidget(savebtn)
        hbox.addWidget(clearbtn)
        hbox.addWidget(sshotbtn)
        hbox.addWidget(seqbtn)
        hbox.addWidget(platebtn)
        hbox.addWidget(pipettbtn)

        #set layout
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.window)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.resize(800, 600)
        self.setWindowTitle('design')
        self.show()

        #make white background
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background,QtCore.Qt.white)
        self.setPalette(palette)

def main():

    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
