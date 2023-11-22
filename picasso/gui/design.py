"""
    picasso.design-gui
    ~~~~~~~~~~~~~~~~

    GUI for design :
    Design rectangular rothemund origami

    :author: Maximilian Thomas Strauss,  2016
    :copyright: Copyright (c) 2016 Jungmann Lab,  MPI of Biochemistry
"""

import glob, os, sys, traceback, importlib, pkgutil
import os.path as _ospath
from math import sqrt

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as _np
from matplotlib.backends.backend_pdf import PdfPages
from PyQt5 import QtCore, QtGui, QtWidgets, QtPrintSupport

from .. import io as _io
from .. import design, design_sequences
from .. import lib, __version__

BASE_SEQUENCES = design_sequences.base_sequences
PAINT_SEQUENCES = design_sequences.paint_sequences


def plotPlate(selection, selectioncolors, platename):
    inch = 25.4
    radius = 4.5 / inch  # diameter of 96 well plates is 9mm
    radiusc = 4 / inch
    rows = 8
    cols = 12
    colsStr = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    rowsStr = ["A", "B", "C", "D", "E", "F", "G", "H"]
    rowsStr = rowsStr[::-1]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 8)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.cla()
    plt.axis("equal")
    for xcord in range(0, cols):
        for ycord in range(0, rows):
            string = rowsStr[ycord] + colsStr[xcord]
            xpos = xcord * radius * 2 + radius
            ypos = ycord * radius * 2 + radius
            if string in selection:
                circle = plt.Circle(
                    (xpos, ypos),
                    radiusc,
                    facecolor=selectioncolors[selection.index(string)],
                    edgecolor="black",
                )
                ax.text(
                    xpos,
                    ypos,
                    string,
                    fontsize=10,
                    color="white",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            else:
                circle = plt.Circle(
                    (xpos, ypos), radiusc, facecolor="white", edgecolor="black"
                )
            ax.add_artist(circle)
    # inner rectangle
    ax.add_patch(
        patches.Rectangle((0, 0), cols * 2 * radius, rows * 2 * radius, fill=False)
    )
    # outer Rectangle
    ax.add_patch(
        patches.Rectangle(
            (0 - 2 * radius, 0),
            (cols + 1) * 2 * radius,
            (rows + 1) * 2 * radius,
            fill=False,
        )
    )

    # add rows and columns
    for xcord in range(0, cols):
        ax.text(
            xcord * 2 * radius + radius,
            rows * 2 * radius + radius,
            colsStr[xcord],
            fontsize=10,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )
    for ycord in range(0, rows):
        ax.text(
            -radius,
            ycord * 2 * radius + radius,
            rowsStr[ycord],
            fontsize=10,
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

    ax.set_xlim([-2 * radius, cols * 2 * radius])
    ax.set_ylim([0, (rows + 1) * 2 * radius])
    plt.title(platename + " - " + str(len(selection)) + " Staples")
    ax.set_xticks([])
    ax.set_yticks([])
    xsize = 13 * 2 * radius
    ysize = 9 * 2 * radius
    fig.set_size_inches(xsize, ysize)

    return fig


BasePlate = BASE_SEQUENCES.copy()
PaintHandles = PAINT_SEQUENCES.copy()

# list of standard paint sequences
allSeqShort = "None"
allSeqLong = " "

for i in range(1, len(PaintHandles)):

    allSeqShort = allSeqShort + ", " + PaintHandles[i][0]
    allSeqLong = allSeqLong + ", " + PaintHandles[i][1]


HEX_SIDE_HALF = 20
HEX_SCALE = 1
HEX_PEN = QtGui.QPen(QtGui.QBrush(QtGui.QColor("black")), 2)

LINE_PEN = QtGui.QPen(QtGui.QBrush(QtGui.QColor(205, 205, 205, 255)), 2)

# origami definition
rows = 12
rowIndex = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]
columns = 16
columnIndex = range(1, columns + 1)
ORIGAMI_SITES = [(x, y) for x in range(rows) for y in range(columns)]

ind2remove = [
    (1, 2),
    (2, 2),
    (8, 2),
    (9, 2),
    (1, 6),
    (2, 6),
    (8, 6),
    (9, 6),
    (1, 10),
    (2, 10),
    (8, 10),
    (9, 10),
    (1, 14),
    (2, 14),
    (8, 14),
    (9, 14),
]
for element1 in ind2remove:
    for element2 in ORIGAMI_SITES:
        if element1 == element2:
            ORIGAMI_SITES.remove(element2)

# add color palette
maxbinding = max(ORIGAMI_SITES)
COLOR_SITES = [
    (0, maxbinding[1] + 3),
    (2, maxbinding[1] + 3),
    (3, maxbinding[1] + 3),
    (4, maxbinding[1] + 3),
    (5, maxbinding[1] + 3),
    (6, maxbinding[1] + 3),
    (7, maxbinding[1] + 3),
    (8, maxbinding[1] + 3),
    (10, maxbinding[1] + 3),
]

BINDING_SITES = ORIGAMI_SITES + COLOR_SITES

colorscheme = 0


rgbcolors = dict()
allcolors = dict()
if colorscheme:
    rgbcolors[0] = [205, 205, 205]
    rgbcolors[1] = [166, 206, 227]
    rgbcolors[2] = [31, 120, 180]
    rgbcolors[3] = [178, 223, 138]
    rgbcolors[4] = [51, 160, 44]
    rgbcolors[5] = [251, 154, 153]
    rgbcolors[6] = [227, 26, 28]
    rgbcolors[7] = [255, 255, 255]
    rgbcolors[8] = [205, 205, 205]
    allcolors[0] = QtGui.QColor(205, 205, 205, 255)  # DEFAULT,  GREY
    allcolors[1] = QtGui.QColor(166, 206, 227, 255)
    allcolors[2] = QtGui.QColor(31, 120, 180, 255)
    allcolors[3] = QtGui.QColor(178, 223, 138, 255)
    allcolors[4] = QtGui.QColor(51, 160, 44, 255)
    allcolors[5] = QtGui.QColor(251, 154, 153, 255)
    allcolors[6] = QtGui.QColor(227, 26, 28, 255)
    allcolors[7] = QtGui.QColor(0, 0, 0, 255)  # BLACK
    allcolors[8] = QtGui.QColor(255, 255, 255, 255)  # WHITE

else:
    rgbcolors[0] = [0, 0, 0]  # is black to increase visibility
    rgbcolors[1] = [166, 206, 227]
    rgbcolors[2] = [31, 120, 180]
    rgbcolors[3] = [178, 223, 138]
    rgbcolors[4] = [51, 160, 44]
    rgbcolors[5] = [251, 154, 153]
    rgbcolors[6] = [227, 26, 28]
    rgbcolors[7] = [253, 191, 111]
    rgbcolors[8] = [205, 205, 205]
    allcolors[0] = QtGui.QColor(205, 205, 205, 255)  # DEFAULT,  GREY
    allcolors[1] = QtGui.QColor(166, 206, 227, 255)
    allcolors[2] = QtGui.QColor(31, 120, 180, 255)
    allcolors[3] = QtGui.QColor(178, 223, 138, 255)
    allcolors[4] = QtGui.QColor(51, 160, 44, 255)
    allcolors[5] = QtGui.QColor(251, 154, 153, 255)
    allcolors[6] = QtGui.QColor(227, 26, 28, 255)
    allcolors[7] = QtGui.QColor(253, 191, 111, 255)  # B
    allcolors[8] = QtGui.QColor(255, 255, 255, 255)  # WHITE

for element in rgbcolors:
    rgbcolors[element][:] = [x / 255 for x in rgbcolors[element]]


defaultcolor = allcolors[0]
maxcolor = 8


def indextoHex(y, x):
    hex_center_x = x * 1.5 * HEX_SIDE_HALF
    if _np.mod(x, 2) == 0:
        hex_center_y = -y * sqrt(3) / 2 * HEX_SIDE_HALF * 2
    else:
        hex_center_y = -(y + 0.5) * sqrt(3) / 2 * HEX_SIDE_HALF * 2
    return hex_center_x, hex_center_y


def indextoStr(x, y):
    rowStr = rowIndex[y]
    colStr = columnIndex[x]
    strIndex = (rowStr, colStr)
    return strIndex


class PipettingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(PipettingDialog, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.setWindowTitle("Pipetting dialog")

        self.loadButton = QtWidgets.QPushButton("Select folder")
        self.folderEdit = QtWidgets.QLabel("")
        self.csvCounter = QtWidgets.QLabel("")
        self.plateCounter = QtWidgets.QLabel("")
        self.uniqueCounter = QtWidgets.QLabel("")

        self.fulllist = []

        self.loadButton.clicked.connect(self.loadFolder)

        layout.addWidget(self.loadButton)
        layout.addWidget(self.folderEdit)
        layout.addWidget(self.csvCounter)
        layout.addWidget(self.plateCounter)
        layout.addWidget(self.uniqueCounter)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )

        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def loadFolder(self):
        if hasattr(self, "pwd"):
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Directory", self.pwd
            )
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self.folderEdit.setText(path)
            csvFiles = glob.glob(os.path.join(path, "*.csv"))
            csvCounter = len(csvFiles)
            self.csvCounter.setText(
                "A total of " + str(csvCounter) + " *.csv files detected."
            )
            platelist = []
            sequencelist = []
            for element in csvFiles:
                os.chdir(path)
                data = design.readPlate(element)
                for i in range(0, len(data)):
                    platelist.append(data[i][0])
                    sequencelist.append(data[i][3])
                    self.fulllist.append(data[i][0:4])

            self.plateCounter.setText(
                "A total of " + str(len(set(platelist)) - 1) + "  plates detected."
            )
            self.uniqueCounter.setText(
                "A total of "
                + str(len(set(sequencelist)) - 2)
                + "  unique sequences detected."
            )

    def getfulllist(self):
        fulllist = self.fulllist
        return fulllist

    @staticmethod
    def getSchemes(parent=None, pwd=None):
        dialog = PipettingDialog(parent)
        if pwd:
            dialog.pwd = pwd
        result = dialog.exec_()
        fulllist = dialog.getfulllist()

        return (fulllist, result == QtWidgets.QDialog.Accepted)


class SeqDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SeqDialog, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget()
        self.table.setWindowTitle("Extension table")
        self.setWindowTitle("Extension table")
        self.resize(500, 285)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ("Pos, Color, Preselection, Shortname, Sequence").split(", ")
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()

        layout.addWidget(self.table)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )

        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def initTable(self, colorcounts, tableshort, tablelong):
        noseq = _np.count_nonzero(colorcounts)
        # table defintion
        self.table.setRowCount(noseq - 1)
        rowRunner = 0
        for i in range(len(colorcounts) - 1):
            if colorcounts[i] > 0:
                self.table.setItem(rowRunner, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
                self.table.setItem(
                    rowRunner, 1, QtWidgets.QTableWidgetItem("Ext " + str(i + 1))
                )
                self.table.item(rowRunner, 1).setBackground(allcolors[i + 1])
                self.table.setItem(
                    rowRunner, 3, QtWidgets.QTableWidgetItem(tableshort[i])
                )
                self.table.setItem(
                    rowRunner, 4, QtWidgets.QTableWidgetItem(tablelong[i])
                )
                rowRunner += 1

        self.ImagersShort = allSeqShort.split(", ")
        self.ImagersLong = allSeqLong.split(", ")

        comb = dict()

        for i in range(self.table.rowCount()):
            comb[i] = QtWidgets.QComboBox()

        for element in self.ImagersShort:
            for i in range(self.table.rowCount()):
                comb[i].addItem(element)

        for i in range(self.table.rowCount()):
            self.table.setCellWidget(i, 2, comb[i])
            comb[i].currentIndexChanged.connect(
                lambda state, indexval=i: self.changeComb(indexval)
            )

    def changeComb(self, indexval):

        sender = self.sender()
        comboval = sender.currentIndex()
        if comboval == 0:
            self.table.setItem(indexval, 3, QtWidgets.QTableWidgetItem("None"))
            self.table.setItem(indexval, 4, QtWidgets.QTableWidgetItem("None"))
        else:
            self.table.setItem(
                indexval,
                3,
                QtWidgets.QTableWidgetItem(self.ImagersShort[comboval]),
            )
            self.table.setItem(
                indexval, 4, QtWidgets.QTableWidgetItem(self.ImagersLong[comboval])
            )

    def readoutTable(self):
        tableshort = ["None", "None", "None", "None", "None", "None", "None"]
        tablelong = ["None", "None", "None", "None", "None", "None", "None"]

        for i in range(self.table.rowCount()):
            try:
                tableshort[int(self.table.item(i, 0).text()) - 1] = self.table.item(
                    i, 3
                ).text()
                if tableshort[i] == "":
                    tableshort[i] = "None"
            except AttributeError:
                tableshort[i] = "None"

            try:
                tablelong[int(self.table.item(i, 0).text()) - 1] = self.table.item(
                    i, 4
                ).text()
                if tablelong[i] == "":
                    tablelong[i] = "None"
            except AttributeError:
                tablelong[i] = "None"
        return tablelong, tableshort


class FoldingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(FoldingDialog, self).__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget()
        self.table.setWindowTitle("Folding table")
        self.setWindowTitle("Folding table")
        self.resize(800, 285)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.table.setRowCount(maxcolor - 1)
        self.table.setColumnCount(8)
        # PRE-SET LABELS
        self.table.setHorizontalHeaderLabels(
            [
                "Component",
                "Initial concentration[uM]",
                "Parts",
                "Pool-concentration[nM]",
                "Target concentration[nM]",
                "Volume[ul]",
                "Excess",
                "Colorcode",
            ]
        )
        self.clcButton = QtWidgets.QPushButton("Recalculate")
        self.clcButton.clicked.connect(self.clcExcess)
        self.exportButton = QtWidgets.QPushButton("Export")
        self.exportButton.clicked.connect(self.exportTable)
        layout.addWidget(self.table)
        layout.addWidget(self.clcButton)
        layout.addWidget(self.exportButton)
        self.table.resizeColumnsToContents()

    def exportTable(self):

        table = dict()
        tablecontent = []
        tablecontent.append(
            [
                "Component",
                "Initial Concentration[uM]",
                "Parts",
                "Pool-Concentration[nM]",
                "Target Concentration[nM]",
                "Volume[ul]",
                "Excess",
                "Colorcode",
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
        if hasattr(self, "pwd"):
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export folding table to.", self.pwd, filter="*.csv"
            )
        else:
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export folding table to.", filter="*.csv"
            )
        if path:
            design.savePlate(path, table)

    def clcExcess(self):
        rowCount = self.table.rowCount()
        self.resize(800, 285 + (rowCount - 6) * 30)

        # Calculate pool concentration for all except the last 3 ()
        totalvolume = float(self.table.item(rowCount - 1, 5).text())

        # Calculate the Target concentration based on the excess
        for i in range(1, rowCount - 3):
            target = float(self.table.item(0, 4).text())
            excess = int(self.table.item(i, 6).text())
            self.writeTable(i, 4, str(target * excess))

        # Calculate the pool concentration
        for i in range(rowCount - 3):
            iconc = float(self.table.item(i, 1).text())
            parts = int(self.table.item(i, 2).text())
            content = _np.round((iconc / parts * 1000), decimals = 3)
            self.writeTable(i, 3, str(content))

        # Calculate Volume based on pool and final concentration
        volume = _np.zeros(rowCount - 3)
        for i in range(rowCount - 3):
            target = float(self.table.item(i, 4).text())
            pool = float(self.table.item(i, 3).text())
            volume[i] = target / pool * totalvolume
            self.writeTable(i, 5, str(_np.round(volume[i], decimals=3)))
        foldingbuffer = totalvolume / 10

        # Calculate Folding Buffer
        self.writeTable(rowCount - 2, 5, str(foldingbuffer))

        # Calculate remainging H20
        water = totalvolume - foldingbuffer - _np.sum(volume)

        self.writeTable(rowCount - 3, 5, str(_np.round(water, decimals=3)))
        if water < 0:
            self.table.item(rowCount - 3, 5).setBackground(QtGui.QColor("red"))
        else:
            self.table.item(rowCount - 3, 5).setBackground(QtGui.QColor("white"))

    def writeTable(self, row, col, content):
        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(content))

    def colorTable(self, row, col, color):
        self.table.item(row, col).setBackground(color)

    def setExt(parent=None):
        dialog = FoldingDialog(parent)
        result = dialog.exec_()
        tablelong, tableshort = dialog.evalTable()
        return (tablelong, tableshort, result == QtWidgets.QDialog.Accepted)


class PlateDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(PlateDialog, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.info = QtWidgets.QLabel("Please make selection:  ")
        self.radio1 = QtWidgets.QRadioButton(
            (
                "Export only the sequences needed for this design."
                " (176 staples in 2 plates)"
            )
        )
        self.radio2 = QtWidgets.QRadioButton(
            (
                "Export full 2 full plates for all sequences used"
                " (176 staples * number of unique sequences)"
            )
        )

        self.setWindowTitle("Plate export")
        layout.addWidget(self.info)
        layout.addWidget(self.radio1)
        layout.addWidget(self.radio2)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )

        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def evalSelection(self):
        if self.radio1.isChecked():
            selection = 1
        elif self.radio2.isChecked():
            selection = 2
        else:
            selection = 0
        return selection

    @staticmethod
    def getSelection(parent=None):
        dialog = PlateDialog(parent)
        result = dialog.exec_()
        selection = dialog.evalSelection()
        return (selection, result == QtWidgets.QDialog.Accepted)


class BindingSiteItem(QtWidgets.QGraphicsPolygonItem):
    def __init__(self, y, x):
        hex_center_x, hex_center_y = indextoHex(y, x)
        center = QtCore.QPointF(hex_center_x, hex_center_y)
        points = [
            HEX_SCALE * HEX_SIDE_HALF * QtCore.QPointF(-1, 0) + center,
            HEX_SCALE * HEX_SIDE_HALF * QtCore.QPointF(-0.5, sqrt(3) / 2) + center,
            HEX_SCALE * HEX_SIDE_HALF * QtCore.QPointF(0.5, sqrt(3) / 2) + center,
            HEX_SCALE * HEX_SIDE_HALF * QtCore.QPointF(1, 0) + center,
            HEX_SCALE * HEX_SIDE_HALF * QtCore.QPointF(0.5, -sqrt(3) / 2) + center,
            HEX_SCALE * HEX_SIDE_HALF * QtCore.QPointF(-0.5, -sqrt(3) / 2) + center,
        ]

        hexagonPointsF = QtGui.QPolygonF(points)
        super().__init__(hexagonPointsF)
        self.setPen(HEX_PEN)
        self.setBrush(defaultcolor)  # initialize all as grey


class Scene(QtWidgets.QGraphicsScene):
    def __init__(self, window):
        super().__init__()
        self.window = window
        for coords in BINDING_SITES:
            self.addItem(BindingSiteItem(*coords))
        self.allcords = []
        self.indices = []

        self.tableshort = [
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
        ]
        self.tablelong = [
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
        ]

        for coords in BINDING_SITES:
            y = coords[0]
            x = coords[1]
            hex_center_x, hex_center_y = indextoHex(y, x)
            self.allcords.append(
                (
                    hex_center_x * 0.125 * 4 / 3,
                    2.5 - hex_center_y * 0.125 * 2 / sqrt(3),
                )
            )  # 5nm spacing

        # Pprepare file format for origami definition
        self.origamicoords = []
        self.origamiindices = []
        for coords in ORIGAMI_SITES:
            y = coords[0]
            x = coords[1]

            self.origamiindices.append((indextoStr(y, x)))
            hex_center_x, hex_center_y = indextoHex(y, x)
            self.origamicoords.append((hex_center_x * 0.125, hex_center_y * 0.125))

        # color palette
        allitems = self.items()
        lenitems = len(allitems)
        bindingitems = len(BINDING_SITES)
        paletteindex = lenitems - bindingitems

        allitems[paletteindex].setBrush(allcolors[1])
        allitems[paletteindex + maxcolor].setBrush(allcolors[0])
        for i in range(1, maxcolor):
            allitems[paletteindex + i].setBrush(allcolors[i])

        self.alllbl = dict()  # LABELS FOR COUNTING
        self.alllblseq = dict()  # LABELS FOR EXTENSIONS

        labelspacer = 0.5
        yoff = 0.2
        xoff = 0.2
        xofflbl = 1

        for i in range(0, maxcolor):
            self.alllbl[i] = QtWidgets.QGraphicsTextItem("   ")
            self.alllbl[i].setPos(
                *(
                    1.5
                    * HEX_SIDE_HALF
                    * (labelspacer + COLOR_SITES[7 - paletteindex - i][1] + xoff),
                    -labelspacer
                    - sqrt(3)
                    * HEX_SIDE_HALF
                    * (COLOR_SITES[7 - paletteindex - i][0] + yoff),
                )
            )
            self.addItem(self.alllbl[i])

            self.alllblseq[i] = QtWidgets.QGraphicsTextItem("   ")
            self.alllblseq[i].setPos(
                *(
                    1.5
                    * HEX_SIDE_HALF
                    * (labelspacer + COLOR_SITES[7 - paletteindex - i][1] + xofflbl),
                    -labelspacer
                    - sqrt(3)
                    * HEX_SIDE_HALF
                    * (COLOR_SITES[7 - paletteindex - i][0] + yoff),
                )
            )
            self.addItem(self.alllblseq[i])

        # MAKE A LABEL FOR THE CURRENTCOLOR
        self.cclabel = QtWidgets.QGraphicsTextItem("Selected color")
        self.cclabel.setPos(
            *(
                1.5
                * HEX_SIDE_HALF
                * (labelspacer + COLOR_SITES[8 - paletteindex][1] + xofflbl),
                -labelspacer
                - sqrt(3) * HEX_SIDE_HALF * (COLOR_SITES[8 - paletteindex][0] + yoff),
            )
        )
        self.addItem(self.cclabel)

        # MAKE A LABEL FOR THE HEXAGONS POINTING DOWNWARDS Ideally: Gray V
        # isolated ones:
        index1 = [3, 3, 3, 3, 10, 10, 10, 10]
        index2 = [2, 6, 10, 14, 2, 6, 10, 14]
        line1 = dict()
        line2 = dict()

        for i in range(len(index1)):
            hex_center_x, hex_center_y = indextoHex(index1[i], index2[i])

            startpoint = QtCore.QPointF(
                hex_center_x - 0.5 * HEX_SIDE_HALF,
                hex_center_y - 0.5 * HEX_SIDE_HALF + 2,
            )
            midpoint = QtCore.QPointF(hex_center_x, hex_center_y + 0.5 * HEX_SIDE_HALF)
            endpoint = QtCore.QPointF(
                hex_center_x + 0.5 * HEX_SIDE_HALF,
                hex_center_y - 0.5 * HEX_SIDE_HALF + 2,
            )

            line1[i] = QtWidgets.QGraphicsLineItem(QtCore.QLineF(startpoint, midpoint))
            line2[i] = QtWidgets.QGraphicsLineItem(QtCore.QLineF(midpoint, endpoint))

            line1[i].setPen(LINE_PEN)
            line2[i].setPen(LINE_PEN)

            self.addItem(line1[i])
            self.addItem(line2[i])

        self.evaluateCanvas()

    def mousePressEvent(self, event):
        clicked_item = self.itemAt(event.scenePos(), self.window.view.transform())
        if clicked_item:
            if clicked_item.type() == 5:
                allitems = self.items()
                lenitems = len(allitems)
                bindingitems = len(BINDING_SITES)
                paletteindex = lenitems - bindingitems
                selectedcolor = allitems[paletteindex].brush().color()

                if clicked_item == allitems[paletteindex]:  # DO NOTHING
                    pass
                elif clicked_item == allitems[paletteindex + 1]:
                    allitems[paletteindex].setBrush(allcolors[1])
                    selectedcolor = allcolors[1]
                elif clicked_item == allitems[paletteindex + 2]:
                    allitems[paletteindex].setBrush(allcolors[2])
                    selectedcolor = allcolors[2]
                elif clicked_item == allitems[paletteindex + 3]:
                    allitems[paletteindex].setBrush(allcolors[3])
                    selectedcolor = allcolors[3]
                elif clicked_item == allitems[paletteindex + 4]:
                    allitems[paletteindex].setBrush(allcolors[4])
                    selectedcolor = allcolors[4]
                elif clicked_item == allitems[paletteindex + 5]:
                    selectedcolor = allcolors[5]
                    allitems[paletteindex].setBrush(allcolors[5])
                elif clicked_item == allitems[paletteindex + 6]:
                    allitems[paletteindex].setBrush(allcolors[6])
                    selectedcolor = allcolors[6]
                elif clicked_item == allitems[paletteindex + 7]:
                    allitems[paletteindex].setBrush(allcolors[7])
                    selectedcolor = allcolors[7]
                elif clicked_item == allitems[paletteindex + 8]:
                    allitems[paletteindex].setBrush(QtGui.QBrush(defaultcolor))
                    selectedcolor = defaultcolor
                else:
                    currentcolor = clicked_item.brush().color()
                    if currentcolor == selectedcolor:
                        clicked_item.setBrush(
                            defaultcolor
                        )  # TURN WHITE AGAIN IF NOT USED
                    else:
                        clicked_item.setBrush(QtGui.QBrush(selectedcolor))
                self.evaluateCanvas()

    def evaluateCanvas(self):
        # READS OUT COLOR VALUES AND MAKES A LIST WITH CORRESPONDING COLORS
        allitems = self.items()
        lenitems = len(allitems)
        bindingitems = len(BINDING_SITES)
        origamiitems = len(ORIGAMI_SITES)
        paletteindex = lenitems - bindingitems + 9
        canvascolors = []
        self.colorcounts = []
        for i in range(0, origamiitems):
            currentcolor = allitems[paletteindex + i].brush().color()
            for j in range(0, maxcolor):
                if currentcolor == allcolors[j]:
                    canvascolors.append(j)
        for i in range(0, maxcolor):
            tocount = i + 1
            if i == maxcolor - 1:
                tocount = 0
            count = canvascolors.count(tocount)
            if count == 0:
                self.alllbl[i].setPlainText("   ")
            else:
                self.alllbl[i].setPlainText(str(canvascolors.count(tocount)))
            self.colorcounts.append(count)

        return canvascolors

    def updateExtensions(self, tableshort):
        # Takes a list of tableshort and updates the display
        allitems = self.items()
        lenitems = len(allitems)
        bindingitems = len(BINDING_SITES)

        for i in range(0, maxcolor - 1):
            if tableshort[i] == "None":
                self.alllblseq[i].setPlainText("   ")
            else:
                self.alllblseq[i].setPlainText(tableshort[i])
                #

    def saveExtensions(self, tableshort, tablelong):
        self.tableshort = tableshort
        self.tablelong = tablelong

    def clearCanvas(self):
        allitems = self.items()
        lenitems = len(allitems)
        bindingitems = len(BINDING_SITES)
        origamiitems = len(ORIGAMI_SITES)
        paletteindex = lenitems - bindingitems + 9

        for i in range(0, origamiitems):
            allitems[paletteindex + i].setBrush(QtGui.QBrush(defaultcolor))

        self.saveExtensions(
            ["None", "None", "None", "None", "None", "None", "None"],
            ["None", "None", "None", "None", "None", "None", "None"],
        )
        self.updateExtensions(["None", "None", "None", "None", "None", "None", "None"])
        self.evaluateCanvas()

    def vectorToString(self, x):
        x_arrstr = _np.char.mod("%f", x)
        x_str = ", ".join(x_arrstr)
        return x_str

    def vectorToStringInt(self, x):
        x_arrstr = _np.char.mod("%i", x)
        x_str = ", ".join(x_arrstr)
        return x_str

    def saveCanvas(self, path):
        canvascolors = self.evaluateCanvas()[::-1]
        structurec = []
        structureInd = []
        for x in range(0, len(canvascolors)):
            structurec.append(
                (self.allcords[x][0], self.allcords[x][1], canvascolors[x])
            )

        for x in range(0, len(canvascolors)):
            structureInd.append(
                [
                    self.origamiindices[x][0],
                    self.origamiindices[x][1],
                    canvascolors[x],
                ]
            )

        # Coordinates for picasso simulate
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

        info = {
            "Generated by": "Picasso design",
            "Structure": structureInd,
            "Extensions Short": self.tableshort,
            "Extensions Long": self.tablelong,
            "Structure.StructureX": structurex,
            "Structure.StructureY": structurey,
            "Structure.StructureEx": structureex,
        }
        design.saveInfo(path, info)

    def loadCanvas(self, path):
        info = _io.load_info(path)
        try:
            structure = info[0]["Structure"]
        except KeyError:
            self.window.statusBar().showMessage("Error. Filetype not recognized")
        structure = structure[::-1]
        allitems = self.items()
        lenitems = len(allitems)
        bindingitems = len(BINDING_SITES)
        origamiitems = len(ORIGAMI_SITES)
        paletteindex = lenitems - bindingitems + 9

        for i in range(0, origamiitems):
            colorindex = structure[i][2]
            allitems[paletteindex + i].setBrush(QtGui.QBrush(allcolors[colorindex]))

        self.evaluateCanvas()
        self.tableshort = info[0]["Extensions Short"]
        self.tablelong = info[0]["Extensions Long"]
        self.updateExtensions(self.tableshort)

    def readCanvas(self):
        allplates = dict()
        canvascolors = self.evaluateCanvas()[::-1]
        ExportPlate = BASE_SEQUENCES.copy()

        ExportPlate[0] = ["Position", "Name", "Sequence", "Color"]
        for i in range(0, len(canvascolors)):
            if canvascolors[i] == 0:
                ExportPlate[1 + i] = [
                    ExportPlate[1 + i][0],
                    ExportPlate[1 + i][1],
                    ExportPlate[1 + i][2],
                    canvascolors[i],
                ]
                pass
            else:
                ExportPlate[1 + i][2] = (
                    ExportPlate[1 + i][2] + " " + self.tablelong[canvascolors[i] - 1]
                )
                ExportPlate[1 + i][1] = (
                    ExportPlate[1 + i][1][:-3] + self.tableshort[canvascolors[i] - 1]
                )
                ExportPlate[1 + i] = [
                    ExportPlate[1 + i][0],
                    ExportPlate[1 + i][1],
                    ExportPlate[1 + i][2],
                    canvascolors[i],
                ]

        allplates[0] = design.convertPlateIndexColor(ExportPlate, "CUSTOM")
        return allplates

    def preparePlate(self, mode):
        # reads out the canvas, modifies BasePlate
        canvascolors = self.evaluateCanvas()[::-1]

        colors = list(set(canvascolors))
        allplates = dict()

        if mode == 2:  # generate a full plate for each used extension
            # get number of plates
            noplates = len(set(canvascolors))
            for j in range(0, noplates):
                if colors[j] == 0:
                    allplates[j] = design.convertPlateIndex(
                        BASE_SEQUENCES.copy(), "BLK"
                    )
                else:
                    ExportPlate = BASE_SEQUENCES.copy()
                    for i in range(0, len(canvascolors)):
                        ExportPlate[1 + i][2] = (
                            ExportPlate[1 + i][2] + " " + self.tablelong[colors[j] - 1]
                        )
                        ExportPlate[1 + i][1] = (
                            ExportPlate[1 + i][1][:-3] + self.tableshort[colors[j] - 1]
                        )
                    allplates[j] = design.convertPlateIndex(
                        ExportPlate, self.tableshort[colors[j] - 1]
                    )

        elif mode == 1:  # only one plate with the modifications
            ExportPlate = BASE_SEQUENCES.copy()
            for i in range(0, len(canvascolors)):
                if canvascolors[i] == 0:
                    pass
                else:
                    ExportPlate[1 + i][2] = (
                        ExportPlate[1 + i][2]
                        + " "
                        + self.tablelong[canvascolors[i] - 1]
                    )
                    ExportPlate[1 + i][1] = (
                        ExportPlate[1 + i][1][:-3]
                        + self.tableshort[canvascolors[i] - 1]
                    )

            allplates[0] = design.convertPlateIndex(ExportPlate, "CUSTOM")

        return allplates


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.mainscene = Scene(self)
        self.view = QtWidgets.QGraphicsView(self.mainscene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setCentralWidget(self.view)
        self.statusBar().showMessage(
            "Ready."
        )  # . . Sequences loaded from " + BaseSequencesFile + ".")

    def openDialog(self):
        if hasattr(self, "pwd"):
            path, ext = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open design", self.pwd, filter="*.yaml"
            )
        else:
            path, ext = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open design", filter="*.yaml"
            )
        if path:
            self.mainscene.loadCanvas(path)
            self.statusBar().showMessage("File loaded from: " + path)
            self.pwd = os.path.dirname(path)
        else:
            self.statusBar().showMessage("Filename not specified. File not loaded.")

    def saveDialog(self):
        if hasattr(self, "pwd"):
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save design to..", self.pwd, filter="*.yaml"
            )
        else:
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save design to..", filter="*.yaml"
            )
        if path:
            self.mainscene.saveCanvas(path)
            self.statusBar().showMessage("File saved as: " + path)
            self.pwd = os.path.dirname(path)
        else:
            self.statusBar().showMessage("Filename not specified. Design not saved.")

    def clearDialog(self):
        self.mainscene.clearCanvas()
        self.statusBar().showMessage("Clearead.")

    def takeScreenshot(self):
        filetypes = "*.png;;*.pdf"
        if hasattr(self, "pwd"):
            path, filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Screenshot to..", self.pwd, filter=filetypes
            )
        else:
            path, filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Screenshot to..", filter=filetypes
            )
        if path:
            if filter == "*.png":
                # p = QtGui.QPixmap.grab(self.view)
                p = self.view.grab()
                p.save(path, filter[2:])
            else:
                pdf_printer = QtPrintSupport.QPrinter()
                pdf_printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
                pdf_printer.setOutputFileName(path)
                pdf_painter = QtGui.QPainter()
                pdf_painter.begin(pdf_printer)
                self.view.render(pdf_painter)
                pdf_painter.end()

            self.statusBar().showMessage("Screenshot saved to {}".format(path))
        else:
            self.statusBar().showMessage(
                "Filename not specified. Screenshot not saved."
            )

    def setSeq(self):

        colorcounts = self.mainscene.colorcounts

        if sum(colorcounts[0:-1]) > 0:

            sdialog = SeqDialog()
            sdialog.initTable(
                colorcounts,
                self.mainscene.tableshort,
                self.mainscene.tablelong,
            )
            ok = sdialog.exec()

            tablelong, tableshort = sdialog.readoutTable()

            if ok:
                self.mainscene.updateExtensions(tableshort)
                self.mainscene.saveExtensions(tableshort, tablelong)
                self.statusBar().showMessage("Extensions set.")

        else:
            self.statusBar().showMessage("No hexagons marked. Please select first.")

    def checkSeq(self):
        colorcounts = self.mainscene.colorcounts
        tableshort = self.mainscene.tableshort
        errors = 0
        for i in range(len(colorcounts) - 1):
            if colorcounts[i] != 0 and tableshort[i] == "None":
                errors += 1

        return errors

    def generatePlates(self):
        seqcheck = self.checkSeq()
        if self.mainscene.tableshort == [
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
        ]:
            self.statusBar().showMessage(
                ("Error: No extensions have been set." " Please set extensions first.")
            )
        elif seqcheck >= 1:
            self.statusBar().showMessage(
                "Error: "
                + str(seqcheck)
                + " Color(s) do not have extensions. Please set first."
            )
        else:
            selection, ok = PlateDialog.getSelection()
            if ok:
                if selection == 0:
                    pass
                else:
                    allplates = self.mainscene.preparePlate(selection)
                    self.statusBar().showMessage(
                        "A total of " + str(len(allplates) * 2) + " Plates generated."
                    )
                    if hasattr(self, "pwd"):
                        path, ext = QtWidgets.QFileDialog.getSaveFileName(
                            self,
                            "Save csv files to.",
                            self.pwd,
                            filter="*.csv",
                        )
                    else:
                        path, ext = QtWidgets.QFileDialog.getSaveFileName(
                            self, "Save csv files to.", filter="*.csv"
                        )
                    if path:
                        design.savePlate(path, allplates)
                        self.statusBar().showMessage("Plates saved to : " + path)
                        self.pwd = os.path.dirname(path)
                    else:
                        self.statusBar().showMessage(
                            "Filename not specified. Plates not saved."
                        )

    def pipettingScheme(self):
        seqcheck = self.checkSeq()
        if self.mainscene.tableshort == [
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
            "None",
        ]:
            self.statusBar().showMessage(
                ("Error: No extensions have been set." " Please set extensions first.")
            )
        elif seqcheck >= 1:
            self.statusBar().showMessage(
                "Error: "
                + str(seqcheck)
                + " Color(s) do not have extensions. Please set first."
            )
        else:
            structureData = self.mainscene.readCanvas()[0]
            fullpipettlist = [
                [
                    "PLATE NAME",
                    "PLATE POSITION",
                    "OLIGO NAME",
                    "SEQUENCE",
                    "COLOR",
                ]
            ]
            if hasattr(self, "pwd"):
                pwd = self.pwd
            else:
                pwd = []
            fulllist, ok = PipettingDialog.getSchemes(pwd=pwd)
            if fulllist == []:
                self.statusBar().showMessage("No *.csv found. Scheme not created.")
            else:
                pipettlist = []
                platelist = []
                for i in range(1, len(structureData)):
                    sequencerow = structureData[i]
                    sequence = sequencerow[3]
                    fullpipettlist.append(sequencerow)
                    fullpipettlist[i][0] = "NOT FOUND"
                    if fullpipettlist[i][2] == " ":
                        fullpipettlist[i][0] = "BIOTIN PLACEHOLDER"
                    if sequence == " ":
                        pass
                    else:
                        for j in range(0, len(fulllist)):
                            fulllistrow = fulllist[j]
                            fulllistseq = fulllistrow[3]
                            if sequence == fulllistseq:
                                pipettlist.append(
                                    [
                                        fulllist[j][0],
                                        fulllist[j][1],
                                        fulllist[j][2],
                                        fulllist[j][3],
                                        rgbcolors[sequencerow[4]],
                                    ]
                                )
                                platelist.append(fulllist[j][0])
                                del fullpipettlist[-1]
                                fullpipettlist.append(fulllist[j])
                                break  # first found will be taken

                exportlist = dict()
                exportlist[0] = fullpipettlist
                noplates = len(set(platelist))
                platenames = list(set(platelist))
                platenames.sort()
                if (len(structureData) - 1 - 16) == (len(pipettlist)):
                    self.statusBar().showMessage(
                        "All sequences found in "
                        + str(noplates)
                        + " Plates. Pipetting scheme complete."
                    )
                else:
                    self.statusBar().showMessage(
                        (
                            "Error: Sequences sequences missing."
                            " Please check *.csv file.."
                        )
                    )

                allfig = dict()
                for x in range(0, len(platenames)):
                    platename = platenames[x]

                    selection = []
                    selectioncolors = []
                    for y in range(0, len(platelist)):
                        if pipettlist[y][0] == platename:
                            selection.append(pipettlist[y][1])
                            selectioncolors.append(pipettlist[y][4])

                    allfig[x] = plotPlate(selection, selectioncolors, platename)
                if hasattr(self, "pwd"):
                    path, ext = QtWidgets.QFileDialog.getSaveFileName(
                        self,
                        "Save pipetting schemes to.",
                        self.pwd,
                        filter="*.pdf",
                    )
                else:
                    path, ext = QtWidgets.QFileDialog.getSaveFileName(
                        self, "Save pipetting schemes to.", filter="*.pdf"
                    )

                if path:
                    progress = lib.ProgressDialog(
                        "Exporting PDFs", 0, len(platenames), self
                    )
                    progress.set_value(0)
                    progress.show()
                    with PdfPages(path) as pdf:
                        for x in range(0, len(platenames)):
                            progress.set_value(x)
                            # pdf.savefig(allfig[x])
                            pdf.savefig(
                                allfig[x],
                                bbox_inches="tight",
                                pad_inches=0.2,
                                dpi=200,
                            )
                            base, ext = _ospath.splitext(path)
                            csv_path = base + ".csv"
                            design.savePlate(csv_path, exportlist)
                    progress.close()
                    self.statusBar().showMessage("Pippetting scheme saved to: " + path)
                    self.pwd = os.path.dirname(path)

    def foldingScheme(self):

        fdialog = FoldingDialog()  # Intitialize FoldingDialog Class
        # Fill with Data
        colorcounts = self.mainscene.colorcounts
        noseq = _np.count_nonzero(colorcounts)

        if hasattr(self, "pwd"):
            fdialog.pwd = self.pwd

        fdialog.table.setRowCount(noseq + 5)

        fdialog.writeTable(0, 0, "Scaffold")
        fdialog.writeTable(0, 1, str(0.1))
        fdialog.writeTable(0, 2, str(1))
        fdialog.writeTable(0, 4, str(10))
        fdialog.writeTable(0, 6, str(1))

        # BLK STAPLES: 10 x
        fdialog.writeTable(1, 0, "Core Mix")
        fdialog.writeTable(1, 1, str(100))
        fdialog.writeTable(1, 2, str(colorcounts[len(colorcounts) - 1]))
        fdialog.writeTable(1, 6, str(10))
        fdialog.writeTable(1, 7, "")
        fdialog.colorTable(1, 7, allcolors[0])

        mixno = 0
        # MODIFIED STAPLES: 100x

        for i in range(len(colorcounts) - 1):
            if colorcounts[i] != 0:
                fdialog.writeTable(mixno + 2, 0, self.mainscene.tableshort[i] + " Mix")
                fdialog.writeTable(mixno + 2, 2, str(colorcounts[i]))
                fdialog.writeTable(mixno + 2, 1, str(100))
                fdialog.writeTable(mixno + 2, 6, str(100))
                fdialog.writeTable(mixno + 2, 7, "")
                index = i + 1
                fdialog.colorTable(mixno + 2, 7, allcolors[index])

                mixno = mixno + 1

        fdialog.writeTable(mixno + 2, 0, "Biotin 1:10")
        fdialog.writeTable(mixno + 2, 1, str(100))
        fdialog.writeTable(mixno + 2, 2, str(80))
        fdialog.writeTable(mixno + 2, 6, str(1))

        fdialog.writeTable(mixno + 3, 0, "H2O")

        fdialog.writeTable(mixno + 4, 0, "10x Folding Buffer")

        fdialog.writeTable(mixno + 5, 0, "Total Volume")
        fdialog.writeTable(mixno + 5, 5, str(40))

        fdialog.clcExcess()

        # MAKE TABLE INTERACTIVE
        result = fdialog.exec()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle(f"Picasso v{__version__}: Design")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "design.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(800, 600)
        self.initUI()

    def initUI(self):

        # create window with canvas
        self.window = Window()

        # define buttons
        loadbtn = QtWidgets.QPushButton("Load")
        savebtn = QtWidgets.QPushButton("Save")
        clearbtn = QtWidgets.QPushButton("Clear")
        sshotbtn = QtWidgets.QPushButton("Screenshot")
        seqbtn = QtWidgets.QPushButton("Extensions")
        platebtn = QtWidgets.QPushButton("Get plates")
        pipettbtn = QtWidgets.QPushButton("Pipetting scheme")
        foldbtn = QtWidgets.QPushButton("Folding scheme")

        loadbtn.clicked.connect(self.window.openDialog)
        savebtn.clicked.connect(self.window.saveDialog)
        clearbtn.clicked.connect(self.window.clearDialog)
        sshotbtn.clicked.connect(self.window.takeScreenshot)
        seqbtn.clicked.connect(self.window.setSeq)
        platebtn.clicked.connect(self.window.generatePlates)
        pipettbtn.clicked.connect(self.window.pipettingScheme)
        foldbtn.clicked.connect(self.window.foldingScheme)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(loadbtn)
        hbox.addWidget(savebtn)
        hbox.addWidget(clearbtn)
        hbox.addWidget(sshotbtn)
        hbox.addWidget(seqbtn)
        hbox.addWidget(platebtn)
        hbox.addWidget(pipettbtn)
        hbox.addWidget(foldbtn)

        # set layout
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.window)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        # make white background
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background, QtCore.Qt.white)
        self.setPalette(palette)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

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
        if p.name == "design":
            p.execute()

    window.show()
    sys.exit(app.exec_())

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(window, "An error occured", message)
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook


if __name__ == "__main__":
    main()
