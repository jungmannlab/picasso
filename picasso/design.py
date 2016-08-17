"""
    picasso.design
    ~~~~~~~~~~~~~~~~

    Design rectangular rothemund origami

    :author: Maximilian Thomas Strauss, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, Max Planck Institute of Biochemistry
"""
import numpy as _np
from . import io as _io


def saveInfo(filename,info):
    _io.save_info(filename, [info], default_flow_style=True)

def convertToPlate(plate,platename):
    # convert the origami from format [ROW,COL,COLOR] to a format for ordering [PLATE NAME, PLATE POSITION, OLIGONAME, SEQUENCE]

    platerow = ['A','B','C','D','E','F','G','H','A','B','C','D','E','F','G','H']
    platecol = [1,2,3,4,5,6,7,8,9,10,11,12]
    structurerow = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
    structurecol = [1,2,3,4,5,6,7,8,9,10,11,12]

    newplate = [['PLATE NAME','PLATE POSITION','OLIGO NAME','SEQUENCE']]
    for row in range(0,len(platerow)):
        for col in range(0,len(platecol)):
            if row < 8:
                platenameindex = platename+'_1'
            else:
                platenameindex = platename+'_2'
            structureindex = structurerow[row]+str(structurecol[col])
            oligoname = ' '
            sequence = ' '
            for i in range(0,len(plate)):
                if plate[i][0] == structureindex:
                    oligoname = plate[i][1]
                    sequence = plate[i][2]
            newplate.append([platenameindex,platerow[row]+str(platecol[col]),oligoname,sequence])

    return newplate
