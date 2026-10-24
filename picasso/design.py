"""
picasso.design
~~~~~~~~~~~~~~

Design rectangular rothemund origami (RRO).

:authors: Maximilian Thomas Strauss
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

import csv

from . import io


def save_info(filename: str, info: dict) -> None:
    """Save information to a YAML file.

    Parameters
    ----------
    filename : str
        Path of the file the metadata belongs to; the YAML is written next to
        it (see ``io.save_info``).
    info : dict
        The metadata to save.
    """
    io.save_info(filename, [info], default_flow_style=True)


def convert_plate_index(plate: list, platename: str) -> list:
    """Convert plate index from canvas index format to a structured
    format for ordering the sequences.

    Parameters
    ----------
    plate : list
        List of lists containing plate information in canvas index
        format.
    platename : str
        Name of the plate to be used in the output.

    Returns
    -------
    newplate : list
        List of lists containing the plate information in a structured
        format for ordering the sequences.
    """
    # convert from canvas index [CANVAS_INDEX, OLIGONAME, SEQUENCE]
    # format for ordering [PLATE NAME, PLATE POSITION, OLIGONAME, SEQUENCE]
    platerow = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
    ]
    platecol = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    structurerow = [
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
    structurecol = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    newplate = [["PLATE NAME", "PLATE POSITION", "OLIGO NAME", "SEQUENCE"]]
    for row in range(0, len(platerow)):
        for col in range(0, len(platecol)):
            if row < 8:
                platenameindex = platename + "_1"
            else:
                platenameindex = platename + "_2"
            structureindex = structurerow[row] + str(structurecol[col])
            oligoname = " "
            sequence = " "
            for i in range(0, len(plate)):
                if plate[i][0] == structureindex:
                    oligoname = plate[i][1]
                    sequence = plate[i][2]
            newplate.append(
                [
                    platenameindex,
                    platerow[row] + str(platecol[col]),
                    oligoname,
                    sequence,
                ]
            )

    return newplate


def convert_plate_index_color(plate: list, platename: str) -> list:
    """Convert plate index from canvas index format to a structured
    format for ordering the sequences, including color information.

    Parameters
    ----------
    plate : list
        List of lists containing plate information in canvas index
        format.
    platename : str
        Name of the plate to be used in the output.

    Returns
    -------
    newplate : list
        List of lists containing the plate information in a structured
        format for ordering the sequences, including color information.
    """
    # convert from canvas index [CANVAS_INDEX, OLIGONAME, SEQUENCE]
    # format for ordering [PLATE NAME, PLATE POSITION, OLIGONAME, SEQUENCE,
    # COLOR]
    platerow = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
    ]
    platecol = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    structurerow = [
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
    structurecol = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    newplate = [
        ["PLATE NAME", "PLATE POSITION", "OLIGO NAME", "SEQUENCE", "COLOR"]
    ]
    for row in range(0, len(platerow)):
        for col in range(0, len(platecol)):
            if row < 8:
                platenameindex = platename + "_1"
            else:
                platenameindex = platename + "_2"
            structureindex = structurerow[row] + str(structurecol[col])
            oligoname = " "
            sequence = " "
            color = " "  # an empty well has no color
            for i in range(0, len(plate)):
                if plate[i][0] == structureindex:
                    oligoname = plate[i][1]
                    sequence = plate[i][2]
                    color = plate[i][3]
            newplate.append(
                [
                    platenameindex,
                    platerow[row] + str(platecol[col]),
                    oligoname,
                    sequence,
                    color,
                ]
            )

    return newplate


def read_plate(filename: str) -> list:
    """Read a plate file and return its content as a list of lists.

    Parameters
    ----------
    filename : str
        The name of the file to read.

    Returns
    -------
    data : list
        A list of lists containing the data from the file.
    """
    with open(filename, newline="") as file:
        data = list(csv.reader(file))
    return data


def save_plate(filename: str, data: list | dict) -> None:
    """Save the plate data to a CSV file.

    Parameters
    ----------
    filename : str
        The name of the file to save the data to.
    data : list or dict
        The plates to save, one after the other: a list of plates or a
        dict of plates by number (0, 1, ...). Each plate is a list of
        rows.
    """
    plates = data.values() if isinstance(data, dict) else data
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for plate in plates:
            for row in plate:
                writer.writerow(row)
