"""Test ``picasso.design``: the plate files and the conversion of the
canvas index to plate positions for ordering the staples.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from picasso import design, io

CANVAS = [
    ["A1", "staple_a1", "AAAA"],
    ["I1", "staple_i1", "CCCC"],
    ["P12", "staple_p12", "GGGG"],
]


def _wells(plate):
    """``{(plate name, position): row}`` of a converted plate."""
    return {(row[0], row[1]): row for row in plate[1:]}


def test_plates_round_trip(tmp_path):
    path = str(tmp_path / "plates.csv")
    plates = [
        [["PLATE NAME", "PLATE POSITION"], ["CUSTOM_1", "A1"]],
        [["CUSTOM_2", "B2"]],
    ]
    design.save_plate(path, plates)
    assert design.read_plate(path) == [
        ["PLATE NAME", "PLATE POSITION"],
        ["CUSTOM_1", "A1"],
        ["CUSTOM_2", "B2"],
    ]


def test_canvas_index_to_plate_positions():
    plate = design.convert_plate_index(CANVAS, "CUSTOM")
    assert plate[0] == [
        "PLATE NAME",
        "PLATE POSITION",
        "OLIGO NAME",
        "SEQUENCE",
    ]
    assert len(plate) == 1 + 2 * 96  # two 96-well plates
    wells = _wells(plate)
    # canvas rows A-H fill the first plate, I-P the second
    assert wells[("CUSTOM_1", "A1")][2:] == ["staple_a1", "AAAA"]
    assert wells[("CUSTOM_2", "A1")][2:] == ["staple_i1", "CCCC"]
    assert wells[("CUSTOM_2", "H12")][2:] == ["staple_p12", "GGGG"]
    assert wells[("CUSTOM_1", "B1")][2:] == [" ", " "]


def test_empty_wells_have_no_color():
    canvas = [row + ["red"] for row in CANVAS[1:]]  # no oligo in A1
    plate = design.convert_plate_index_color(canvas, "CUSTOM")
    wells = _wells(plate)
    assert wells[("CUSTOM_1", "A1")][2:] == [" ", " ", " "]
    assert wells[("CUSTOM_2", "A1")][4] == "red"
    assert wells[("CUSTOM_2", "A2")][4] == " "  # not the color before it


def test_save_info(tmp_path):
    path = str(tmp_path / "design.yaml")
    design.save_info(path, {"Structure": 1, "Extensions Short": [1, 2]})
    with open(path) as file:
        saved = io._parse_yaml(file, all_documents=True)
    assert saved == [{"Structure": 1, "Extensions Short": [1, 2]}]
