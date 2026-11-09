"""Tests of release/collect_third_party_licenses.py, which writes the
license notices shipped with the one-click installers.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import email.message
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[1] / "release" / "collect_third_party_licenses.py"
)
_spec = importlib.util.spec_from_file_location("collect_licenses", _SCRIPT)
collect = importlib.util.module_from_spec(_spec)
# registered first: its dataclass looks its module up there
sys.modules[_spec.name] = collect
_spec.loader.exec_module(collect)


def _meta(license=None, classifiers=()):
    meta = email.message.Message()
    if license is not None:
        meta["License"] = license
    for classifier in classifiers:
        meta["Classifier"] = classifier
    return meta


@pytest.mark.parametrize(
    "declared", ["LICENSE.txt", "LICENSE", "COPYING", "license.md"]
)
def test_a_declared_file_name_is_not_a_license_name(declared):
    assert collect.names_a_file(declared)


@pytest.mark.parametrize(
    "declared", ["MIT", "BSD-3-Clause", "LicenseRef-Proprietary", "GPLv3"]
)
def test_license_names_are_kept(declared):
    assert not collect.names_a_file(declared)
    assert collect.license_name(_meta(declared)) == declared


def test_a_declared_file_name_falls_back_to_the_classifiers():
    """vectorfit-py declares "License: LICENSE.txt"."""
    meta = _meta(
        "LICENSE.txt",
        ["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"],
    )
    assert collect.license_name(meta) == (
        "GNU General Public License v3 (GPLv3)"
    )
    assert collect.license_name(_meta("LICENSE.txt")) == (
        "see the license text below"
    )


@pytest.mark.parametrize("name", ["vectorfit-py", "vectorfit-py-cu12"])
def test_vectorfit_carries_the_notice_of_its_bundled_fftw(name):
    ((label, text),) = collect.BUNDLED_NOTICES[name]
    assert "FFTW" in label
    assert "GNU General Public License" in text
    assert "https://www.fftw.org" in text
