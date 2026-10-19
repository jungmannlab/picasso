"""Calibrations written before Picasso 1.0 in Picasso: Localize.

Their spot widths and lateral corrections are in camera pixels; Localize
converts them to nm with the camera pixel size of its Parameters dialog.
Changing that pixel size after loading (e.g., when the camera's is set)
converts them again, so the fit never uses a stale conversion. Spline
calibrations record the pixel size they were made with and use it.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from picasso import io, lib, transforms
from picasso.gui import localize as gui_localize

X_COEFFS = [0.0, 0.0, 1e-6, 2e-4, 1.1]  # camera pixels (spot width)
Y_COEFFS = [0.0, 0.0, 1e-6, -2e-4, 1.1]
SHIFT = (1.0, -0.5)  # camera pixels


@pytest.fixture
def dialog(qt_offscreen, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    window = gui_localize.Window()
    dialog = window.parameters_dialog
    dialog.pixelsize.setValue(100.0)
    yield dialog
    window.close()


def _z_calibration(path, **extra):
    io.save_yaml(
        str(path),
        {
            "X Coefficients": X_COEFFS,
            "Y Coefficients": Y_COEFFS,
            "Magnification factor": 0.79,
            **extra,
        },
    )
    return str(path)


def _lateral_entry():
    shift = transforms.TranslationTransform.from_shift(SHIFT)
    return {"Type": "chromatic", "Transform": shift.to_dict()}


def _shift_of(entries):
    model = transforms.from_dict(entries[0]["Transform"])
    return model.apply([[0.0, 0.0]])[0]


def test_legacy_z_calibration_follows_the_pixel_size(dialog, tmp_path):
    path = _z_calibration(tmp_path / "old_z.yaml")  # no format marker
    dialog.update_z_calib(path)
    np.testing.assert_allclose(
        dialog.z_calibration["X Coefficients"], np.multiply(X_COEFFS, 100)
    )
    # the user turned the z fit off; setting the camera's pixel size
    # converts the calibration again and keeps that choice
    dialog.fit_z_checkbox.setChecked(False)
    dialog.pixelsize.setValue(130.0)
    np.testing.assert_allclose(
        dialog.z_calibration["X Coefficients"], np.multiply(X_COEFFS, 130)
    )
    assert dialog.z_calibration_path == path
    assert not dialog.fit_z_checkbox.isChecked()


def test_current_z_calibration_ignores_the_pixel_size(dialog, tmp_path):
    nm = list(np.multiply(X_COEFFS, 130))
    path = str(tmp_path / "new_z.yaml")
    io.save_calibration(
        path,
        {
            "X Coefficients": nm,
            "Y Coefficients": nm,
            "Magnification factor": 0.79,
        },
    )
    dialog.update_z_calib(path)
    dialog.pixelsize.setValue(160.0)
    np.testing.assert_allclose(dialog.z_calibration["X Coefficients"], nm)
    assert dialog.fit_z_checkbox.isChecked()


def test_legacy_lateral_correction_follows_the_pixel_size(dialog, tmp_path):
    path = str(tmp_path / "old_lateral.yaml")
    io.save_yaml(path, {lib.LATERAL_TRANSFORMS_KEY: [_lateral_entry()]})
    dialog.update_affine_calib([path])
    np.testing.assert_allclose(
        _shift_of(dialog.lateral_transforms), np.multiply(SHIFT, 100)
    )
    dialog.pixelsize.setValue(130.0)
    np.testing.assert_allclose(
        _shift_of(dialog.lateral_transforms), np.multiply(SHIFT, 130)
    )
    assert dialog.affine_calibration_paths == [path]


def test_legacy_spline_calibration_uses_its_own_pixel_size(dialog, tmp_path):
    path = str(tmp_path / "old_spline.hdf5")
    metadata = {
        "model": "spline-3d",
        "pixelsize": 130.0,
        lib.LATERAL_TRANSFORMS_KEY: [_lateral_entry()],
    }
    with h5py.File(path, "w") as f:
        f.create_dataset("coefficients", data=np.zeros((2, 64), "f4"))
        f.attrs["metadata"] = json.dumps(metadata)
    dialog.update_spline_calib(path)  # the dialog shows 100 nm
    np.testing.assert_allclose(
        _shift_of(dialog.spline_calibration[lib.LATERAL_TRANSFORMS_KEY]),
        np.multiply(SHIFT, 130),
    )
