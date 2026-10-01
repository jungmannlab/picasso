"""Tests for ``picasso.localize.localize_frames`` — the GUI-free streaming
wrapper around identify + fit (S0B-2 contract 3).

Kept in its own file, deliberately free of any Qt/GUI import, so it doubles
as evidence that the wrapper is importable and runnable without a display.

Verification tiers (scientific correctness):

* ORACLE      — the wrapper must reproduce the existing batch identify+fit
                (``localize.localize``) spot for spot on the same frames and
                parameters; it changes no result.
* PROPERTY    — every frame is localized and the absolute ``frame`` indices
                are contiguous across arbitrary batch boundaries, i.e.
                batch-of-1 == batch-of-N concatenated == the whole movie.
* KNOWN-ANSWER — on simulated frames with known emitter positions the
                recovered positions fall within tolerance of ground truth.
* GUI-FREE    — importing and running the wrapper pulls in no Qt.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from picasso import localize

# Bundled-movie parameters, matching the rest of the suite (see conftest).
BOX = 7
MIN_NG = 5000
CAMERA_INFO = {"Baseline": 0, "Sensitivity": 1, "Gain": 1, "Pixelsize": 130}
PARAMS = {"Min. Net Gradient": MIN_NG, "Box Size": BOX}


def _assert_locs_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Assert two localization tables are equal column by column.

    Integer columns (``frame``) must match exactly; floating-point columns
    within a tight tolerance. The rows are aligned by sorting on the fitted
    coordinates first, so a difference in row order does not read as a
    difference in the result.
    """
    assert list(a.columns) == list(b.columns)
    a = a.sort_values(["frame", "x", "y"]).reset_index(drop=True)
    b = b.sort_values(["frame", "x", "y"]).reset_index(drop=True)
    assert len(a) == len(b)
    for col in a.columns:
        va, vb = a[col].to_numpy(), b[col].to_numpy()
        assert va.dtype == vb.dtype, f"{col}: dtype {va.dtype} != {vb.dtype}"
        if va.dtype.kind in "iu":
            assert np.array_equal(va, vb), f"{col}: integer columns differ"
        else:
            np.testing.assert_allclose(
                va,
                vb,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
                err_msg=f"{col}: float columns differ",
            )


# ---------------------------------------------------------------------------
# ORACLE — parity with the existing batch identify+fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fitting_method", ["gausslq", "gaussmle"])
def test_matches_batch_localize(movie, movie_info, fitting_method):
    """localize_frames == localize on the same frames and parameters."""
    ref, _ = localize.localize(
        movie,
        camera_info=CAMERA_INFO,
        identification_parameters=PARAMS,
        movie_info=movie_info,
        fitting_method=fitting_method,
        threaded=True,
        return_info=True,
    )
    out = localize.localize_frames(
        np.asarray(movie),
        list(movie_info) + [CAMERA_INFO],
        PARAMS,
        fitting_method=fitting_method,
    )
    _assert_locs_equal(ref, out)


def test_camera_info_read_from_info(movie, movie_info):
    """The camera parameters are taken from the info list-of-dicts when no
    ``camera_info`` is passed (the streaming contract carries them there)."""
    explicit = localize.localize_frames(
        np.asarray(movie), movie_info, PARAMS, camera_info=CAMERA_INFO
    )
    from_info = localize.localize_frames(
        np.asarray(movie), list(movie_info) + [CAMERA_INFO], PARAMS
    )
    _assert_locs_equal(explicit, from_info)


def test_does_not_mutate_camera_info(movie, movie_info):
    """A caller-supplied camera_info is not mutated (fit fills a missing
    "Pixelsize" in place; the wrapper copies first so a dict reused across
    streaming batches is left untouched)."""
    camera_info = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
    before = dict(camera_info)
    localize.localize_frames(
        np.asarray(movie), movie_info, PARAMS, camera_info=camera_info
    )
    assert camera_info == before


def test_accepts_loaded_movie_unchanged(movie, movie_info):
    """Passing an already-loaded movie (a memmap) is used as-is and matches
    passing the same data as a plain ndarray."""
    as_array = localize.localize_frames(
        np.asarray(movie), movie_info, PARAMS, camera_info=CAMERA_INFO
    )
    as_movie = localize.localize_frames(
        movie, movie_info, PARAMS, camera_info=CAMERA_INFO
    )
    _assert_locs_equal(as_array, as_movie)


# ---------------------------------------------------------------------------
# PROPERTY — absolute, contiguous frame indices across arbitrary batches
# ---------------------------------------------------------------------------


def _run_in_batches(frames, boundaries, **kwargs):
    """Localize ``frames`` split at ``boundaries`` (cumulative frame counts),
    each batch offset by its absolute start, then concatenate."""
    parts, start = [], 0
    for end in boundaries:
        chunk = frames[start:end]
        if len(chunk):
            parts.append(
                localize.localize_frames(
                    chunk, None, PARAMS, start_frame=start, **kwargs
                )
            )
        start = end
    return pd.concat(parts, ignore_index=True)


def test_batch_boundaries_are_invariant(movie):
    """batch-of-1 == arbitrary batches == the whole movie in one call."""
    frames = np.asarray(movie)
    n = len(frames)
    full = localize.localize_frames(
        frames, None, PARAMS, camera_info=CAMERA_INFO
    )
    one = _run_in_batches(
        frames, list(range(1, n + 1)), camera_info=CAMERA_INFO
    )
    arbitrary = _run_in_batches(
        frames, [3, 3, 7, 12, 25, n], camera_info=CAMERA_INFO
    )
    _assert_locs_equal(full, one)
    _assert_locs_equal(full, arbitrary)


def test_start_frame_offsets_indices(movie):
    """``start_frame`` shifts every frame index by a constant and preserves
    the column dtype (so a batch starting at 0 is unchanged)."""
    frames = np.asarray(movie)
    base = localize.localize_frames(
        frames, None, PARAMS, camera_info=CAMERA_INFO
    )
    shifted = localize.localize_frames(
        frames, None, PARAMS, camera_info=CAMERA_INFO, start_frame=1000
    )
    assert base["frame"].dtype == shifted["frame"].dtype
    np.testing.assert_array_equal(
        shifted["frame"].to_numpy(), base["frame"].to_numpy() + 1000
    )
    # start_frame=0 is a no-op on the frame column.
    zero = localize.localize_frames(
        frames, None, PARAMS, camera_info=CAMERA_INFO, start_frame=0
    )
    np.testing.assert_array_equal(
        zero["frame"].to_numpy(), base["frame"].to_numpy()
    )


# ---------------------------------------------------------------------------
# KNOWN-ANSWER — recover known emitter positions from simulated frames
# ---------------------------------------------------------------------------


def _simulate_movie(positions, n_frames, size, photons, sigma, bg, seed):
    """Render Gaussian emitters at fixed ``positions`` onto a Poisson-noisy
    movie. Returns ``(frames, camera_info)``; the trivial ``CAMERA_INFO``
    (Baseline 0, Sensitivity/Gain 1) makes the counts photons directly."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    signal = np.zeros((size, size), dtype=np.float64)
    for x0, y0 in positions:
        signal += photons * np.exp(
            -(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
        )
    frames = rng.poisson(signal + bg, size=(n_frames, size, size))
    frames = frames.astype(np.uint16)
    return frames, CAMERA_INFO


def test_recovers_known_positions():
    """Recovered positions fall within a tolerance of the ground truth."""
    positions = [(8.0, 8.0), (20.0, 24.0), (25.0, 10.0)]
    frames, camera_info = _simulate_movie(
        positions,
        n_frames=20,
        size=32,
        photons=800.0,
        sigma=1.2,
        bg=5.0,
        seed=42,
    )
    params = {"Min. Net Gradient": 2000, "Box Size": 7}
    locs = localize.localize_frames(
        frames,
        None,
        params,
        camera_info=camera_info,
        fitting_method="gaussmle",
    )
    # Every frame yields all emitters (bright, well separated).
    assert set(locs["frame"].unique()) == set(range(20))
    # Each ground-truth emitter has a nearby recovered localization.
    for x0, y0 in positions:
        d = np.hypot(locs["x"].to_numpy() - x0, locs["y"].to_numpy() - y0)
        best = d.min()
        assert best < 0.3, f"emitter ({x0},{y0}) recovered {best:.3f} px off"
    # Averaged over frames the bias is well below a tenth of a pixel.
    for x0, y0 in positions:
        near = locs[np.hypot(locs["x"] - x0, locs["y"] - y0) < 1.0]
        assert abs(near["x"].mean() - x0) < 0.1
        assert abs(near["y"].mean() - y0) < 0.1


# ---------------------------------------------------------------------------
# GUI-FREE — importable and runnable with no display, no Qt on the call path
# ---------------------------------------------------------------------------


def test_no_qt_on_call_path():
    """A fresh interpreter can import and run localize_frames without Qt.

    Run in a subprocess with no ``DISPLAY`` and ``QT_QPA_PLATFORM`` cleared so
    that any accidental Qt import would surface, then assert no Qt binding was
    imported.
    """
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from picasso import localize

        frames = np.zeros((3, 16, 16), dtype=np.uint16)
        frames[:, 8, 8] = 500
        camera_info = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        params = {"Min. Net Gradient": 10, "Box Size": 7}
        locs = localize.localize_frames(frames, None, params,
                                        camera_info=camera_info)
        assert "frame" in locs.columns

        leaked = [m for m in ("PyQt5", "PyQt6", "PySide2", "PySide6")
                  if m in sys.modules]
        assert not leaked, f"Qt imported on the localize path: {leaked}"
        print("ok")
        """
    )
    env = {
        k: v
        for k, v in __import__("os").environ.items()
        if k not in ("DISPLAY", "QT_QPA_PLATFORM")
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ok" in result.stdout
