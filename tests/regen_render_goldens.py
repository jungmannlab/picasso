"""Regenerate the render golden archive from the current implementation.

Running this script redefines what "correct" rendering output is:
``tests/test_render_goldens.py`` compares every future render against the
arrays written here. Only run it when a rendering change is *intended*
(and review the resulting ``render_goldens.npz`` diff in git like any
other change).

Usage (from the repository root):

    python tests/regen_render_goldens.py

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from picasso import io  # noqa: E402
from tests._render_golden_scenes import (  # noqa: E402
    GOLDENS_PATH,
    SCENES,
    prepare_inputs,
    run_scene,
)


def main():
    locs, info = io.load_locs("./tests/data/testdata_locs.hdf5")
    inputs = prepare_inputs(locs)

    arrays = {}
    for scene in SCENES:
        out = run_scene(scene, inputs, info)
        for key, arr in out.items():
            # freezing an accidentally-empty render would make the test
            # a no-op, so refuse it here
            assert (
                np.abs(arr.astype(np.float64)).sum() > 0
            ), f"scene {scene['name']} produced an empty '{key}' array"
            arrays[f"{scene['name']}::{key}"] = arr
            print(
                f"{scene['name']:26s} {key:5s} {str(arr.shape):15s} "
                f"{str(arr.dtype):8s} sum={arr.astype(np.float64).sum():.6g}"
            )

    np.savez_compressed(GOLDENS_PATH, **arrays)
    size_kb = os.path.getsize(GOLDENS_PATH) / 1024
    print(f"\nwrote {len(arrays)} arrays to {GOLDENS_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
