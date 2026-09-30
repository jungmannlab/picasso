"""Scene table shared by the render golden tests and their regeneration
script.

The golden tests freeze the pixel output of the current rendering
implementation so that refactors (kernel threading, fused
post-processing, GPU backends) can prove they change performance without
changing pixels. Every scene is derived deterministically from the
committed ``tests/data/testdata_locs.hdf5`` fixture -- no extra data
files beyond the goldens archive itself.

Scene kinds:

* ``render``       -- raw float image from ``render.render`` (kernels)
* ``scene_single`` -- uint8 RGB + raw float image from
                      ``render._render_single_channel`` (colorize chain)
* ``scene_multi``  -- uint8 RGB + raw float stack from
                      ``render._render_multi_channel`` (multi-channel
                      colorize chain)

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np

from picasso import lib, render

GOLDENS_PATH = "./tests/data/render_goldens.npz"

#: nontrivial rotation (radians, legacy Euler convention around x, y, z)
ANG = (0.35, -0.6, 0.8)
FULL_VIEWPORT = ((0.0, 0.0), (32.0, 32.0))
#: offset sub-viewport with non-integer bounds (catches edge/rounding bugs)
SUB_VIEWPORT = ((6.7, 4.3), (21.9, 17.2))

#: keys each scene kind writes into the goldens archive
KIND_KEYS = {
    "render": ("image",),
    "scene_single": ("rgb", "raw"),
    "scene_multi": ("rgb", "raw"),
}

SCENES = [
    # --- kernel level: render.render raw float images ---
    dict(
        name="hist_full",
        kind="render",
        input="locs",
        blur=None,
        oversampling=10,
        viewport=FULL_VIEWPORT,
    ),
    dict(
        name="hist_sub",
        kind="render",
        input="locs",
        blur=None,
        oversampling=7.3,
        viewport=SUB_VIEWPORT,
    ),
    dict(
        name="gaussian_full",
        kind="render",
        input="locs",
        blur="gaussian",
        oversampling=10,
        viewport=FULL_VIEWPORT,
    ),
    dict(
        name="gaussian_sub",
        kind="render",
        input="locs",
        blur="gaussian",
        oversampling=7.3,
        viewport=SUB_VIEWPORT,
    ),
    dict(
        name="gaussian_iso_full",
        kind="render",
        input="locs",
        blur="gaussian_iso",
        oversampling=10,
        viewport=FULL_VIEWPORT,
    ),
    dict(
        name="smooth_full",
        kind="render",
        input="locs",
        blur="smooth",
        oversampling=10,
        viewport=FULL_VIEWPORT,
    ),
    dict(
        name="convolve_full",
        kind="render",
        input="locs",
        blur="convolve",
        oversampling=10,
        viewport=FULL_VIEWPORT,
    ),
    dict(
        name="gaussian_minblur_full",
        kind="render",
        input="locs",
        blur="gaussian",
        min_blur_width=0.2,
        oversampling=10,
        viewport=FULL_VIEWPORT,
    ),
    # per-loc in-plane precision-ellipse rotation (theta kernels)
    dict(
        name="gaussian_angle_full",
        kind="render",
        input="locs_angle",
        blur="gaussian",
        oversampling=10,
        viewport=FULL_VIEWPORT,
    ),
    # --- global 3D rotation paths ---
    dict(
        name="hist_rot",
        kind="render",
        input="locs3d",
        blur=None,
        oversampling=10,
        viewport=FULL_VIEWPORT,
        ang=ANG,
    ),
    dict(
        name="gaussian_rot",
        kind="render",
        input="locs3d",
        blur="gaussian",
        oversampling=10,
        viewport=FULL_VIEWPORT,
        ang=ANG,
    ),
    dict(
        name="gaussian_rot_angle",
        kind="render",
        input="locs3d_angle",
        blur="gaussian",
        oversampling=10,
        viewport=FULL_VIEWPORT,
        ang=ANG,
    ),
    # --- scene level: contrast -> LUT/colormap -> blend -> 8 bit ---
    dict(
        name="scene_single_magma",
        kind="scene_single",
        input="locs",
        blur="gaussian",
        oversampling=10,
        viewport=FULL_VIEWPORT,
        contrast=(0.0, 5.0),
        colormap="magma",
    ),
    dict(
        name="scene_multi_lut_bg",
        kind="scene_multi",
        input="channels3",
        blur="smooth",
        oversampling=10,
        viewport=FULL_VIEWPORT,
        contrast=(0.0, 4.0),
        colors="lut",
        relative_intensities=[1.0, 0.7, 1.3],
        background=(0.08, 0.08, 0.12),
    ),
    dict(
        name="scene_multi_solid_invert",
        kind="scene_multi",
        input="channels3",
        blur=None,
        oversampling=10,
        viewport=FULL_VIEWPORT,
        contrast=(0.0, 4.0),
        colors="solid",
        invert=True,
    ),
]


def prepare_inputs(locs):
    """Build every locs variant the scene table refers to,
    deterministically, from the committed fixture."""
    angle = ((np.arange(len(locs)) * 37) % 180).astype(np.float32)
    locs_angle = locs.copy()
    locs_angle["angle"] = angle

    # same recipe as the ``locs_3d`` fixture in test_render.py
    rng = np.random.default_rng(0)
    locs3d = locs.copy()
    locs3d["z"] = rng.uniform(-100.0, 100.0, size=len(locs)).astype(np.float32)
    locs3d_angle = locs3d.copy()
    locs3d_angle["angle"] = angle

    return {
        "locs": locs,
        "locs_angle": locs_angle,
        "locs3d": locs3d,
        "locs3d_angle": locs3d_angle,
        # strided split -> three channels with similar densities
        "channels3": [locs.iloc[i::3] for i in range(3)],
    }


def _resolve_colors(spec, n_channels):
    if spec == "lut":
        return [render.solid_to_lut(rgb) for rgb in lib.get_colors(n_channels)]
    if spec == "solid":
        return lib.get_colors(n_channels)
    raise ValueError(f"unknown colors spec: {spec}")


def run_scene(scene, inputs, info, via_backend=False):
    """Render one scene with the current implementation. Returns a dict
    of arrays keyed as in ``KIND_KEYS``."""
    return run_scene_counted(scene, inputs, info, via_backend)[1]


def run_scene_counted(scene, inputs, info, via_backend=False):
    """``run_scene`` that also returns the number of localizations
    rendered: ``(n, arrays)``.

    ``via_backend`` routes the kernel-level ``render`` scenes through
    ``render.scene._render_channels`` -- the splat-backend seam every
    scene-level render and the GUI go through -- instead of calling
    ``render.render`` (which always runs the CPU kernels). That is how
    the GPU golden comparison renders the whole table on a non-CPU
    backend; the scene-level kinds already go through the seam.
    """
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    disp_px_size = pixelsize / scene["oversampling"]
    kind = scene["kind"]

    if kind == "render":
        kwargs = dict(
            disp_px_size=disp_px_size,
            viewport=scene["viewport"],
            blur_method=scene["blur"],
            min_blur_width=scene.get("min_blur_width", 0.0),
            ang=scene.get("ang"),
        )
        if via_backend:
            ((n, image),) = render.scene._render_channels(
                [inputs[scene["input"]]], [info], **kwargs
            )
        else:
            n, image = render.render(inputs[scene["input"]], info, **kwargs)
        return n, {"image": image}

    if kind == "scene_single":
        n, rgb, _, raw = render._render_single_channel(
            inputs[scene["input"]],
            info,
            disp_px_size=disp_px_size,
            viewport=scene["viewport"],
            blur_method=scene["blur"],
            min_blur_width=scene.get("min_blur_width", 0.0),
            ang=scene.get("ang"),
            contrast=scene["contrast"],
            invert_colors=scene.get("invert", False),
            single_channel_colormap=scene["colormap"],
        )
        return n, {"rgb": rgb, "raw": raw}

    if kind == "scene_multi":
        channels = inputs[scene["input"]]
        n, rgb, _, raw = render._render_multi_channel(
            channels,
            [info] * len(channels),
            disp_px_size=disp_px_size,
            colors=_resolve_colors(scene["colors"], len(channels)),
            viewport=scene["viewport"],
            blur_method=scene["blur"],
            min_blur_width=scene.get("min_blur_width", 0.0),
            ang=scene.get("ang"),
            contrast=scene["contrast"],
            relative_intensities=scene.get("relative_intensities"),
            invert_colors=scene.get("invert", False),
            background_color=scene.get("background"),
        )
        return n, {"rgb": rgb, "raw": raw}

    raise ValueError(f"unknown scene kind: {kind}")
