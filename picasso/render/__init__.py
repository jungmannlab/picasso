"""
picasso.render
~~~~~~~~~~~~~~

Render single molecule localizations to a super-resolution image.

Provides functions for painting onto rendered images (QImage), such as
scale bar and picks.

This package splits the former single-module implementation into
``kernels`` (numba CPU kernels), ``geometry`` (viewport and rotation
math), ``backend`` (splat backend contract), ``splat`` (raw splat
stage and the CPU backend), ``overlays_qt`` (Qt overlay drawing and
export), ``scene`` (multi-channel composition), ``image_overlay`` (overlay
of an image, e.g., a widefield PNG or TIFF), ``grid`` (reference grid
in 2D and 3D), ``tracks`` (single-particle tracks), ``color_vision``
(simulation of color blindness) and ``animation``.
``picasso.render`` re-exports the full former surface, so
``render.<name>`` keeps working unchanged.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

# Every name imported below stays available as ``render.<name>`` (the
# GUI and the tests use many of them); ``__all__`` lists the ones meant
# for scripts, which the API documentation shows.
from .kernels import (  # noqa: F401
    _DRAW_MAX_SIGMA,
    _render_setup,
    _render_setup_anisotropic,
    _render_setup3d,
    _render_setup3d_anisotropic,
    _fill,
    _fill3d,
    _draw_gaussian_loc,
    _fill_gaussian,
    _draw_gaussian_theta_loc,
    _fill_gaussian_theta,
    _draw_gaussian_cov3d_loc,
    _draw_gaussian_rot_loc,
    _draw_gaussian_rot_theta_loc,
    _fill_gaussian_rot,
    _fill_gaussian_rot_theta,
    inverse_3x3,
    determinant_3x3,
    render_hist_numba,
    _compose_multi_lut,
    _quantize_rgb,
    _compose_single,
)
from .geometry import (  # noqa: F401
    rotation_matrix,
    to_rotation,
    closest_rotvec,
    viewport_height,
    viewport_width,
    viewport_size,
    viewport_center,
    shift_viewport,
    zoom_viewport,
    adjust_viewport_to_aspect_ratio,
    adjust_viewport_decorator,
    map_to_view,
)
from .backend import SplatBackend, SplatBackendError  # noqa: F401
from .splat import (  # noqa: F401
    render,
    CpuBackend,
    _CHUNKABLE_BLUR_METHODS,
    _MIN_CHUNK_LOCS,
    _render_worker_budget,
    _chunk_tasks,
    _RenderColumns,
    _extract_render_columns,
    _render_arrays,
    _render_hist,
    _render_hist_arrays,
    render_hist3d,
    render_hist3d_anisotropic,
    _render_gaussian,
    _render_gaussian_iso,
    _render_convolve,
    _render_smooth,
    _fftconvolve,
    locs_rotation,
    _locs_rotation_arrays,
)
from .overlays_qt import (  # noqa: F401
    POLYGON_POINTER_SIZE,
    BRUSH_FILL_ALPHA,
    LINE_STYLES,
    OverlayStyle,
    export_qimage_to_pdf,
    export_qimage_to_svg,
    get_rectangle_pick_polygon,
    _draw_picks_circle,
    _draw_picks_rectangle,
    _draw_picks_polygon,
    _draw_picks_square,
    _draw_picks_box,
    brush_pick_path,
    _draw_picks_brush,
    draw_picks,
    draw_points,
    draw_scalebar,
    draw_legend,
    _format_tick,
    _colorbar_layout,
    _paint_colorbar,
    colorbar_image,
    colorbar_svg,
    save_colorbar,
    draw_minimap,
    draw_rotation,
    draw_rotation_angles,
    rgb_to_qimage,
    optimal_scalebar_length,
)
from .image_overlay import (  # noqa: F401
    SCALING_MODES as OVERLAY_SCALING_MODES,
    BLEND_MODES as OVERLAY_BLEND_MODES,
    GRAYSCALE_COLORS as OVERLAY_GRAYSCALE_COLORS,
    IMAGE_EXTENSIONS as OVERLAY_IMAGE_EXTENSIONS,
    load_overlay_image,
    count_image_pages,
    overlay_extent,
    overlay_to_qimage,
    draw_image_overlay,
    composite_behind,
)
from .grid import (  # noqa: F401
    AUTO_GRID_FRACTION,
    GRID_BLEND_MODES,
    GRID_PLANES,
    GRID_POSITIONS,
    GRID_KINDS,
    GridOverlay,
    grid_levels,
    grid_box,
    grid_segments_3d,
    format_grid_labels,
    grid_label_unit,
    project_points,
    draw_grid,
    draw_grid_3d,
)
from .color_vision import (  # noqa: F401
    COLOR_VISION_DEFICIENCIES,
    simulate_color_vision,
    simulate_color_vision_qimage,
)
from .tracks import (  # noqa: F401
    MAX_TRACKS_DEFAULT,
    select_track_segments,
    draw_tracks,
)
from .scene import (  # noqa: F401
    N_GROUP_COLORS,
    solid_to_lut,
    stops_to_lut,
    get_colors_from_colormap,
    get_group_color,
    render_scene,
    color_range,
    _render_channels,
    _contrast_limits,
    _resolve_cmap,
    _render_multi_channel,
    _render_single_channel,
    scale_contrast,
    to_8bit,
    apply_colormap,
    split_locs_by_property,
    split_locs_by_group,
)
from .animation import (  # noqa: F401
    ANIMATION_TRANSITIONS,
    _normalize_animation_positions,
    _animation_sequence,
    build_animation,
    _build_animation,
    _adjust_disp_px_size,
    _adjust_contrast,
)

__all__ = [
    # rendering
    "render",
    "render_scene",
    "render_hist3d",
    "render_hist3d_anisotropic",
    "locs_rotation",
    "rotation_matrix",
    "build_animation",
    "ANIMATION_TRANSITIONS",
    # colors and contrast
    "scale_contrast",
    "apply_colormap",
    "to_8bit",
    "solid_to_lut",
    "stops_to_lut",
    "get_colors_from_colormap",
    "get_group_color",
    "color_range",
    "split_locs_by_property",
    "split_locs_by_group",
    "rgb_to_qimage",
    "COLOR_VISION_DEFICIENCIES",
    "simulate_color_vision",
    "simulate_color_vision_qimage",
    # viewport
    "viewport_center",
    "viewport_size",
    "viewport_width",
    "viewport_height",
    "shift_viewport",
    "zoom_viewport",
    "map_to_view",
    # drawing onto rendered images
    "OverlayStyle",
    "LINE_STYLES",
    "draw_picks",
    "draw_points",
    "draw_scalebar",
    "optimal_scalebar_length",
    "draw_legend",
    "GridOverlay",
    "draw_grid",
    "draw_grid_3d",
    "select_track_segments",
    "draw_tracks",
    # image overlays (e.g., widefield images)
    "load_overlay_image",
    "overlay_extent",
    "overlay_to_qimage",
    "draw_image_overlay",
    "composite_behind",
    # color bars and export
    "colorbar_image",
    "colorbar_svg",
    "save_colorbar",
    "export_qimage_to_pdf",
    "export_qimage_to_svg",
]
