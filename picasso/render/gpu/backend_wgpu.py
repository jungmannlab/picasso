"""
picasso.render.gpu.backend_wgpu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offscreen GPU splat backend on wgpu (WebGPU: Metal/D3D12/Vulkan).

Splatting runs in compute shaders rather than rasterized quads: on
tile-based GPUs (Apple in particular) the per-primitive cost of tens of
millions of tiny quads dwarfs the actual arithmetic, and measured
compute is more than an order of magnitude faster.

Histogram counts scatter with integer ``atomicAdd`` (bit-exact and
deterministic); ``smooth`` and ``convolve`` are that histogram followed
by the CPU image filter the CPU backend uses (``splat._fftconvolve``),
so they match it exactly. Gaussian splatting is a *gather* — every
output pixel is owned by one thread that sums the contributions
reaching it in a register, so no atomics touch the image, the sums are
plain float32, and nothing can overflow. Localizations are routed by
footprint into three tiers, because the work profile of SMLM data
spans a few orders of magnitude (sub-pixel blurs by the milion at
overview zoom, blurs covering the whole view at deep zoom):

- **pixel tier** (3σ below one pixel — nearly all localizations at
  overview zoom): a per-render counting sort by center pixel; each
  pixel's thread evaluates only the segments of its 3×3 neighborhood,
  which is where such a footprint can reach;
- **tile tier** (footprints spanning up to ``_WHALE_MAX_TILES`` 16×16
  tiles): binned into every tile they overlap, then streamed through
  workgroup memory by the tile's 256 threads;
- **whales** (wider still): a pixel-parallel pass over the whole
  image, looping the (short) whale list per pixel.

Every Gaussian variant is one bivariate Gaussian with a 2x2 covariance:
``gaussian``/``gaussian_iso`` are diagonal, a per-localization
``angle`` rotates the precision ellipse in-plane
(``kernels._draw_gaussian_theta_loc``), and a 3D rotation ``ang``
rotates the localizations about the view center in-shader and
projects ``R·cov·Rᵀ`` (``kernels._draw_gaussian_cov3d_loc``). The
arithmetic mirrors the CPU kernels exactly (strict in-view mask,
truncation-based ±3σ footprint bounds, the same normalized Gaussian at
pixel centers, the same degenerate-covariance guard), so both backends
touch the identical pixel set and differ only by float32-vs-float64
rounding. One caveat: the order in which a pixel's contributions are
summed follows the (parallel, nondeterministic) bin order, so repeated
renders may differ in the last float bits — unlike the CPU backend,
which is deterministic for a given worker budget.

Every blur method is GPU-rendered, rotated or not: ``convolve``'s blur
is the channel's global precision (``splat.global_blur``), so no
in-view mask is ever needed.

Uploads are resident: a channel's columns are transferred once and
reused by every later render of the same memory (the cache is keyed
by the data pointers behind the arrays, so the GUI hands over whole
channels rather than viewport slices — see
``SplatBackend.persistent_uploads``). Residency is bounded by
``settings["Render"]["gpu"]["vram_budget_mb"]`` with least recently
rendered channels evicted first; a single channel larger than the
budget is rendered in single-use row chunks whose images are summed.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from math import ceil
from typing import Literal, TYPE_CHECKING

import numpy as np
import wgpu

from ... import lib
from ..backend import SplatBackend, SplatBackendError, vram_budget_bytes
from ..geometry import to_rotation

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation

    from ..splat import _RenderColumns

_WORKGROUP = 256
_MAX_WORKGROUPS = 65_535  # per dispatch dimension; grid-stride covers more
_TILE = 16  # gather tile edge in pixels; 16x16 = one thread per pixel
#: a localization whose footprint overlaps more tiles than this is a
#: "whale" and takes the pixel-parallel pass instead of the bins
_WHALE_MAX_TILES = 64
_WHALE_CAPACITY = 1_000_000  # whale list slots at the start of the bins
_UNIFORM_BYTES = 144
#: storage buffers bound per compute stage (8 scratch + 7 channel
#: columns + the index list); WebGPU's baseline guarantee is 8, desktop
#: adapters offer far more — a refusal simply leaves rendering on the CPU
_STORAGE_BINDINGS_WANTED = 16
_MODE_ISO = 1  # sx = sy = mean of the two precisions
_MODE_THETA = 2  # per-localization in-plane ellipse angle
_MODE_ROT = 4  # 3D rotation about the view center
_MODE_INDEXED = 8  # rows come from an index list into the columns

#: the uniform block shared by both shader modules
_UNIFORMS_WGSL = """
struct Uniforms {
    os: f32,
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
    min_blur: f32,
    npx_f: f32,
    npy_f: f32,
    n_locs: u32,
    npx: u32,
    npy: u32,
    ntx: u32,
    nty: u32,
    tile_base: u32,   // per-channel slice of the persisted tile offsets
    whale_cap: u32,
    bin_cap: u32,     // tile-tier bin entries (after the whale slots)
    px_cap: u32,      // pixel-tier sorted entries
    mode: u32,        // MODE_* bits
    stride: u32,      // column index = offset + i * stride (strided
    offset: u32,      // previews read the resident buffers directly)
    rot: mat3x3<f32>,     // 3D rotation (MODE_ROT)
    center: vec2<f32>,    // rotation center (camera px)
    _pad2: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
const MODE_ISO: u32 = 1u;
const MODE_THETA: u32 = 2u;
const MODE_ROT: u32 = 4u;
const MODE_INDEXED: u32 = 8u;
"""

_CHANNEL_WGSL = """
@group(1) @binding(0) var<storage, read> xs: array<f32>;
@group(1) @binding(1) var<storage, read> ys: array<f32>;
@group(1) @binding(2) var<storage, read> lpxs: array<f32>;
@group(1) @binding(3) var<storage, read> lpys: array<f32>;
@group(1) @binding(4) var<storage, read> zs: array<f32>;
@group(1) @binding(5) var<storage, read> lpzs: array<f32>;
@group(1) @binding(6) var<storage, read> angles: array<f32>;
// rows to render when MODE_INDEXED (the viewport pyramid's selection);
// otherwise unused and bound to a placeholder
@group(1) @binding(7) var<storage, read> indices: array<u32>;

// index into the column buffers: a request may be a strided view
// (interactive preview) of the resident channel, or an index list
fn src(i: u32) -> u32 {
    let k = u.offset + i * u.stride;
    if ((u.mode & MODE_INDEXED) != 0u) {
        return indices[k];
    }
    return k;
}

// camera-pixel position, rotated about the view center when MODE_ROT
// (as _locs_rotation_arrays does, before the in-view mask)
fn position(i: u32) -> vec2<f32> {
    let k = src(i);
    var x = xs[k];
    var y = ys[k];
    if ((u.mode & MODE_ROT) != 0u) {
        let p = u.rot * vec3<f32>(x - u.center.x, y - u.center.y, zs[k]);
        x = p.x + u.center.x;
        y = p.y + u.center.y;
    }
    return vec2<f32>(x, y);
}

fn visible(p: vec2<f32>) -> bool {
    // strict in-view mask, identical to _render_setup's
    return p.x > u.x_min && p.x < u.x_max && p.y > u.y_min && p.y < u.y_max;
}
"""

# --------------------------------------------------------------------
# scatter module: histogram (integer atomics on the image)
# --------------------------------------------------------------------
_SCATTER_WGSL = (
    f"const WORKGROUP: u32 = {_WORKGROUP}u;\n"
    + _UNIFORMS_WGSL
    + """
@group(0) @binding(1) var<storage, read_write> image: array<atomic<u32>>;
// aux[0] = in-view count
@group(0) @binding(2) var<storage, read_write> aux: array<atomic<u32>, 4>;
"""
    + _CHANNEL_WGSL
    + """
var<workgroup> wg_count: atomic<u32>;

// Reduces the per-invocation in-view counts of a workgroup into
// aux[0]. Workgroup memory is zeroed explicitly: WGSL promises
// zero-initialization, but an NVIDIA RTX A5500 on Windows handed each
// workgroup the previous one's count (n inflated ~6x) while Metal was
// clean -- never rely on the driver for it.
fn flush_count(local_index: u32, my_count: u32) {
    if (local_index == 0u) {
        atomicStore(&wg_count, 0u);
    }
    workgroupBarrier();
    atomicAdd(&wg_count, my_count);
    workgroupBarrier();
    if (local_index == 0u) {
        atomicAdd(&aux[0], atomicLoad(&wg_count));
    }
}

@compute @workgroup_size(WORKGROUP)
fn cs_hist(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WORKGROUP;
    var my_count = 0u;
    for (var i = gid.x; i < u.n_locs; i += stride) {
        let p = position(i);
        if (!visible(p)) {
            continue;
        }
        my_count += 1u;
        let j = min(u32(trunc(u.os * (p.x - u.x_min))), u.npx - 1u);
        let ii = min(u32(trunc(u.os * (p.y - u.y_min))), u.npy - 1u);
        atomicAdd(&image[ii * u.npx + j], 1u);
    }
    flush_count(lid, my_count);
}
"""
)

# --------------------------------------------------------------------
# tiled module: Gaussian three-tier bin/scan/scatter/gather
# --------------------------------------------------------------------
_TILED_WGSL = (
    f"""
const WORKGROUP: u32 = {_WORKGROUP}u;
const TILE: u32 = {_TILE}u;
const TILE_PX: u32 = {_TILE * _TILE}u;
const WHALE_MAX_TILES: u32 = {_WHALE_MAX_TILES}u;
"""
    + _UNIFORMS_WGSL
    + """
@group(0) @binding(1) var<storage, read_write> imagef: array<f32>;
// aux[0] = in-view count, aux[1] = bin entries, aux[2] = whale count
@group(0) @binding(2) var<storage, read_write> aux: array<atomic<u32>, 4>;
// tile tier: tile_atomics[2t] = count, [2t + 1] = cursor; bins hold
// the whale list (first whale_cap slots) followed by the tile-tier
// entries. One offsets buffer holds, in this order: the pixel tier's
// per-pixel prefix within its tile (tile-major pixel index q = tile *
// 256 + local, ntiles*256 entries), the pixel tier's per-tile global
// base (ntiles entries), then the tile tier's offsets persisted per
// channel (ntiles per channel, sliced by u.tile_base)
@group(0) @binding(3)
var<storage, read_write> tile_atomics: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> bins: array<u32>;
// pixel tier: px_atomics[2q] = count, px_atomics[2q + 1] = cursor;
// sorted[e] = (xp, yp, sx, sy), sorted_off[e] = the off-diagonal
// covariance s01
@group(0) @binding(6) var<storage, read_write> px_atomics: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> sorted: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> sorted_off: array<f32>;

// start of the tile-tier offsets inside the offsets buffer
fn tile_offsets_at(t: u32) -> u32 {
    return u.ntx * u.nty * (TILE_PX + 1u) + u.tile_base + t;
}
"""
    + _CHANNEL_WGSL
    + """
struct Splat {
    visible: bool,  // passes the in-view mask (counts as rendered)
    tier: u32,      // 0 = pixel, 1 = tile, 2 = whale, 3 = nothing to draw
    j_min: i32,     // CPU footprint arithmetic: trunc then clamp
    j_max: i32,
    i_min: i32,
    i_max: i32,
    xp: f32,
    yp: f32,
    sx: f32,        // sqrt of the 2x2 covariance diagonal (display px)
    sy: f32,
    s01: f32,       // off-diagonal covariance; 0 = axis-aligned
};

fn splat_params(i: u32) -> Splat {
    var s: Splat;
    s.tier = 3u;
    let p = position(i);
    s.visible = visible(p);
    if (!s.visible) {
        return s;
    }
    let xp = u.os * (p.x - u.x_min);
    let yp = u.os * (p.y - u.y_min);
    let k = src(i);
    var sx = u.os * max(lpxs[k], u.min_blur);
    var sy = u.os * max(lpys[k], u.min_blur);
    if ((u.mode & MODE_ISO) != 0u) {
        let mean = 0.5 * (sx + sy);
        sx = mean;
        sy = mean;
    }
    var vxx = sx * sx;
    var vyy = sy * sy;
    var vxy = 0.0;
    if ((u.mode & MODE_THETA) != 0u) {
        // precision ellipse rotated in-plane by the localization's angle
        let c = cos(angles[k]);
        let sn = sin(angles[k]);
        vxx = sx * sx * c * c + sy * sy * sn * sn;
        vyy = sx * sx * sn * sn + sy * sy * c * c;
        vxy = (sx * sx - sy * sy) * c * sn;
    }
    if ((u.mode & MODE_ROT) != 0u) {
        // 3D covariance rotated with the localizations, projected
        let sz = u.os * max(lpzs[k], u.min_blur);
        let cov = mat3x3<f32>(
            vec3<f32>(vxx, vxy, 0.0),
            vec3<f32>(vxy, vyy, 0.0),
            vec3<f32>(0.0, 0.0, sz * sz),
        );
        let cr = u.rot * cov * transpose(u.rot);
        vxx = cr[0][0];
        vyy = cr[1][1];
        vxy = cr[1][0];
    }
    if ((u.mode & (MODE_THETA | MODE_ROT)) != 0u) {
        if (vxx * vyy - vxy * vxy < 1e-10) {  // CPU degenerate guard
            return s;
        }
    }
    if (vxy != 0.0 || (u.mode & MODE_ROT) != 0u) {
        sx = sqrt(vxx);
        sy = sqrt(vyy);
    }
    let off_x = 3.0 * sx;
    let off_y = 3.0 * sy;
    s.j_min = max(0, i32(trunc(xp - off_x)));
    s.j_max = min(i32(u.npx), i32(trunc(xp + off_x)) + 1);
    s.i_min = max(0, i32(trunc(yp - off_y)));
    s.i_max = min(i32(u.npy), i32(trunc(yp + off_y)) + 1);
    if (s.j_max <= s.j_min || s.i_max <= s.i_min) {
        return s;
    }
    s.xp = xp;
    s.yp = yp;
    s.sx = sx;
    s.sy = sy;
    s.s01 = vxy;
    if (off_x < 1.0 && off_y < 1.0) {
        // footprint provably within the center pixel's 3x3 neighborhood
        s.tier = 0u;
        return s;
    }
    let touched = u32((s.j_max - 1) / i32(TILE) - s.j_min / i32(TILE) + 1)
        * u32((s.i_max - 1) / i32(TILE) - s.i_min / i32(TILE) + 1);
    s.tier = select(1u, 2u, touched > WHALE_MAX_TILES);
    return s;
}

// tile-major pixel index of a localization's center pixel
fn center_q(s: Splat) -> u32 {
    let jc = min(u32(trunc(s.xp)), u.npx - 1u);
    let ic = min(u32(trunc(s.yp)), u.npy - 1u);
    let tile = (ic / TILE) * u.ntx + jc / TILE;
    return tile * TILE_PX + (ic % TILE) * TILE + jc % TILE;
}

// contribution of a Gaussian (v = xp, yp, sx, sy; s01) to pixel
// (px, py); zero outside its CPU footprint bounds
fn weight(px: i32, py: i32, v: vec4<f32>, s01: f32) -> f32 {
    let off_x = 3.0 * v.z;
    let j_min = max(0, i32(trunc(v.x - off_x)));
    let j_max = min(i32(u.npx), i32(trunc(v.x + off_x)) + 1);
    if (px < j_min || px >= j_max) {
        return 0.0;
    }
    let off_y = 3.0 * v.w;
    let i_min = max(0, i32(trunc(v.y - off_y)));
    let i_max = min(i32(u.npy), i32(trunc(v.y + off_y)) + 1);
    if (py < i_min || py >= i_max) {
        return 0.0;
    }
    let a = f32(px) + 0.5 - v.x;
    let b = f32(py) + 0.5 - v.y;
    if (s01 == 0.0) {
        // axis-aligned: kernels._draw_gaussian_loc's separable form
        return exp(-(a * a / (2.0 * v.z * v.z) + b * b / (2.0 * v.w * v.w)))
            / (6.283185307179586 * v.z * v.w);
    }
    // general bivariate form: kernels._draw_gaussian_cov3d_loc
    let s00 = v.z * v.z;
    let s11 = v.w * v.w;
    let det = s00 * s11 - s01 * s01;
    let inv00 = s11 / det;
    let inv11 = s00 / det;
    let inv01 = -s01 / det;
    let exponent = a * a * inv00 + 2.0 * a * b * inv01 + b * b * inv11;
    return exp(-0.5 * exponent) / (6.283185307179586 * sqrt(det));
}

var<workgroup> wg_count: atomic<u32>;

// Reduces the per-invocation in-view counts of a workgroup into
// aux[0]. Workgroup memory is zeroed explicitly: WGSL promises
// zero-initialization, but an NVIDIA RTX A5500 on Windows handed each
// workgroup the previous one's count (n inflated ~6x) while Metal was
// clean -- never rely on the driver for it.
fn flush_count(local_index: u32, my_count: u32) {
    if (local_index == 0u) {
        atomicStore(&wg_count, 0u);
    }
    workgroupBarrier();
    atomicAdd(&wg_count, my_count);
    workgroupBarrier();
    if (local_index == 0u) {
        atomicAdd(&aux[0], atomicLoad(&wg_count));
    }
}

// ---- phase A: size the tile-tier bins (needs a readback) ----

@compute @workgroup_size(WORKGROUP)
fn cs_bin_count(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WORKGROUP;
    for (var i = gid.x; i < u.n_locs; i += stride) {
        let s = splat_params(i);
        if (s.tier == 2u) {
            atomicAdd(&aux[2], 1u);
        }
        if (s.tier != 1u) {
            continue;
        }
        let ty_max = (s.i_max - 1) / i32(TILE);
        let tx_max = (s.j_max - 1) / i32(TILE);
        for (var ty = s.i_min / i32(TILE); ty <= ty_max; ty++) {
            for (var tx = s.j_min / i32(TILE); tx <= tx_max; tx++) {
                atomicAdd(&tile_atomics[2u * (u32(ty) * u.ntx + u32(tx))], 1u);
            }
        }
    }
}

// serial exclusive prefix sum over the (small) tile grid; also stores
// the total number of bin entries for the Python side to size the bins
@compute @workgroup_size(1)
fn cs_scan() {
    var acc = 0u;
    let n_tiles = u.ntx * u.nty;
    for (var t = 0u; t < n_tiles; t++) {
        offsets[tile_offsets_at(t)] = acc;
        acc += atomicLoad(&tile_atomics[2u * t]);
    }
    atomicStore(&aux[1], acc);
}

// ---- phase B: pixel-tier sort, scatter, gather, whales ----

@compute @workgroup_size(WORKGROUP)
fn cs_px_count(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WORKGROUP;
    for (var i = gid.x; i < u.n_locs; i += stride) {
        let s = splat_params(i);
        if (s.tier == 0u) {
            atomicAdd(&px_atomics[2u * center_q(s)], 1u);
        }
    }
}

var<workgroup> scan_buf: array<u32, TILE_PX>;

// per tile: exclusive scan of its 256 pixel counts (Hillis-Steele in
// workgroup memory); the tile total goes to the base slot
@compute @workgroup_size(TILE, TILE)
fn cs_px_scan(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let tile = wid.y * u.ntx + wid.x;
    let q = tile * TILE_PX + lid;
    let own = atomicLoad(&px_atomics[2u * q]);
    scan_buf[lid] = own;
    workgroupBarrier();
    for (var step = 1u; step < TILE_PX; step *= 2u) {
        var add = 0u;
        if (lid >= step) {
            add = scan_buf[lid - step];
        }
        workgroupBarrier();
        scan_buf[lid] += add;
        workgroupBarrier();
    }
    offsets[q] = scan_buf[lid] - own;  // inclusive -> exclusive
    if (lid == TILE_PX - 1u) {
        offsets[u.ntx * u.nty * TILE_PX + tile] = scan_buf[lid];
    }
}

// serial exclusive prefix sum of the tile totals -> per-tile bases
@compute @workgroup_size(1)
fn cs_px_tile_scan() {
    var acc = 0u;
    let base = u.ntx * u.nty * TILE_PX;
    for (var t = 0u; t < u.ntx * u.nty; t++) {
        let total = offsets[base + t];
        offsets[base + t] = acc;
        acc += total;
    }
}

@compute @workgroup_size(WORKGROUP)
fn cs_bin_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let stride = nwg.x * WORKGROUP;
    let n_tiles = u.ntx * u.nty;
    var my_count = 0u;
    for (var i = gid.x; i < u.n_locs; i += stride) {
        let s = splat_params(i);
        if (s.visible) {
            my_count += 1u;  // degenerate footprints still count
        }
        if (s.tier == 0u) {
            let q = center_q(s);
            let slot = atomicAdd(&px_atomics[2u * q + 1u], 1u);
            let e = offsets[n_tiles * TILE_PX + q / TILE_PX]
                + offsets[q] + slot;
            if (e < u.px_cap) {
                sorted[e] = vec4<f32>(s.xp, s.yp, s.sx, s.sy);
                if ((u.mode & (MODE_THETA | MODE_ROT)) != 0u) {
                    sorted_off[e] = s.s01;  // axis-aligned modes: 0
                }
            }
        } else if (s.tier == 1u) {
            let ty_max = (s.i_max - 1) / i32(TILE);
            let tx_max = (s.j_max - 1) / i32(TILE);
            for (var ty = s.i_min / i32(TILE); ty <= ty_max; ty++) {
                for (var tx = s.j_min / i32(TILE); tx <= tx_max; tx++) {
                    let tile = u32(ty) * u.ntx + u32(tx);
                    let slot = atomicAdd(&tile_atomics[2u * tile + 1u], 1u);
                    let idx = offsets[tile_offsets_at(tile)] + slot;
                    if (idx < u.bin_cap) {
                        bins[u.whale_cap + idx] = i;
                    }
                }
            }
        } else if (s.tier == 2u) {
            let slot = atomicAdd(&aux[2], 1u);
            if (slot < u.whale_cap) {
                bins[slot] = i;
            }
        }
    }
    flush_count(lid, my_count);
}

var<workgroup> s_loc: array<vec4<f32>, WORKGROUP>;
var<workgroup> s_off: array<f32, WORKGROUP>;

@compute @workgroup_size(TILE, TILE)
fn cs_gather(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lxy: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let n_tiles = u.ntx * u.nty;
    let tile = wid.y * u.ntx + wid.x;
    let px = i32(wid.x * TILE + lxy.x);
    let py = i32(wid.y * TILE + lxy.y);
    var acc = 0.0;

    // pixel tier: the 3x3 neighborhood's sorted segments
    for (var dy = -1; dy <= 1; dy++) {
        let qy = py + dy;
        if (qy < 0 || qy >= i32(u.npy)) {
            continue;
        }
        for (var dx = -1; dx <= 1; dx++) {
            let qx = px + dx;
            if (qx < 0 || qx >= i32(u.npx)) {
                continue;
            }
            let nt = u32(qy) / TILE * u.ntx + u32(qx) / TILE;
            let q = nt * TILE_PX + (u32(qy) % TILE) * TILE + u32(qx) % TILE;
            let start = offsets[n_tiles * TILE_PX + nt] + offsets[q];
            let end = start + atomicLoad(&px_atomics[2u * q + 1u]);
            // the off-diagonal term is only stored (and read) when the
            // mode can produce one; uniform branch, no divergence
            if ((u.mode & (MODE_THETA | MODE_ROT)) != 0u) {
                for (var e = start; e < end; e++) {
                    acc += weight(px, py, sorted[e], sorted_off[e]);
                }
            } else {
                for (var e = start; e < end; e++) {
                    acc += weight(px, py, sorted[e], 0.0);
                }
            }
        }
    }

    // tile tier: stream this tile's bin through workgroup memory
    let start = u.whale_cap + offsets[tile_offsets_at(tile)];
    let count = atomicLoad(&tile_atomics[2u * tile + 1u]);
    var done = 0u;
    while (done < count) {
        let chunk = min(WORKGROUP, count - done);
        if (lid < chunk) {
            let s = splat_params(bins[start + done + lid]);
            s_loc[lid] = vec4<f32>(s.xp, s.yp, s.sx, s.sy);
            s_off[lid] = s.s01;
        }
        workgroupBarrier();
        for (var k = 0u; k < chunk; k++) {
            acc += weight(px, py, s_loc[k], s_off[k]);
        }
        workgroupBarrier();
        done += chunk;
    }

    if (px < i32(u.npx) && py < i32(u.npy)) {
        imagef[u32(py) * u.npx + u32(px)] = acc;
    }
}

// whales: pixel-parallel gather over the whole image, added on top
@compute @workgroup_size(WORKGROUP)
fn cs_whales(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n_whales = min(atomicLoad(&aux[2]), u.whale_cap);
    if (n_whales == 0u) {
        return;
    }
    let n_pixels = u.npx * u.npy;
    let stride = nwg.x * WORKGROUP;
    for (var p = gid.x; p < n_pixels; p += stride) {
        let px = i32(p % u.npx);
        let py = i32(p / u.npx);
        var acc = 0.0;
        for (var w = 0u; w < n_whales; w++) {
            let s = splat_params(bins[w]);
            acc += weight(
                px, py, vec4<f32>(s.xp, s.yp, s.sx, s.sy), s.s01
            );
        }
        imagef[p] += acc;
    }
}
"""
)


def _storage_entry(binding, read_only=False):
    kind = (
        wgpu.BufferBindingType.read_only_storage
        if read_only
        else wgpu.BufferBindingType.storage
    )
    return {
        "binding": binding,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {"type": kind},
    }


class WgpuBackend(SplatBackend):
    """GPU splat backend computing offscreen through wgpu.

    Raises
    ------
    SplatBackendError
        If no suitable adapter/device is available (at construction) or
        a render request is outside the supported scope / fails.
    """

    name = "wgpu"
    persistent_uploads = True

    def __init__(self, adapter: str = "high-performance"):
        try:
            adapter = self._request_adapter(adapter)
            # raise the storage ceilings to what the adapter has;
            # channels are one f32 buffer per column, 4 bytes per loc
            wanted = (
                "max-storage-buffer-binding-size",
                "max-buffer-size",
            )
            limits = {
                key: adapter.limits[key]
                for key in wanted
                if key in adapter.limits
            }
            key = "max-storage-buffers-per-shader-stage"
            if key in adapter.limits:
                limits[key] = min(
                    adapter.limits[key], _STORAGE_BINDINGS_WANTED
                )
            self._device = adapter.request_device_sync(required_limits=limits)
        except Exception as error:
            raise SplatBackendError(
                f"wgpu initialization failed: {error}"
            ) from error
        self._adapter_info = dict(adapter.info)
        self._adapter_preference = adapter
        self._max_storage_bytes = limits.get(
            "max-storage-buffer-binding-size", 128 * 2**20
        )
        self._max_channel_locs = self._max_storage_bytes // 16  # sorted
        device = self._device

        uniform_entry = {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }
        # channel columns (group 1), shared by every pipeline so the
        # cached per-channel bind groups fit them all
        self._group1_layout = device.create_bind_group_layout(
            entries=[_storage_entry(b, read_only=True) for b in range(8)]
        )
        # scatter module (histogram): uniforms, u32 image, aux
        self._scatter_group0_layout = device.create_bind_group_layout(
            entries=[uniform_entry, _storage_entry(1), _storage_entry(2)]
        )
        # tiled module (gaussian): + tier scratch
        self._tiled_group0_layout = device.create_bind_group_layout(
            entries=[uniform_entry] + [_storage_entry(b) for b in range(1, 9)]
        )
        scatter_module = device.create_shader_module(code=_SCATTER_WGSL)
        tiled_module = device.create_shader_module(code=_TILED_WGSL)
        scatter_layout = device.create_pipeline_layout(
            bind_group_layouts=[
                self._scatter_group0_layout,
                self._group1_layout,
            ]
        )
        tiled_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._tiled_group0_layout, self._group1_layout]
        )
        self._pipelines = {
            "cs_hist": device.create_compute_pipeline(
                layout=scatter_layout,
                compute={"module": scatter_module, "entry_point": "cs_hist"},
            )
        }
        for entry in (
            "cs_bin_count",
            "cs_scan",
            "cs_px_count",
            "cs_px_scan",
            "cs_px_tile_scan",
            "cs_bin_scatter",
            "cs_gather",
            "cs_whales",
        ):
            self._pipelines[entry] = device.create_compute_pipeline(
                layout=tiled_layout,
                compute={"module": tiled_module, "entry_point": entry},
            )

        self._uniforms = device.create_buffer(
            size=_UNIFORM_BYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        # aux[0] = in-view count, aux[1] = bin entries, aux[2] = whales
        self._aux = device.create_buffer(
            size=16,
            usage=(
                wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            ),
        )
        # bound as the index list of non-indexed renders (never read)
        self._no_indices = device.create_buffer(
            size=4, usage=wgpu.BufferUsage.STORAGE
        )
        # one render at a time per device queue; the seam allows
        # concurrent callers (async worker + synchronous renders)
        self._lock = threading.Lock()
        # scratch buffers, reused across calls and grown as needed
        self._scratch = {}  # name -> (buffer, size)
        self._scatter_group0 = None
        self._tiled_group0 = None
        # resident uploads, one per column array, keyed by the memory
        # behind the array (data pointer, size, dtype, strides) so fresh
        # views of an unchanged DataFrame hit and columns shared between
        # blur methods (x, y) are uploaded once; each entry keeps a
        # reference to its source array so that memory cannot be
        # recycled under the key. Least recently rendered first; bounded
        # by the VRAM budget. Bind groups are cached per channel and
        # dropped when any of their arrays is evicted.
        self._arrays = OrderedDict()  # array key -> (array, buffer, nbytes)
        self._groups = {}  # channel key -> bind group
        self._cache_bytes = 0
        self._temp_buffers = []  # chunked (uncached) uploads of a render
        self.upload_count = 0  # statistics, e.g. for the bench
        self.uploaded_bytes = 0
        self._base_f32 = None
        self._rotation = None  # (3x3 float32, center) when rotated

    @staticmethod
    def _request_adapter(preference: str):
        """``"high-performance"`` / ``"low-power"`` ask the system for
        that kind of adapter; any other string selects the first
        enumerated adapter whose name or backend contains it
        (case-insensitive), e.g. ``"NVIDIA"`` on a dual-GPU laptop."""
        if preference in ("high-performance", "low-power"):
            return wgpu.gpu.request_adapter_sync(power_preference=preference)
        wanted = preference.lower()
        for candidate in wgpu.gpu.enumerate_adapters_sync():
            info = candidate.info
            label = f"{info.get('device', '')} {info.get('backend_type', '')}"
            if wanted in label.lower():
                return candidate
        raise SplatBackendError(f"no GPU adapter matches '{preference}'")

    def describe(self) -> str:
        """E.g. ``"Apple M4 via Metal"``."""
        info = self._adapter_info
        device = info.get("device", "unknown GPU")
        backend_type = info.get("backend_type", "?")
        return f"{device} via {backend_type}"

    def _validate_render_request(self, columns, blur_method, viewport, ang):
        """Raise ``SplatBackendError`` for any request ``render_channels``
        cannot honor."""
        if blur_method not in (
            None,
            "gaussian",
            "gaussian_iso",
            "smooth",
            "convolve",
        ):
            raise SplatBackendError(f"unknown blur_method '{blur_method}'")
        if viewport is None:
            raise SplatBackendError("GPU rendering needs an explicit viewport")
        if ang is not None:
            if any(c.z is None for c in columns):
                raise SplatBackendError("3D rotation needs z")
        if blur_method in ("gaussian", "gaussian_iso"):
            if any(c.lpx is None or c.lpy is None for c in columns):
                raise SplatBackendError("missing localization precision")
            if ang is not None and any(c.lpz is None for c in columns):
                raise SplatBackendError("3D rotation needs lpz")
        if any(len(c) > self._max_channel_locs for c in columns):
            raise SplatBackendError(
                "channel exceeds the GPU storage-binding limit"
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def render_channels(
        self,
        columns: list[_RenderColumns],
        info: list[list[dict]],
        *,
        disp_px_size: float,
        viewport: tuple[tuple[float, float], tuple[float, float]] | None,
        blur_method: (
            Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
        ),
        min_blur_width: float,
        ang: tuple | Rotation | None,
        quadtree_capacity: int | None = None,
        triangulation_passes: int | None = None,
        triangulation_jitter: float | None = None,
    ) -> list[tuple[int, lib.FloatArray2D]]:
        """Render each channel offscreen on the GPU.

        See ``backend.SplatBackend.render_channels``. The ``quadtree``
        and ``triangulation`` methods are CPU-only
        (``scene._render_channels`` never sends them here; their
        parameters are accepted for the contract).

        Parameters
        ----------
        columns : list of splat._RenderColumns
            Localization columns, one per channel (already extracted,
            angle in radians and lpz fallback applied).
        info : list of list of dict
            Metadata, one entry per channel.
        disp_px_size : float
            Display pixel size in nm, see ``splat.render``.
        viewport : tuple or None
            Field of view ``((y_min, x_min), (y_max, x_max))`` in
            camera pixels, see ``splat.render``.
        blur_method : {"gaussian", "gaussian_iso", "smooth", \
                "convolve"} or None
            Blur method, see ``splat.render``.
        min_blur_width : float
            Minimum size of blur (camera pixels), see ``splat.render``.
        ang : tuple or scipy.spatial.transform.Rotation or None
            Rotation of the localizations, see ``splat.render``.
        quadtree_capacity : int, optional
            Accepted for the backend contract and ignored. Default is
            None.
        triangulation_passes : int, optional
            Accepted for the backend contract and ignored. Default is
            None.
        triangulation_jitter : float, optional
            Accepted for the backend contract and ignored. Default is
            None.

        Returns
        -------
        renderings : list of (int, lib.FloatArray2D)
            ``(number of locs in view, image)`` per channel, in input
            order.

        Raises
        ------
        SplatBackendError
            If the request cannot be rendered on the GPU or the GPU
            render fails.
        """
        self._validate_render_request(columns, blur_method, viewport, ang)
        try:
            with self._lock:
                return self._render(
                    columns,
                    info,
                    disp_px_size=disp_px_size,
                    viewport=viewport,
                    blur_method=blur_method,
                    min_blur_width=min_blur_width,
                    ang=ang,
                )
        except SplatBackendError:
            raise
        except Exception as error:
            raise SplatBackendError(f"GPU render failed: {error}") from error

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def _render(
        self,
        columns,
        info,
        *,
        disp_px_size,
        viewport,
        blur_method,
        min_blur_width,
        ang,
    ):
        (y_min, x_min), (y_max, x_max) = viewport
        pixelsize = lib.get_from_metadata(
            info[0], "Pixelsize", raise_error=True
        )
        oversampling = pixelsize / disp_px_size
        # image size exactly as _render_setup computes it (float64)
        n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
        n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
        if n_pixel_x <= 0 or n_pixel_y <= 0:
            raise SplatBackendError("empty viewport")
        ntx = -(-n_pixel_x // _TILE)
        nty = -(-n_pixel_y // _TILE)
        self._base_f32 = (
            oversampling,
            x_min,
            y_min,
            x_max,
            y_max,
            min_blur_width,
            float(n_pixel_x),
            float(n_pixel_y),
        )
        if ang is None:
            self._rotation = None
        else:
            # rotation about the view center, as _locs_rotation_arrays
            matrix = np.ascontiguousarray(
                to_rotation(ang).as_matrix(), dtype=np.float32
            )
            center = (
                x_min + (x_max - x_min) / 2,
                y_min + (y_max - y_min) / 2,
            )
            self._rotation = (matrix, center)
        geometry = {
            "npx": n_pixel_x,
            "npy": n_pixel_y,
            "ntx": ntx,
            "nty": nty,
        }
        plan = self._plan_uploads(columns)
        entries = [(chunk, cache) for chunk, _, cache in plan]
        max_locs = max(len(chunk) for chunk, _, _ in plan)
        self._ensure_scratch(n_pixel_x * n_pixel_y, ntx * nty, len(plan))
        try:
            if blur_method in ("gaussian", "gaussian_iso"):
                self._ensure_buffer(
                    "sorted", max_locs * 16, wgpu.BufferUsage.STORAGE
                )
                self._ensure_buffer(
                    "sorted_off", max_locs * 4, wgpu.BufferUsage.STORAGE
                )
                iso = blur_method == "gaussian_iso"
                bin_cap = self._bin_phase(entries, geometry, iso)
                renderings = self._splat_phase(entries, geometry, iso, bin_cap)
            else:
                renderings = self._render_hist(entries, geometry)
        finally:
            # chunk uploads are single-use; the readbacks above completed,
            # so the GPU is done with them
            for buffer in self._temp_buffers:
                buffer.destroy()
            self._temp_buffers.clear()
        renderings = self._merge_chunks(renderings, plan, len(columns))
        if blur_method in (None, "gaussian", "gaussian_iso"):
            return renderings
        return self._filter(
            renderings, columns, blur_method, oversampling, min_blur_width
        )

    # ------------------------------------------------------------------
    # resident uploads: budget, chunking, eviction
    # ------------------------------------------------------------------
    @staticmethod
    def _column_arrays(channel):
        """The seven column arrays bound for a channel (``x`` stands in
        for absent ones)."""
        return tuple(
            channel.x if column is None else column
            for column in (
                channel.x,
                channel.y,
                channel.lpx,
                channel.lpy,
                channel.z,
                channel.lpz,
                channel.angle,
            )
        )

    @staticmethod
    def _array_key(array):
        interface = array.__array_interface__
        return (
            interface["data"][0],
            array.nbytes,
            array.dtype.str,
            interface.get("strides"),
        )

    def _channel_key(self, channel):
        return tuple(self._array_key(a) for a in self._column_arrays(channel))

    def _resolve_view(self, array):
        """Map a column array onto a resident upload: the array itself,
        or a strided/offset 1-D view of a resident array (the GUI's
        interactive previews are ``iloc[::step]`` of whole channels).
        Returns ``(resident key, stride, offset)`` in elements, or
        None."""
        key = self._array_key(array)
        if key in self._arrays:
            return key, 1, 0
        if array.ndim != 1 or array.dtype != np.float32:
            return None
        strides = array.__array_interface__.get("strides")
        step = 1 if strides is None else strides[0] // 4
        if strides is not None and (strides[0] % 4 or step < 1):
            return None
        pointer = array.__array_interface__["data"][0]
        for base_key in self._arrays:
            base_pointer, base_bytes, base_dtype, base_strides = base_key
            if base_dtype != key[2] or base_strides is not None:
                continue
            if pointer < base_pointer or (pointer - base_pointer) % 4:
                continue
            offset = (pointer - base_pointer) // 4
            if offset + max(len(array) - 1, 0) * step < base_bytes // 4:
                return base_key, step, offset
        return None

    def _resolve_channel(self, channel):
        """``(resident keys, stride, offset)`` when every column of the
        channel maps onto resident uploads with one common stride and
        offset, else None."""
        resolved = [
            self._resolve_view(a) for a in self._column_arrays(channel)
        ]
        if any(r is None for r in resolved):
            return None
        steps = {(step, offset) for _, step, offset in resolved}
        if len(steps) != 1:
            return None
        ((step, offset),) = steps
        return tuple(key for key, _, _ in resolved), step, offset

    @staticmethod
    def _channel_bytes(channel):
        """GPU bytes for one channel: 4 bytes per unique column."""
        arrays = WgpuBackend._column_arrays(channel)
        unique = {WgpuBackend._array_key(a): a for a in arrays}
        return sum(len(a) * 4 for a in unique.values())

    def _plan_channel_chunks(self, index, channel, budget):
        """Plan for one channel: a single cacheable entry, or (when
        larger than ``budget``) single-use row chunks.

        Returns a list of ``(chunk_columns, channel_index, cacheable)``
        entries."""
        nbytes = self._channel_bytes(channel)
        if (
            channel.indices is not None
            and budget is not None
            and nbytes > budget
        ):
            # an index list needs its whole base resident; a base
            # over the budget is gathered and chunked as rows
            channel = channel.materialize()
            nbytes = self._channel_bytes(channel)
        if budget is not None and nbytes > budget and len(channel) > 1:
            n_chunks = ceil(nbytes / budget)
            rows = ceil(len(channel) / n_chunks)
            entries = []
            for start in range(0, len(channel), rows):
                stop = min(start + rows, len(channel))
                entries.append((channel.slice(start, stop), index, False))
            return entries
        return [(channel, index, True)]

    def _reclaim_upload_budget(self, plan, budget):
        """Evict least recently rendered resident uploads until the
        cache fits ``plan``'s new uploads within ``budget``."""
        needed = {}
        for chunk, _, cache in plan:
            if not cache:
                continue
            resolved = self._resolve_channel(chunk)
            if resolved is not None:  # served by resident uploads
                for key in resolved[0]:
                    needed[key] = self._arrays[key][2]
                continue
            for array in self._column_arrays(chunk):
                needed[self._array_key(array)] = len(array) * 4
        new_bytes = sum(
            nbytes for key, nbytes in needed.items() if key not in self._arrays
        )
        for key in list(self._arrays):
            if self._cache_bytes + new_bytes <= budget:
                break
            if key in needed:
                continue
            self._evict(key)

    def _plan_uploads(self, columns):
        """Decide per channel whether it renders from a resident upload
        or, when larger than the VRAM budget, in single-use row chunks;
        then evict least recently rendered channels to make room.

        Returns ``(chunk_columns, channel_index, cacheable)`` entries in
        render order."""
        budget = vram_budget_bytes()
        plan = []
        for index, channel in enumerate(columns):
            plan.extend(self._plan_channel_chunks(index, channel, budget))
        if budget is not None:
            self._reclaim_upload_budget(plan, budget)
        return plan

    def _evict(self, key):
        _, buffer, nbytes = self._arrays.pop(key)
        buffer.destroy()
        self._cache_bytes -= nbytes
        # bind groups over the evicted buffer are stale
        for channel_key in [k for k in self._groups if key in k]:
            del self._groups[channel_key]

    @staticmethod
    def _merge_chunks(renderings, plan, n_channels):
        """Sum the chunk renderings of each channel back together."""
        merged = [None] * n_channels
        for (n, image), (_, index, _) in zip(renderings, plan):
            if merged[index] is None:
                merged[index] = [n, image]
            else:
                merged[index][0] += n
                merged[index][1] += image
        return [tuple(entry) for entry in merged]

    def _mode(self, channel, iso):
        mode = _MODE_ISO if iso else 0
        if not iso and channel.angle is not None:
            mode |= _MODE_THETA
        if self._rotation is not None:
            mode |= _MODE_ROT
        if channel.indices is not None:
            mode |= _MODE_INDEXED
        return mode

    def _write_uniforms(
        self,
        n_locs,
        geometry,
        mode=0,
        tile_base=0,
        bin_cap=0,
        stride=1,
        offset=0,
    ):
        uniforms = np.zeros(_UNIFORM_BYTES // 4, dtype=np.float32)
        uniforms[:8] = self._base_f32
        uniforms[8:20].view(np.uint32)[:] = (
            n_locs,
            geometry["npx"],
            geometry["npy"],
            geometry["ntx"],
            geometry["nty"],
            tile_base,
            _WHALE_CAPACITY,
            bin_cap,
            n_locs,  # px_cap: at most one sorted entry per localization
            mode,
            stride,
            offset,
        )
        if self._rotation is not None:
            matrix, center = self._rotation
            # mat3x3 uniform layout: three vec4-padded columns
            for column in range(3):
                uniforms[20 + 4 * column : 23 + 4 * column] = matrix[:, column]
            uniforms[32:34] = center
        self._device.queue.write_buffer(self._uniforms, 0, uniforms.tobytes())

    def _readback(self, encoder, image_bytes):
        readback = self._device.create_buffer(
            size=image_bytes + 16,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder.copy_buffer_to_buffer(
            self._scratch["image"][0], 0, readback, 0, image_bytes
        )
        encoder.copy_buffer_to_buffer(self._aux, 0, readback, image_bytes, 16)
        return readback

    def _collect(self, readbacks, geometry, as_float):
        npx, npy = geometry["npx"], geometry["npy"]
        renderings = []
        for readback in readbacks:
            readback.map_sync(wgpu.MapMode.READ)
            raw = np.frombuffer(readback.read_mapped(), dtype=np.uint32)
            n = int(raw[npx * npy])
            pixels = raw[: npx * npy]
            if as_float:
                image = pixels.view(np.float32).reshape(npy, npx).copy()
            else:
                image = pixels.astype(np.float32).reshape(npy, npx)
            readback.unmap()
            renderings.append((n, image))
        return renderings

    def _render_hist(self, entries, geometry):
        """Histogram: one scatter dispatch per channel (integer
        atomics; exact and deterministic)."""
        self._ensure_group0s()
        device = self._device
        image_bytes = geometry["npx"] * geometry["npy"] * 4
        image = self._scratch["image"][0]
        readbacks = []
        for channel, cache in entries:
            group, stride, offset = self._channel_bind_group(channel, cache)
            self._write_uniforms(
                len(channel),
                geometry,
                mode=self._mode(channel, False),
                stride=stride,
                offset=offset,
            )
            encoder = device.create_command_encoder()
            encoder.clear_buffer(image, 0, image_bytes)
            encoder.clear_buffer(self._aux, 0, 16)
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(self._pipelines["cs_hist"])
            compute_pass.set_bind_group(0, self._scatter_group0)
            compute_pass.set_bind_group(1, group)
            compute_pass.dispatch_workgroups(self._grid(len(channel)))
            compute_pass.end()
            readbacks.append(self._readback(encoder, image_bytes))
            # per-channel submit so the shared uniform buffer can be
            # rewritten for the next channel
            device.queue.submit([encoder.finish()])
        return self._collect(readbacks, geometry, as_float=False)

    def _filter(
        self, renderings, columns, blur_method, oversampling, min_blur_width
    ):
        """``smooth`` / ``convolve``: the CPU image filter on the GPU
        histogram, with the CPU backend's exact blur widths."""
        from ..splat import _fftconvolve, global_blur

        out = []
        for channel, (n, image) in zip(columns, renderings):
            if n == 0:
                out.append((n, image))
                continue
            if blur_method == "smooth":
                out.append((n, _fftconvolve(image, 1, 1)))
                continue
            # convolve: the channel's global blur, as on the CPU
            lpx, lpy = global_blur(channel)
            blur_width = oversampling * max(lpx, min_blur_width)
            blur_height = oversampling * max(lpy, min_blur_width)
            out.append((n, _fftconvolve(image, blur_width, blur_height)))
        return out

    def _bin_phase(self, entries, geometry, iso):
        """Count tile-tier bin entries per tile per channel and
        prefix-sum the (persisted) offsets; returns the bin capacity."""
        self._ensure_group0s()
        device = self._device
        ntiles = geometry["ntx"] * geometry["nty"]
        tile_atomics = self._scratch["tile_atomics"][0]
        totals_rb = device.create_buffer(
            size=16 * len(entries),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        for c, (channel, cache) in enumerate(entries):
            group, stride, offset = self._channel_bind_group(channel, cache)
            self._write_uniforms(
                len(channel),
                geometry,
                mode=self._mode(channel, iso),
                tile_base=c * ntiles,
                stride=stride,
                offset=offset,
            )
            encoder = device.create_command_encoder()
            encoder.clear_buffer(self._aux, 0, 16)
            encoder.clear_buffer(tile_atomics, 0, ntiles * 8)
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_bind_group(0, self._tiled_group0)
            compute_pass.set_bind_group(1, group)
            compute_pass.set_pipeline(self._pipelines["cs_bin_count"])
            compute_pass.dispatch_workgroups(self._grid(len(channel)))
            compute_pass.set_pipeline(self._pipelines["cs_scan"])
            compute_pass.dispatch_workgroups(1)
            compute_pass.end()
            encoder.copy_buffer_to_buffer(self._aux, 0, totals_rb, 16 * c, 16)
            device.queue.submit([encoder.finish()])
        totals_rb.map_sync(wgpu.MapMode.READ)
        aux = np.frombuffer(totals_rb.read_mapped(), dtype=np.uint32).copy()
        totals_rb.unmap()
        totals = aux.reshape(-1, 4)[:, 1]
        whale_counts = aux.reshape(-1, 4)[:, 2]
        if int(whale_counts.max(initial=0)) > _WHALE_CAPACITY:
            raise SplatBackendError(
                "too many wide-blur localizations for the GPU whale list; "
                "falling back to CPU rendering"
            )
        bin_cap = max(1, int(totals.max(initial=0)))
        bins_bytes = (_WHALE_CAPACITY + bin_cap) * 4
        if bins_bytes > self._max_storage_bytes:
            raise SplatBackendError(
                "tile bins exceed the GPU storage limit; falling back to "
                "CPU rendering"
            )
        self._ensure_buffer("bins", bins_bytes, wgpu.BufferUsage.STORAGE)
        return bin_cap

    def _splat_phase(self, entries, geometry, iso, bin_cap):
        """Per channel: sort the pixel tier, scatter the tile tier and
        whales, gather every pixel, add the whale pass, read back."""
        self._ensure_group0s()
        device = self._device
        npx, npy = geometry["npx"], geometry["npy"]
        ntx, nty = geometry["ntx"], geometry["nty"]
        ntiles = ntx * nty
        image_bytes = npx * npy * 4
        image = self._scratch["image"][0]
        tile_atomics = self._scratch["tile_atomics"][0]
        px_atomics = self._scratch["px_atomics"][0]
        readbacks = []
        for c, (channel, cache) in enumerate(entries):
            group, stride, offset = self._channel_bind_group(channel, cache)
            self._write_uniforms(
                len(channel),
                geometry,
                mode=self._mode(channel, iso),
                tile_base=c * ntiles,
                bin_cap=bin_cap,
                stride=stride,
                offset=offset,
            )
            encoder = device.create_command_encoder()
            encoder.clear_buffer(self._aux, 0, 16)
            encoder.clear_buffer(tile_atomics, 0, ntiles * 8)
            encoder.clear_buffer(px_atomics, 0, ntiles * _TILE * _TILE * 8)
            encoder.clear_buffer(image, 0, image_bytes)
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_bind_group(0, self._tiled_group0)
            compute_pass.set_bind_group(1, group)
            grid = self._grid(len(channel))
            compute_pass.set_pipeline(self._pipelines["cs_px_count"])
            compute_pass.dispatch_workgroups(grid)
            compute_pass.set_pipeline(self._pipelines["cs_px_scan"])
            compute_pass.dispatch_workgroups(ntx, nty)
            compute_pass.set_pipeline(self._pipelines["cs_px_tile_scan"])
            compute_pass.dispatch_workgroups(1)
            compute_pass.set_pipeline(self._pipelines["cs_bin_scatter"])
            compute_pass.dispatch_workgroups(grid)
            compute_pass.set_pipeline(self._pipelines["cs_gather"])
            compute_pass.dispatch_workgroups(ntx, nty)
            compute_pass.set_pipeline(self._pipelines["cs_whales"])
            compute_pass.dispatch_workgroups(self._grid(npx * npy))
            compute_pass.end()
            readbacks.append(self._readback(encoder, image_bytes))
            device.queue.submit([encoder.finish()])
        return self._collect(readbacks, geometry, as_float=True)

    # ------------------------------------------------------------------
    # resources
    # ------------------------------------------------------------------
    @staticmethod
    def _grid(n_items):
        return min(_MAX_WORKGROUPS, -(-n_items // _WORKGROUP) or 1)

    def _ensure_buffer(self, name, size, usage):
        """Create or grow a named scratch buffer; returns True when the
        bind groups referencing it must be rebuilt."""
        current = self._scratch.get(name)
        if current is not None and current[1] >= size:
            return False
        buffer = self._device.create_buffer(size=size, usage=usage)
        self._scratch[name] = (buffer, size)
        self._scatter_group0 = None
        self._tiled_group0 = None
        return True

    def _ensure_scratch(self, n_pixels, ntiles, n_channels):
        clearable = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        tile_px = _TILE * _TILE
        self._ensure_buffer(
            "image", n_pixels * 4, clearable | wgpu.BufferUsage.COPY_SRC
        )
        self._ensure_buffer("tile_atomics", ntiles * 8, clearable)
        self._ensure_buffer("px_atomics", ntiles * tile_px * 8, clearable)
        # pixel-tier prefixes + bases, then per-channel tile offsets
        self._ensure_buffer(
            "offsets",
            ntiles * (tile_px + 1 + n_channels) * 4,
            wgpu.BufferUsage.STORAGE,
        )
        # grown on demand by _bin_phase / _render
        self._ensure_buffer(
            "bins", _WHALE_CAPACITY * 4 + 4, wgpu.BufferUsage.STORAGE
        )
        self._ensure_buffer("sorted", 16, wgpu.BufferUsage.STORAGE)
        self._ensure_buffer("sorted_off", 4, wgpu.BufferUsage.STORAGE)

    def _ensure_group0s(self):
        """Rebuild the scratch bind groups if any buffer was regrown."""
        if self._scatter_group0 is None or self._tiled_group0 is None:
            self._build_group0s()

    def _build_group0s(self):
        device = self._device

        def entry(binding, buffer, size):
            return {
                "binding": binding,
                "resource": {"buffer": buffer, "offset": 0, "size": size},
            }

        def scratch(binding, name):
            buffer, size = self._scratch[name]
            return entry(binding, buffer, size)

        self._scatter_group0 = device.create_bind_group(
            layout=self._scatter_group0_layout,
            entries=[
                entry(0, self._uniforms, _UNIFORM_BYTES),
                scratch(1, "image"),
                entry(2, self._aux, 16),
            ],
        )
        self._tiled_group0 = device.create_bind_group(
            layout=self._tiled_group0_layout,
            entries=[
                entry(0, self._uniforms, _UNIFORM_BYTES),
                scratch(1, "image"),
                entry(2, self._aux, 16),
                scratch(3, "tile_atomics"),
                scratch(4, "offsets"),
                scratch(5, "bins"),
                scratch(6, "px_atomics"),
                scratch(7, "sorted"),
                scratch(8, "sorted_off"),
            ],
        )

    def _channel_bind_group(self, channel, cache=True):
        """``(bind group, stride, offset)`` over one channel's column
        buffers (x, y, lpx, lpy, z, lpz, angle; ``x`` stands in for
        absent ones) and its index list, if any. With ``cache`` the
        column upload is resident and reused by later renders of the
        same memory — including strided or offset views of it, which
        render straight from the resident buffers through
        ``stride``/``offset``; otherwise the upload is single-use (a
        chunk of a channel larger than the VRAM budget) and freed after
        the render. An index list is always a single-use upload, and
        its bind group is not cached."""
        buffers, stride, offset, group_key = self._column_buffers(
            channel, cache
        )
        if channel.indices is not None and len(channel.indices):
            index_buffer = self._device.create_buffer_with_data(
                data=np.ascontiguousarray(channel.indices, dtype=np.uint32),
                usage=wgpu.BufferUsage.STORAGE,
            )
            self._temp_buffers.append(index_buffer)
            return self._bind_group(buffers + [index_buffer]), stride, offset
        if group_key is None:
            return (
                self._bind_group(buffers + [self._no_indices]),
                stride,
                offset,
            )
        group = self._groups.get(group_key)
        if group is None:
            group = self._bind_group(buffers + [self._no_indices])
            self._groups[group_key] = group
        return group, stride, offset

    def _column_buffers(self, channel, cache):
        """``(column buffers, stride, offset, group key)`` for a
        channel: resident buffers (possibly through a strided/offset
        view) or fresh uploads; the group key is None for single-use
        uploads."""
        arrays = self._column_arrays(channel)
        if cache:
            resolved = self._resolve_channel(channel)
            if resolved is not None:
                base_keys, stride, offset = resolved
                for array_key in set(base_keys):
                    self._arrays.move_to_end(array_key)  # most recently used
                return (
                    [self._arrays[k][1] for k in base_keys],
                    stride,
                    offset,
                    base_keys,
                )
        key = self._channel_key(channel)
        buffers = []
        temp = {}  # placeholders alias x: upload each array once
        for array in arrays:
            array_key = self._array_key(array)
            if array_key in temp:
                buffers.append(temp[array_key])
                continue
            resident = self._arrays.get(array_key) if cache else None
            if resident is not None:
                self._arrays.move_to_end(array_key)
                buffers.append(resident[1])
                continue
            buffer = self._device.create_buffer_with_data(
                data=np.ascontiguousarray(array, dtype=np.float32),
                usage=wgpu.BufferUsage.STORAGE,
            )
            self.upload_count += 1
            self.uploaded_bytes += buffer.size
            if cache:
                self._arrays[array_key] = (array, buffer, buffer.size)
                self._cache_bytes += buffer.size
            else:
                self._temp_buffers.append(buffer)
                temp[array_key] = buffer
            buffers.append(buffer)
        return buffers, 1, 0, key if cache else None

    def _bind_group(self, buffers):
        return self._device.create_bind_group(
            layout=self._group1_layout,
            entries=[
                {
                    "binding": binding,
                    "resource": {
                        "buffer": buffer,
                        "offset": 0,
                        "size": buffer.size,
                    },
                }
                for binding, buffer in enumerate(buffers)
            ],
        )

    @property
    def resident_bytes(self) -> int:
        """GPU bytes currently held by resident uploads."""
        return self._cache_bytes

    def clear_cache(self) -> None:
        """Drop resident column uploads (frees GPU memory)."""
        self._groups.clear()
        for _, buffer, _ in self._arrays.values():
            buffer.destroy()
        self._arrays.clear()
        self._cache_bytes = 0

    def release_uploads(self) -> None:
        """Drop resident uploads (a dataset was closed)."""
        self.clear_cache()

    def close(self) -> None:
        """Release the device and all cached resources."""
        self.clear_cache()
        self._scratch.clear()
        self._scatter_group0 = None
        self._tiled_group0 = None
        self._device.destroy()
