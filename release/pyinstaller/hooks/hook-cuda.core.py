"""PyInstaller hook: bundle NVIDIA ``cuda.core``'s compiled Cython extensions.

GPU-accelerated (numba.cuda) code imports ``cuda.core`` (via numba-cuda).
``cuda.core``'s actual functionality lives in compiled Cython extensions
(``.pyd``) under a
version-specific subpackage -- ``cuda/core/cu12`` for CUDA 12 -- which it selects
at import time and merges into ``cuda.core`` itself by rewriting ``cuda.core``'s
``__path__``. That runtime ``__path__`` merge is why ``cuda.core._utils.cuda_utils``
resolves to ``cuda/core/cu12/_utils/cuda_utils*.pyd``.

PyInstaller's ``--collect-all cuda`` collects the ``.py``/``.pxd`` data files but
drops these compiled extensions (the dynamic ``cu12`` dispatch confuses its
module graph -- unlike ``cuda.bindings``, which uses no such dispatch and bundles
fine), so the frozen app dies with ``No module named
'cuda.core._utils.cuda_utils'`` and reports no GPU.

We copy every ``.pyd`` under ``cuda/core`` verbatim to its real on-disk path, as
*data* rather than as a *binary*: added as a binary, PyInstaller relocates it by
inferred module name and breaks the ``cu12`` layout the runtime merge depends on;
copied as data it lands exactly where ``cuda.core``'s import machinery looks.
"""

import glob
import os

from PyInstaller.utils.hooks import get_package_paths

# Anchor on cuda.bindings: a normal package with a stable path. cuda.core rewrites
# its own __path__ at import time (the cu12/cu13 dispatch), so resolving paths
# through it is unreliable. get_package_paths returns (base, pkg_dir); base is the
# site-packages directory that contains the top-level ``cuda`` namespace.
_site_packages, _ = get_package_paths("cuda.bindings")
_core_dir = os.path.join(_site_packages, "cuda", "core")

datas = [
    (src, os.path.relpath(os.path.dirname(src), _site_packages))
    for src in glob.glob(
        os.path.join(_core_dir, "**", "*.pyd"), recursive=True
    )
]

# cuda-core's wheel is delvewheel-repaired too: those extensions link a private,
# hash-mangled copy of msvcp140 vendored in a top-level ``cuda_core.libs``
# directory, which ``cuda/core/__init__.py`` registers via os.add_dll_directory
# (resolved as ``cuda/core/../../cuda_core.libs``). Copying the .pyd as data
# skips PyInstaller's dependency analysis, so bundle the DLLs at that path.
_libs_dir = os.path.join(_site_packages, "cuda_core.libs")
datas += [
    (dll, "cuda_core.libs")
    for dll in glob.glob(os.path.join(_libs_dir, "*.dll"))
]
