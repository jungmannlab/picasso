"""PyInstaller hook: bundle ``cuda.bindings``' vendored C++ runtime DLL.

``cuda-bindings`` wheels are delvewheel-repaired: their compiled extensions
(``runtime``, ``driver``, ``nvrtc``, ...) link a private, hash-mangled copy of
msvcp140 (e.g. ``msvcp140-<hash>.dll``) vendored in a top-level
``cuda_bindings.libs`` directory next to the ``cuda`` namespace.
``cuda/bindings/__init__.py`` registers that directory with
``os.add_dll_directory`` (resolved as ``cuda/bindings/../../cuda_bindings.libs``)
before any extension loads.

``--collect-all cuda`` does not pick up that sibling directory, so the frozen
app dies with ``DLL load failed while importing runtime`` as soon as numba-cuda
imports ``cuda.bindings.runtime``. Copy the DLLs to ``cuda_bindings.libs`` in the
bundle, exactly where the delvewheel patch looks for them.
"""

import glob
import os

from PyInstaller.utils.hooks import get_package_paths

# get_package_paths returns (base, pkg_dir); base is the site-packages directory
# that contains both the ``cuda`` namespace and ``cuda_bindings.libs``.
_site_packages, _ = get_package_paths("cuda.bindings")
_libs_dir = os.path.join(_site_packages, "cuda_bindings.libs")

datas = [
    (dll, "cuda_bindings.libs")
    for dll in glob.glob(os.path.join(_libs_dir, "*.dll"))
]
