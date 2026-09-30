"""Make ``from numba import cuda`` resolve to NVIDIA's ``numba_cuda`` target.

The ``numba-cuda`` package replaces Numba's built-in (deprecated) CUDA target
with NVIDIA's out-of-tree one. It normally does this via
``_numba_cuda_redirector.pth`` -- a top-level *site-packages* file that the
interpreter runs at startup to install a meta-path finder which maps
``numba.cuda`` onto ``numba_cuda/numba/cuda``.

That mechanism is lost in a frozen (PyInstaller) build:

  1. PyInstaller never executes ``.pth`` files, so the redirect is never set
    up.
  2. The stock redirector rewrites the module *search path* and asks every
     meta-path finder to load ``numba.cuda`` from numba_cuda's directory. In a
     frozen app the frozen importer comes first and resolves modules by *name*,
     ignoring that path, so it keeps returning Numba's built-in ``numba.cuda``.

The net effect is that ``numba.cuda.is_available()`` returns False inside the
frozen app even when the GPU and CUDA runtime are fine, so any GPU-accelerated
(numba.cuda) code paths silently fall back to CPU or disappear.

This module reinstalls the redirect with a finder that, for every
``numba.cuda`` / ``numba.cuda.*`` import, asks :class:`PathFinder` to load the
module from numba_cuda's on-disk directory (where ``--collect-all numba_cuda``
places it). Because the spec is created with the correct ``numba.cuda.*`` name
and numba_cuda's search path, every module executes exactly once with the right
``__name__``/``__file__`` and its internal imports resolve recursively.

``install()`` is called from :mod:`picasso` at import time, before anything
imports ``numba.cuda``. It only acts in a frozen build (in a normal install the
``.pth`` redirector already did the job) and is a no-op when ``numba_cuda`` is
not present. Set ``PICASSO_DEBUG_CUDA=1`` to see what it decides.
"""

import importlib.util
import os
import sys
from importlib.machinery import PathFinder

_SRC = "numba.cuda"


def _log(msg):
    if os.environ.get("PICASSO_DEBUG_CUDA"):
        print("[picasso][cuda-compat] " + msg, file=sys.stderr)


def _numba_cuda_parent_paths():
    """Directories that contain numba_cuda's ``cuda`` package (i.e. the dirs
    that should masquerade as the parent of ``numba.cuda``).

    numba_cuda ships its target at ``numba_cuda/numba/cuda``, so the parent of
    the redirected ``numba.cuda`` is ``numba_cuda/numba``.
    """
    try:
        spec = importlib.util.find_spec("numba_cuda")
    except Exception as exc:
        _log(f"find_spec('numba_cuda') raised: {exc!r}")
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return [
        os.path.join(loc, "numba") for loc in spec.submodule_search_locations
    ]


class _NumbaCudaRedirector:
    """Map ``numba.cuda[.*]`` onto numba_cuda's on-disk files by name+path."""

    def __init__(self, parent_paths):
        self._parent_paths = parent_paths

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _SRC and not fullname.startswith(_SRC + "."):
            return None
        # Already imported (or mid-import): let the normal machinery use it.
        if fullname in sys.modules:
            return None

        # For "numba.cuda" itself, point PathFinder at numba_cuda's "numba"
        # directory. For submodules, Python passes the parent package's
        # __path__ (which our top-level spec set to numba_cuda's "cuda"
        # directory), so we forward that unchanged.
        search = self._parent_paths if fullname == _SRC else path
        try:
            spec = PathFinder.find_spec(fullname, search)
        except Exception as exc:
            _log(f"PathFinder.find_spec({fullname!r}) raised: {exc!r}")
            spec = None

        if spec is not None:
            _log(f"redirecting {fullname!r} -> {spec.origin}")
            return spec

        _log(f"could not locate {fullname!r} under {search}")
        return None


def install():
    """Install the numba.cuda -> numba_cuda redirect for frozen builds."""
    # In a normal (non-frozen) install the site-packages .pth redirector
    # already runs at startup, so there is nothing to do.
    if not getattr(sys, "frozen", False):
        return

    if any(type(f).__name__ == "_NumbaCudaRedirector" for f in sys.meta_path):
        _log("redirector already installed")
        return

    if "numba.cuda" in sys.modules:
        _log("numba.cuda already imported; cannot redirect")
        return

    parent_paths = _numba_cuda_parent_paths()
    if not parent_paths:
        _log(
            "numba_cuda not bundled; leaving numba.cuda untouched (CPU build?)"
        )
        return

    sys.meta_path.insert(0, _NumbaCudaRedirector(parent_paths))
    _log(f"installed numba.cuda redirector (parent paths: {parent_paths})")
