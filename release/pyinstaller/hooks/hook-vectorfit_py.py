"""PyInstaller hook: bundle VectorFit's Python bindings (``vectorfit_py``).

Picasso imports ``vectorfit_py`` with ``importlib.import_module`` (see
``picasso/vectorfit.py``), which PyInstaller's import analysis cannot see,
so the build scripts name it with ``--hidden-import vectorfit_py`` and this
hook adds what the package needs at run time:

- its compiled extension, ``vectorfit_py._vectorfit_py``;
- its distribution metadata: ``vectorfit_py/__init__.py`` reads its own
  version with ``importlib.metadata.version`` and fails to import without
  it;
- the DLLs delvewheel (Windows) or auditwheel (Linux) vendored next to the
  package, in ``vectorfit_py.libs`` (FFTW and the MSVC runtime). The
  package adds that folder to the DLL search path relative to its own
  location (``<package>/../vectorfit_py.libs``), so the DLLs go to the same
  relative place in the bundle.

The CPU (``vectorfit-py``) and the GPU (``vectorfit-py-cu12``) builds install
the same module, so either is handled.
"""

import os

from PyInstaller.compat import importlib_metadata
from PyInstaller.utils.hooks import copy_metadata, get_package_paths

hiddenimports = ["vectorfit_py._vectorfit_py"]

datas = []
for _distribution in ("vectorfit-py", "vectorfit-py-cu12"):
    try:
        datas += copy_metadata(_distribution)
    except importlib_metadata.PackageNotFoundError:
        pass

_site_packages, _ = get_package_paths("vectorfit_py")
binaries = []
for _libs in ("vectorfit_py.libs", "vectorfit_py_cu12.libs"):
    _libs_dir = os.path.join(_site_packages, _libs)
    if os.path.isdir(_libs_dir):
        binaries += [
            (os.path.join(_libs_dir, name), _libs)
            for name in sorted(os.listdir(_libs_dir))
        ]
