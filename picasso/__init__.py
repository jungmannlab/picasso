"""
picasso.__init__.py
~~~~~~~~~~~~~~~~~~~

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

import os.path
import re
import warnings

import yaml
from .version import __version__

# In frozen (PyInstaller) builds the numba-cuda redirect (a site-packages .pth)
# never runs, so "from numba import cuda" would fall back to Numba's built-in
# CUDA stub and GPU-accelerated (numba.cuda) code silently vanishes. Install
# the redirect here, before anything imports numba.cuda. No-op outside frozen
# builds.
from . import _numba_cuda_compat as _numba_cuda_compat

_numba_cuda_compat.install()

_this_file = os.path.abspath(__file__)
_this_dir = os.path.dirname(_this_file)

DOCS_ROOT_URL = "https://picassosr.readthedocs.io/en"


def docs_version(version: str = __version__) -> str:
    """Return the Read the Docs version slug matching a Picasso version.

    Final releases read the default documentation (``latest``, built
    from ``master``). Pre-releases (``a``, ``b``, ``rc`` or ``.dev``,
    e.g. ``1.0.0a2``) read the documentation built from their
    development branch ``vX.Y`` (e.g. ``v1.0``), which describes
    features not yet merged into ``master``.

    Parameters
    ----------
    version : str, optional
        Picasso version string. Default is the installed version.

    Returns
    -------
    slug : str
        ``"latest"`` or ``"vX.Y"``.
    """
    match = re.match(r"(\d+)\.(\d+)(?:\.\d+)*(a|b|rc|\.dev)?", version)
    if match is None or match.group(3) is None:
        return "latest"
    return f"v{match.group(1)}.{match.group(2)}"


def docs_url(page: str = "") -> str:
    """Return the URL of a documentation page for this Picasso version.

    Parameters
    ----------
    page : str, optional
        Page (and anchor) relative to the documentation root, e.g.
        ``"render/analysis.html#render-resi"``. Default is the
        documentation root.

    Returns
    -------
    url : str
        Full URL, see ``docs_version`` for the version it points to.
    """
    return f"{DOCS_ROOT_URL}/{docs_version()}/{page}"


def user_config_dir() -> str:
    """Return the user Picasso directory (``~/.picasso``).

    Shared with ``~/.picasso/settings.yaml`` and the other per-user files
    (see ``picasso.io``) so every install type (one-click installer, PyPI,
    source) keeps a single, user-writable, uninstall-surviving location.

    Returns
    -------
    path : str
        ``~/.picasso`` expanded for the current user.
    """
    return os.path.join(os.path.expanduser("~"), ".picasso")


def config_filename() -> str:
    """Return the path to the user camera config file
    (``~/.picasso/config.yaml``).

    This is where Picasso Localize reads and writes its camera
    configuration (see https://picassosr.readthedocs.io/en/latest/
    localize/camera.html#localize-camera-config). Keeping it next to the other
    ``~/.picasso`` files means it no longer hides inside the installed
    package directory, where it was hard to find.

    Returns
    -------
    path : str
        ``~/.picasso/config.yaml``.
    """
    return os.path.join(user_config_dir(), "config.yaml")


def _legacy_config_filename() -> str:
    """Path to the pre-0.11 in-package config (``picasso/config.yaml``)."""
    return os.path.join(_this_dir, "config.yaml")


def resolve_config_path() -> str | None:
    """Return the path of the config file to read, or None if none exists.

    Resolution:

    1. ``~/.picasso/config.yaml`` (preferred, user-writable);
    2. the legacy in-package ``config.yaml`` (older installs), read in
       place and never moved, so a user who keeps editing it there still
       sees their changes take effect.

    Returns
    -------
    path : str or None
        Path of the config file to read; None when neither exists.
    """
    user_config = config_filename()
    if os.path.isfile(user_config):
        return user_config
    if os.path.isfile(_legacy_config_filename()):
        return _legacy_config_filename()
    return None


#: Why the camera configuration could not be read, or None; set by
#: ``load_config`` and reported by Localize when it opens.
CONFIG_LOAD_ERROR: str | None = None


def load_config() -> dict:
    """Load the camera configuration used by Picasso Localize.

    Reads ``~/.picasso/config.yaml`` if present, otherwise the legacy
    in-package ``config.yaml`` (see ``resolve_config_path``). A file
    that cannot be read does not stop Picasso: it is warned about, the
    reason is kept in ``CONFIG_LOAD_ERROR`` and no camera is configured.

    Returns
    -------
    config : dict
        The parsed configuration; an empty dict if neither file exists,
        it cannot be read or parsed, or it parses to nothing.
    """
    global CONFIG_LOAD_ERROR
    CONFIG_LOAD_ERROR = None
    path = resolve_config_path()
    if path is None:
        return {}
    try:
        with open(path, "r") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError:
        return {}
    except OSError as error:
        reason = f"it cannot be opened ({error})"
    except yaml.YAMLError as error:
        reason = f"it is not valid YAML:\n{error}"
    else:
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        reason = "it does not hold settings such as 'Cameras: ...'"
    CONFIG_LOAD_ERROR = (
        f"The camera configuration {path} could not be read, so no camera "
        f"is configured: {reason}"
    )
    warnings.warn(CONFIG_LOAD_ERROR, stacklevel=2)
    return {}


CONFIG = load_config()
