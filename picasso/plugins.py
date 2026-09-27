"""
picasso.plugins
~~~~~~~~~~~~~~~

Discovery, loading and online management of user plugins from
``~/.picasso/plugins``.

A plugin is a single ``.py`` file. It may extend Picasso in three ways,
in any combination:

* **a GUI** — a module-level ``Plugin`` class with ``__init__(self, window)``
  setting ``self.name`` (the target GUI app, e.g. ``"render"``) and
  ``self.window``, plus ``execute(self)`` adding actions to
  ``window.plugin_menu``. Loaded by ``picasso.gui.plugins_loader``;
* **the Python API** — a module-level ``PICASSO_API`` mapping names to
  objects (or a callable returning such a mapping), reachable from scripts
  as ``picasso.plugins.api.<name>`` (see ``load_api_plugins``);
* **the command line** — a module-level ``register_cli(subparsers)`` that
  adds ``picasso <command>`` subcommands, each calling
  ``set_defaults(func=...)`` to name its handler (see
  ``register_cli_plugins``).

See ``plugin_template.py``, which shows all three.

This module imports no Qt, so the command line and ordinary scripts can
use plugins without a GUI. Everything that needs Qt — the plugin menu
and the store dialog — lives in ``picasso.gui.plugins_loader``, which
re-exports the names defined here.

Plugins are executed as ordinary Python inside the Picasso process, so
they cannot be sandboxed: a loaded plugin can do anything Picasso can.
The safeguards here therefore aim at *provenance and reviewability*
rather than containment:

* nothing is loaded until it has been explicitly enabled, so a file that
  merely appears in the plugins folder never runs (``is_enabled``). This
  holds identically in the GUI, in the CLI and in a script — importing
  ``picasso.plugins`` alone executes no plugin code;
* registry downloads are verified against a ``sha256`` pinned in the
  manifest, so the code behind a plugin cannot change without a reviewed
  manifest commit (``install``);
* ids and repository paths taken from the manifest are validated before
  they are used as file names or URLs (``is_safe_id``,
  ``is_safe_repo_path``);
* the store dialog can show a plugin's source before installing it and a
  diff of what changed before updating it.

Loading is deliberately tolerant: a broken plugin prints a traceback and
is skipped so that it can never crash app startup or the command line.

The second half of this module implements the client side of an online
plugin store for browsing, installing, updating and uninstalling plugins
from the ``picasso_plugins`` GitHub repository. The registry is a single
``index.json`` manifest at the repo root; plugins are plain ``.py`` files
downloaded into the user plugins directory. Installed plugins, their
versions and which files are enabled are tracked in a hidden
``.installed.json`` sidecar there.

:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import difflib
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
import traceback
import urllib.request
from types import ModuleType
from typing import Any

from . import io
from .version import __version__ as PICASSO_VERSION


# =============================================================================
# Safety helpers: identifiers, paths and integrity
# =============================================================================

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMAND_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def is_safe_id(value) -> bool:
    """Whether ``value`` may be used as a plugin id and file name stem.

    Ids come from a remote manifest and are turned into file names inside
    the user plugins directory, so they must not be able to escape it:
    only ASCII letters, digits, ``_`` and ``-`` are allowed.

    Parameters
    ----------
    value : object
        Candidate plugin id.

    Returns
    -------
    safe : bool
        True if ``value`` is a string of allowed characters only.
    """
    return isinstance(value, str) and bool(_ID_RE.match(value))


def is_safe_repo_path(value) -> bool:
    """Whether ``value`` is a safe relative path inside the registry repo.

    Rejects absolute paths, Windows separators, ``..`` segments and
    anything that is not a ``.py`` file, so a manifest entry cannot point
    the download at another repository or at a path outside it.

    Parameters
    ----------
    value : object
        Candidate repository path.

    Returns
    -------
    safe : bool
        True if ``value`` is a safe relative ``.py`` path.
    """
    if not isinstance(value, str) or not value.endswith(".py"):
        return False
    if value.startswith("/") or "\\" in value or ":" in value:
        return False
    return all(part not in ("", ".", "..") for part in value.split("/"))


def is_safe_filename(value) -> bool:
    """Whether ``value`` is a bare ``.py`` file name with no directory part."""
    return (
        isinstance(value, str)
        and value.endswith(".py")
        and len(value) > len(".py")
        and os.path.basename(value) == value
    )


def is_safe_command(value) -> bool:
    """Whether ``value`` may be used as a ``picasso`` subcommand name.

    Plugin command names end up in ``picasso -h`` next to the built-in
    ones, so they are held to the same shape: lowercase ASCII letters,
    digits, ``_`` and ``-``.

    Parameters
    ----------
    value : object
        Candidate subcommand name.

    Returns
    -------
    safe : bool
        True if ``value`` is a string of allowed characters only.
    """
    return isinstance(value, str) and bool(_COMMAND_RE.match(value))


def is_valid_digest(value) -> bool:
    """Whether ``value`` looks like a hex-encoded SHA-256 digest."""
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def plugin_path(filename: str) -> str:
    """Absolute path of ``filename`` inside the user plugins directory.

    Parameters
    ----------
    filename : str
        Bare ``.py`` file name, without a directory part.

    Returns
    -------
    path : str
        Absolute path of the file in the plugins directory.

    Raises
    ------
    ValueError
        If ``filename`` is not a bare ``.py`` name or if the resolved path
        would land outside the plugins directory.
    """
    if not is_safe_filename(filename):
        raise ValueError(f"Unsafe plugin file name: {filename!r}")
    directory = os.path.abspath(io.plugins_directory())
    path = os.path.abspath(os.path.join(directory, filename))
    if os.path.dirname(path) != directory:
        raise ValueError(
            f"Plugin file escapes the plugins folder: {filename!r}"
        )
    return path


# =============================================================================
# Discovery and loading
# =============================================================================

# Imported plugin modules, keyed by path -> (stat signature, module). One
# plugin file can be wanted by the GUI, the CLI and the Python API in the
# same process; executing it once per consumer would run its import-time
# side effects repeatedly and hand out duplicate class objects.
_module_cache: dict[str, tuple[tuple, ModuleType]] = {}

# Plugin files already reported as failing to import. A failed import is not
# cached (the next load should retry the file), so without this a broken
# plugin would print its warning once per consumer within a single command.
_reported_failures: set[str] = set()


def _discover_plugin_files() -> list[str]:
    """Return sorted ``.py`` files in the user plugins directory, skipping
    files whose name starts with ``_``."""
    directory = io.plugins_directory()
    files = sorted(glob.glob(os.path.join(directory, "*.py")))
    return [f for f in files if not os.path.basename(f).startswith("_")]


def _module_signature(path: str) -> tuple:
    """Cheap identity of a file's contents: modification time and size."""
    stat = os.stat(path)
    return (stat.st_mtime_ns, stat.st_size)


def clear_module_cache() -> None:
    """Forget every imported plugin module so the next load re-executes it.

    Drops the ``sys.modules`` entries as well, so that a reload really
    starts from the file rather than from what a previous load left behind.
    """
    for _, module in _module_cache.values():
        name = getattr(module, "__name__", None)
        if name and sys.modules.get(name) is module:
            del sys.modules[name]
    _module_cache.clear()
    _reported_failures.clear()


def _load_module_from_path(path: str) -> ModuleType:
    """Import a standalone ``.py`` file that is not part of any package.

    The imported module is cached here, so loading the same unchanged file
    again in the same process returns the module already executed; a file
    edited on disk (a different mtime or size) is executed afresh.

    The module is registered in ``sys.modules`` under a
    ``picasso_plugin_`` name *before* it is executed. Plenty of ordinary
    Python looks a class's own module up there while the class statement is
    still running — ``@dataclass`` does, and so do ``typing.get_type_hints``,
    ``pickle`` and ``enum`` — and all of it fails with a confusing
    ``AttributeError`` on ``None`` if the module cannot be found. Registering
    first also lets a plugin split itself over helper imports.

    The source is compiled directly rather than through
    ``loader.exec_module``, which would go via ``__pycache__``. That cache
    decides a ``.pyc`` is still valid from the source's mtime *in whole
    seconds* plus its size, so a plugin edited and reloaded within the same
    second without changing length would run its old bytecode — precisely
    the edit-and-reload loop the Plugins menu exists for. Compiling here
    also keeps ``__pycache__`` directories out of the user plugins folder.
    """
    try:
        signature = _module_signature(path)
    except OSError:
        signature = None
    if signature is not None:
        cached = _module_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]

    name = "picasso_plugin_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path!r}")
    source = spec.loader.get_source(name)
    if source is None:
        raise ImportError(f"Could not read the source of {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(source, path, "exec"), module.__dict__)
    except BaseException:
        # Leave no half-executed module behind for the next importer to
        # find and believe.
        sys.modules.pop(name, None)
        raise
    if signature is not None:
        _module_cache[path] = (signature, module)
    return module


def enabled_plugin_files(state: dict | None = None) -> list[str]:
    """Paths of the plugin files that are enabled, in load order."""
    state = load_state() if state is None else state
    return [
        path
        for path in _discover_plugin_files()
        if is_enabled(state, os.path.basename(path))
    ]


def disabled_plugin_files(state: dict | None = None) -> list[str]:
    """File names present in the plugins folder that are not enabled."""
    state = load_state() if state is None else state
    return [
        os.path.basename(p)
        for p in _discover_plugin_files()
        if not is_enabled(state, os.path.basename(p))
    ]


def load_enabled_modules(
    verbose: bool = False,
) -> list[tuple[str, ModuleType]]:
    """Import every enabled plugin file and return ``(path, module)`` pairs.

    Only files that have been explicitly enabled are imported; a ``.py``
    file that has merely been copied into the plugins folder is discovered
    but not run until the user enables it, because importing it means
    running it. A plugin that fails to import is skipped, so one broken
    file never takes down the rest.

    Parameters
    ----------
    verbose : bool, optional
        Print the full traceback of a failing plugin instead of a one-line
        warning. This runs on every ``picasso`` command, so the default is
        the short form: a single broken plugin must not bury the output of
        every command the user runs in a traceback they did not ask for.
        ``picasso plugins list`` is where the details belong.

    Returns
    -------
    modules : list of (str, ModuleType)
        The successfully imported plugin modules with their file paths.
    """
    modules: list[tuple[str, ModuleType]] = []
    for path in enabled_plugin_files():
        try:
            modules.append((path, _load_module_from_path(path)))
        except Exception as exc:
            _report_load_failure(path, exc, verbose)
    return modules


def _report_load_failure(path: str, exc: BaseException, verbose: bool) -> None:
    """Report a plugin that would not import, once per file per process."""
    if verbose:
        print(f"Failed to load plugin {path!r}:")
        traceback.print_exc()
        return
    if path in _reported_failures:
        return
    _reported_failures.add(path)
    print(
        f"Warning: plugin {os.path.basename(path)!r} failed to load "
        f"({type(exc).__name__}: {exc}); run 'picasso plugins list' for "
        "the full error.",
        file=sys.stderr,
    )


# =============================================================================
# Python API plugins
# =============================================================================


def _api_names(module: ModuleType) -> dict[str, Any]:
    """The ``PICASSO_API`` mapping a plugin module contributes, if any.

    ``PICASSO_API`` may be the mapping itself or a callable returning one,
    so that a plugin can build its exports lazily (e.g. after an optional
    import). Anything else is rejected.
    """
    exported = getattr(module, "PICASSO_API", None)
    if exported is None:
        return {}
    if callable(exported):
        exported = exported()
    if not isinstance(exported, dict):
        raise TypeError(
            "PICASSO_API must be a dict (or a callable returning one), got "
            f"{type(exported).__name__}"
        )
    for key in exported:
        if not isinstance(key, str) or not key.isidentifier():
            raise TypeError(
                f"PICASSO_API keys must be valid Python names, got {key!r}"
            )
    return dict(exported)


def load_api_plugins() -> dict[str, Any]:
    """Import the enabled plugins and merge their ``PICASSO_API`` exports.

    This is the explicit trigger for running plugin code from a script:
    importing ``picasso.plugins`` executes nothing by itself.

    On a name collision the first plugin to export the name keeps it and a
    warning naming both files is printed, so an added plugin can never
    silently replace a name a script already relies on.

    Returns
    -------
    exports : dict
        Mapping of exported name to the object the plugin provides.

    Examples
    --------
    >>> from picasso import plugins
    >>> exports = plugins.load_api_plugins()  # doctest: +SKIP
    >>> exports["analyze"]("locs.hdf5")  # doctest: +SKIP
    """
    exports: dict[str, Any] = {}
    origin: dict[str, str] = {}
    for path, module in load_enabled_modules():
        name = os.path.basename(path)
        try:
            contributed = _api_names(module)
        except Exception:
            print(f"Failed to read PICASSO_API from plugin {path!r}:")
            traceback.print_exc()
            continue
        for key, value in contributed.items():
            if key in exports:
                print(
                    f"Plugin {name!r} exports {key!r}, which "
                    f"{origin[key]!r} already provides; keeping the one from "
                    f"{origin[key]!r}."
                )
                continue
            exports[key] = value
            origin[key] = name
    return exports


class _Api:
    """Attribute access over the names exported by API plugins.

    The plugins are imported on the first attribute access, not on import
    of this module, so ``import picasso`` never runs plugin code. Use
    ``refresh()`` after enabling or editing a plugin.
    """

    def __init__(self):
        self._exports: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._exports is None:
            self._exports = load_api_plugins()
        return self._exports

    def refresh(self) -> dict[str, Any]:
        """Re-read the plugins folder and re-import the enabled plugins."""
        clear_module_cache()
        self._exports = None
        return self._load()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        exports = self._load()
        try:
            return exports[name]
        except KeyError:
            raise AttributeError(
                f"No enabled Picasso plugin exports {name!r}. Available: "
                + (", ".join(sorted(exports)) or "(none)")
            ) from None

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._load()))

    def __contains__(self, name: str) -> bool:
        return name in self._load()


api = _Api()
"""Lazy namespace of the names exported by enabled API plugins."""


# =============================================================================
# Command line plugins
# =============================================================================


class _GuardedSubparsers:
    """``add_parser`` proxy that keeps a plugin from breaking the CLI.

    ``argparse`` raises when a subcommand name is added twice, which would
    turn one careless plugin into a crash of the whole ``picasso`` command
    for everyone. This wrapper rejects a name that is already taken (by a
    built-in command or an earlier plugin) or that is not a plausible
    command name, before argparse ever sees it.
    """

    def __init__(self, subparsers, source: str):
        self._subparsers = subparsers
        self._source = source

    def add_parser(self, name, **kwargs):
        if not is_safe_command(name):
            raise ValueError(
                f"Plugin {self._source!r} tried to add the command {name!r}; "
                "command names may contain only lowercase letters, digits, "
                "'_' and '-', and must start with a letter or digit."
            )
        if name in self._subparsers.choices:
            raise ValueError(
                f"Plugin {self._source!r} tried to add the command {name!r}, "
                "which already exists. Rename the command in the plugin."
            )
        return self._subparsers.add_parser(name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._subparsers, name)


def register_cli_plugins(subparsers) -> set[str]:
    """Let the enabled plugins add their subcommands to the ``picasso`` CLI.

    Each plugin module may define ``register_cli(subparsers)``; it is
    called with a guarded view of ``subparsers`` on which ``add_parser``
    refuses names that are already taken. A plugin subcommand names its
    handler with ``parser.set_defaults(func=handler)``, and ``handler`` is
    called with the parsed ``argparse.Namespace``.

    Nothing is imported when no plugin is enabled, so the common case
    costs one directory listing. A plugin that raises is reported and
    skipped, leaving the rest of the CLI intact.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The ``argparse`` subparsers action of the ``picasso`` parser.

    Returns
    -------
    commands : set of str
        The subcommand names contributed by plugins.
    """
    if not enabled_plugin_files():
        return set()

    existing = set(subparsers.choices)
    for path, module in load_enabled_modules():
        register = getattr(module, "register_cli", None)
        if register is None:
            continue
        name = os.path.basename(path)
        before = set(subparsers.choices)
        try:
            register(_GuardedSubparsers(subparsers, name))
        except Exception:
            print(f"Failed to register CLI commands from plugin {path!r}:")
            traceback.print_exc()
            # Drop any parser the plugin managed to add before it failed,
            # so a half-registered command cannot be invoked.
            for added in set(subparsers.choices) - before:
                subparsers.choices.pop(added, None)
    return set(subparsers.choices) - existing


def plugin_cli_commands(module: ModuleType) -> list[str]:
    """The subcommand names ``module`` would register, for reporting.

    Uses a throwaway parser so that listing what a plugin provides never
    touches the real CLI.

    Parameters
    ----------
    module : ModuleType
        Imported plugin module.

    Returns
    -------
    commands : list of str
        Sorted subcommand names; empty if the plugin has no
        ``register_cli`` or if it fails.
    """
    register = getattr(module, "register_cli", None)
    if register is None:
        return []
    import argparse

    probe = argparse.ArgumentParser(add_help=False)
    subparsers = probe.add_subparsers()
    try:
        register(_GuardedSubparsers(subparsers, "probe"))
    except Exception:
        return []
    return sorted(subparsers.choices)


# =============================================================================
# Online plugin store
# =============================================================================

# --- Registry location -------------------------------------------------------

REPO = "jungmannlab/picasso_plugins"
BRANCH = "main"  # the registry's default branch
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
MANIFEST_URL = f"{RAW_BASE}/index.json"
REPO_URL = f"https://github.com/{REPO}"

_TIMEOUT = 15  # seconds, for every network request
_SIDECAR = ".installed.json"

# Registry ``app`` value for plugins that extend the Python API or the
# command line rather than one GUI. They are relevant everywhere, so the
# store dialog shows them next to the current app's plugins.
API_APP = "api"


# --- Version helpers ---------------------------------------------------------


def parse_version(value: str | None) -> tuple[int, ...]:
    """Parse a version string into a tuple of its numeric components.

    Non-numeric suffixes (e.g. the ``a0`` in ``0.11.0a0``) are split on so
    that ``0.11.0a0`` -> ``(0, 11, 0, 0)``. Good enough to order the simple
    versions plugins use; it is not a full PEP 440 implementation.

    Parameters
    ----------
    value : str or None
        Version string.

    Returns
    -------
    version : tuple of int
        Numeric components; empty if ``value`` is None or empty.
    """
    if not value:
        return ()
    return tuple(int(p) for p in re.findall(r"\d+", str(value)))


def compare_versions(a: str | None, b: str | None) -> int:
    """Return -1, 0 or 1 for ``a`` <, == or > ``b`` (zero-padded compare)."""
    ta, tb = parse_version(a), parse_version(b)
    n = max(len(ta), len(tb))
    ta = ta + (0,) * (n - len(ta))
    tb = tb + (0,) * (n - len(tb))
    return (ta > tb) - (ta < tb)


def is_compatible(entry: dict) -> bool:
    """Whether the running Picasso satisfies the plugin's minimum version."""
    minimum = entry.get("min_picasso_version")
    if not minimum:
        return True
    return compare_versions(PICASSO_VERSION, minimum) >= 0


# --- Installed-state sidecar -------------------------------------------------


def _sidecar_path() -> str:
    return os.path.join(io.plugins_directory(), _SIDECAR)


def load_state() -> dict:
    """Load the installed-state sidecar, tolerant of a missing/corrupt file.

    The state holds three things: ``plugins`` (per-id records of what was
    installed from the registry), ``enabled`` (per-file-name flags deciding
    what may be loaded) and ``trust_acknowledged`` (the one-time warning).

    Returns
    -------
    state : dict
        The installed state; the defaults (no plugins, nothing enabled,
        trust not acknowledged) if the file is missing or corrupt.
    """
    path = _sidecar_path()
    state = {"plugins": {}, "enabled": {}, "trust_acknowledged": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            state["plugins"] = data.get("plugins", {}) or {}
            enabled = data.get("enabled", {}) or {}
            if isinstance(enabled, dict):
                state["enabled"] = {
                    k: bool(v)
                    for k, v in enabled.items()
                    if isinstance(k, str)
                }
            state["trust_acknowledged"] = bool(data.get("trust_acknowledged"))
    except (OSError, ValueError):
        pass
    return state


def save_state(state: dict) -> None:
    """Write the installed state to the sidecar file.

    Parameters
    ----------
    state : dict
        Installed state, as returned by :func:`load_state`.
    """
    with open(_sidecar_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def is_enabled(state: dict, filename: str) -> bool:
    """Whether ``filename`` is allowed to be loaded and executed.

    Unknown files are *not* enabled: enabling is the explicit consent step,
    so a plugin only ever runs after the user has said so.

    Parameters
    ----------
    state : dict
        Installed state, as returned by :func:`load_state`.
    filename : str
        Bare file name of the plugin.

    Returns
    -------
    enabled : bool
        True if the user has enabled ``filename``.
    """
    return bool(state.get("enabled", {}).get(filename))


def set_enabled(state: dict, filename: str, value: bool) -> None:
    """Record whether ``filename`` may be loaded, and save the sidecar."""
    state.setdefault("enabled", {})[filename] = bool(value)
    save_state(state)


# --- Network + install operations -------------------------------------------


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "picasso"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def _is_usable_entry(entry: dict) -> bool:
    """Whether a manifest entry is structurally safe to work with.

    Entries with an unsafe id or file path are dropped outright — they can
    never be legitimate, since both are used to build a local file name and
    a download URL. A missing or malformed ``sha256`` is *not* filtered here:
    such entries are kept and surfaced as ``UNVERIFIED`` so the user sees
    why they cannot be installed.
    """
    return is_safe_id(entry.get("id")) and is_safe_repo_path(entry.get("file"))


def fetch_manifest() -> list[dict]:
    """Download and parse the registry manifest.

    Network and parse errors propagate so the caller can show a message
    and offer to retry.

    Returns
    -------
    entries : list of dict
        The plugin entries whose id and file path pass validation.
    """
    data = json.loads(_get(MANIFEST_URL).decode("utf-8"))
    plugins = data.get("plugins", []) if isinstance(data, dict) else []
    entries = []
    for entry in plugins:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        if not _is_usable_entry(entry):
            print(
                "Skipping plugin registry entry with an unsafe id or file "
                f"path: {entry.get('id')!r} -> {entry.get('file')!r}"
            )
            continue
        entries.append(entry)
    return entries


def _local_filename(entry: dict) -> str:
    """The on-disk ``.py`` name for an entry: the plugin id plus ``.py``."""
    if not is_safe_id(entry.get("id")):
        raise ValueError(f"Unsafe plugin id: {entry.get('id')!r}")
    return f"{entry['id']}.py"


def download_source(entry: dict) -> bytes:
    """Download an entry's ``.py`` and verify it against the pinned hash.

    Unverified code is never written to disk or shown as if it were the
    published plugin.

    Parameters
    ----------
    entry : dict
        Registry manifest entry.

    Returns
    -------
    content : bytes
        The verified source of the plugin.

    Raises
    ------
    ValueError
        If the entry is unsafe, carries no valid ``sha256``, or if the
        downloaded bytes do not match it.
    """
    if not _is_usable_entry(entry):
        raise ValueError(f"Unsafe registry entry for {entry.get('id')!r}")
    expected = entry.get("sha256")
    if not is_valid_digest(expected):
        raise ValueError(
            f"The registry entry for {entry['id']!r} has no valid sha256 "
            "integrity hash, so it cannot be verified. Refusing to install."
        )
    content = _get(f"{RAW_BASE}/{entry['file']}")
    actual = sha256_bytes(content)
    if actual != expected.lower():
        raise ValueError(
            f"Integrity check failed for {entry['id']!r}: the downloaded file "
            f"does not match the hash pinned in the registry.\n\n"
            f"expected {expected.lower()}\ngot      {actual}\n\n"
            "The file may have been changed after it was published. Nothing "
            "was installed."
        )
    return content


def install(entry: dict, state: dict) -> None:
    """Download, verify and store ``entry``'s ``.py``, then enable it.

    Installing is an explicit user action, so the file is enabled on
    success. Raises before touching the disk if verification fails.

    Parameters
    ----------
    entry : dict
        Registry manifest entry.
    state : dict
        Installed state, updated in place and saved.
    """
    content = download_source(entry)
    filename = _local_filename(entry)
    target = plugin_path(filename)
    with open(target, "wb") as f:
        f.write(content)
    state["plugins"][entry["id"]] = {
        "file": filename,
        "version": entry.get("version"),
        "app": entry.get("app"),
        "display_name": entry.get("display_name", entry["id"]),
        "sha256": sha256_bytes(content),
    }
    state.setdefault("enabled", {})[filename] = True
    save_state(state)


def uninstall(plugin_id: str, state: dict) -> None:
    """Delete the installed ``.py`` for ``plugin_id`` and forget it."""
    record = state["plugins"].get(plugin_id, {})
    filename = record.get("file")
    if not is_safe_filename(filename):
        filename = f"{plugin_id}.py" if is_safe_id(plugin_id) else None
    if filename:
        try:
            os.remove(plugin_path(filename))
        except (OSError, ValueError):
            pass
        state.get("enabled", {}).pop(filename, None)
    state["plugins"].pop(plugin_id, None)
    save_state(state)


def read_local_source(filename: str) -> str:
    """Read an on-disk plugin file as text, replacing undecodable bytes."""
    path = plugin_path(filename)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def diff_sources(old: str, new: str, filename: str) -> str:
    """Unified diff between the installed and the incoming source."""
    lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{filename} (installed)",
            tofile=f"{filename} (update)",
        )
    )
    if not lines:
        return (
            "The new version is byte-for-byte identical to the installed one."
        )
    return "".join(lines)


# --- Status model ------------------------------------------------------------

NOT_INSTALLED = "not_installed"
UP_TO_DATE = "up_to_date"
UPDATE_AVAILABLE = "update_available"
INCOMPATIBLE = "incompatible"
UNVERIFIED = "unverified"  # registry entry without a usable sha256
UNMANAGED = "unmanaged"  # registry plugin whose file was copied in by hand
LOCAL_ONLY = "local_only"  # file in the folder with no registry entry
ORPHAN = "orphan"  # installed but no longer in the manifest


def status_for(entry: dict, state: dict) -> str:
    """Status of a plugin entry relative to what is installed.

    Parameters
    ----------
    entry : dict
        Entry as returned by :func:`merged_entries`.
    state : dict
        Installed state, as returned by :func:`load_state`.

    Returns
    -------
    status : str
        One of ``ORPHAN``, ``LOCAL_ONLY``, ``INCOMPATIBLE``,
        ``UNVERIFIED``, ``UNMANAGED``, ``NOT_INSTALLED``,
        ``UPDATE_AVAILABLE`` or ``UP_TO_DATE``.
    """
    installed = state["plugins"].get(entry["id"])
    if entry.get("_orphan"):
        return ORPHAN
    if entry.get("_local"):
        return LOCAL_ONLY
    if not is_compatible(entry):
        return INCOMPATIBLE
    if not is_valid_digest(entry.get("sha256")):
        return UNVERIFIED
    if installed is None:
        try:
            exists = os.path.exists(plugin_path(_local_filename(entry)))
        except ValueError:
            exists = False
        return UNMANAGED if exists else NOT_INSTALLED
    if compare_versions(entry.get("version"), installed.get("version")) > 0:
        return UPDATE_AVAILABLE
    return UP_TO_DATE


def entry_filename(entry: dict) -> str | None:
    """The on-disk file name an entry maps to, or ``None`` if unsafe."""
    if entry.get("_local"):
        return entry.get("file")
    try:
        return _local_filename(entry)
    except ValueError:
        return None


def merged_entries(manifest: list[dict], state: dict) -> list[dict]:
    """Manifest entries, plus installed plugins and loose files not in it.

    Orphans (installed locally but dropped from the registry) are surfaced
    so they can still be uninstalled; they are flagged with ``_orphan``.
    Files sitting in the plugins folder that the registry knows nothing
    about are surfaced too — flagged with ``_local`` — so that manually
    shared plugins can be reviewed and enabled from the same place.

    Parameters
    ----------
    manifest : list of dict
        Entries as returned by :func:`fetch_manifest`.
    state : dict
        Installed state, as returned by :func:`load_state`.

    Returns
    -------
    entries : list of dict
        All entries, sorted by display name.
    """
    by_id = {e["id"]: e for e in manifest}
    entries = list(manifest)
    accounted = {
        name
        for name in (entry_filename(e) for e in manifest)
        if name is not None
    }
    for pid, rec in state["plugins"].items():
        if rec.get("file"):
            accounted.add(rec["file"])
        if pid not in by_id:
            entries.append(
                {
                    "id": pid,
                    "display_name": rec.get("display_name", pid),
                    "app": rec.get("app"),
                    "description": "(no longer in the online registry)",
                    "version": rec.get("version"),
                    "file": rec.get("file"),
                    "_orphan": True,
                }
            )
    for path in _discover_plugin_files():
        name = os.path.basename(path)
        if name in accounted:
            continue
        entries.append(
            {
                "id": os.path.splitext(name)[0],
                "display_name": name,
                "app": None,
                "description": (
                    "(local file, not from the online registry — review its "
                    "source before enabling it)"
                ),
                "file": name,
                "_local": True,
            }
        )
    return sorted(
        entries, key=lambda e: str(e.get("display_name", e["id"])).lower()
    )
