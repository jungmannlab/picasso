"""
picasso.deprecation
~~~~~~~~~~~~~~~~~~~

Deprecation of Picasso's public functions, classes, modules and command
line names, following the stability policy (``docs/versioning.rst``):
a name deprecated in Picasso 1.x keeps working, with a warning, until
1.(x+2) at the earliest.

Every deprecation is registered here when its module is imported, with
the version that deprecated it and the version that may remove it, so
that ``tests/test_stability.py`` can fail once a removal is due. The
warnings are ``FutureWarning``s, which Python shows by default, pointing
at the line of the caller's code.

Only the standard library is imported, so the command line can use this
module without loading the rest of Picasso.

:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import re
import warnings
from typing import Callable, TypeVar

#: Minor releases a deprecated name keeps working for, at least
DEPRECATION_MINOR_RELEASES = 2

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class Deprecation:
    """A deprecated name and the versions that deprecate and remove it.

    Parameters
    ----------
    name : str
        The deprecated name, e.g. ``"picasso.localize.locs_from_fits"`` or
        ``"picasso undrift_fiducials"`` for a command.
    since : str
        Version that deprecated it, ``"MAJOR.MINOR"``.
    remove_in : str
        Version that may remove it, ``"MAJOR.MINOR"``.
    alternative : str or None, optional
        What to use instead. Default is None.
    message : str or None, optional
        The complete warning text, replacing the generated one. Default
        is None.
    """

    name: str
    since: str
    remove_in: str
    alternative: str | None = None
    message: str | None = None

    def text(self) -> str:
        """The warning text."""
        if self.message is not None:
            return self.message
        text = (
            f"{self.name} is deprecated since Picasso {self.since} and "
            f"will be removed in Picasso {self.remove_in}"
        )
        if self.alternative:
            text += f"; use {self.alternative} instead"
        return text + "."

    def warn(self, stacklevel: int = 2) -> None:
        """Issue the ``FutureWarning``.

        Parameters
        ----------
        stacklevel : int, optional
            As for ``warnings.warn``, counted from the caller of this
            method: the default 2 points at the code that called the
            function calling ``warn``, i.e., the deprecated function.
        """
        warnings.warn(self.text(), FutureWarning, stacklevel=stacklevel + 1)


#: Every deprecation registered by the modules imported so far
REGISTRY: list[Deprecation] = []


def register(
    name: str,
    *,
    since: str,
    remove_in: str,
    alternative: str | None = None,
    message: str | None = None,
) -> Deprecation:
    """Register a deprecation; call at module level.

    Parameters
    ----------
    name, since, remove_in, alternative, message
        See ``Deprecation``.

    Returns
    -------
    deprecation : Deprecation
        Its ``warn`` method issues the warning where the deprecated
        name is used.

    Raises
    ------
    ValueError
        If ``remove_in`` comes sooner than the stability policy allows,
        see ``earliest_removal``.
    """
    if not removal_allowed(since, remove_in):
        raise ValueError(
            f"{name}: deprecated in {since}, it may be removed in "
            f"{earliest_removal(since)} at the earliest, not {remove_in}."
        )
    deprecation = Deprecation(name, since, remove_in, alternative, message)
    REGISTRY.append(deprecation)
    return deprecation


def deprecated(
    *, since: str, remove_in: str, alternative: str | None = None
) -> Callable[[_T], _T]:
    """Decorator deprecating a public function or class.

    Calls warn (for a class: creating an instance warns), and the
    docstring gets a ``.. deprecated::`` note below its summary line.

    Parameters
    ----------
    since, remove_in, alternative
        See ``Deprecation``.

    Returns
    -------
    decorator : callable
        The decorator.

    Examples
    --------
    >>> @deprecated(since="1.1", remove_in="1.3", alternative="new_name")
    ... def old_name(locs, info):
    ...     return new_name(locs, info)
    """

    def decorate(obj):
        name = f"{obj.__module__}.{obj.__qualname__}"
        deprecation = register(
            name, since=since, remove_in=remove_in, alternative=alternative
        )
        if isinstance(obj, type):
            init = obj.__init__

            @functools.wraps(init)
            def __init__(self, *args, **kwargs):
                deprecation.warn()
                init(self, *args, **kwargs)

            obj.__init__ = __init__
            obj.__doc__ = _note_in_docstring(obj.__doc__, deprecation)
            return obj

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            deprecation.warn()
            return obj(*args, **kwargs)

        wrapper.__doc__ = _note_in_docstring(obj.__doc__, deprecation)
        return wrapper

    return decorate


def _note_in_docstring(doc: str | None, deprecation: Deprecation) -> str:
    """``doc`` with a ``.. deprecated::`` note after its summary, where
    numpydoc expects it."""
    note = f".. deprecated:: {deprecation.since}\n   "
    note += f"Removed in Picasso {deprecation.remove_in}."
    if deprecation.alternative:
        note += f" Use {deprecation.alternative} instead."
    if not doc:
        return note
    doc = inspect.cleandoc(doc)
    summary, _, rest = doc.partition("\n\n")
    return f"{summary}\n\n{note}\n\n{rest}".rstrip() + "\n"


def _minor_version(version: str) -> tuple[int, int]:
    """``(major, minor)`` of a version such as ``"1.2"`` or
    ``"1.2.0rc1"``."""
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"Not a version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def earliest_removal(since: str) -> str:
    """The first version that may remove a name deprecated in ``since``:
    ``DEPRECATION_MINOR_RELEASES`` minor releases later.

    Parameters
    ----------
    since : str
        Version that deprecated the name.

    Returns
    -------
    version : str
        ``"MAJOR.MINOR"``, e.g. ``"1.3"`` for ``"1.1"``.
    """
    major, minor = _minor_version(since)
    return f"{major}.{minor + DEPRECATION_MINOR_RELEASES}"


def removal_allowed(since: str, remove_in: str) -> bool:
    """Whether ``remove_in`` keeps a name deprecated in ``since`` for
    long enough: ``DEPRECATION_MINOR_RELEASES`` minor releases, or
    until a later major release.

    Parameters
    ----------
    since, remove_in : str
        Versions that deprecate and remove the name.

    Returns
    -------
    allowed : bool
        True if the stability policy allows the removal.
    """
    major, minor = _minor_version(since)
    remove_major, remove_minor = _minor_version(remove_in)
    if remove_major > major:
        return True
    return (
        remove_major == major
        and remove_minor >= minor + DEPRECATION_MINOR_RELEASES
    )


def removal_due(deprecation: Deprecation, version: str) -> bool:
    """Whether the Picasso ``version`` must no longer contain a
    deprecated name: it is the release ``remove_in`` or later.
    Pre-releases of ``remove_in`` (e.g., ``"1.0.0a1"``) are not due
    yet, so a removal can land during the release cycle.

    Parameters
    ----------
    deprecation : Deprecation
        The deprecation.
    version : str
        A Picasso version, e.g. ``picasso.__version__``.

    Returns
    -------
    due : bool
        True if the name should have been removed.
    """
    current = _minor_version(version)
    target = _minor_version(deprecation.remove_in)
    if current != target:
        return current > target
    return re.search(r"(a|b|rc|\.dev)\d*$", version) is None
