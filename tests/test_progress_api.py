"""Test that the public API reports progress in one way: every public
function or method that reports progress takes a single ``progress``
argument (a ``lib.Progress`` tracker, "console" or None, see
``picasso.lib.Progress``), annotated and documented in the same words,
and none uses the old callback arguments.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

from tests.regen_stability_snapshots import public_api

#: argument names that reported progress before the progress API was
#: unified in Picasso 1.0
OLD_NAMES = {
    "progress_callback",
    "callback",
    "callback_parent",
    "kinetics_progress",
    "groupprops_progress",
    "segmentation_callback",
    "rcc_callback",
    "identification_progress_callback",
    "fit_progress_callback",
    "fit_z_progress_callback",
    "cut_progress_callback",
}

#: arguments with "progress" in their name that are not trackers
NOT_TRACKERS = {"progress_title"}

#: how a public ``progress`` argument is annotated and documented, in the
#: same words
ANNOTATION = 'lib.Progress | Literal["console"] | None'
DOC_TYPE = 'lib.Progress, "console" or None, optional'


def _public_callables():
    """(qualified name, callable, documented object) of every public
    function, and of every public method (and ``__init__``) of every
    public class; ``__init__`` is documented in the class docstring."""
    out = []
    for module_name, names in public_api().items():
        module = importlib.import_module(module_name)
        for name in names:
            obj = getattr(module, name)
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr != "__init__" and attr.startswith("_"):
                        continue
                    if isinstance(member, (staticmethod, classmethod)):
                        member = member.__func__
                    if inspect.isfunction(member):
                        documented = obj if attr == "__init__" else member
                        qualname = f"{module_name}.{name}.{attr}"
                        out.append((qualname, member, documented))
            elif inspect.isfunction(obj):
                out.append((f"{module_name}.{name}", obj, obj))
    return out


PUBLIC_CALLABLES = _public_callables()
WITH_PROGRESS = [
    (name, fn, documented)
    for name, fn, documented in PUBLIC_CALLABLES
    if "progress" in inspect.signature(fn).parameters
]


def test_public_api_has_progress_functions():
    # guards the parametrization below against silently testing nothing
    assert len(WITH_PROGRESS) > 30


@pytest.mark.parametrize(
    "name, fn",
    [(name, fn) for name, fn, _ in PUBLIC_CALLABLES],
    ids=[name for name, _, _ in PUBLIC_CALLABLES],
)
def test_no_old_progress_arguments(name, fn):
    params = inspect.signature(fn).parameters
    old = sorted(OLD_NAMES & set(params))
    assert not old, f"{name} still takes {old}; use `progress` instead."
    extra = sorted(
        p
        for p in params
        if re.search("progress", p) and p not in NOT_TRACKERS | {"progress"}
    )
    assert (
        not extra
    ), f"{name} takes {extra}; report every step through one `progress`."


@pytest.mark.parametrize(
    "name, fn, documented",
    WITH_PROGRESS,
    ids=[name for name, _, _ in WITH_PROGRESS],
)
def test_progress_argument_is_consistent(name, fn, documented):
    param = inspect.signature(fn).parameters["progress"]
    assert param.default is None, f"{name}: progress must default to None."
    # postponed annotations are stored re-unparsed, with single quotes
    assert str(param.annotation).replace("'", '"') == ANNOTATION, (
        f"{name}: annotate progress as {ANNOTATION}, not "
        f"{param.annotation!r}."
    )
    doc = inspect.getdoc(documented) or ""
    assert re.search(
        rf"^progress : {re.escape(DOC_TYPE)}$", doc, re.MULTILINE
    ), f"{name}: document progress as `progress : {DOC_TYPE}`."
    assert "picasso.lib.Progress" in doc, (
        f"{name}: the progress docstring should point to "
        "``picasso.lib.Progress``."
    )
