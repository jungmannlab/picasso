"""Test that the public API documents its numeric defaults truthfully:
where the description of a parameter says "Default is X", X must be the
default in the signature.

A default left in camera pixels by the switch to nm in Picasso 1.0
(``postprocess.select_binding_event_cores`` kept ``r_max=0.1`` while
its docstring said 10 nm) passed every other test, since those passed
``r_max`` explicitly or accepted an empty result.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import inspect
import math
import re

import pytest

from tests.test_progress_api import PUBLIC_CALLABLES

#: "Default is 10", "Default is ``0.5``", "Default is 20,000",
#: "Default is 20/130"
DEFAULT = re.compile(
    r"[Dd]efault is (?:``)?"
    r"(?P<number>[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?)"
    r"(?:\s*/\s*(?P<divisor>\d+(?:\.\d+)?))?"
)


def _parameter_descriptions(doc: str) -> dict[str, str]:
    """Description of each parameter in the Parameters and Other
    Parameters sections of a numpydoc docstring, by parameter name."""
    lines = doc.splitlines()
    descriptions = {}
    names = None
    in_section = False
    for i, line in enumerate(lines):
        underlined = (
            i + 1 < len(lines)
            and lines[i + 1].strip()
            and set(lines[i + 1].strip()) == {"-"}
        )
        if underlined and not line.startswith(" "):
            in_section = line.strip() in ("Parameters", "Other Parameters")
            names = None
            continue
        if not in_section or set(line.strip()) == {"-"}:
            continue
        if line and not line.startswith(" "):
            names = [
                name.strip().lstrip("*")
                for name in line.split(" : ")[0].split(",")
            ]
            for name in names:
                descriptions[name] = ""
        elif names is not None:
            for name in names:
                descriptions[name] += " " + line.strip()
    return descriptions


def _documented_defaults():
    """(name, documented value, default in the code) of every numeric
    default of the public API that its docstring states."""
    out = []
    for name, fn, documented in PUBLIC_CALLABLES:
        doc = inspect.getdoc(documented) or ""
        descriptions = _parameter_descriptions(doc)
        for param_name, param in inspect.signature(fn).parameters.items():
            default = param.default
            if isinstance(default, bool) or not isinstance(
                default, (int, float)
            ):
                continue
            match = DEFAULT.search(descriptions.get(param_name, ""))
            if match is None:
                continue
            value = float(match["number"].replace(",", ""))
            if match["divisor"]:
                value /= float(match["divisor"])
            out.append((f"{name}({param_name})", value, default))
    return out


DOCUMENTED_DEFAULTS = _documented_defaults()


def test_public_api_has_documented_defaults():
    # guards the parametrization below against silently testing nothing,
    # e.g., after a change of the docstring style
    assert len(DOCUMENTED_DEFAULTS) > 50


@pytest.mark.parametrize(
    "name, documented, default",
    DOCUMENTED_DEFAULTS,
    ids=[name for name, _, _ in DOCUMENTED_DEFAULTS],
)
def test_documented_default_is_the_default(name, documented, default):
    assert math.isclose(documented, default, rel_tol=1e-9), (
        f"{name} defaults to {default!r}, but its docstring says "
        f"{documented:g}."
    )


def test_reads_the_documented_default():
    # the parsing the check relies on, on a default that disagrees with
    # its docstring
    def link(locs, r_max=0.1):
        """Link localizations.

        Parameters
        ----------
        locs : pd.DataFrame
            Localizations.
        r_max : float, optional
            Maximum distance (nm). Default is 10.
        """

    descriptions = _parameter_descriptions(inspect.getdoc(link))
    match = DEFAULT.search(descriptions["r_max"])
    assert float(match["number"]) == 10
    assert "Default" not in descriptions["locs"]
