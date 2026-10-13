"""
picasso.expression
~~~~~~~~~~~~~~~~~~

Conditions on localization columns written as text, e.g.
``photons > 500 and 0.8 < sx < 1.2``, used to filter localizations.

An expression is parsed, not executed: only the syntax listed below is
accepted, so a typo gives a readable error instead of running code.

Syntax
------
Names
    Column names (``x``, ``photons``, ``lpx``, ...), the constants
    ``pi``, ``inf`` and ``nan``, and ``pixelsize`` (camera pixel size in
    nm, from the metadata) when filtering through :func:`filter_locs`.
Arithmetic
    ``+ - * / // % **``. Unsigned integer columns (e.g. ``frame``) are
    converted to signed integers, so differences may be negative.
Comparisons
    ``< <= > >= == !=``, also chained: ``0.8 < sx < 1.2``.
Logic
    ``and``, ``or``, ``not``. ``&``, ``|`` and ``~`` work on conditions
    too, but bind tighter than comparisons, so each comparison then
    needs parentheses: ``(photons > 500) & (sx < 1.2)``.
Functions applied to each localization
    ``abs sqrt log log10 exp floor ceil isfinite``.
Statistics over the localizations being filtered
    ``mean median std min max sum`` (one argument) and
    ``percentile(values, q)`` with ``q`` in 0-100. Non-finite values
    (NaN, infinity) are ignored. Their argument can be any expression,
    e.g. ``median(sqrt(lpx**2 + lpy**2))``.

A localization is kept where the condition is True. Comparisons with
NaN are False, so ``photons > 500`` removes localizations whose
``photons`` is NaN (and ``not photons > 500`` keeps them).

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import lib

__all__ = [
    "ExpressionError",
    "filter_locs",
    "condition",
    "evaluate",
    "variables_from_info",
]

#: Constants available in every expression.
CONSTANTS = {"pi": np.pi, "inf": np.inf, "nan": np.nan}

#: Functions applied to each localization separately.
ELEMENTWISE_FUNCTIONS: dict[str, Callable] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "floor": np.floor,
    "ceil": np.ceil,
    "isfinite": np.isfinite,
}

#: Statistics over all localizations being filtered, with their number
#: of arguments.
STATISTICS: dict[str, tuple[Callable, int]] = {
    "mean": (np.mean, 1),
    "median": (np.median, 1),
    "std": (np.std, 1),
    "min": (np.min, 1),
    "max": (np.max, 1),
    "sum": (np.sum, 1),
    "percentile": (np.percentile, 2),
}

#: Names :func:`variables_from_info` can provide from the metadata.
METADATA_VARIABLES = ("pixelsize",)

_COMPARISONS = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}

_ARITHMETIC = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.FloorDiv: np.floor_divide,
    ast.Mod: np.mod,
    ast.Pow: np.power,
}

_SYMBOLS = {
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.MatMult: "@",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
}


class ExpressionError(ValueError):
    """An expression that cannot be parsed or evaluated."""


def _parse(expression: str) -> ast.expr:
    """Syntax tree of ``expression``, or ExpressionError."""
    if not expression.strip():
        raise ExpressionError("The expression is empty.")
    try:
        return ast.parse(expression.strip(), mode="eval").body
    except SyntaxError as e:
        raise ExpressionError(f"Invalid syntax: {e.msg}.") from None


def _is_condition(value: Any) -> bool:
    """Whether ``value`` is a boolean scalar or array."""
    return np.asarray(value).dtype == bool


class _Evaluator:
    """Evaluates a parsed expression against columns and variables."""

    def __init__(
        self,
        columns: Mapping[str, Any],
        variables: Mapping[str, float],
    ) -> None:
        self.columns = columns
        self.variables = variables
        self.cache: dict[str, np.ndarray] = {}

    def column(self, name: str) -> np.ndarray:
        if name not in self.cache:
            values = np.asarray(self.columns[name])
            if values.dtype.kind == "u":
                # frame - 10 must not wrap around below 0
                values = values.astype(np.int64)
            self.cache[name] = values
        return self.cache[name]

    def visit(self, node: ast.AST) -> Any:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ExpressionError(
                f"'{ast.unparse(node)}' is not supported in expressions."
            )
        return method(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (bool, int, float)):
            return node.value
        raise ExpressionError(
            f"Only numbers are allowed as constants, not {node.value!r}."
        )

    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id
        if name in self.columns:
            return self.column(name)
        if name in self.variables:
            return self.variables[name]
        if name in CONSTANTS:
            return CONSTANTS[name]
        if name in ELEMENTWISE_FUNCTIONS or name in STATISTICS:
            raise ExpressionError(
                f"'{name}' is a function; call it, e.g., {name}(photons)."
            )
        known = list(self.columns) + list(self.variables)
        hint = (
            " Use 'pixelsize' only with metadata that contains the pixel"
            " size."
            if name in METADATA_VARIABLES
            else ""
        )
        raise ExpressionError(
            f"Unknown name '{name}'. Available: {', '.join(known)}.{hint}"
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return np.negative(value)
        if isinstance(node.op, ast.UAdd):
            return value
        # 'not' and '~'
        if not _is_condition(value):
            op = "not" if isinstance(node.op, ast.Not) else "~"
            raise ExpressionError(
                f"'{op}' needs a condition, e.g., not (photons > 500)."
            )
        return np.logical_not(value)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = type(node.op)
        if op in _ARITHMETIC:
            return _ARITHMETIC[op](left, right)
        if op in (ast.BitAnd, ast.BitOr):
            if _is_condition(left) and _is_condition(right):
                if op is ast.BitAnd:
                    return np.logical_and(left, right)
                return np.logical_or(left, right)
            symbol = _SYMBOLS[op]
            raise ExpressionError(
                f"'{symbol}' binds tighter than comparisons, so"
                f" '{ast.unparse(node)}' combines numbers. Put each"
                f" comparison in parentheses, e.g., (photons > 500)"
                f" {symbol} (sx < 1.2), or use"
                f" '{'and' if op is ast.BitAnd else 'or'}'."
            )
        if op is ast.BitXor:
            raise ExpressionError(
                "'^' is not supported; use ** for powers, e.g., lpx**2."
            )
        raise ExpressionError(
            f"'{_SYMBOLS.get(op, op.__name__)}' is not supported in"
            " expressions."
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        values = [self.visit(v) for v in node.values]
        word = "and" if isinstance(node.op, ast.And) else "or"
        for value, sub in zip(values, node.values):
            if not _is_condition(value):
                raise ExpressionError(
                    f"'{word}' combines conditions, but '{ast.unparse(sub)}'"
                    " is a number."
                )
        combine = np.logical_and if word == "and" else np.logical_or
        return combine.reduce(np.broadcast_arrays(*values))

    def visit_Compare(self, node: ast.Compare) -> Any:
        result = None
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _COMPARISONS:
                raise ExpressionError(
                    f"'{_SYMBOLS.get(type(op), type(op).__name__)}' is not"
                    " supported; use < <= > >= == !=."
                )
            right = self.visit(comparator)
            current = _COMPARISONS[type(op)](left, right)
            result = (
                current if result is None else np.logical_and(result, current)
            )
            left = right
        return result

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ExpressionError(
                f"'{ast.unparse(node.func)}' is not a supported function."
            )
        name = node.func.id
        if node.keywords:
            raise ExpressionError(f"{name}() takes no keyword arguments.")
        if name in ELEMENTWISE_FUNCTIONS:
            if len(node.args) != 1:
                raise ExpressionError(f"{name}() takes one argument.")
            return ELEMENTWISE_FUNCTIONS[name](self.visit(node.args[0]))
        if name in STATISTICS:
            function, n_args = STATISTICS[name]
            if len(node.args) != n_args:
                usage = (
                    "one argument"
                    if n_args == 1
                    else "two arguments, e.g., percentile(photons, 95)"
                )
                raise ExpressionError(f"{name}() takes {usage}.")
            values = np.asarray(self.visit(node.args[0]), dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return np.nan
            if n_args == 2:
                q = self.visit(node.args[1])
                if np.ndim(q) != 0 or not 0 <= q <= 100:
                    raise ExpressionError(
                        "The percentile must be one number from 0 to 100."
                    )
                return function(values, q)
            return function(values)
        functions = list(ELEMENTWISE_FUNCTIONS) + list(STATISTICS)
        raise ExpressionError(
            f"Unknown function '{name}'. Available: {', '.join(functions)}."
        )


def names(expression: str) -> set[str]:
    """Names an expression refers to, without functions and constants.

    These are the columns it needs, plus any variables such as
    ``pixelsize``.

    Parameters
    ----------
    expression : str
        Expression, see the module docstring for the syntax.

    Returns
    -------
    names : set of str
        Referenced names.

    Raises
    ------
    ExpressionError
        If the expression cannot be parsed.
    """
    tree = _parse(expression)
    called = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and id(node) not in called
        and node.id not in CONSTANTS
    }


def evaluate(
    expression: str,
    columns: Mapping[str, Any],
    variables: Mapping[str, float] | None = None,
) -> Any:
    """Value of an expression: one value per localization, or a scalar
    if it uses no column.

    Parameters
    ----------
    expression : str
        Expression, see the module docstring for the syntax.
    columns : Mapping[str, array-like]
        Column name -> values, e.g. a ``pd.DataFrame``. Only the columns
        the expression uses are read.
    variables : Mapping[str, float] or None, optional
        Extra scalar names, e.g. from :func:`variables_from_info`.
        Columns take precedence over variables with the same name.
        Default None.

    Returns
    -------
    value : np.ndarray or scalar
        Result of the expression.

    Raises
    ------
    ExpressionError
        If the expression cannot be parsed or evaluated.
    """
    tree = _parse(expression)
    evaluator = _Evaluator(columns, variables or {})
    with np.errstate(all="ignore"):
        return evaluator.visit(tree)


def condition(
    expression: str,
    columns: Mapping[str, Any],
    variables: Mapping[str, float] | None = None,
) -> lib.BoolArray1D:
    """Which localizations satisfy a condition.

    Parameters
    ----------
    expression : str
        Condition, e.g. ``"photons > 500 and 0.8 < sx < 1.2"``. See the
        module docstring for the syntax.
    columns : Mapping[str, array-like]
        Column name -> values, e.g. a ``pd.DataFrame``. Only the columns
        the condition uses are read.
    variables : Mapping[str, float] or None, optional
        Extra scalar names, e.g. from :func:`variables_from_info`.
        Default None.

    Returns
    -------
    keep : lib.BoolArray1D
        True for each localization that satisfies the condition.

    Raises
    ------
    ExpressionError
        If the expression cannot be evaluated, is not a condition (e.g.
        ``photons * 2``), or does not depend on any column (e.g.
        ``mean(photons) > 500``).
    """
    value = evaluate(expression, columns, variables)
    if not _is_condition(value):
        raise ExpressionError(
            "The expression must be a condition that is True or False,"
            " e.g., photons > 500."
        )
    if np.ndim(value) != 1:
        raise ExpressionError(
            "The condition is the same for every localization; compare a"
            " column, e.g., photons > mean(photons)."
        )
    return value


def variables_from_info(info: list[dict] | dict | None) -> dict[str, float]:
    """Scalars that expressions may use, taken from the metadata.

    Parameters
    ----------
    info : list of dicts, dict or None
        Localization metadata.

    Returns
    -------
    variables : dict[str, float]
        ``{"pixelsize": nm per camera pixel}`` if the metadata has
        ``Pixelsize``, else empty.
    """
    if not info:
        return {}
    pixelsize = lib.get_from_metadata(info, "Pixelsize")
    if pixelsize is None:
        return {}
    return {"pixelsize": float(pixelsize)}


def filter_locs(
    locs: pd.DataFrame,
    expression: str,
    info: list[dict] | dict | None = None,
) -> pd.DataFrame:
    """Keep the localizations that satisfy a condition.

    Statistics such as ``median(photons)`` are computed over ``locs``.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    expression : str
        Condition, e.g. ``"photons > 500 and lpx * pixelsize < 10"``.
        See the module docstring for the syntax.
    info : list of dicts, dict or None, optional
        Metadata of ``locs``; provides ``pixelsize``. Default None.

    Returns
    -------
    filtered_locs : pd.DataFrame
        Localizations that satisfy the condition, with a new index.

    Raises
    ------
    ExpressionError
        If the expression is not a valid condition on ``locs``.

    Examples
    --------
    >>> locs = filter_locs(locs, "photons > median(photons)", info)
    """
    keep = condition(expression, locs, variables_from_info(info))
    return locs[keep].reset_index(drop=True)


def range_expression(
    column: str, vmin: float, vmax: float, inclusive: bool = False
) -> str:
    """Condition that keeps values of a column within a range.

    Parameters
    ----------
    column : str
        Column name.
    vmin, vmax : float
        Range limits.
    inclusive : bool, optional
        If True, values equal to a limit are kept. Default False.

    Returns
    -------
    expression : str
        E.g. ``"0.8 < sx < 1.2"``; :func:`parse_range` reads it back.
    """
    op = "<=" if inclusive else "<"
    return f"{float(vmin)!r} {op} {column} {op} {float(vmax)!r}"


def _number(node: ast.expr) -> float | None:
    """Value of a numeric literal such as ``1.5``, ``-inf`` or ``nan``."""
    sign = 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.USub, ast.UAdd)
    ):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        node = node.operand
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return sign * float(node.value)
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return sign * float(CONSTANTS[node.id])
    return None


def parse_range(expression: str) -> tuple[str, float, float] | None:
    """Column and limits of a range condition ``vmin < column < vmax``
    (or with ``<=``), as written by :func:`range_expression`.

    Parameters
    ----------
    expression : str
        Expression.

    Returns
    -------
    column_range : tuple or None
        ``(column, vmin, vmax)``, or None if the expression is not such
        a range.
    """
    try:
        tree = _parse(expression)
    except ExpressionError:
        return None
    if not (
        isinstance(tree, ast.Compare)
        and len(tree.ops) == 2
        and all(isinstance(op, (ast.Lt, ast.LtE)) for op in tree.ops)
        and isinstance(tree.comparators[0], ast.Name)
        and tree.comparators[0].id not in CONSTANTS
    ):
        return None
    vmin = _number(tree.left)
    vmax = _number(tree.comparators[1])
    if vmin is None or vmax is None:
        return None
    return tree.comparators[0].id, vmin, vmax
