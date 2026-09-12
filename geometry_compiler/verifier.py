"""Independent re-verification pass.

This is deliberately a SEPARATE code path from solver.py's residual
construction, even though it re-parses the same equation strings: the
solver may have used only a subset of an over-determined system (sympy
picks whichever rows it needs), so trusting the solver's own residuals
would not catch a case where the *full* equation set is inconsistent
but the subset solver happened to use is not. Per CONTRACT.md hard rule
#1 ("verify algebraically, never eyeball"), we substitute the solved
values back into every declared equation from scratch and require an
EXACT zero residual, using a completely numeric Environment (no sympy
symbols left at all).
"""
from __future__ import annotations

import sympy

from .evaluator import Environment, evaluate
from .expr_parser import parse_equation

import sys
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schema import ProblemIR, VerificationResult, rational  # noqa: E402
from fractions import Fraction  # noqa: E402


def _residual_fraction(lhs_val, rhs_val) -> Fraction:
    """A single nonnegative exact Fraction that is 0 iff lhs == rhs
    exactly. For vectors this is the squared Euclidean distance between
    them (0 iff every component matches exactly); for scalars it is the
    squared difference. Squaring keeps the residual a single scalar
    (matching VerificationResult.residual's shape) while staying exact
    and unambiguous about the "is it exactly zero" question -- there is
    no sign-cancellation subtlety to worry about."""
    lv, rv = hasattr(lhs_val, "shape"), hasattr(rhs_val, "shape")
    if lv != rv:
        raise ValueError("cannot compare a vector to a scalar")
    if lv:
        diff = lhs_val - rhs_val
        sq = (diff.T * diff)[0, 0]
    else:
        sq = (lhs_val - rhs_val) ** 2
    sq = sympy.nsimplify(sympy.together(sq))
    if not sq.is_Rational:
        raise ValueError(f"residual {sq!r} is not an exact rational (unresolved irrational?)")
    return Fraction(int(sq.p), int(sq.q))


def verify_equations(
    problem_ir: ProblemIR, numeric_env: Environment
) -> list[VerificationResult]:
    """Re-evaluate every equation in problem_ir.equations against a
    fully-numeric Environment and report an exact pass/fail + residual
    for each, independent of whatever sympy.solve internally used."""
    results: list[VerificationResult] = []
    for eq_str in problem_ir.equations:
        lhs_node, rhs_node = parse_equation(eq_str)
        lhs_val = evaluate(lhs_node, numeric_env)
        rhs_val = evaluate(rhs_node, numeric_env)
        residual = _residual_fraction(lhs_val, rhs_val)
        results.append(
            VerificationResult(
                check=eq_str,
                passed=(residual == 0),
                residual=rational(residual),
            )
        )
    return results
