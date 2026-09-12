"""Build the full symbolic equation system from ProblemIR.equations and
solve it with sympy, using exact rational arithmetic throughout.

Generic vs. special-cased (see geometry_compiler/README.md for the full
write-up): this module is deliberately NOT keyed on `problem_type`. It
turns every entity with status == "unknown" (3 sympy symbols for a
point/vector, 6 for a parametric_line) plus every unknown_scalars name
into a symbol, evaluates every equation string against that symbolic
environment to get a scalar residual expression per component, and
hands the whole flat list to `sympy.solve`. All three fixture families
happen to be exactly linear in these unknowns (a scalar or an unknown
vector is never multiplied by another unknown), so this reduces to one
linear solve either way -- but we do not special-case on linearity or
on `problem_type`; `sympy.solve` is used because it also degrades
gracefully to polynomial systems if a future problem introduces one.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy

from .context import UnknownSymbols, build_symbolic_context, to_fraction
from .evaluator import evaluate
from .exceptions import (
    AmbiguousSystemError,
    SystemInconsistentError,
    UnderdeterminedSystemError,
)
from .expr_parser import parse_equation

import sys
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schema import ProblemIR  # noqa: E402


@dataclass
class SolveResult:
    solved_vectors: dict[str, sympy.Matrix]  # entity id (or "line.A"/"line.d") -> Matrix(3,1) of Fractions-as-Rational
    solved_scalars: dict[str, Fraction]       # unknown_scalars name -> Fraction


def _residuals_for_equation(eq_str: str, env) -> list[sympy.Expr]:
    lhs_node, rhs_node = parse_equation(eq_str)
    lhs_val = evaluate(lhs_node, env)
    rhs_val = evaluate(rhs_node, env)
    lv, rv = hasattr(lhs_val, "shape"), hasattr(rhs_val, "shape")
    if lv != rv:
        raise SystemInconsistentError(
            f"equation {eq_str!r} compares a vector to a scalar -- malformed equation"
        )
    if lv:
        diff = lhs_val - rhs_val
        return [sympy.expand(diff[i, 0]) for i in range(diff.shape[0])]
    return [sympy.expand(lhs_val - rhs_val)]


def solve_system(problem_ir: ProblemIR) -> SolveResult:
    env, unk = build_symbolic_context(problem_ir)

    residuals: list[sympy.Expr] = []
    for eq_str in problem_ir.equations:
        residuals.extend(_residuals_for_equation(eq_str, env))

    if not unk.all_symbols:
        # No unknowns at all -- nothing to solve, but we still want to
        # fall through to the caller's independent verification pass on
        # `residuals` (all should already be exactly zero if the given
        # data is self-consistent).
        return SolveResult(solved_vectors={}, solved_scalars={})

    solutions = sympy.solve(residuals, unk.all_symbols, dict=True)

    if not solutions:
        raise SystemInconsistentError(
            "no exact solution exists for the equation system built from "
            f"ProblemIR.equations={problem_ir.equations!r} over unknowns "
            f"{[str(s) for s in unk.all_symbols]!r} -- the equations are "
            "mutually inconsistent."
        )
    if len(solutions) > 1:
        raise AmbiguousSystemError(
            f"found {len(solutions)} distinct solution branches for unknowns "
            f"{[str(s) for s in unk.all_symbols]!r}; geometry_compiler expects "
            "a uniquely determined problem. Branches: "
            f"{[{str(k): v for k, v in sol.items()} for sol in solutions]!r}"
        )

    solution = solutions[0]

    missing_or_free = []
    for sym in unk.all_symbols:
        if sym not in solution:
            missing_or_free.append(sym)
            continue
        remaining_free = solution[sym].free_symbols & set(unk.all_symbols)
        if remaining_free:
            missing_or_free.append(sym)
    if missing_or_free:
        raise UnderdeterminedSystemError(
            f"the equation system does not uniquely pin down "
            f"{[str(s) for s in missing_or_free]!r} -- free parameter(s) remain. "
            "geometry_compiler requires fully determined problems."
        )

    solved_scalar_values: dict[sympy.Symbol, Fraction] = {
        sym: to_fraction(solution[sym]) for sym in unk.all_symbols
    }

    solved_vectors: dict[str, sympy.Matrix] = {}
    for key, (sx, sy, sz) in unk.vector_symbols.items():
        solved_vectors[key] = sympy.Matrix(
            [
                sympy.Rational(solved_scalar_values[sx].numerator, solved_scalar_values[sx].denominator),
                sympy.Rational(solved_scalar_values[sy].numerator, solved_scalar_values[sy].denominator),
                sympy.Rational(solved_scalar_values[sz].numerator, solved_scalar_values[sz].denominator),
            ]
        )

    solved_scalars: dict[str, Fraction] = {
        name: solved_scalar_values[sym] for name, sym in unk.scalar_symbols.items()
    }

    return SolveResult(solved_vectors=solved_vectors, solved_scalars=solved_scalars)
