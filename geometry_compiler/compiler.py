"""The `compile()` entrypoint: ProblemIR -> SceneGraph.

Pipeline (see README.md for the long-form write-up):
  1. solver.solve_system      -- build the full symbolic equation system
                                  from every `unknown` entity's components
                                  + unknown_scalars, and solve it exactly
                                  with sympy. Raises a GeometryCompilerError
                                  subclass if the system has no solution,
                                  more than one, or leaves free parameters.
  2. populate every entity    -- `unknown` -> `computed`, with
                                  coordinates/components/anchor+direction
                                  always filled in (never left None).
  3. verifier.verify_equations -- INDEPENDENTLY re-substitute the solved
                                  values into every one of
                                  ProblemIR.equations from scratch (fresh
                                  parse, fully-numeric environment, no
                                  reliance on whatever subset sympy.solve
                                  happened to use) and require an exact
                                  zero residual for each.
  4. evaluate query_expr       -- against the same fully-numeric
                                  environment, producing final_answer as
                                  an exact Rational string.
  5. verified = True iff every verification passed.
"""
from __future__ import annotations

import dataclasses
import sys
import pathlib

from .context import build_numeric_context, matrix_to_vec3, to_fraction
from .evaluator import evaluate
from .exceptions import GeometryCompilerError, IrrationalResultError
from .expr_parser import parse_expr
from .solver import solve_system
from .verifier import verify_equations

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schema import Entity, ProblemIR, SceneGraph, rational  # noqa: E402


def _populate_entity(entity: Entity, solve_result) -> Entity:
    """Return a NEW Entity: `given`/`computed` entities pass through
    unchanged (as a fresh object, so callers can't accidentally mutate
    the compiler's internals via the input ProblemIR); `unknown`
    entities become `computed` with their geometric field(s) filled in
    from the solved values."""
    if entity.status in ("given", "computed"):
        return dataclasses.replace(entity)

    if entity.kind in ("point", "vector"):
        vec = solve_result.solved_vectors[entity.id]
        v3 = matrix_to_vec3(vec)
        new = dataclasses.replace(entity, status="computed")
        if entity.kind == "point":
            new.coordinates = v3
        else:
            new.components = v3
        return new

    if entity.kind == "parametric_line":
        anchor = solve_result.solved_vectors[f"{entity.id}.A"]
        direction = solve_result.solved_vectors[f"{entity.id}.d"]
        new = dataclasses.replace(entity, status="computed")
        new.anchor = matrix_to_vec3(anchor)
        new.direction = matrix_to_vec3(direction)
        return new

    raise GeometryCompilerError(f"entity {entity.id!r} has unrecognized kind {entity.kind!r}")


def compile_problem_ir(problem_ir: ProblemIR) -> SceneGraph:
    """ProblemIR -> SceneGraph. See module docstring for the pipeline."""

    solve_result = solve_system(problem_ir)

    populated_entities = [_populate_entity(e, solve_result) for e in problem_ir.entities]

    solved_scalars_str = {
        name: rational(frac) for name, frac in solve_result.solved_scalars.items()
    }

    numeric_env = build_numeric_context(
        problem_ir,
        solved_vectors=solve_result.solved_vectors,
        solved_scalars=solve_result.solved_scalars,
    )

    verifications = verify_equations(problem_ir, numeric_env)
    verified = all(v.passed for v in verifications)

    query_node = parse_expr(problem_ir.query_expr)
    query_value = evaluate(query_node, numeric_env)
    if hasattr(query_value, "shape"):
        raise GeometryCompilerError(
            f"query_expr {problem_ir.query_expr!r} evaluated to a 3-vector, but "
            "final_answer must be a scalar Rational"
        )
    try:
        final_answer_fraction = to_fraction(query_value)
    except GeometryCompilerError as exc:
        raise IrrationalResultError(
            f"query_expr {problem_ir.query_expr!r} evaluated to {query_value!r}, "
            "which is not an exact rational -- see README.md 'Known limitations' "
            "(genuinely irrational final answers, e.g. a non-perfect-square norm, "
            "are not currently supported rather than being silently floated)."
        ) from exc

    relations = [dataclasses.replace(r) for r in problem_ir.relations]

    return SceneGraph(
        problem_id=problem_ir.problem_id,
        entities=populated_entities,
        relations=relations,
        solved_scalars=solved_scalars_str,
        verifications=verifications,
        verified=verified,
        final_answer=rational(final_answer_fraction),
    )
