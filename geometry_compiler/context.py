"""Build an evaluator.Environment from a ProblemIR (or from a dict of
already-solved values), and small Vec3 <-> sympy helpers.

This is the one place that knows how ProblemIR.entities map onto the
mini-language's notion of "vector" (dot/cross/norm/+/- all operate on
raw 3-vectors, whether the underlying Entity.kind is "point" or
"vector" -- CONTRACT.md does not distinguish them once resolved).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy

from .evaluator import Environment
from .exceptions import GeometryCompilerError

import sys
import pathlib

# shared/schema.py is a frozen, dependency-free contract file both
# sibling packages import as-is (see CONTRACT.md) -- we only read from
# it, never modify it.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.schema import Entity, ProblemIR, Vec3  # noqa: E402


def vec3_to_matrix(v: Vec3) -> sympy.Matrix:
    return sympy.Matrix([sympy.Rational(v.x), sympy.Rational(v.y), sympy.Rational(v.z)])


def matrix_to_vec3(m: sympy.Matrix) -> Vec3:
    fracs = [to_fraction(m[i, 0]) for i in range(3)]
    return Vec3(x=str(fracs[0]), y=str(fracs[1]), z=str(fracs[2]))


def to_fraction(value: sympy.Expr) -> Fraction:
    """Convert a fully-evaluated sympy scalar to an exact Python Fraction.
    Raises if the value is not an exact rational (e.g. a leftover
    symbolic sqrt) -- see exceptions.IrrationalResultError callers."""
    value = sympy.nsimplify(sympy.sympify(value))
    value = sympy.together(value)
    if not value.is_Rational:
        raise GeometryCompilerError(
            f"expected an exact rational value but got {value!r} "
            f"(not a sympy Rational -- likely an unresolved sqrt/irrational)"
        )
    return Fraction(int(value.p), int(value.q))


def _entity_vector_value(entity: Entity) -> Vec3 | None:
    """The Vec3 payload for a point/vector entity, regardless of which
    of coordinates/components is populated (kind-dependent per schema)."""
    return entity.coordinates if entity.coordinates is not None else entity.components


@dataclass
class UnknownSymbols:
    """Bookkeeping produced while building a *symbolic* Environment:
    which sympy symbols correspond to which (entity_id, component) or
    (scalar_name), so the solver knows what to solve for and how to map
    solutions back onto entities."""

    all_symbols: list[sympy.Symbol]
    # entity_id -> (sym_x, sym_y, sym_z) for unknown point/vector entities
    vector_symbols: dict[str, tuple[sympy.Symbol, sympy.Symbol, sympy.Symbol]]
    # unknown_scalars name -> symbol
    scalar_symbols: dict[str, sympy.Symbol]


def build_symbolic_context(problem_ir: ProblemIR) -> tuple[Environment, UnknownSymbols]:
    """Build the Environment used while SOLVING: given entities resolve
    to concrete sympy Rationals, `unknown` entities resolve to fresh
    sympy symbols (3 per point/vector, 6 per parametric_line -- anchor +
    direction -- though no fixture currently exercises an unknown line),
    and every name in unknown_scalars resolves to a fresh scalar symbol.
    """
    env = Environment()
    all_symbols: list[sympy.Symbol] = []
    vector_symbols: dict[str, tuple[sympy.Symbol, sympy.Symbol, sympy.Symbol]] = {}
    scalar_symbols: dict[str, sympy.Symbol] = {}

    for entity in problem_ir.entities:
        if entity.kind in ("point", "vector"):
            if entity.status == "given" or entity.status == "computed":
                v = _entity_vector_value(entity)
                if v is None:
                    raise GeometryCompilerError(
                        f"entity {entity.id!r} has status {entity.status!r} but no "
                        f"coordinates/components populated"
                    )
                env.vectors[entity.id] = vec3_to_matrix(v)
            elif entity.status == "unknown":
                sx, sy, sz = sympy.symbols(f"{entity.id}__x {entity.id}__y {entity.id}__z")
                env.vectors[entity.id] = sympy.Matrix([sx, sy, sz])
                vector_symbols[entity.id] = (sx, sy, sz)
                all_symbols.extend((sx, sy, sz))
            else:
                raise GeometryCompilerError(
                    f"entity {entity.id!r} has unrecognized status {entity.status!r}"
                )
        elif entity.kind == "parametric_line":
            if entity.status == "given" or entity.status == "computed":
                if entity.anchor is None or entity.direction is None:
                    raise GeometryCompilerError(
                        f"parametric_line {entity.id!r} has status {entity.status!r} "
                        f"but anchor/direction not both populated"
                    )
                env.lines[entity.id] = (
                    vec3_to_matrix(entity.anchor),
                    vec3_to_matrix(entity.direction),
                )
            elif entity.status == "unknown":
                # Not exercised by any of the three fixtures, but supported
                # for the "generalize to a similarly-shaped fourth problem"
                # goal in CONTRACT.md.
                ax, ay, az = sympy.symbols(f"{entity.id}__Ax {entity.id}__Ay {entity.id}__Az")
                dx, dy, dz = sympy.symbols(f"{entity.id}__dx {entity.id}__dy {entity.id}__dz")
                anchor_m = sympy.Matrix([ax, ay, az])
                direction_m = sympy.Matrix([dx, dy, dz])
                env.lines[entity.id] = (anchor_m, direction_m)
                vector_symbols[f"{entity.id}.A"] = (ax, ay, az)
                vector_symbols[f"{entity.id}.d"] = (dx, dy, dz)
                all_symbols.extend((ax, ay, az, dx, dy, dz))
            else:
                raise GeometryCompilerError(
                    f"entity {entity.id!r} has unrecognized status {entity.status!r}"
                )
        else:
            raise GeometryCompilerError(f"entity {entity.id!r} has unrecognized kind {entity.kind!r}")

    for name in problem_ir.unknown_scalars:
        sym = sympy.Symbol(name)
        env.scalars[name] = sym
        scalar_symbols[name] = sym
        all_symbols.append(sym)

    return env, UnknownSymbols(all_symbols, vector_symbols, scalar_symbols)


def build_numeric_context(
    problem_ir: ProblemIR,
    solved_vectors: dict[str, sympy.Matrix] | None = None,
    solved_scalars: dict[str, Fraction] | None = None,
) -> Environment:
    """Build a fully-numeric Environment: every entity id resolves to a
    concrete sympy Rational vector, and every unknown_scalars name
    resolves to a concrete sympy Rational. `given`/`computed` entities
    come straight from the ProblemIR; entities with status == "unknown"
    MUST have a value supplied via `solved_vectors` (keyed by entity id).

    Used for the independent re-verification pass and for evaluating
    query_expr -- both want zero symbols left in the environment.
    """
    solved_vectors = solved_vectors or {}
    solved_scalars = solved_scalars or {}
    env = Environment()

    for entity in problem_ir.entities:
        if entity.kind in ("point", "vector"):
            if entity.status in ("given", "computed"):
                v = _entity_vector_value(entity)
                if v is None:
                    raise GeometryCompilerError(
                        f"entity {entity.id!r} has status {entity.status!r} but no "
                        f"coordinates/components populated"
                    )
                env.vectors[entity.id] = vec3_to_matrix(v)
            elif entity.status == "unknown":
                if entity.id not in solved_vectors:
                    raise GeometryCompilerError(
                        f"no solved value supplied for unknown entity {entity.id!r}"
                    )
                env.vectors[entity.id] = solved_vectors[entity.id]
        elif entity.kind == "parametric_line":
            if entity.status in ("given", "computed"):
                env.lines[entity.id] = (
                    vec3_to_matrix(entity.anchor),
                    vec3_to_matrix(entity.direction),
                )
            elif entity.status == "unknown":
                key_a, key_d = f"{entity.id}.A", f"{entity.id}.d"
                if key_a not in solved_vectors or key_d not in solved_vectors:
                    raise GeometryCompilerError(
                        f"no solved anchor/direction supplied for unknown line {entity.id!r}"
                    )
                env.lines[entity.id] = (solved_vectors[key_a], solved_vectors[key_d])

    for name in problem_ir.unknown_scalars:
        if name not in solved_scalars:
            raise GeometryCompilerError(f"no solved value supplied for unknown scalar {name!r}")
        frac = solved_scalars[name]
        env.scalars[name] = sympy.Rational(frac.numerator, frac.denominator)

    return env
