"""Evaluate mini-language AST nodes (see expr_parser.py) against a locked
down `Environment`, producing sympy values: either a 3x1 sympy.Matrix
("vector") or a scalar sympy.Expr ("scalar").

This module has no notion of ProblemIR/SceneGraph or of solving --
geometry_compiler.solver builds the Environment (mapping entity ids to
either concrete sympy Rationals or fresh sympy symbols for `unknown`
entities) and geometry_compiler.verifier rebuilds a second, fully
numeric Environment for the independent re-check. Both reuse this one
evaluator so equations and query_expr are always interpreted exactly
the same way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import sympy

from .exceptions import GeometryCompilerError, UnknownIdentifierError
from .expr_parser import BinOp, Call, Ident, Node, Num, UnaryOp

ZERO_VECTOR = sympy.Matrix([sympy.Integer(0), sympy.Integer(0), sympy.Integer(0)])


@dataclass
class Environment:
    """Symbol table used while evaluating one mini-language expression.

    vectors: entity id -> sympy.Matrix(3, 1)   (points AND vectors alike;
             the mini-language does not distinguish "point" from "vector"
             once resolved -- both are 3-vectors, per CONTRACT.md)
    scalars: unknown_scalars name -> sympy.Symbol (or, once solved, a
             concrete sympy Rational)
    lines:   parametric_line entity id -> (anchor_matrix, direction_matrix),
             only reachable via A(line_id) / d(line_id), never as a bare
             identifier (a raw line id is not itself a 3-vector).
    """

    vectors: dict[str, sympy.Matrix] = field(default_factory=dict)
    scalars: dict[str, sympy.Expr] = field(default_factory=dict)
    lines: dict[str, tuple[sympy.Matrix, sympy.Matrix]] = field(default_factory=dict)

    def resolve_ident(self, name: str):
        if name == "zero":
            return ZERO_VECTOR
        if name in self.vectors:
            return self.vectors[name]
        if name in self.scalars:
            return self.scalars[name]
        raise UnknownIdentifierError(
            f"identifier {name!r} is not a declared Entity id, a declared "
            f"unknown_scalars name, or the reserved constant 'zero'"
        )


def _is_vector(value) -> bool:
    return isinstance(value, sympy.MatrixBase)


def exact_sqrt(value: sympy.Expr) -> sympy.Expr:
    """sqrt() that returns an EXACT sympy Rational when the radicand is a
    perfect-square rational (numerator and denominator both perfect
    squares), and otherwise falls back to a symbolic sympy.sqrt (left
    irrational -- see geometry_compiler/README.md "Known limitations").

    This is what lets M26S1J21Q64's `norm((6,2,3))` collapse to the
    exact integer 7 instead of a symbolic sqrt(49) or a float 7.0,
    which in turn is what lets `dot(F, v) / norm(v)` come out as the
    exact Rational 18/7 rather than a lingering sqrt or a float.
    """
    value = sympy.nsimplify(value)
    if value.is_Rational:
        frac = Fraction(int(value.p), int(value.q))
        if frac < 0:
            # norm() should never be called on a negative radicand in a
            # well-formed problem, but guard rather than silently
            # returning a complex result.
            raise GeometryCompilerError(
                f"norm(): radicand {frac} is negative, cannot take a real sqrt"
            )
        num, den = frac.numerator, frac.denominator
        sq_num, sq_den = math.isqrt(num), math.isqrt(den)
        if sq_num * sq_num == num and sq_den * sq_den == den:
            return sympy.Rational(sq_num, sq_den)
    # Not (exactly) a perfect square -- leave symbolic/irrational.
    return sympy.sqrt(value)


def evaluate(node: Node, env: Environment):
    """Evaluate one AST node. Returns a sympy.Matrix(3, 1) for
    vector-valued nodes or a sympy.Expr for scalar-valued nodes."""

    if isinstance(node, Num):
        return sympy.Integer(node.value)

    if isinstance(node, Ident):
        return env.resolve_ident(node.name)

    if isinstance(node, UnaryOp):
        val = evaluate(node.operand, env)
        if node.op == "-":
            return -val
        raise GeometryCompilerError(f"unsupported unary operator {node.op!r}")

    if isinstance(node, BinOp):
        return _eval_binop(node, env)

    if isinstance(node, Call):
        return _eval_call(node, env)

    raise GeometryCompilerError(f"unrecognized AST node: {node!r}")


def _require_vector(value, where: str) -> sympy.Matrix:
    if not _is_vector(value):
        raise GeometryCompilerError(f"{where}: expected a 3-vector, got a scalar ({value})")
    return value


def _require_scalar(value, where: str) -> sympy.Expr:
    if _is_vector(value):
        raise GeometryCompilerError(f"{where}: expected a scalar, got a 3-vector ({value.T})")
    return value


def _eval_binop(node: BinOp, env: Environment):
    left = evaluate(node.left, env)
    right = evaluate(node.right, env)
    lv, rv = _is_vector(left), _is_vector(right)

    if node.op in ("+", "-"):
        if lv != rv:
            raise GeometryCompilerError(
                f"cannot {'add' if node.op == '+' else 'subtract'} a vector and a scalar "
                f"in expression around {node!r}"
            )
        return (left + right) if node.op == "+" else (left - right)

    if node.op == "*":
        if lv and rv:
            raise GeometryCompilerError(
                "vector * vector is not defined in the mini-language -- use dot(...) "
                "or cross(...) instead"
            )
        if lv:
            return left * right  # vector * scalar
        if rv:
            return right * left  # scalar * vector
        return left * right  # scalar * scalar

    if node.op == "/":
        if rv:
            raise GeometryCompilerError("cannot divide by a vector")
        if lv:
            return left / right  # vector / scalar (componentwise)
        return left / right  # scalar / scalar

    if node.op == "**":
        base = _require_scalar(left, "** base")
        exponent = _require_scalar(right, "** exponent")
        return base**exponent

    raise GeometryCompilerError(f"unsupported binary operator {node.op!r}")


def _eval_call(node: Call, env: Environment):
    name = node.name

    if name in ("A", "d"):
        if len(node.args) != 1 or not isinstance(node.args[0], Ident):
            raise GeometryCompilerError(
                f"{name}(...) expects exactly one bare line-id argument, got {node.args!r}"
            )
        line_id = node.args[0].name
        if line_id not in env.lines:
            raise UnknownIdentifierError(
                f"{line_id!r} is not a declared parametric_line entity id "
                f"(referenced via {name}({line_id}))"
            )
        anchor, direction = env.lines[line_id]
        return anchor if name == "A" else direction

    args = [evaluate(a, env) for a in node.args]

    if name == "dot":
        if len(args) != 2:
            raise GeometryCompilerError(f"dot(...) expects 2 arguments, got {len(args)}")
        u = _require_vector(args[0], "dot() arg 1")
        v = _require_vector(args[1], "dot() arg 2")
        return (u.T * v)[0, 0]

    if name == "cross":
        if len(args) != 2:
            raise GeometryCompilerError(f"cross(...) expects 2 arguments, got {len(args)}")
        u = _require_vector(args[0], "cross() arg 1")
        v = _require_vector(args[1], "cross() arg 2")
        return u.cross(v)

    if name == "norm":
        if len(args) != 1:
            raise GeometryCompilerError(f"norm(...) expects 1 argument, got {len(args)}")
        v = _require_vector(args[0], "norm() arg")
        radicand = (v.T * v)[0, 0]
        return exact_sqrt(radicand)

    raise UnknownIdentifierError(
        f"{name!r} is not a recognized mini-language function "
        f"(expected one of: dot, cross, norm, A, d)"
    )
