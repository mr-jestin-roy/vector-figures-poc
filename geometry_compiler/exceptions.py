"""Exception types raised by geometry_compiler.

These are used to "clearly surface" problems rather than crashing with a
bare traceback or (worse) silently returning a wrong-but-confident answer.
See CONTRACT.md hard rule #1 ("verify algebraically, never eyeball") --
the same spirit applies to the compiler's own failure modes.
"""
from __future__ import annotations


class GeometryCompilerError(Exception):
    """Base class for all errors raised by geometry_compiler."""


class MiniLanguageSyntaxError(GeometryCompilerError):
    """The equations/query_expr mini-language string could not be parsed."""


class UnknownIdentifierError(GeometryCompilerError):
    """An identifier in a mini-language expression is not a declared
    Entity, a declared unknown_scalar, or a reserved constant (zero, 0)."""


class SystemInconsistentError(GeometryCompilerError):
    """The equation system built from ProblemIR.equations has NO exact
    solution -- i.e. the problem's own equations contradict each other.
    This is a distinct failure mode from `verified=False`: there is no
    numeric assignment at all to populate the SceneGraph with, so we
    raise instead of fabricating placeholder coordinates."""


class UnderdeterminedSystemError(GeometryCompilerError):
    """The equation system does not pin every unknown down to a single
    value (free parameters remain). geometry_compiler expects fully
    determined problems, matching the three fixture families."""


class AmbiguousSystemError(GeometryCompilerError):
    """The solver found more than one discrete solution branch and had
    no principled way to choose between them."""


class IrrationalResultError(GeometryCompilerError):
    """A value that must be reported as an exact Rational (e.g.
    final_answer) turned out to be genuinely irrational (a sqrt that is
    not a perfect square). Known current limitation -- see README."""
