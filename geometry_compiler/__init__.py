"""geometry_compiler: ProblemIR -> SceneGraph, the second half of the
pipeline described in CONTRACT.md.

    from geometry_compiler import compile
    scene_graph = compile(problem_ir)

`compile` is an alias for `compiler.compile_problem_ir`; it is exposed
under both names (`compile` for the semantics the pipeline contract
asks for, `compile_problem_ir` because shadowing the `compile` builtin
at module level is worth avoiding in the implementation itself).

See geometry_compiler/README.md for the approach, how generic vs.
special-cased the solver ended up being, and known limitations.
"""
from .compiler import compile_problem_ir
from .exceptions import (
    AmbiguousSystemError,
    GeometryCompilerError,
    IrrationalResultError,
    MiniLanguageSyntaxError,
    SystemInconsistentError,
    UnderdeterminedSystemError,
    UnknownIdentifierError,
)

compile = compile_problem_ir

__all__ = [
    "compile",
    "compile_problem_ir",
    "GeometryCompilerError",
    "MiniLanguageSyntaxError",
    "UnknownIdentifierError",
    "SystemInconsistentError",
    "UnderdeterminedSystemError",
    "AmbiguousSystemError",
    "IrrationalResultError",
]
