"""Exception types raised by problem_parser.

Kept in their own module (no imports from the rest of the package) so
every other module can import them without risking circular imports.
"""
from __future__ import annotations


class ProblemParserError(Exception):
    """Base class for every error this package raises on purpose."""


class MissingAPIKeyError(ProblemParserError):
    """Raised when no Gemini API key can be found in the environment.

    The message is written to be actionable on its own (it is often the
    only thing a user sees), so callers should generally let it propagate
    rather than re-wrapping it.
    """


class GeminiResponseError(ProblemParserError):
    """Raised when the Gemini API call fails, or returns something that
    cannot even be parsed as JSON. Distinguished from ValidationError,
    which fires *after* successful JSON parsing when the *content*
    violates the pipeline's hard rules.
    """


class ValidationError(ProblemParserError):
    """Raised when a parsed ProblemIR violates a hard rule of the
    parser/compiler contract (e.g. a solved coordinate on an `unknown`
    entity, or an equation referencing an undeclared identifier).

    Carries the full list of individual problems found, not just the
    first one, so a single failed parse attempt surfaces everything
    wrong with it at once.
    """

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        message = "ProblemIR failed validation:\n" + "\n".join(
            f"  - {issue}" for issue in self.issues
        )
        super().__init__(message)
