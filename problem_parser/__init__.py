"""problem_parser: LaTeX "Given/Find" text -> Gemini -> validated ProblemIR.

Half of a two-stage pipeline (see ../CONTRACT.md). This package's one
job is extraction -- it must never solve any geometry itself. See
validators.py for the code-level enforcement of that rule, and
README.md for usage, setup, and design notes.
"""
from __future__ import annotations

import os
import sys

# `shared/schema.py` lives one directory above this package (a sibling
# of problem_parser/, not a subpackage of it -- see CONTRACT.md). Make
# sure the repo root is importable as `shared.schema` regardless of the
# caller's cwd or how this package was invoked (script, pytest, REPL,
# ...). This runs once, the first time anything under `problem_parser`
# is imported, because Python always initializes a package's __init__
# before any of its submodules.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from problem_parser.errors import (  # noqa: E402
    GeminiResponseError,
    MissingAPIKeyError,
    ProblemParserError,
    ValidationError,
)
from problem_parser.gemini_client import (  # noqa: E402
    GeminiClient,
    GoogleGenAIClient,
    create_default_client,
)
from problem_parser.parser import parse_problem, parse_problem_live  # noqa: E402
from problem_parser.validators import validate_problem_ir  # noqa: E402

__all__ = [
    "parse_problem",
    "parse_problem_live",
    "validate_problem_ir",
    "GeminiClient",
    "GoogleGenAIClient",
    "create_default_client",
    "ProblemParserError",
    "MissingAPIKeyError",
    "GeminiResponseError",
    "ValidationError",
]
