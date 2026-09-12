"""Shared test helpers: loading the golden fixtures under fixtures/.

We only ever READ fixtures/problem_ir/*.json here -- never write to
them. Reading them is exactly what CONTRACT.md expects ("both agents
read this file and shared/schema.py before writing code"); the golden
fixtures are the test targets problem_parser is graded against.
"""
from __future__ import annotations

import copy
import json
import os

# tests/ -> problem_parser/ -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES_DIR = os.path.join(REPO_ROOT, "fixtures", "problem_ir")

ALL_PROBLEM_IDS = ["M26S1J21Q55", "M26S1J21Q64", "M26S2J21Q3"]

# The fixture held out of prompts.py's few-shot examples -- see
# prompts.py's module docstring and README.md. Only this one is used as
# the live end-to-end test target.
HELD_OUT_PROBLEM_ID = "M26S2J21Q3"


def load_fixture_dict(problem_id: str) -> dict:
    """Raw JSON dict straight off disk, untouched."""
    path = os.path.join(FIXTURES_DIR, f"{problem_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fixture_source_text(problem_id: str) -> str:
    """The exact input text a caller would hand to parse_problem()."""
    return load_fixture_dict(problem_id)["source_text"]


def mock_gemini_response_for_fixture(problem_id: str) -> dict:
    """Builds the canned JSON a mocked Gemini call would return for this
    problem: the golden fixture with `problem_id` and `source_text`
    stripped out, since the real pipeline supplies those two fields
    itself rather than trusting them from the model (see parser.py).
    """
    data = copy.deepcopy(load_fixture_dict(problem_id))
    data.pop("problem_id", None)
    data.pop("source_text", None)
    return data
