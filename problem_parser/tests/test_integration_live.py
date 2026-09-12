"""One real, live Gemini API call against the HELD-OUT fixture.

Held out means: M26S2J21Q3 never appears in prompts.py's few-shot
examples (only M26S1J21Q55 and M26S1J21Q64 do). Testing the parser
end-to-end against a problem it has never been shown, using the real
API, is the only honest test of whether the extraction actually
generalizes -- testing against a few-shotted example would just be
checking whether the model can echo back something it was already
handed.

Automatically skipped when none of gemini_client.API_KEY_ENV_VARS
(GEMINI_API_KEY, GEMINI_JEE_LATEX_API_KEY, GOOGLE_API_KEY) is set -- via
a real environment variable or a project-root .env file -- so the suite
stays green with zero configuration and exercises the real API the
moment a key is present either way.

Assertions here are deliberately loose on exact wording/ids (an LLM's
phrasing of `problem_type`/`target`/entity ids can reasonably vary run
to run) but strict on the two hard rules: no solved geometry on any
entity the model marks "unknown", and every equation/query_expr
identifier resolves to something declared.
"""
from __future__ import annotations

import os

import pytest

from problem_parser.gemini_client import API_KEY_ENV_VARS, _load_dotenv_once
from problem_parser.parser import parse_problem_live
from problem_parser.tests.conftest import HELD_OUT_PROBLEM_ID, load_fixture_source_text
from problem_parser.validators import validate_problem_ir

# Load .env (if present) BEFORE evaluating the skip condition below --
# otherwise a key that only lives in .env (not a real exported env var)
# would cause an incorrect skip, since pytest evaluates `skipif` at
# collection time, before create_default_client() would normally trigger
# the .env load itself.
_load_dotenv_once()

pytestmark = pytest.mark.skipif(
    not any(os.environ.get(name) for name in API_KEY_ENV_VARS),
    reason=(
        f"None of {', '.join(API_KEY_ENV_VARS)} is set (checked both real "
        "env vars and a project-root .env file) -- skipping the live "
        "Gemini integration test. Set one to actually exercise the real API."
    ),
)


def test_live_parse_of_held_out_fixture():
    raw_text = load_fixture_source_text(HELD_OUT_PROBLEM_ID)

    ir = parse_problem_live(raw_text, HELD_OUT_PROBLEM_ID)

    # parse_problem_live already calls validate_problem_ir internally
    # and would have raised if it failed; re-running it here documents
    # the expectation and guards against a future refactor silently
    # dropping that call.
    validate_problem_ir(ir)

    assert ir.problem_id == HELD_OUT_PROBLEM_ID
    assert ir.source_text == raw_text.strip()
    assert ir.equations, "expected at least one equation"
    assert ir.query_expr.strip()
    assert ir.target.strip()

    # The problem describes two given lines (L1, L2), a given third
    # direction, and two unknown intersection points (C, D) -- the
    # extraction should reflect that shape, even if ids/wording differ
    # from the fixture's own choices.
    line_entities = [e for e in ir.entities if e.kind == "parametric_line"]
    unknown_entities = [e for e in ir.entities if e.status == "unknown"]
    assert len(line_entities) >= 2, "expected at least L1 and L2 as parametric_line entities"
    assert len(unknown_entities) >= 2, "expected at least C and D as unknown points"

    for e in unknown_entities:
        assert e.coordinates is None
        assert e.anchor is None
        assert e.direction is None
        assert e.components is None
