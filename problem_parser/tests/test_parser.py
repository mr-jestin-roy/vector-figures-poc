"""Unit tests for parser.py using a mocked Gemini client.

Zero network calls, no API key needed -- FakeGeminiClient (tests/fakes.py)
returns pre-baked JSON built from the golden fixtures. This is exactly
scenario (a) from the task brief: mocked-client tests for all three
problems, plus a deliberately-poisoned response (a solved coordinate on
an "unknown" entity) asserting the validator rejects it.
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from problem_parser.errors import GeminiResponseError, MissingAPIKeyError, ValidationError
from problem_parser.gemini_client import create_default_client
from problem_parser.parser import parse_problem
from problem_parser.schema_json import PROBLEM_IR_RESPONSE_SCHEMA
from problem_parser.tests.conftest import (
    ALL_PROBLEM_IDS,
    load_fixture_source_text,
    mock_gemini_response_for_fixture,
)
from problem_parser.tests.fakes import FakeGeminiClient

from shared.schema import ProblemIR


@pytest.mark.parametrize("problem_id", ALL_PROBLEM_IDS)
def test_parse_problem_matches_fixture_shape_via_mocked_client(problem_id):
    raw_text = load_fixture_source_text(problem_id)
    mock_response = mock_gemini_response_for_fixture(problem_id)
    fake_client = FakeGeminiClient(mock_response)

    ir = parse_problem(raw_text, problem_id, fake_client)

    assert isinstance(ir, ProblemIR)
    assert fake_client.call_count == 1
    # The schema handed to the (fake) client must be the real contract
    # schema -- i.e. parser.py isn't silently using something else.
    assert fake_client.last_json_schema == PROBLEM_IR_RESPONSE_SCHEMA

    # problem_id / source_text are stamped by the caller, never trusted
    # from the model's JSON (which doesn't even contain them).
    assert ir.problem_id == problem_id
    assert ir.source_text == raw_text.strip()

    expected_entity_ids = {e["id"] for e in mock_response["entities"]}
    assert {e.id for e in ir.entities} == expected_entity_ids
    assert ir.equations == mock_response["equations"]
    assert ir.query_expr == mock_response["query_expr"]
    assert ir.unknown_scalars == mock_response["unknown_scalars"]

    # And it must independently re-validate clean (parse_problem already
    # calls validate_problem_ir internally and would have raised if not,
    # but this documents the expectation for readers of the test).
    unknown_entities = [e for e in ir.entities if e.status == "unknown"]
    for e in unknown_entities:
        assert e.coordinates is None
        assert e.anchor is None
        assert e.direction is None
        assert e.components is None


def test_parse_problem_rejects_response_that_solves_an_unknown_point():
    """The core hard-rule regression test: if Gemini's response smuggles
    a solved coordinate onto an entity it itself marked "unknown", the
    pipeline must reject it -- not silently accept a "helpful" answer.
    """
    problem_id = "M26S1J21Q64"
    raw_text = load_fixture_source_text(problem_id)
    poisoned_response = copy.deepcopy(mock_gemini_response_for_fixture(problem_id))

    # Find the unknown foot-of-perpendicular entity "F" and sneak in the
    # actual solved answer from the LaTeX source (F = (1,6,0)), exactly
    # the kind of "shortcut" CONTRACT.md hard rule 4 forbids.
    f_entity = next(e for e in poisoned_response["entities"] if e["id"] == "F")
    assert f_entity["status"] == "unknown"
    f_entity["coordinates"] = {"x": "1", "y": "6", "z": "0"}

    fake_client = FakeGeminiClient(poisoned_response)

    with pytest.raises(ValidationError) as exc_info:
        parse_problem(raw_text, problem_id, fake_client)

    joined = " ".join(exc_info.value.issues)
    assert "F" in joined
    assert "unknown" in joined
    assert "coordinates" in joined


def test_parse_problem_rejects_response_that_solves_an_unknown_vector():
    """Same regression, but for a `vector`-kind unknown (components
    instead of coordinates) -- M26S1J21Q55's "c".
    """
    problem_id = "M26S1J21Q55"
    raw_text = load_fixture_source_text(problem_id)
    poisoned_response = copy.deepcopy(mock_gemini_response_for_fixture(problem_id))

    c_entity = next(e for e in poisoned_response["entities"] if e["id"] == "c")
    assert c_entity["status"] == "unknown"
    c_entity["components"] = {"x": "2", "y": "-1", "z": "3"}  # the real solved c

    fake_client = FakeGeminiClient(poisoned_response)

    with pytest.raises(ValidationError) as exc_info:
        parse_problem(raw_text, problem_id, fake_client)

    joined = " ".join(exc_info.value.issues)
    assert "c" in joined
    assert "components" in joined


def test_parse_problem_rejects_equation_with_undeclared_identifier():
    problem_id = "M26S1J21Q55"
    raw_text = load_fixture_source_text(problem_id)
    bad_response = copy.deepcopy(mock_gemini_response_for_fixture(problem_id))
    bad_response["equations"].append("cross(a, q) == b")  # 'q' undeclared

    fake_client = FakeGeminiClient(bad_response)
    with pytest.raises(ValidationError) as exc_info:
        parse_problem(raw_text, problem_id, fake_client)
    assert any("q" in issue for issue in exc_info.value.issues)


def test_parse_problem_raises_gemini_response_error_on_non_json():
    fake_client = FakeGeminiClient("this is not json { at all")
    with pytest.raises(GeminiResponseError):
        parse_problem("Given: ... Find: ...", "BOGUS", fake_client)


def test_parse_problem_raises_gemini_response_error_on_wrong_top_level_type():
    fake_client = FakeGeminiClient(json.dumps([1, 2, 3]))
    with pytest.raises(GeminiResponseError):
        parse_problem("Given: ... Find: ...", "BOGUS", fake_client)


def _make_tests_ignore_the_real_dotenv_file(monkeypatch):
    # These tests assert behavior for specific combinations of env vars
    # being *absent*. They must stay hermetic regardless of what a real
    # .env file at the project root happens to contain (e.g. a
    # developer's own GEMINI_JEE_LATEX_API_KEY) -- so short-circuit the
    # module's "load .env once per process" flag instead of relying on
    # the filesystem.
    import problem_parser.gemini_client as gemini_client_module

    monkeypatch.setattr(gemini_client_module, "_dotenv_loaded", True)
    for name in gemini_client_module.API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_missing_api_key_raises_actionable_error(monkeypatch):
    _make_tests_ignore_the_real_dotenv_file(monkeypatch)
    with pytest.raises(MissingAPIKeyError) as exc_info:
        create_default_client()
    message = str(exc_info.value)
    assert "GEMINI_API_KEY" in message
    assert "GOOGLE_API_KEY" in message


def test_api_key_from_google_api_key_fallback_is_accepted(monkeypatch):
    _make_tests_ignore_the_real_dotenv_file(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key-for-test")
    client = create_default_client()
    assert client is not None


def test_api_key_from_jee_latex_env_var_is_accepted(monkeypatch):
    _make_tests_ignore_the_real_dotenv_file(monkeypatch)
    monkeypatch.setenv("GEMINI_JEE_LATEX_API_KEY", "jee-latex-key-for-test")
    client = create_default_client()
    assert client is not None


def test_dotenv_file_supplies_key_when_no_real_env_var_is_set(monkeypatch, tmp_path):
    import problem_parser.gemini_client as gemini_client_module

    for name in gemini_client_module.API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gemini_client_module, "_dotenv_loaded", False)

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_JEE_LATEX_API_KEY=key-from-dotenv-file\n")
    # gemini_client.py resolves ".env" as two directories up from its own
    # file location (problem_parser/../.env); patch Path(__file__) via
    # the module's __file__ attribute so the loader looks in tmp_path
    # instead of touching the real project-root .env.
    fake_module_path = tmp_path / "problem_parser" / "gemini_client.py"
    fake_module_path.parent.mkdir()
    monkeypatch.setattr(gemini_client_module, "__file__", str(fake_module_path))

    client = create_default_client()
    assert client is not None
    assert os.environ.get("GEMINI_JEE_LATEX_API_KEY") == "key-from-dotenv-file"
