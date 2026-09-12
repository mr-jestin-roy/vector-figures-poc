"""Orchestrates one Given/Find text -> validated ProblemIR extraction.

This is the only module most callers need. `parse_problem` takes an
injected `GeminiClient` (see gemini_client.py) so it never needs a real
API key or network access to be unit tested; `parse_problem_live` is a
convenience wrapper that builds the real client for you.
"""
from __future__ import annotations

import json
from typing import Optional

from problem_parser.errors import GeminiResponseError
from problem_parser.gemini_client import GeminiClient, create_default_client
from problem_parser.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from problem_parser.schema_json import PROBLEM_IR_RESPONSE_SCHEMA
from problem_parser.validators import validate_problem_ir

from shared.schema import ProblemIR, problem_ir_from_dict


def parse_problem(
    raw_text: str,
    problem_id: str,
    client: GeminiClient,
) -> ProblemIR:
    """Extracts one problem's raw "Given/Find" text into a validated
    ProblemIR, using `client` to talk to Gemini (or a fake, in tests).

    `problem_id` and the resulting `ProblemIR.source_text` are NOT
    trusted from the model's response -- `source_text` is stamped as a
    verbatim (whitespace-trimmed) copy of `raw_text`, and `problem_id`
    is stamped as exactly what the caller passed in. Everything else
    (`entities`, `relations`, `equations`, ...) comes from Gemini's
    structured-output response, then run through `validate_problem_ir`,
    which raises `ValidationError` (see errors.py) listing every
    contract violation found -- including, critically, any solved
    geometry on an entity the model itself marked "unknown".

    Raises:
        GeminiResponseError: the API call failed, or the response
            wasn't valid JSON / didn't match the expected top-level
            shape enough to even construct a ProblemIR.
        problem_parser.errors.ValidationError: the constructed ProblemIR
            violates a hard rule (unsolved-unknown, bad equation
            identifier, duplicate id, etc).
    """
    prompt = build_user_prompt(raw_text, problem_id)
    raw_response = client.generate_structured_json(
        system_instruction=SYSTEM_INSTRUCTION,
        prompt=prompt,
        json_schema=PROBLEM_IR_RESPONSE_SCHEMA,
    )

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise GeminiResponseError(
            f"Gemini response was not valid JSON: {exc}\nRaw response:\n{raw_response}"
        ) from exc

    if not isinstance(data, dict):
        raise GeminiResponseError(
            f"Expected a JSON object from Gemini, got {type(data).__name__}: {data!r}"
        )

    # problem_id / source_text are never trusted from the model -- see
    # docstring above and schema_json.py's module docstring.
    data["problem_id"] = problem_id
    data["source_text"] = raw_text.strip()

    try:
        ir = problem_ir_from_dict(data)
    except (TypeError, KeyError, ValueError) as exc:
        raise GeminiResponseError(
            f"Gemini response did not match the ProblemIR shape: {exc}\n"
            f"Parsed JSON:\n{json.dumps(data, indent=2)}"
        ) from exc

    validate_problem_ir(ir)
    return ir


def parse_problem_live(
    raw_text: str,
    problem_id: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ProblemIR:
    """Convenience wrapper: builds the real Gemini client (reading the
    API key from GEMINI_API_KEY / GOOGLE_API_KEY unless `api_key` is
    given) and calls `parse_problem`.

    Raises MissingAPIKeyError immediately, before any network call, if
    no key is configured -- see gemini_client.create_default_client.
    """
    client = create_default_client(api_key=api_key, model=model)
    return parse_problem(raw_text, problem_id, client)
