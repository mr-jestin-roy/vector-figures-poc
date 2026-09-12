"""Thin, dependency-injectable wrapper around the Gemini API.

`parser.py` never imports `google.genai` directly and never reads
environment variables directly -- it only depends on the `GeminiClient`
Protocol below. That means:

  * Unit tests can pass in any object with a matching
    `generate_structured_json` method (see tests/fakes.py) and never
    touch the network or need an API key.
  * The real `google.genai` package is imported lazily, inside the
    functions/methods that actually need it, so importing this module
    (or the rest of the package) never fails just because the SDK
    isn't installed -- only constructing a *real* client does.

SDK choice: `google-genai` (imports as `google.genai`), Google's current
unified Python SDK for the Gemini API, is what's actually pip-installable
in this environment (`pip install google-genai`, verified working with
python 3.13). This is the SDK that supports the `response_json_schema`
config field used by prompts.py's schema-constrained structured output;
the older `google-generativeai` package is not used. See README.md for
more detail on this choice.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from problem_parser.errors import GeminiResponseError, MissingAPIKeyError

DEFAULT_MODEL = "gemini-3.6-flash"

# Names checked, in this order, for the Gemini API key -- both as real
# environment variables and as keys in a project-root .env file. Kept as
# a tuple (not hardcoded inline) so create_default_client's docstring and
# the actual lookup can never drift apart.
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GEMINI_JEE_LATEX_API_KEY", "GOOGLE_API_KEY")

_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """Loads a `.env` file from the project root into os.environ, once
    per process. Never overrides a variable that's already set in the
    real environment -- .env is a convenience default, not an override.

    Resolved relative to this file's location (problem_parser/../.env),
    not the current working directory, so this works the same whether
    tests are run from the project root or from inside problem_parser/.

    If python-dotenv isn't installed, this is a silent no-op: real
    environment variables (`export GEMINI_API_KEY=...`) still work
    exactly as before -- .env support is additive, not a new requirement.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)


@runtime_checkable
class GeminiClient(Protocol):
    """The only surface parser.py depends on. Any object satisfying this
    (real or fake) can be injected into `parser.parse_problem`.
    """

    def generate_structured_json(
        self,
        *,
        system_instruction: str,
        prompt: str,
        json_schema: dict,
    ) -> str:
        """Returns the raw JSON text of the model's response, constrained
        to `json_schema` via the provider's structured-output mode.
        Implementations should raise GeminiResponseError on any
        transport/API failure so callers see one exception type.
        """
        ...


class GoogleGenAIClient:
    """Real GeminiClient backed by google.genai.Client.

    Constructed by `create_default_client`; you would not normally
    build this directly in a test.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - exercised only
            # when the optional dependency truly isn't installed.
            raise GeminiResponseError(
                "The 'google-genai' package is required to talk to the "
                "real Gemini API. Install it with: pip install google-genai"
            ) from exc

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate_structured_json(
        self,
        *,
        system_instruction: str,
        prompt: str,
        json_schema: dict,
    ) -> str:
        from google.genai import types
        from google.genai import errors as genai_errors

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=json_schema,
            temperature=0,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except genai_errors.APIError as exc:
            raise GeminiResponseError(f"Gemini API call failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise GeminiResponseError(
                "Gemini returned an empty response (no candidate text). "
                f"Full response: {response!r}"
            )
        return text


def create_default_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> GoogleGenAIClient:
    """Builds the real Gemini client, reading the API key from the
    environment (or a project-root `.env` file) if not passed explicitly.

    Resolution order: explicit `api_key` argument, then each name in
    `API_KEY_ENV_VARS` in turn (`GEMINI_API_KEY`, then
    `GEMINI_JEE_LATEX_API_KEY`, then `GOOGLE_API_KEY` as a last-resort
    fallback). Before checking any of them, a `.env` file at the project
    root is loaded into os.environ if python-dotenv is installed and the
    file exists -- so a `.env` with `GEMINI_JEE_LATEX_API_KEY=...` works
    with no shell `export` needed. Real environment variables always win
    over `.env` values. Raises MissingAPIKeyError with an actionable
    message if none of those are set -- this is expected to happen in any
    environment that hasn't configured Gemini yet, so the message is
    written to be useful on its own.
    """
    _load_dotenv_once()
    resolved_key = api_key
    if not resolved_key:
        for name in API_KEY_ENV_VARS:
            resolved_key = os.environ.get(name)
            if resolved_key:
                break
    if not resolved_key:
        checked = ", ".join(API_KEY_ENV_VARS)
        raise MissingAPIKeyError(
            "No Gemini API key found. problem_parser needs one to call "
            "the real Gemini API.\n"
            f"Fix: set one of these environment variables ({checked}), "
            "either with `export` or in a `.env` file at the project root, "
            "to a key from https://aistudio.google.com/app/apikey, e.g.:\n"
            "    export GEMINI_API_KEY=your-key-here\n"
            "or in .env:\n"
            "    GEMINI_API_KEY=your-key-here\n"
            "Until then, only the mocked unit tests (tests/test_parser.py, "
            "tests/test_validators.py) can run -- the live integration test "
            "(tests/test_integration_live.py) will skip itself automatically."
        )
    resolved_model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    return GoogleGenAIClient(api_key=resolved_key, model=resolved_model)
