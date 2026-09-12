"""A fake GeminiClient for unit tests -- zero network calls, no key.

Satisfies the same `GeminiClient` protocol (structurally, via duck
typing / Protocol) that the real `GoogleGenAIClient` does, so
`parser.parse_problem` cannot tell the difference.
"""
from __future__ import annotations

import json
from typing import Union


class FakeGeminiClient:
    def __init__(self, response_data: Union[dict, str]):
        """`response_data`: either a dict (will be JSON-serialized on
        each call) or an already-serialized JSON string (e.g. to test
        malformed/non-JSON responses).
        """
        self._response_data = response_data
        self.call_count = 0
        self.last_system_instruction: str | None = None
        self.last_prompt: str | None = None
        self.last_json_schema: dict | None = None

    def generate_structured_json(
        self,
        *,
        system_instruction: str,
        prompt: str,
        json_schema: dict,
    ) -> str:
        self.call_count += 1
        self.last_system_instruction = system_instruction
        self.last_prompt = prompt
        self.last_json_schema = json_schema
        if isinstance(self._response_data, str):
            return self._response_data
        return json.dumps(self._response_data)
