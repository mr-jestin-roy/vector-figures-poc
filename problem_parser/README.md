# problem_parser

The extraction half of a two-stage pipeline for turning vector-geometry
word problems into a verified 3D scene graph (see
[`../CONTRACT.md`](../CONTRACT.md) for the full spec). This package
takes one problem's raw "Given/Find" text, calls the Gemini API to
extract it into a `ProblemIR` (defined in
[`../shared/schema.py`](../shared/schema.py)), and **never solves
anything** -- that is the `geometry_compiler` package's job, built
independently and in parallel, under `../geometry_compiler/`.

## Install

```bash
pip install -r requirements.txt
```

## Set your API key

Either export it as a real environment variable:

```bash
export GEMINI_API_KEY=your-key-here
```

or drop it in a `.env` file at the project root (one directory up from
this package) -- `create_default_client()` auto-loads it, no `export` or
`source` needed:

```
GEMINI_API_KEY=your-key-here
```

Checked, in order: `GEMINI_API_KEY`, then `GEMINI_JEE_LATEX_API_KEY`,
then `GOOGLE_API_KEY` (see `gemini_client.API_KEY_ENV_VARS`) -- each name
is checked as both a real env var and a `.env` entry. A real env var
always wins over a `.env` value of the same name. Get a key at
<https://aistudio.google.com/app/apikey>. If none of the three names are
set anywhere, `create_default_client()` / `parse_problem_live()` raise
`MissingAPIKeyError` immediately (before any network call) with the same
instructions.

`.env` loading requires `python-dotenv` (in `requirements.txt`); if it
isn't installed, `.env` is silently ignored and only real environment
variables work -- this package's own tests never depend on `.env` being
present. This has been verified against the real API: with a key set via
`GEMINI_JEE_LATEX_API_KEY` in `.env`, all 36 tests pass, including the
one live call against the held-out fixture (see "Held-out fixture"
below).

## Usage

```python
from problem_parser import parse_problem_live

raw_text = (
    "Given: L1 through (2,6,7) parallel to (-3,2,4); "
    "L2 through (4,3,5) parallel to (2,1,3); "
    "L3 parallel to (-3,5,16) meets L1 at C, L2 at D. "
    "Find: |CD|^2"
)
ir = parse_problem_live(raw_text, problem_id="M26S2J21Q3")
# ir is a validated shared.schema.ProblemIR, ready to hand to
# geometry_compiler.
```

For testing, or to reuse one client across many calls, inject a client
explicitly instead:

```python
from problem_parser import create_default_client, parse_problem

client = create_default_client()          # reads GEMINI_API_KEY/GOOGLE_API_KEY
ir = parse_problem(raw_text, "M26S2J21Q3", client)
```

`parse_problem` never imports `google.genai` or reads an environment
variable itself -- it only calls `client.generate_structured_json(...)`.
Any object with that one method (see `problem_parser/tests/fakes.py`)
can stand in for the real client, which is how the unit test suite runs
with zero network calls and no key.

## Package layout

| File | Purpose |
|---|---|
| `errors.py` | `MissingAPIKeyError`, `GeminiResponseError`, `ValidationError` (carries every issue found, not just the first). |
| `schema_json.py` | Builds the JSON Schema handed to Gemini's structured-output mode (`response_json_schema`). Mirrors `shared/schema.py`'s dataclasses field-for-field. |
| `prompts.py` | System instruction (grammar, ROLE_STYLE, the never-solve rule) + two worked few-shot examples. |
| `gemini_client.py` | The injectable `GeminiClient` Protocol, the real `GoogleGenAIClient` implementation, and `create_default_client()` (env var resolution). |
| `validators.py` | Pure functions enforcing the contract's hard rules on an already-constructed `ProblemIR`. This is where "never solve" is actually *enforced*, not just requested in a prompt. |
| `parser.py` | `parse_problem(raw_text, problem_id, client)` orchestrates: build prompt -> call client -> parse JSON -> stamp `problem_id`/`source_text` -> construct `ProblemIR` -> validate -> return. |
| `tests/` | See below. |

## Why `google-genai`, not `google-generativeai`

Google currently ships two Python packages for Gemini. This project
uses **`google-genai`** (imports as `google.genai`), Google's current
unified SDK, because:

- It is what's actually `pip install`-able and importable in this
  environment (verified: `google-genai==2.22.0` on Python 3.13).
- It exposes `GenerateContentConfig.response_json_schema`, which
  accepts a **standard JSON Schema** dict directly (`$defs`, `$ref`,
  `anyOf`, `required`, etc.) -- a much better fit for `ProblemIR`'s
  shape (optional-but-nullable geometry fields, enum-constrained
  `Literal` types) than hand-rolling Gemini's older, more restrictive
  `Schema` proto (`response_schema`) would have been. `schema_json.py`
  documents exactly which JSON Schema keywords Gemini's
  `response_json_schema` supports and builds the schema within that
  subset.
- The older `google-generativeai` package was not needed as a fallback
  -- `google-genai` installed and worked on the first try.

The real Gemini client defaults to `gemini-3.6-flash` (`gemini-2.5-flash`
was the original default but returned a 404 "no longer available to new
users" from the live API and was replaced) -- override via the
`GEMINI_MODEL` environment variable or the `model=` argument to
`create_default_client`/`parse_problem_live`).

`google.genai` is imported lazily, inside the functions/methods that
actually need it (`GoogleGenAIClient.__init__` and
`.generate_structured_json`), not at module import time -- so importing
`problem_parser` itself, and running the mocked unit tests, never
requires the SDK to even be installed. Only constructing a *real*
client does.

## Held-out fixture (for honest end-to-end testing)

Three golden fixtures exist in `../fixtures/problem_ir/`:
`M26S1J21Q55`, `M26S1J21Q64`, `M26S2J21Q3`.

**`M26S2J21Q3` ("two lines through given points, forced parallel to a
third given direction") is held out of `prompts.py`'s few-shot examples
entirely.** Only `M26S1J21Q55` (cross-product + plane-dot system) and
`M26S1J21Q64` (foot-of-perpendicular + projection) are embedded as
worked examples in the system instruction. `tests/test_integration_live.py`
targets exclusively the held-out `M26S2J21Q3` -- it is the one and only
live-Gemini test, and it is an honest test of generalization precisely
*because* the model has never seen that problem's shape in its prompt.
Few-shotting with all three and then "testing" against all three would
have proven nothing.

(The **mocked** unit tests, by contrast, legitimately use canned
responses built from all three fixtures, including the held-out one --
that's testing the parse+validate *pipeline* against a hypothetical
Gemini response, not testing what Gemini itself would generalize to.
Those are different claims, and the task brief's circularity warning is
about the latter, not the former.)

## Design decisions worth calling out

- **`problem_id` and `source_text` are never trusted from the model.**
  The caller already knows which LaTeX section it's feeding in, so
  `parse_problem(raw_text, problem_id, client)` stamps both fields onto
  the result itself (`problem_id` verbatim, `source_text` as
  `raw_text.strip()`) rather than asking Gemini to echo them back.
  `problem_id` is an exact identifier, not something to "extract", and
  `source_text` should be a mechanical copy of the input, never a
  paraphrase. The JSON schema sent to Gemini (`schema_json.py`)
  excludes both fields entirely.
- **The parser only ever emits `status: "given"` or `status: "unknown"`,
  never `"computed"`.** `"computed"` describes the geometry_compiler's
  *output* state (`shared/schema.py` rule 4) -- by the same "the parser
  never solves" logic as CONTRACT.md hard rule 1, the parser itself
  should never claim something is already solved. CONTRACT.md doesn't
  state this in so many words, so this is a judgment call, but it's
  consistent with all three golden fixtures (none uses `"computed"` in
  a `ProblemIR`) and is now enforced two ways: the JSON schema
  restricts `status` to `["given", "unknown"]` for the parser's output,
  and `validators.validate_parser_never_emits_computed_status` is a
  second, independent check.
- **Structured output is a strong prior, not the enforcement
  mechanism.** Gemini's `response_json_schema` supports a restricted
  JSON Schema subset (documented in `schema_json.py`'s module
  docstring) that has no clean way to express "field X must be null
  when sibling field Y equals a specific value." So the schema makes
  every geometry field nullable-but-required (forcing the model to be
  explicit about "this doesn't apply here" rather than silently
  omitting a key), and `validators.py` is the actual, independent,
  code-level enforcement of the "never solve" rule -- exactly as
  CONTRACT.md hard rule 4 asks for ("that is a bug in the parser's
  prompt or its post-processing validator, not an acceptable
  shortcut").

## Ambiguities / judgment calls found in CONTRACT.md and shared/schema.py

Per the task brief, these are named explicitly rather than silently
resolved:

1. **CONTRACT.md's grammar section says the reserved constants `zero`
   and `0` are "the only literals allowed", but the golden fixture
   `fixtures/problem_ir/M26S1J21Q55.json` has the equation
   `"dot(c, n) == 4"` -- a bare scalar literal `4` that is neither
   `zero` nor `0`. Read literally, the fixture violates the contract's
   own prose.

   Resolution used here: `shared/schema.py`'s design-rule comment #2
   scopes the "no bare literals" concern to *vectors/points*
   ("no bare numeric-literal tuples smuggling in problem data"), and
   the mini-language has no syntax for a vector literal at all (no
   `(x,y,z)` tuple form is defined anywhere in the grammar). So
   `validators.py` treats any bare **scalar** numeric token (matched by
   a number regex, not the identifier regex) as always allowed,
   regardless of whether it's `0` or something else, and only polices
   *identifier* tokens (named references) against the declared-id /
   `unknown_scalars` / `zero` allowlist. This interpretation is what
   lets `fixtures/problem_ir/M26S1J21Q55.json` pass
   `validate_problem_ir` at all (see
   `tests/test_validators.py::test_golden_fixture_passes_validation`
   and
   `tests/test_validators.py::test_reserved_zero_and_bare_numeric_literals_are_allowed`).
   If the intended rule really is "no scalar literal besides 0", that
   fixture needs to change, not this package.

2. **The `**` (power) operator is documented ("for query_expr like
   squared length") but never actually used in any of the three golden
   fixtures** -- all three express squared length as `dot(v, v)`
   instead (e.g. `query_expr: "dot(a + c, a + c)"` rather than something
   like `"norm(a + c) ** 2"`). Since no worked example demonstrates it,
   `prompts.py`'s few-shot examples also never use `**`, and there's no
   fixture-backed evidence for exactly when a parser should reach for
   `**` instead of `dot(v, v)`. The grammar and the validator both still
   accept `**` (it's a defined token; `validators.py`'s tokenizer treats
   it as an operator, not an identifier, and is agnostic which
   arithmetic operators actually appear), but nothing here can claim to
   have tested that path against a real, contract-endorsed example.

3. **Whether `A(...)`/`d(...)`'s argument must specifically be a
   `parametric_line`-kind entity is implied, not stated.** CONTRACT.md
   says "`A(line_id)`, `d(line_id)` -- anchor point / direction of a
   parametric line", which strongly implies the argument must be a
   `parametric_line` entity, but never says what should happen if it
   isn't. This package treats that as a hard validation error
   (`validators._validate_expression_identifiers`, tested in
   `test_A_call_on_non_line_entity_is_rejected`) rather than a soft
   warning, on the theory that `A(some_vector)` is unambiguously
   meaningless and should fail loudly rather than be silently passed
   through to `geometry_compiler`.

4. **`Entity.param_name` isn't mentioned by CONTRACT.md hard rule 4's
   list of fields that must stay null for an `unknown` entity**
   (`coordinates`/`anchor`/`direction`/`components`). This package reads
   that omission as intentional: `param_name` is just the *name* of a
   line's parameter (e.g. `"lambda"`), not a solved value, so an
   `unknown`-status `parametric_line` entity may still declare it (see
   `validators.py`'s `GEOMETRY_FIELDS` tuple, which deliberately
   excludes `param_name`, and
   `tests/test_validators.py::test_unknown_entity_may_keep_param_name`).
   None of the three fixtures actually exercise an `unknown`
   `parametric_line` entity, so this is untested against golden data --
   flagging it here rather than asserting confidence it doesn't matter.

## Tests

```bash
pytest problem_parser/tests/ -v
```

- **`tests/test_validators.py`** -- pure unit tests of every validator,
  including feeding it all three real golden fixtures (to check for
  false positives) and hand-built `ProblemIR`s that violate each rule
  (to check for false negatives). No Gemini, no network, no key.
- **`tests/test_parser.py`** -- unit tests of `parse_problem` against a
  `FakeGeminiClient` (`tests/fakes.py`) returning canned JSON built from
  each of the three fixtures (task requirement (a)), plus two
  deliberately-poisoned responses -- one with a solved `coordinates` on
  an `unknown` point (`F` in `M26S1J21Q64`), one with a solved
  `components` on an `unknown` vector (`c` in `M26S1J21Q55`) -- both
  asserted to raise `ValidationError`. Also covers the missing-API-key
  path and malformed/non-JSON Gemini responses. No network, no key.
- **`tests/test_integration_live.py`** -- task requirement (b): one real
  Gemini API call against the held-out fixture `M26S2J21Q3`.
  Auto-skips (via `pytest.mark.skipif`) whenever neither
  `GEMINI_API_KEY` nor `GOOGLE_API_KEY` is set, so the suite is green
  with zero configuration.

**Results in this environment (no API key set):** 33 passed, 1 skipped
(the live test, skipped as designed). The live test itself has not been
run against the real API in this environment because no key is
available here -- set `GEMINI_API_KEY` and re-run to exercise it.
