"""Builds the JSON Schema handed to Gemini's structured-output mode.

This schema constrains the *shape* Gemini is allowed to emit -- it cannot
by itself enforce cross-field semantic rules like "an `unknown` entity may
not have populated geometry" (JSON Schema has no clean way to say "field X
must be null when sibling field Y equals SOME_VALUE" within the restricted
subset Gemini's `response_json_schema` supports: `$id`, `$defs`, `$ref`,
`$anchor`, `type`, `format`, `title`, `description`, `enum`, `items`,
`prefixItems`, `minItems`, `maxItems`, `minimum`, `maximum`, `anyOf`,
`oneOf`, `properties`, `additionalProperties`, `required`, and the
non-standard `propertyOrdering`). That is exactly why `validators.py`
exists as a second, independent enforcement layer -- see CONTRACT.md's
hard rule 4 ("the parser never solves") and shared/schema.py's docstring
rule 3. Structured output here is a *strong prior*, not the enforcement
mechanism.

Field mirroring: every property name and enum value below is copied
verbatim from shared/schema.py (EntityKind, EntityStatus, VisualRole,
AnchorDir, Relation.kind) so a conformant response round-trips through
`shared.schema.problem_ir_from_dict` without translation.

Deliberately NOT part of the requested shape: `problem_id` and
`source_text`. Both are supplied by the caller (the caller already knows
which LaTeX section it is feeding in) and are stamped onto the model's
JSON after the fact in `parser.py`, rather than trusted from the LLM --
`problem_id` is an exact identifier, not something to "extract", and
`source_text` should be a mechanical copy of the input, not a
paraphrase.
"""
from __future__ import annotations

ENTITY_KIND = ["point", "parametric_line", "vector"]
ENTITY_STATUS = ["given", "unknown", "computed"]  # full range from shared/schema.py

# problem_parser's own output is restricted to a strict subset of
# EntityStatus: "computed" is the geometry_compiler's output state
# (shared/schema.py rule 4) -- by the same "the parser never solves"
# logic as CONTRACT.md hard rule 1, the parser itself should never
# claim status "computed" for anything it emits. None of the three
# golden fixtures use "computed" in a ProblemIR. This is a judgment
# call (CONTRACT.md doesn't spell it out) -- see README.md and
# validators.validate_parser_never_emits_computed_status, which is the
# actual enforcement point; this schema restriction is just a second,
# earlier line of defense at the structured-output layer.
PARSER_ENTITY_STATUS = ["given", "unknown"]
VISUAL_ROLE = [
    "given_external_point",
    "given_line",
    "given_vector",
    "computed_point",
    "computed_line",
    "computed_vector",
]
ANCHOR_DIR = [
    "north", "south", "east", "west",
    "north east", "north west", "south east", "south west",
]
RELATION_KIND = ["lies_on", "perpendicular_to", "parallel_to", "projects_onto"]

_RATIONAL_STRING = {
    "type": "string",
    "description": (
        "An EXACT rational number written as a string parseable by "
        "Python's fractions.Fraction, e.g. \"7/2\", \"-1\", \"4\", \"18/7\". "
        "Never a float/decimal like \"3.5\"."
    ),
}

_VEC3 = {
    "type": "object",
    "description": "A 3-vector of exact rational components.",
    "properties": {
        "x": _RATIONAL_STRING,
        "y": _RATIONAL_STRING,
        "z": _RATIONAL_STRING,
    },
    "required": ["x", "y", "z"],
    "additionalProperties": False,
}

_NULLABLE_VEC3 = {"anyOf": [_VEC3, {"type": "null"}]}

_LABEL = {
    "type": "object",
    "description": "The on-figure text label for an entity and its anchor.",
    "properties": {
        "text": {"type": "string"},
        "anchor": {
            "type": "string",
            "enum": ANCHOR_DIR,
            "description": (
                "One of the eight compass directions -- never a bare "
                "xshift/yshift offset."
            ),
        },
    },
    "required": ["text", "anchor"],
    "additionalProperties": False,
}

_ENTITY = {
    "type": "object",
    "description": (
        "One geometric object (point, parametric_line, or vector) that "
        "appears in the problem. If status == \"unknown\", coordinates, "
        "anchor, direction, and components MUST ALL be null -- do not "
        "solve for them."
    ),
    "properties": {
        "id": {
            "type": "string",
            "description": (
                "Short, stable identifier referenced by `equations` and "
                "`query_expr`, e.g. \"P\", \"F\", \"L\", \"c\"."
            ),
        },
        "kind": {"type": "string", "enum": ENTITY_KIND},
        "label": _LABEL,
        "status": {
            "type": "string",
            "enum": PARSER_ENTITY_STATUS,
            "description": (
                "Only \"given\" or \"unknown\" -- never \"computed\". "
                "\"computed\" describes the geometry_compiler's output "
                "state, not something problem_parser ever claims."
            ),
        },
        "visual_role": {"type": "string", "enum": VISUAL_ROLE},
        "coordinates": {
            **_NULLABLE_VEC3,
            "description": (
                "Only for kind == \"point\" AND status != \"unknown\". "
                "Null otherwise."
            ),
        },
        "anchor": {
            **_NULLABLE_VEC3,
            "description": (
                "Only for kind == \"parametric_line\" AND status != "
                "\"unknown\" (the line's anchor point A). Null otherwise."
            ),
        },
        "direction": {
            **_NULLABLE_VEC3,
            "description": (
                "Only for kind == \"parametric_line\" AND status != "
                "\"unknown\" (the line's direction vector d). Null "
                "otherwise."
            ),
        },
        "param_name": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": (
                "Only for kind == \"parametric_line\", e.g. \"lambda\", "
                "\"t\", \"s\". This is just the parameter's NAME, not a "
                "solved value, so it may be present even when status == "
                "\"unknown\". Null for point/vector kinds."
            ),
        },
        "components": {
            **_NULLABLE_VEC3,
            "description": (
                "Only for kind == \"vector\" AND status != \"unknown\". "
                "Null otherwise."
            ),
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Free-text, human/render-facing notes only (e.g. \"foot of "
                "the perpendicular from P\"). NEVER parsed/solved -- do not "
                "put algebra here. Use [] if there is nothing to note."
            ),
        },
    },
    "required": [
        "id", "kind", "label", "status", "visual_role",
        "coordinates", "anchor", "direction", "param_name", "components",
        "notes",
    ],
    "additionalProperties": False,
}

_RELATION = {
    "type": "object",
    "description": (
        "A cosmetic/semantic hint for a renderer. NEVER solved directly -- "
        "the real algebra lives in `equations`."
    ),
    "properties": {
        "kind": {"type": "string", "enum": RELATION_KIND},
        "subject": {"type": "string", "description": "An entity id."},
        "object": {"type": "string", "description": "An entity id."},
        "via": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": (
                "Optional third entity id, e.g. perpendicular_to(F, L) "
                "via P means segment(P,F) is perpendicular to direction(L)."
            ),
        },
    },
    "required": ["kind", "subject", "object", "via"],
    "additionalProperties": False,
}

# Top-level shape requested from Gemini. `problem_id` and `source_text`
# are intentionally excluded -- see module docstring.
PROBLEM_IR_RESPONSE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ProblemIRExtraction",
    "type": "object",
    "properties": {
        "problem_type": {
            "type": "string",
            "description": (
                "Short snake_case description of the constraint pattern, "
                "e.g. \"vector_cross_and_dot_system\", "
                "\"point_and_line_foot_of_perpendicular\", "
                "\"two_lines_with_third_direction_constraint\". "
                "Informational only -- never parsed by the compiler."
            ),
        },
        "entities": {"type": "array", "items": _ENTITY, "minItems": 1},
        "relations": {"type": "array", "items": _RELATION},
        "equations": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The authoritative algebra, each entry `LHS == RHS`, "
                "written in the equations mini-language (see system "
                "instructions). Every identifier must be a declared "
                "entity id, a declared unknown_scalars name, one of "
                "A(...)/d(...)/dot(...)/cross(...)/norm(...), or the "
                "reserved zero/0."
            ),
        },
        "unknown_scalars": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Free scalar parameters solved for by the compiler, e.g. "
                "[\"lambda\"] or [\"t\", \"s\"]. [] if none."
            ),
        },
        "target": {
            "type": "string",
            "description": "Human-readable description of what \"Find\" asks for.",
        },
        "query_expr": {
            "type": "string",
            "description": (
                "The mini-language expression the compiler evaluates for "
                "the final answer, e.g. \"dot(a + c, a + c)\"."
            ),
        },
    },
    "required": [
        "problem_type", "entities", "relations", "equations",
        "unknown_scalars", "target", "query_expr",
    ],
    "additionalProperties": False,
}
