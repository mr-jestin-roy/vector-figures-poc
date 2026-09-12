"""Post-processing validators enforcing the parser/compiler contract.

Structured output (see schema_json.py) constrains the *shape* of a
Gemini response, but JSON Schema cannot express "field X must be null
whenever sibling field Y equals a specific value" in the restricted
subset Gemini supports. So this module is the actual enforcement point
for CONTRACT.md's hard rule 4 ("the parser never solves") and
shared/schema.py's design rules 2 and 3. Every validator here is a pure
function over an already-constructed `ProblemIR` -- no I/O, no Gemini
calls -- so they are trivial to unit test in isolation.

`validate_problem_ir` is the single entry point `parser.py` calls; the
individual `validate_*` functions are exposed too so tests can target
one rule at a time.
"""
from __future__ import annotations

import re

from problem_parser.errors import ValidationError

from shared.schema import Entity, ProblemIR, Relation

# The only function names the equations mini-language defines
# (CONTRACT.md, "equations mini-language" section).
ALLOWED_FUNCTIONS = {"A", "d", "dot", "cross", "norm"}

# The only bare-word literal the grammar reserves. (The scalar literal
# `0`, and any other bare integer literal such as the `4` in
# `dot(c, n) == 4` from fixtures/problem_ir/M26S1J21Q55.json, is a
# *number* token, not an *identifier* token, so it never reaches the
# identifier-membership check below at all -- see the "Ambiguity" note
# in README.md for why bare scalar numeric literals are treated as
# always-allowed even though CONTRACT.md's prose can be read as banning
# every literal except `zero`/`0`.)
RESERVED_BARE_IDENTIFIERS = {"zero"}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Order matters only for readability of error messages; matching is by
# regex alternation, longest operators first so `==` is never split
# into two `=` (which isn't even a valid token, but be defensive) and
# `**` is never split into two `*`.
_TOKEN_RE = re.compile(r"\*\*|==|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[(),+\-*/]")

# Fields that hold solved/given geometry. Exactly the four fields named
# in CONTRACT.md hard rule 4 and shared/schema.py rule 3. `param_name` is
# deliberately excluded: it is just the *name* of a line's parameter
# (e.g. "lambda"), not a solved value, so it may be present even when
# status == "unknown" (see schema_json.py's field description).
GEOMETRY_FIELDS = ("coordinates", "anchor", "direction", "components")

# Which geometry fields are meaningful for each entity kind. Used by
# validate_kind_field_population to catch e.g. a "vector" entity that
# was accidentally given `coordinates` instead of `components`.
_FIELDS_ALLOWED_BY_KIND = {
    "point": {"coordinates"},
    "parametric_line": {"anchor", "direction", "param_name"},
    "vector": {"components"},
}
_ALL_KIND_SCOPED_FIELDS = {"coordinates", "anchor", "direction", "param_name", "components"}


def _tokenize(expr: str) -> list[str]:
    return _TOKEN_RE.findall(expr)


def _entity_maps(ir: ProblemIR) -> tuple[dict[str, Entity], list[str]]:
    """Returns (id -> Entity, [duplicate-id issue strings])."""
    by_id: dict[str, Entity] = {}
    issues: list[str] = []
    for e in ir.entities:
        if e.id in by_id:
            issues.append(
                f"duplicate entity id '{e.id}' -- ids must be unique."
            )
        else:
            by_id[e.id] = e
    return by_id, issues


def validate_unique_entity_ids(ir: ProblemIR) -> list[str]:
    """Every Entity.id must be unique within the ProblemIR."""
    _, issues = _entity_maps(ir)
    return issues


def validate_unknown_entities_have_no_geometry(ir: ProblemIR) -> list[str]:
    """CONTRACT.md hard rule 4 / schema.py rule 3.

    An entity with status == "unknown" must have coordinates, anchor,
    direction, and components ALL unpopulated (None). Populating any of
    them means the parser solved something -- that is the geometry
    compiler's job, never the parser's.
    """
    issues: list[str] = []
    for e in ir.entities:
        if e.status != "unknown":
            continue
        for field_name in GEOMETRY_FIELDS:
            value = getattr(e, field_name, None)
            if value is not None:
                issues.append(
                    f"entity '{e.id}' has status 'unknown' but field "
                    f"'{field_name}' is populated ({value!r}) -- the "
                    "parser must never solve for an unknown entity's "
                    "geometry (CONTRACT.md hard rule 4)."
                )
    return issues


def validate_kind_field_population(ir: ProblemIR) -> list[str]:
    """Extra rigor beyond the explicit contract: a field that doesn't
    even apply to an entity's `kind` (e.g. `components` on a `point`)
    should never be populated, regardless of status. Catches the model
    mixing up which geometric field goes with which kind.
    """
    issues: list[str] = []
    for e in ir.entities:
        allowed = _FIELDS_ALLOWED_BY_KIND.get(e.kind, set())
        for field_name in _ALL_KIND_SCOPED_FIELDS - allowed:
            value = getattr(e, field_name, None)
            if value is not None:
                issues.append(
                    f"entity '{e.id}' has kind '{e.kind}' but field "
                    f"'{field_name}' is populated ({value!r}); that field "
                    f"only applies to kind(s) that have it in "
                    f"{sorted(_FIELDS_ALLOWED_BY_KIND.items())}."
                )
    return issues


def validate_relations_reference_declared_entities(ir: ProblemIR) -> list[str]:
    """Relation.subject/object/via (when set) must be declared entity ids."""
    entity_ids = {e.id for e in ir.entities}
    issues: list[str] = []
    for r in ir.relations:
        for role, value in (("subject", r.subject), ("object", r.object), ("via", r.via)):
            if value is None:
                continue
            if value not in entity_ids:
                issues.append(
                    f"relation {r.kind}({r.subject}, {r.object}"
                    f"{', via=' + r.via if r.via else ''}): "
                    f"{role} '{value}' is not a declared entity id."
                )
    return issues


def _validate_expression_identifiers(
    expr: str,
    source_label: str,
    entity_ids: set[str],
    entity_kind_by_id: dict[str, str],
    unknown_scalar_names: set[str],
) -> list[str]:
    issues: list[str] = []
    tokens = _tokenize(expr)
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if _IDENTIFIER_RE.match(tok):
            next_tok = tokens[i + 1] if i + 1 < n else None
            if next_tok == "(":
                # `tok` is being used as a function name.
                if tok not in ALLOWED_FUNCTIONS:
                    issues.append(
                        f"{source_label}: unknown function '{tok}(...)' -- "
                        f"allowed functions are {sorted(ALLOWED_FUNCTIONS)}."
                    )
                elif tok in ("A", "d"):
                    # A(line_id) / d(line_id): single-argument, and the
                    # argument must specifically be a parametric_line.
                    arg_tok = tokens[i + 2] if i + 2 < n else None
                    if arg_tok is None or not _IDENTIFIER_RE.match(arg_tok):
                        issues.append(
                            f"{source_label}: '{tok}(...)' must take a "
                            "single entity id as its argument."
                        )
                    elif arg_tok not in entity_ids:
                        issues.append(
                            f"{source_label}: '{tok}({arg_tok})' references "
                            f"undeclared entity id '{arg_tok}'."
                        )
                    elif entity_kind_by_id.get(arg_tok) != "parametric_line":
                        issues.append(
                            f"{source_label}: '{tok}({arg_tok})' requires "
                            f"'{arg_tok}' to be a parametric_line entity, "
                            f"but it is '{entity_kind_by_id.get(arg_tok)}'."
                        )
                # else: dot/cross/norm -- their arguments are validated
                # generically as this same loop continues over the
                # tokens inside the parens.
            else:
                # `tok` is a bare identifier reference, not a function
                # call. Must resolve to something declared.
                if tok in ALLOWED_FUNCTIONS:
                    issues.append(
                        f"{source_label}: '{tok}' is used as a bare "
                        "identifier but is a reserved function name -- "
                        f"did you mean '{tok}(...)'? "
                    )
                elif tok in RESERVED_BARE_IDENTIFIERS:
                    pass
                elif tok in entity_ids:
                    pass
                elif tok in unknown_scalar_names:
                    pass
                else:
                    issues.append(
                        f"{source_label}: undeclared identifier '{tok}' -- "
                        "must be a declared entity id, a declared "
                        "unknown_scalars name, or the reserved 'zero'."
                    )
        i += 1
    return issues


def validate_equation_identifiers(ir: ProblemIR) -> list[str]:
    """Every identifier appearing in `equations` or `query_expr` must be
    one of: a declared Entity id, A(...)/d(...)/dot(...)/cross(...)/
    norm(...), a declared `unknown_scalars` name, or the reserved
    `zero` (bare scalar numeric literals like `0` or `4` are *not*
    identifiers at all and are always allowed -- see the module-level
    comment on RESERVED_BARE_IDENTIFIERS for why).

    Also enforces the A(...)/d(...) argument must specifically name a
    parametric_line entity, since `A`/`d` are only meaningful for lines.
    """
    entity_by_id, dup_issues = _entity_maps(ir)
    entity_ids = set(entity_by_id.keys())
    entity_kind_by_id = {eid: e.kind for eid, e in entity_by_id.items()}
    unknown_scalar_names = set(ir.unknown_scalars)

    issues: list[str] = list(dup_issues)
    for eq in ir.equations:
        if "==" not in eq:
            issues.append(
                f"equation '{eq}' has no '==' -- every entry in "
                "`equations` must be of the form LHS == RHS."
            )
        issues.extend(
            _validate_expression_identifiers(
                eq, f"equation '{eq}'", entity_ids, entity_kind_by_id, unknown_scalar_names
            )
        )

    issues.extend(
        _validate_expression_identifiers(
            ir.query_expr,
            f"query_expr '{ir.query_expr}'",
            entity_ids,
            entity_kind_by_id,
            unknown_scalar_names,
        )
    )
    return issues


def validate_parser_never_emits_computed_status(ir: ProblemIR) -> list[str]:
    """Judgment call, not something CONTRACT.md states in so many words:
    CONTRACT.md hard rule 1 says "the parser never solves"; shared/
    schema.py rule 4 says an entity's status is only "computed" once
    the geometry_compiler has actually solved for it. By that same
    logic, problem_parser's own output should never claim status ==
    "computed" -- only "given" (stated directly in the problem) or
    "unknown" (defined by a condition, not yet solved). None of the
    three golden fixtures ever use "computed" in a ProblemIR, which is
    consistent with this reading. See README.md's "Ambiguities" section.
    """
    issues: list[str] = []
    for e in ir.entities:
        if e.status == "computed":
            issues.append(
                f"entity '{e.id}' has status 'computed' in a ProblemIR -- "
                "problem_parser should never claim something is already "
                "solved; only 'given' or 'unknown' are valid statuses for "
                "its output (CONTRACT.md hard rule 1)."
            )
    return issues


def validate_problem_ir(ir: ProblemIR) -> None:
    """Run every validator and raise ValidationError with ALL issues
    found (not just the first), if any. This is the single function
    parser.py calls after constructing a ProblemIR from a Gemini
    response.
    """
    issues: list[str] = []
    issues += validate_unique_entity_ids(ir)
    issues += validate_unknown_entities_have_no_geometry(ir)
    issues += validate_kind_field_population(ir)
    issues += validate_relations_reference_declared_entities(ir)
    issues += validate_equation_identifiers(ir)
    issues += validate_parser_never_emits_computed_status(ir)
    if issues:
        raise ValidationError(issues)
