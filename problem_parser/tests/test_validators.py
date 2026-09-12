"""Unit tests for validators.py -- no Gemini, no network, no key.

These target the two validators the task explicitly calls for:
  1. rejecting a solved geometry value on an "unknown" entity, and
  2. rejecting an equation/query_expr identifier that isn't a declared
     entity id, A()/d()/dot()/cross()/norm(), a declared
     unknown_scalars name, or the reserved zero/0.

Plus a pass over the three golden fixtures to make sure the validators
don't have false positives on data that is, by construction, already
contract-compliant (this doubles as the regression check for the
"is a bare scalar literal like `4` allowed" ambiguity -- see README.md).
"""
from __future__ import annotations

import pytest

from problem_parser.errors import ValidationError
from problem_parser.validators import (
    validate_equation_identifiers,
    validate_kind_field_population,
    validate_parser_never_emits_computed_status,
    validate_problem_ir,
    validate_relations_reference_declared_entities,
    validate_unique_entity_ids,
    validate_unknown_entities_have_no_geometry,
)
from problem_parser.tests.conftest import ALL_PROBLEM_IDS, load_fixture_dict

from shared.schema import Entity, Label, ProblemIR, Vec3, problem_ir_from_dict


def _minimal_ir(entities, equations=None, unknown_scalars=None, query_expr="0"):
    return ProblemIR(
        problem_id="TEST",
        problem_type="test",
        source_text="test",
        entities=entities,
        relations=[],
        equations=equations or [],
        unknown_scalars=unknown_scalars or [],
        target="test",
        query_expr=query_expr,
    )


# ---------------------------------------------------------------------------
# Golden fixtures must validate cleanly (no false positives).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem_id", ALL_PROBLEM_IDS)
def test_golden_fixture_passes_validation(problem_id):
    ir = problem_ir_from_dict(load_fixture_dict(problem_id))
    validate_problem_ir(ir)  # must not raise


# ---------------------------------------------------------------------------
# Hard rule: unknown entities must never carry solved geometry.
# ---------------------------------------------------------------------------


def test_unknown_entity_with_populated_coordinates_is_rejected():
    entities = [
        Entity(
            id="F",
            kind="point",
            label=Label(text="F", anchor="east"),
            status="unknown",
            visual_role="computed_point",
            coordinates=Vec3(x="1", y="6", z="0"),  # <- the bug we must catch
        )
    ]
    issues = validate_unknown_entities_have_no_geometry(_minimal_ir(entities))
    assert len(issues) == 1
    assert "F" in issues[0]
    assert "unknown" in issues[0]
    assert "coordinates" in issues[0]


def test_unknown_entity_with_populated_components_is_rejected():
    entities = [
        Entity(
            id="c",
            kind="vector",
            label=Label(text="c", anchor="east"),
            status="unknown",
            visual_role="computed_vector",
            components=Vec3(x="2", y="-1", z="3"),
        )
    ]
    issues = validate_unknown_entities_have_no_geometry(_minimal_ir(entities))
    assert len(issues) == 1
    assert "components" in issues[0]


def test_unknown_entity_with_populated_anchor_and_direction_is_rejected():
    entities = [
        Entity(
            id="L3",
            kind="parametric_line",
            label=Label(text="L3", anchor="north"),
            status="unknown",
            visual_role="computed_line",
            anchor=Vec3(x="0", y="0", z="0"),
            direction=Vec3(x="-3", y="5", z="16"),
        )
    ]
    issues = validate_unknown_entities_have_no_geometry(_minimal_ir(entities))
    assert len(issues) == 2
    joined = " ".join(issues)
    assert "anchor" in joined and "direction" in joined


def test_unknown_entity_may_keep_param_name():
    # param_name is just the parameter's NAME (e.g. "lambda"), not a
    # solved value -- it must NOT be flagged even while status is
    # "unknown".
    entities = [
        Entity(
            id="L3",
            kind="parametric_line",
            label=Label(text="L3", anchor="north"),
            status="unknown",
            visual_role="computed_line",
            param_name="u",
        )
    ]
    assert validate_unknown_entities_have_no_geometry(_minimal_ir(entities)) == []


def test_given_entity_with_populated_geometry_is_fine():
    entities = [
        Entity(
            id="P",
            kind="point",
            label=Label(text="P", anchor="north"),
            status="given",
            visual_role="given_external_point",
            coordinates=Vec3(x="5", y="4", z="2"),
        )
    ]
    assert validate_unknown_entities_have_no_geometry(_minimal_ir(entities)) == []


def test_end_to_end_validate_problem_ir_raises_validation_error():
    entities = [
        Entity(
            id="F",
            kind="point",
            label=Label(text="F", anchor="east"),
            status="unknown",
            visual_role="computed_point",
            coordinates=Vec3(x="1", y="6", z="0"),
        )
    ]
    ir = _minimal_ir(entities)
    with pytest.raises(ValidationError) as exc_info:
        validate_problem_ir(ir)
    assert any("F" in issue for issue in exc_info.value.issues)


# ---------------------------------------------------------------------------
# Hard rule: every equation/query_expr identifier must be declared.
# ---------------------------------------------------------------------------


def _given_point(entity_id):
    return Entity(
        id=entity_id,
        kind="point",
        label=Label(text=entity_id, anchor="north"),
        status="given",
        visual_role="given_external_point",
        coordinates=Vec3(x="0", y="0", z="0"),
    )


def _given_line(entity_id):
    return Entity(
        id=entity_id,
        kind="parametric_line",
        label=Label(text=entity_id, anchor="north"),
        status="given",
        visual_role="given_line",
        anchor=Vec3(x="0", y="0", z="0"),
        direction=Vec3(x="1", y="0", z="0"),
        param_name="t",
    )


def test_undeclared_identifier_in_equation_is_rejected():
    entities = [_given_point("P")]
    ir = _minimal_ir(entities, equations=["P == q"])  # 'q' is not declared anywhere
    issues = validate_equation_identifiers(ir)
    assert any("q" in issue and "undeclared" in issue for issue in issues)


def test_undeclared_identifier_in_query_expr_is_rejected():
    entities = [_given_point("P")]
    ir = _minimal_ir(entities, query_expr="dot(P, ghost)")
    issues = validate_equation_identifiers(ir)
    assert any("ghost" in issue for issue in issues)


def test_declared_entity_ids_are_accepted():
    entities = [_given_point("P"), _given_point("Q")]
    ir = _minimal_ir(entities, equations=["P == Q"], query_expr="dot(P, Q)")
    assert validate_equation_identifiers(ir) == []


def test_declared_unknown_scalar_is_accepted():
    entities = [_given_point("P"), _given_line("L")]
    ir = _minimal_ir(
        entities,
        equations=["P == A(L) + lambda * d(L)"],
        unknown_scalars=["lambda"],
    )
    assert validate_equation_identifiers(ir) == []


def test_undeclared_scalar_param_is_rejected():
    entities = [_given_point("P"), _given_line("L")]
    # 'mu' used but never declared in unknown_scalars
    ir = _minimal_ir(entities, equations=["P == A(L) + mu * d(L)"], unknown_scalars=[])
    issues = validate_equation_identifiers(ir)
    assert any("mu" in issue for issue in issues)


def test_reserved_zero_and_bare_numeric_literals_are_allowed():
    entities = [_given_point("P")]
    ir = _minimal_ir(
        entities,
        equations=["cross(P, P) == zero", "dot(P, P) == 4"],
    )
    # Neither 'zero' nor the bare scalar literal '4' should be flagged --
    # see the module docstring in validators.py on RESERVED_BARE_IDENTIFIERS.
    assert validate_equation_identifiers(ir) == []


def test_unknown_function_name_is_rejected():
    entities = [_given_point("P")]
    ir = _minimal_ir(entities, equations=["foo(P) == zero"])
    issues = validate_equation_identifiers(ir)
    assert any("foo" in issue and "unknown function" in issue for issue in issues)


def test_A_call_on_non_line_entity_is_rejected():
    entities = [_given_point("P")]
    # A(...) requires its argument to be a parametric_line entity; P is a point.
    ir = _minimal_ir(entities, equations=["P == A(P)"])
    issues = validate_equation_identifiers(ir)
    assert any("A(P)" in issue and "parametric_line" in issue for issue in issues)


def test_A_call_on_line_entity_is_accepted():
    entities = [_given_point("P"), _given_line("L")]
    ir = _minimal_ir(entities, equations=["P == A(L)"])
    assert validate_equation_identifiers(ir) == []


def test_equation_missing_double_equals_is_flagged():
    entities = [_given_point("P")]
    ir = _minimal_ir(entities, equations=["P + P"])  # no '=='
    issues = validate_equation_identifiers(ir)
    assert any("==" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Supplementary validators (not explicitly required, but cheap and useful).
# ---------------------------------------------------------------------------


def test_duplicate_entity_ids_are_rejected():
    entities = [_given_point("P"), _given_point("P")]
    issues = validate_unique_entity_ids(_minimal_ir(entities))
    assert len(issues) == 1
    assert "P" in issues[0]


def test_kind_field_mismatch_is_rejected():
    # A "point" with components populated instead of coordinates.
    entity = Entity(
        id="P",
        kind="point",
        label=Label(text="P", anchor="north"),
        status="given",
        visual_role="given_external_point",
        components=Vec3(x="1", y="2", z="3"),
    )
    issues = validate_kind_field_population(_minimal_ir([entity]))
    assert any("components" in issue for issue in issues)


def test_parser_output_may_not_claim_computed_status():
    entity = Entity(
        id="F",
        kind="point",
        label=Label(text="F", anchor="east"),
        status="computed",  # never valid for problem_parser's own output
        visual_role="computed_point",
        coordinates=Vec3(x="1", y="6", z="0"),
    )
    issues = validate_parser_never_emits_computed_status(_minimal_ir([entity]))
    assert len(issues) == 1
    assert "F" in issues[0] and "computed" in issues[0]


def test_relation_referencing_undeclared_entity_is_rejected():
    from shared.schema import Relation

    entities = [_given_point("P")]
    ir = _minimal_ir(entities)
    ir.relations = [Relation(kind="lies_on", subject="P", object="ghost_line")]
    issues = validate_relations_reference_declared_entities(ir)
    assert any("ghost_line" in issue for issue in issues)
