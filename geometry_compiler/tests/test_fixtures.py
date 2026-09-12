"""geometry_compiler is graded on: fixture ProblemIR in -> does it
produce the exact fixture SceneGraph? (CONTRACT.md, "Ground truth: the
fixtures"). This exercises all three constraint families -- cross
product + plane-dot system, foot-of-perpendicular + projection, and two
parametrized lines forced parallel to a third direction -- through the
one generic `compile()` entrypoint, with no per-problem special casing
in the test itself.
"""
import json
from fractions import Fraction

import pytest

from conftest import FIXTURE_IDS, PROBLEM_IR_DIR, SCENE_GRAPH_DIR

from shared.schema import problem_ir_from_dict, scene_graph_from_dict
from geometry_compiler import compile


def _load(fixture_id: str):
    problem_ir = problem_ir_from_dict(
        json.loads((PROBLEM_IR_DIR / f"{fixture_id}.json").read_text())
    )
    expected = scene_graph_from_dict(
        json.loads((SCENE_GRAPH_DIR / f"{fixture_id}.json").read_text())
    )
    return problem_ir, expected


def _fr(rational_str: str) -> Fraction:
    return Fraction(rational_str)


def _assert_vec3_equal(actual, expected, where: str):
    assert actual is not None, f"{where}: expected a populated Vec3, got None"
    assert _fr(actual.x) == _fr(expected.x), f"{where}.x: {actual.x!r} != {expected.x!r}"
    assert _fr(actual.y) == _fr(expected.y), f"{where}.y: {actual.y!r} != {expected.y!r}"
    assert _fr(actual.z) == _fr(expected.z), f"{where}.z: {actual.z!r} != {expected.z!r}"


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_compile_matches_fixture_scene_graph(fixture_id):
    problem_ir, expected = _load(fixture_id)

    actual = compile(problem_ir)

    assert actual.problem_id == expected.problem_id

    expected_by_id = {e.id: e for e in expected.entities}
    actual_by_id = {e.id: e for e in actual.entities}
    assert set(actual_by_id) == set(expected_by_id), "entity id sets differ"

    for entity_id, expected_entity in expected_by_id.items():
        actual_entity = actual_by_id[entity_id]
        where = f"{fixture_id}:{entity_id}"
        assert actual_entity.status == expected_entity.status, f"{where}: status"
        assert actual_entity.kind == expected_entity.kind, f"{where}: kind"

        if expected_entity.kind == "point":
            _assert_vec3_equal(actual_entity.coordinates, expected_entity.coordinates, f"{where}.coordinates")
        elif expected_entity.kind == "vector":
            _assert_vec3_equal(actual_entity.components, expected_entity.components, f"{where}.components")
        elif expected_entity.kind == "parametric_line":
            _assert_vec3_equal(actual_entity.anchor, expected_entity.anchor, f"{where}.anchor")
            _assert_vec3_equal(actual_entity.direction, expected_entity.direction, f"{where}.direction")

    # solved_scalars: same names, exact-rational-equal values (Fraction
    # comparison, not string equality -- "6/1" and "6" must count as equal).
    assert set(actual.solved_scalars) == set(expected.solved_scalars), "solved_scalars keys differ"
    for name, expected_val in expected.solved_scalars.items():
        assert _fr(actual.solved_scalars[name]) == _fr(expected_val), f"solved_scalars[{name}]"

    # Every declared equation must have been independently re-verified
    # with an exact zero residual.
    assert len(actual.verifications) == len(problem_ir.equations)
    for v in actual.verifications:
        assert v.passed, f"{fixture_id}: verification failed for {v.check!r} (residual {v.residual})"
        assert _fr(v.residual) == 0

    assert actual.verified is True
    assert expected.verified is True  # sanity-check the fixture itself

    assert _fr(actual.final_answer) == _fr(expected.final_answer), (
        f"{fixture_id}: final_answer {actual.final_answer!r} != {expected.final_answer!r}"
    )
