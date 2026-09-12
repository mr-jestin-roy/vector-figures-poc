"""Deliberately-inconsistent input must be surfaced clearly, never
crash uninformatively and never silently emit a wrong-but-confident
answer (see the task brief and CONTRACT.md hard rule #1: "verify
algebraically, never eyeball").

We build this by mutating M26S1J21Q55's `b` entity -- the RHS of the
equation "cross(a, c) == b" -- on a deep copy of the fixture.

Why mutate `b` (the cross-product RHS) rather than the "4" in
"dot(c, n) == 4" (the other equation's RHS): for THIS constraint family,
cross(a, c) == b is rank-deficient in c (cross(a, ·) always has a
2-dimensional image, the plane perpendicular to a), so it alone leaves
one free parameter along a; the dot(c, n) == k equation is exactly what
pins that parameter down. Changing k to some other value still yields a
unique (different) but perfectly consistent c -- it does not create an
inconsistency, it just changes the answer. Genuine inconsistency
instead requires breaking the identity that any cross(a, c) is
necessarily perpendicular to a (i.e. dot(a, cross(a, c)) == 0 for all
c) -- so if we corrupt `b` such that dot(a, b) != 0, NO c can satisfy
cross(a, c) == b at all, regardless of the second equation. That is
exactly what this test does.
"""
import copy
import json

import pytest

from conftest import PROBLEM_IR_DIR

from shared.schema import problem_ir_from_dict
from geometry_compiler import compile
from geometry_compiler.exceptions import GeometryCompilerError, SystemInconsistentError


def _load_q55():
    return problem_ir_from_dict(
        json.loads((PROBLEM_IR_DIR / "M26S1J21Q55.json").read_text())
    )


def test_corrupted_cross_product_rhs_is_reported_as_inconsistent():
    problem_ir = _load_q55()

    # Sanity: a . b == 0 in the original (valid) fixture, which is why
    # cross(a, c) == b is solvable at all.
    a = next(e for e in problem_ir.entities if e.id == "a").components
    b = next(e for e in problem_ir.entities if e.id == "b").components
    a_dot_b = int(a.x) * int(b.x) + int(a.y) * int(b.y) + int(a.z) * int(b.z)
    assert a_dot_b == 0

    corrupted = copy.deepcopy(problem_ir)
    b_entity = next(e for e in corrupted.entities if e.id == "b")
    b_entity.components.z = "-2"  # was "-3" -- now dot(a, b) == 2 != 0

    with pytest.raises(GeometryCompilerError) as excinfo:
        compile(corrupted)

    # It must be the specific, informative "no exact solution" failure
    # mode, not some other accidental error, and the message must name
    # what went wrong rather than just being a bare traceback.
    assert isinstance(excinfo.value, SystemInconsistentError)
    message = str(excinfo.value)
    assert "no exact solution" in message.lower() or "inconsistent" in message.lower()


def test_valid_fixture_still_compiles_after_the_corrupted_copy():
    """Guard against the corrupted-copy test accidentally mutating
    shared state (e.g. if deepcopy were missing) and breaking the
    happy-path fixture for anyone running the suite in this order."""
    problem_ir = _load_q55()
    result = compile(problem_ir)
    assert result.verified is True
    assert result.final_answer == "27"
