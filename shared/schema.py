"""Shared data contract between problem_parser and geometry_compiler.

Both packages MUST import these types rather than redefining their own.
This file has zero third-party dependencies (stdlib only) so it can be
imported unmodified from either package regardless of what else they
each install (google-genai in problem_parser, sympy in geometry_compiler).

Design rules (do not violate without updating CONTRACT.md first):

1. All numeric fields are exact-rational strings (e.g. "7/2", "-1", "18/7"),
   parsed via fractions.Fraction -- NEVER floats. This lets the compiler
   check algebraic identities as exact zero, not "close enough".
2. Every 3D vector/point that appears in `equations` or `query_expr` must
   be a declared Entity -- no bare vector-literal tuples like (8,7,-3)
   smuggling in problem data that bypassed parsing. Bare *scalar* number
   literals (e.g. `4`, `7/2`) are fine (they're not "data" in the same
   sense -- see CONTRACT.md). The reserved constant `zero` is the zero
   vector, for `cross(u, v) == zero` parallelism checks.
3. problem_parser produces ProblemIR and must NEVER populate `coordinates`
   / `anchor` / `direction` / `components` for an entity whose status is
   "unknown". Solving is the geometry_compiler's job, not the parser's.
4. geometry_compiler consumes ProblemIR and produces SceneGraph, in which
   every entity's status is "given" or "computed" and every geometric
   field is populated -- and `verified` is True only if every equation's
   residual is an exact 0 (Fraction(0)), not merely small.
5. Visual encoding (color, line style) is derived from `visual_role` via
   the fixed ROLE_STYLE table below -- it is never chosen ad hoc by the
   parser, the compiler, or a downstream renderer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Literal, Optional
import json

Rational = str  # a string parseable by fractions.Fraction, e.g. "7/2", "-3", "18/7"


def R(x: Rational) -> Fraction:
    return Fraction(x)


def rational(f: Fraction) -> Rational:
    return str(f)


EntityKind = Literal["point", "parametric_line", "vector"]
EntityStatus = Literal["given", "unknown", "computed"]

VisualRole = Literal[
    "given_external_point",
    "given_line",
    "given_vector",
    "computed_point",
    "computed_line",
    "computed_vector",
]

AnchorDir = Literal[
    "north", "south", "east", "west",
    "north east", "north west", "south east", "south west",
]

# "Color by role, not by arbitrary choice" (lesson 5) + "color AND line
# style, never angle alone" (lesson 4). This table is the single source
# of truth for visual encoding; nothing downstream should invent colors.
ROLE_STYLE: dict[VisualRole, dict[str, str]] = {
    "given_external_point": {"color": "red", "line_style": "none"},
    "given_line": {"color": "navy", "line_style": "solid"},
    "given_vector": {"color": "navy", "line_style": "dashed"},
    "computed_point": {"color": "darkgreen", "line_style": "none"},
    "computed_line": {"color": "darkgreen", "line_style": "solid"},
    "computed_vector": {"color": "darkgreen", "line_style": "dotted"},
}


@dataclass
class Vec3:
    x: Rational
    y: Rational
    z: Rational

    def as_fractions(self) -> tuple[Fraction, Fraction, Fraction]:
        return (R(self.x), R(self.y), R(self.z))

    @staticmethod
    def from_fractions(t: tuple[Fraction, Fraction, Fraction]) -> "Vec3":
        return Vec3(rational(t[0]), rational(t[1]), rational(t[2]))


@dataclass
class Label:
    text: str
    anchor: AnchorDir  # explicit anchor, never a bare xshift/yshift (lesson 2)


@dataclass
class Entity:
    id: str  # short stable id referenced by `equations`/`query_expr`, e.g. "P", "F", "L", "c"
    kind: EntityKind
    label: Label
    status: EntityStatus
    visual_role: VisualRole

    # Populated depending on `kind`; all must stay None while status == "unknown".
    coordinates: Optional[Vec3] = None            # kind == "point"
    anchor: Optional[Vec3] = None                 # kind == "parametric_line" (A)
    direction: Optional[Vec3] = None               # kind == "parametric_line" (d)
    param_name: Optional[str] = None               # kind == "parametric_line", e.g. "lambda"
    components: Optional[Vec3] = None               # kind == "vector"

    # Free-text, human/render-facing notes only (e.g. "defined as foot of
    # perpendicular from P"). NOT parsed or solved -- see ProblemIR.equations
    # for the actual algebra the compiler executes.
    notes: list[str] = field(default_factory=list)


@dataclass
class Relation:
    """Cosmetic/semantic hint for a renderer (e.g. draw a right-angle mark,
    draw a dashed drop-line for a projection). Never solved directly --
    the compiler solves `equations`, not `relations`."""
    kind: Literal["lies_on", "perpendicular_to", "parallel_to", "projects_onto"]
    subject: str            # entity id
    object: str             # entity id
    via: Optional[str] = None  # e.g. perpendicular_to(F, L) via P means segment(P,F) ⟂ direction(L)


@dataclass
class ProblemIR:
    """Output of problem_parser.

    `equations` is the authoritative algebra the geometry_compiler solves.
    Mini-language grammar (every 3D-vector identifier must be a declared
    Entity id; bare scalar number literals are fine):
        dot(u, v), cross(u, v), norm(v), A(line_id), d(line_id)
        + - * / (scalar*vector, vector/scalar, scalar/scalar, vector+-vector)
        ==, **
        zero    -- the zero vector (0,0,0), for parallelism checks
    Every equation is `LHS == RHS`. No bare vector-literal tuples like
    (8,7,-3) -- such constants must be declared as their own `given`
    Entity (see CONTRACT.md for the full rationale and examples).
    """
    problem_id: str
    problem_type: str
    source_text: str
    entities: list[Entity]
    relations: list[Relation]
    equations: list[str]
    unknown_scalars: list[str]   # free parameters to solve for, e.g. ["lambda"], ["t", "s"]
    target: str                 # human-readable description of what "Find" asks for
    query_expr: str              # mini-language expression evaluated for the final answer


@dataclass
class VerificationResult:
    check: str          # the equation checked, verbatim from ProblemIR.equations (or derived)
    passed: bool
    residual: Rational   # exact residual as a rational string; "0" iff passed


@dataclass
class SceneGraph:
    """Output of geometry_compiler. Every entity is fully resolved: status
    is only "given" or "computed", and the relevant geometric field is
    always populated. `verified` is True only if every VerificationResult
    has residual == "0" exactly."""
    problem_id: str
    entities: list[Entity]
    relations: list[Relation]
    solved_scalars: dict[str, Rational]   # e.g. {"lambda": "1"}
    verifications: list[VerificationResult]
    verified: bool
    final_answer: Rational


def to_json(obj) -> str:
    return json.dumps(asdict(obj), indent=2)


def entity_from_dict(d: dict) -> Entity:
    d = dict(d)
    if d.get("label"):
        d["label"] = Label(**d["label"])
    for k in ("coordinates", "anchor", "direction", "components"):
        if d.get(k):
            d[k] = Vec3(**d[k])
    return Entity(**d)


def problem_ir_from_dict(d: dict) -> ProblemIR:
    d = dict(d)
    d["entities"] = [entity_from_dict(e) for e in d["entities"]]
    d["relations"] = [Relation(**r) for r in d["relations"]]
    return ProblemIR(**d)


def scene_graph_from_dict(d: dict) -> SceneGraph:
    d = dict(d)
    d["entities"] = [entity_from_dict(e) for e in d["entities"]]
    d["relations"] = [Relation(**r) for r in d["relations"]]
    d["verifications"] = [VerificationResult(**v) for v in d["verifications"]]
    return SceneGraph(**d)
