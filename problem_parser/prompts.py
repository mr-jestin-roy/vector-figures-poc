"""Prompt template for the Gemini extraction call.

Held-out fixture: M26S2J21Q3 ("two lines with a third-direction
constraint") deliberately does NOT appear anywhere below. The other two
problems (M26S1J21Q55: cross-product + plane-dot system; M26S1J21Q64:
foot-of-perpendicular + projection) are embedded as worked few-shot
examples. This means the one live end-to-end test in
tests/test_integration_live.py (which targets M26S2J21Q3) is an honest
test of generalization, not a check that the model can echo back
something it was already shown. See README.md for more on this choice.

The few-shot JSON blocks below are hand-written to match
fixtures/problem_ir/M26S1J21Q55.json and .../M26S1J21Q64.json (with
`problem_id` and `source_text` stripped out, since the real pipeline
supplies those two fields itself -- see schema_json.py's module
docstring for why). They are inlined as string literals rather than
read from fixtures/ at import time, so this package has no runtime
dependency on that directory.
"""
from __future__ import annotations

import json

ROLE_STYLE_TEXT = """\
ROLE_STYLE convention (visual_role must be exactly one of these six
strings -- never invent a new one, never leave it to a renderer to guess):
  - "given_external_point": a point given directly in the problem that is
    not itself defined as lying on a line/curve (e.g. the point P you drop
    a perpendicular *from*).
  - "given_line": a line given directly (has a known anchor + direction).
  - "given_vector": a vector given directly by its components.
  - "computed_point": a point that is NOT given directly -- it is defined
    by a condition (foot of perpendicular, intersection, etc.) and will be
    solved by the geometry_compiler. Use this even while the parser leaves
    the entity's status as "unknown" and its coordinates unpopulated.
  - "computed_line": a line defined by a condition rather than given
    directly (rare -- none of the worked examples below need this, but it
    exists for symmetry).
  - "computed_vector": a vector that is NOT given directly -- e.g. a
    vector pinned down by other equations, still unknown until solved.
Rule of thumb: "given_*" <=> Entity.status == "given"; "computed_*" <=>
Entity.status is "unknown" (not yet solved) or "computed" (already
resolved elsewhere, which never happens in your output -- you only ever
emit "given" or "unknown", never "computed"; "computed" status is the
geometry_compiler's own output state, not yours).
"""

GRAMMAR_TEXT = """\
`equations` / `query_expr` mini-language grammar (CONTRACT.md):
  - Every declared entity id resolves to a 3-vector: a point's
    coordinates, a vector's components, or (for a parametric_line) via
    A(id) / d(id).
  - A(line_id), d(line_id) -- anchor point / direction of a
    parametric_line entity. The argument MUST be the id of an entity
    whose kind is "parametric_line".
  - dot(u, v), cross(u, v), norm(v) -- the only functions besides A/d.
  - Operators: + - * (scalar*vector or vector +- vector), ** (power, for
    query_expr things like squared length), == (equation only).
  - Reserved literals: `zero` (the zero vector) and bare scalar numbers
    like `0` or `4` (e.g. "dot(c, n) == 4"). Every other constant MUST be
    a declared Entity -- never smuggle in a raw vector/point tuple like
    "(1,2,3)"; there is no syntax for that at all, so if you need a fixed
    direction or point as a constant, declare it as its own Entity
    (given status, populated components/coordinates) and reference its
    id, the way "n" and "n3" are declared as Entities below instead of
    writing "(1,1,1)" or "(-3,5,16)" inline.
  - Scalar parameters solved for by the compiler must be declared in
    `unknown_scalars` (e.g. ["lambda"], ["t", "s"]) and referenced by
    that exact bare name in `equations` (e.g. "lambda * d(L)").
  - Every identifier you use in `equations`/`query_expr` MUST be one of:
    a declared Entity id, A(...)/d(...)/dot(...)/cross(...)/norm(...), a
    name declared in `unknown_scalars`, or the reserved `zero`. Anything
    else is a bug -- a validator downstream will reject it.
"""

NEVER_SOLVE_TEXT = """\
THE SINGLE MOST IMPORTANT RULE: you never solve anything. You are the
"parser" half of a two-stage pipeline; a separate "geometry_compiler"
does 100% of the algebra later, working ONLY from the `equations` you
write down. Concretely:
  - If an Entity's status is "unknown" (i.e. the problem defines it by a
    condition rather than giving it directly -- a foot of a
    perpendicular, an intersection point, a vector pinned down by two
    equations, etc.), then coordinates / anchor / direction / components
    for that entity MUST ALL be null/omitted. Do not compute, guess, or
    "helpfully" fill in a value even if you are confident what it is.
  - Never carry a numeric answer from your own mental math into any
    entity field. All arithmetic belongs in `equations`, to be executed
    later by the compiler, not by you.
  - A downstream validator inspects every entity you mark "unknown" and
    will reject the whole response if any of those four fields is
    non-null. There is no partial credit for "solving it right but in
    the wrong stage" -- populating an unknown entity's geometry is
    always treated as a bug, never an acceptable shortcut.
"""

FEWSHOT_1_INPUT = (
    "Given: a = -i + 2j + 2k, b = 8i + 7j - 3k, with a x c = b and "
    "c . (i + j + k) = 4. Find: |a + c|^2"
)
FEWSHOT_1_PROBLEM_ID = "M26S1J21Q55"
FEWSHOT_1_OUTPUT = {
    "problem_type": "vector_cross_and_dot_system",
    "entities": [
        {
            "id": "a",
            "kind": "vector",
            "label": {"text": "a", "anchor": "north west"},
            "status": "given",
            "visual_role": "given_vector",
            "coordinates": None,
            "anchor": None,
            "direction": None,
            "param_name": None,
            "components": {"x": "-1", "y": "2", "z": "2"},
            "notes": [],
        },
        {
            "id": "b",
            "kind": "vector",
            "label": {"text": "b", "anchor": "north east"},
            "status": "given",
            "visual_role": "given_vector",
            "coordinates": None,
            "anchor": None,
            "direction": None,
            "param_name": None,
            "components": {"x": "8", "y": "7", "z": "-3"},
            "notes": [],
        },
        {
            "id": "n",
            "kind": "vector",
            "label": {"text": "(1,1,1)", "anchor": "south"},
            "status": "given",
            "visual_role": "given_vector",
            "coordinates": None,
            "anchor": None,
            "direction": None,
            "param_name": None,
            "components": {"x": "1", "y": "1", "z": "1"},
            "notes": ["direction defining the plane constraint c . n = 4"],
        },
        {
            "id": "c",
            "kind": "vector",
            "label": {"text": "c", "anchor": "east"},
            "status": "unknown",
            "visual_role": "computed_vector",
            "coordinates": None,
            "anchor": None,
            "direction": None,
            "param_name": None,
            "components": None,
            "notes": [
                "only determined up to a multiple of a before the plane "
                "constraint is imposed"
            ],
        },
    ],
    "relations": [],
    "equations": ["cross(a, c) == b", "dot(c, n) == 4"],
    "unknown_scalars": [],
    "target": "the squared length of the vector a + c",
    "query_expr": "dot(a + c, a + c)",
}

FEWSHOT_2_INPUT = (
    "Given: (alpha,beta,gamma) is the foot of the perpendicular from "
    "(5,4,2) to the line r = (-i+3j+k) + lambda(2i+3j-k). Find: the "
    "projection of alpha*i+beta*j+gamma*k onto 6i+2j+3k"
)
FEWSHOT_2_PROBLEM_ID = "M26S1J21Q64"
FEWSHOT_2_OUTPUT = {
    "problem_type": "point_and_line_foot_of_perpendicular",
    "entities": [
        {
            "id": "P",
            "kind": "point",
            "label": {"text": "P(5,4,2)", "anchor": "north"},
            "status": "given",
            "visual_role": "given_external_point",
            "coordinates": {"x": "5", "y": "4", "z": "2"},
            "anchor": None,
            "direction": None,
            "param_name": None,
            "components": None,
            "notes": [],
        },
        {
            "id": "L",
            "kind": "parametric_line",
            "label": {"text": "L", "anchor": "south west"},
            "status": "given",
            "visual_role": "given_line",
            "coordinates": None,
            "anchor": {"x": "-1", "y": "3", "z": "1"},
            "direction": {"x": "2", "y": "3", "z": "-1"},
            "param_name": "lambda",
            "components": None,
            "notes": [],
        },
        {
            "id": "F",
            "kind": "point",
            "label": {"text": "F=(alpha,beta,gamma)", "anchor": "east"},
            "status": "unknown",
            "visual_role": "computed_point",
            "coordinates": None,
            "anchor": None,
            "direction": None,
            "param_name": None,
            "components": None,
            "notes": ["foot of the perpendicular from P to L"],
        },
        {
            "id": "v",
            "kind": "vector",
            "label": {"text": "(6,2,3)", "anchor": "south east"},
            "status": "given",
            "visual_role": "given_vector",
            "coordinates": None,
            "anchor": None,
            "direction": None,
            "param_name": None,
            "components": {"x": "6", "y": "2", "z": "3"},
            "notes": ["projection target direction"],
        },
    ],
    "relations": [
        {"kind": "lies_on", "subject": "F", "object": "L", "via": None},
        {"kind": "perpendicular_to", "subject": "F", "object": "L", "via": "P"},
        {"kind": "projects_onto", "subject": "F", "object": "v", "via": None},
    ],
    "equations": ["F == A(L) + lambda * d(L)", "dot(P - F, d(L)) == 0"],
    "unknown_scalars": ["lambda"],
    "target": "the scalar projection of F (as a vector from the origin) onto v",
    "query_expr": "dot(F, v) / norm(v)",
}


def _fewshot_block(problem_id: str, raw_text: str, output: dict) -> str:
    return (
        f"### Example input (Problem ID: {problem_id})\n"
        f"{raw_text}\n\n"
        "### Example output (JSON)\n"
        f"{json.dumps(output, indent=2)}\n"
    )


FEWSHOT_EXAMPLES_TEXT = (
    "Worked examples (study the pattern, especially which fields stay "
    "null and why; do NOT copy their entity ids or numbers into an "
    "unrelated problem):\n\n"
    + _fewshot_block(FEWSHOT_1_PROBLEM_ID, FEWSHOT_1_INPUT, FEWSHOT_1_OUTPUT)
    + "\n"
    + _fewshot_block(FEWSHOT_2_PROBLEM_ID, FEWSHOT_2_INPUT, FEWSHOT_2_OUTPUT)
)


SYSTEM_INSTRUCTION = f"""\
You are the extraction stage of a two-stage math pipeline for turning
vector-geometry word problems into a verified 3D scene graph. You read a
problem's "Given/Find" text and structure it into a ProblemIR: a list of
geometric entities, the relations between them, and the algebra
(`equations`) a separate solver will use to actually solve the problem.

{NEVER_SOLVE_TEXT}
{GRAMMAR_TEXT}
{ROLE_STYLE_TEXT}
Other rules:
  - Every point/line/vector mentioned in the "Given" or "Find" text must
    become its own Entity with a short, stable id (e.g. "P", "F", "L",
    "c") -- reuse the letters the problem itself uses where natural.
  - `label.anchor` must be one of the eight compass directions (north,
    south, east, west, north east, north west, south east, south west)
    -- never a bare offset.
  - `notes` are free-text, human-facing only. They are never parsed or
    solved -- put ALL of the real algebra in `equations`, never in prose
    inside a note.
  - `target` is a short human-readable description of what "Find" asks
    for. `query_expr` is the mini-language expression the compiler
    evaluates to actually produce that answer.
  - Output must conform exactly to the JSON schema you were given for
    this response. Output ONLY the JSON -- no prose, no markdown code
    fences.

{FEWSHOT_EXAMPLES_TEXT}
Now extract the ProblemIR for the new problem given in the user message,
following every rule above exactly -- in particular, remember that ANY
entity you mark "unknown" must have null coordinates/anchor/direction/
components. Never solve.
"""


def build_user_prompt(raw_text: str, problem_id: str) -> str:
    """The per-call user message. `problem_id` is included only so the
    model has full context (e.g. to pick a sensible `problem_type`); it
    is never trusted back out of the response -- parser.py stamps the
    caller-supplied problem_id onto the result unconditionally.
    """
    return (
        f"Problem ID: {problem_id}\n\n"
        f"{raw_text.strip()}\n\n"
        "Extract this into the ProblemIR JSON structure described in the "
        "system instructions. Remember: never solve for an unknown "
        "entity's geometry. Output only the JSON."
    )
