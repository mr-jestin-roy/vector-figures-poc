# Pipeline contract: Problem Parser → Geometry Compiler

This document is the shared spec for two independently-built components.
Neither one should redefine data shapes that live in [`shared/schema.py`](shared/schema.py) --
that file is the single source of truth and has zero third-party
dependencies so both sides can import it unmodified.

```
LaTeX problem block (from *3 vector figures problem latex code.tex*)
        │
        ▼
┌───────────────────┐
│  problem_parser    │   Gemini API extracts structure ONLY.
│                    │   Never solves, never guesses a coordinate.
└─────────┬──────────┘
          │  ProblemIR  (shared/schema.py)
          ▼
┌───────────────────┐
│ geometry_compiler   │   Exact-rational symbolic solve of every
│                    │   equation. Verifies every relation with an
│                    │   EXACT zero residual, not a float tolerance.
└─────────┬──────────┘
          │  SceneGraph  (shared/schema.py)
          ▼
   (renderer / TikZ / validator -- out of scope for this pass)
```

## Why this split

Getting *one* figure right in this project (documented in
[`Instructions for Figures.pdf`](Instructions%20for%20Figures.pdf)) took many
rounds of manual iteration -- and most of that pain was rendering/layout
engineering (font size, baseline alignment, label collisions), not geometry.
Keeping the math engine authoritative and independent of rendering means
the compiler can be fully unit-tested with zero rendering code in the loop.

## Ground truth: the fixtures

[`fixtures/problem_ir/*.json`](fixtures/problem_ir/) and
[`fixtures/scene_graph/*.json`](fixtures/scene_graph/) are the golden
input/output pairs for **all three** problems in the `.tex` file, not just
the one worked example in the PDF. I hand-derived and independently
verified all three by exact arithmetic before writing them:

| problem_id | type | final_answer |
|---|---|---|
| `M26S1J21Q55` | cross-product + plane-dot system | `27` |
| `M26S1J21Q64` | foot-of-perpendicular + projection | `18/7` |
| `M26S2J21Q3` | two lines + third-direction constraint | `290` |

- `problem_parser` is graded on: LaTeX source text in → does it produce a
  `ProblemIR` structurally equivalent to the fixture (same entities, same
  equations up to algebraic restatement, same `query_expr` semantics)?
  It should **never** need the compiler to pass its own tests.
- `geometry_compiler` is graded on: fixture `ProblemIR` in → does it
  produce the exact fixture `SceneGraph` (same coordinates, `verified:
  true`, matching `final_answer`)? It should **never** need Gemini or the
  parser to pass its own tests.

This lets both components be built and tested fully in parallel. A later
integration pass (not part of this task) wires parser output directly into
the compiler.

## Hard rules carried over from `Instructions for Figures.pdf`

1. **Verify algebraically, never eyeball.** Every `lies_on` claim must be
   checked by substituting the solved parameter back into `A + t·d` and
   confirming an exact match, not a plausible-looking drawing. This is why
   `geometry_compiler` uses exact rationals (`fractions.Fraction` or
   `sympy.Rational`), never floats.
2. **Explicit label anchors, not offsets.** `Label.anchor` is one of the
   eight compass directions. No `xshift`/`yshift`-only positioning.
3. **Color + line style together, never either alone**, and **always by
   role** (see `ROLE_STYLE` in `shared/schema.py`), never chosen ad hoc.
4. **The parser never solves.** If you catch a Gemini response populating
   `coordinates`/`components`/`anchor`/`direction` for an entity with
   `status: "unknown"`, that is a bug in the parser's prompt or its
   post-processing validator, not an acceptable shortcut.

## `equations` mini-language (for `geometry_compiler` to parse/solve)

Every entry in `ProblemIR.equations` and `ProblemIR.query_expr` is built from:

- entity ids (each resolves to a 3-vector: a point's coordinates, a
  vector's components, or -- for a `parametric_line` -- via `A(id)`/`d(id)`)
- `A(line_id)`, `d(line_id)` -- anchor point / direction of a parametric line
- `dot(u, v)`, `cross(u, v)`, `norm(v)`
- `+ - * /` (vector ± vector, scalar * vector, scalar / scalar or vector /
  scalar), `**` (power, for `query_expr` like squared length), `==`
- reserved constant `zero` (the zero vector), for `cross(u, v) == zero`
  parallelism checks
- bare scalar number literals (integers, or rationals like `7/2`) ARE
  allowed directly in an equation -- e.g. `dot(c, n) == 4` (fixture
  `M26S1J21Q55`) is valid. What is NOT allowed is a bare *vector* literal
  tuple like `(8, 7, -3)`: any 3D quantity must be a declared `Entity`
  (see `n` and `n3` in the fixtures for how a bare direction, e.g. a
  plane normal, is declared rather than inlined). This was corrected
  after `geometry_compiler`'s first implementation hit the literal `4`
  in `M26S1J21Q55` and the `/` in `M26S1J21Q64`'s `query_expr` -- the
  original draft of this section said scalars were restricted the same
  way vectors are, which the fixtures never actually required.
- scalar parameters declared in `unknown_scalars` (e.g. `lambda`, `t`, `s`)

Do **not** literal-`eval()` these strings. Parse them safely (a small
hand-rolled recursive-descent parser, or `sympy.parsing.sympy_parser` with
a locked-down local symbol table) since the strings originate from an LLM.

## Division of labor

- **Agent A -- `problem_parser/`**: LaTeX text → Gemini API call →
  `ProblemIR`. Owns prompt design, response-schema enforcement (Gemini
  structured output / function-calling, not free-form prose parsing),
  and a validator that rejects any solved value for an `unknown` entity.
- **Agent B -- `geometry_compiler/`**: `ProblemIR` → solve → `SceneGraph`.
  Owns the equation parser/solver and the exact-arithmetic verification
  pass. Should support the three constraint families already present in
  the fixtures (linear cross-product system, foot-of-perpendicular, two
  parametrized lines forced parallel to a third direction) generically
  enough that a fourth problem of a similar shape would not need new code.

Both agents read this file and `shared/schema.py` before writing code, and
must flag here (or in their own README) any point where they found the
contract ambiguous or had to make a judgment call.
