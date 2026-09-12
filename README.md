# JEE Vector Geometry Solver

A pipeline that turns JEE-style 3D vector-geometry word problems into
**exactly verified** solutions and diagrams — not approximated, not
eyeballed. Given a problem like *"F is the foot of the perpendicular
from P(5,4,2) to a line — find the projection of F onto (6,2,3)"*, the
system extracts the problem's structure, solves it with exact rational
arithmetic, independently re-checks every geometric claim algebraically,
and only then renders a figure.

```
LaTeX word problem
        │
        ▼
┌─────────────────────┐   Gemini API extracts entities, constraints,
│   problem_parser      │   and the target — never solves anything.
└─────────┬────────────┘   A code-level validator rejects any response
          │  ProblemIR       that tries to sneak in a solved coordinate.
          ▼
┌─────────────────────┐   sympy-based solver, exact rationals only.
│  geometry_compiler     │   Every claim (incidence, perpendicularity,
└─────────┬────────────┘   parallelism) is re-verified from scratch —
          │  SceneGraph      "verified: true" means an exact zero
          ▼                  residual, never "close enough".
   diagram (Gemini image model) / TikZ / anything else
```

## Why this is more than a wrapper around an LLM

The interesting engineering problem here isn't "ask an AI to solve a
math problem" — it's **keeping the LLM out of the loop on anything it
could hallucinate**, while still using it for what it's genuinely good
at (turning messy natural language into structured data).

- **The parser is forbidden from solving.** Its JSON schema and a
  separate, independent validator both reject any response that
  populates a coordinate for something the problem hasn't given —
  enforced in code, not just requested in a prompt.
- **Everything is exact, not floating-point.** `18/7`, `27`, `290` are
  the real answers to the three worked problems below, computed and
  verified as exact `Fraction`/`sympy.Rational` values. A `< 1e-9`
  tolerance check would have been *weaker* than what these problems
  actually demand.
- **The two solving stages were built by independent agents in
  parallel, against one shared contract** (`CONTRACT.md`,
  `shared/schema.py`) neither was allowed to modify — and they still
  converged on identical fixes for the same ambiguity they each found
  independently. That convergence is itself a signal the contract was
  well-specified.
- **A rendering bug was caught by actually looking at the output**, not
  assumed away: an image-generation pass initially placed two
  intersection points at the *visual* crossing of two lines that are
  actually skew in 3D — a subtle projection artifact that's easy to
  miss and easy for a student to be misled by. Fixed by requiring an
  explicit depth cue.

## The three worked problems

Source: [`3 vector figures problem latex code.tex`](<3%20vector%20figures%20problem%20latex%20code.tex>)

| Problem | Type | Verified answer |
|---|---|---|
| M26S1J21Q55 | Cross-product + plane-dot system | **27** |
| M26S1J21Q64 | Foot of perpendicular + projection | **18/7** |
| M26S2J21Q3 | Two skew lines forced parallel to a third direction | **290** |

<p float="left">
  <img src="generate%203D%20vector%20figures/output/M26S1J21Q55.jpeg" width="32%" alt="Generated figure for M26S1J21Q55" />
  <img src="generate%203D%20vector%20figures/output/M26S1J21Q64.jpeg" width="32%" alt="Generated figure for M26S1J21Q64" />
  <img src="generate%203D%20vector%20figures/output/M26S2J21Q3.jpeg" width="32%" alt="Generated figure for M26S2J21Q3" />
</p>

Every coordinate in these images was computed and verified by
`geometry_compiler` before any pixel was rendered — colors and line
styles are assigned deterministically by role (given data vs. computed
answer), never chosen by the image model.

## Repository layout

| Path | What it is |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | The shared data contract both solving stages build against |
| [`shared/schema.py`](shared/schema.py) | The `ProblemIR` / `SceneGraph` types — dependency-free, imported by everything |
| [`fixtures/`](fixtures/) | Hand-verified golden input/output pairs for all three problems |
| [`problem_parser/`](problem_parser/) | LaTeX text → Gemini API → `ProblemIR` (36 tests, incl. one live API call) |
| [`geometry_compiler/`](geometry_compiler/) | `ProblemIR` → exact symbolic solve → verified `SceneGraph` (5/5 tests) |
| [`generate 3D vector figures/`](<generate%203D%20vector%20figures/>) | `SceneGraph` → diagram, via Gemini's image model |

## Quickstart

```bash
pip install -r problem_parser/requirements.txt -r geometry_compiler/requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY

pytest geometry_compiler/tests/ problem_parser/tests/ -v
python3 "generate 3D vector figures/scripts/generate_figures.py"
```

## Status / what's next

Both solving stages are independently tested and verified; they are not
yet wired together into one live end-to-end call (parser output →
compiler input directly). A deterministic TikZ renderer — exact
projection instead of an image model's best guess — is the natural next
step for production-quality figures.
