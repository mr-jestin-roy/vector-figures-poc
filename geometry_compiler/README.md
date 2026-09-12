# geometry_compiler

`ProblemIR -> SceneGraph`, the second half of the pipeline described in
[`../CONTRACT.md`](../CONTRACT.md). Solves every equation in
`ProblemIR.equations` with exact rational/symbolic arithmetic (never
floats), independently re-verifies each equation by substituting the
solved values back in, and evaluates `query_expr` to produce
`final_answer`.

```python
from shared.schema import problem_ir_from_dict
from geometry_compiler import compile  # == geometry_compiler.compiler.compile_problem_ir

scene_graph = compile(problem_ir)
```

## Layout

| file | responsibility |
|---|---|
| `expr_parser.py` | Tokenizer + hand-rolled recursive-descent parser for the mini-language. No `eval()` anywhere. Produces a tiny AST (`Num`/`Ident`/`Call`/`BinOp`/`UnaryOp`). |
| `evaluator.py` | Evaluates that AST against a locked-down `Environment` (entity id -> sympy value). Implements `dot`, `cross`, `norm` (with exact-perfect-square sqrt collapsing), `A()`/`d()`, and `+ - * / **`. |
| `context.py` | Builds an `Environment` from a `ProblemIR`: either *symbolic* (unknown entities/scalars become fresh sympy symbols, for solving) or *fully numeric* (for re-verification and `query_expr`). Also the `Vec3 <-> sympy` conversions and the `sympy value -> exact Fraction` conversion. |
| `solver.py` | Builds the residual system from every equation string and calls `sympy.solve`. Raises a specific `GeometryCompilerError` subclass for "no solution" / "underdetermined" / "ambiguous" rather than guessing. |
| `verifier.py` | The **independent** re-check: re-parses every equation string from scratch and substitutes a fully-numeric environment, requiring an exact-zero residual. Deliberately does not reuse the solver's own residual expressions -- see the module docstring for why. |
| `compiler.py` | Orchestrates the above into `compile_problem_ir(problem_ir) -> SceneGraph`. |
| `exceptions.py` | The `GeometryCompilerError` hierarchy. |
| `tests/` | `test_fixtures.py` (all three golden fixtures), `test_inconsistent.py` (deliberately-broken input). |

## How generic is the solver, really?

**Fully generic, not `problem_type`-keyed.** `solver.py` never branches
on `ProblemIR.problem_type`. For every entity with `status == "unknown"`
it allocates 3 fresh sympy symbols (a point/vector) or 6 (an anchor +
direction, for a hypothetically-unknown `parametric_line` -- untested,
since no fixture needs it, but wired up for the "generalize to a fourth
similarly-shaped problem" goal), plus one symbol per name in
`unknown_scalars`. It then evaluates every string in
`ProblemIR.equations` against that fully-symbolic environment to
produce a flat list of scalar residual expressions (3 per vector
equation, 1 per scalar equation), and hands the whole list to
`sympy.solve(residuals, unknowns, dict=True)`.

This one code path is what solves all three fixture families:

- **`M26S1J21Q55`** (cross-product + plane-dot system): 3 unknowns
  (`c`'s components), 4 residuals (3 from `cross(a,c)==b`, 1 from
  `dot(c,n)==4`) -- over-determined (the cross-product residual is
  rank-2 for a fixed `a`) but consistent.
- **`M26S1J21Q64`** (foot-of-perpendicular + projection): 4 unknowns
  (`F`'s components + `lambda`), 4 residuals (3 from
  `F==A(L)+lambda*d(L)`, 1 from `dot(P-F,d(L))==0`) -- exactly determined.
- **`M26S2J21Q3`** (two lines + third-direction constraint): 8 unknowns
  (`C`, `D`, `t`, `s`), 9 residuals (3+3 from the two line equations, 3
  from `cross(D-C,n3)==zero`, one of which is dependent) -- consistent
  and exactly determined once the dependency is accounted for.

`sympy.solve` handles all three uniformly because in every one of these
families, **an unknown is never multiplied by another unknown** --
`cross`/`dot` always have at least one fully-known operand, and
`lambda * d(L)` / `t * d(L1)` / `s * d(L2)` multiply a scalar unknown by
a *constant* vector. So every one of these systems happens to be
exactly linear in the unknowns, and `sympy.solve` resolves them via
ordinary linear elimination. I used the general `sympy.solve` (not
`linsolve`) specifically so that a fourth problem with a genuinely
nonlinear constraint (e.g. two unknown vectors dotted together) would
still get a real attempt via sympy's polynomial-system machinery,
rather than hitting a hard `linsolve`-only ceiling -- but I have not
constructed or tested such a nonlinear fixture, so treat "handles
nonlinear systems too" as an untested aspiration of the design, not a
verified claim. **No part of the solve itself is special-cased per
problem.**

What *is* a fixed, non-generic assumption baked into `solver.py`
(documented, not hidden): the solved system must have **exactly one**
discrete solution with **no free parameters** left over.
`UnderdeterminedSystemError` / `AmbiguousSystemError` are raised
otherwise. This matches all three fixtures (each is a fully-determined
"Find the missing point/vector" problem) but would need extending if a
future problem legitimately has a one-parameter family of valid
answers.

## Independent re-verification (not just trusting the solver)

`verifier.py` does not reuse any residual expression built inside
`solver.py`. After solving, it builds a brand-new, fully-numeric
`Environment` (§`context.build_numeric_context`) and **re-parses each
equation string from `ProblemIR.equations` from scratch**, evaluates
LHS and RHS independently, and computes an exact residual (squared
Euclidean distance for a vector equation, squared difference for a
scalar one -- squaring keeps a single scalar `Fraction` per
`VerificationResult` while staying an unambiguous "is it exactly zero"
test). `SceneGraph.verified` is `True` iff *every* one of those
residuals is exactly `Fraction(0)`.

This matters because `sympy.solve`, given an over-determined system
(e.g. `M26S1J21Q55`'s 4 residuals for 3 unknowns), could in principle
use a subset internally; re-deriving every check independently means an
equation the solver didn't strictly need to reach *a* solution can still
fail the compiler's own verification, rather than being silently
trusted. In practice, for all three fixtures, `sympy.solve` is itself
consistent-or-empty on the full linear system (it does not silently
drop a contradictory row), so this mostly shows up as
`SystemInconsistentError` at the solve stage rather than
`verified=False` after a "successful" solve -- see `tests/test_inconsistent.py`
for exactly which failure mode a corrupted input triggers, and why.

## Exact-sqrt handling (`M26S1J21Q64`)

`M26S1J21Q64`'s `query_expr` is `dot(F, v) / norm(v)`, and
`norm((6,2,3)) = sqrt(49) = 7` -- exact only because 49 happens to be a
perfect square. `evaluator.exact_sqrt` special-cases this: it takes the
radicand (always an exact sympy `Rational` once its vector argument is
fully numeric), reduces it to a Python `Fraction`, and checks whether
both the numerator and denominator are perfect squares via
`math.isqrt`. If so it returns an exact `sympy.Rational`; final answer
comes out as the exact `18/7`, never a float `2.571...` or a lingering
symbolic `sqrt(49)/1` (which sympy would actually auto-simplify to `7`
on its own for this particular case, but the explicit perfect-square
check also covers ratios like `sqrt(9/25) = 3/5` that sympy's default
`sqrt()` would leave as `3/5` too, and more importantly makes the
"is this exact" check explicit and intentional rather than incidental).

**Known limitation:** if a future problem's final answer is genuinely
irrational (norm of a non-perfect-square, e.g. `sqrt(2)`), this raises
`IrrationalResultError` rather than silently coercing to a float or
leaving a symbolic `sqrt` masquerading as a `Rational` string. None of
the three fixtures hit this case.

## Where CONTRACT.md's grammar section was ambiguous or insufficient

Both of these are things I *did* implement (the fixtures require them
to compile at all), rather than things I worked around quietly --
flagging them here per the task brief:

1. **Bare non-zero numeric literals.** CONTRACT.md's grammar bullet
   says the reserved constants `zero` and `0` are "**the only literals
   allowed**"; every other constant must be a declared `Entity`. But
   `fixtures/problem_ir/M26S1J21Q55.json`'s own equations list contains
   `"dot(c, n) == 4"` -- a bare `4`, not a declared entity. I implemented
   the tokenizer/parser to accept arbitrary non-negative integer
   literals (not just `0`), since otherwise this fixture is literally
   unparseable under the contract's stated grammar. `shared/schema.py`
   and `CONTRACT.md` are frozen for this task, so I did not edit either
   -- this is a genuine inconsistency between the prose spec and the
   ground-truth fixture, not a defect in my code.
2. **Division (`/`).** The grammar bullet lists `+ - *` (vector-vector
   and scalar-vector) and `**` (power), but no division operator. Yet
   `fixtures/problem_ir/M26S1J21Q64.json`'s `query_expr` is
   `"dot(F, v) / norm(v)"`, which needs scalar `/` scalar division to
   parse at all. I added `/` (scalar/scalar, and vector/scalar for
   symmetry, though no fixture needs the latter) as a grammar
   extension. `**` itself, meanwhile, is never actually used by any of
   the three fixtures' `equations` or `query_expr` -- it's implemented
   (right-associative, scalar-only) but exercised only by a manual
   sanity check, not by fixture-driven tests.

Neither gap required a judgment call about problem *semantics* -- both
literal `4` and the `/` in `dot(F,v)/norm(v)` have an unambiguous
reading -- so I implemented the evident intent rather than rejecting
valid fixture input on a technicality.

## Deliberately-inconsistent input (`tests/test_inconsistent.py`)

The test mutates a deep copy of `M26S1J21Q55`'s `b` entity (the RHS of
`cross(a, c) == b`) so that `dot(a, b) != 0`. Since `cross(a, c)` is
*always* perpendicular to `a` for any `c`, this makes the equation
unsatisfiable for **any** value of `c` at all, independent of the
second equation (`dot(c, n) == 4`) -- a clean, unconditional
inconsistency, as opposed to just picking a different (but still
consistent) target value. `compile()` raises `SystemInconsistentError`
(a `GeometryCompilerError` subclass) with a message naming the
equations and unknowns involved, rather than crashing with a bare
traceback or returning a confident-looking wrong answer. See that
file's module docstring for why mutating the *other* equation's RHS
(the literal `4`) would **not** have produced a genuinely inconsistent
system for this constraint family (it would just change the unique
answer) -- and would have been the wrong test to write.

## Running the tests

```bash
cd geometry_compiler   # or the repo root -- conftest.py adds the repo
                        # root to sys.path either way
python3 -m pip install -r requirements.txt
python3 -m pytest tests/ -v
```

## Other limitations / non-goals

- `norm(...)` (and any equation containing a `sqrt`) appearing *inside*
  `ProblemIR.equations` for an entity that is still partly symbolic at
  parse time (as opposed to appearing only in `query_expr`, after
  everything is already solved) is not exercised by any fixture and is
  not specifically hardened -- `sympy.solve` would be asked to solve a
  system containing a literal `sympy.sqrt(...)` of an unknown
  expression, which can work for simple cases but is not something this
  package specifically tests or guarantees.
- Parametric lines with `status == "unknown"` are wired up in
  `context.py` (6 symbols: 3 for the anchor, 3 for the direction) but
  untested -- no fixture has one.
- `sympy.solve`'s general polynomial-system solving path (for a
  hypothetical nonlinear fourth problem) is unexercised; only the
  linear path that all three real fixtures hit is verified.
