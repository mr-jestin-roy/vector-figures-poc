# Generate 3D vector figures

The image-generation renderer stage: turns a verified `SceneGraph` (from
`geometry_compiler`) into a 2D depiction of a 3D vector-geometry figure,
using Gemini's image model (`gemini-3.1-flash-image`, "Nano Banana 2").

This is one possible consumer of the scene graph -- see `../CONTRACT.md`
for why the pipeline treats the geometry compiler, not the renderer, as
the mathematical authority. A TikZ-based renderer (deterministic, exact
projection) is the recommended production path; this folder is the
image-model alternative, useful as a fast PoC / comparison point.

## Layout

- `input/` -- local copies of the three golden `SceneGraph` fixtures this
  script was run against (see `input/README.md`).
- `scripts/generate_figures.py` -- builds one detailed prompt per scene
  graph (exact coordinates, `ROLE_STYLE` colors/line-styles, explicit
  label anchors, relation-driven cosmetic markers like right-angle ticks)
  and calls Gemini's image API.
- `output/` -- the generated images, one per problem.

## Usage

```bash
python3 "generate 3D vector figures/scripts/generate_figures.py"          # all fixtures in input/
python3 "generate 3D vector figures/scripts/generate_figures.py" M26S2J21Q3  # just one
```

Needs a Gemini API key -- see the project-root `.env` / `.env.example`
and `problem_parser/README.md`'s "Set your API key" section (same
resolution order is used here).

## Known limitation

An image model does not do real geometric projection, so it can still
misplace things a true renderer wouldn't -- e.g. the first pass on
`M26S2J21Q3` (two skew lines) planted both intersection points at the
2D projection's false crossing point. The prompt now explicitly asks for
a depth-break cue to avoid that, but the model is still not guaranteed
to get spatial relationships exactly right the way a coordinate-driven
renderer would.
