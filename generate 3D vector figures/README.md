# Generate 3D vector figures

The image-generation renderer stage: turns a verified `SceneGraph` (from
`geometry_compiler`) into a 2D depiction of a 3D vector-geometry figure,
using a Gemini image model (`MODEL` in `scripts/generate_figures.py`,
currently `gemini-3-pro-image`; `gemini-3.1-flash-image` / "Nano Banana 2"
also works and was used for earlier rounds).

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

## Known limitations

An image model does not do real geometric projection, so it can still
misplace things a true renderer wouldn't:

- **Skew lines can look like they cross.** The first pass on `M26S2J21Q3`
  (two skew lines) planted both intersection points at the 2D
  projection's false crossing point. Fixed by explicitly asking for a
  depth-break cue and forbidding C/D from sitting at the apparent
  crossing.
- **Prompt text can leak onto the image as if it were a label.** An
  earlier version described each entity's anchor point / direction
  vector in the same flowing sentence as its label, and the model
  sometimes echoed fragments of that instruction prose (raw coordinate
  tuples, even words like "given") onto the figure as extra text. Fixed
  by cleanly separating, per entity, the literal on-image label (quoted,
  verbatim) from everything else (which is explicitly marked "for your
  placement only -- not written on the image"), plus a closing
  whitelist rule naming every string the model is allowed to render.
- **Placement instructions themselves can leak.** Describing label
  placement with compass jargon ("the north-west edge") caused
  `gemini-3-pro-image` to literally write `"(NW edge)"` onto one figure.
  Fixed by rewording placement as plain phrases ("just above and to the
  left of") and adding those instruction words to the forbidden-text list.
- **Axis orientation can drift between figures.** Nothing pinned z to
  the vertical axis, so one regeneration swapped which axis was
  vertical -- correct in isolation, but visually inconsistent against
  the other two figures in the set. Fixed with an explicit fixed-
  orientation rule (z vertical, y horizontal-right, x receding to the
  lower-left) in every prompt.
- **Coordinate captions next to labels -- accepted, not fixed.** Despite
  an explicit text whitelist, the model very consistently adds a
  coordinate caption next to vector/point/line labels (e.g. `a
  (-1,2,2)`) beyond the plain letter the fixture specifies. This
  persisted across multiple prompt rewordings and both models tried,
  which suggests it's a strong prior from textbook-diagram training data
  rather than something prompt wording alone will reliably suppress. It
  is not factually wrong -- unlike the issues above, nothing it adds
  contradicts the underlying geometry -- so this was a deliberate
  decision to accept it rather than keep spending regenerations chasing
  strict whitelist compliance.

None of these fixes make the model an exact geometric renderer -- it's
still not guaranteed to get spatial relationships exactly right the way
a coordinate-driven renderer (e.g. generated TikZ) would. These are
mitigations for specific failure modes actually observed, not a
guarantee against new ones.
