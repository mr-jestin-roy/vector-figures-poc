# Input

Local copies of `fixtures/scene_graph/*.json` (the geometry_compiler's
verified, golden output) -- kept here so this feature's input, script,
and output all live together under one folder.

These are copies, not the source of truth: `fixtures/scene_graph/` at
the project root is still what `geometry_compiler`'s own tests are
graded against (see `../../CONTRACT.md`). If a scene graph fixture
changes there, re-copy it here before regenerating figures:

```bash
cp ../../fixtures/scene_graph/*.json .
```
