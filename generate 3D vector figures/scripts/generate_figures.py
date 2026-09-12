"""Renders each fixture's SceneGraph into a 2D depiction of a 3D vector
figure, using Gemini's image-generation model (gemini-3.1-flash-image,
"Nano Banana 2").

This is the "renderer" stage of the pipeline described in CONTRACT.md --
consuming the already-solved, already-verified SceneGraph JSON so the
image model never has to do any geometry itself, only follow explicit
coordinates and a fixed visual-encoding convention.

Usage:
    python3 "generate 3D vector figures/scripts/generate_figures.py"

Reads GEMINI_API_KEY / GEMINI_JEE_LATEX_API_KEY / GOOGLE_API_KEY from a
real env var or the project-root .env (same resolution order as
problem_parser.gemini_client). Reads scene graphs from the sibling
input/ directory (a local copy of fixtures/scene_graph/*.json, kept
in sync manually -- see input/README.md) and writes one image per
fixture into the sibling output/ directory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FEATURE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = FEATURE_ROOT.parent
FIXTURES_DIR = FEATURE_ROOT / "input"
OUT_DIR = FEATURE_ROOT / "output"
MODEL = "gemini-3.1-flash-image"

API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GEMINI_JEE_LATEX_API_KEY", "GOOGLE_API_KEY")

ROLE_STYLE = {
    "given_external_point": {"color": "red", "line_style": "none"},
    "given_line": {"color": "navy", "line_style": "solid"},
    "given_vector": {"color": "navy", "line_style": "dashed"},
    "computed_point": {"color": "darkgreen", "line_style": "none"},
    "computed_line": {"color": "darkgreen", "line_style": "solid"},
    "computed_vector": {"color": "darkgreen", "line_style": "dotted"},
}


def resolve_api_key() -> str:
    try:
        from dotenv import load_dotenv

        env_path = PROJECT_ROOT / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit(
        "No Gemini API key found. Set one of "
        f"{', '.join(API_KEY_ENV_VARS)} as a real env var or in the "
        "project-root .env file."
    )


def fmt_vec(v: dict) -> str:
    return f"({v['x']}, {v['y']}, {v['z']})"


def describe_entity(e: dict) -> str:
    style = ROLE_STYLE[e["visual_role"]]
    label = e["label"]
    lines = [
        f"- id `{e['id']}` ({e['kind']}, status={e['status']}, role={e['visual_role']})",
        f"  color: {style['color']}, line style: {style['line_style']}",
        f"  label: \"{label['text']}\" anchored at the {label['anchor']} edge of its reference point/line "
        f"(the label text must not overlap the point or line it names)",
    ]
    if e["kind"] == "point":
        lines.append(f"  coordinates: {fmt_vec(e['coordinates'])}")
    elif e["kind"] == "parametric_line":
        lines.append(
            f"  passes through anchor point {fmt_vec(e['anchor'])} with direction vector "
            f"{fmt_vec(e['direction'])} (parameter name: {e.get('param_name') or 't'})"
        )
        lines.append(
            "  draw this as a full straight line (extending a bit past both the anchor "
            "point and any other labelled point on it), not just a segment between two dots"
        )
    elif e["kind"] == "vector":
        lines.append(
            f"  components {fmt_vec(e['components'])}, drawn as an arrow from the origin "
            "(or from wherever makes the figure clearest, but its direction and length must "
            "match these exact components)"
        )
    return "\n".join(lines)


def describe_relation(r: dict, entities_by_id: dict) -> str:
    kind = r["kind"]
    if kind == "lies_on":
        return f"- draw a small tick/dot showing `{r['subject']}` sitting exactly on the line `{r['object']}`"
    if kind == "perpendicular_to":
        seg = f"the segment from `{r['via']}` to `{r['subject']}`" if r.get("via") else f"`{r['subject']}`"
        return (
            f"- {seg} must be visually perpendicular to `{r['object']}`; mark the right angle "
            "with a small square/right-angle tick where they meet"
        )
    if kind == "parallel_to":
        if r.get("via"):
            a, b = r["via"], r["subject"]
            return (
                f"- draw an actual, visible line segment directly connecting `{a}` to `{b}` "
                f"(this segment IS the line referred to as `{r['object']}`'s direction -- do not "
                f"draw `{r['object']}`'s direction as a separate floating arrow disconnected from "
                f"`{a}` and `{b}`). This segment must be visually parallel to the given direction "
                f"labelled `{r['object']}`.\n"
                f"  IMPORTANT: the two lines that `{a}` and `{b}` each sit on are skew lines in 3D "
                f"-- they do NOT actually intersect. If your 2D oblique projection makes them appear "
                f"to cross, draw `{a}` and `{b}` at their own given coordinates, clearly NOT at that "
                f"apparent crossing point, and use a visible depth cue (e.g. a small gap/break in "
                f"whichever line passes behind) so it's clear the two lines pass each other in space "
                f"rather than meeting."
            )
        return f"- `{r['subject']}` must be drawn visually parallel to the direction of `{r['object']}`"
    if kind == "projects_onto":
        return (
            f"- show a thin dashed drop-line from `{r['subject']}` onto the direction of "
            f"`{r['object']}`, illustrating the projection being asked for"
        )
    return f"- {kind}: {r['subject']} -> {r['object']}"


def build_prompt(scene_graph: dict) -> str:
    entities_by_id = {e["id"]: e for e in scene_graph["entities"]}
    entity_lines = "\n".join(describe_entity(e) for e in scene_graph["entities"])
    relation_lines = "\n".join(
        describe_relation(r, entities_by_id) for r in scene_graph.get("relations", [])
    ) or "(no incidence relations for this problem -- it is a pure algebraic system; do not invent any)"

    return f"""Create a clean, textbook-style 2D diagram depicting a 3D vector-geometry
figure (problem {scene_graph['problem_id']}), the way it would appear as a
hand-drawn illustration in a math textbook -- NOT a photorealistic 3D render.

Use an oblique 3D coordinate system: draw x, y, z axes as three lines meeting
at a labelled origin, with the axes clearly but unobtrusively visible in the
background (thin, gray, behind the geometric entities).

Every coordinate below has already been computed and verified exactly by a
symbolic solver -- do not alter, round, or re-derive any of them. Plot the
entities using these EXACT values and this EXACT visual encoding. This is
the whole point of the figure: a student should be able to read off which
elements are given data and which is the answer being solved for, at a
glance, without reading the original word problem.

ENTITIES:
{entity_lines}

GEOMETRIC RELATIONSHIPS TO SHOW:
{relation_lines}

HARD RULES (violating any of these makes the figure wrong, not just less
polished):
1. Color and line style are BOTH significant and must be used TOGETHER --
   never rely on color alone or line style alone, and never rely only on
   the angle two lines/vectors meet at to distinguish them (a shallow angle
   in 3D-on-2D can look like the same line continuing).
2. Every label needs an explicit anchor edge (as specified per entity above)
   -- position each label so it never overlaps the point, line, or vector it
   names.
3. Labels and all mathematical notation should be a normal, legible text
   size matching clean textbook typography -- not tiny, not oversized.
4. Do not add any point, line, vector, plane, or grid that is not listed
   above. No decorative elements, no shading, no photorealistic lighting.
5. Keep the background plain/white so the figure reads clearly as a
   reference diagram, not an illustration.

Render as a single static image, roughly square, high contrast, suitable
for direct inclusion in a printed math worksheet."""


def main() -> None:
    api_key = resolve_api_key()
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise SystemExit("Install the SDK first: pip install google-genai")

    client = genai.Client(api_key=api_key)
    OUT_DIR.mkdir(exist_ok=True)

    only = set(sys.argv[1:]) or None
    fixture_paths = sorted(FIXTURES_DIR.glob("*.json"))
    if only:
        fixture_paths = [p for p in fixture_paths if p.stem in only]
    if not fixture_paths:
        raise SystemExit(f"No scene graph fixtures found under {FIXTURES_DIR} matching {only}")

    for path in fixture_paths:
        scene_graph = json.loads(path.read_text())
        problem_id = scene_graph["problem_id"]
        prompt = build_prompt(scene_graph)
        print(f"--- Generating figure for {problem_id} ---")

        config = types.GenerateContentConfig(response_modalities=["IMAGE"])
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=config,
        )

        candidates = response.candidates or []
        saved = False
        for candidate in candidates:
            parts = candidate.content.parts if candidate.content else []
            for part in parts:
                if part.inline_data and part.inline_data.data:
                    mime = part.inline_data.mime_type or "image/png"
                    ext = "png" if "png" in mime else mime.split("/")[-1]
                    out_path = OUT_DIR / f"{problem_id}.{ext}"
                    out_path.write_bytes(part.inline_data.data)
                    print(f"  saved {out_path}")
                    saved = True
        if not saved:
            print(f"  WARNING: no image data returned for {problem_id}. "
                  f"Full response: {response!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
