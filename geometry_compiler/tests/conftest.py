import pathlib
import sys

# Make both `shared` (repo_root/shared/schema.py) and `geometry_compiler`
# importable regardless of where pytest is invoked from.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIXTURES_DIR = _REPO_ROOT / "fixtures"
PROBLEM_IR_DIR = FIXTURES_DIR / "problem_ir"
SCENE_GRAPH_DIR = FIXTURES_DIR / "scene_graph"

FIXTURE_IDS = ["M26S1J21Q55", "M26S1J21Q64", "M26S2J21Q3"]
