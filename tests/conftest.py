import sys
from pathlib import Path

# Ensure repository root is on sys.path for package imports (ops, data_layer, etc.)
ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
