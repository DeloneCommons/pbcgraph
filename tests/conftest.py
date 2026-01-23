import sys
from pathlib import Path

# Ensure src/ is importable when running tests without an installed wheel.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
