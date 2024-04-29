import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
GEN_DIR = os.path.join(ROOT_DIR, "gen")
