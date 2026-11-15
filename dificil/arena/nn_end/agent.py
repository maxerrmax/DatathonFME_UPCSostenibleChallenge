"""Combinació: end_aware,nn_pre."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import combo

BatteryAgent = combo(["end_aware", "nn_pre"], None)
