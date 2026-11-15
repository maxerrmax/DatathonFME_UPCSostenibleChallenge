"""Combinació: end_aware,v12."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import combo

BatteryAgent = combo(["end_aware", "v12"], None)
