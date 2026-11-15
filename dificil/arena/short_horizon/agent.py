"""Planifica només 12 h endavant (horizon 12)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent
OVERRIDE = {"horizon": 12}


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__({**OVERRIDE, **(params or {})})
