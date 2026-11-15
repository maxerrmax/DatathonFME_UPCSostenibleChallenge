"""Previsió solar = persistència de cel clar pura (pv_blend 0,0)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent
OVERRIDE = {"pv_blend": 0.0}


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__({**OVERRIDE, **(params or {})})
