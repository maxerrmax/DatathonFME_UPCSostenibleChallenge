"""Pesa més el preu (ratio 0,80): el nostre estalvi de cost va per darrere dels líders."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent
OVERRIDE = {"ratio": 0.80}


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__({**OVERRIDE, **(params or {})})
