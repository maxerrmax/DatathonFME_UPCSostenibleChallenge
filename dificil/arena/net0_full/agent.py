"""Confia més en el model directe de la càrrega neta de l'hora actual (net0_mix 0,8)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent
OVERRIDE = {"net0_mix": 0.8}


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__({**OVERRIDE, **(params or {})})
