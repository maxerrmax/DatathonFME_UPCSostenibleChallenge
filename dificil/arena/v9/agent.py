"""Referència: v9 = v12 sense el calibratge en línia del CO2 (públic 23,45)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent


class _NoMemory(dict):
    """Diccionari que mai recorda res: el v12 no troba cap previsió antiga i no corregeix el CO2."""
    def __setitem__(self, k, v):
        pass


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        self._co2_raw_forecasts = _NoMemory()
