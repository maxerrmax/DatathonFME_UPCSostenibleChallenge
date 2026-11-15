"""stoch_mpc + analog_co2 (els dos guanyadors de la primera ronda)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import combo

BatteryAgent = combo(["stoch_mpc", "analog_co2"], None)
