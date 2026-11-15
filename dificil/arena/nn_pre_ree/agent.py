"""nn_pre amb LightGBM de preu/CO2 re-entrenats amb historial REE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import combo

BatteryAgent = combo(["v12_ree", "nn_pre"], None)
