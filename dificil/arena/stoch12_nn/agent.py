"""stoch_mpc (12 escenaris) + previsions de xarxa neuronal (GPU) de CO2/preu."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import combo

BatteryAgent = combo(["stoch_mpc", "nn_co2"], {"S": 12})
