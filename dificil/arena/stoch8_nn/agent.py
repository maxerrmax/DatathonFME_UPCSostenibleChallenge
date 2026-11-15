"""stoch_mpc (8, més ràpid) + xarxa neuronal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import combo

BatteryAgent = combo(["stoch_mpc", "nn_co2"], None)
