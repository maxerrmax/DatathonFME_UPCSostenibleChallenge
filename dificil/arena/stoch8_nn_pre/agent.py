"""stoch_mpc (8) + xarxes pre-entrenades amb historial REE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import combo

BatteryAgent = combo(["stoch_mpc", "nn_pre"], None)
