"""stoch_mpc mantenint la prudència futura del v12 (k_future 0,4)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import combo

BatteryAgent = combo(["stoch_mpc"], {"k_future": 0.4})
