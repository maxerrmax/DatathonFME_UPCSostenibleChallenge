"""nn_co2_full: com nn_co2 però confiant totalment en la xarxa (nn_mix=1.0, nn_mix_price=0.5).
Llegeix els npz de dificil/arena/nn_co2/."""
import importlib.util
from pathlib import Path

_f = Path(__file__).resolve().parents[1] / "nn_co2" / "agent.py"
_spec = importlib.util.spec_from_file_location("nn_co2_agent_impl", _f)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


class BatteryAgent(_m.BatteryAgent):
    DEFAULTS = {"nn_mix": 1.0, "nn_price": True, "nn_mix_price": 0.5}
