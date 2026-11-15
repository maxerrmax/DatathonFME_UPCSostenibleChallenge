"""vf_mix (mixin): mitjana de la primera acció de la base i la del valor après (vf_adp).
Combinable: combo(["vf_mix", "<base>"]); param mix (0,5)."""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base, load_strategy

Base = load_base().BatteryAgent
V = load_strategy("vf_adp")
HERE_VF = Path(V.__file__).resolve().parent


class BatteryAgent(Base):
    _lam = V.BatteryAgent._lam
    _vlp = V.BatteryAgent._vlp

    def __init__(self, params=None):
        super().__init__(params)
        self.vf = None; self.lam_log = None
        try:
            import lightgbm as lgb
            m = str(self.p.get("models", "models_all7.pkl"))
            name = "vf_all.pkl" if "all" in m else "vf_" + m.replace("models_", "").replace("7.pkl", ".pkl")
            self.vf = lgb.Booster(model_str=pickle.load(open(HERE_VF / name, "rb"))["vf"])
        except Exception:
            self.vf = None

    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        a = super()._lp(w, net, soc0, v_term, dmax0, scen)
        try:
            if a is not None and self.vf is not None and len(w) >= 24 and abs(self.cap - 6000.0) < 1:
                b = self._vlp(w, net, soc0, dmax0, scen)
                if b is not None and np.isfinite(b).all():
                    m = float(self.p.get("mix", 0.5))
                    out = np.array(a, float).copy(); out[0] = (1 - m) * a[0] + m * b[0]
                    return out
        except Exception:
            pass
        return a
