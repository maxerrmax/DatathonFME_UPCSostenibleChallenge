"""vf_blend: mitjana de la primera acció del v12 (LP de 24 h amb previsions) i del vf_adp (valor après).
Dues famílies de decisió diferents: la mitjana és menys sensible a l'error de qualsevol de les dues."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_strategy

V = load_strategy("vf_adp")
Base = V.BatteryAgent


class BatteryAgent(Base):
    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        a = V.Base._lp(self, w, net, soc0, v_term, dmax0, scen)
        try:
            if self.vf is not None and len(w) >= 24 and abs(self.cap - 6000.0) < 1:
                b = self._vlp(w, net, soc0, dmax0, scen)
                if a is not None and b is not None and np.isfinite(b).all():
                    m = float(self.p.get("mix", 0.5))
                    out = np.array(a, float).copy(); out[0] = (1 - m) * a[0] + m * b[0]
                    return out
        except Exception:
            pass
        return a
