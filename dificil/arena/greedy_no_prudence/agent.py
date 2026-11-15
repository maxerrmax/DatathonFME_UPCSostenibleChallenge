"""greedy_no_prudence: v12 sense cap marge de seguretat (k_dis=0, scen=0, k_future=0, q_term=0.5).
Opcionalment el valor terminal del LP es barreja amb la w mitjana de les mateixes 24 h d'ahir
(param yday_mix, per defecte 0.5): mes mirada endavant. Mesura quant valen els marges de prudencia.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

_B = load_base()
Base = _B.BatteryAgent
ETA = _B.ETA


class BatteryAgent(Base):
    def __init__(self, params=None):
        p = {"k_dis": 0.0, "scen": 0, "k_future": 0.0, "q_term": 0.5, "yday_mix": 0.5}
        p.update(params or {})
        super().__init__(p)

    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        try:
            m = float(self.p.get("yday_mix", 0.5))
            if m > 0 and v_term > 0 and len(self.hist_w) >= 24:
                vy = self.p["term_scale"] * ETA * float(np.mean(self.hist_w[-24:]))
                if np.isfinite(vy):
                    v_term = (1 - m) * v_term + m * vy
        except Exception:
            pass
        return super()._lp(w, net, soc0, v_term, dmax0, scen)
