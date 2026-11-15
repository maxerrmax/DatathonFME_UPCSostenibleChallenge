"""cautious: v12 amb mes prudencia: k_future=0.7, scen_scale=1.3 (sobreescrivibles via params) i reserva de SOC.

Reserva: no es descarrega per sota de reserve_soc (0.10 del cap) tret que el pes w de l'hora actual estigui al
top 20% (>= quantil reserve_q=0.8) de la finestra de 24 h prevista. Post-processat de l'accio.
Params: k_future, scen_scale, reserve_soc (0.10), reserve_q (0.8).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base
import numpy as np

Base = load_base().BatteryAgent
ETA = 0.9 ** 0.5


class BatteryAgent(Base):
    def __init__(self, params=None):
        p = {"k_future": 0.7, "scen_scale": 1.3}
        p.update(params or {})
        super().__init__(p)
        self._w = None

    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        try:
            self._w = np.asarray(w, float).copy()
        except Exception:
            self._w = None
        return super()._lp(w, net, soc0, v_term, dmax0, scen)

    def _act(self, obs):
        self._w = None
        a = super()._act(obs)
        try:
            if a < 0 and self._w is not None and len(self._w) > 1:
                soc = min(self.cap, max(0.0, float(obs.get("soc_kwh"))))
                floor = self.p.get("reserve_soc", 0.10) * self.cap
                if self._w[0] < np.quantile(self._w, self.p.get("reserve_q", 0.8)):
                    a = max(a, -max(0.0, (soc - floor) * ETA))
                    a = self._clamp(soc, a)
        except Exception:
            pass
        return float(a)
