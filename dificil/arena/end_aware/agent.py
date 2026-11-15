"""end_aware (mixin): el test acaba el 31/8/2026 23:00 (l'última setmana de test són els 7 últims dies d'agost).
Quan l'horitzó de planificació hi arriba, es planifica només fins al final i l'energia restant val 0 (abans l'agent
només ho notava 2 h abans, quan s'escurça la finestra de mercat). Si l'ordre real fos un altre, només descarregaria
abans d'hora un vespre. Combinable: combo(["end_aware", "<base>"])."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent
END = int(np.datetime64("2026-08-31T23", "h").astype(np.int64))   # última hora del test (hores des d'epoch)


class BatteryAgent(Base):
    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        try:
            t = self.t_prev
            end = int(self.p.get("end_epoch", END))
            if t is not None and t <= end:
                R = end - t + 1                                     # hores que queden, inclosa l'actual
                if R < len(w):
                    w = np.asarray(w)[:R]; net = np.asarray(net)[:R]; v_term = 0.0
                    fc = getattr(self, "_fc", None)                 # previsions de l'MPC estocàstic
                    if fc is not None:
                        self._fc = tuple(np.asarray(a)[:R] for a in fc)
        except Exception:
            pass
        return super()._lp(w, net, soc0, v_term, dmax0, scen)
