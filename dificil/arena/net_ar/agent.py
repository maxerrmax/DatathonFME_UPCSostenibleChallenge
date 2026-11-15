"""net_ar: persistencia a curt termini del residu de consum net sobre el v12.

A t es calcula e = net realitzat(t-1) - net previst a t-1 (prediccio crua, abans de cap desplacament d'aquest
modul). S'estima en linia l'autocorrelacio lag-1 rho dels errors (EW, alpha ar_alpha; rho = cov1/var, retallada a
[0, ar_rho_max]; rho=0 fins a ar_nmin mostres). Es desplaca el consum previst per a h=0..ar_hmax-1 en
ar_scale*rho^(h+1)*e (el solar no es toca).
Params: ar_hmax (6), ar_scale (1.0), ar_alpha (0.03), ar_nmin (12), ar_rho_max (0.9), ar_cap (600 kW).
Causal; si el timestamp salta, l'error previ s'oblida.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base
import numpy as np

Base = load_base().BatteryAgent


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        self._ar_t = None
        self._ar_pred = None
        self._ar_prev_e = None
        self._c0 = 0.0; self._c1 = 0.0; self._n = 0

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = super()._forecast(t, price3, co2_3)
        try:
            consec = self._ar_t is not None and self._ar_t == t - 1
            e = None
            if consec and self._ar_pred is not None:
                ld, pvr = self.hist.get("load", t - 1), self.hist.get("pv", t - 1)
                if np.isfinite(ld) and np.isfinite(pvr):
                    e = float(ld - pvr - self._ar_pred)
            if e is not None:
                a = self.p.get("ar_alpha", 0.03)
                if self._ar_prev_e is not None:
                    self._n += 1
                    k = max(a, 1.0 / self._n)
                    self._c1 += k * (e * self._ar_prev_e - self._c1)
                    self._c0 += k * (self._ar_prev_e ** 2 - self._c0)
                self._ar_prev_e = e
            else:
                self._ar_prev_e = None
            self._ar_pred = float(load[0] - pv[0])      # prediccio crua de net per a t
            self._ar_t = t
            if e is not None and self._n >= self.p.get("ar_nmin", 12) and self._c0 > 1e-9:
                rho = float(np.clip(self._c1 / self._c0, 0.0, self.p.get("ar_rho_max", 0.9)))
                cap = self.p.get("ar_cap", 600.0); sc = self.p.get("ar_scale", 1.0)
                for h in range(min(int(self.p.get("ar_hmax", 6)), len(load))):
                    s = float(np.clip(sc * rho ** (h + 1) * e, -cap, cap))
                    load[h] = max(0.0, load[h] + s)
        except Exception:
            pass
        return price, co2, load, pv
