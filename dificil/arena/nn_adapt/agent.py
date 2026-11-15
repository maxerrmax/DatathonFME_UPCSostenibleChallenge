"""nn_adapt: com nn_pre, però el pes de la xarxa al CO2 s'adapta en línia (causal).

Cada hora es revela un valor nou de CO2 (t+2). Amb ell es puntuen les previsions que LightGBM (v12, ja calibrat) i
la xarxa van fer 3..23 h abans per a aquella hora; es manté l'error quadràtic amb decaïment exponencial (semivida
`ad_hl` hores) i el pes de la xarxa és l'invers de l'error relatiu: a = SL / (SL + SN), acotat a [ad_min, 1-ad_min].
Fins que hi ha prou dades, a = nn_mix (0,5). Preu: barreja fixa com a nn_pre.
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_strategy

_M = load_strategy("nn_co2")
_PRE = load_strategy("nn_pre").BatteryAgent
nf = _M.nf
Base = _M.Base


class BatteryAgent(_PRE):
    def __init__(self, params=None):
        super().__init__(params)
        self.p.setdefault("ad_hl", 72.0); self.p.setdefault("ad_min", 0.1)
        self._st = {}                    # origen t -> (base co2[0:24], nn co2[0:24])
        self._SL = 0.0; self._SN = 0.0; self._nup = 0
        self.a_log = self.p.get("_a_log")

    def _update(self, t):
        y_e = t + 2
        y = self.hist.get("co2", y_e)
        if not np.isfinite(y):
            return
        d = 0.5 ** (1.0 / (21.0 * float(self.p["ad_hl"])))
        for h in range(3, 24):
            s = y_e - h
            f = self._st.get(s)
            if f is None:
                continue
            eL = (f[0][h] - y) ** 2; eN = (f[1][h] - y) ** 2
            if np.isfinite(eL) and np.isfinite(eN):
                self._SL = d * self._SL + eL; self._SN = d * self._SN + eN; self._nup += 1
        for s in [s for s in self._st if s < t - 24]:
            del self._st[s]

    def _weight(self):
        if self._nup < 21 * 12 or self._SL + self._SN <= 0:
            return float(self.p["nn_mix"])
        a = self._SL / (self._SL + self._SN)
        m = float(self.p["ad_min"])
        return float(np.clip(a, m, 1 - m))

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = Base._forecast(self, t, price3, co2_3)
        if self._nn is None or len(co2_3) < 3 or len(price3) < 3:
            return price, co2, load, pv
        try:
            self._update(t)
            sl = slice(nf.H0, nf.H1 + 1)
            x = nf.build(self.hist, t, "co2")
            if x is not None:
                pr = self._nn.predict("co2", x, nf.last_known(self.hist, t, "co2"))
                if pr is not None and np.isfinite(pr).all():
                    nnf = co2.copy(); nnf[sl] = pr
                    self._st[t] = (co2.copy(), nnf)
                    a = self._weight()
                    if self.a_log is not None:
                        self.a_log[t] = a
                    co2 = co2.copy(); co2[sl] = np.clip(a * pr + (1 - a) * co2[sl], 0.0, 1.0)
            if self.p["nn_price"] and self.p["nn_mix_price"] > 0:
                x = nf.build(self.hist, t, "price")
                if x is not None:
                    pr = self._nn.predict("price", x, nf.last_known(self.hist, t, "price"))
                    if pr is not None and np.isfinite(pr).all():
                        b = float(self.p["nn_mix_price"])
                        price = price.copy(); price[sl] = b * pr + (1 - b) * price[sl]
        except Exception:
            pass
        return price, co2, load, pv
