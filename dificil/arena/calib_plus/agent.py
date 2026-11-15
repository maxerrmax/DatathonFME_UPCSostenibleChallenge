"""calib_plus: calibratge en linia millorat (CO2 i preu) sobre el v12.

Es desen les previsions CRUES (sense correccio) de les ultimes ~8 h. A l'hora t, cada previsio feta a t-j (j=1..6)
per a les hores t..t+2 (ara conegudes) amb horitzo >=3 dona una mostra d'error (conegut - previst). El biaix es
la mitjana exponencialment ponderada (per recencia, tau=calib_tau) de les mostres retallades a +-clip, i
s'aplica a h>=3 amb decaiment exp(-(h-3)/calib_decay). Opcionalment (calib_hdep) les mostres d'horitzo mes gran
pesen mes.
Params: calib_co2 (1), calib_price (1), co2_clip (0.06), price_clip (0.15), calib_tau (3.0), calib_decay (8.0),
        price_decay (8.0), calib_hdep (0), calib_nmax (6).
Causal: nomes usa valors coneguts a t. Sense mostres -> previsio crua de la base.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base
import numpy as np

Base = load_base().BatteryAgent


class _NoMemory(dict):
    def __setitem__(self, k, v):
        pass


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        self._co2_raw_forecasts = _NoMemory()       # desactiva la correccio v12
        self._raw = {}                               # t -> (price_raw, co2_raw)

    def _bias(self, key_idx, known, t, clip, hdep):
        nmax = int(self.p.get("calib_nmax", 6)); tau = float(self.p.get("calib_tau", 3.0))
        num = den = 0.0
        for j in range(1, nmax + 1):
            old = self._raw.get(t - j)
            if old is None:
                continue
            o = old[key_idx]
            for i in range(min(3, len(known))):
                hz = j + i
                if hz < 3 or hz >= len(o):
                    continue
                e = known[i] - o[hz]
                if not np.isfinite(e):
                    continue
                wgt = np.exp(-(j - 1) / tau)
                if hdep:
                    wgt *= 1.0 + 0.25 * (hz - 3)
                num += wgt * float(np.clip(e, -clip, clip)); den += wgt
        return num / den if den > 0 else 0.0

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = super()._forecast(t, price3, co2_3)
        try:
            raw_p, raw_c = price.copy(), co2.copy()
            hdep = int(self.p.get("calib_hdep", 0))
            h = np.arange(3, len(co2))
            if self.p.get("calib_co2", 1):
                b = self._bias(1, list(co2_3), t, self.p.get("co2_clip", 0.06), hdep)
                co2[3:] = np.clip(co2[3:] + b * np.exp(-(h - 3) / self.p.get("calib_decay", 8.0)), 0.0, 1.0)
            if self.p.get("calib_price", 1):
                b = self._bias(0, list(price3), t, self.p.get("price_clip", 0.15), hdep)
                price[3:] = np.maximum(price[3:] + b * np.exp(-(h - 3) / self.p.get("price_decay", 8.0)), 0.0)
            self._raw[t] = (raw_p, raw_c)
            for k in list(self._raw):
                if k < t - 8:
                    del self._raw[k]
        except Exception:
            pass
        return price, co2, load, pv
