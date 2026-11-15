"""calib_hz: calibratge CO2 (i opcionalment preu) per horitzo, multi-origen, sobre el v12.

Es desen les previsions CRUES fetes a cada hora de les ultimes ~24 h (es desactiva la correccio del v12). Cada hora
t, les hores t..t+2 que acaben de quedar conegudes (nomes les noves, sense duplicar mostres) comparades amb la
previsio feta a l'origen s=T-h (h=3..23) donen una mostra d'error (conegut - previst) per a l'horitzo h.
Per horitzo es mante un biaix b_h mitjana exponencial (semivida hz_half mostres), encongit cap a 0 amb pes
n/(n+hz_n0) i retallat a +-hz_clip. Opcional (hz_hod) biaix separat per franja horaria (4x6 h) de l'hora objectiu,
barrejat 50/50 amb el global. S'aplica co2[h] += b_h per h>=3. Igual per al preu amb hz_price (clip hz_price_clip).
Params: hz_half (24), hz_n0 (12), hz_clip (0.06), hz_hod (False), hz_co2 (True), hz_price (False), hz_price_clip (0.15).
Causal; si el timestamp salta (nova setmana de test) s'oblida l'emmagatzematge d'origens (els biaixos es conserven).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base
import numpy as np

Base = load_base().BatteryAgent
HMIN, HMAX = 3, 23


class _NoMemory(dict):
    def __setitem__(self, k, v):
        pass


class _HzBias:
    """Biaix per horitzo (global i per franja horaria) amb mitjana exponencial i encongiment."""
    def __init__(self, half, n0, clip, hod):
        self.a = 1.0 - 0.5 ** (1.0 / max(half, 1e-6)); self.n0 = float(n0); self.clip = float(clip); self.hod = hod
        self.g = np.zeros(HMAX + 1); self.gn = np.zeros(HMAX + 1)
        self.b = np.zeros((4, HMAX + 1)); self.bn = np.zeros((4, HMAX + 1))

    def _upd(self, m, n, i, e):
        n[i] += 1.0
        m[i] += max(self.a, 1.0 / n[i]) * (e - m[i])

    def update(self, target, h, e):
        e = float(np.clip(e, -3 * self.clip, 3 * self.clip))
        self._upd(self.g, self.gn, h, e)
        if self.hod:
            k = int(target % 24) // 6
            self._upd(self.b[k], self.bn[k], h, e)

    def get(self, target, h):
        v = self.g[h] * self.gn[h] / (self.gn[h] + self.n0)
        if self.hod:
            k = int(target % 24) // 6
            vb = self.b[k][h] * self.bn[k][h] / (self.bn[k][h] + self.n0)
            v = 0.5 * v + 0.5 * vb
        return float(np.clip(v, -self.clip, self.clip))


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        self._co2_raw_forecasts = _NoMemory()       # desactiva la correccio v12
        self._raw = {}                               # s -> (price_raw, co2_raw)
        self._hz_last_t = None
        self._hz_last_target = None
        hod = bool(self.p.get("hz_hod", False)); half = self.p.get("hz_half", 24); n0 = self.p.get("hz_n0", 12)
        self._hz_co2 = _HzBias(half, n0, self.p.get("hz_clip", 0.06), hod)
        self._hz_pr = _HzBias(half, n0, self.p.get("hz_price_clip", 0.15), hod)

    def _learn(self, bias, idx, known, t, last_target):
        for T in range(max(last_target + 1, t), t + len(known)):
            truth = known[T - t]
            if not np.isfinite(truth):
                continue
            for h in range(HMIN, HMAX + 1):
                old = self._raw.get(T - h)
                if old is None:
                    continue
                o = old[idx]
                if h >= len(o) or not np.isfinite(o[h]):
                    continue
                bias.update(T, h, truth - o[h])

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = super()._forecast(t, price3, co2_3)
        try:
            if self._hz_last_t is None or t != self._hz_last_t + 1:
                self._raw = {}                       # salt de timestamp: origens antics no valen
                self._hz_last_target = t - 1
            raw_p, raw_c = price.copy(), co2.copy()
            use_c, use_p = self.p.get("hz_co2", True), self.p.get("hz_price", False)
            lt = self._hz_last_target
            if use_c:
                self._learn(self._hz_co2, 1, list(co2_3), t, lt)
            if use_p:
                self._learn(self._hz_pr, 0, list(price3), t, lt)
            self._hz_last_target = max(lt, t + max(len(co2_3), len(price3)) - 1)
            if use_c:
                for h in range(HMIN, min(len(co2), HMAX + 1)):
                    co2[h] = min(1.0, max(0.0, co2[h] + self._hz_co2.get(t + h, h)))
            if use_p:
                for h in range(HMIN, min(len(price), HMAX + 1)):
                    price[h] = max(0.0, price[h] + self._hz_pr.get(t + h, h))
            self._raw[t] = (raw_p, raw_c)
            for k in list(self._raw):
                if k < t - 24:
                    del self._raw[k]
            self._hz_last_t = t
        except Exception:
            pass
        return price, co2, load, pv
