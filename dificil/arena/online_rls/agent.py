"""online_rls: correccio lineal en linia per horitzo (h=3..23) de la previsio de CO2 (i opcionalment preu).

Per cada horitzo h hi ha un model RLS (oblit `rls_lambda`=0.995) del residu y - f amb regressors
[1, f, ultim_conegut, sin(2pi hod/24), cos(2pi hod/24)] (valors de CO2/preu escalats x10), on f es la previsio crua
del v12 feta h hores abans i y el valor realitzat (es coneix quan entra a la finestra de 3 h: a l'hora t es coneix
t+2, aixi que cada pas aporta una parella nova per horitzo). La correccio s'encongeix cap a la identitat fins
a `rls_min` (48) actualitzacions: corr * min(1, n/rls_min). El correcor propi de v12 es desactiva (_NoMemory).
Params: rls_lambda, rls_min, rls_price (False), rls_p0, rls_max (limit relatiu de la correccio).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent
SC = 10.0
HS = list(range(3, 24))


class _NoMemory(dict):
    def __setitem__(self, k, v):
        pass


class _Bank:
    def __init__(self, lam, p0, nmin):
        self.lam, self.nmin = lam, nmin
        self.th = {h: np.zeros(5) for h in HS}
        self.P = {h: np.eye(5) * p0 for h in HS}
        self.n = {h: 0 for h in HS}
        self.raw = {}                                   # t -> (f array 24 crua, ultim conegut)

    @staticmethod
    def x(f, last, tgt_hod):
        a = 2 * np.pi * tgt_hod / 24.0
        return np.array([1.0, f * SC, last * SC, np.sin(a), np.cos(a)])

    def update(self, h, x, e):
        P, th = self.P[h], self.th[h]
        Px = P @ x
        k = Px / (self.lam + x @ Px)
        th += k * (e - x @ th)
        self.P[h] = (P - np.outer(k, Px)) / self.lam
        # evita explosio de P amb poca excitacio
        tr = np.trace(self.P[h])
        if not np.isfinite(tr) or tr > 1e4:
            self.P[h] = np.eye(5) * 10.0
        self.n[h] += 1


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        for k, v in {"rls_lambda": 0.995, "rls_min": 48, "rls_price": False, "rls_p0": 10.0, "rls_max": 0.4}.items():
            if not (params and k in params):
                self.p[k] = v
        self._co2_raw_forecasts = _NoMemory()
        self.banks = {"co2": _Bank(self.p["rls_lambda"], self.p["rls_p0"], self.p["rls_min"])}
        if self.p["rls_price"]:
            self.banks["price"] = _Bank(self.p["rls_lambda"], self.p["rls_p0"], self.p["rls_min"])

    def _learn(self, key, bank, t, n_known):
        """Parelles noves: el valor realitzat de t+n_known-1 (acabat d'entrar a la finestra)."""
        if n_known < 1:
            return
        te = t + n_known - 1
        y = self.hist.get(key, te)
        if not np.isfinite(y):
            return
        for h in HS:
            tau = te - h
            rec = bank.raw.get(tau)
            if rec is None:
                continue
            f, last = rec
            if not np.isfinite(f[h]) or not np.isfinite(last):
                continue
            x = bank.x(f[h], last, (tau + h) % 24)
            e = (y - f[h]) * SC
            if np.isfinite(e) and np.isfinite(x).all():
                bank.update(h, x, e)

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = super()._forecast(t, price3, co2_3)
        try:
            for key, arr, k3 in (("co2", co2, co2_3), ("price", price, price3)):
                bank = self.banks.get(key)
                if bank is None or len(k3) < 1:
                    continue
                self._learn(key, bank, t, len(k3))
                last = float(k3[-1])
                raw = arr.copy()
                bank.raw[t] = (raw, last)
                for k in [k for k in bank.raw if k < t - 30]:
                    del bank.raw[k]
                lim = self.p["rls_max"]
                for h in range(max(3, len(k3)), 24):
                    n = bank.n[h]
                    if n < 1 or not np.isfinite(raw[h]):
                        continue
                    x = bank.x(raw[h], last, (t + h) % 24)
                    corr = float(x @ bank.th[h]) / SC * min(1.0, n / float(self.p["rls_min"]))
                    if not np.isfinite(corr):
                        continue
                    cap = lim * abs(raw[h]) + 0.01
                    arr[h] = raw[h] + float(np.clip(corr, -cap, cap))
                if key == "co2":
                    arr[:] = np.clip(arr, 0.0, 1.0)
        except Exception:
            pass
        return price, co2, load, pv
