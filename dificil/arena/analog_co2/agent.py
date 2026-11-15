"""analog_co2: previsio per analegs (veins mes propers a l'historial) de CO2 (i opcionalment preu), barrejada amb LightGBM.

A l'hora t la consulta es el vector de les ultimes 24 h de CO2 + les 3 h conegudes (t-23..t+2) normalitzat pel
darrer valor conegut, mes l'hora del dia (nomes es miren hores tau amb el mateix tau%24, tau <= t-24, de manera que
la continuacio tau+3..tau+23 ja es coneguda). Es prenen els k mes semblants (distancia amb pesos creixents cap al
present, petita penalitzacio si cap de setmana difereix) i la previsio de les hores 3..23 es la continuacio
(escalada pel ratio) mitjana ponderada. Mescla: x = (1-mix)*lgb + mix*analeg, mix = `analog_mix` (0.4).
Preu: mateix metode en diferencies normalitzades (`analog_mix_price`, per defecte 0.0).
Params: analog_k, analog_mix, analog_mix_price, analog_decay. Qualsevol error -> previsio del v12.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base

Base = load_base().BatteryAgent
Q = 26                                    # t-23 .. t+2


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        for k, v in {"analog_k": 10, "analog_mix": 0.4, "analog_mix_price": 0.0, "analog_decay": 12.0}.items():
            if not (params and k in params):
                self.p[k] = v

    def _dense(self, key):
        d = self.hist.d[key]
        keys = np.fromiter(d.keys(), np.int64, len(d)); vals = np.fromiter(d.values(), float, len(d))
        e0 = int(keys.min())
        arr = np.full(int(keys.max()) - e0 + 1, np.nan)
        arr[keys - e0] = vals
        return arr, e0

    def _analog(self, key, t, n_known, mode):
        """Retorna array (24,) amb la previsio per analegs de les hores n_known..23 (nan on no es pot)."""
        arr, e0 = self._dense(key)
        out = np.full(24, np.nan)
        tl = t + n_known - 1                                         # darrera hora coneguda
        if tl - e0 + 1 > len(arr) or n_known < 1:
            return out
        qi = np.arange(t - 23, tl + 1) - e0
        if qi.min() < 0 or qi.max() >= len(arr):
            return out
        q = arr[qi]
        if not np.isfinite(q).all() or len(q) < 10:
            return out
        L = len(q)
        # candidats: tau amb tau%24==t%24, tau <= t-24, finestra [tau-23, tau+23] disponible
        lo = e0 + 23; hi = min(t - 24, e0 + len(arr) - 1 - 23)
        if hi < lo:
            return out
        first = lo + ((t - lo) % 24)
        taus = np.arange(first, hi + 1, 24)
        if len(taus) < 3:
            return out
        qoff = np.arange(-23, -23 + L)                               # mateix offset que la consulta
        Qm = arr[(taus - e0)[:, None] + qoff[None, :]]
        C = arr[(taus - e0)[:, None] + np.arange(n_known, 24)[None, :]]       # tau+h, h = n_known..23
        good = np.isfinite(Qm).all(1) & np.isfinite(C).all(1)
        if good.sum() < 3:
            return out
        taus, Qm, C = taus[good], Qm[good], C[good]
        lastq = q[-1]; lastc = Qm[:, -1]
        wts = np.exp(-(L - 1 - np.arange(L)) / self.p["analog_decay"])
        if mode == "ratio":
            if lastq <= 1e-6:
                return out
            okc = lastc > 1e-6
            if okc.sum() < 3:
                return out
            taus, Qm, C, lastc = taus[okc], Qm[okc], C[okc], lastc[okc]
            qn = q / lastq; Qn = Qm / lastc[:, None]
        else:
            sd = max(float(np.std(q)), 1e-3)
            qn = (q - lastq) / sd; Qn = (Qm - lastc[:, None]) / sd
        dist = np.sqrt((wts * (Qn - qn) ** 2).sum(1) / wts.sum())
        wk = ((taus // 24 + 3) % 7 >= 5) != (((t // 24) + 3) % 7 >= 5)
        dist = dist + 0.05 * wk
        k = int(min(self.p["analog_k"], len(taus)))
        idx = np.argsort(dist)[:k]
        ww = 1.0 / (dist[idx] + 0.02); ww /= ww.sum()
        if mode == "ratio":
            cont = (C[idx] / lastc[idx, None]) * lastq
        else:
            cont = (C[idx] - lastc[idx, None]) + lastq
        res = (ww[:, None] * cont).sum(0)                            # indexat per h - n_known + 1... ver abaix
        # C columna j = tau + (n_known-1) + j; hora t+h <-> j = h - (n_known-1)
        out[n_known:] = res
        return out

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = super()._forecast(t, price3, co2_3)
        try:
            for key, arr, n, mix, mode in (("co2", co2, len(co2_3), self.p["analog_mix"], "ratio"),
                                           ("price", price, len(price3), self.p["analog_mix_price"], "diff")):
                if mix <= 0 or n < 1:
                    continue
                a = self._analog(key, t, n, mode)
                ok = np.isfinite(a)
                ok[:max(n, 3)] = False
                if ok.any():
                    new = arr.copy()
                    new[ok] = (1 - mix) * arr[ok] + mix * a[ok]
                    if key == "co2":
                        new = np.clip(new, 0.0, 1.0)
                    arr[:] = new
        except Exception:
            pass
        return price, co2, load, pv
