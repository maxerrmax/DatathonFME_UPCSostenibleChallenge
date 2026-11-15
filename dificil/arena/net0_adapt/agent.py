"""net0_adapt: mescla adaptativa de l'hora actual (model directe net0 vs consum-solar dels models separats).

El v12 barreja amb pes fix net0_mix=0.5. Aqui es calculen les dues prediccions de l'hora 0 per separat, i amb el net
realitzat (ultim_consum - ultima_solar de t-1) s'actualitza una MSE exponencial per hora del dia (na_alpha) i una de
global (na_alpha_g) per a cada predictor. Pes del directe w = e_sep^2/(e_dir^2+e_sep^2), retallat a [na_lo, na_hi].
Fallback: hod amb >= na_minn mostres -> global amb >= na_ming -> 0.5.
Params: na_alpha (0.2), na_alpha_g (0.03), na_minn (3), na_ming (6), na_lo (0.2), na_hi (0.9).
Causal. Es forca net0_mix=0 a la base per obtenir la previsio separada i despres es barreja aqui.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_base
import numpy as np

Base = load_base().BatteryAgent
from feats import load_features2, meas_features


class BatteryAgent(Base):
    def __init__(self, params=None):
        super().__init__(params)
        self._mix_default = self.p.get("net0_mix", 0.5)
        self.p["net0_mix"] = 0.0                    # la base no barreja; ho fem aqui
        self._na_last = None                          # (t, pred_dir, pred_sep)
        self._m_h = {}                                # hod -> [mse_dir, mse_sep, n]
        self._m_g = [0.0, 0.0, 0]

    @staticmethod
    def _ew(m, ed, es, a):
        if m[2] == 0:
            m[0], m[1] = ed * ed, es * es
        else:
            m[0] += a * (ed * ed - m[0]); m[1] += a * (es * es - m[1])
        m[2] += 1

    def _learn(self, t):
        if self._na_last is None or self._na_last[0] != t - 1:
            return
        ld, pv = self.hist.get("load", t - 1), self.hist.get("pv", t - 1)
        if not (np.isfinite(ld) and np.isfinite(pv)):
            return
        real = ld - pv
        _, pdir, psep = self._na_last
        ed, es = real - pdir, real - psep
        h = (t - 1) % 24
        self._ew(self._m_h.setdefault(h, [0.0, 0.0, 0]), ed, es, self.p.get("na_alpha", 0.2))
        self._ew(self._m_g, ed, es, self.p.get("na_alpha_g", 0.03))

    def _weight(self, t):
        m = self._m_h.get(t % 24)
        if m is None or m[2] < self.p.get("na_minn", 3):
            m = self._m_g if self._m_g[2] >= self.p.get("na_ming", 6) else None
        if m is None:
            return 0.5
        d, s = m[0], m[1]
        w = s / (d + s) if (d + s) > 1e-9 else 0.5
        return float(np.clip(w, self.p.get("na_lo", 0.2), self.p.get("na_hi", 0.9)))

    def _forecast(self, t, price3, co2_3):
        price, co2, load, pv = super()._forecast(t, price3, co2_3)
        try:
            self._learn(t)
            self._na_last = None
            if self.n0 is not None and self._mix_default > 0:
                x0 = np.concatenate([load_features2(self.hist, t, [0])[0], meas_features(self.hist, "pv", t, [0])[0]])
                nd = float(self.n0.predict(x0.reshape(1, -1), num_threads=1)[0])
                ns = float(load[0] - pv[0])
                if np.isfinite(nd) and np.isfinite(ns):
                    w = self._weight(t)
                    load[0] = w * nd + (1 - w) * ns + pv[0]
                    self._na_last = (t, nd, ns)
        except Exception:
            pass
        return price, co2, load, pv
