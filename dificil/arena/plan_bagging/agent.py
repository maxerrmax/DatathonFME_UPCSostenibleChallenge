"""plan_bagging: bagging d'accions sobre hipotesis de previsio.

Cada hora es resol el mateix LP del v12 sobre 4-6 conjunts de previsio alternatius i es retorna la
mediana de la primera accio (robust a que un unic model s'equivoqui):
  a) previsio v12; b) preu/CO2 h>=3 = forma d'ahir reescalada al nivell conegut;
  c+/c-) CO2 llunya x(1 +/- delta) amb rampa; d+/d-) consum net +/- error tipic (h>=1).
Implementat sobreescrivint _forecast (guarda preu/CO2) i _lp (fa el bagging); la resta es el v12.
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
        super().__init__(params)
        for k, v in {"bag_hyp": "a,b,c+,c-,d+,d-", "bag_delta": 0.15, "bag_load_k": 1.5,
                     "bag_ramp": 10.0, "bag_agg": "median"}.items():
            self.p.setdefault(k, v)
        self._fc = None

    def _forecast(self, t, price3, co2_3):
        out = super()._forecast(t, price3, co2_3)
        self._fc = (t, np.array(out[0], float), np.array(out[1], float), len(price3), len(co2_3))
        return out

    def _yday(self, key, t, base, nk, H):
        """Forma d'ahir (mateixes hores) reescalada perque el valor conegut coincideixi."""
        y = np.array([self.hist.get(key, t + h - 24) for h in range(H)], float)
        if not np.isfinite(y).all():
            return None
        n = max(1, min(nk, H))
        yk, bk = y[:n].mean(), base[:n].mean()
        f = float(np.clip(bk / yk, 0.5, 2.0)) if yk > 1e-9 else 1.0
        alt = base.copy()
        alt[n:] = y[n:] * f
        return alt

    def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
        base_sol = super()._lp(w, net, soc0, v_term, dmax0, scen)
        try:
            fc = self._fc
            if fc is None or base_sol is None:
                return base_sol
            t, price, co2, npr, nco = fc
            H = len(w)
            price, co2 = price[:H], co2[:H]
            w = np.asarray(w, float); net = np.asarray(net, float)
            r = self.p["ratio"]
            sig = float(np.mean([abs(e) for _, e in self.err])) if len(self.err) >= 12 else 80.0
            d = float(self.p["bag_delta"])
            ramp = np.minimum(1.0, np.maximum(0.0, np.arange(H) - 2) / self.p["bag_ramp"])
            q = self.p["q_term"]
            acts = []
            for hy in str(self.p["bag_hyp"]).split(","):
                hy = hy.strip()
                w2, n2 = w, net
                if hy == "a":
                    acts.append(float(base_sol[0]))
                    continue
                if hy == "b":
                    cb = self._yday("co2", t, co2, nco, H)
                    pb = self._yday("price", t, price, npr, H)
                    if cb is None or pb is None:
                        continue
                    w2 = w + 0.7 * (cb - co2) + 0.3 * r * (pb - price)
                elif hy in ("c+", "c-"):
                    s = 1.0 if hy == "c+" else -1.0
                    w2 = w + 0.7 * co2 * (s * d * ramp)
                elif hy in ("d+", "d-"):
                    s = 1.0 if hy == "d+" else -1.0
                    n2 = net.copy()
                    n2[1:] += s * self.p["bag_load_k"] * sig
                else:
                    continue
                v2 = 0.0 if v_term == 0 else self.p["term_scale"] * ETA * float(
                    np.quantile(np.concatenate([w2, self.hist_w[-72:]]), q))
                P = super()._lp(w2, n2, soc0, v2, dmax0, scen)
                if P is not None and np.isfinite(P[0]):
                    acts.append(float(P[0]))
            if not acts:
                return base_sol
            a = float(np.median(acts)) if self.p["bag_agg"] == "median" else float(np.mean(acts))
            out = np.array(base_sol, float)
            out[0] = a
            return out
        except Exception:
            return base_sol
