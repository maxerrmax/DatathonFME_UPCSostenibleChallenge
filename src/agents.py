"""
Agents de referència per comparar en local (Mode Difícil i Estàndard).
"""
import numpy as np
import pandas as pd

from src.optim import solve_dispatch

ETA = 0.9 ** 0.5


def clamp(p, obs):
    cap = float(obs.get("cap_max_kwh", 2000.0))
    pmax = float(obs.get("power_max_kw", 500.0))
    soc = float(obs["soc_kwh"])
    hi = max(0.0, min(pmax, (cap - soc) / ETA - 1e-4))
    lo = max(0.0, min(pmax, soc * ETA - 1e-4))
    return float(max(-lo, min(hi, p)))


class KitHeuristic:
    """Heurística del kit, escalada a la potència disponible."""

    def act(self, obs):
        k = float(obs.get("power_max_kw", 500.0)) / 500.0
        h, pr, soc, co2 = obs["hour"], obs["preu_actual"], obs["soc_pct"], obs["co2_actual"]
        p = 0.0
        if 1 <= h <= 6 and pr < 0.08 and soc < 0.85:
            p = 350.0 * k
        elif 11 <= h <= 15 and pr < 0.10 and soc < 0.95:
            p = 250.0 * k
        elif 18 <= h <= 22 and (pr > 0.16 or co2 > 0.10) and soc > 0.15:
            p = -450.0 * k
        return clamp(p, obs)


class OracleMPC:
    """MPC de 24 h amb previsions PERFECTES (llegeix el futur del DataFrame). Només per mesurar."""

    def __init__(self, data, kc, kp, horizon=24):
        self.d = data.reset_index(drop=True)
        self.kc, self.kp, self.H = kc, kp, horizon
        self.t = 0

    def act(self, obs):
        s = self.d.iloc[self.t:self.t + self.H]
        self.t += 1
        w = 0.7 * s.co2_kg_per_kwh.values / self.kc + 0.3 * s.precio_eur_per_kwh.values / self.kp
        net = (s.load_kw - s.pv_kw).values
        p = solve_dispatch(w, net, obs["soc_kwh"], obs["cap_max_kwh"], obs["power_max_kw"])
        return clamp(p[0], obs)


class NaiveForecastMPC:
    """MPC de 24 h amb previsions ingènues: el que no es coneix = mateixa hora d'ahir."""

    def __init__(self, kc, kp, horizon=24):
        self.kc, self.kp, self.H = kc, kp, horizon
        self.hist = {}  # timestamp -> (price, co2, load, pv)
        self.prev_ts = None

    def _get(self, ts, i, default):
        v = self.hist.get(ts)
        return default if v is None or v[i] is None else v[i]

    def act(self, obs):
        ts = obs["timestamp"]
        n5 = obs["noves_dades_5min"]
        # guardem el que sabem
        for k, (pr, c) in enumerate(zip(obs["preus_properes_24h"], obs["co2_properes_24h"])):
            t = ts + pd.Timedelta(hours=k)
            old = self.hist.get(t, (None, None, None, None))
            self.hist[t] = (pr, c, old[2], old[3])
        tp = ts - pd.Timedelta(hours=1)
        old = self.hist.get(tp, (None, None, None, None))
        self.hist[tp] = (old[0], old[1], n5["ultim_consum_kw"], n5["ultima_solar_kw"])

        H = self.H
        price, co2, net = np.zeros(H), np.zeros(H), np.zeros(H)
        last_load, last_pv = n5["ultim_consum_kw"], n5["ultima_solar_kw"]
        for k in range(H):
            t = ts + pd.Timedelta(hours=k)
            y = t - pd.Timedelta(hours=24)
            price[k] = self._get(t, 0, self._get(y, 0, obs["preu_actual"]))
            co2[k] = self._get(t, 1, self._get(y, 1, obs["co2_actual"]))
            load = self._get(y, 2, last_load)
            pv = self._get(y, 3, 0.0)
            net[k] = load - pv
        w = 0.7 * co2 / self.kc + 0.3 * price / self.kp
        p = solve_dispatch(w, net, obs["soc_kwh"], obs["cap_max_kwh"], obs["power_max_kw"])
        return clamp(p[0], obs)
