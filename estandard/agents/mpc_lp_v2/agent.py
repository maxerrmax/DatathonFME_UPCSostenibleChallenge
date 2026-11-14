"""
MPC-LP v2: com mpc_lp, més
 1) previsió de demanda neta per hora (historial propi: última hora, mateixa hora ahir i fa 7 dies)
    per limitar la descàrrega a hores de consum baix (<500 kW) a TOT l'horitzó, no només la 1a hora;
 2) ràtio CO2_base/Cost_base estimada en línia amb la demanda observada (prior 0.605, del leaderboard).
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

CAP, PMAX, ETA_C, ETA_D, EPS = 2000.0, 500.0, 0.9 ** 0.5, 0.9 ** 0.5, 1e-3


def solve_lp(w, soc0, v_term, dmax):
    H = len(w)
    L = np.tril(np.ones((H, H)))
    A_soc = np.hstack([L * ETA_C, -L / ETA_D])
    A_ub = np.vstack([A_soc, -A_soc])
    b_ub = np.concatenate([np.full(H, CAP - EPS - soc0), np.full(H, soc0 - EPS)])
    cost = np.concatenate([w, -w]) - v_term * A_soc[-1]
    bounds = [(0, PMAX)] * H + [(0, float(u)) for u in dmax]
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return r.x[:H] - r.x[H:] if r.status == 0 else np.zeros(H)


class BatteryAgent:
    def __init__(self, params=None):
        p = {"ratio_prior": 0.605, "prior_hours": 168, "q_term": 0.6, "safety": 0.95}
        f = Path(__file__).parent / "parametres.json"
        if f.exists():
            p.update(json.loads(f.read_text()))
        if params:
            p.update(params)
        self.p = p
        self.hist_w, self.net_hist, self.co2_hist, self.pr_hist = [], [], [], []

    def _ratio(self):
        """CO2_base/Cost_base estimada amb la demanda observada, encongida cap al prior."""
        n = len(self.net_hist); r0, k = self.p["ratio_prior"], self.p["prior_hours"]
        if n == 0:
            return r0
        net = np.maximum(np.array(self.net_hist), 0)
        r = float(net @ np.array(self.co2_hist)) / max(1e-9, float(net @ np.array(self.pr_hist)))
        return (r * n + r0 * k) / (n + k)

    def _net_forecast(self, H):
        """Previsió conservadora de demanda neta per a les H hores següents."""
        n = len(self.net_hist); out = np.full(H, 1e4)
        if n == 0:
            return out
        last = self.net_hist[-1]
        for h in range(H):
            c = [last] if h == 0 else []
            # self.net_hist[-1] és l'hora t-1; l'hora t+h té lag 24 a l'índex t+h-24
            for lag in (24, 168):
                i = n + h - lag  # índex a net_hist de l'hora t+h-lag
                if 0 <= i < n:
                    c.append(self.net_hist[i])
            if c:
                out[h] = self.p["safety"] * min(c)
        return out

    def act(self, obs):
        nd = obs.get("noves_dades_5min", {}) or {}
        soc = float(obs["soc_kwh"])
        price = list(obs["preus_properes_24h"]) or [obs["preu_actual"]]
        co2 = list(obs["co2_properes_24h"]) or [obs["co2_actual"]]
        H = min(len(price), len(co2))
        # historial: la dada "última" correspon a l'hora anterior (t-1)
        if self.hist_w:
            self.net_hist.append(float(nd.get("ultim_consum_kw", 1e4)) - float(nd.get("ultima_solar_kw", 0.0)))
            self.co2_hist.append(self._prev[0]); self.pr_hist.append(self._prev[1])
        self._prev = (float(co2[0]), float(price[0]))
        ratio = self._ratio()
        w = 0.7 * np.asarray(co2[:H]) + 0.3 * ratio * np.asarray(price[:H])
        self.hist_w.append(float(w[0]))
        v = 0.0 if H < 24 else ETA_D * float(np.quantile(np.concatenate([w, self.hist_w[-72:]]), self.p["q_term"]))
        dmax = np.clip(self._net_forecast(H), 0, PMAX)
        P = solve_lp(w, soc, v, dmax)
        a = float(P[0])
        max_c = min(PMAX, (CAP - soc) / ETA_C) - EPS
        max_d = min(PMAX, soc * ETA_D) - EPS
        a = max(-max(0.0, max_d), min(max(0.0, max_c), a))
        return 0.0 if abs(a) < 1e-6 else a
