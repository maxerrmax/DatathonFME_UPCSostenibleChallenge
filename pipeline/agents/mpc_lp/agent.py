"""
BatteryAgent — MPC amb programació lineal (Datathon UPC 2026).
A cada hora resol un LP sobre l'horitzó de previsions (24 h de preu i CO2)
minimitzant el senyal ponderat w_t = 0.7*co2_t/CO2B + 0.3*preu_t/CB,
i aplica només la primera acció (receding horizon).
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

CAP = 2000.0
PMAX = 500.0
ETA_C = 0.9 ** 0.5
ETA_D = 0.9 ** 0.5
EPS = 1e-3  # marge per no tocar mai els límits exactes (evita infraccions per coma flotant)


def solve_lp(w, soc0, v_term, dmax=None):
    """w: pesos per kWh importat (len H). Retorna potències netes P_t (kW)."""
    H = len(w)
    # variables: c_0..c_{H-1}, d_0..d_{H-1}
    cost = np.concatenate([w, -w])
    # SOC_k = soc0 + sum_{t<k} (eta_c c_t - d_t/eta_d), k=1..H
    L = np.tril(np.ones((H, H)))
    A_soc = np.hstack([L * ETA_C, -L / ETA_D])
    # 0 + EPS <= SOC_k <= CAP - EPS
    A_ub = np.vstack([A_soc, -A_soc])
    b_ub = np.concatenate([np.full(H, CAP - EPS - soc0), np.full(H, soc0 - EPS)])
    # valor terminal: -v_term * SOC_H
    cost = cost - v_term * A_soc[-1]
    ub_d = np.full(H, PMAX) if dmax is None else np.minimum(PMAX, dmax)
    bounds = [(0, PMAX)] * H + [(0, float(u)) for u in ub_d]
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if r.status != 0:
        return np.zeros(H)
    return r.x[:H] - r.x[H:]


class BatteryAgent:
    def __init__(self, params=None):
        p = {}
        f = Path(__file__).parent / "parametres.json"
        if f.exists():
            p = json.loads(f.read_text())
        if params:
            p.update(params)
        self.ratio = p.get("ratio_co2b_cb", 0.587)   # CO2_base / Cost_base (estimat a train)
        self.q_term = p.get("q_term", 0.6)           # quantil per valorar l'energia final
        self.term_scale = p.get("term_scale", 1.0)
        self.hist_w = []

    def weights(self, co2, price):
        # en unitats proporcionals a la puntuació: 0.7*co2/CO2B + 0.3*preu/CB  (escalat per CO2B)
        return 0.7 * np.asarray(co2) + 0.3 * self.ratio * np.asarray(price)

    def act(self, obs):
        soc = float(obs["soc_kwh"])
        price = list(obs["preus_properes_24h"]) or [obs["preu_actual"]]
        co2 = list(obs["co2_properes_24h"]) or [obs["co2_actual"]]
        H = min(len(price), len(co2))
        w = self.weights(co2[:H], price[:H])
        self.hist_w.append(float(w[0]))

        if H < 24:
            v = 0.0  # final del període: l'energia restant no val res
        else:
            ref = np.concatenate([w, self.hist_w[-72:]])
            v = self.term_scale * ETA_D * float(np.quantile(ref, self.q_term))

        # protecció: no descarregar per sobre del consum net observat (forats de dades a 0)
        last = obs.get("noves_dades_5min", {}) or {}
        net_last = float(last.get("ultim_consum_kw", 1e4)) - float(last.get("ultima_solar_kw", 0.0))
        dmax = np.full(H, PMAX)
        if net_last < PMAX:
            dmax[0] = max(0.0, net_last)

        P = solve_lp(w, soc, v, dmax)
        a = float(P[0])
        # clamp físic exacte
        max_c = min(PMAX, (CAP - soc) / ETA_C) - EPS
        max_d = min(PMAX, soc * ETA_D) - EPS
        a = max(-max(0.0, max_d), min(max(0.0, max_c), a))
        if abs(a) < 1e-6:
            a = 0.0
        return a
