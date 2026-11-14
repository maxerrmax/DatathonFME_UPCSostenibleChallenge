"""Heurística de rànquing: carrega a les k hores amb menor senyal de les properes 24 h,
descarrega a les k amb major senyal (si el diferencial supera la pèrdua del 10%)."""
import numpy as np

CAP, PMAX, ETA = 2000.0, 500.0, 0.9 ** 0.5


class BatteryAgent:
    def __init__(self, k_charge=4, k_dis=4, ratio=0.587):
        self.kc, self.kd, self.ratio = k_charge, k_dis, ratio

    def act(self, obs):
        w = 0.7 * np.asarray(obs["co2_properes_24h"]) + 0.3 * self.ratio * np.asarray(obs["preus_properes_24h"])
        if len(w) == 0:
            return 0.0
        order = np.argsort(w)
        rank = int(np.where(order == 0)[0][0])
        soc = float(obs["soc_kwh"])
        a = 0.0
        if rank < self.kc and w[0] < 0.9 * np.sort(w)[-self.kd:].mean():
            a = PMAX
        elif rank >= len(w) - self.kd and w[0] * 0.9 > np.sort(w)[:self.kc].mean():
            a = -PMAX
        max_c = min(PMAX, (CAP - soc) / ETA) - 1e-3
        max_d = min(PMAX, soc * ETA) - 1e-3
        return float(max(-max(0.0, max_d), min(max(0.0, max_c), a)))
