"""Programació dinàmica sobre SOC discretitzat (horitzó = previsió de 24 h).
Serveix per validar que el LP és òptim i com a base per a variants estocàstiques."""
import numpy as np

CAP, PMAX, ETA = 2000.0, 500.0, 0.9 ** 0.5


class BatteryAgent:
    def __init__(self, n_soc=81, ratio=0.587, q_term=0.6):
        self.grid = np.linspace(0, CAP, n_soc)
        self.ratio, self.q = ratio, q_term
        d = self.grid[None, :] - self.grid[:, None]          # canvi de SOC i -> j
        p = np.where(d >= 0, d / ETA, d * ETA)               # potència a xarxa necessària
        self.P = np.where(np.abs(p) <= PMAX + 1e-9, p, np.nan)

    def act(self, obs):
        w = 0.7 * np.asarray(obs["co2_properes_24h"]) + 0.3 * self.ratio * np.asarray(obs["preus_properes_24h"])
        H = len(w)
        v_term = 0.0 if H < 24 else ETA * float(np.quantile(w, self.q))
        V = -v_term * self.grid                               # cost-to-go terminal
        pol = []
        for t in range(H - 1, -1, -1):
            Q = w[t] * self.P + V[None, :]                    # (i, j)
            Q = np.where(np.isnan(Q), np.inf, Q)
            j = np.argmin(Q, axis=1); pol.append(j); V = Q[np.arange(len(V)), j]
        j0 = pol[-1]
        soc = float(obs["soc_kwh"])
        i = int(np.argmin(np.abs(self.grid - soc)))
        target = self.grid[j0[i]]
        dsoc = target - soc
        a = dsoc / ETA if dsoc >= 0 else dsoc * ETA
        max_c = min(PMAX, (CAP - soc) / ETA) - 1e-3
        max_d = min(PMAX, soc * ETA) - 1e-3
        return float(max(-max(0.0, max_d), min(max(0.0, max_c), a)))
