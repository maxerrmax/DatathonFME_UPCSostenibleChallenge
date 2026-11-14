"""Heurística de rànquing amb paràmetres optimitzats per evolució diferencial (tune.py)."""
import json
from pathlib import Path
import numpy as np

CAP, PMAX, ETA = 2000.0, 500.0, 0.9 ** 0.5


class BatteryAgent:
    def __init__(self, params=None):
        p = {"k_charge": 4, "k_dis": 4, "ratio": 0.587, "margin": 0.9}
        f = Path(__file__).parent / "parametres.json"
        if f.exists():
            p.update(json.loads(f.read_text()))
        if params:
            p.update(params)
        self.kc, self.kd = int(round(p["k_charge"])), int(round(p["k_dis"]))
        self.ratio, self.margin = p["ratio"], p["margin"]

    def act(self, obs):
        w = 0.7 * np.asarray(obs["co2_properes_24h"]) + 0.3 * self.ratio * np.asarray(obs["preus_properes_24h"])
        if len(w) == 0:
            return 0.0
        s = np.sort(w); rank = int((w < w[0]).sum())
        soc = float(obs["soc_kwh"]); a = 0.0
        if rank < self.kc and w[0] < self.margin * s[-self.kd:].mean():
            a = PMAX
        elif rank >= len(w) - self.kd and w[0] * self.margin > s[:self.kc].mean():
            a = -PMAX
        max_c = min(PMAX, (CAP - soc) / ETA) - 1e-3
        max_d = min(PMAX, soc * ETA) - 1e-3
        return float(max(-max(0.0, max_d), min(max(0.0, max_c), a)))
