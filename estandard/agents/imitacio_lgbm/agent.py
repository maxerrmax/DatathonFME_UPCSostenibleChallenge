"""Imitation learning: LightGBM entrenat per imitar l'oracle LP. Inclou safety clamp."""
from pathlib import Path
import numpy as np, joblib

CAP, PMAX, ETA, RATIO = 2000.0, 500.0, 0.9 ** 0.5, 0.587


def obs_features(soc, co2_24, price_24, hour, dow):
    w = 0.7 * np.asarray(co2_24, float) + 0.3 * RATIO * np.asarray(price_24, float)
    H = len(w)
    if H < 24:
        w = np.concatenate([w, np.full(24 - H, w[-1] if H else 0.0)])
    rel = (w - w.mean()) / (w.std() + 1e-9)
    rank0 = float((w < w[0]).sum()) / 23.0
    return np.concatenate([[soc / 2000.0, rank0, w[0] / (w.mean() + 1e-9), hour, dow, H],
                           rel, [w.min(), w.max(), w.mean()]])


class BatteryAgent:
    def __init__(self):
        self.m = joblib.load(Path(__file__).parent / "model.pkl")

    def act(self, obs):
        soc = float(obs["soc_kwh"])
        x = obs_features(soc, obs["co2_properes_24h"], obs["preus_properes_24h"], obs["hour"], obs["day_of_week"])
        a = float(self.m.predict(x[None, :])[0])
        max_c = min(PMAX, (CAP - soc) / ETA) - 1e-3
        max_d = min(PMAX, soc * ETA) - 1e-3
        return float(max(-max(0.0, max_d), min(max(0.0, max_c), a)))
