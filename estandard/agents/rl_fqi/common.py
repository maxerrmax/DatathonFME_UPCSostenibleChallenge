"""Codi compartit entre l'entrenament i la inferència de l'agent FQI (no importa res prohibit)."""
import numpy as np

CAP, PMAX, ETA, RATIO = 2000.0, 500.0, 0.9 ** 0.5, 0.587
ACTIONS = np.array([-500.0, -250.0, 0.0, 250.0, 500.0])


def wvec(co2, price):
    w = 0.7 * np.asarray(co2, float) + 0.3 * RATIO * np.asarray(price, float)
    if len(w) < 24:
        w = np.concatenate([w, np.full(24 - len(w), w[-1] if len(w) else 0.0)])
    return w


def feat(soc, w):
    rel = (w - w.mean()) / (w.std() + 1e-9)
    return np.concatenate([[soc / CAP, (w < w[0]).sum() / 23.0], rel, [w.std() / (w.mean() + 1e-9)]])


def clamp(soc, a):
    max_c = min(PMAX, (CAP - soc) / ETA) - 1e-3
    max_d = min(PMAX, soc * ETA) - 1e-3
    return float(max(-max(0.0, max_d), min(max(0.0, max_c), a)))


def step_soc(soc, p):
    return soc + p * ETA if p >= 0 else soc + p / ETA
