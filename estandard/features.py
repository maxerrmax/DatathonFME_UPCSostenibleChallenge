"""Features compartides per als agents apresos (copiades dins de cada agent per ser autocontinguts)."""
import numpy as np

RATIO = 0.587


def obs_features(soc, co2_24, price_24, hour, dow):
    w = 0.7 * np.asarray(co2_24, float) + 0.3 * RATIO * np.asarray(price_24, float)
    H = len(w)
    if H < 24:
        w = np.concatenate([w, np.full(24 - H, w[-1] if H else 0.0)])
    rel = (w - w.mean()) / (w.std() + 1e-9)
    rank0 = float((w < w[0]).sum()) / 23.0
    return np.concatenate([[soc / 2000.0, rank0, w[0] / (w.mean() + 1e-9), hour, dow, H],
                           rel, [w.min(), w.max(), w.mean()]])
