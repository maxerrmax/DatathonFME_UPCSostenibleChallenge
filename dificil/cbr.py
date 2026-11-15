"""CBR (raonament basat en casos) per al CO2: recupera els k casos més semblants de la base de casos (només fit)
i adapta la seva continuació (+3..+23 h) a l'últim valor conegut. Avaluació per replay combinat amb la v12."""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exp import FIT, VAL
from train_forecast import epoch_hours

def ser(df):
    eh = epoch_hours(df.index).values
    return {k: dict(zip(eh.tolist(), df[c].values.astype(float))) for k, c in (("co2", "co2_kg_per_kwh"), ("price", "precio_eur_per_kwh"))}, eh

def desc(S, t):
    k2 = S["co2"].get(t + 2, np.nan); p2 = S["price"].get(t + 2, np.nan)
    if not (np.isfinite(k2) and k2 > 0 and np.isfinite(p2) and p2 > 0):
        return None
    c = np.array([S["co2"].get(t + 2 - i, np.nan) for i in range(27)]); p = np.array([S["price"].get(t + 2 - i, np.nan) for i in range(27)])
    if np.isnan(c).sum() > 3:
        return None
    c = np.nan_to_num(np.log(np.maximum(c, 1e-4) / k2)); p = np.nan_to_num(np.log(np.maximum(p, 1e-4) / p2))
    ts = pd.Timestamp(int(t) * 3600, unit="s"); h = ts.hour
    wk = 1.0 if ts.dayofweek >= 5 else 0.0
    doy = ts.dayofyear / 365.25 * 2 * np.pi
    return np.concatenate([c * 3, p, [np.sin(2 * np.pi * h / 24) * 3, np.cos(2 * np.pi * h / 24) * 3, wk * 2, np.sin(doy), np.cos(doy), np.log(k2) * 2]]), k2

def base(df):
    S, eh = ser(df); D, F, K = [], [], []
    for t in eh:
        r = desc(S, t)
        fut = np.array([S["co2"].get(t + h, np.nan) for h in range(3, 24)])
        if r is None or np.isnan(fut).any():
            continue
        D.append(r[0]); F.append(np.log(fut / r[1])); K.append(t)
    return np.array(D), np.array(F)

if __name__ == "__main__":
    D, F = base(FIT)
    S, ev = ser(pd.concat([FIT, VAL]).sort_index()); _, tv = ser(VAL)
    out = {}
    for k in (5, 15, 40):
        P = {}
        for t in tv:
            r = desc(S, t)
            if r is None:
                continue
            d = ((D - r[0]) ** 2).sum(1); nn = np.argpartition(d, k)[:k]; w = 1 / (np.sqrt(d[nn]) + 1e-3)
            P[int(t)] = r[1] * np.exp((F[nn] * w[:, None]).sum(0) / w.sum())
        out[k] = P
    print("casos a la base:", len(D)); pickle.dump(out, open(HERE / "results" / "cbr.pkl", "wb"))
