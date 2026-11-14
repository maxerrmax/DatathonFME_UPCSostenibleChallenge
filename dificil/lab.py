"""Laboratori de previsió: matrius de features (train=FIT, val=VAL amb historial FIT+VAL) i avaluació per MAE i per score."""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "agent_hard")); sys.path.insert(0, str(HERE.parent / "estandard"))
from exp import FIT, VAL, _df, replay, BLOCKS
from train_forecast import full_hist, epoch_hours, make_cs, COLS
import feats

CACHE = HERE / "results" / "lab_mkt2.pkl"


def build_mkt2():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    feats.set_cs(make_cs(FIT))
    hist, _ = full_hist(pd.concat([FIT, VAL]).sort_index())
    out = {}
    for part, d in (("fit", FIT), ("val", VAL)):
        eh = epoch_hours(d.index).values; present = set(eh.tolist())
        for key in ("price", "co2"):
            X, y, T, Hh = [], [], [], []
            for t in eh:
                hs = [h for h in range(3, 24) if t + h in present]
                if not hs:
                    continue
                X.append(feats.mkt_features2(hist, key, t, hs)); y += [hist.get(key, t + h) for h in hs]
                T += [t] * len(hs); Hh += hs
            out[part, key] = (np.vstack(X), np.array(y), np.array(T), np.array(Hh))
    pickle.dump(out, open(CACHE, "wb")); return out


def swap(table, key, T, Hh, pred):
    j = {"price": 0, "co2": 1, "load": 2, "pv": 3}[key]
    new = {t: tuple(a.copy() for a in v) for t, v in table.items()}
    for t, h, p in zip(T, Hh, pred):
        if t in new and h < len(new[t][j]):
            new[t][j][h] = p
    return new


if __name__ == "__main__":
    t0 = time.time(); D = build_mkt2(); print({k: v[0].shape for k, v in D.items()}, f"{time.time()-t0:.0f}s")
