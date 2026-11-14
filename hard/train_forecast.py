"""Entrena 4 LightGBM (preu, CO2, consum, solar) amb features idèntiques a les de l'agent.
Ús: python hard/train_forecast.py [fit|all]   (fit = sense la validació; all = tot train per al lliurament)"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib, lightgbm as lgb
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "agent_hard")); sys.path.insert(0, str(HERE.parent / "pipeline"))
from hdata import load_hard
from data import split_val
from feats import Hist, mkt_features, meas_features, MKT_NAMES, MEAS_NAMES

COLS = {"price": "precio_eur_per_kwh", "co2": "co2_kg_per_kwh", "load": "load_kw", "pv": "pv_kw",
        "irr": "irradiancia_wm2", "temp": "temp_ambient_c", "vent": "vent_ms"}


def epoch_hours(idx):
    return ((idx - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h")).astype(int)


def full_hist(df):
    h = Hist(); eh = epoch_hours(df.index)
    for k, c in COLS.items():
        for e, v in zip(eh, df[c].values):
            h.put(k, e, v)
    return h, eh


def build(df, step=1):
    hist, eh = full_hist(df)
    Xs = {k: [] for k in ("price", "co2", "load", "pv")}; Ys = {k: [] for k in Xs}
    present = set(eh.tolist())
    for t in eh[::step]:
        hs_m = [h for h in range(3, 24) if t + h in present]
        hs_l = [h for h in range(0, 24) if t + h in present]
        for k in ("price", "co2"):
            if hs_m:
                Xs[k].append(mkt_features(hist, k, t, hs_m)); Ys[k].append([hist.get(k, t + h) for h in hs_m])
        for k in ("load", "pv"):
            if hs_l:
                Xs[k].append(meas_features(hist, k, t, hs_l)); Ys[k].append([hist.get(k, t + h) for h in hs_l])
    return {k: (np.vstack(Xs[k]), np.concatenate(Ys[k])) for k in Xs}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "fit"
    df = load_hard(); fit, val = split_val(df)
    tr = fit if which == "fit" else df
    t0 = time.time(); D = build(tr); print("features", {k: v[0].shape for k, v in D.items()}, f"{time.time()-t0:.0f}s")
    models = {}
    P = dict(n_estimators=500, learning_rate=0.04, num_leaves=63, min_child_samples=20,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
    for k, (X, y) in D.items():
        lag = X[:, 5]; ok = np.isfinite(lag)
        models[k + "_dir"] = lgb.LGBMRegressor(**P).fit(X, y).booster_            # model directe
        models[k + "_res"] = lgb.LGBMRegressor(**P).fit(X[ok], (y - lag)[ok]).booster_  # y - lag24
    out = HERE / "agent_hard" / f"models_{which}.pkl"
    joblib.dump({k: b.model_to_string() for k, b in models.items()}, out)
    print("desat", out)

