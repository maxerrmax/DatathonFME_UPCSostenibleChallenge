"""Entrena 4 LightGBM (preu, CO2, consum, solar) amb features idèntiques a les de l'agent.
Ús: python dificil/train_forecast.py [fit|all]   (fit = sense la validació; all = tot train per al lliurament)"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib, lightgbm as lgb
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "agent_hard")); sys.path.insert(0, str(HERE.parent / "estandard"))
from hdata import load_hard
from data import split_val
from feats import Hist, mkt_features, meas_features, MKT_NAMES, MEAS_NAMES, cs_table, set_cs

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


def make_cs(df):
    # taula en hora estàndard (hivern): treu l'offset d'hora d'estiu abans de dia de l'any / hora
    import feats as _f
    eh = epoch_hours(df.index).values
    es = np.array([_f.std_epoch(e) for e in eh], dtype=np.int64)
    day = es.astype("int64") // 24; hr = es % 24
    d64 = day.astype("datetime64[D]")
    doy = (d64 - d64.astype("datetime64[Y]")).astype(np.int64)
    return cs_table(doy, hr, df["pv_kw"].values)


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
    CS = make_cs(tr); set_cs(CS)
    t0 = time.time(); D = build(tr); print("features", {k: v[0].shape for k, v in D.items()}, f"{time.time()-t0:.0f}s")
    models = {}
    P = dict(n_estimators=500, learning_rate=0.04, num_leaves=63, min_child_samples=20,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
    for k, (X, y) in D.items():
        lag = X[:, 5]; ok = np.isfinite(lag)
        models[k + "_dir"] = lgb.LGBMRegressor(**P).fit(X, y).booster_            # model directe
        models[k + "_res"] = lgb.LGBMRegressor(**P).fit(X[ok], (y - lag)[ok]).booster_  # y - lag24
    out = HERE / "agent_hard" / f"models_{which}.pkl"
    dump = {k: b.model_to_string() for k, b in models.items()}; dump["pv_cs"] = CS
    eh = epoch_hours(tr.index).values
    dump["hist0"] = {k: (eh.astype(np.int64), tr[c].values.astype(float)) for k, c in COLS.items()}
    joblib.dump(dump, out)
    print("desat", out)

