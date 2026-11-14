"""
Entrena els models de previsió (LightGBM) per al Mode Difícil.

Imita el que veu l'agent al test: blocs de 7 dies sense historial previ.
  - validació: bloc dels dies 15-21 de cada mes (fora de l'entrenament)
  - final:     tots els blocs -> models a submission_hard/

Ús:  python scripts/train_forecast.py [--final]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data import load_raw_hourly
from src.forecast import (PRICE_FEATS, LOAD_FEATS, PV_FEATS, price_features, load_features,
                          pv_features, ts_to_hour)

OUT = ROOT / "submission_hard"
COLS = {"p": "precio_eur_per_kwh", "c": "co2_kg_per_kwh", "load": "load_kw", "pv": "pv_kw",
        "irr": "irradiancia_wm2", "temp": "temp_ambient_c"}
KNOWN_AHEAD = {"p": 2, "c": 2}  # la resta: fins a t0-1


def build_store(raw):
    hours = np.array([ts_to_hour(t) for t in raw.index])
    store = {k: dict(zip(hours, raw[c].values)) for k, c in COLS.items()}
    return store, hours


def block_of(ts):
    d = ts.day
    b = 0 if d <= 7 else 1 if d <= 14 else 2 if d <= 21 else 3
    return (ts.year, ts.month, b)


def make_samples(raw, store, hours, blocks_keep):
    blk = np.array([block_of(t) for t in raw.index], dtype=object)
    blk_of_hour = dict(zip(hours, map(tuple, blk)))
    X = {"p": [], "c": [], "load": [], "pv": []}
    Y = {"p": [], "c": [], "load": [], "pv": []}
    for t0, b in zip(hours, map(tuple, blk)):
        if b not in blocks_keep:
            continue

        def get(col, h, t0=t0, b=b):
            if blk_of_hour.get(h) != b or h > t0 + KNOWN_AHEAD.get(col, -1):
                return float("nan")
            v = store[col].get(h, float("nan"))
            return float(v)

        for k in range(24):
            T = t0 + k
            if blk_of_hour.get(T) != b:
                break
            if k >= 3:
                for tg in ("p", "c"):
                    y = store[tg].get(T, np.nan)
                    if y == y:
                        X[tg].append(price_features(get, t0, k, tg)); Y[tg].append(y)
            y = store["load"].get(T, np.nan)
            if y == y:
                X["load"].append(load_features(get, t0, k)); Y["load"].append(y)
            y = store["pv"].get(T, np.nan)
            if y == y:
                X["pv"].append(pv_features(get, t0, k)); Y["pv"].append(y)
    return {k: (np.array(X[k], float), np.array(Y[k], float)) for k in X}


FEATS = {"p": PRICE_FEATS, "c": PRICE_FEATS, "load": LOAD_FEATS, "pv": PV_FEATS}
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
REG = dict(PARAMS, num_leaves=15, min_data_in_leaf=150, learning_rate=0.03, lambda_l2=10, feature_fraction=0.7)
# Configuració triada per validació (MAE): veure historial a la memòria del projecte
TARGET_PARAMS = {
    "p": REG,
    "c": dict(REG, objective="huber"),          # alpha s'ajusta a la desviació de y
    "load": PARAMS,
    "pv": dict(REG, objective="l1"),            # molts zeros (nit): L1 va millor
}
ROUNDS = {"p": 800, "c": 800, "load": 600, "pv": 800}


def train(samples):
    models = {}
    for tg, (X, y) in samples.items():
        ds = lgb.Dataset(X, y, feature_name=FEATS[tg], free_raw_data=False)
        prm = dict(TARGET_PARAMS[tg])
        if prm.get("objective") == "huber":
            prm["alpha"] = float(np.std(y)) * 0.5
        models[tg] = lgb.train(prm, ds, num_boost_round=ROUNDS[tg])
    return models


def main(final=False):
    raw = load_raw_hourly()
    store, hours = build_store(raw)
    all_blocks = {block_of(t) for t in raw.index}
    if final:
        models = train(make_samples(raw, store, hours, all_blocks))
        OUT.mkdir(exist_ok=True)
        for tg, m in models.items():
            m.save_model(str(OUT / f"model_{tg}.txt"))
        print("models desats a", OUT)
        return
    val_blocks = {b for b in all_blocks if b[2] == 2}
    tr = make_samples(raw, store, hours, all_blocks - val_blocks)
    va = make_samples(raw, store, hours, val_blocks)
    models = train(tr)
    for tg in models:
        X, y = va[tg]
        pred = models[tg].predict(X)
        k = X[:, FEATS[tg].index("k")]
        l24 = {"p": "x_l24", "c": "x_l24", "load": "load_l24", "pv": "pv_l24"}[tg]
        naive = X[:, FEATS[tg].index(l24)]
        ok = ~np.isnan(naive)
        blend = np.where(ok, 0.5 * pred + 0.5 * np.nan_to_num(naive), pred)
        print(f"      MAE mescla 50/50 model+ahir={np.mean(np.abs(blend - y)):.4f}")
        print(f"{tg:5s} n={len(y):6d}  MAE model={np.mean(np.abs(pred - y)):.4f}  "
              f"MAE model (amb ahir)={np.mean(np.abs(pred[ok] - y[ok])):.4f}  MAE 'ahir'={np.mean(np.abs(naive[ok] - y[ok])):.4f}"
              f"  mitjana y={np.mean(np.abs(y)):.4f}")
        for kk in (0, 3, 6, 12, 23):
            m = k == kk
            if m.any():
                print(f"       k={kk:2d}  MAE={np.mean(np.abs(pred[m] - y[m])):.4f}")
    OUT.mkdir(exist_ok=True)
    for tg, m in models.items():
        m.save_model(str(OUT / f"val_model_{tg}.txt"))


if __name__ == "__main__":
    main(final="--final" in sys.argv)
