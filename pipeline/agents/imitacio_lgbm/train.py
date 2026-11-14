"""Behavior cloning: LightGBM imita les accions de l'oracle LP (previsió perfecta)
sobre la partició 'fit'. Ús: python pipeline/agents/imitacio_lgbm/train.py"""
import sys
from pathlib import Path
import numpy as np, joblib, lightgbm as lgb
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from data import load_hourly, split_val, blocks
from oracle import oracle_actions
from features import obs_features

fit, _ = split_val(load_hourly())
X, y = [], []
P, S = oracle_actions(fit)
soc_before = np.concatenate([[1000.0], S[:-1]])
co2 = fit.co2_kg_per_kwh.values; pr = fit.precio_eur_per_kwh.values; idx = fit.index
for t in range(len(fit)):
    X.append(obs_features(soc_before[t], co2[t:t + 24], pr[t:t + 24], idx[t].hour, idx[t].dayofweek))
    y.append(P[t])
m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=10, verbose=-1)
m.fit(np.array(X), np.array(y))
joblib.dump(m, HERE / "model.pkl")
print("ok", len(X))
