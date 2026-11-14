"""Fitted Q-Iteration (RL offline per lots) amb LightGBM.
Transicions sintètiques: per a cada hora de 'fit', SOCs aleatoris × 5 accions discretes.
Recompensa = -w_t * P_t / 50 (estalvi ponderat CO2+cost). Ús: python pipeline/agents/rl_fqi/train.py"""
import sys
from pathlib import Path
import numpy as np, joblib, lightgbm as lgb
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
from common import ACTIONS, wvec, feat, clamp, step_soc
from data import load_hourly, split_val

GAMMA, ITERS, NSOC = 0.97, 30, 12
rng = np.random.default_rng(0)
fit, _ = split_val(load_hourly())
co2 = fit.co2_kg_per_kwh.values; pr = fit.precio_eur_per_kwh.values
T = len(fit) - 1
W = [wvec(co2[t:t + 24], pr[t:t + 24]) for t in range(T + 1)]
S, A, R, S2 = [], [], [], []
for t in range(T):
    for soc in rng.uniform(0, 2000, NSOC):
        for a in ACTIONS:
            p = clamp(soc, a)
            S.append(feat(soc, W[t])); A.append(a); R.append(-W[t][0] * p / 50.0)
            S2.append((step_soc(soc, p), t + 1))
X = np.column_stack([np.array(S), np.array(A) / 500.0]); R = np.array(R)
nxt_states = np.array([feat(s, W[t]) for s, t in S2])
nxt_soc = np.array([s for s, _ in S2])
print("transicions", len(X))
model = None
for it in range(ITERS):
    if model is None:
        y = R
    else:
        qs = []
        for a in ACTIONS:
            # accions no factibles es clampen, igual que a l'agent
            qs.append(model.predict(np.column_stack([nxt_states, np.full(len(nxt_states), a / 500.0)])))
        y = R + GAMMA * np.max(np.column_stack(qs), axis=1)
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63, verbose=-1)
    model.fit(X, y)
    print("iter", it, "mean target", round(float(y.mean()), 3), flush=True)
joblib.dump(model, HERE / "model.pkl")
