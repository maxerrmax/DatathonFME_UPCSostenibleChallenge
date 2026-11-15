"""Branca C: correcció residual apresa de l'acció de la v12 (etiquetes = acció de l'agent amb informació perfecta
en el MATEIX estat, estil DAgger). Avaluació en llaç tancat amb models leave-one-block-out."""
import sys, pickle
from pathlib import Path
import numpy as np, lightgbm as lgb
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from stack import raw, cal_v12, TRUTH, POS, BID, EV, P as PBASE
from exp import VAL, BLOCKS
from henv import HardEnv
from heval import load_agent
from train_forecast import COLS

C = cal_v12(raw())
TBL = {k: dict(zip(EV.tolist(), VAL[c].values.tolist())) for k, c in COLS.items() if k in ("price", "co2", "load", "pv")}


def feats(ag, o, t):
    fc = C.get(t)
    soc = float(o["soc_kwh"]); cap = 6000.0
    if fc is None:
        return None
    pr, co, ld, pv = [np.asarray(x, float) for x in fc]
    w = 0.7 * co + 0.3 * 0.6213 * pr; net = ld - pv
    return [soc / cap, t % 24, (t // 24 + 4) % 7, w[0], w[:3].mean(), w.max(), w.min(), (w[0] - w.mean()) / (w.std() + 1e-9),
            float(np.mean(w[1:] > w[0])), net[0], net[:6].sum(), net.sum(), np.clip(-net, 0, None).sum(), pv[:12].sum()]


def collect():
    ag = load_agent({**PBASE, "_fc_table": C}); pf = load_agent({"models": "models_fit7.pkl", "_cheat": (TBL, {"price", "co2", "load", "pv"})})
    env = HardEnv(VAL.reset_index()); o = env.reset(); d = False; X, A, Y, B = [], [], [], []
    i = 0
    while not d:
        t = int(EV[i]); a = ag.act(o); b = pf.act(o)          # mateix estat
        f = feats(ag, o, t)
        if f is not None:
            X.append(f + [a]); A.append(a); Y.append(b - a); B.append(BID[i])
        o, d = env.step(a); i += 1
    return np.array(X), np.array(A), np.array(Y), np.array(B)


def run_with(models, lam):
    ag = load_agent({**PBASE, "_fc_table": C}); env = HardEnv(VAL.reset_index()); o = env.reset(); d = False; i = 0
    while not d:
        t = int(EV[i]); a = ag.act(o); f = feats(ag, o, t)
        if f is not None and lam > 0:
            a = ag._clamp(float(o["soc_kwh"]), a + lam * float(models[BID[i]].predict(np.array([f + [a]]))[0]))
        o, d = env.step(a); i += 1
    per = [env.evaluate_results(mask=m)["score"] for m in BLOCKS]
    return env.evaluate_results()["score"], per


if __name__ == "__main__":
    X, A, Y, B = collect()
    print("estats", X.shape, "|residu| mitjà", round(np.abs(Y).mean(), 1), "kW", flush=True)
    pp = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1)
    models = {b: lgb.LGBMRegressor(**pp).fit(X[B != b], Y[B != b]) for b in np.unique(B)}
    pred = np.concatenate([models[b].predict(X[B == b]) for b in np.unique(B)])
    yy = np.concatenate([Y[B == b] for b in np.unique(B)])
    print("R2 fora de bloc", round(1 - np.mean((yy - pred) ** 2) / np.var(yy), 3), flush=True)
    for lam in (0.0, 0.3, 0.6, 1.0):
        s, per = run_with(models, lam); print("lambda", lam, s, " ".join(f"{x:5.1f}" for x in per), flush=True)
