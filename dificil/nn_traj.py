"""Branca A: previsor neuronal de la trajectòria de CO2 (+3..+23 h) amb una MLP sobre l'última setmana.
Entrenament només amb fit; avaluació per replay combinant-lo amb els LightGBM de la v12."""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exp import FIT, VAL
from train_forecast import epoch_hours

LB = 168
KEYS = ("co2", "price")
COL = {"co2": "co2_kg_per_kwh", "price": "precio_eur_per_kwh"}


def series(df):
    eh = epoch_hours(df.index).values
    d = {k: dict(zip(eh.tolist(), df[COL[k]].values.astype(float))) for k in KEYS}
    for k, c in (("irr", "irradiancia_wm2"), ("temp", "temp_ambient_c"), ("vent", "vent_ms")):
        if c in df.columns:
            d[k] = dict(zip(eh.tolist(), df[c].values.astype(float)))
    return d, eh


def build(df_hist, df_t, target=True):
    S, _ = series(df_hist); _, ts = series(df_t)
    X, Y, M, T, K2 = [], [], [], [], []
    for t in ts:
        k2 = S["co2"].get(t + 2, np.nan); p2 = S["price"].get(t + 2, np.nan)
        if not (np.isfinite(k2) and k2 > 0 and np.isfinite(p2) and p2 > 0):
            continue
        feats = []
        for k, ref in (("co2", k2), ("price", p2)):
            v = np.array([S[k].get(t + 2 - i, np.nan) for i in range(LB + 3)])
            feats.append(np.nan_to_num(np.log(np.maximum(v, 1e-4) / ref), nan=0.0).clip(-3, 3))
            feats.append(np.isfinite(v).astype(float)[::12])
        ts_ = pd.Timestamp(int(t) * 3600, unit="s")
        cal = np.zeros(24 + 7 + 2); cal[ts_.hour] = 1; cal[24 + ts_.dayofweek] = 1
        doy = ts_.dayofyear / 365.25 * 2 * np.pi; cal[31] = np.sin(doy); cal[32] = np.cos(doy)
        met = [np.nan_to_num(S.get("irr", {}).get(t - 1, np.nan)) / 1000, np.nan_to_num(S.get("temp", {}).get(t - 1, np.nan)) / 40,
               np.nan_to_num(S.get("vent", {}).get(t - 1, np.nan)) / 10, np.log(k2), np.log(p2)]
        X.append(np.concatenate(feats + [cal, met]))
        y = np.array([S["co2"].get(t + h, np.nan) for h in range(3, 24)]) if target else np.full(21, np.nan)
        M.append(np.isfinite(y)); Y.append(np.nan_to_num(np.log(np.maximum(y, 1e-4) / k2))); T.append(t); K2.append(k2)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(M), np.array(T), np.array(K2)


class Net(nn.Module):
    def __init__(self, d, hdim=256, p=0.2):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, hdim), nn.GELU(), nn.Dropout(p), nn.Linear(hdim, hdim), nn.GELU(), nn.Dropout(p), nn.Linear(hdim, 21))

    def forward(self, x):
        return self.f(x)


def train(X, Y, M, seed=0, epochs=60, loss="l1", hdim=256, p=0.2, wd=1e-4, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(X); idx = np.random.permutation(n); va = idx[: n // 10]; tr = idx[n // 10:]
    net = Net(X.shape[1], hdim, p); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt, Mt = map(torch.tensor, (X, Y, M.astype(np.float32)))
    best, bst = 1e9, None
    for ep in range(epochs):
        net.train(); perm = torch.tensor(np.random.permutation(tr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; out = net(Xt[b])
            e = (out - Yt[b]) * Mt[b]
            l = e.abs().mean() if loss == "l1" else (e ** 2).mean()
            opt.zero_grad(); l.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            vl = (((net(Xt[va]) - Yt[va]) * Mt[va]).abs().sum() / Mt[va].sum()).item()
        if vl < best:
            best, bst = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bst); return net, best


def predict(net, X, K2):
    net.eval()
    with torch.no_grad():
        return K2[:, None] * np.exp(net(torch.tensor(X)).numpy())


if __name__ == "__main__":
    torch.set_num_threads(2)
    t0 = time.time()
    Xf, Yf, Mf, Tf, _ = build(FIT, FIT)
    Xv, Yv, Mv, Tv, K2v = build(pd.concat([FIT, VAL]).sort_index(), VAL)
    print("dades", Xf.shape, Xv.shape, f"{time.time()-t0:.0f}s", flush=True)
    out = {}
    for name, kw in (("l1", dict(loss="l1")), ("l1_s1", dict(loss="l1", seed=1)), ("l1_s2", dict(loss="l1", seed=2))):
        net, b = train(Xf, Yf, Mf, **kw); P = predict(net, Xv, K2v); out[name] = P
        y = K2v[:, None] * np.exp(Yv); err = np.abs(P - y)[Mv]
        print(name, "val-intern", round(b, 4), "MAE CO2 val", round(err.mean(), 5), f"{time.time()-t0:.0f}s", flush=True)
    out["ens3"] = np.mean([out[k] for k in ("l1", "l1_s1", "l1_s2")], axis=0)
    y = K2v[:, None] * np.exp(Yv); print("ens3 MAE", round(np.abs(out["ens3"] - y)[Mv].mean(), 5))
    pickle.dump((Tv, out), open(HERE / "results" / "nn_traj.pkl", "wb"))
