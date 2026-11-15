"""Models de preu/CO2 especialitzats en l'horitzó proper (h=3..11), on hi ha tot el valor de la informació de mercat
(previsió perfecta de 3..11 = +0,60 = previsió perfecta de 3..23 als 3 plecs).

Mateixes features, bases i hiperparàmetres que dificil/train_v7.py; només canvia el conjunt d'entrenament (h=3..11).
L'agent fa servir els models propers per a h<=11 i els originals per a h>=12.

    python dificil/diag/mkt_near.py train      # dificil/agent_hard/models_f<k>7near.pkl (només els models propers)
    python dificil/diag/mkt_near.py mae
    python dificil/diag/mkt_near.py score      # v12 + models propers (CO2 i preu / només CO2), 3 plecs
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dificil"), str(ROOT / "dificil" / "agent_hard"), str(ROOT / "estandard"),
                str(ROOT / "dificil" / "arena"), str(ROOT / "scripts")]
AH = ROOT / "dificil" / "agent_hard"
HS = range(3, 12)


def matrix(hist, eh, key):
    import feats
    present = set(np.asarray(eh).tolist()); X, y = [], []
    for t in eh:
        hs = [h for h in HS if t + h in present]
        if hs:
            X.append(feats.mkt_features2(hist, key, t, hs)); y += [hist.get(key, t + h) for h in hs]
    return np.vstack(X), np.array(y)


def fit_models(X, y, key):
    """Còpia de la part de mercat de train_v7.py (mateixes bases i conjunts de features)."""
    import lightgbm as lgb
    from train_v7 import FS, P, N, base_of
    ms = []
    if key == "price":
        for fs, names in FS.items():
            cols = [N.index(n) for n in names]
            for kind in ("k2", "xs"):
                b = base_of(X, kind); ok = np.isfinite(b)
                m = lgb.LGBMRegressor(**P).fit(X[ok][:, cols], (y - b)[ok])
                ms.append((m.booster_.model_to_string(), cols, kind))
    else:
        k2 = X[:, N.index("k2")]; a = X[:, N.index("x_shift")]
        bases = {"logk2": k2, "logxs": np.where(np.isfinite(a) & (a > 0.3 * k2), a, k2)}
        for names in (FS["core"] + ["km1", "km2", "km3", "slope", "l23", "l25"],
                      [n for n in N if n not in ("lag168", "wk7", "wk7_o", "wk3")]):
            cols = [N.index(n) for n in names]
            for kind, b in bases.items():
                ok = np.isfinite(b) & (b > 0) & (y > 0)
                m = lgb.LGBMRegressor(**P).fit(X[ok][:, cols], np.log(y[ok] / b[ok]))
                ms.append((m.booster_.model_to_string(), cols, kind))
    return ms


def train():
    import joblib, feats
    from hdata import load_hard
    from folds import split_fold
    from train_forecast import full_hist
    df = load_hard()
    for k in (1, 2, 3):
        t0 = time.time()
        tr = split_fold(df, k)[0]
        feats.set_cs(joblib.load(AH / f"models_f{k}7.pkl")["pv_cs"])
        hist, eh = full_hist(tr)
        out = {key: fit_models(*matrix(hist, eh, key), key) for key in ("price", "co2")}
        joblib.dump(out, AH / f"models_f{k}7near.pkl", compress=3)
        print(f"plec {k}: {time.time() - t0:.0f} s", flush=True)


def _pred(models, X):
    """Igual que BatteryAgent._pred_mkt2 però amb una matriu de features donada."""
    import lightgbm as lgb
    k2 = X[:, 7]; xs = np.where(np.isfinite(X[:, 13]), X[:, 13], k2)
    xsl = np.where(np.isfinite(X[:, 13]) & (X[:, 13] > 0.3 * k2), X[:, 13], k2)
    out = []
    for b, cols, base in models:
        b = lgb.Booster(model_str=b) if isinstance(b, str) else b
        pr = b.predict(X[:, cols], num_threads=1)
        if base == "logk2":
            out.append(np.where(k2 > 0, k2 * np.exp(pr), k2))
        elif base == "logxs":
            out.append(np.where(xsl > 0, xsl * np.exp(pr), xsl))
        else:
            out.append((xs if base == "xs" else k2) + pr)
    return np.mean(out, axis=0)


def mae():
    import joblib, feats
    from hdata import load_hard
    from folds import split_fold
    from train_forecast import full_hist
    df = load_hard(); hist, _ = full_hist(df)
    for k in (1, 2, 3):
        val = split_fold(df, k)[1]
        eh = ((val.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h")).astype(int).values
        present = set(eh.tolist())
        base = joblib.load(AH / f"models_f{k}7.pkl"); near = joblib.load(AH / f"models_f{k}7near.pkl")
        feats.set_cs(base["pv_cs"])
        for key in ("co2", "price"):
            X, y, H = [], [], []
            for t in eh:
                hs = [h for h in HS if t + h in present]
                if hs:
                    X.append(feats.mkt_features2(hist, key, t, hs)); y += [hist.get(key, t + h) for h in hs]; H += hs
            X = np.vstack(X); y = np.array(y); H = np.array(H)
            pb = _pred(base["mkt2"][key], X); pn = _pred(near[key], X)
            eb, en = np.abs(pb - y), np.abs(pn - y)
            by = "  ".join(f"h{h}:{100 * (en[H == h].mean() / eb[H == h].mean() - 1):+.1f}%" for h in (3, 5, 8, 11))
            print(f"plec {k} {key:<5} MAE base {eb.mean():.5f} near {en.mean():.5f} ({100 * (en.mean() / eb.mean() - 1):+.1f}%)  {by}",
                  flush=True)


VARS = {"near": ("price", "co2"), "near_co2": ("co2",)}


def one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    name, k = job
    import joblib, lightgbm as lgb
    from arena import fold_data
    from common import load_base
    from henv import HardEnv
    import feats
    val = fold_data(k)
    near = joblib.load(AH / f"models_f{k}7near.pkl")
    near = {key: [(lgb.Booster(model_str=ms), list(cols), base) for ms, cols, base in v] for key, v in near.items()}
    keys = VARS[name]
    Base = load_base().BatteryAgent

    class A(Base):
        def _pred_mkt2(self, key, t, hs):
            out = super()._pred_mkt2(key, t, hs)
            if key in keys:
                idx = [i for i, h in enumerate(hs) if h <= 11]
                if idx:
                    X = feats.mkt_features2(self.hist, key, t, [hs[i] for i in idx])
                    out = np.asarray(out, float).copy(); out[idx] = _pred(near[key], X)
            return out

    ag = A({"models": f"models_f{k}7.pkl"})
    env = HardEnv(val.reset_index()); o = env.reset(); d = False
    while not d:
        o, d = env.step(ag.act(o))
    r = env.evaluate_results()
    return name, k, r["score"], r["co2_estalviat_pct"], r["cost_estalviat_pct"]


def score(names):
    from multiprocessing import Pool
    V12 = {1: 20.74, 2: 17.42, 3: 18.90}
    R = {}
    with Pool(4) as p:
        for n, k, s, c, e in p.imap_unordered(one, [(n, k) for n in names for k in (1, 2, 3)]):
            R[(n, k)] = s; print(n, k, s, c, e, flush=True)
    for n in names:
        v = [R[(n, k)] for k in (1, 2, 3)]
        print(f"{n:<10}", v, f"mitjana {np.mean(v):.3f}  delta v12 {np.mean(v) - np.mean(list(V12.values())):+.3f}",
              "per plec", [round(R[(n, k)] - V12[k], 2) for k in (1, 2, 3)])


if __name__ == "__main__":
    for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "mae": mae}.get(cmd, lambda: score(sys.argv[2:] or list(VARS)))()
