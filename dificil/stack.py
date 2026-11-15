"""Experiments v13: correccions de les previsions a partir dels errors recents (causals), avaluades per replay."""
import sys, pickle
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exp import VAL, BLOCKS, replay
from train_forecast import COLS, epoch_hours

EV = epoch_hours(VAL.index).values
POS = {e: i for i, e in enumerate(EV)}
BID = np.zeros(len(EV), int)
for b, m in enumerate(BLOCKS):
    BID[m] = b
TRUTH = {k: dict(zip(EV.tolist(), VAL[COLS[k]].values.astype(float))) for k in ("price", "co2", "load", "pv")}
P = {"models": "models_fit7.pkl", "co2_cal": False}


def raw():
    return pickle.load(open(HERE / "results" / "fc_v12raw.pkl", "rb"))


def cal_v12(R, j=1, cap=0.06, tau=8.0, lag=3):
    """Calibratge de la v12 aplicat fora de línia (mateixa lògica causal)."""
    out = {}
    for t, v in R.items():
        a = [x.copy() for x in v]; f = a[j]
        old = R.get(t - lag)
        if old is not None and len(old[j]) >= 6 and all(t + i in TRUTH["co2" if j == 1 else "price"] for i in range(3)):
            key = "co2" if j == 1 else "price"
            e = np.array([TRUTH[key][t + i] - old[j][lag + i] for i in range(3)])
            if np.isfinite(e).all():
                b = float(np.clip(np.median(e), -cap, cap)); h = np.arange(3, len(f))
                f[3:] = np.clip(f[3:] + b * np.exp(-(h - 3) / tau), 0.0, 1.0 if j == 1 else 10.0)
        out[t] = tuple(a)
    return out


def score(tbl, extra=None):
    s, per, tot = replay(tbl, {**P, **(extra or {})})
    return s, per, tot


if __name__ == "__main__":
    R = raw()
    for n, t in (("raw", R), ("v12 offline", cal_v12(R))):
        s, per, tot = score(t)
        print(f"{n:12s}", s, " ".join(f"{x:5.1f}" for x in per), flush=True)


# ---------------------------------------------------------------- E1: corrector après del CO2
LAGS = range(1, 9)


def err_feats(R, t, j, key, ref=None):
    """Errors causals: valors ja coneguts (t..t+2) contra previsions fetes a t-L."""
    tr = TRUTH[key]; out = []
    for L in LAGS:
        old = R.get(t - L)
        for i in range(3):
            hh = L + i
            if hh < 3:
                continue
            v = np.nan
            if old is not None and hh < len(old[j]) and (t + i) in tr and (t - L) in POS and BID[POS[t - L]] == BID[POS[t]]:
                v = tr[t + i] - old[j][hh]
            out.append(v)
    return out


def dataset(R, j=1, key="co2"):
    X, y, T, H, B = [], [], [], [], []
    tr = TRUTH[key]
    for t, v in R.items():
        if t not in POS:
            continue
        f = v[j]; ef = err_feats(R, t, j, key)
        e3 = [x for x in ef[:0]]  # placeholder
        k = [tr.get(t + i, np.nan) for i in range(3)]
        prev = R.get(t - 1)
        med3 = np.nanmedian(ef[3:6]) if np.isfinite(ef[3:6]).any() else np.nan   # L=3 (horitzons 3,4,5) = v12
        for h in range(3, len(f)):
            if (t + h) not in POS or BID[POS[t + h]] != BID[POS[t]]:
                continue
            rev = f[h] - prev[j][h + 1] if (prev is not None and h + 1 < len(prev[j])) else np.nan
            X.append([h, t % 24, (t + h) % 24, f[h], k[0], k[1], k[2], f[h] - k[2], k[2] - k[0], rev, med3, *ef])
            y.append(tr[t + h] - f[h]); T.append(t); H.append(h); B.append(BID[POS[t]])
    return np.array(X, float), np.array(y), np.array(T), np.array(H), np.array(B)


def lobo(X, y, B, params, mono=None):
    import lightgbm as lgb
    pred = np.zeros(len(y))
    for b in np.unique(B):
        tr = B != b
        m = lgb.LGBMRegressor(**params).fit(X[tr], y[tr])
        pred[B == b] = m.predict(X[B == b])
    return pred


def apply(R, T, H, delta, j=1, hi=1.0):
    out = {t: tuple(x.copy() for x in v) for t, v in R.items()}
    for t, h, d in zip(T, H, delta):
        out[t][j][h] = float(np.clip(out[t][j][h] + d, 0.0, hi))
    return out


# ---------------------------------------------------------------- E2: escenaris de mercat correlacionats
def residuals(C):
    """Trajectòries d'error (veritat - previsió) per a cada t, preu i CO2, horitzons 0..23 (NaN fora de bloc)."""
    out = {}
    for t, v in C.items():
        if t not in POS:
            continue
        rp = np.full(24, np.nan); rc = np.full(24, np.nan)
        for h in range(3, min(24, len(v[0]))):
            if (t + h) in POS and BID[POS[t + h]] == BID[POS[t]]:
                rp[h] = TRUTH["price"][t + h] - v[0][h]; rc[h] = TRUTH["co2"][t + h] - v[1][h]
        out[t] = (rp, rc)
    return out


def make_scen(C, K=10, win=1, center=True, seed=0, scale=1.0):
    rng = np.random.default_rng(seed); Rs = residuals(C)
    full = [t for t, (rp, rc) in Rs.items() if np.isfinite(rc[3:]).all()]
    byh = {}
    for t in full:
        byh.setdefault(t % 24, []).append(t)
    out = {}
    for t, v in C.items():
        if t not in POS:
            continue
        b = BID[POS[t]]; h0 = t % 24
        pool = [u for d in range(-win, win + 1) for u in byh.get((h0 + d) % 24, []) if BID[POS[u]] != b]
        if len(pool) < K:
            continue
        pick = rng.choice(pool, K, replace=False)
        rp = np.array([Rs[u][0] for u in pick]); rc = np.array([Rs[u][1] for u in pick])
        rp = np.nan_to_num(rp); rc = np.nan_to_num(rc)
        if center:
            rp -= rp.mean(0); rc -= rc.mean(0)
        n = len(v[0])
        pS = v[0][None, :] + scale * rp[:, :n]; cS = np.clip(v[1][None, :] + scale * rc[:, :n], 0.001, 1.0)
        out[t] = (pS, cS)
    return out


def cal_meas(R, key, lags=(1, 2, 3), tau=4.0, cap=None, h0=1, shrink=1.0):
    """Calibratge online del consum i la solar amb els errors de previsions anteriors per a l hora t-1."""
    j = {"load": 2, "pv": 3}[key]; tr = TRUTH[key]; out = {}
    for t, v in R.items():
        a = [x.copy() for x in v]; f = a[j]
        e = []
        for L in lags:
            old = R.get(t - 1 - L)
            if old is not None and L < len(old[j]) and (t - 1) in tr and np.isfinite(tr[t - 1]):
                e.append(tr[t - 1] - old[j][L])
        if e:
            b = float(np.median(e)) * shrink
            if cap is not None:
                b = float(np.clip(b, -cap, cap))
            h = np.arange(h0, len(f))
            f[h0:] = f[h0:] + b * np.exp(-(h - h0) / tau)
            if key == "pv":
                f[h0:] = np.maximum(f[h0:], 0.0)
        out[t] = tuple(a)
    return out
