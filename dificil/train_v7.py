"""v7: preu/CO2 amb features ampliades (ensemble de 4 models residuals), consum/solar sense lag setmanal,
i historial inicial (dades d'entrenament) dins del fitxer de models.
Ús: python dificil/train_v7.py [fit|all]"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, joblib, lightgbm as lgb
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "agent_hard")); sys.path.insert(0, str(HERE.parent / "estandard"))
from hdata import load_hard
from data import split_val
import feats
from train_forecast import full_hist, epoch_hours, make_cs, build, COLS

N = feats.MKT2_NAMES
FS = {"no168": [n for n in N[:35] if n not in ("lag168", "wk7", "wk7_o")],
      "core": ["h", "hod", "dow", "doy_s", "doy_c", "k0", "k1", "k2", "lag24", "lag48", "shift", "x_shift", "m24",
               "mn24", "mx24", "sd24", "rel24", "o_k2", "o_lag24", "o_shift", "o_m24", "hod_t"]}
P = dict(n_estimators=500, learning_rate=0.04, num_leaves=63, min_child_samples=20,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)


def base_of(X, kind):
    k2 = X[:, N.index("k2")]
    return np.where(np.isfinite(X[:, N.index("x_shift")]), X[:, N.index("x_shift")], k2) if kind == "xs" else k2


def mkt2_matrix(tr, key):
    hist, eh = full_hist(tr); present = set(eh.tolist()); X, y = [], []
    for t in eh:
        hs = [h for h in range(3, 24) if t + h in present]
        if hs:
            X.append(feats.mkt_features2(hist, key, t, hs)); y += [hist.get(key, t + h) for h in hs]
    return np.vstack(X), np.array(y)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "fit"
    df = load_hard(); fit, val = split_val(df); tr = fit if which == "fit" else df
    CS = make_cs(tr); feats.set_cs(CS); t0 = time.time()
    dump = {"pv_cs": CS, "mkt2": {}}
    for key in ("price", "co2"):
        X, y = mkt2_matrix(tr, key); ms = []
        if key == "price":                               # preu: residu additiu
            for fs, names in FS.items():
                cols = [N.index(n) for n in names]
                for kind in ("k2", "xs"):
                    b = base_of(X, kind); ok = np.isfinite(b)
                    m = lgb.LGBMRegressor(**P).fit(X[ok][:, cols], (y - b)[ok])
                    ms.append((m.booster_.model_to_string(), cols, kind))
        else:                                            # CO2: logaritme del quocient (millor MAE i score)
            k2 = X[:, N.index("k2")]; a = X[:, N.index("x_shift")]
            bases = {"logk2": k2, "logxs": np.where(np.isfinite(a) & (a > 0.3 * k2), a, k2)}
            for names in (FS["core"] + ["km1", "km2", "km3", "slope", "l23", "l25"],
                          [n for n in N if n not in ("lag168", "wk7", "wk7_o", "wk3")]):
                cols = [N.index(n) for n in names]
                for kind, b in bases.items():
                    ok = np.isfinite(b) & (b > 0) & (y > 0)
                    m = lgb.LGBMRegressor(**P).fit(X[ok][:, cols], np.log(y[ok] / b[ok]))
                    ms.append((m.booster_.model_to_string(), cols, kind))
        dump["mkt2"][key] = ms
        # models per tram d'horitzó (+3..+5, +6..+11): CO2 en log-quocient, preu en residu additiu
        cols = [N.index(n) for n in N if n not in ("lag168", "wk7", "wk7_o", "wk3")]
        Hh = X[:, N.index("h")]; k2 = X[:, N.index("k2")]; grp = []
        for a, b_ in ((3, 5), (6, 11)):
            sel = (Hh >= a) & (Hh <= b_) & np.isfinite(y)
            if key == "co2":
                sel &= (k2 > 0) & (y > 0)
                m = lgb.LGBMRegressor(**P).fit(X[sel][:, cols], np.log(y[sel] / k2[sel])); kind = "log"
            else:
                sel &= np.isfinite(k2)
                m = lgb.LGBMRegressor(**P).fit(X[sel][:, cols], (y - k2)[sel]); kind = "add"
            grp.append((a, b_, m.booster_.model_to_string(), cols, kind))
        dump.setdefault("mktg", {})[key] = grp
    # consum v2: residu respecte a lag24 i respecte al perfil d'ahir desplaçat
    L = feats.LOAD2_NAMES; hist, eh = full_hist(tr); present = set(eh.tolist()); X, y = [], []
    for t in eh:
        hs = [h for h in range(24) if t + h in present]
        X.append(feats.load_features2(hist, t, hs)); y += [hist.get("load", t + h) for h in hs]
    X = np.vstack(X); y = np.array(y)
    def fb(*names):
        b = X[:, L.index(names[0])]
        for n in names[1:]:
            b = np.where(np.isfinite(b), b, X[:, L.index(n)])
        return np.where(np.isfinite(b), b, 1000.0)
    cols = [L.index(n) for n in L if n not in ("wk7", "same", "same_m", "same_shift")]; ok = np.isfinite(y)
    dump["load2"] = []
    for kind, b in (("lag24", fb("lag24", "last", "m24")), ("xs", fb("x_shift", "lag24", "last", "m24"))):
        m = lgb.LGBMRegressor(**P).fit(X[ok][:, cols], (y - b)[ok])
        dump["load2"].append((m.booster_.model_to_string(), cols, kind))
    # model directe del consum net de l'hora actual (features de consum + solar a h=0)
    Xn, yn = [], []
    for t in eh:
        Xn.append(np.concatenate([feats.load_features2(hist, t, [0])[0], feats.meas_features(hist, "pv", t, [0])[0]]))
        yn.append(hist.get("load", t) - hist.get("pv", t))
    Xn = np.array(Xn); yn = np.array(yn); okn = np.isfinite(yn)
    dump["net0"] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=30, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(Xn[okn], yn[okn]).booster_.model_to_string()
    D = build(tr)
    for k in ("pv",):
        X, y = D[k]; X = X.copy(); X[:, 7] = np.nan; lag = X[:, 5]; ok = np.isfinite(lag)
        dump[k + "_dir"] = lgb.LGBMRegressor(**P).fit(X, y).booster_.model_to_string()
        dump[k + "_res"] = lgb.LGBMRegressor(**P).fit(X[ok], (y - lag)[ok]).booster_.model_to_string()
    eh = epoch_hours(tr.index).values
    dump["hist0"] = {k: (eh.astype(np.int64), tr[c].values.astype(float)) for k, c in COLS.items()}
    out = HERE / "agent_hard" / f"models_{which}7.pkl"; joblib.dump(dump, out, compress=3)
    print("desat", out, f"{time.time()-t0:.0f}s", f"{out.stat().st_size/1e6:.1f} MB")
