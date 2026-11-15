"""Model net0 (consum net de l'hora actual) entrenat amb finestres horàries desplaçades cada 5 min.

Les dades crues són cinc-minutals: a més de la graella oficial (hh:00-hh:59) es poden construir 11 graelles horàries més
desplaçades 5..55 min. Cada graella és una sèrie horària plausible del mateix campus, amb les mateixes relacions
consum(t) ~ f(consum(t-1), solar(t-1), meteo, calendari). Cada finestra s'etiqueta amb l'hora sencera més propera
(desplaçaments de -25 a +30 min), de manera que l'hora del dia s'equivoca com a molt 30 min.

    python dificil/diag/net0_aug.py train      # models_f<k>7aug.pkl (aug) i models_f<k>7ctl.pkl (control, només o=0)
    python dificil/diag/net0_aug.py mae        # MAE de l'hora actual a cada plec
    python dificil/diag/net0_aug.py score      # puntuació v12 amb cada net0 als 3 plecs (4 processos, 1 fil)
Només canvia el model net0; la resta de models són els mateixos de models_f<k>7.pkl.
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
TRAIN = ROOT / "data" / "train"
OFFSETS = list(range(-25, 35, 5))          # minuts; 0 = graella oficial
NET0_P = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def raw5():
    """Sèries cinc-minutals amb el mateix tractament que estandard/data.py i dificil/hdata.py (solar x12)."""
    c = pd.read_csv(TRAIN / "contador.csv", parse_dates=["Fecha"], usecols=["Fecha", "Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).set_index("Fecha")
    inv = pd.read_csv(TRAIN / "inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
    me = pd.read_csv(TRAIN / "meteo.csv", parse_dates=["Fecha"]).set_index("Fecha")
    load = (-c["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(0, 5000)
    cols = [[x for x in inv.columns if f"INVERSOR {i}" in x.upper() and "POTENCIA ACTIVA" in x.upper()
             and "FASE" not in x.upper()][0] for i in (1, 2, 3)]
    pv = sum(inv[k].fillna(0) for k in cols).clip(lower=0) * 12.0
    return pd.DataFrame({"load": load, "pv": pv,
                         "irr": me["Atersa - IRRADIANCIA (W/m2)"],
                         "temp": me["Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)"],
                         "vent": me["Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)"]})


def shifted_hist(r5, off, allowed):
    """Historial horari de la graella desplaçada `off` minuts. Finestra [H+off, H+off+1h) -> hora H.
    Només finestres totalment dins d'hores oficials permeses (sense tocar la validació del plec)."""
    from feats import Hist
    r = r5.copy(); r.index = r.index - pd.Timedelta(minutes=off)      # ara la finestra comença a H
    g = r.resample("1h")
    hr = g.mean(); n = g["load"].count()
    hr = hr[n >= 10]
    # la finestra [H+off, H+off+1h) toca les hores oficials floor(H+off) i floor(H+off+59min)
    a = (hr.index + pd.Timedelta(minutes=off)).floor("h"); b = (hr.index + pd.Timedelta(minutes=off + 55)).floor("h")
    ok = a.isin(allowed) & b.isin(allowed)
    hr = hr[ok]
    h = Hist(); eh = ((hr.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h")).astype(int).values
    for k in ("load", "pv", "irr", "temp", "vent"):
        for e, v in zip(eh, hr[k].values):
            h.put(k, e, v)
    return h, eh


def net0_rows(hist, eh):
    import feats
    X, y = [], []
    for t in eh:
        yv = hist.get("load", t) - hist.get("pv", t)
        if not np.isfinite(yv):
            continue
        X.append(np.concatenate([feats.load_features2(hist, t, [0])[0], feats.meas_features(hist, "pv", t, [0])[0]]))
        y.append(yv)
    return np.array(X), np.array(y)


def train(only=()):
    import joblib, lightgbm as lgb, feats
    from hdata import load_hard
    from folds import split_fold
    df = load_hard(); r5 = raw5()
    for k in (1, 2, 3):
        t0 = time.time()
        tr = split_fold(df, k)[0]
        base = joblib.load(AH / f"models_f{k}7.pkl")
        feats.set_cs(base["pv_cs"])
        allowed = pd.DatetimeIndex(tr.index)
        data = {}
        for off in OFFSETS:
            data[off] = net0_rows(*shifted_hist(r5, off, allowed))
        for name, offs, fo in (("ctl", [0], False), ("aug", OFFSETS, False), ("aug10", [-10, -5, 0, 5, 10], False),
                               ("augf", OFFSETS, True)):
            if only and name not in only:
                continue
            X = np.vstack([np.column_stack([data[o][0], np.full(len(data[o][1]), o)]) if fo else data[o][0] for o in offs])
            y = np.concatenate([data[o][1] for o in offs])
            m = lgb.LGBMRegressor(**NET0_P).fit(X, y)
            out = dict(base); out["net0"] = m.booster_.model_to_string()
            joblib.dump(out, AH / f"models_f{k}7{name}.pkl", compress=3)
            print(f"plec {k} {name}: {len(y)} files", flush=True)
        print(f"plec {k}: {time.time() - t0:.0f} s", flush=True)


def mae():
    import joblib, lightgbm as lgb, feats
    from hdata import load_hard
    from folds import split_fold
    from train_forecast import full_hist
    df = load_hard()
    hist, _ = full_hist(df)                       # a l'hora t l'agent coneix consum/solar fins a t-1 (tot real)
    for k in (1, 2, 3):
        val = split_fold(df, k)[1]
        eh = ((val.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h")).astype(int).values
        res = {}
        for name in ("", "ctl", "aug", "aug10", "augf"):
            d = joblib.load(AH / f"models_f{k}7{name}.pkl"); feats.set_cs(d["pv_cs"])
            b = lgb.Booster(model_str=d["net0"])
            X = np.array([np.concatenate([feats.load_features2(hist, t, [0])[0], feats.meas_features(hist, "pv", t, [0])[0]]) for t in eh])
            if name == "augf":
                X = np.column_stack([X, np.zeros(len(X))])
            res[name or "v12"] = b.predict(X)
        y = (val.load_kw - val.pv_kw).values
        hod = val.index.hour.values
        line = "  ".join(f"{n} {np.mean(np.abs(p - y)):.1f}" for n, p in res.items())
        day = (hod >= 9) & (hod <= 16)
        line2 = "  ".join(f"{n} {np.mean(np.abs(p - y)[day]):.1f}" for n, p in res.items())
        print(f"plec {k}  MAE net h0: {line}   | 9-16 h: {line2}", flush=True)


VARS = {"ctl": {}, "aug": {}, "aug10": {}, "aug_mix8": {"net0_mix": 0.8}}


def one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    name, k = job
    from arena import fold_data
    from common import load_base
    from henv import HardEnv
    val = fold_data(k)
    tag = name.split("_")[0]
    ag = load_base().BatteryAgent({"models": f"models_f{k}7{tag}.pkl", **VARS[name]})
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
    if cmd == "train":
        train(tuple(sys.argv[2:]))
    elif cmd == "mae":
        mae()
    else:
        score(sys.argv[2:] or ["ctl", "aug"])
