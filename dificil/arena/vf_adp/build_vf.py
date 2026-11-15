"""Aprèn el valor marginal de l'energia emmagatzemada (programació dinàmica aproximada).

Objectiu: per a cada hora t del train i cada nivell de SOC s, lambda(t+3, s) = dual de la restricció de SOC inicial
d'un LP amb previsió perfecta de 48 h que comença a t+3 amb SOC s (= quant val 1 kWh més a la bateria).
Model: LightGBM monòton decreixent en el SOC, sobre lambda / escala (escala = mitjana de w de les últimes 24 h).

    python dificil/arena/vf_adp/build_vf.py f1 f2 f3 all     -> dificil/arena/vf_adp/vf_<k>.pkl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, time, pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "dificil"), str(ROOT / "estandard"), str(ROOT / "dificil" / "agent_hard"), str(HERE)]
from vf_feats import base_features, with_soc, SOC_MID, SOC_COL  # noqa: E402
from feats import Hist  # noqa: E402

ETA = 0.9 ** 0.5
RATIO = 0.6213
H = 48
CAP, PM = 6000.0, 1500.0


def lp_struct(H):
    I = sp.identity(H, format="csr"); Z = sp.csr_matrix((H, H))
    D = sp.diags([np.ones(H), -np.ones(H - 1)], [0, -1], format="csr")
    Aeq = sp.hstack([-ETA * I, I / ETA, D, Z]).tocsr()
    Aub = sp.hstack([I, -I, Z, -I]).tocsr()
    bnd = [(0, PM)] * (2 * H) + [(0, CAP)] * H + [(0, None)] * H
    return Aeq, Aub, bnd


def lambdas(w, net, socs, st):
    Aeq, Aub, bnd = st
    Hh = len(w)
    vend = ETA * float(np.median(w))
    cost = np.concatenate([np.zeros(2 * Hh), np.zeros(Hh), w]); cost[3 * Hh - 1] = -vend
    out = []
    for s in socs:
        beq = np.zeros(Hh); beq[0] = s
        r = linprog(cost, A_ub=Aub, b_ub=-net, A_eq=Aeq, b_eq=beq, bounds=bnd, method="highs")
        out.append(-r.eqlin.marginals[0] if r.status == 0 else np.nan)
    return np.array(out)


def segments(df):
    gid = (df.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
    return [df[gid == g] for g in np.unique(gid)]


def build(tag):
    t0 = time.time()
    from hdata import load_hard
    from folds import split_fold
    df = load_hard()
    fit = df if tag == "all" else split_fold(df, int(tag[1:]))[0]
    ep = fit.index.values.astype("datetime64[h]").astype(np.int64)
    hist = Hist()
    for col, key in (("precio_eur_per_kwh", "price"), ("co2_kg_per_kwh", "co2"), ("load_kw", "load"),
                     ("pv_kw", "pv"), ("irradiancia_wm2", "irr")):
        for e, v in zip(ep, fit[col].values):
            hist.put(key, e, v)
    st = lp_struct(H)
    X, Y = [], []
    for seg in segments(fit):
        e = seg.index.values.astype("datetime64[h]").astype(np.int64)
        w = 0.7 * seg.co2_kg_per_kwh.values + 0.3 * RATIO * seg.precio_eur_per_kwh.values
        net = (seg.load_kw - seg.pv_kw).values
        for i in range(0, len(seg) - 3 - H + 1):
            lam = lambdas(w[i + 3:i + 3 + H], net[i + 3:i + 3 + H], SOC_MID, st)
            f, scale = base_features(hist.get, int(e[i]), RATIO)
            X.append(with_soc(f, SOC_MID)); Y.append(lam / scale)
    X = np.vstack(X); Y = np.concatenate(Y)
    ok = np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    import lightgbm as lgb
    mono = [0] * X.shape[1]; mono[SOC_COL] = -1
    prm = dict(objective="l2", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, num_threads=1,
               verbose=-1, seed=1)
    b = lgb.train(prm, lgb.Dataset(X, Y), num_boost_round=800)
    pickle.dump({"vf": b.model_to_string(), "n": len(Y), "H": H, "ratio": RATIO},
                open(HERE / f"vf_{tag}.pkl", "wb"))
    np.savez_compressed(HERE / f"_train_{tag}.npz", X=X.astype(np.float32), Y=Y.astype(np.float32))
    return tag, len(Y), time.time() - t0


if __name__ == "__main__":
    tags = sys.argv[1:] or ["f1"]
    with Pool(len(tags)) as p:
        for tag, n, dt in p.imap_unordered(build, tags):
            print(f"{tag}: {n} mostres, {dt:.0f} s", flush=True)
