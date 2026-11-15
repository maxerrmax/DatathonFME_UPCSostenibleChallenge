"""Variants al voltant dels models de CO2 propers (mkt_near.py), als 3 plecs.

  near_co2   v12 + CO2 propers (h=3..11)                                     (referència d'aquesta tanda)
  nocal      near_co2 sense el calibratge en línia del CO2 del v12              (es solapen?)
  avg        mitjana CO2 proper / original per a h=3..11
  bands      CO2 amb models per franges d'horitzó (3-5, 6-8, 9-11)
  stack      v19 (previsió al canvi de setmana) + CO2 propers
  v19        v19 sol (referència del stack)

    python dificil/diag/strats_near.py train_bands
    python dificil/diag/strats_near.py score [variants]
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dificil" / "diag"), str(ROOT / "dificil"), str(ROOT / "dificil" / "agent_hard"),
                str(ROOT / "estandard"), str(ROOT / "dificil" / "arena"), str(ROOT / "scripts")]
AH = ROOT / "dificil" / "agent_hard"
BANDS = [range(3, 6), range(6, 9), range(9, 12)]


def train_bands():
    import joblib, feats, mkt_near
    from hdata import load_hard
    from folds import split_fold
    from train_forecast import full_hist
    df = load_hard()
    for k in (1, 2, 3):
        t0 = time.time()
        tr = split_fold(df, k)[0]
        feats.set_cs(joblib.load(AH / f"models_f{k}7.pkl")["pv_cs"])
        hist, eh = full_hist(tr)
        out = []
        for hs in BANDS:
            mkt_near.HS = hs
            out.append((list(hs), mkt_near.fit_models(*mkt_near.matrix(hist, eh, "co2"), "co2")))
        joblib.dump(out, AH / f"models_f{k}7bands.pkl", compress=3)
        print(f"plec {k}: {time.time() - t0:.0f} s", flush=True)


class _NoGet(dict):
    """Fa que el calibratge del v12 no trobi mai la previsió antiga (= calibratge desactivat)."""
    def get(self, *a, **kw):
        return None


def _boosters(v):
    import lightgbm as lgb
    return [(lgb.Booster(model_str=ms), list(cols), base) for ms, cols, base in v]


def one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    name, k = job
    import joblib, feats
    from mkt_near import _pred
    from arena import fold_data
    from henv import HardEnv
    val = fold_data(k)
    prm = {"models": f"models_f{k}7.pkl"}
    if name in ("stack", "v19"):
        from boundary_test import agent_cls
        Base = agent_cls(k)
        gid = (val.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
        weeks = [val[gid == g] for g in np.unique(gid)]
        ep = lambda ts: int(ts.to_datetime64().astype("datetime64[h]").astype(np.int64))
        starts = [ep(w.index[0]) for w in weeks]
        prm = {"boundary": True, "_week_ends": [ep(w.index[-1]) for w in weeks], "_next_starts": starts[1:] + [None]}
    else:
        from common import load_base
        Base = load_base().BatteryAgent
    near = _boosters(joblib.load(AH / f"models_f{k}7near.pkl")["co2"])
    bands = [(set(hs), _boosters(v)) for hs, v in joblib.load(AH / f"models_f{k}7bands.pkl")] if name == "bands" else None

    class A(Base):
        def __init__(self, p):
            super().__init__(p)
            if name == "nocal":
                self._co2_raw_forecasts = _NoGet()

        def _pred_mkt2(self, key, t, hs):
            out = super()._pred_mkt2(key, t, hs)
            if key != "co2" or name == "v19":
                return out
            out = np.asarray(out, float).copy()
            idx = [i for i, h in enumerate(hs) if h <= 11]
            if not idx:
                return out
            X = feats.mkt_features2(self.hist, key, t, [hs[i] for i in idx])
            if name == "bands":
                p = np.empty(len(idx))
                for hset, ms in bands:
                    j = [n for n, i in enumerate(idx) if hs[i] in hset]
                    if j:
                        p[j] = _pred(ms, X[j])
            else:
                p = _pred(near, X)
                if name == "avg":
                    p = 0.5 * (p + out[idx])
            out[idx] = p
            return out

    ag = A(prm)
    env = HardEnv(val.reset_index()); o = env.reset(); d = False; t0 = time.time()
    while not d:
        o, d = env.step(ag.act(o))
    r = env.evaluate_results()
    return name, k, r["score"], r["co2_estalviat_pct"], r["cost_estalviat_pct"], 1000 * (time.time() - t0) / len(val)


def score(names):
    from multiprocessing import Pool
    V12 = {1: 20.74, 2: 17.42, 3: 18.90}
    R = {}
    with Pool(4) as p:
        for n, k, s, c, e, ms in p.imap_unordered(one, [(n, k) for n in names for k in (1, 2, 3)]):
            R[(n, k)] = s; print(n, k, s, c, e, f"{ms:.0f} ms/pas", flush=True)
    for n in names:
        v = [R[(n, k)] for k in (1, 2, 3)]
        print(f"{n:<9}", v, f"mitjana {np.mean(v):.3f}  delta v12 {np.mean(v) - np.mean(list(V12.values())):+.3f}",
              "per plec", [round(R[(n, k)] - V12[k], 2) for k in (1, 2, 3)], flush=True)


if __name__ == "__main__":
    for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    if sys.argv[1] == "train_bands":
        train_bands()
    else:
        score(sys.argv[2:] or ["nocal", "avg", "bands", "stack", "v19"])
