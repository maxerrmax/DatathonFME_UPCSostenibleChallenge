"""Final conegut, mesurat amb dades reals dels plecs (sense pseudo-test).

  v12            referència
  end_real       v12 + final de l'episodi conegut (l'última hora del plec): guany a l'última setmana
  end_cada_setm  v12 + "final" a l'última hora de CADA setmana (l'episodi continua): cost de cobrir-se a cada setmana
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dificil"), str(ROOT / "estandard"), str(ROOT / "dificil" / "arena"), str(ROOT / "scripts")]


def week_ends(val):
    gid = (val.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
    ends = [val.index[gid == g][-1] for g in np.unique(gid)]
    return [int(e.to_datetime64().astype("datetime64[h]").astype(np.int64)) for e in ends], gid


def one(job):
    mode, k = job
    from arena import fold_data
    from common import load_base
    from henv import HardEnv
    val = fold_data(k)
    ends, gid = week_ends(val)
    Base = load_base().BatteryAgent

    class A(Base):
        def _lp(self, w, net, soc0, v_term, dmax0, scen=None):
            t = self.t_prev
            cand = [ends[-1]] if mode == "end_real" else (ends if mode == "end_cada_setm" else [])
            for e in cand:
                R = e - t + 1
                if 0 < R < len(w):
                    w = np.asarray(w)[:R]; net = np.asarray(net)[:R]; v_term = 0.0
                    break
            return super()._lp(w, net, soc0, v_term, dmax0, scen)
    ag = A({"models": f"models_f{k}7.pkl"})
    env = HardEnv(val.reset_index()); o = env.reset(); d = False
    while not d:
        o, d = env.step(ag.act(o))
    wk = [env.evaluate_results(mask=(gid == g))["score"] for g in np.unique(gid)]
    return mode, k, env.evaluate_results()["score"], wk


if __name__ == "__main__":
    jobs = [(m, k) for m in ("v12", "end_real", "end_cada_setm") for k in (1, 2, 3)]
    R = {}
    with Pool(6) as p:
        for m, k, s, wk in p.imap_unordered(one, jobs):
            R[(m, k)] = (s, wk)
    for m in ("end_real", "end_cada_setm"):
        for k in (1, 2, 3):
            d = np.array(R[(m, k)][1]) - np.array(R[("v12", k)][1])
            print(f"{m:<14} plec {k}: total {R[(m, k)][0] - R[('v12', k)][0]:+.3f} | per setmana "
                  + " ".join(f"{x:+.2f}" for x in d))
