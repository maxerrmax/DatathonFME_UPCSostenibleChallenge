"""Grava les previsions del v12 (models del plec k) i les accions/SOC, per poder reproduir decisions ràpidament.

    python dificil/diag/record.py 1 2 3      -> dificil/diag/rec_f<k>.pkl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, pickle, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dificil"), str(ROOT / "estandard"), str(ROOT / "dificil" / "arena"), str(ROOT / "scripts")]


def one(k):
    from arena import fold_data
    from common import load_base
    from henv import HardEnv
    val = fold_data(k)
    rec, scen = {}, {}
    A = load_base().BatteryAgent
    ag = A({"models": f"models_f{k}7.pkl", "_record": rec})
    # captura també els escenaris de l'hora actual
    orig = ag._lp
    def lp(w, net, soc0, v, dmax0, sc=None):
        scen[ag.t_prev] = (None if sc is None else np.asarray(sc) + net[0], v)
        return orig(w, net, soc0, v, dmax0, sc)
    ag._lp = lp
    env = HardEnv(val.reset_index()); o = env.reset(); d = False; t0 = time.time()
    ts = []
    while not d:
        ts.append(ag.t_prev if ag.t_prev is not None else None)
        o, d = env.step(ag.act(o))
    r = env.evaluate_results()
    out = dict(rec=rec, scen=scen, actions=np.array(env.actions), soc=np.array(env.soc_series), res=r)
    pickle.dump(out, open(ROOT / "dificil" / "diag" / f"rec_f{k}.pkl", "wb"))
    return k, r["score"], time.time() - t0


if __name__ == "__main__":
    ks = [int(x) for x in sys.argv[1:]] or [1]
    with Pool(len(ks)) as p:
        for k, s, dt in p.imap_unordered(one, ks):
            print(f"plec {k}: {s:.3f}  ({dt:.0f} s)", flush=True)
