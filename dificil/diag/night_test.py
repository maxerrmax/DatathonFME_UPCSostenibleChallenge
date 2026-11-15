"""v20 (consum nocturn exacte) als plecs. Basat en regime_test.py.

v18 als plecs (dades reals) sota els dos règims possibles del servidor.

  cont      un sol episodi continu (SOC arrossegat entre setmanes), com l'arena
  reset     cada setmana és un episodi (SOC 3.000 kWh a l'inici), el mateix objecte agent
  reset_new cada setmana és un episodi i l'agent es crea de nou

Variants: ref (week_mode=never = v12 als plecs), auto (v18), always.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, shutil, importlib.util, tempfile
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dificil"), str(ROOT / "estandard"), str(ROOT / "dificil" / "arena"), str(ROOT / "scripts")]
V18 = ROOT / "dificil" / "v20"


def agent_cls(k):
    d = Path(tempfile.mkdtemp(prefix=f"v18f{k}_"))
    for f in ("agent.py", "feats.py", "parametres.json"):
        shutil.copy(V18 / f, d / f)
    shutil.copy(ROOT / "dificil" / "agent_hard" / f"models_f{k}7.pkl", d / "models_all7.pkl")
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(f"v18_{k}", d / "agent.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.BatteryAgent


def one(job):
    regime, variant, k = job
    from arena import fold_data
    from henv import HardEnv
    val = fold_data(k)
    gid = (val.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
    weeks = [val[gid == g] for g in np.unique(gid)]
    ends = [int(w.index[-1].to_datetime64().astype("datetime64[h]").astype(np.int64)) for w in weeks]
    A = agent_cls(k)
    prm = {"night_exact": variant != "ref"}
    tot = np.zeros(4); per = []
    if regime == "cont":
        ag = A(prm)
        env = HardEnv(val.reset_index()); o = env.reset(); d = False
        while not d:
            o, d = env.step(ag.act(o))
        for g in np.unique(gid):
            r = env.evaluate_results(mask=(gid == g)); per.append(r["score"])
        r = env.evaluate_results()
        return regime, variant, k, r["score"], per, getattr(ag, "regime", None)
    ag = A(prm)
    for w in weeks:
        if regime == "reset_new":
            ag = A(prm)
        env = HardEnv(w.reset_index()); o = env.reset(); d = False
        while not d:
            o, d = env.step(ag.act(o))
        r = env.evaluate_results(); per.append(r["score"])
        tot += [r["co2_baseline_kg"], r["co2_participant_kg"], r["cost_baseline_eur"], r["cost_participant_eur"]]
    score = 100 * (0.7 * (1 - tot[1] / tot[0]) + 0.3 * (1 - tot[3] / tot[2]))
    return regime, variant, k, round(score, 3), per, getattr(ag, "regime", None)


if __name__ == "__main__":
    jobs = [(r, v, k) for r in ("cont",) for v in ("ref", "auto") for k in (1, 2, 3)]
    R = {}
    with Pool(6) as p:
        for r, v, k, s, per, reg in p.imap_unordered(one, jobs):
            R[(r, v, k)] = (s, per, reg)
    for r in ("cont",):
        for k in (1, 2, 3):
            a, b = R[(r, "auto", k)], R[(r, "ref", k)]
            d = np.array(a[1]) - np.array(b[1])
            print(f"{r:<9} plec {k}: ref {b[0]:.3f}  v20 {a[0]:.3f}  Δ {a[0] - b[0]:+.3f}  règim detectat: {a[2]}"
                  f" | per setmana " + " ".join(f"{x:+.2f}" for x in d))
