"""Escombrat d'un paràmetre cada vegada de la v19 + hora solar als 3 plecs (models `fit` de cada plec).

    python dificil/diag/sweep_params.py '{"ratio": [0.55, 0.7], "net0_mix": [0.3, 0.7]}'
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT = ROOT / "dificil" / "diag" / "extra_feats_agent"   # agent.py de la v19 + hora solar; feats.py ve de dificil/agent_hard
AH = ROOT / "dificil" / "agent_hard"


def one(job):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    name, prm, k = job
    import importlib.util, joblib
    import numpy as np, pandas as pd
    sys.path[:0] = [str(AGENT), str(AH), str(ROOT / "dificil"), str(ROOT / "estandard"), str(ROOT / "scripts")]
    from arena import fold_data
    from henv import HardEnv
    spec = importlib.util.spec_from_file_location("ag", AGENT / "agent.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    val = fold_data(k)
    gid = (val.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
    weeks = [val[gid == g] for g in np.unique(gid)]
    eh = lambda ts: int(ts.to_datetime64().astype("datetime64[h]").astype(np.int64))
    ends = [eh(w.index[-1]) for w in weeks]; starts = [eh(w.index[0]) for w in weeks]
    ag = m.BatteryAgent({**prm, "_models": joblib.load(AH / f"models_f{k}7.pkl"), "_week_ends": ends,
                         "_next_starts": starts[1:] + [None], "end_epoch": ends[-1]})
    env = HardEnv(val.reset_index()); o = env.reset(); d = False
    while not d:
        o, d = env.step(ag.act(o))
    return name, k, env.evaluate_results()["score"]


if __name__ == "__main__":
    from multiprocessing import Pool
    grid = json.loads(sys.argv[1])
    jobs = [("base", {}, k) for k in (1, 2, 3)]
    for key, vals in grid.items():
        jobs += [(f"{key}={v}", {key: v}, k) for v in vals for k in (1, 2, 3)]
    R = {}
    with Pool(2) as p:
        for n, k, s in p.imap_unordered(one, jobs):
            R[(n, k)] = s
    names = list(dict.fromkeys(j[0] for j in jobs))
    b = [R[("base", k)] for k in (1, 2, 3)]
    for n in names:
        v = [R[(n, k)] for k in (1, 2, 3)]
        print(f"{n:<18}", " ".join(f"{x:.3f}" for x in v), f" mitjana {sum(v) / 3:.3f}  Δ {sum(v) / 3 - sum(b) / 3:+.3f}",
              " per plec", " ".join(f"{x - y:+.2f}" for x, y in zip(v, b)), flush=True)
