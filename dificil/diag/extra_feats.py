"""Features opcionals (JAMM_EXTRA de dificil/agent_hard/feats.py) sobre la v19 + hora solar, als 3 plecs.

    python dificil/diag/extra_feats.py dificil/diag/extra_feats_agent base heat thr sun nat pan heat,sun ...

Per a cada variant entrena els models dels plecs 1..3 (seqüencial, tots els nuclis) i simula els 3 plecs
(1 fil per procés). Escriu la puntuació per plec i la diferència respecte a `base`.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AH = ROOT / "dificil" / "agent_hard"


def pkl(var, k):
    return AH / (f"models_f{k}7.pkl" if var == "base" else f"models_f{k}7_x{'-'.join(sorted(var.split(',')))}.pkl")


def one(job):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ.pop("JAMM_EXTRA", None)
    agent_dir, var, k = job
    import importlib.util, joblib
    import numpy as np, pandas as pd
    sys.path[:0] = [agent_dir, str(ROOT / "dificil"), str(ROOT / "estandard"), str(ROOT / "scripts")]
    from arena import fold_data
    from henv import HardEnv
    spec = importlib.util.spec_from_file_location("ag", Path(agent_dir) / "agent.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    val = fold_data(k)
    gid = (val.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
    weeks = [val[gid == g] for g in np.unique(gid)]
    eh = lambda ts: int(ts.to_datetime64().astype("datetime64[h]").astype(np.int64))
    ends = [eh(w.index[-1]) for w in weeks]; starts = [eh(w.index[0]) for w in weeks]
    ag = m.BatteryAgent({"_models": joblib.load(pkl(var, k)), "_week_ends": ends, "_next_starts": starts[1:] + [None],
                         "end_epoch": ends[-1]})
    env = HardEnv(val.reset_index()); o = env.reset(); d = False
    while not d:
        o, d = env.step(ag.act(o))
    r = env.evaluate_results()
    return var, k, r["score"], r["co2_estalviat_pct"], r["cost_estalviat_pct"]


if __name__ == "__main__":
    from multiprocessing import Pool
    agent_dir = sys.argv[1]; names = sys.argv[2:]
    R = {}
    for var in names:
        for k in (1, 2, 3):
            if not pkl(var, k).exists() or var == "base" and "--retrain" in sys.argv:
                env = dict(os.environ); env.pop("JAMM_EXTRA", None)
                if var != "base":
                    env["JAMM_EXTRA"] = var
                subprocess.run([sys.executable, "dificil/train_v7.py", f"f{k}"], cwd=ROOT, env=env, check=True,
                               stdout=subprocess.DEVNULL)
        with Pool(2) as p:
            for v, k, s, c, e in p.imap_unordered(one, [(agent_dir, var, k) for k in (1, 2, 3)]):
                R[(v, k)] = s
        v = [R[(var, k)] for k in (1, 2, 3)]
        line = f"{var:<14} " + " ".join(f"{x:.3f}" for x in v) + f"  mitjana {sum(v) / 3:.3f}"
        if ("base", 1) in R:
            line += "  Δ " + " ".join(f"{R[(var, k)] - R[('base', k)]:+.3f}" for k in (1, 2, 3)) + \
                    f"  ({sum(v) / 3 - sum(R[('base', k)] for k in (1, 2, 3)) / 3:+.3f})"
        print(line, flush=True)
