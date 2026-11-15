"""
Avalua agents del Mode Difícil sobre setmanes concretes del train (holdout), per provar casos especials que la
validació de l'equip no té (festius, Setmana Santa, vacances...). Els models s'han d'haver entrenat SENSE
aquestes setmanes (p. ex. `train_v7.py hol`, que treu les setmanes HOL de més avall).

Ús:
    python scripts/eval_weeks.py <dir_agent>['|{json params}'] [...] [--weeks 2025-09-08:2025-09-14,...]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dificil")); sys.path.insert(0, str(ROOT / "estandard"))

HOL = [("2025-09-08", "2025-09-14"), ("2025-12-08", "2025-12-14"), ("2026-01-02", "2026-01-08"),
       ("2026-03-30", "2026-04-06"), ("2026-08-03", "2026-08-09")]


def weeks_df(weeks):
    from hdata import load_hard
    df = load_hard()
    parts, names = [], []
    for a, b in weeks:
        w = df[(df.index >= pd.Timestamp(a)) & (df.index < pd.Timestamp(b) + pd.Timedelta("1D"))]
        parts.append(w); names.append((a, len(w)))
    return pd.concat(parts), names


def one(job):
    d, params, weeks = job
    sys.path.insert(0, d)
    df, names = weeks_df(weeks)
    from henv import HardEnv
    sp = importlib.util.spec_from_file_location("agent_eval", os.path.join(d, "agent.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    ag = m.BatteryAgent(params) if params else m.BatteryAgent()
    env = HardEnv(df.reset_index()); o = env.reset(); done = False
    while not done:
        o, done = env.step(ag.act(o))
    tot = env.evaluate_results()
    per, p0 = [], 0
    for k, n in names:
        mask = np.zeros(len(df), bool); mask[p0:p0 + n] = True; p0 += n
        per.append((k, env.evaluate_results(mask=mask)["score"]))
    return os.path.basename(d.rstrip("/")), params, tot, per


if __name__ == "__main__":
    weeks = HOL
    if "--weeks" in sys.argv:
        weeks = [tuple(x.split(":")) for x in sys.argv[sys.argv.index("--weeks") + 1].split(",")]
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--weeks"]
    jobs = []
    for a in args:
        d, _, js = a.partition("|")
        jobs.append((os.path.abspath(d), json.loads(js) if js else {}, weeks))
    with Pool(min(len(jobs), os.cpu_count() or 1)) as p:
        for name, prm, t, per in p.map(one, jobs):
            print(f"{t['score']:6.2f}  CO2={t['co2_estalviat_pct']:6.2f}  cost={t['cost_estalviat_pct']:6.2f}  "
                  f"infr={t['infraccions']}  {name} {json.dumps(prm)}  | " + "  ".join(f"{w[5:]}:{v:5.2f}" for w, v in per))
