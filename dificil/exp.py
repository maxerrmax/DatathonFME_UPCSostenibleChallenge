"""Banc de proves ràpid: enregistra previsions una vegada i re-juga només la decisió (LP), amb puntuació per blocs."""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "estandard"))
from hdata import load_hard; from data import split_val, blocks
from henv import HardEnv
from heval import load_agent

_df = load_hard(); FIT, VAL = split_val(_df)
_gid = (VAL.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
BLOCKS = [(_gid == g) for g in np.unique(_gid)]


def run_blocks(agent, df=VAL):
    env = HardEnv(df.reset_index()); o = env.reset(); d = False
    while not d:
        o, d = env.step(agent.act(o))
    tot = env.evaluate_results()
    per = [env.evaluate_results(mask=m)["score"] for m in BLOCKS] if df is VAL else []
    return tot["score"], per, tot


def record(params, path):
    rec = {}
    s, per, _ = run_blocks(load_agent({"models": "models_fit.pkl", **params, "_record": rec}))
    pickle.dump(rec, open(path, "wb")); return s, per


def replay(table, params):
    return run_blocks(load_agent({"models": "models_fit.pkl", **params, "_fc_table": table}))


if __name__ == "__main__":
    tblp = HERE / "results" / "fc_v6_fit.pkl"
    if not tblp.exists():
        print("record", record({}, tblp))
    tbl = pickle.load(open(tblp, "rb"))
    grid = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [{}]
    for g in grid:
        s, per, _ = replay(tbl, g)
        print(json.dumps(g), s, " ".join(f"{x:5.1f}" for x in per), flush=True)
