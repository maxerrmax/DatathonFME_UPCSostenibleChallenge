"""Avalua un agent a validació passant-li el calendari de setmanes de validació (_weeks)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import json
from multiprocessing import Pool
D = None; V = None; WK = None
def weeks_of(df):
    eh = df.index.values.astype("datetime64[h]").astype(np.int64)
    cuts = np.where(np.diff(eh) != 1)[0]
    st = np.r_[0, cuts + 1]; en = np.r_[cuts, len(eh) - 1]
    return [(int(eh[a]), int(eh[b])) for a, b in zip(st, en)]
def init(d):
    global D, V, WK
    D = d; V = val(); WK = weeks_of(V)
def one(cfg):
    m = load_agent(D); p = dict(cfg)
    if p.pop("_valweeks", True): p.setdefault("_weeks", WK)
    r = run(m.BatteryAgent(p), V)
    return cfg, r["score"], r["co2_estalviat_pct"], r["cost_estalviat_pct"], r["infraccions"], r["ms_per_step"]
if __name__ == "__main__":
    d = os.path.abspath(sys.argv[1]); grid = [json.loads(a) for a in sys.argv[2:]] or [{}]
    with Pool(min(4, len(grid)), initializer=init, initargs=(d,)) as p:
        for cfg, s, c, k, i, ms in p.map(one, grid):
            print(f"{s:6.2f} CO2={c:6.2f} cost={k:6.2f} infr={i} {ms}ms {json.dumps(cfg)}", flush=True)
