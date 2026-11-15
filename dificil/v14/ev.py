"""python ev.py <agent_dir> [--mA a.pkl] [--mB b.pkl] cfg_json ... -> score a A i B (1 fil per procés, 4 processos)."""
from common import *
import json
from multiprocessing import Pool
from splits import split_B, split_C
W = os.path.dirname(os.path.abspath(__file__))
DATA = {}
def init():
    df = load_hard(); DATA["A"] = split_val(df)[1]; DATA["B"] = split_B(df)[1]; DATA["C"] = split_C(df)[1]
def one(job):
    d, split, pkl, cfg = job
    m = load_agent(d); p = dict(cfg); p["models"] = pkl
    r = run(m.BatteryAgent(p), DATA[split])
    return split, json.dumps(cfg), r["score"], r["co2_estalviat_pct"], r["cost_estalviat_pct"], r["infraccions"], r["ms_per_step"]
if __name__ == "__main__":
    a = sys.argv[1:]; d = os.path.abspath(a.pop(0)); mA, mB = W + "/models_A.pkl", W + "/models_B.pkl"; mC = W + "/models_C.pkl"
    SPL = os.environ.get("SPL", "AB")
    while a and a[0].startswith("--"):
        k = a.pop(0); v = os.path.abspath(a.pop(0))
        if k == "--mA": mA = v
        elif k == "--mC": mC = v
        else: mB = v
    cfgs = [json.loads(x) for x in a] or [{}]
    jobs = [(d, s, {"A": mA, "B": mB, "C": mC}[s], c) for c in cfgs for s in SPL]
    with Pool(4, initializer=init) as p:
        res = p.map(one, jobs)
    tab = {}
    for s, c, sc, co, ct, i, ms in res: tab.setdefault(c, {})[s] = (sc, co, ct, i, ms)
    for c, v in tab.items():
        print("  ".join(f"{s} {v[s][0]:6.2f}" for s in SPL), f" sum {sum(v[s][0] for s in SPL):6.2f}  infr {sum(v[s][3] for s in SPL)}  {c}", flush=True)
