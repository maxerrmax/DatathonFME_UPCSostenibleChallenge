"""python ev.py <agent_dir> [--mA a.pkl] [--mB b.pkl] cfg_json ... -> score a A i B (1 fil per procés, 4 processos)."""
from common import *
import json
from multiprocessing import Pool
from splits import split_B
W = os.path.dirname(os.path.abspath(__file__))
DATA = {}
def init():
    df = load_hard(); DATA["A"] = split_val(df)[1]; DATA["B"] = split_B(df)[1]
def one(job):
    d, split, pkl, cfg = job
    m = load_agent(d); p = dict(cfg); p["models"] = pkl
    r = run(m.BatteryAgent(p), DATA[split])
    return split, json.dumps(cfg), r["score"], r["co2_estalviat_pct"], r["cost_estalviat_pct"], r["infraccions"], r["ms_per_step"]
if __name__ == "__main__":
    a = sys.argv[1:]; d = os.path.abspath(a.pop(0)); mA, mB = W + "/models_A.pkl", W + "/models_B.pkl"
    while a and a[0].startswith("--"):
        k = a.pop(0); v = os.path.abspath(a.pop(0))
        if k == "--mA": mA = v
        else: mB = v
    cfgs = [json.loads(x) for x in a] or [{}]
    jobs = [(d, s, mA if s == "A" else mB, c) for c in cfgs for s in ("A", "B")]
    with Pool(4, initializer=init) as p:
        res = p.map(one, jobs)
    tab = {}
    for s, c, sc, co, ct, i, ms in res: tab.setdefault(c, {})[s] = (sc, co, ct, i, ms)
    for c, v in tab.items():
        A, B = v["A"], v["B"]
        print(f"A {A[0]:6.2f}  B {B[0]:6.2f}  sum {A[0]+B[0]:6.2f}  infr {A[3]+B[3]}  {A[4]}ms  {c}", flush=True)
