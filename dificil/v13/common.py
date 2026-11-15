import os, sys, importlib.util, pickle
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
ROOT = "/home/user/DatathonFME_UPCSostenibleChallenge"
sys.path.insert(0, ROOT + "/dificil"); sys.path.insert(0, ROOT + "/estandard")
from hdata import load_hard
from data import split_val
from henv import run
S = ROOT + "/dificil/agent_hard"
def val():
    return split_val(load_hard())[1]
def load_agent(d):
    if d not in sys.path: sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("agent_" + str(abs(hash(d))), os.path.join(d, "agent.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def truth(df):
    eh = (df.index.values.astype("datetime64[h]").astype(np.int64))
    tb = {}
    for k, c in (("price", "precio_eur_per_kwh"), ("co2", "co2_kg_per_kwh"), ("load", "load_kw"), ("pv", "pv_kw")):
        tb[k] = dict(zip(eh.tolist(), df[c].values.tolist()))
    return tb
