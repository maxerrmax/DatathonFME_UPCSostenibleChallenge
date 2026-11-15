"""Valor de la informació del v12 als 3 plecs (dades reals), per variable i per horitzó.

Cada variant substitueix la previsió per la realitat en un conjunt d'hores de l'horitzó (h=0 és l'hora actual).
Quan l'hora actual és perfecta s'apaguen els escenaris i la prudència de l'hora 0 (no hi ha incertesa).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dificil"), str(ROOT / "estandard"), str(ROOT / "dificil" / "arena"), str(ROOT / "scripts")]

SHRINK = {"net_h0_err50": 0.5, "net_h0_err75": 0.75, "net_h0_perf_hedged": 0.0, "co2_h3_7_err50": 0.5,
          "both_err50": 0.5}
VARIANTS = {
    "net_h0_err50": {"load": [0], "pv": [0]},
    "net_h0_err75": {"load": [0], "pv": [0]},
    "net_h0_perf_hedged": {"load": [0], "pv": [0]},
    "co2_h3_7_err50": {"co2": range(3, 8)},
    "both_err50": {"load": [0], "pv": [0], "co2": range(3, 8)},
    "v12": {},
    "net_h0": {"load": [0], "pv": [0]},
    "pv_h0": {"pv": [0]},
    "net_h0_5": {"load": range(0, 6), "pv": range(0, 6)},
    "net_all": {"load": range(24), "pv": range(24)},
    "co2_h3_7": {"co2": range(3, 8)},
    "co2_all": {"co2": range(3, 24)},
    "price_all": {"price": range(3, 24)},
    "net_h0+co2_all": {"load": [0], "pv": [0], "co2": range(3, 24)},
}


def one(job):
    name, k = job
    from arena import fold_data
    from common import load_base
    from henv import HardEnv
    val = fold_data(k)
    ep = val.index.values.astype("datetime64[h]").astype(np.int64)
    true = {"price": dict(zip(ep, val.precio_eur_per_kwh)), "co2": dict(zip(ep, val.co2_kg_per_kwh)),
            "load": dict(zip(ep, val.load_kw)), "pv": dict(zip(ep, val.pv_kw))}
    spec = VARIANTS[name]
    Base = load_base().BatteryAgent

    class A(Base):
        def _forecast(self, t, price3, co2_3):
            out = list(super()._forecast(t, price3, co2_3))
            for key, hs in spec.items():
                i = {"price": 0, "co2": 1, "load": 2, "pv": 3}[key]
                for h in hs:
                    v = true[key].get(t + h)
                    if v is not None and h < len(out[i]):
                        a = SHRINK.get(name, 0.0)          # 0 = perfecte; 0,5 = la meitat de l'error
                        out[i][h] = v + a * (out[i][h] - v)
            return tuple(out)
    prm = {"models": f"models_f{k}7.pkl"}
    if 0 in spec.get("load", []) and name not in SHRINK:
        prm.update({"scen": 0, "k_dis": 0.0})
    ag = A(prm)
    env = HardEnv(val.reset_index()); o = env.reset(); d = False
    while not d:
        o, d = env.step(ag.act(o))
    return name, k, env.evaluate_results()["score"]


if __name__ == "__main__":
    from horacle import oracle
    from arena import fold_data
    names = sys.argv[1:] or list(VARIANTS)
    jobs = [(n, k) for n in names for k in (1, 2, 3)]
    R = {}
    with Pool(4) as p:
        for n, k, s in p.imap_unordered(one, jobs):
            R[(n, k)] = s
    orc = {k: oracle(fold_data(k))[0]["score"] for k in (1, 2, 3)}
    print(f"{'variant':<16}" + "".join(f"{'plec'+str(k):>9}" for k in (1, 2, 3)) + f"{'mitjana':>9}{'guany':>8}")
    base = np.mean([R[("v12", k)] for k in (1, 2, 3)])
    for n in names:
        m = np.mean([R[(n, k)] for k in (1, 2, 3)])
        print(f"{n:<16}" + "".join(f"{R[(n, k)]:9.2f}" for k in (1, 2, 3)) + f"{m:9.2f}{m - base:+8.2f}")
    print(f"{'òptim':<16}" + "".join(f"{orc[k]:9.2f}" for k in (1, 2, 3)) + f"{np.mean(list(orc.values())):9.2f}"
          f"{np.mean(list(orc.values())) - base:+8.2f}")
    out = ROOT / "dashboard" / "model_data" / "voi.json"           # llegit pel dashboard (pàgina «On són els punts»)
    import json
    out.write_text(json.dumps({
        "base": float(base),
        "oracle": {"mean": float(np.mean(list(orc.values()))), "gain": float(np.mean(list(orc.values())) - base)},
        "variants": {n: {"folds": [R[(n, k)] for k in (1, 2, 3)], "mean": float(np.mean([R[(n, k)] for k in (1, 2, 3)])),
                         "gain": float(np.mean([R[(n, k)] for k in (1, 2, 3)]) - base)} for n in names},
    }, indent=1), encoding="utf-8")
