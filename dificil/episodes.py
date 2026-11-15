"""Avaluació amb un episodi per setmana (bateria al 50 % a l'inici de cada setmana, agent nou), com sembla fer el servidor."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exp import VAL, BLOCKS
from henv import HardEnv
from heval import load_agent


def run_episodes(params, df=VAL, rem=False):
    gc = gb = cc = cb = 0.0; infr = 0; soc_end = []
    for m in BLOCKS:
        d = df[m]
        prm = dict(params)
        if rem:                                      # final de l'episodi conegut (hores que queden)
            ts = d.index.values.astype("datetime64[h]").astype(np.int64)
            prm["_rem"] = {int(t): int(len(ts) - i) for i, t in enumerate(ts)}
        ag = load_agent(prm)
        env = HardEnv(d.reset_index()); o = env.reset(); done = False
        while not done:
            o, done = env.step(ag.act(o))
        g = np.asarray(env.grid); ex = np.asarray(env.exports)
        base = np.maximum(d.load_kw.values - d.pv_kw.values, 0)
        cc += float(np.sum(g * d.co2_kg_per_kwh.values)); cb += float(np.sum(base * d.co2_kg_per_kwh.values))
        gc += float(np.sum(g * d.precio_eur_per_kwh.values)); gb += float(np.sum(base * d.precio_eur_per_kwh.values))
        infr += int(env.clipped + env.violations); soc_end.append(float(env.soc))
    dco2 = (cb - cc) / cb; dcost = (gb - gc) / gb
    return round(100 * (0.7 * dco2 + 0.3 * dcost) - 0.2 * infr, 3), round(100 * dco2, 2), round(100 * dcost, 2), infr, soc_end
