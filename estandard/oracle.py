"""Cota superior: LP amb previsió perfecta de tota la sèrie (consum inclòs)."""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog

ETA = 0.9 ** 0.5


def weights(df, ratio):
    return (0.7 * df["co2_kg_per_kwh"] + 0.3 * ratio * df["precio_eur_per_kwh"]).values


def oracle_actions(df, ratio=None, soc0=1000.0, cap=2000.0, pmax=500.0):
    if ratio is None:
        net = (df.load_kw - df.pv_kw).clip(lower=0)
        ratio = (net * df.co2_kg_per_kwh).sum() / (net * df.precio_eur_per_kwh).sum()
    w = weights(df, ratio)
    net = (df.load_kw - df.pv_kw).clip(lower=0).values
    H = len(w)
    I = sp.identity(H, format="csr")
    D = sp.diags([np.ones(H), -np.ones(H - 1)], [0, -1], format="csr")
    Aeq = sp.hstack([-ETA * I, I / ETA, D])
    beq = np.zeros(H); beq[0] = soc0
    c = np.concatenate([w, -w, np.zeros(H)])
    b = [(0, pmax)] * H + [(0, min(pmax, n)) for n in net] + [(0, cap)] * H
    r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=b, method="highs")
    x = r.x
    return x[:H] - x[H:2 * H], x[2 * H:]


def score_from_actions(df, P):
    net = (df.load_kw - df.pv_kw).values
    base = np.maximum(net, 0); g = np.maximum(net + P, 0)
    co2 = df.co2_kg_per_kwh.values; pr = df.precio_eur_per_kwh.values
    dco2 = 1 - (g * co2).sum() / (base * co2).sum()
    dc = 1 - (g * pr).sum() / (base * pr).sum()
    return {"score": round(100 * (0.7 * dco2 + 0.3 * dc), 3),
            "co2_estalviat_pct": round(100 * dco2, 3), "cost_estalviat_pct": round(100 * dc, 3)}
