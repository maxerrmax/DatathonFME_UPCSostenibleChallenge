"""Òptim teòric Mode Difícil: LP amb previsió perfecta de tot (consum, solar, preu, CO2).
imp_t >= net_t + c_t - d_t, imp_t >= 0  (l'excedent es perd a 0 €/0 kg)."""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog

ETA = 0.9 ** 0.5


def oracle(df, cap=6000.0, pmax=1500.0, soc0=3000.0):
    net = (df.load_kw - df.pv_kw).values; co2 = df.co2_kg_per_kwh.values; pr = df.precio_eur_per_kwh.values
    base = np.maximum(net, 0); B = (base * co2).sum(); C = (base * pr).sum()
    w = 100 * (0.7 * co2 / B + 0.3 * pr / C)
    H = len(net); I = sp.identity(H, format="csr"); Z = sp.csr_matrix((H, H))
    D = sp.diags([np.ones(H), -np.ones(H - 1)], [0, -1], format="csr")
    Aeq = sp.hstack([-ETA * I, I / ETA, D, Z]); beq = np.zeros(H); beq[0] = soc0
    Aub = sp.hstack([I, -I, Z, -I]); bub = -net                    # net + c - d - imp <= 0
    cost = np.concatenate([np.zeros(3 * H), w])
    bnd = [(0, pmax)] * (2 * H) + [(0, cap)] * H + [(0, None)] * H
    r = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bnd, method="highs")
    x = r.x; P = x[:H] - x[H:2 * H]
    imp = np.maximum(net + P, 0)
    dco2 = 1 - (imp * co2).sum() / B; dc = 1 - (imp * pr).sum() / C
    return {"score": round(100 * (0.7 * dco2 + 0.3 * dc), 3), "co2": round(100 * dco2, 2), "cost": round(100 * dc, 2)}, P
