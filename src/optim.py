"""
Programa lineal de despatx de la bateria.

Variables per hora t:  c_t (càrrega, kW >= 0), d_t (descàrrega, kW >= 0), g_t (importació de xarxa >= 0)
    SOC_k = soc0 + sum_{i<=k} (eta_c * c_i - d_i / eta_d)        0 <= SOC_k <= cap
    g_t  >= net_t + c_t - d_t        (net_t = consum - solar; l'excedent s'aboca a 0 €)
Objectiu: min  sum_t w_t * g_t  -  v_term * SOC_H
    w_t = pes de l'hora (combinació de CO2 i preu)
    v_term = valor de l'energia que queda a la bateria al final de l'horitzó
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

ETA = 0.9 ** 0.5


def solve_dispatch(w, net, soc0, cap, pmax, v_term=0.0, eta=ETA):
    w = np.asarray(w, float)
    net = np.asarray(net, float)
    H = len(w)
    # Escalat numèric: amb pesos ~1e-7 HiGHS s'atura abans de l'òptim (tolerància dual).
    scale = float(np.mean(np.abs(w))) or 1.0
    w = w / scale
    v_term = v_term / scale
    I = sparse.identity(H, format="csr")
    L = sparse.csr_matrix(np.tril(np.ones((H, H))))
    Z = sparse.csr_matrix((H, H))

    # Objectiu sobre [c, d, g]
    cost = np.concatenate([np.zeros(H), np.zeros(H), w])
    if v_term:
        cost[:H] -= v_term * eta
        cost[H:2 * H] += v_term / eta

    # SOC <= cap  ;  -SOC <= 0
    A_soc_hi = sparse.hstack([eta * L, -L / eta, Z])
    A_soc_lo = sparse.hstack([-eta * L, L / eta, Z])
    # net + c - d - g <= 0
    A_imp = sparse.hstack([I, -I, -I])
    A = sparse.vstack([A_soc_hi, A_soc_lo, A_imp]).tocsc()
    b = np.concatenate([np.full(H, cap - soc0), np.full(H, soc0), -net])

    bounds = [(0, pmax)] * (2 * H) + [(0, None)] * H
    r = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    return r.x[:H] - r.x[H:2 * H]


def score_weights(df, alpha=0.7, beta=0.3):
    """Pesos exactes de la fórmula oficial: 0.7·ΔCO2/CO2_base + 0.3·ΔCost/Cost_base."""
    base = np.maximum(df["load_kw"].values - df["pv_kw"].values, 0.0)
    co2 = df["co2_kg_per_kwh"].values
    pr = df["precio_eur_per_kwh"].values
    return alpha * co2 / (base * co2).sum() + beta * pr / (base * pr).sum()


def score_from_actions(df, p, alpha=0.7, beta=0.3):
    net = df["load_kw"].values - df["pv_kw"].values
    g = np.maximum(net + p, 0.0)
    base = np.maximum(net, 0.0)
    co2 = df["co2_kg_per_kwh"].values
    pr = df["precio_eur_per_kwh"].values
    dco2 = 1 - (g * co2).sum() / (base * co2).sum()
    dcost = 1 - (g * pr).sum() / (base * pr).sum()
    return {"score": 100 * (alpha * dco2 + beta * dcost), "co2_pct": 100 * dco2, "cost_pct": 100 * dcost}
