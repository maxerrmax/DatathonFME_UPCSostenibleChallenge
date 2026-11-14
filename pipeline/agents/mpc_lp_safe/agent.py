"""
BatteryAgent — MPC amb programació lineal, versió blindada (Datathon UPC 2026).

Mateixa política que mpc_lp (LP sobre les 24 h de preu i CO2, horitzó lliscant), més:
  - neteja de l'observació: si falta l'hora actual s'usa preu_actual/co2_actual; la resta de forats
    s'interpolen entre hores veïnes (ancorant a l'hora anterior); els valors extrems reals NO es toquen;
  - si el solver falla, reutilitza el pla de l'hora anterior;
  - qualsevol excepció -> 0.0 (repòs), mai peta;
  - la sortida sempre és un float finit dins dels límits físics exactes (0 infraccions).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

CAP = 2000.0
PMAX = 500.0
ETA_C = 0.9 ** 0.5
ETA_D = 0.9 ** 0.5
EPS = 1e-3


def _num(x, default):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _clean_series(values, current, previous):
    """Llista de 24 h sense NaN.
    - posició 0 (hora actual): si falta, s'usa `current` (preu_actual / co2_actual) si és vàlid;
    - la resta de forats s'interpolen linealment entre veïns vàlids, usant l'hora anterior coneguda
      (`previous`) com a punt d'ancoratge a l'esquerra; a la dreta es repeteix l'últim vàlid.
    Els valors extrems reals NO es toquen: són els mateixos que puntuen."""
    raw = [_num(v, float("nan")) for v in list(values or [])]
    if not raw:
        return []
    if not math.isfinite(raw[0]) and current is not None and math.isfinite(current):
        raw[0] = current
    y = np.asarray([previous] + raw, dtype=float)       # índex 0 = hora anterior (ancoratge)
    ok = np.isfinite(y)
    if not ok.all():
        idx = np.arange(len(y))
        y = np.interp(idx, idx[ok], y[ok]) if ok.any() else np.full(len(y), previous)
    return [float(v) for v in y[1:]]


def solve_lp(w, soc0, v_term, dmax):
    H = len(w)
    L = np.tril(np.ones((H, H)))
    A_soc = np.hstack([L * ETA_C, -L / ETA_D])
    A_ub = np.vstack([A_soc, -A_soc])
    b_ub = np.concatenate([np.full(H, CAP - EPS - soc0), np.full(H, soc0 - EPS)])
    cost = np.concatenate([w, -w]) - v_term * A_soc[-1]
    bounds = [(0, PMAX)] * H + [(0, float(u)) for u in dmax]
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if r.status != 0 or r.x is None or not np.all(np.isfinite(r.x)):
        return None
    return r.x[:H] - r.x[H:]


class BatteryAgent:
    def __init__(self, params=None):
        p = {}
        try:
            f = Path(__file__).parent / "parametres.json"
            if f.exists():
                p = json.loads(f.read_text())
        except Exception:
            p = {}
        if params:
            p.update(params)
        self.ratio = _num(p.get("ratio_co2b_cb", 0.587), 0.587)
        self.q_term = _num(p.get("q_term", 0.6), 0.6)
        self.term_scale = _num(p.get("term_scale", 1.0), 1.0)
        self.hist_w = []
        self.last_price = 0.12      # valors per defecte raonables si mai n'arriba cap de vàlid
        self.last_co2 = 0.10
        self.last_soc = CAP / 2
        self.plan = []              # accions planificades per a les hores següents (fallback)

    # ------------------------------------------------------------------ física
    @staticmethod
    def _clamp(soc, a):
        max_c = min(PMAX, (CAP - soc) / ETA_C) - EPS
        max_d = min(PMAX, soc * ETA_D) - EPS
        a = max(-max(0.0, max_d), min(max(0.0, max_c), a))
        return 0.0 if abs(a) < 1e-6 or not math.isfinite(a) else float(a)

    # ------------------------------------------------------------------ política
    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            try:
                return self._clamp(self.last_soc, 0.0)
            except Exception:
                return 0.0

    def _act(self, obs):
        obs = obs or {}
        soc = min(CAP, max(0.0, _num(obs.get("soc_kwh"), self.last_soc)))
        self.last_soc = soc

        p_now = _num(obs.get("preu_actual"), float("nan"))
        c_now = _num(obs.get("co2_actual"), float("nan"))
        price = _clean_series(obs.get("preus_properes_24h"), p_now, self.last_price) \
            or [p_now if math.isfinite(p_now) else self.last_price]
        co2 = _clean_series(obs.get("co2_properes_24h"), c_now, self.last_co2) \
            or [c_now if math.isfinite(c_now) else self.last_co2]
        self.last_price, self.last_co2 = price[0], co2[0]
        H = min(len(price), len(co2))

        w = 0.7 * np.asarray(co2[:H]) + 0.3 * self.ratio * np.asarray(price[:H])
        self.hist_w.append(float(w[0]))
        if H < 24:
            v = 0.0                                 # final de la simulació: buidar la bateria
        else:
            ref = np.concatenate([w, self.hist_w[-72:]])
            v = self.term_scale * ETA_D * float(np.quantile(ref, self.q_term))

        nd = obs.get("noves_dades_5min") or {}
        net_last = _num(nd.get("ultim_consum_kw"), 1e4) - _num(nd.get("ultima_solar_kw"), 0.0)
        dmax = np.full(H, PMAX)
        if net_last < PMAX:
            dmax[0] = max(0.0, net_last)

        P = solve_lp(w, soc, v, dmax)
        if P is None:                               # solver fallit: segueix el pla anterior
            a = self.plan.pop(0) if self.plan else 0.0
        else:
            a = float(P[0])
            self.plan = [float(x) for x in P[1:]]
        return self._clamp(soc, a)
