"""
agent.py — Mode Difícil (Repte UPC Sostenible, Datathon 2026)
MPC de 24 h amb previsions LightGBM de preu, CO2, consum i solar + blindatge de dades.
Fitxer generat per scripts/build_agent.py: no editar a mà.
"""
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

try:
    import lightgbm as lgb
except Exception:
    lgb = None


NAN = float("nan")
EPOCH = datetime(1970, 1, 1)

# Festius (Espanya, Catalunya, Barcelona) i tancaments del campus
FESTIUS = {
    "2025-09-11", "2025-09-24", "2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08",
    "2025-12-25", "2025-12-26", "2026-01-01", "2026-01-06", "2026-04-03", "2026-04-06",
    "2026-05-01", "2026-05-25", "2026-06-24", "2026-08-15", "2026-09-11", "2026-09-24",
    "2026-10-12", "2026-11-01", "2026-12-08", "2026-12-25", "2026-12-26",
}


def hour_to_dt(h):
    return EPOCH + timedelta(hours=int(h))


def ts_to_hour(ts):
    """Hores des de 1970 a partir de pd.Timestamp, datetime, np.datetime64 o text ISO. None si no es pot."""
    if ts is None:
        return None
    if hasattr(ts, "value") and not isinstance(ts, (str, bytes)):          # pd.Timestamp
        try:
            return int(ts.value // 3_600_000_000_000)
        except Exception:
            pass
    if isinstance(ts, datetime):
        return int((ts.replace(tzinfo=None) - EPOCH).total_seconds() // 3600)
    try:
        s = str(ts).strip().replace("T", " ")[:19]
        return int((datetime.fromisoformat(s) - EPOCH).total_seconds() // 3600)
    except Exception:
        return None


def calendar(h):
    d = hour_to_dt(h)
    iso = d.strftime("%Y-%m-%d")
    md = (d.month, d.day)
    nadal = md >= (12, 22) or md <= (1, 7)
    pasqua = date(2026, 3, 28) <= d.date() <= date(2026, 4, 6)
    agost = d.month == 8
    festiu = iso in FESTIUS
    off = festiu or d.weekday() >= 5
    doy = d.timetuple().tm_yday
    return {
        "hour": d.hour, "dow": d.weekday(), "month": d.month, "doy": doy,
        "festiu": int(festiu), "off": int(off), "nadal": int(nadal), "pasqua": int(pasqua), "agost": int(agost),
    }


def solar_elev(h):
    """Elevació solar aproximada (Barcelona) a la meitat de l'hora, en graus."""
    d = hour_to_dt(h)
    doy = d.timetuple().tm_yday
    lat = math.radians(41.39)
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    hour_utc = d.hour + 0.5  # les marques temporals són hora local; aproximem
    # hora solar ~ hora local - 1 (hivern) / -2 (estiu); fem servir -1.5 de mitjana
    ha = math.radians(15 * (hour_utc - 1.5 - 12 + 0.15))
    s = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


PV_CLEAR_KW = 150.0  # potència aprox. de la planta real (sense escalar) amb sol al zenit


def _mean(vals):
    v = [x for x in vals if x == x]
    return sum(v) / len(v) if v else NAN


PRICE_FEATS = ["k", "hour", "dow", "month", "off", "festiu", "nadal", "pasqua", "agost",
               "p0", "p1", "p2", "c0", "c1", "c2", "x_l24", "x_l48", "x_m24", "y_l24", "x_l24_0",
               "x_shift", "x_cur", "elev", "irr_l1", "irr_l24", "irr_m24", "temp_l1", "y_cur"]
LOAD_FEATS = ["k", "hour", "dow", "month", "off", "festiu", "nadal", "pasqua", "agost",
              "load_l1", "load_l24", "load_l48", "load_m24", "load_l24_0", "temp_l1", "temp_m24", "irr_l1"]
PV_FEATS = ["k", "hour", "month", "doy", "elev", "pv_l1", "irr_l1", "pv_l24", "irr_l24", "pv_m24max",
            "temp_l1", "elev_l1", "kt_l1", "kt_l24", "kt_d1", "pv_l48", "pv_clear"]


def price_features(get, t0, k, target):
    """target = 'p' (preu) o 'c' (CO2)."""
    other = "c" if target == "p" else "p"
    T = t0 + k
    f = calendar(T)
    f["k"] = k
    for i in range(3):
        f[f"p{i}"] = get("p", t0 + i)
        f[f"c{i}"] = get("c", t0 + i)
    f["x_l24"] = get(target, T - 24)
    f["x_l48"] = get(target, T - 48)
    f["y_l24"] = get(other, T - 24)
    f["x_m24"] = _mean([get(target, t0 + 2 - i) for i in range(24)])
    f["x_l24_0"] = get(target, t0 + 2 - 24)  # mateix origen ahir (per mesurar el canvi de nivell)
    f["x_cur"] = get(target, t0 + 2)
    f["y_cur"] = get(other, t0 + 2)
    f["x_shift"] = f["x_l24"] + (f["x_cur"] - f["x_l24_0"])
    f["elev"] = solar_elev(T)
    f["irr_l1"] = get("irr", t0 - 1)
    f["irr_l24"] = get("irr", T - 24)
    f["irr_m24"] = _mean([get("irr", t0 - 1 - i) for i in range(24)])
    f["temp_l1"] = get("temp", t0 - 1)
    return [f[c] for c in PRICE_FEATS]


def load_features(get, t0, k):
    T = t0 + k
    f = calendar(T)
    f["k"] = k
    f["load_l1"] = get("load", t0 - 1)
    f["load_l24"] = get("load", T - 24)
    f["load_l48"] = get("load", T - 48)
    f["load_m24"] = _mean([get("load", t0 - 1 - i) for i in range(24)])
    f["load_l24_0"] = get("load", t0 - 1 - 24)
    f["temp_l1"] = get("temp", t0 - 1)
    f["temp_m24"] = _mean([get("temp", t0 - 1 - i) for i in range(24)])
    f["irr_l1"] = get("irr", t0 - 1)
    return [f[c] for c in LOAD_FEATS]


def pv_features(get, t0, k):
    T = t0 + k
    f = calendar(T)
    f["k"] = k
    f["elev"] = solar_elev(T)
    f["elev_l1"] = solar_elev(t0 - 1)
    f["pv_l1"] = get("pv", t0 - 1)
    f["irr_l1"] = get("irr", t0 - 1)
    f["pv_l24"] = get("pv", T - 24)
    f["irr_l24"] = get("irr", T - 24)
    vals = [get("pv", t0 - 1 - i) for i in range(24)]
    vals = [v for v in vals if v == v]
    f["pv_m24max"] = max(vals) if vals else NAN
    f["temp_l1"] = get("temp", t0 - 1)
    s = math.sin(math.radians(max(f["elev_l1"], 0.0)))
    f["kt_l1"] = f["irr_l1"] / (1000 * s) if s > 0.1 and f["irr_l1"] == f["irr_l1"] else NAN
    s24 = math.sin(math.radians(max(solar_elev(T - 24), 0.0)))
    f["kt_l24"] = f["irr_l24"] / (1000 * s24) if s24 > 0.1 and f["irr_l24"] == f["irr_l24"] else NAN
    # claredat mitjana de les darreres 24 h (energia irradiada / energia de cel clar)
    num = den = 0.0
    for i in range(24):
        hh = t0 - 1 - i
        ir = get("irr", hh)
        se = math.sin(math.radians(max(solar_elev(hh), 0.0)))
        if ir == ir and se > 0.1:
            num += ir; den += 1000 * se
    f["kt_d1"] = num / den if den > 0 else NAN
    f["pv_l48"] = get("pv", T - 48)
    f["pv_clear"] = PV_CLEAR_KW * math.sin(math.radians(max(f["elev"], 0.0)))
    return [f[c] for c in PV_FEATS]


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




# ---------------------------------------------------------------------------
# BatteryAgent — Mode Difícil (6.000 kWh, ±1.500 kW, solar x12, mercat a 3 h)
#
# Cada hora:
#   1. Guarda a la memòria el que sap (preu/CO2 de 3 h, consum/solar/meteo de l'hora passada).
#   2. Preveu 24 h de preu, CO2, consum i solar amb models LightGBM (fitxers model_*.txt).
#   3. Resol un programa lineal de 24 h (minimitza 0.7·CO2 + 0.3·cost importats de la xarxa,
#      aprofitant l'excedent solar gratuït) i aplica només la primera hora (MPC).
#   4. Blindatge: dades absents/NaN, descàrrega limitada al consum previst, pla B si falla,
#      i clamp físic per tenir 0 infraccions.
# ---------------------------------------------------------------------------


try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

HERE = Path(__file__).resolve().parent
ETA = 0.9 ** 0.5

DEFAULTS = {
    "KC": 78.09690663686939,     # mitjana(consum_net_base · CO2) al train (solar x12)
    "KP": 127.89844467070672,    # mitjana(consum_net_base · preu)
    "alpha": 0.7, "beta": 0.3,
    "horizon": 24,
    "pv_scale_hard": 12.0,
    "v_term_q": 0.0,             # valor terminal de l'energia (quantil dels pesos de l'horitzó)
    "v_term_f": 0.0,
    "dis_margin_kw": 0.0,        # marge de seguretat en descarregar respecte al consum net previst
    "load_bias": 0.0,
    "pv_bias": 1.0,
    "outage_kw": 100.0,
    "pc_contrast": 1.0,          # amplifica el contrast entre hores previstes de preu/CO2
    "res_w": 0.0,                # pes del model residual de preu/CO2 (0 = només directe)          # consum mesurat per sota d'això = avaria / campus aturat
}


def _f(x, default=float("nan")):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else default
    except Exception:
        return default


class BatteryAgent:
    def __init__(self, model_dir=None, params=None):
        d = Path(model_dir) if model_dir else HERE
        self.p = dict(DEFAULTS)
        try:
            pf = d / "params.json"
            if pf.exists():
                self.p.update(json.loads(pf.read_text()))
        except Exception:
            pass
        if params:
            self.p.update(params)
        self.models = {}
        if lgb is not None:
            for tg in ("p", "c", "load", "pv", "p_res", "c_res"):
                f = d / f"model_{tg}.txt"
                if f.exists():
                    try:
                        self.models[tg] = lgb.Booster(model_file=str(f))
                    except Exception:
                        pass
        self.store = {k: {} for k in ("p", "c", "load", "pv", "irr", "temp")}
        self.last_load = 900.0
        self.last_price, self.last_co2 = 0.13, 0.08
        self.last_t0 = None
        self.repeat_guard = True   # si el timestamp no avança, avancem el rellotge intern

    # ------------------------------------------------------------------ memòria
    def _get(self, col, h):
        return self.store[col].get(h, float("nan"))

    def _remember(self, obs, t0, pv_scale):
        prices = list(obs.get("preus_properes_24h") or [])
        co2s = list(obs.get("co2_properes_24h") or [])
        p_now = _f(obs.get("preu_actual"))
        c_now = _f(obs.get("co2_actual"))
        if not prices:
            prices = [p_now]
        if not co2s:
            co2s = [c_now]
        for i, v in enumerate(prices):
            v = _f(v)
            if v == v:
                self.store["p"][t0 + i] = v
        for i, v in enumerate(co2s):
            v = _f(v)
            if v == v:
                self.store["c"][t0 + i] = v
        n5 = obs.get("noves_dades_5min") or {}
        load = _f(n5.get("ultim_consum_kw"))
        pv = _f(n5.get("ultima_solar_kw"))
        if load == load and t0 > 0:
            self.store["load"][t0 - 1] = load
        if pv == pv:
            self.store["pv"][t0 - 1] = pv / pv_scale
        irr = _f(n5.get("irradiancia_wm2"))
        tmp = _f(n5.get("temp_ambient_c"))
        if irr == irr:
            self.store["irr"][t0 - 1] = irr
        if tmp == tmp:
            self.store["temp"][t0 - 1] = tmp
        # neteja de memòria antiga (> 3 dies)
        if t0 % 168 == 0:
            for col in self.store:
                self.store[col] = {h: v for h, v in self.store[col].items() if h >= t0 - 72}

    # ------------------------------------------------------------------ previsions
    def _forecast(self, t0, H, pv_scale):
        get = self._get
        price = np.full(H, np.nan)
        co2 = np.full(H, np.nan)
        for k in range(H):
            price[k] = get("p", t0 + k)
            co2[k] = get("c", t0 + k)
        unk = [k for k in range(H) if price[k] != price[k] or co2[k] != co2[k]]
        if unk:
            for tg, arr in (("p", price), ("c", co2)):
                m = self.models.get(tg)
                ks = [k for k in unk if arr[k] != arr[k]]
                if m is not None and ks:
                    X = np.array([price_features(get, t0, k, tg) for k in ks], float)
                    pred = m.predict(X)
                    mr = self.models.get(tg + "_res")
                    rw = float(self.p["res_w"])
                    if mr is not None and rw > 0:
                        # model residual: diferència respecte a l'últim valor conegut (t0+2)
                        base = X[:, PRICE_FEATS.index("x_cur")]
                        known = next((v for v in (get(tg, t0 + i) for i in (2, 1, 0)) if v == v), float("nan"))
                        pr = np.where(np.isnan(base), known, base) + mr.predict(X)
                        pred = np.where(np.isnan(pr), pred, (1 - rw) * pred + rw * pr)
                    g = float(self.p["pc_contrast"])
                    if g != 1.0 and len(ks) > 1:
                        mu = float(np.mean(pred))
                        pred = mu + g * (pred - mu)
                    arr[ks] = pred
        # forats que quedin: últim conegut
        lp = next((v for v in (get("p", t0 + i) for i in (2, 1, 0)) if v == v), self.last_price)
        lc = next((v for v in (get("c", t0 + i) for i in (2, 1, 0)) if v == v), self.last_co2)
        price = np.where(np.isnan(price), lp, price)
        co2 = np.where(np.isnan(co2), lc, co2)
        self.last_price, self.last_co2 = lp, lc

        ks = list(range(H))
        m = self.models.get("load")
        if m is not None:
            load = m.predict(np.array([load_features(get, t0, k) for k in ks], float))
        else:
            ll = get("load", t0 - 1)
            load = np.full(H, ll if ll == ll else self.last_load)
        m = self.models.get("pv")
        if m is not None:
            pv = m.predict(np.array([pv_features(get, t0, k) for k in ks], float))
        else:
            pv = np.zeros(H)
        pv = np.clip(pv, 0, None) * pv_scale * self.p["pv_bias"]
        load = np.clip(load, 0, None) + self.p["load_bias"]
        # Blindatge: si el comptador marca ~0 (avaria del sensor o campus aturat), suposem que continua
        ll = get("load", t0 - 1)
        if ll == ll and ll < self.p["outage_kw"]:
            load = np.full(H, max(ll, 0.0))
        return price, co2, np.clip(load, 0, None), pv
    # ------------------------------------------------------------------ decisió
    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            try:
                return self._fallback(obs)
            except Exception:
                return 0.0

    def _limits(self, obs):
        cap = _f(obs.get("cap_max_kwh"), 6000.0)
        pmax = _f(obs.get("power_max_kw"), 1500.0)
        soc = min(max(_f(obs.get("soc_kwh"), cap / 2), 0.0), cap)
        return cap, pmax, soc

    def _clamp(self, p, obs):
        cap, pmax, soc = self._limits(obs)
        hi = max(0.0, min(pmax, (cap - soc) / ETA - 1e-4))
        lo = max(0.0, min(pmax, soc * ETA - 1e-4))
        p = _f(p, 0.0)
        return float(max(-lo, min(hi, p)))

    def _act(self, obs):
        cap, pmax, soc = self._limits(obs)
        hard = cap > 2500
        pv_scale = self.p["pv_scale_hard"] if hard else 1.0
        t0 = ts_to_hour(obs.get("timestamp"))
        if t0 is None or (self.last_t0 is not None and t0 <= self.last_t0 and self.repeat_guard):
            # sense data fiable: rellotge intern (1 h per crida)
            t0 = (self.last_t0 + 1) if self.last_t0 is not None else 0
        self.last_t0 = t0
        self._remember(obs, t0, pv_scale)
        H = int(self.p["horizon"])
        price, co2, load, pv = self._forecast(t0, H, pv_scale)
        net = load - pv
        w = self.p["alpha"] * co2 / self.p["KC"] + self.p["beta"] * price / self.p["KP"]
        v_term = 0.0
        if self.p["v_term_f"]:
            v_term = self.p["v_term_f"] * float(np.quantile(w, self.p["v_term_q"]))
        plan = solve_dispatch(w, net, soc, cap, pmax, v_term=v_term)
        p0 = float(plan[0])
        if p0 < 0:
            # blindatge: no descarregar per sobre del consum net previst (l'excedent no compta)
            p0 = -min(-p0, max(0.0, net[0] - self.p["dis_margin_kw"]))
        return self._clamp(p0, obs)

    def _fallback(self, obs):
        """Pla B: carregar en hores barates/netes, descarregar en hores cares/brutes."""
        cap, pmax, soc = self._limits(obs)
        pr = _f(obs.get("preu_actual"), 0.13)
        c = _f(obs.get("co2_actual"), 0.08)
        s = 0.7 * c / 0.085 + 0.3 * pr / 0.138
        n5 = obs.get("noves_dades_5min") or {}
        net = _f(n5.get("ultim_consum_kw"), 0.0) - _f(n5.get("ultima_solar_kw"), 0.0)
        if net < 0:
            p = min(pmax, -net)
        elif s < 0.85:
            p = pmax * 0.6
        elif s > 1.15:
            p = -min(pmax, max(0.0, net))
        else:
            p = 0.0
        return self._clamp(p, obs)
