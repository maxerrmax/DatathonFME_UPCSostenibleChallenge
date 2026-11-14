"""
Variables explicatives per a les previsions (compartides entre entrenament i agent).

Convenció d'hores: enter = hores des de 1970-01-01 (h = int(ts.value // 3_600_000_000_000)).
A l'origen t0 l'agent coneix:
    preu i CO2  fins a t0+2   (obs, horitzó de 3 h al Mode Difícil)
    consum, solar i meteo fins a t0-1   (obs["noves_dades_5min"])
`get(col, h)` ha de retornar el valor o NaN si no és conegut.
"""
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np

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
