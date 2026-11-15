"""
Features de previsió, compartides per l'entrenament i l'agent (només numpy).
Totes les sèries es consulten per marca de temps (hores des d'epoch) en un historial `Hist`.
Disponibilitat a l'instant t (com al simulador):
  - preu i CO2 coneguts fins a t+2 (finestra de 3 h);
  - consum, solar i meteo coneguts fins a t-1.
Per a l'objectiu tau = t+h es generen features amb lags relatius a tau i a t.
"""
import numpy as np

H_MAX = 24
KNOWN_MKT = 2      # última hora de mercat coneguda: t+2
KNOWN_MEAS = -1    # última hora mesurada: t-1


class Hist:
    """Historial per hora (clau = hora entera des d'epoch)."""
    def __init__(self):
        self.d = {k: {} for k in ("price", "co2", "load", "pv", "irr", "temp", "vent")}

    def put(self, key, hour, val):
        if val is not None and np.isfinite(val):
            self.d[key][int(hour)] = float(val)

    def get(self, key, hour):
        return self.d[key].get(int(hour), np.nan)

    def mean_back(self, key, last_hour, n):
        v = [self.d[key].get(int(last_hour) - i, np.nan) for i in range(n)]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else np.nan


def calendar(hour_epoch):
    """hour_epoch: hores des de 1970-01-01 00:00 (hora local del dataset)."""
    days = hour_epoch // 24
    hod = hour_epoch % 24
    dow = (days + 3) % 7                      # 1970-01-01 va ser dijous (dow=3, dilluns=0)
    doy = (days % 365.25)                     # aproximació suficient per a l'estacionalitat
    return hod, dow, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)



# ---------------------------------------------------------------- cel clar (solar màxima per dia de l'any i hora)
_CS = None


def set_cs(table):
    global _CS
    _CS = None if table is None else np.asarray(table, dtype=float)
    _CSC.clear()


_CSC = {}


def cs_value(hour_epoch):
    if _CS is None:
        return np.nan
    v = _CSC.get(hour_epoch)
    if v is None:
        v = _CSC[hour_epoch] = _cs_value(hour_epoch)
    return v


def _cs_value(hour_epoch):
    day = np.datetime64(int(hour_epoch // 24), "D")
    doy = int((day - day.astype("datetime64[Y]")).astype(np.int64))
    return float(_CS[min(doy, 365), int(hour_epoch % 24)])


def cs_table(day_of_year, hour, pv, win=12, q=0.95):
    """Taula 366x24 amb el quantil q de la solar en una finestra de ±win dies (entrenament)."""
    T = np.zeros((366, 24))
    for h in range(24):
        m = hour == h; dd, vv = day_of_year[m], pv[m]
        for k in range(366):
            dist = np.minimum(np.abs(dd - k), 366 - np.abs(dd - k))
            sel = vv[(dist <= win) & np.isfinite(vv)]
            T[k, h] = np.quantile(sel, q) if len(sel) else 0.0
    return T


# ---------------------------------------------------------------- calendari (Barcelona / UPC 2025-26)
def _days(*dates):
    out = set()
    for d in dates:
        out.add(int(np.datetime64(d, "D").astype(np.int64)))
    return out


def _range(a, b):
    a = int(np.datetime64(a, "D").astype(np.int64)); b = int(np.datetime64(b, "D").astype(np.int64))
    return set(range(a, b + 1))


# festius laborals Catalunya + locals Barcelona (2025-2026)
FESTIUS = _days("2025-09-11", "2025-09-24", "2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08",
                "2025-12-25", "2025-12-26", "2026-01-01", "2026-01-06", "2026-04-03", "2026-04-06",
                "2026-05-01", "2026-05-25", "2026-06-24", "2026-08-15", "2026-09-11", "2026-09-24")
# períodes no lectius UPC (Nadal; Setmana Santa; estiu)
VACANCES = _range("2025-12-24", "2026-01-06") | _range("2026-03-30", "2026-04-06") | _range("2026-08-03", "2026-08-21")
# edificis TANCATS i sense climatització (horaris d'edificis UPC: Nadal 25/12-6/1, agost 3-21)
TANCAT = _range("2025-12-25", "2026-01-06") | _range("2026-08-03", "2026-08-21")
# HORARI REDUÏT de matí (8:30-14:00): estiu 22/6-31/7, represa 24/8-4/9, vigílies de Nadal 22-24/12
REDUIT = _range("2025-12-22", "2025-12-24") | _range("2026-06-22", "2026-07-31") | _range("2026-08-24", "2026-09-04")
# Calendari de graus FIB (Campus Nord) 2025-26
EXAMENS = _range("2025-10-30", "2025-11-05") | _range("2026-01-07", "2026-01-19") | \
          _range("2026-04-07", "2026-04-13") | _range("2026-06-04", "2026-06-19")
CLASSES = _range("2025-09-08", "2025-12-19") | _range("2026-02-09", "2026-05-29")


_DFC = {}


def day_flags(hour_epoch):
    d = int(hour_epoch // 24)
    v = _DFC.get(d)
    if v is None:
        v = _DFC[d] = _day_flags(d)
    return v


def _day_flags(d):
    dow = (d + 3) % 7
    fest = d in FESTIUS; vac = d in VACANCES
    off = fest or vac or dow >= 5
    sense_classes = float(d not in CLASSES and d not in EXAMENS)
    return (float(fest), float(vac), float(d in EXAMENS), sense_classes, float(d in TANCAT),
            float(d in REDUIT), float(off))


MKT_NAMES = ["h", "hod", "dow", "doy_s", "doy_c", "lag24", "lag48", "lag168", "last_known",
             "last_known_m24", "mean24", "hod_known", "other_lag24", "other_last", "other_last_m24",
             "irr_last", "vent_last", "temp_last", "irr_lag24_tau", "vent_mean24"]
MEAS_NAMES = ["h", "hod", "dow", "doy_s", "doy_c", "lag24", "lag48", "lag168", "last",
              "last_m24", "mean24", "irr_last", "temp_last", "irr_m24_at_tau", "hod_last"]


def mkt_features(hist, key, t, hs):
    """Features per preu o CO2 a tau=t+h (inclou l'altre senyal de mercat i la meteo)."""
    rows = []
    other = "co2" if key == "price" else "price"
    lk = t + KNOWN_MKT; lm = t + KNOWN_MEAS
    last = hist.get(key, lk); last24 = hist.get(key, lk - 24); m24 = hist.mean_back(key, lk, 24)
    olast = hist.get(other, lk); olast24 = hist.get(other, lk - 24)
    irr, vent, temp = hist.get("irr", lm), hist.get("vent", lm), hist.get("temp", lm)
    vm24 = hist.mean_back("vent", lm, 24)
    for h in hs:
        tau = t + h; hod, dow, s, c = calendar(tau)
        rows.append([h, hod, dow, s, c, hist.get(key, tau - 24), hist.get(key, tau - 48),
                     hist.get(key, tau - 168), last, last - last24, m24, lk % 24,
                     hist.get(other, tau - 24), olast, olast - olast24, irr, vent, temp,
                     hist.get("irr", tau - 24), vm24])
    return np.array(rows, dtype=float)


def meas_features(hist, key, t, hs):
    """Features per consum o solar a tau=t+h."""
    rows = []
    lm = t + KNOWN_MEAS
    last = hist.get(key, lm); last24 = hist.get(key, lm - 24); m24 = hist.mean_back(key, lm, 24)
    irr = hist.get("irr", lm); temp = hist.get("temp", lm)
    pv_last = hist.get("pv", lm); cs_last = cs_value(lm)
    k_last = min(1.3, max(0.0, pv_last / cs_last)) if (np.isfinite(pv_last) and np.isfinite(cs_last) and cs_last > 30) else np.nan
    for h in hs:
        tau = t + h; hod, dow, s, c = calendar(tau)
        rows.append([h, hod, dow, s, c, hist.get(key, tau - 24), hist.get(key, tau - 48),
                     hist.get(key, tau - 168), last, last - last24, m24, irr, temp,
                     hist.get("irr", tau - 24), lm % 24, *day_flags(tau), day_flags(tau - 24)[6],
                     day_flags(tau - 168)[6], cs_value(tau), k_last, k_last * cs_value(tau)])
    return np.array(rows, dtype=float)


class AvgModel:
    """Ensemble: mitjana de les prediccions de diversos models (picklable)."""
    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


# ---------------------------------------------------------------- v7: features de mercat ampliades
MKT2_NAMES = ["h", "hod", "dow", "doy_s", "doy_c", "k0", "k1", "k2", "lag24", "lag48", "lag168", "wk7",
              "shift", "x_shift", "m24", "mn24", "mx24", "sd24", "rel24", "o_k2", "o_lag24", "o_shift", "o_m24",
              "off", "fest", "cs_tau", "kt_d1", "irr_last", "vent_last", "temp_last", "vent_m24", "irr_lag24",
              "hod_t", "d2_24", "wk7_o", "km1", "km2", "km3", "slope", "l23", "l25", "o_k0", "wk3", "o_l23", "o_l25"]


def _nanmean(v):
    v = [x for x in v if x == x and abs(x) != _INF]
    return float(np.mean(v)) if len(v) > 7 else (sum(v) / len(v) if v else np.nan)


_INF = float("inf")


def mkt_features2(hist, key, t, hs):
    other = "co2" if key == "price" else "price"
    lk = t + KNOWN_MKT; lm = t + KNOWN_MEAS
    g = hist.d[key].get; go = hist.d[other].get; nan = np.nan
    k0, k1, k2 = g(t, nan), g(t + 1, nan), g(lk, nan)
    if not np.isfinite(k2):
        k2 = k1 if np.isfinite(k1) else k0
    w = np.array([g(lk - i, nan) for i in range(24)], float); ok = np.isfinite(w)
    m24 = w[ok].mean() if ok.any() else nan; mn = w[ok].min() if ok.any() else nan
    mx = w[ok].max() if ok.any() else nan; sd = w[ok].std() if ok.any() else nan
    shift = k2 - g(lk - 24, nan)
    ok2 = go(lk, nan); oshift = ok2 - go(lk - 24, nan)
    om24 = _nanmean([go(lk - i, nan) for i in range(24)])
    num = den = 0.0
    for i in range(24):
        c = cs_value(lm - i); v = hist.d["pv"].get(lm - i, nan)
        if np.isfinite(c) and c > 30 and np.isfinite(v):
            num += v; den += c
    kt = num / den if den > 0 else nan
    irr, vent, temp = hist.get("irr", lm), hist.get("vent", lm), hist.get("temp", lm)
    vm24 = hist.mean_back("vent", lm, 24)
    rows = []
    for h in hs:
        tau = t + h; hod, dow, s, c = calendar(tau); fl = day_flags(tau)
        l24 = g(tau - 24, nan)
        wk = _nanmean([g(tau - 24 * d, nan) for d in range(1, 8)])
        wko = _nanmean([go(tau - 24 * d, nan) for d in range(1, 8)])
        rows.append([h, hod, dow, s, c, k0, k1, k2, l24, g(tau - 48, nan), g(tau - 168, nan), wk,
                     shift, l24 + shift, m24, mn, mx, sd, l24 - m24, ok2, go(tau - 24, nan), oshift, om24,
                     fl[6], fl[0], cs_value(tau), kt, irr, vent, temp, vm24, hist.get("irr", tau - 24),
                     t % 24, l24 - g(tau - 48, nan), wko, g(t - 1, nan), g(t - 2, nan), g(t - 3, nan), k2 - k0,
                     g(tau - 23, nan), g(tau - 25, nan), go(t, nan),
                     _nanmean([g(tau - 24 * d, nan) for d in range(1, 4)]), go(tau - 23, nan), go(tau - 25, nan)])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------- v7: features de consum ampliades
LOAD2_NAMES = ["h", "hod", "dow", "doy_s", "doy_c", "last", "last2", "last3", "d1", "lag24", "lag48", "lm24",
               "shift", "x_shift", "x_ratio", "m24", "mn24", "mx24", "temp_last", "temp_m24", "irr_last",
               "fest", "vac", "exam", "nocls", "tancat", "reduit", "off", "off_l24", "off_lm", "same", "same_m",
               "same_shift", "hod_lm", "wk7"]


def load_features2(hist, t, hs):
    g = hist.d["load"].get; nan = np.nan; lm = t + KNOWN_MEAS
    last, last2, last3 = g(lm, nan), g(lm - 1, nan), g(lm - 2, nan)
    lm24 = g(lm - 24, nan); shift = last - lm24
    w = np.array([g(lm - i, nan) for i in range(24)], float); ok = np.isfinite(w)
    m24 = w[ok].mean() if ok.any() else nan; mn = w[ok].min() if ok.any() else nan; mx = w[ok].max() if ok.any() else nan
    temp = hist.get("temp", lm); tm24 = hist.mean_back("temp", lm, 24); irr = hist.get("irr", lm)
    off_lm = day_flags(lm)[6]
    # com va l'última hora respecte al mateix tipus de dia (laborable/no) més recent
    same_lm = nan
    for d in range(1, 8):
        if day_flags(lm - 24 * d)[6] == off_lm and np.isfinite(g(lm - 24 * d, nan)):
            same_lm = g(lm - 24 * d, nan); break
    rows = []
    for h in hs:
        tau = t + h; hod, dow, s, c = calendar(tau); fl = day_flags(tau)
        l24 = g(tau - 24, nan)
        vals = [(g(tau - 24 * d, nan), day_flags(tau - 24 * d)[6]) for d in range(1, 8)]
        st = [v for v, o in vals if o == fl[6] and np.isfinite(v)]
        same = st[0] if st else nan; same_m = float(np.mean(st)) if st else nan
        rows.append([h, hod, dow, s, c, last, last2, last3, last - last2, l24, g(tau - 48, nan), lm24,
                     shift, l24 + shift, l24 * last / lm24 if (np.isfinite(lm24) and lm24 > 50) else nan,
                     m24, mn, mx, temp, tm24, irr, *fl, day_flags(tau - 24)[6], off_lm, same, same_m,
                     same + (last - same_lm), lm % 24, _nanmean([v for v, _ in vals])])
    return np.array(rows, dtype=float)
