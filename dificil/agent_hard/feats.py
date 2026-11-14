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
# períodes no lectius UPC (Nadal ampliat al 24/12 i fins al 6/1; Setmana Santa; estiu)
VACANCES = _range("2025-12-24", "2026-01-06") | _range("2026-03-30", "2026-04-06") | _range("2026-08-03", "2026-08-31")
# Calendari de graus FIB (Campus Nord) 2025-26
EXAMENS = _range("2025-10-30", "2025-11-05") | _range("2026-01-07", "2026-01-19") | \
          _range("2026-04-07", "2026-04-13") | _range("2026-06-04", "2026-06-19")
CLASSES = _range("2025-09-08", "2025-12-19") | _range("2026-02-09", "2026-05-29")


def day_flags(hour_epoch):
    d = int(hour_epoch // 24)
    dow = (d + 3) % 7
    fest = d in FESTIUS; vac = d in VACANCES
    off = fest or vac or dow >= 5
    sense_classes = float(d not in CLASSES and d not in EXAMENS)
    return float(fest), float(vac), float(d in EXAMENS), sense_classes, float(off)


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
    for h in hs:
        tau = t + h; hod, dow, s, c = calendar(tau)
        rows.append([h, hod, dow, s, c, hist.get(key, tau - 24), hist.get(key, tau - 48),
                     hist.get(key, tau - 168), last, last - last24, m24, irr, temp,
                     hist.get("irr", tau - 24), lm % 24, *day_flags(tau), day_flags(tau - 24)[4],
                     day_flags(tau - 168)[4]])
    return np.array(rows, dtype=float)


class AvgModel:
    """Ensemble: mitjana de les prediccions de diversos models (picklable)."""
    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)
