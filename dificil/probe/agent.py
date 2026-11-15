"""
Agent de DIAGNÒSTIC (no competeix). Escriu el que rep del servidor dins de les dues coses que mostra la pàgina de
resultats: el gràfic de consigna/SOC de les primeres 72 hores i el comptador d'infraccions.

  Pas 0 de cada objecte nou   consigna = potència màxima + 200 kW (s'acota -> 1 infracció per instància).
                              "Infraccions = N" vol dir que el servidor ha creat l'agent N vegades.
  Hores 1..9 (capçalera)      |consigna| codifica un fet (vegeu LLEGENDA més avall). El signe NO porta informació:
                              sempre s'empeny el SOC cap al 50 % perquè no toqui cap límit.
  Hores 10..11                0 kW (separador)
  Hores 12..41                |consigna| = ultim_consum_kw / 2   (comparar amb la corba de demanda del gràfic)
  Hores 42..71                |consigna| = ultima_solar_kw / 2   (comparar amb la corba de solar del gràfic)
  Hores >= 72                 0 kW

LLEGENDA de la capçalera (hora des de l'inici):
  1  50 x longitud de preus_properes_24h          (3 h -> 150 kW, 24 h -> 1200 kW)
  2  50 x longitud de co2_properes_24h
  3  50 x nombre de claus del diccionari obs
  4  50 x nombre de claus de noves_dades_5min
  5  100 x tipus de timestamp: 1 text, 2 pandas sense zona, 3 pandas amb zona, 4 datetime sense zona,
                              5 datetime amb zona, 6 numpy datetime64, 7 altre / absent
  6  cap_max_kwh / 10         (6.000 -> 600 kW; 0 si no hi és)
  7  power_max_kw / 2         (1.500 -> 750 kW; 0 si no hi és)
  8  100 x (1 + diferència en hores entre l'hora del timestamp (rellotge sense zona) i obs["hour"], mòdul 24)
                              (100 kW = coincideixen)
  9  100 x codi de la finestra: 1 = preus[0] == preu_actual i co2[0] == co2_actual; 2 = només el preu coincideix;
                              3 = només el CO2 coincideix; 4 = cap; 5 = finestres buides
"""
import math

ETA = 0.9 ** 0.5


def _f(x, d=float("nan")):
    try:
        v = float(x)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def _ts_code(ts):
    if ts is None:
        return 7
    if isinstance(ts, str):
        return 1
    mod = type(ts).__module__
    name = type(ts).__name__
    if "pandas" in mod and name == "Timestamp":
        return 3 if getattr(ts, "tzinfo", None) is not None else 2
    if name == "datetime":
        return 5 if getattr(ts, "tzinfo", None) is not None else 4
    if name == "datetime64":
        return 6
    return 7


def _ts_hour(ts):
    """Hora del rellotge del timestamp tal com ve (sense convertir de zona)."""
    try:
        if isinstance(ts, str):
            return int(ts.strip()[11:13])
        h = getattr(ts, "hour", None)
        if h is not None:
            return int(h)
        s = str(ts)
        return int(s[11:13])
    except Exception:
        return None


class BatteryAgent:
    def __init__(self, params=None):
        self.k = 0                       # passos d'aquesta instància

    def _push(self, obs, mag):
        """Magnitud |mag| amb el signe que porta el SOC cap al 50 %, dins dels límits físics."""
        cap = _f(obs.get("cap_max_kwh"), 6000.0)
        pmax = _f(obs.get("power_max_kw"), 1500.0)
        soc = _f(obs.get("soc_kwh"), cap / 2)
        mag = max(0.0, min(abs(mag), pmax))
        if soc > cap / 2:
            return -min(mag, max(0.0, soc * ETA - 1.0))
        return min(mag, max(0.0, (cap - soc) / ETA - 1.0))

    def _header(self, obs, i):
        pr = obs.get("preus_properes_24h") or []
        cr = obs.get("co2_properes_24h") or []
        nd = obs.get("noves_dades_5min") or {}
        ts = obs.get("timestamp")
        if i == 1:
            return 50.0 * len(pr)
        if i == 2:
            return 50.0 * len(cr)
        if i == 3:
            return 50.0 * len(obs)
        if i == 4:
            return 50.0 * (len(nd) if hasattr(nd, "__len__") else 0)
        if i == 5:
            return 100.0 * _ts_code(ts)
        if i == 6:
            return _f(obs.get("cap_max_kwh"), 0.0) / 10.0
        if i == 7:
            return _f(obs.get("power_max_kw"), 0.0) / 2.0
        if i == 8:
            h = _ts_hour(ts)
            oh = obs.get("hour")
            if h is None or oh is None:
                return 0.0
            return 100.0 * (1 + (int(h) - int(oh)) % 24)
        if i == 9:
            if not pr or not cr:
                return 500.0
            p_ok = abs(_f(pr[0]) - _f(obs.get("preu_actual"))) < 1e-9
            c_ok = abs(_f(cr[0]) - _f(obs.get("co2_actual"))) < 1e-9
            return 100.0 * (1 if p_ok and c_ok else 2 if p_ok else 3 if c_ok else 4)
        return 0.0

    def act(self, obs):
        k = self.k
        self.k += 1
        try:
            obs = obs or {}
            if k == 0:
                return float(_f(obs.get("power_max_kw"), 1500.0) + 200.0)     # marca d'instància (1 infracció)
            if k <= 9:
                return float(self._push(obs, self._header(obs, k)))
            if k <= 11:
                return 0.0
            nd = obs.get("noves_dades_5min") or {}
            if k <= 41:
                return float(self._push(obs, _f(nd.get("ultim_consum_kw"), 0.0) / 2.0))
            if k <= 71:
                return float(self._push(obs, _f(nd.get("ultima_solar_kw"), 0.0) / 2.0))
            return 0.0
        except Exception:
            return 0.0
