"""Features del valor marginal de l'energia emmagatzemada (només numpy; compartides per l'entrenament i l'agent).

Disponibilitat a l'instant t (com al simulador): preu i CO2 fins a t+2; consum, solar i meteo fins a t-1.
Tot el mercat s'expressa en w = 0.7*CO2 + 0.3*ratio*preu i es normalitza per la mitjana de w de les últimes 24 h
(`scale`), perquè el model aprengui formes relatives i no nivells (generalitza entre estacions).
"""
import numpy as np
from feats import calendar, day_flags

SEG = 600.0                     # trams de SOC del valor terminal (10 x 600 kWh = 6.000 kWh)
NSEG = 10
SOC_MID = (np.arange(NSEG) + 0.5) * SEG


def wval(get, e, ratio):
    return 0.7 * get("co2", e) + 0.3 * ratio * get("price", e)


def base_features(get, t, ratio):
    """Vector de features a l'instant t (sense el SOC) i l'escala de normalització."""
    wk = np.array([wval(get, t + i, ratio) for i in range(3)])                 # t..t+2 (exactes)
    wb = np.array([wval(get, t - 1 - i, ratio) for i in range(48)])            # t-1..t-48
    last24 = wb[:24]
    ok = np.isfinite(last24)
    scale = float(np.mean(last24[ok])) if ok.sum() >= 6 else float(np.nanmean(wk))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 0.08
    hod, dow, ds, dc = calendar(t)
    off0 = day_flags(t)[6]; off1 = day_flags(t + 24)[6]
    # dia anterior a les mateixes hores que t+3..t+26 (perfil d'ahir de l'horitzó que ve)
    wy = np.array([wval(get, t + 3 + i - 24, ratio) for i in range(24)])
    ld = np.array([get("load", t - 1 - i) for i in range(24)])
    pv = np.array([get("pv", t - 1 - i) for i in range(24)])
    irr = get("irr", t - 1)
    with np.errstate(all="ignore"):
        f = np.concatenate([
            [hod, dow, ds, dc, off0, off1, scale],
            wk / scale, wb / scale, wy / scale,
            [np.nanmin(last24) / scale, np.nanmax(last24) / scale, np.nanstd(last24) / scale,
             (wk[2] - wb[0]) / scale],
            [ld[0], np.nanmean(ld), pv[0], np.nanmean(pv), np.nanmax(pv), irr, ld[0] - pv[0]],
        ])
    return f, scale


def with_soc(f, soc_levels, cap=6000.0):
    X = np.repeat(f[None, :], len(soc_levels), axis=0)
    return np.column_stack([X, np.asarray(soc_levels, float) / cap])


N_BASE = 7 + 3 + 48 + 24 + 4 + 7
SOC_COL = N_BASE                  # índex de la columna del SOC (monòtona decreixent)
