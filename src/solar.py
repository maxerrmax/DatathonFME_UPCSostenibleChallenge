"""
Model físic de previsió solar:  solar(T) = cel_clar(dia de l'any, hora) × claredat prevista(k)

    claredat prevista = w_ara[k] · claredat de l'última hora
                      + w_ahir[k] · claredat de les darreres 24 h
                      + (1 - w_ara - w_ahir) · claredat mitjana (mes, hora)

Taules (es desen a solar.json i les llegeix l'agent):
    cs   [366][24]  potència solar de cel clar (quantil 97 per finestra de ±12 dies), kW reals (sense escalar)
    ktc  [12][24]   claredat mitjana per mes i hora
    w_now, w_d1 [24] pesos per horitzó k
"""
import json

import numpy as np

CS_MIN = 8.0  # kW: per sota, es considera nit/crepuscle


def build_tables(times, pv, keep_mask):
    """times: DatetimeIndex horari; pv: array kW reals (NaN = sense dada); keep_mask: hores usables."""
    doy = times.dayofyear.values
    hr = times.hour.values
    v_all = np.where(keep_mask, pv, np.nan)
    cs = np.zeros((366, 24))
    for h in range(24):
        sel = hr == h
        d, v = doy[sel], v_all[sel]
        for dd in range(1, 367):
            dist = np.minimum(np.abs(d - dd), 366 - np.abs(d - dd))
            m = (dist <= 12) & ~np.isnan(v)
            cs[dd - 1, h] = np.quantile(v[m], 0.97) if m.sum() > 5 else np.nan
        col = cs[:, h]
        bad = np.isnan(col)
        if bad.all():
            cs[:, h] = 0.0
        elif bad.any():
            idx = np.arange(366)
            cs[bad, h] = np.interp(idx[bad], idx[~bad], col[~bad], period=366)
    csv = cs[doy - 1, hr]
    kt = np.where(csv > CS_MIN, np.clip(pv / np.maximum(csv, 1e-9), 0, 1.3), np.nan)
    kt = np.where(keep_mask, kt, np.nan)
    ktc = np.full((12, 24), 0.6)
    mon = times.month.values
    for m in range(1, 13):
        for h in range(24):
            x = kt[(mon == m) & (hr == h)]
            x = x[~np.isnan(x)]
            if len(x) > 5:
                ktc[m - 1, h] = float(x.mean())
    return cs, ktc


def fit_weights(samples):
    """samples[k] = array (y, cs, kt_now, kt_d1, ktc) -> pesos per k minimitzant MAE (cerca en graella)."""
    w_now, w_d1 = [], []
    for k in range(24):
        a = np.asarray(samples[k], float)
        y, c, kn, kd, kc = a.T
        kd = np.where(np.isnan(kd), kc, kd)
        kn = np.where(np.isnan(kn), kd, kn)
        best = (np.inf, 0.0, 0.0)
        for an in np.linspace(0, 1, 21):
            for ad in np.linspace(0, 1 - an, int(round((1 - an) * 20)) + 1):
                mae = np.mean(np.abs(c * (an * kn + ad * kd + (1 - an - ad) * kc) - y))
                if mae < best[0]:
                    best = (mae, an, ad)
        w_now.append(best[1]); w_d1.append(best[2])
    return w_now, w_d1


def save(path, cs, ktc, w_now, w_d1):
    with open(path, "w") as f:
        json.dump({"cs": np.round(cs, 2).tolist(), "ktc": np.round(ktc, 4).tolist(),
                   "w_now": list(map(float, w_now)), "w_d1": list(map(float, w_d1))}, f)
