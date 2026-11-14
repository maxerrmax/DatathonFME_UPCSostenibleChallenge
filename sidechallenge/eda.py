"""Robust, Tukey-style exploratory statistics with explicit data-quality flags.

Medians and quartiles instead of bare means, day-level bootstrap intervals for
cohort comparisons, and a data-quality layer that makes measurement artefacts
(frozen meter readings, imputed solar, test gaps) visible before any conclusion.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

ENERGY = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]


def robust_profile(df: pd.DataFrame, metric: str, by: str = "hour") -> pd.DataFrame:
    """Median and inter-quartile range per group (resistant to outliers)."""
    g = df.groupby(by)[metric]
    out = pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                        "mean": g.mean(), "n": g.count()})
    return out.reindex(range(24)) if by == "hour" else out


def frozen_hours(df: pd.DataFrame) -> list[int]:
    """Hours where the meter repeats its previous reading on most days."""
    if "load_congelada" not in df or "hour" not in df:
        return []
    share = df.groupby("hour")["load_congelada"].mean()
    return [int(h) for h, v in share.items() if v >= 0.5]


def data_quality(df: pd.DataFrame) -> dict:
    """Facts an analyst must know before trusting a chart of this dataset."""
    out: dict = {"rows": int(len(df)), "issues": []}
    if "Fecha" not in df:
        return out
    first = df["Fecha"].min().to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    last = df["Fecha"].max().to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0) + _dt.timedelta(days=1)
    full = pd.date_range(first, last, freq="h", inclusive="left")
    missing = full.difference(df["Fecha"])
    out["expected_hours"] = int(len(full))
    out["missing_hours"] = int(len(missing))
    months_present = sorted(int(m) for m in df["month"].dropna().unique()) if "month" in df else []
    out["months_present"] = months_present
    out["months_absent"] = [m for m in range(1, 13) if m not in months_present]
    if "load_congelada" in df:
        fr = frozen_hours(df)
        share = float(df["load_congelada"].mean())
        out["frozen_share_pct"] = round(100 * share, 1)
        out["frozen_hours"] = fr
        if share > 0.01:
            out["issues"].append({
                "level": "alta", "title": "Lectures congelades del comptador",
                "detail": f"{ca(100 * share)}% de les hores repeteixen exactament la lectura anterior, concentrades a les hores "
                          f"{_hours_label(fr)}. El perfil nocturn de consum és un valor mantingut, no una mesura.",
                "action": "No interpretar la forma nocturna del consum; tractar-la com a constant coneguda."})
    if "pv_imputada" in df:
        imp = float(df["pv_imputada"].mean())
        out["pv_imputed_pct"] = round(100 * imp, 1)
        if imp > 0.005:
            out["issues"].append({
                "level": "mitjana", "title": "Solar absent imputada a 0",
                "detail": f"{ca(100 * imp)}% de les hores no tenen telemetria dels inversors i s'han posat a 0 kW.",
                "action": "La generació solar total està lleugerament infravalorada."})
    if out["missing_hours"]:
        out["issues"].append({
            "level": "informativa", "title": "Forats de calendari (test ocult)",
            "detail": f"Falten {ca(out['missing_hours'], 0)} hores: els últims 7 dies de cada mes (test) i"
                      + (" el mes de " + ", ".join(_month_name(m) for m in out["months_absent"]) if out["months_absent"] else " cap mes sencer")
                      + ".",
            "action": "Les mitjanes mensuals no cobreixen el final de mes; el backtest usa els 7 dies anteriors."})
    if all(c in df for c in ("load_kw", "pv_kw")):
        net = (df["load_kw"] - df["pv_kw"]).dropna()
        out["min_net_load_kw"] = round(float(net.min()), 1)
        out["pv_surplus_hours"] = int((net < 0).sum())
        out["issues"].append({
            "level": "informativa", "title": "El campus sempre importa de la xarxa",
            "detail": f"La demanda neta mínima és {ca(net.min(), 0)} kW i la solar màxima {ca(df['pv_kw'].max(), 0)} kW: "
                      f"no hi ha excedents solars ({int((net < 0).sum())} hores).",
            "action": "Carregar la bateria sempre implica importar: el valor ve de moure consum entre hores."})
    per_month = []
    if "month" in df:
        for m, part in df.groupby("month"):
            item = {"month": int(m), "hours": int(len(part)), "days": int(part["date"].nunique())}
            if "load_congelada" in part:
                item["frozen_pct"] = round(100 * float(part["load_congelada"].mean()), 1)
            per_month.append(item)
    out["per_month"] = per_month
    return out


def bootstrap_diff(a_days: pd.DataFrame, b_days: pd.DataFrame, metric: str, n_boot: int = 400, seed: int = 7) -> dict:
    """Hourly difference (b − a) of means with a day-level bootstrap 95% interval.

    Days, not hours, are resampled: consecutive hours of one day are not
    independent observations.
    """
    rng = np.random.default_rng(seed)
    A = a_days.pivot_table(index="date", columns="hour", values=metric).reindex(columns=range(24))
    B = b_days.pivot_table(index="date", columns="hour", values=metric).reindex(columns=range(24))
    if A.empty or B.empty:
        return {}
    a, b = A.to_numpy(float), B.to_numpy(float)
    boot_a = np.array([np.nanmean(a[rng.integers(0, len(a), len(a))], 0) for _ in range(n_boot)])
    boot_b = np.array([np.nanmean(b[rng.integers(0, len(b), len(b))], 0) for _ in range(n_boot)])
    diff = boot_b - boot_a
    lo, hi = np.nanpercentile(diff, [2.5, 97.5], axis=0)
    return {"point": np.nanmean(b, 0) - np.nanmean(a, 0), "lo": lo, "hi": hi,
            "mean_a": np.nanmean(a, 0), "mean_b": np.nanmean(b, 0),
            "a_lo": np.nanpercentile(boot_a, 2.5, axis=0), "a_hi": np.nanpercentile(boot_a, 97.5, axis=0),
            "b_lo": np.nanpercentile(boot_b, 2.5, axis=0), "b_hi": np.nanpercentile(boot_b, 97.5, axis=0),
            "days_a": int(len(a)), "days_b": int(len(b)),
            "significant_hours": [int(h) for h in range(24) if np.isfinite(lo[h]) and (lo[h] > 0 or hi[h] < 0)]}


def signal_by_month(df: pd.DataFrame) -> dict:
    """70/30 hourly signal per month and the best charge→discharge pair (after 90% round trip)."""
    clean = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh", "hour", "month"])
    pm, cm = clean["precio_eur_per_kwh"].mean(), clean["co2_kg_per_kwh"].mean()
    clean = clean.assign(signal=0.7 * clean["co2_kg_per_kwh"] / cm + 0.3 * clean["precio_eur_per_kwh"] / pm)
    matrix = clean.pivot_table(index="month", columns="hour", values="signal", aggfunc="mean").reindex(columns=range(24))
    best = []
    for m, row in matrix.iterrows():
        sig = row.to_numpy(float)
        if np.isnan(sig).any():
            continue
        spread, c, d = max((0.9 * sig[d] - sig[c], c, d) for c in range(24) for d in range(c + 1, 24))
        days = clean[clean["month"] == m].pivot_table(index="date", columns="hour", values="signal")
        per_day = (0.9 * days.get(d) - days.get(c)).dropna() if c in days and d in days else pd.Series(dtype=float)
        best.append({"month": int(m), "charge_hour": int(c), "discharge_hour": int(d), "spread": round(float(spread), 3),
                     "day_spread_q25": round(float(per_day.quantile(.25)), 3) if len(per_day) else None,
                     "day_spread_median": round(float(per_day.median()), 3) if len(per_day) else None,
                     "day_spread_q75": round(float(per_day.quantile(.75)), 3) if len(per_day) else None,
                     "days": int(len(per_day))})
    return {"matrix": matrix, "best": best}


def _hours_label(hours: list[int]) -> str:
    if not hours:
        return "—"
    hs = sorted(hours)
    # compress a wrap-around run such as 22,23,0..5 into "22–05 h"
    runs, start, prev = [], hs[0], hs[0]
    for h in hs[1:]:
        if h != prev + 1:
            runs.append((start, prev)); start = h
        prev = h
    runs.append((start, prev))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == 23:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return ", ".join(f"{a:02d}–{b:02d} h" if a != b else f"{a:02d} h" for a, b in runs)


def _month_name(m: int) -> str:
    return ["gener", "febrer", "març", "abril", "maig", "juny", "juliol", "agost", "setembre", "octubre", "novembre", "desembre"][m - 1]


def ca(x, dec: int = 1) -> str:
    """Catalan number formatting: 1.234,5"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    s = f"{v:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")
