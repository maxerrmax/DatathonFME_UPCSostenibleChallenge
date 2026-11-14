"""
pipeline/data.py
Construcció del dataset horari IDÈNTIC al del tutorial oficial (el que probablement fa servir
el servidor) i particions temporals per a validació.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
CACHE = ROOT / "data" / "processed" / "pipeline_hourly.csv"
COLS = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]


def build_hourly() -> pd.DataFrame:
    c = pd.read_csv(TRAIN_DIR / "contador.csv", parse_dates=["Fecha"]).set_index("Fecha")
    inv = pd.read_csv(TRAIN_DIR / "inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
    pv = pd.read_csv(TRAIN_DIR / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co = pd.read_csv(TRAIN_DIR / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")
    load = (-c["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(0, 5000).resample("1h").mean().dropna()
    cols = [[x for x in inv.columns if f"INVERSOR {i}" in x.upper() and "POTENCIA ACTIVA" in x.upper()
             and "FASE" not in x.upper()][0] for i in (1, 2, 3)]
    pvk = sum(inv[k].fillna(0) for k in cols).clip(lower=0).resample("1h").mean().dropna()
    df = pd.DataFrame({"load_kw": load, "pv_kw": pvk,
                       "precio_eur_per_kwh": pv["precio_eur_per_kwh"],
                       "co2_kg_per_kwh": co["co2_kg_per_kwh"]}).dropna()
    df.index.name = "Fecha"
    return df


def load_hourly(force: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE, parse_dates=["Fecha"], index_col="Fecha")
    df = build_hourly()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE)
    return df


def blocks(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Trossos d'hores consecutives (els salts són els forats de test / dades)."""
    gid = (df.index.to_series().diff() != pd.Timedelta("1h")).cumsum()
    return [g for _, g in df.groupby(gid)]


def split_val(df: pd.DataFrame, val_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imita el test: els darrers `val_days` dies de cada bloc mensual són validació.
    Retorna (fit, val). Els dos conserven l'ordre temporal i es poden concatenar com el servidor."""
    fit, val = [], []
    for b in blocks(df):
        if len(b) < 24 * (val_days + 3):
            fit.append(b)
            continue
        # VAL_SHIFT (dies, variable d'entorn): desplaça la setmana de validació cap enrere (segona partició)
        sh = int(os.environ.get("VAL_SHIFT", "0"))
        cut = b.index[-1] - pd.Timedelta(days=val_days + sh) + pd.Timedelta("1h")
        end = cut + pd.Timedelta(days=val_days)
        fit.append(b[(b.index < cut) | (b.index >= end)])
        val.append(b[(b.index >= cut) & (b.index < end)])
    return pd.concat(fit), pd.concat(val)
