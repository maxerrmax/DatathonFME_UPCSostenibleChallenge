"""
Construcció del dataset horari del campus (Mode Estàndard i Mode Difícil).

Ús:
    from src.data import load_hourly
    df = load_hourly()             # solar real
    df_hard = load_hourly(pv_scale=12)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "train"

METEO_COLS = {
    "Atersa - IRRADIANCIA (W/m2)": "irradiancia_wm2",
    "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)": "temp_ambient_c",
    "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)": "temp_panel_c",
    "Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)": "vent_ms",
}


def _read(name, **kw):
    return pd.read_csv(DATA_DIR / name, parse_dates=["Fecha"], **kw).set_index("Fecha")


def load_raw_hourly(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Totes les sèries a resolució horària, SENSE eliminar forats (NaN on falten dades)."""
    global DATA_DIR
    DATA_DIR = Path(data_dir)

    c = _read("contador.csv", usecols=["Fecha", "Contador - POTENCIA ACTIVA EXPORTADA (kW)"])
    load = (-c.iloc[:, 0]).clip(lower=0, upper=5000).resample("1h").mean()

    inv_cols = [f"Inversor {i} - POTENCIA ACTIVA (kW)" for i in (1, 2, 3)]
    inv = _read("inversor.csv", usecols=["Fecha"] + inv_cols)
    pv = inv[inv_cols].fillna(0).sum(axis=1).clip(lower=0).resample("1h").mean()

    met = _read("meteo.csv", usecols=["Fecha"] + list(METEO_COLS)).rename(columns=METEO_COLS)
    met = met.resample("1h").mean()

    price = _read("precio_pvpc.csv")["precio_eur_per_kwh"]
    co2 = _read("co2_red.csv")["co2_kg_per_kwh"]

    idx = pd.date_range(price.index.min(), price.index.max(), freq="1h")
    df = pd.DataFrame(index=idx)
    df.index.name = "Fecha"
    df["load_kw"] = load
    df["pv_kw"] = pv
    df["precio_eur_per_kwh"] = price
    df["co2_kg_per_kwh"] = co2
    for col in METEO_COLS.values():
        df[col] = met[col]
    return df


def load_hourly(pv_scale: float = 1.0, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Dataset horari llest per al simulador (com el tutorial oficial: s'eliminen les hores amb forats)."""
    df = load_raw_hourly(data_dir)
    df = df.dropna(subset=["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]).copy()
    df["pv_kw"] = df["pv_kw"] * pv_scale
    for col in METEO_COLS.values():
        df[col] = df[col].interpolate(limit_direction="both")
    return df


def validation_weeks(df: pd.DataFrame, start_day: int = 15, n_days: int = 7) -> pd.DataFrame:
    """Selecciona un bloc de 7 dies per mes (imita les 12 setmanes de test) i els concatena."""
    blocks = []
    for _, g in df.groupby([df.index.year, df.index.month]):
        w = g[(g.index.day >= start_day) & (g.index.day < start_day + n_days)]
        if len(w) == 24 * n_days:
            blocks.append(w)
    return pd.concat(blocks)
