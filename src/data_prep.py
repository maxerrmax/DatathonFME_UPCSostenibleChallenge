"""
src/data_prep.py
Preprocessat offline: CSV de 5 minuts de data/train/ -> taula horaria neta.
Usat per train_pipeline.ipynb i run_hard_sim.py (NO forma part de submission.zip).
"""
from pathlib import Path

import numpy as np
import pandas as pd

def find_train_dir():
    """Localitza data/train (repo local o dataset de Kaggle a /kaggle/input/**)."""
    here = Path(__file__).resolve().parent
    cands = [here.parent / "data" / "train", here / "data" / "train", Path("data/train"),
             Path("../data/train"), Path("../../data/train")]
    for c in cands:
        if (c / "contador.csv").exists():
            return c.resolve()
    kag = Path("/kaggle/input")
    if kag.exists():
        for f in kag.rglob("contador.csv"):
            return f.parent
    return cands[0]


TRAIN_DIR = find_train_dir()
PV_SCALE = 12.0          # Mode Dificil: solar x12
MAX_LOAD_KW = 5000.0     # tall fisic recomanat a GUIA_VARIABLES.md

COL_LOAD = "Contador - POTENCIA ACTIVA EXPORTADA (kW)"
METEO_COLS = {
    "Atersa - IRRADIANCIA (W/m2)": "irradiancia_wm2",
    "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)": "temp_ambient_c",
    "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)": "temp_panel_c",
    "Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)": "vent_ms",
}


def _hourly_mean(df):
    # Mateixa convencio que starter_kit/tutorial.ipynb (i per tant que el test ocult).
    return df.resample("1h").mean()


def build_hourly(train_dir=None, pv_scale=PV_SCALE):
    """Taula horaria: load_kw (0.0/aberrants -> NaN), pv_kw (x pv_scale), meteo, preu, CO2.
    No s'imputa res aqui: els forats (novembre 2025, ultims 7 dies de cada mes) queden com NaN."""
    train_dir = Path(train_dir or TRAIN_DIR)
    rd = lambda n, cols: pd.read_csv(train_dir / n, usecols=["Fecha", *cols],
                                     parse_dates=["Fecha"]).set_index("Fecha").sort_index()
    cont = rd("contador.csv", [COL_LOAD])
    load = -cont[COL_LOAD]
    load = load.where((load > 0.0) & (load < MAX_LOAD_KW))      # 0.0 = caiguda de sensor -> NaN
    load_h = _hourly_mean(load.to_frame("load_kw"))

    inv_cols = [c for c in pd.read_csv(train_dir / "inversor.csv", nrows=0).columns
                if c.endswith("- POTENCIA ACTIVA (kW)")]
    inv = rd("inversor.csv", inv_cols)
    pv = inv.clip(lower=0).sum(axis=1, min_count=1) * pv_scale
    pv_h = _hourly_mean(pv.to_frame("pv_kw"))

    met = rd("meteo.csv", list(METEO_COLS)).rename(columns=METEO_COLS)
    met_h = _hourly_mean(met)

    price = rd("precio_pvpc.csv", ["precio_eur_per_kwh"])
    co2 = rd("co2_red.csv", ["co2_kg_per_kwh"])
    df = pd.concat([price, co2, load_h, pv_h, met_h], axis=1, sort=True)
    df = df[df["precio_eur_per_kwh"].notna()]
    df.index.name = "Fecha"
    return df


def build_env_frame(train_dir=None, pv_scale=PV_SCALE):
    """Taula 'oficial' per a CampusBatteryEnv, calcada de starter_kit/tutorial.ipynb
    (load clip [0, 5000] amb els 0.0 conservats, PV fillna(0), dropna) + PV x pv_scale
    + meteo horaria (per a noves_dades_5min en Mode Dificil)."""
    train_dir = Path(train_dir or TRAIN_DIR)
    cont = pd.read_csv(train_dir / "contador.csv", usecols=["Fecha", COL_LOAD],
                       parse_dates=["Fecha"]).set_index("Fecha")
    load = (-cont[COL_LOAD]).clip(lower=0, upper=5000).resample("1h").mean().dropna()
    inv_cols = [c for c in pd.read_csv(train_dir / "inversor.csv", nrows=0).columns
                if c.endswith("- POTENCIA ACTIVA (kW)")]
    inv = pd.read_csv(train_dir / "inversor.csv", usecols=["Fecha", *inv_cols],
                      parse_dates=["Fecha"]).set_index("Fecha")
    pv = inv.fillna(0).sum(axis=1).clip(lower=0).resample("1h").mean().dropna() * pv_scale
    met = pd.read_csv(train_dir / "meteo.csv", usecols=["Fecha", *METEO_COLS],
                      parse_dates=["Fecha"]).set_index("Fecha").rename(columns=METEO_COLS)
    met = met.resample("1h").mean()
    price = pd.read_csv(train_dir / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(train_dir / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")
    df = pd.DataFrame({"load_kw": load, "pv_kw": pv,
                       "precio_eur_per_kwh": price["precio_eur_per_kwh"],
                       "co2_kg_per_kwh": co2["co2_kg_per_kwh"]}).dropna()
    df = df.join(met)
    df.index.name = "Fecha"
    return df
