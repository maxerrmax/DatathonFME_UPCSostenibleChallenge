"""
dashboard/data.py
Carga, limpieza y agregación horaria de los CSV de data/train/.

Genera una única tabla horaria con todo lo necesario para el dashboard y la
guarda en data/processed/hourly.parquet para que las siguientes cargas sean
instantáneas (se regenera sola si cambia algún CSV).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
CACHE = ROOT / "data" / "processed" / "hourly.parquet"

MAX_LOAD_KW = 5000.0  # corte físico recomendado en GUIA_VARIABLES.md

# Columnas de interés de cada CSV (nombre original -> nombre corto)
CONTADOR_COLS = {
    "Contador - POTENCIA ACTIVA EXPORTADA (kW)": "p_export_kw",
    "Dispositivo 1 - POTENCIA REACTIVA (kVAr)": "q_kvar",
    "Dispositivo 1 - FACTOR DE POTENCIA": "power_factor",
    "Contador - TENSIÓN AC FASE 1-N (V)": "v1",
    "Contador - TENSIÓN AC FASE 2-N (V)": "v2",
    "Contador - TENSIÓN AC FASE 3-N (V)": "v3",
    "Contador - CORRIENTE AC FASE 1 (A)": "i1",
    "Contador - CORRIENTE AC FASE 2 (A)": "i2",
    "Contador - CORRIENTE AC FASE 3 (A)": "i3",
    "Contador - CALIDAD COM": "contador_calidad",
}
METEO_COLS = {
    "Atersa - IRRADIANCIA (W/m2)": "irradiancia",
    "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)": "temp_amb",
    "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)": "temp_panel",
    "Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)": "viento",
    "Atersa - CALIDAD COM": "meteo_calidad",
}
RELE_COL = "IoLogik 1214 - SALIDA RELE 0"

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
DIAS_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def _read(name: str, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(TRAIN_DIR / name, usecols=usecols, parse_dates=["Fecha"])
    return df.set_index("Fecha").sort_index()


def _hourly_mean(df: pd.DataFrame) -> pd.DataFrame:
    # Las marcas de 5 min van de HH:05 a (HH+1):00 -> la hora "t" agrupa (t, t+1].
    # Usamos label/closed='left' sobre el instante desplazado 5 min para que
    # la hora 10:00 contenga las muestras 10:05 ... 11:00 (la hora 10-11).
    shifted = df.copy()
    shifted.index = shifted.index - pd.Timedelta(minutes=5)
    return shifted.resample("1h").mean()


def _hourly_count(df: pd.DataFrame) -> pd.Series:
    shifted = df.copy()
    shifted.index = shifted.index - pd.Timedelta(minutes=5)
    return shifted.resample("1h").count()


def build_hourly() -> pd.DataFrame:
    """Construye la tabla horaria a partir de los CSV en bruto."""
    # --- Contador (consumo neto del campus en el PCC)
    cont = _read("contador.csv", usecols=["Fecha", *CONTADOR_COLS]).rename(columns=CONTADOR_COLS)
    cont["load_kw_raw"] = -cont["p_export_kw"]
    cont["load_kw"] = cont["load_kw_raw"].where(
        (cont["load_kw_raw"] > 0) & (cont["load_kw_raw"] < MAX_LOAD_KW)
    )
    # Lecturas "congeladas": la muestra repite exactamente el valor anterior.
    # Cada noche (~22:05-06:00) el contador mantiene la última lectura.
    raw = cont["load_kw_raw"]
    cont["repetida"] = (raw == raw.shift()).astype(float).where(raw.notna())
    cont_h = _hourly_mean(cont.drop(columns=["p_export_kw"]))
    cont_h["load_congelada"] = cont_h["repetida"] > 0.5
    cont_h = cont_h.drop(columns=["repetida"])
    cont_h["n_contador"] = _hourly_count(cont[["load_kw"]])["load_kw"]

    # --- Inversores (generación FV)
    inv_cols_all = pd.read_csv(TRAIN_DIR / "inversor.csv", nrows=0).columns
    inv_p = [c for c in inv_cols_all if c.endswith("- POTENCIA ACTIVA (kW)")]
    inv_t = [c for c in inv_cols_all if c.endswith("- TEMPERATURA INT (degC)")]
    inv = _read("inversor.csv", usecols=["Fecha", *inv_p, *inv_t])
    ren = {c: f"pv_inv{c.split()[1]}_kw" for c in inv_p}
    ren.update({c: f"temp_inv{c.split()[1]}" for c in inv_t})
    inv = inv.rename(columns=ren)
    pv_cols = sorted(c for c in inv.columns if c.startswith("pv_inv"))
    inv["pv_kw"] = inv[pv_cols].clip(lower=0).sum(axis=1, min_count=1)
    inv_h = _hourly_mean(inv)
    inv_h["n_inversor"] = _hourly_count(inv[["pv_kw"]])["pv_kw"]

    # --- Meteo
    met = _read("meteo.csv", usecols=["Fecha", *METEO_COLS]).rename(columns=METEO_COLS)
    met_h = _hourly_mean(met)
    met_h["n_meteo"] = _hourly_count(met[["irradiancia"]])["irradiancia"]

    # --- Relé digital
    rel = _read("entradas_salidas.csv", usecols=["Fecha", RELE_COL]).rename(columns={RELE_COL: "rele"})
    rel_h = _hourly_mean(rel)

    # --- Mercado (ya horario)
    pvpc = _read("precio_pvpc.csv", usecols=["Fecha", "precio_eur_per_kwh"])
    co2 = _read("co2_red.csv")

    df = pd.concat([pvpc, co2, cont_h, inv_h, met_h, rel_h], axis=1, sort=True)
    df.index.name = "Fecha"
    df = df[df.index < "2026-09-02"]
    df["load_congelada"] = df["load_congelada"].astype("boolean").fillna(False).astype(bool)

    # --- Columnas derivadas de calendario
    idx = df.index
    df["fecha_dia"] = idx.normalize()
    df["hora"] = idx.hour
    df["dow"] = idx.dayofweek
    df["dia_semana"] = pd.Categorical([DIAS[d] for d in idx.dayofweek], categories=DIAS, ordered=True)
    df["mes_num"] = idx.month
    df["mes"] = idx.to_period("M").astype(str)
    df["finde"] = idx.dayofweek >= 5

    # El test oculto son los últimos 7 días de cada mes natural
    days_in_month = idx.days_in_month
    df["es_test"] = idx.day > (days_in_month - 7)

    # Hora "completa": hay consumo, precio y CO2 (la solar ausente se imputa a 0)
    df["pv_imputada"] = df["pv_kw"].isna() & df["load_kw"].notna()
    df["pv_kw_sim"] = df["pv_kw"].fillna(0.0)
    df["completa"] = df[["load_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]].notna().all(axis=1)

    # Escenario base (sin batería) del enunciado
    df["neto_kw"] = df["load_kw"] - df["pv_kw_sim"]
    df["red_base_kw"] = df["neto_kw"].clip(lower=0)
    df["coste_base_eur"] = df["red_base_kw"] * df["precio_eur_per_kwh"]
    df["co2_base_kg"] = df["red_base_kw"] * df["co2_kg_per_kwh"]

    return df


def _cache_is_fresh() -> bool:
    if not CACHE.exists():
        return False
    mtime = CACHE.stat().st_mtime
    return all(p.stat().st_mtime <= mtime for p in TRAIN_DIR.glob("*.csv"))


def load_hourly(force: bool = False) -> pd.DataFrame:
    """Devuelve la tabla horaria (desde caché parquet si está al día)."""
    if not force and _cache_is_fresh():
        try:
            df = pd.read_parquet(CACHE)
            df["dia_semana"] = pd.Categorical(df["dia_semana"], categories=DIAS, ordered=True)
            return df
        except Exception:
            pass
    df = build_hourly()
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE)
    except Exception:
        pass  # sin pyarrow o sin permisos: seguimos sin caché
    return df


def sim_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla en el formato que espera starter_kit/environment.py."""
    out = df.loc[df["completa"], ["load_kw", "pv_kw_sim", "precio_eur_per_kwh", "co2_kg_per_kwh"]]
    return out.rename(columns={"pv_kw_sim": "pv_kw"})


if __name__ == "__main__":
    d = load_hourly(force=True)
    print(d.shape)
    print(d[["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]].describe().round(3))
