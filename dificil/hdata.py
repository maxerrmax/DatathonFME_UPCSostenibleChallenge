"""Dataset horari Mode Difícil: mateix processament que el tutorial + meteo horària; PV ×12."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train"
CACHE = ROOT / "data" / "processed" / "hard_hourly.csv"
PV_SCALE = 12.0
METEO = {"irradiancia_wm2": "Atersa - IRRADIANCIA (W/m2)",
         "temp_ambient_c": "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)",
         "temp_panel_c": "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)",
         "vent_ms": "Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)"}


def load_hard(force=False):
    if CACHE.exists() and not force:
        return pd.read_csv(CACHE, parse_dates=["Fecha"], index_col="Fecha")
    import sys; sys.path.insert(0, str(ROOT / "estandard"))
    from data import build_hourly
    df = build_hourly()
    me = pd.read_csv(TRAIN / "meteo.csv", parse_dates=["Fecha"]).set_index("Fecha")
    for k, c in METEO.items():
        df[k] = me[c].resample("1h").mean().reindex(df.index)
    df["pv_kw"] = df["pv_kw"] * PV_SCALE
    CACHE.parent.mkdir(parents=True, exist_ok=True); df.to_csv(CACHE)
    return df
