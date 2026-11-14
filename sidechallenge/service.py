"""UPC Energy Intelligence — evidence-first analytics engine.

Standalone FastAPI app. This module intentionally never imports or changes a
participant's BatteryAgent. The Decision Lab uses retrospective dispatch only.
"""
from __future__ import annotations

import io
from contextvars import ContextVar
import json
import math
import os
import re
import unicodedata
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.optimize import linprog
from plotly.subplots import make_subplots

from . import arena, copilot, eda

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SAMPLE = HERE / "sample_hourly.csv"
MAX_UPLOAD_BYTES = 45 * 1024 * 1024
DAYS = ["Dl", "Dt", "Dc", "Dj", "Dv", "Ds", "Dg"]
FIELD_NAMES = {
    "load_kw": "Consum del campus",
    "pv_kw": "Generació solar",
    "net_kw": "Demanda neta",
    "precio_eur_per_kwh": "Preu de l'electricitat",
    "co2_kg_per_kwh": "Intensitat de CO₂",
    "temp_c": "Temperatura ambient",
    "irr_wm2": "Irradiància",
    "baseline_cost": "Cost sense bateria",
    "baseline_co2": "Emissions sense bateria",
}
UNITS = {"load_kw": "kW", "pv_kw": "kW", "net_kw": "kW", "precio_eur_per_kwh": "€/kWh", "co2_kg_per_kwh": "kg CO₂/kWh", "temp_c": "°C", "irr_wm2": "W/m²", "baseline_cost": "€", "baseline_co2": "kg CO₂"}
PALETTE = ["#E77741", "#4A6C9A", "#789A77", "#9C76A9", "#D0A64C", "#4F7580"]

app = FastAPI(title="UPC Energy Intelligence", version="3.0.0")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    source: str
    owner: str | None = None

SESSION: ContextVar[str | None] = ContextVar("upc_session", default=None)

DATASETS: dict[str, Dataset] = {}


@app.middleware("http")
async def anonymous_session(request: Request, call_next):
    """Isolate uploaded datasets per browser, without storing personal details."""
    sid = request.cookies.get("upc_session")
    if not sid or not re.fullmatch(r"[0-9a-f]{32}", sid):
        sid = uuid.uuid4().hex
        new_session = True
    else:
        new_session = False
    context_token = SESSION.set(sid)
    try:
        response = await call_next(request)
    finally:
        SESSION.reset(context_token)
    if new_session:
        response.set_cookie("upc_session", sid, httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return response


def _num(x: Any, decimals: int = 4) -> float | None:
    try:
        n = float(x)
        if math.isfinite(n):
            return round(n, decimals)
    except (TypeError, ValueError):
        pass
    return None


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise relevant campus columns; preserve generic uploaded numeric columns."""
    df = frame.copy()
    aliases = {
        "pv_kw_sim": "pv_kw", "temp_amb": "temp_c", "irradiancia": "irr_wm2",
        "coste_base_eur": "baseline_cost", "co2_base_kg": "baseline_co2",
        "neto_kw": "net_kw", "red_base_kw": "net_kw",
        "precio": "precio_eur_per_kwh", "co2": "co2_kg_per_kwh",
    }
    for a, b in aliases.items():
        if a in df.columns and b not in df.columns:
            df = df.rename(columns={a: b})
    if "Fecha" not in df.columns:
        for c in df.columns:
            if c.lower() in {"timestamp", "datetime", "date", "fecha", "time", "data"}:
                df = df.rename(columns={c: "Fecha"})
                break
    if "Fecha" in df.columns:
        dt = pd.to_datetime(df["Fecha"], errors="coerce", utc=True)
        # The official dataset is naive local datetime; converting with utc=True
        # preserves displayed clock hours for that source (no timezone conversion).
        if dt.notna().mean() > 0.8:
            df["Fecha"] = dt.dt.tz_localize(None)
            df = df.dropna(subset=["Fecha"]).sort_values("Fecha")
            df = df.drop_duplicates(subset=["Fecha"], keep="last")
        else:
            df = df.drop(columns="Fecha")
    for c in df.columns:
        if c != "Fecha" and df[c].dtype == object:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() >= 0.9 * df[c].notna().sum() and converted.notna().sum() > 0:
                df[c] = converted
    if "Fecha" in df.columns:
        df["hour"] = df["Fecha"].dt.hour
        df["weekday"] = df["Fecha"].dt.dayofweek
        df["month"] = df["Fecha"].dt.month
        df["date"] = df["Fecha"].dt.strftime("%Y-%m-%d")
    else:
        df["row_id"] = np.arange(len(df))
    for c in ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "load_kw" in df and "pv_kw" in df:
        if "net_kw" not in df:
            df["net_kw"] = df["load_kw"] - df["pv_kw"]
        if "precio_eur_per_kwh" in df and "baseline_cost" not in df:
            df["baseline_cost"] = df["net_kw"].clip(lower=0) * df["precio_eur_per_kwh"]
        if "co2_kg_per_kwh" in df and "baseline_co2" not in df:
            df["baseline_co2"] = df["net_kw"].clip(lower=0) * df["co2_kg_per_kwh"]
    return df.reset_index(drop=True)


def _load_repo_data() -> pd.DataFrame | None:
    dashboard = ROOT / "dashboard"
    if not (dashboard / "data.py").is_file() or not (ROOT / "data" / "train").exists():
        return None
    sys.path.insert(0, str(dashboard))
    try:
        from data import load_hourly
        raw = load_hourly()
        if "completa" in raw:
            raw = raw[raw["completa"]]
        return normalize_frame(raw.reset_index())
    except Exception as exc:
        print("[UPC Energy] Could not read repo data:", exc)
        return None
    finally:
        sys.path.remove(str(dashboard))


def startup_data() -> None:
    raw = _load_repo_data()
    if raw is not None and len(raw):
        DATASETS["campus"] = Dataset("Campus Nord · Train oficial", raw, "data/train del repositori")
    elif DEFAULT_SAMPLE.exists():
        raw = pd.read_csv(DEFAULT_SAMPLE)
        DATASETS["campus"] = Dataset("Campus Nord · Mostra de train", normalize_frame(raw), "Mostra real consolidada")


startup_data()


def _get_dataset(dataset_id: str) -> Dataset:
    ds = DATASETS.get(dataset_id)
    if not ds or (ds.owner is not None and ds.owner != SESSION.get()):
        raise HTTPException(404, "Dataset desconegut")
    return ds


def _numeric_fields(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if c not in {"hour", "weekday", "month", "row_id"}]


def _energy_ready(df: pd.DataFrame) -> bool:
    return all(x in df.columns for x in ("load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh", "Fecha"))


def _catalogue(ds_id: str, ds: Dataset) -> dict:
    df = ds.frame
    fields = _numeric_fields(df)
    return {
        "id": ds_id, "name": ds.name, "source": ds.source,
        "rows": len(df), "cols": len(df.columns),
        "start": str(df["Fecha"].min().date()) if "Fecha" in df else None,
        "end": str(df["Fecha"].max().date()) if "Fecha" in df else None,
        "metrics": [{"key": c, "label": FIELD_NAMES.get(c, c), "unit": UNITS.get(c, "")} for c in fields],
        "missing_pct": _num(df[fields].isna().mean().mean() * 100, 2) if fields else 0,
        "energy_ready": _energy_ready(df),
        "months": sorted(int(m) for m in df["month"].dropna().unique()) if "month" in df else [],
        "official": ds.source.startswith("data/train"),
    }


@app.get("/")
def home():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    campus = DATASETS.get("campus")
    if campus is not None and campus.source.startswith("data/train") and _energy_ready(campus.frame):
        arena.start_background(campus.frame)
    return {
        "status": "ok", "mode": "helmcode" if os.getenv("HELMCODE_API_KEY") and os.getenv("HELMCODE_MODEL") else "local",
        "datasets": [_catalogue(k, v) for k, v in DATASETS.items() if v.owner is None or v.owner == SESSION.get()],
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "dades.csv"
    if not name.lower().endswith((".csv", ".zip")):
        raise HTTPException(400, "Formats admesos: CSV o ZIP")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Arxiu massa gran (màxim 45 MB)")
    try:
        if name.lower().endswith(".zip"):
            df = _dataset_from_zip(contents)
        else:
            df = normalize_frame(pd.read_csv(io.BytesIO(contents)))
        if len(df) < 5:
            raise ValueError("Calen almenys 5 files")
        if len(df) > 600_000:
            raise ValueError("Màxim 600.000 files per dataset")
        if not _numeric_fields(df):
            raise ValueError("No s'han detectat columnes numèriques")
    except (ValueError, KeyError, zipfile.BadZipFile, pd.errors.ParserError) as exc:
        raise HTTPException(400, f"No es poden interpretar les dades: {exc}") from exc
    dsid = "up-" + uuid.uuid4().hex[:10]
    # Short-lived, in-memory upload cache. Do not write user files to disk.
    if len(DATASETS) > 20:
        old = next((k for k in DATASETS if k != "campus"), None)
        if old: DATASETS.pop(old)
    DATASETS[dsid] = Dataset(name, df, "Pujat en aquesta sessió (memòria)", SESSION.get())
    return _catalogue(dsid, DATASETS[dsid])


def _dataset_from_zip(contents: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(contents)) as z:
        candidates = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
        if len(candidates) > 20:
            raise ValueError("El ZIP té massa CSV")
        for n in candidates:
            if z.getinfo(n).file_size > 110_000_000:
                raise ValueError("CSV descomprimit massa gran")
        if sum(z.getinfo(n).file_size for n in candidates) > 170_000_000:
            raise ValueError("El ZIP descomprimit supera 170 MB")
        names = {Path(n).name.lower(): n for n in candidates}
        if {"contador.csv", "inversor.csv", "precio_pvpc.csv", "co2_red.csv"} <= set(names):
            def read_csv(key, columns):
                with z.open(names[key]) as f:
                    d = pd.read_csv(f, usecols=["Fecha", *columns], parse_dates=["Fecha"])
                return d.set_index("Fecha").sort_index()
            cont = read_csv("contador.csv", ["Contador - POTENCIA ACTIVA EXPORTADA (kW)"])
            inv_header = pd.read_csv(z.open(names["inversor.csv"]), nrows=0).columns
            inv_cols = [c for c in inv_header if re.search(r"Inversor [123] - POTENCIA ACTIVA \(kW\)$", c)]
            if len(inv_cols) != 3:
                raise ValueError("No s'han trobat els tres inversors")
            inv = read_csv("inversor.csv", inv_cols)
            price = read_csv("precio_pvpc.csv", ["precio_eur_per_kwh"])
            carbon = read_csv("co2_red.csv", ["co2_kg_per_kwh"])
            # Sensor samples HH:05..HH+1:00 correspond to the interval HH..HH+1.
            # Align on interval start (5-minute offset).
            cont.index -= pd.Timedelta(minutes=5)
            inv.index -= pd.Timedelta(minutes=5)
            load = (-cont.iloc[:, 0]).clip(lower=0, upper=5000).resample("1h").mean().rename("load_kw")
            solar = inv[inv_cols].clip(lower=0).sum(axis=1, min_count=1).resample("1h").mean().rename("pv_kw")
            merged = pd.concat([load, solar, price, carbon], axis=1).dropna(subset=["load_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"])
            merged["pv_kw"] = merged["pv_kw"].fillna(0)
            return normalize_frame(merged.reset_index())
        if not candidates:
            raise ValueError("No hi ha fitxers CSV al ZIP")
        # For a generic ZIP, use the largest CSV (usually the principal dataset).
        selected = max(candidates, key=lambda p: z.getinfo(p).file_size)
        with z.open(selected) as f:
            return normalize_frame(pd.read_csv(f))


def _filtered(df: pd.DataFrame, month: int | None = None, day_type: str = "all") -> pd.DataFrame:
    if month is not None and "month" in df:
        df = df[df["month"] == month]
    if day_type != "all" and "weekday" in df:
        df = df[df["weekday"] < 5] if day_type == "weekday" else df[df["weekday"] >= 5]
    return df


def _fig_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


def _style_fig(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "family": "Arial, sans-serif", "color": "#23313D"}},
        font={"family": "Arial, sans-serif", "color": "#687681", "size": 11},
        margin=dict(l=30, r=20, t=48 if title else 20, b=38),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"bgcolor": "#17232c", "font": {"color": "#ffffff"}},
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#E5E7E5", tickfont={"size": 11}),
        yaxis=dict(gridcolor="#edf0ed", zeroline=False, linecolor="#E5E7E5", tickfont={"size": 11}),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font={"size": 11}),
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _hour_profile(df: pd.DataFrame) -> list[dict]:
    if "hour" not in df:
        return []
    fields = [c for c in ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"] if c in df]
    g = df.groupby("hour")[fields].mean(numeric_only=True).reindex(range(24))
    return [{"hour": h, **{c: _num(g.loc[h,c], 5) for c in fields}} for h in range(24)]


@app.get("/api/overview")
def overview(dataset_id: str = "campus", month: int | None = None, day_type: str = "all"):
    ds = _get_dataset(dataset_id)
    df = _filtered(ds.frame, month, day_type)
    if df.empty:
        raise HTTPException(400, "El filtre no conté dades")
    profile = _hour_profile(df)
    numerics = _numeric_fields(df)
    robust = {}
    if "hour" in df:
        for m in ("load_kw", "pv_kw", "net_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"):
            if m in df:
                r = eda.robust_profile(df, m)
                robust[m] = {k: [_num(v, 5) for v in r[k]] for k in ("median", "q25", "q75")}
    key = {}
    for k, source, scale, decimals in [
        ("demand_mwh", "load_kw", 1000, 1), ("solar_mwh", "pv_kw", 1000, 1),
        ("emissions_t", "baseline_co2", 1000, 1), ("cost_eur", "baseline_cost", 1, 0),
        ("peak_kw", "load_kw", 1, 0),
    ]:
        key[k] = _num((df[source].max() if k == "peak_kw" else df[source].sum())/scale, decimals) if source in df else None
    corr = _num(df[["precio_eur_per_kwh", "co2_kg_per_kwh"]].corr().iloc[0, 1], 3) if all(c in df for c in ("precio_eur_per_kwh", "co2_kg_per_kwh")) else None
    return {
        "dataset": _catalogue(dataset_id, ds), "kpi": key, "corr": corr,
        "profile": profile, "robust": robust, "frozen_hours": eda.frozen_hours(ds.frame), "rows": len(df), "missing": [
            {"field": FIELD_NAMES.get(c,c), "missing_pct": _num(df[c].isna().mean()*100,1)}
            for c in sorted(numerics, key=lambda c: df[c].isna().mean(), reverse=True)[:12]
        ],
        "recommended": _recommend(df),
    }


def _recommend(df: pd.DataFrame) -> list[str]:
    needed = ["hour", "precio_eur_per_kwh", "co2_kg_per_kwh"]
    if not all(c in df for c in needed):
        return ["Explora la distribució i els valors absents abans de prendre decisions."]
    g = df.groupby("hour")[["precio_eur_per_kwh", "co2_kg_per_kwh"]].mean().dropna()
    if g.empty: return []
    p = int(g["precio_eur_per_kwh"].idxmin()); c = int(g["co2_kg_per_kwh"].idxmin())
    return [f"El preu mitjà mínim de la mostra és a les {p:02d}:00; el CO₂ mitjà mínim és a les {c:02d}:00.",
            "Són patrons històrics, no consignes. Cal decidir amb les previsions a 24 h i el SOC.",
            "El rendiment de cicle és del 90%: no tots els desplaçaments energètics compensen."]


class ChartReq(BaseModel):
    dataset_id: str = "campus"
    kind: str = "heatmap"
    metrics: list[str] = Field(default_factory=lambda: ["precio_eur_per_kwh", "co2_kg_per_kwh"])
    group: str = "hour"
    aggregation: str = "mean"
    month: int | None = None
    day_type: str = "all"


def make_charts(req: ChartReq) -> list[dict]:
    df = _filtered(_get_dataset(req.dataset_id).frame, req.month, req.day_type)
    fields = _numeric_fields(df)
    metrics = [m for m in req.metrics if m in fields][:3]
    if not metrics:
        raise HTTPException(400, "No hi ha mètriques numèriques vàlides")
    group = req.group if req.group in {"hour", "weekday", "month", "date", "row_id"} and req.group in df else ("hour" if "hour" in df else "row_id")
    agg = req.aggregation if req.aggregation in {"mean", "median", "sum", "min", "max"} else "mean"
    results = []
    if req.kind == "heatmap":
        if "hour" not in df or "weekday" not in df:
            raise HTTPException(400, "El heatmap hora × dia necessita una columna de data")
        for metric in metrics:
            m = df.pivot_table(index="weekday", columns="hour", values=metric, aggfunc=agg).reindex(index=range(7), columns=range(24))
            fig = go.Figure(go.Heatmap(
                z=m.values, x=[f"{i:02d}" for i in range(24)], y=DAYS,
                colorscale=[[0,"#f8f0e9"],[0.35,"#e8b49b"],[0.72,"#d77a50"],[1,"#743a26"]],
                colorbar=dict(title=UNITS.get(metric,""), thickness=11, len=0.8),
                hovertemplate="%{y}, %{x}:00<br><b>%{z:.3f}</b> "+UNITS.get(metric,"")+"<extra></extra>",
                xgap=2,ygap=2,
            ))
            fig.update_xaxes(title="Hora del dia", type="category", dtick=2)
            fig.update_yaxes(title="Dia de la setmana", autorange="reversed")
            frozen = eda.frozen_hours(df) if metric in {"load_kw", "net_kw", "baseline_cost", "baseline_co2"} else []
            note = f" · ⚠ {eda._hours_label(frozen)}: lectura congelada" if frozen else ""
            results.append({"title": FIELD_NAMES.get(metric,metric) + note, "unit": UNITS.get(metric,""), "figure": _fig_json(_style_fig(fig))})
    elif req.kind in {"line", "bar"}:
        if group == "date" and "Fecha" in df:
            daily = df.groupby("date")[metrics].agg(agg)
            if len(daily) > 200:
                # Aggregate to weeks instead of skipping rows (no aliasing of the series).
                weekly = df.set_index("Fecha")[metrics].resample("W-MON", label="left", closed="left").agg(agg if agg != "sum" else "sum")
                x = weekly.dropna(how="all"); x.index = x.index.strftime("%Y-%m-%d"); group_label = "Setmana"
            else:
                x = daily; group_label = "Data"
        else:
            x = df.groupby(group)[metrics].agg(agg).sort_index()
            group_label = {"hour": "Hora", "weekday": "Dia", "month": "Mes", "date": "Data"}.get(group, group)
        # Different physical units never share an axis: one panel per unit (small multiples).
        units = list(dict.fromkeys(UNITS.get(m, m) for m in metrics))
        fig = make_subplots(rows=len(units), cols=1, shared_xaxes=True, vertical_spacing=0.09) if len(units) > 1 else go.Figure()
        labels = [DAYS[int(v)] for v in x.index] if group == "weekday" else list(x.index)
        band = req.kind == "line" and group == "hour" and agg in {"mean", "median"}
        for i, m in enumerate(metrics):
            row = units.index(UNITS.get(m, m)) + 1
            pos = dict(row=row, col=1) if len(units) > 1 else {}
            if band:
                q = df.groupby("hour")[m].quantile([0.25, 0.75]).unstack().reindex(x.index)
                fig.add_trace(go.Scatter(x=labels, y=q[0.75], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), **pos)
                fig.add_trace(go.Scatter(x=labels, y=q[0.25], mode="lines", line=dict(width=0), fill="tonexty",
                                         fillcolor=_rgba(PALETTE[i], .14), showlegend=False, hoverinfo="skip"), **pos)
            trace = (go.Bar(x=labels, y=x[m], name=FIELD_NAMES.get(m, m), marker_color=PALETTE[i]) if req.kind == "bar" else
                     go.Scatter(x=labels, y=x[m], mode="lines+markers", name=FIELD_NAMES.get(m, m) + (" (banda: quartils 25–75%)" if band else ""),
                                line=dict(color=PALETTE[i], width=2.5), marker=dict(size=4)))
            fig.add_trace(trace, **pos)
            fig.update_yaxes(title_text=UNITS.get(m, ""), **pos)
        fig.update_xaxes(title=group_label, row=len(units), col=1) if len(units) > 1 else fig.update_xaxes(title=group_label)
        if len(units) > 1:
            fig.update_layout(height=230 * len(units) + 60)
        results.append({"title": " · ".join(FIELD_NAMES.get(m, m) for m in metrics), "unit": "", "figure": _fig_json(_style_fig(fig))})
    elif req.kind == "scatter":
        if len(metrics) < 2: raise HTTPException(400, "Scatter necessita dues mètriques")
        cols = metrics[:2] + (["hour"] if "hour" in df else [])
        vals = df[cols].dropna()
        if len(vals) > 4500: vals = vals.sample(4500, random_state=42)
        marker = dict(size=6, opacity=.5, color=vals["hour"], colorscale="Twilight", colorbar=dict(title="Hora", thickness=10)) if "hour" in vals else dict(size=6, color="#E77741", opacity=.4)
        fig = go.Figure(go.Scattergl(x=vals[metrics[0]], y=vals[metrics[1]], mode="markers", marker=marker,
                                     hovertemplate="%{x:.4g} · %{y:.4g}<extra></extra>"))
        r = vals[metrics[0]].corr(vals[metrics[1]])
        fig.update_xaxes(title=FIELD_NAMES.get(metrics[0], metrics[0]) + " (" + UNITS.get(metrics[0], "") + ")")
        fig.update_yaxes(title=FIELD_NAMES.get(metrics[1], metrics[1]) + " (" + UNITS.get(metrics[1], "") + ")")
        results.append({"title": f"Relació entre variables · r = {r:.3f} (color = hora)", "unit": "", "figure": _fig_json(_style_fig(fig))})
    elif req.kind in {"histogram", "box"}:
        units = list(dict.fromkeys(UNITS.get(m, m) for m in metrics))
        many = len(units) > 1
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=[FIELD_NAMES.get(m, m) for m in metrics]) if many else go.Figure()
        for i, m in enumerate(metrics):
            vals = df[m].dropna()
            pos = dict(row=1, col=i + 1) if many else {}
            if req.kind == "histogram":
                fig.add_trace(go.Histogram(x=vals, name=FIELD_NAMES.get(m, m), marker_color=PALETTE[i], opacity=.7, nbinsx=35), **pos)
                fig.update_xaxes(title_text=UNITS.get(m, ""), **pos)
            else:
                fig.add_trace(go.Box(y=vals, name=FIELD_NAMES.get(m, m), marker_color=PALETTE[i], boxpoints="outliers", boxmean=True), **pos)
                fig.update_yaxes(title_text=UNITS.get(m, ""), **pos)
        fig.update_layout(barmode="overlay", showlegend=not many)
        title = "Distribució de les variables" + (" · un panell per unitat" if many else "")
        results.append({"title": title, "unit": "", "figure": _fig_json(_style_fig(fig))})
    else:
        raise HTTPException(400, "Tipus de gràfic no compatible")
    return results


@app.post("/api/chart")
def chart(req: ChartReq):
    return {"charts":make_charts(req)}


class AssistantReq(BaseModel):
    dataset_id: str = "campus"
    message: str = Field(min_length=2,max_length=900)
    # Only a small, client-provided analytical context is accepted. This is
    # intentionally stateless: no user message is saved by the server.
    previous_query: str | None = Field(default=None, max_length=900)
    previous_plan: dict[str, Any] | None = None
    # Last turns (question + answer text) for the analyst; never stored server-side.
    history: list[dict[str, str]] | None = Field(default=None, max_length=6)


def _local_plan(query: str, metrics: list[str], has_time: bool) -> dict:
    s = query.casefold()
    kind = "heatmap" if any(t in s for t in ("heatmap", "mapa de calor", "mapa calor")) else (
        "scatter" if any(t in s for t in ("correlac", "relaci", "dispersi", "scatter")) else (
        "box" if "boxplot" in s or "caixa" in s else (
        "histogram" if "histogra" in s or "distribuci" in s else "line")))
    keys = [
        ("precio_eur_per_kwh", ["preu", "precio", "price", "cost per kwh", "pvpc"]),
        ("co2_kg_per_kwh", ["co2", "carbon", "emissi", "emisión", "emission"]),
        ("load_kw", ["consum", "demand", "càrrega del campus", "carga del campus"]),
        ("pv_kw", ["solar", "fotovolta", "inversor"]),
        ("temp_c", ["temperatura", "temperature"]),
        ("irr_wm2", ["irradi", "radiació"]),
    ]
    matched = [key for key, words in keys if key in metrics and any(w in s for w in words)]
    if not matched:
        matched = [key for key in ("precio_eur_per_kwh", "co2_kg_per_kwh") if key in metrics] or metrics[:1]
    if kind=="scatter" and len(matched)<2: matched=(matched+metrics)[:2]
    if not has_time and kind=="heatmap": kind="histogram"
    group = "weekday" if any(t in s for t in ("dia de la setmana", "día de la semana", "weekday", "setmana")) and kind!="heatmap" else ("month" if any(t in s for t in ("mensual", "mesos", "meses")) else "hour")
    return {"kind":kind, "metrics":matched[:3], "group":group, "aggregation":"mean"}


def _query_requirements(query: str, fields: list[str]) -> tuple[list[str], bool]:
    """Extract explicit user-requested metrics and heatmap intent, independently of the LLM.

    The model may legally output a syntactically valid one-metric plan despite a
    two-metric prompt. We must not silently drop the user's other metric.
    This intentionally handles *only explicit* metric names/synonyms.
    """
    q = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", query).casefold())
    q = "".join(c for c in q if not unicodedata.combining(c))
    aliases = (
        ("precio_eur_per_kwh", r"\b(?:preu(?:s)?|precio(?:s)?|price(?:s)?|pvpc)\b"),
        ("co2_kg_per_kwh", r"\b(?:co2|carbon|emissi\w*|emision\w*|emission\w*)\b"),
        ("load_kw", r"\b(?:consum\w*|demanda|demand|load)\b"),
        ("pv_kw", r"\b(?:solar|fotovoltaic\w*|pv)\b"),
        ("temp_c", r"\b(?:temperatura|temperature)\b"),
        ("irr_wm2", r"\b(?:irradi\w*|radiacio)\b"),
    )
    explicit = []
    for key, pattern in aliases:
        if key in fields and re.search(pattern, q):
            explicit.append(key)
    heatmap = bool(re.search(r"\b(?:heatmap\w*|mapa(?:s)? de calor)\b", q))
    return explicit, heatmap


def _complete_user_intent(plan: dict, query: str, fields: list[str], has_time: bool) -> tuple[dict, list[str]]:
    """Enforce requested metrics on a *validated* LLM plan, without inventing data.

    Numeric calculations still run locally. This is schema/intent reconciliation,
    not a replacement for model planning or a hidden fallback.
    """
    explicit, wants_heatmap = _query_requirements(query, fields)
    adjusted = dict(plan)
    changes = []
    selected = list(adjusted.get("metrics", []))
    missing = [key for key in explicit if key not in selected]
    if missing:
        # Explicitly requested metrics take priority over unrelated model extras.
        adjusted["metrics"] = (explicit + [key for key in selected if key not in explicit])[:3]
        changes.append("S'han afegit les variables demanades explícitament que faltaven al pla del model.")
    if wants_heatmap and has_time and adjusted.get("kind") != "heatmap":
        adjusted["kind"] = "heatmap"
        changes.append("S'ha respectat el tipus de gràfic heatmap demanat explícitament.")
    if adjusted.get("kind") == "heatmap" and adjusted.get("group") != "weekday":
        adjusted["group"] = "weekday"
        # Heatmaps in this implementation always plot weekday x hour.
    return adjusted, changes


def _extract_json_object(content: Any) -> dict | None:
    """Parse OpenAI-compatible content, including fenced or prefixed JSON.

    Do not parse ``reasoning_content``: it is not the model's final answer.
    Use a JSON decoder rather than a greedy ``{.*}`` regexp so multiple
    objects or markdown do not corrupt the extraction.
    """
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "\n".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content if isinstance(part, (str, dict))
        )
    if not isinstance(content, str):
        return None
    content = content.strip()
    if not content:
        return None
    try:
        result = json.loads(content)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            result, _ = decoder.raw_decode(content[match.start():])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    return None


def _normalize_llm_plan(obj: dict, fields: list[str]) -> tuple[dict, list[str]]:
    """Convert frequent OpenAI-compatible model schema variations to our canonical plan.

    Only known chart types, aggregations, groups and dataset fields are permitted.
    Unsupported model-generated columns are never passed to pandas or Plotly.
    """
    if not isinstance(obj, dict):
        return {}, ["el resultat no és un objecte JSON"]
    if isinstance(obj.get("plan"), dict):
        obj = obj["plan"]

    # LLMs often return one heatmap per variable, not our single combined plan.
    children = next((obj.get(k) for k in ("charts", "visualizations", "plots")
                     if isinstance(obj.get(k), list) and obj.get(k)), None)
    if children is not None:
        plans = [item for item in children if isinstance(item, dict)]
        if not plans:
            return {}, ["llista de gràfics buida"]
        obj = dict(plans[0])
        selected = []
        for item in plans:
            specs = item.get("metrics", item.get("variables", item.get("metric", item.get("y", []))))
            if isinstance(specs, (str, dict)):
                specs = [specs]
            if isinstance(specs, list):
                selected.extend(specs)
        if selected:
            obj["metrics"] = selected

    chart_types = {
        "heatmap":"heatmap", "heat_map":"heatmap", "mapa_de_calor":"heatmap", "heat map":"heatmap",
        "line":"line", "line_chart":"line", "line plot":"line", "timeseries":"line", "time_series":"line", "serie_temporal":"line",
        "bar":"bar", "bar_chart":"bar", "bars":"bar", "barplot":"bar", "barras":"bar",
        "scatter":"scatter", "scatter_plot":"scatter", "scatterplot":"scatter", "dispersio":"scatter", "dispersion":"scatter",
        "hist":"histogram", "histogram":"histogram", "histograma":"histogram",
        "box":"box", "boxplot":"box", "box_plot":"box", "caixa":"box"
    }
    groups = {
        "hour":"hour", "hora":"hour", "hours":"hour", "hour_of_day":"hour", "hora_del_dia":"hour",
        "weekday":"weekday", "day_of_week":"weekday", "dia_setmana":"weekday", "dia_de_la_setmana":"weekday", "day of week":"weekday",
        "month":"month", "mes":"month", "months":"month", "month_of_year":"month",
        "date":"date", "fecha":"date", "data":"date", "day":"date", "row_id":"row_id", "index":"row_id"
    }
    aggregations = {
        "mean":"mean", "average":"mean", "avg":"mean", "mitjana":"mean", "promedio":"mean",
        "median":"median", "mediana":"median", "sum":"sum", "total":"sum", "suma":"sum",
        "min":"min", "minimum":"min", "max":"max", "maximum":"max"
    }
    def pick(*names):
        return next((obj[n] for n in names if n in obj and obj[n] is not None), None)
    def word(value):
        return str(value).casefold().strip().replace("-", "_") if isinstance(value, str) else ""

    kind_val = pick("kind", "chart_type", "chartType", "plot_type", "type", "visualization_type")
    kind = chart_types.get(word(kind_val), word(kind_val))
    group_val = pick("group", "group_by", "groupBy", "x_axis", "x", "dimension") or "hour"
    if isinstance(group_val, list):
        group_val = "weekday" if any(word(x) in {"weekday","day_of_week"} for x in group_val) else group_val[0] if group_val else "hour"
    group = groups.get(word(group_val), word(group_val))
    agg_val = pick("aggregation", "agg", "aggregate", "statistic", "operation") or "mean"
    aggregation = aggregations.get(word(agg_val), word(agg_val))

    raw = pick("metrics", "metric", "variables", "variable", "fields", "field", "columns", "values", "value", "y_axis", "y")
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []

    # Labels and conventional short names are accepted only when unambiguous.
    known_aliases = {
        "price": "precio_eur_per_kwh", "preu":"precio_eur_per_kwh", "precio":"precio_eur_per_kwh",
        "pvpc":"precio_eur_per_kwh", "electricity_price":"precio_eur_per_kwh",
        "co2":"co2_kg_per_kwh", "carbon_intensity":"co2_kg_per_kwh",
        "emissions_intensity":"co2_kg_per_kwh", "emissions_factor":"co2_kg_per_kwh",
        "load":"load_kw", "consum":"load_kw", "consumption":"load_kw", "demand":"load_kw",
        "solar":"pv_kw", "solar_generation":"pv_kw", "pv":"pv_kw",
        "temperature":"temp_c", "irradiance":"irr_wm2"
    }
    lookup = {field.casefold(): field for field in fields}
    for field in fields:
        label = FIELD_NAMES.get(field)
        if label and label.casefold() not in lookup:
            lookup[label.casefold()] = field
    for alias, value in known_aliases.items():
        if value in fields:
            lookup[alias] = value

    clean = []
    unknown = []
    for item in raw:
        if isinstance(item, dict):
            item = next((item[k] for k in ("field", "key", "name", "column", "metric", "value") if isinstance(item.get(k), str)), None)
        if not isinstance(item, str):
            unknown.append(str(item)[:60]); continue
        candidate = lookup.get(item.casefold().strip())
        if candidate:
            if candidate not in clean: clean.append(candidate)
        else:
            unknown.append(item[:60])
    return {"kind":kind,"metrics":clean,"group":group,"aggregation":aggregation}, unknown


def _validate_llm_plan(obj: dict, fields: list[str], has_time: bool) -> tuple[dict | None, str | None]:
    """Normalize and strictly validate, with intelligible error messages."""
    kinds = {"heatmap", "line", "bar", "scatter", "histogram", "box"}
    groups = {"hour", "weekday", "month", "date", "row_id"}
    aggregations = {"mean", "median", "sum", "min", "max"}
    plan, unknown = _normalize_llm_plan(obj, fields)
    kind, group, aggregation, metrics = (plan.get(k) for k in ("kind","group","aggregation","metrics"))
    problems = []
    if kind not in kinds: problems.append("gràfic=" + str(kind)[:40])
    if group not in groups: problems.append("agrupació=" + str(group)[:40])
    if aggregation not in aggregations: problems.append("agregació=" + str(aggregation)[:40])
    if unknown: problems.append("variables desconegudes=" + ", ".join(unknown[:4]))
    if not isinstance(metrics,list) or not metrics: problems.append("cap variable numèrica vàlida")
    if problems:
        return None, "Helmcode ha retornat un pla de visualització amb camps no admesos: " + "; ".join(problems)
    if kind == "heatmap" and not has_time:
        return None, "No es pot fer un heatmap hora × setmana sense marques temporals."
    if kind == "scatter" and len(metrics) < 2:
        return None, "El scatter requereix dues variables numèriques."
    return {"kind":kind, "metrics":metrics[:3], "group":group, "aggregation":aggregation}, None


async def _llm_plan(query: str, fields: list[str], has_time: bool) -> tuple[dict | None, str | None]:
    """Request machine-readable chart specifications from Helmcode.

    Qwen 3.6 reasons by default. With a tiny max_tokens budget it can return
    *only* reasoning_content and an empty final content. Disable reasoning for
    models that support it, and request JSON mode to constrain the response.
    """
    key = os.getenv("HELMCODE_API_KEY")
    model = os.getenv("HELMCODE_MODEL")
    if not key or not model:
        return None, None
    endpoint = os.getenv("HELMCODE_BASE_URL", "https://api.helmcode.com/v1/chat/completions")
    system = ("You create safe data visualization configurations for an energy dashboard. "
              "Respond with exactly one JSON object, no markdown or explanations. "
              "Required JSON keys: kind, metrics, group, aggregation. Example: {\"kind\":\"heatmap\",\"metrics\":[\"precio_eur_per_kwh\",\"co2_kg_per_kwh\"],\"group\":\"weekday\",\"aggregation\":\"mean\"}. "
              "kind: heatmap, line, bar, scatter, histogram, box. "
              "group: hour, weekday, month, date, row_id. "
              "aggregation: mean, median, sum, min, max. "
              "metrics must be a nonempty array of EXACT numeric field names from the supplied list. "
              "Heatmaps are weekday x hour. Select both price and CO2 metrics if requested. "
              "Never invent columns. The server will calculate the plots and insights, not you.")
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps({"query": query, "fields": fields,
                                                                 "has_time": has_time}, ensure_ascii=False)}],
            "temperature": 0.0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"}}
    model_id = model.casefold().strip()
    if model_id.startswith(("qwen3.6", "gemma4", "deepseek-v4-flash")):
        body["reasoning_effort"] = "none"
    elif model_id.startswith(("glm5.3", "glm5.3-flash")):
        body["reasoning_effort"] = "low"
        body["max_tokens"] = 4096  # low reasoning must fit alongside the JSON answer
    else:
        body["max_tokens"] = 4096
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            res = await client.post(endpoint,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=body)
            # Some compatible backends may reject JSON mode. One backward-compatible
            # retry is safe, while all other errors should remain visible.
            if res.status_code == 400 and "response_format" in res.text.lower():
                body.pop("response_format", None)
                res = await client.post(endpoint,
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                    json=body)
            if res.status_code in (401, 403):
                return None, f"Credencial API rebutjada (HTTP {res.status_code})."
            if res.status_code == 404:
                return None, "Model o endpoint no trobat (HTTP 404). Comprova HELMCODE_MODEL."
            if res.status_code == 429:
                return None, "Quota o límit de peticions de Helmcode (HTTP 429)."
            if res.status_code == 402:
                return None, "Helmcode ha indicat saldo o crèdit insuficient (HTTP 402)."
            res.raise_for_status()
            payload = res.json()
            choice = payload["choices"][0]
            message = choice["message"]
            obj = _extract_json_object(message.get("content"))
            if obj is None:
                reason = choice.get("finish_reason")
                has_thinking = bool(message.get("reasoning_content"))
                print("[UPC Energy] Helmcode empty/invalid JSON:",
                      "model=", model, "finish_reason=", reason,
                      "has_reasoning=", has_thinking,
                      "content_length=", len(str(message.get("content") or "")))
                if reason == "length":
                    return None, "Helmcode ha esgotat el límit de tokens abans d'acabar el JSON."
                if has_thinking and not message.get("content"):
                    return None, "Helmcode ha retornat raonament però no una resposta final."
                return None, "El model no ha retornat una configuració JSON vàlida."
            plan, error = _validate_llm_plan(obj, fields, has_time)
            if plan is not None:
                return plan, None
            # Recover once from a semantic schema mismatch (not from an API failure).
            # The second model call cannot execute code or introduce new data fields.
            print("[UPC Energy] Helmcode first plan rejected:", error)
            repair_body = dict(body)
            repair_body["messages"] = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({
                    "task": "REPAIR this chart JSON to exactly kind, metrics, group, aggregation. "
                            "Use only the allowed values and numeric field names. "
                            "Keep the user's requested visualizations. No prose.",
                    "original_query": query,
                    "rejected_plan": obj,
                    "validation_error": error,
                    "allowed_numeric_fields": fields,
                    "has_time": has_time,
                }, ensure_ascii=False)},
            ]
            repair = await client.post(endpoint,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=repair_body)
            if repair.status_code == 400 and "response_format" in repair.text.lower():
                repair_body.pop("response_format", None)
                repair = await client.post(endpoint,
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                    json=repair_body)
            if repair.is_success:
                repair_payload = repair.json()
                repaired = _extract_json_object(repair_payload["choices"][0]["message"].get("content"))
                if repaired is not None:
                    corrected, repair_error = _validate_llm_plan(repaired, fields, has_time)
                    if corrected:
                        return corrected, None
                    print("[UPC Energy] Helmcode repair rejected:", repair_error)
            # Preserve the first specific error if repair fails, rather than claiming
            # that Helmcode is offline or silently executing an unsafe plan.
            return None, error + " (S'ha intentat reparar el format automàticament.)"
    except httpx.TimeoutException:
        return None, "Temps d'espera de Helmcode exhaurit (35 s)."
    except httpx.HTTPStatusError as exc:
        print("[UPC Energy] Helmcode HTTP failure:", exc.response.status_code)
        return None, f"Error del proveïdor (HTTP {exc.response.status_code})."
    except (httpx.RequestError, ValueError, KeyError, IndexError, TypeError) as exc:
        print("[UPC Energy] Helmcode planning failure:", type(exc).__name__)
        return None, "Error de connexió o resposta no vàlida de Helmcode."


def _grounded_insights(df:pd.DataFrame, metrics:list[str], req:ChartReq) -> tuple[list[str],dict]:
    insights=[]; evidence={"rows":len(df),"aggregation":req.aggregation,"selection":req.day_type}
    for m in metrics:
        values=pd.to_numeric(df[m],errors="coerce").dropna()
        if values.empty: continue
        e={"mean":_num(values.mean(),5),"min":_num(values.min(),5),"max":_num(values.max(),5),"n":int(len(values))}
        if "hour" in df:
            grouped=df.groupby("hour")[m].mean().dropna()
            if len(grouped):
                e["lowest_average_hour"]=int(grouped.idxmin())
                e["highest_average_hour"]=int(grouped.idxmax())
                insights.append(f"{FIELD_NAMES.get(m,m)}: mitjana {e['mean']:g} {UNITS.get(m,'')}. Mínim horari mitjà a les {e['lowest_average_hour']:02d}:00 i màxim a les {e['highest_average_hour']:02d}:00.")
        else:
            insights.append(f"{FIELD_NAMES.get(m,m)}: mitjana {e['mean']:g}, entre {e['min']:g} i {e['max']:g} ({e['n']} registres).")
        evidence[m]=e
    if all(m in metrics for m in ("precio_eur_per_kwh","co2_kg_per_kwh")):
        c=_num(df[metrics[:2]].corr().iloc[0,1],3)
        if c is not None: insights.append(f"Correlació entre les dues variables seleccionades: {c:+.3f}. No implica causalitat.")
    if "precio_eur_per_kwh" in metrics or "co2_kg_per_kwh" in metrics:
        insights.append("Decisió: utilitza aquests patrons com a diagnòstic; per a consignes reals, compara les 24 hores futures i comprova l'energia de la bateria i el 90% de rendiment.")
    return insights,evidence


FOLLOWUP_RE = (r"\b(ara|nomes|sols|mateix|mateixa|mateixos|filtra|separa|compara|despres|pero|aixo|ho|igual|abans|anterior|"
               r"canvia|enlloc|en lloc|tambe|i si)\b")


def is_followup(plain: str, earlier: dict | None, explicit: list[str]) -> bool:
    """A follow-up must say so ('ara…', '…-ho', 'el mateix…'); short new questions are new questions."""
    return bool(earlier and not explicit and re.search(FOLLOWUP_RE, plain))


def _continuation(query: str, previous_query: str | None, previous_plan: dict | None, available: list[str], has_time: bool) -> tuple[str, dict | None, str, bool]:
    """Resolve short references and filters without trusting previous plans.

    The client context is advisory and cannot grant access to a new metric.
    Specific new metric requests take precedence over the earlier turn.
    """
    plain = unicodedata.normalize("NFKD", query.casefold())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    day_type = "all"
    if re.search(r"cap[s]? de setmana|weekend|fin(?:es)? de semana", plain):
        day_type = "weekend"
    elif re.search(r"laborable|feiner|weekday|entre setmana", plain):
        day_type = "weekday"
    comparison = bool(re.search(r"compar|diferenc|versus| vs\b", plain)) and bool(re.search(r"setmana|laborable|feiner|weekend", plain))
    earlier = previous_plan if isinstance(previous_plan, dict) else None
    explicit, explicit_heatmap = _query_requirements(query, available)
    # Direct questions about metrics are new questions. Follow-ups like
    # 'ara només caps de setmana' must reuse the earlier chart type/metrics.
    followup = is_followup(plain, earlier, explicit)
    if earlier and followup:
        kind = earlier.get("kind") if earlier.get("kind") in {"heatmap","line","bar","scatter","histogram","box"} else "line"
        metrics = [m for m in earlier.get("metrics", []) if isinstance(m,str) and m in available][:3]
        group = earlier.get("group") if earlier.get("group") in {"hour","weekday","month","date"} else "hour"
        aggregation = earlier.get("aggregation") if earlier.get("aggregation") in {"mean","median","sum","min","max"} else "mean"
        # A follow-up may change the chart form or aggregation without
        # restating any columns. Explicit refinements win over inherited form.
        if re.search(r"\b(?:barres|barras|barchart|bar chart)\b",plain): kind="bar"
        elif re.search(r"\b(?:linies|lineas|line chart|serie temporal|serie de temps)\b",plain): kind="line"
        elif re.search(r"\b(?:heatmap|mapa de calor)\b",plain) and has_time: kind="heatmap"
        elif re.search(r"\b(?:boxplot|diagrama de caixa)\b",plain): kind="box"
        elif re.search(r"\b(?:histograma|histogram)\b",plain): kind="histogram"
        if re.search(r"\b(?:mediana|median)\b",plain): aggregation="median"
        elif re.search(r"\b(?:suma|sumatori|sum)\b",plain): aggregation="sum"
        elif re.search(r"\b(?:maxim|maximo|maximum)\b",plain): aggregation="max"
        elif re.search(r"\b(?:minim|minimo|minimum)\b",plain): aggregation="min"
        if kind in {"line","bar"}:
            if re.search(r"\b(?:per mes|cada mes|mensual|mesos)\b",plain): group="month"
            elif re.search(r"\b(?:per hora|cada hora|horari)\b",plain): group="hour"
        if metrics:
            return query, {"kind":kind,"metrics":metrics,"group":group,"aggregation":aggregation}, day_type, comparison
    return query, None, day_type, comparison


def _compare_daytypes(dataset_id: str, plan: dict) -> tuple[list[dict], list[str], dict]:
    """Weekday vs weekend on shared axes, with day-level bootstrap 95% intervals.

    Always compares hourly *means*: sums would mostly reflect that there are 2,5×
    more working days than weekend days.
    """
    df = _get_dataset(dataset_id).frame
    if "weekday" not in df or "hour" not in df:
        raise HTTPException(400,"La comparació horària necessita una data reconeguda")
    weekday=df[df["weekday"]<5]
    weekend=df[df["weekday"]>=5]
    charts: list[dict] = []
    summaries: list[str] = []
    evidence: dict = {"comparison":"laborables vs cap de setmana", "statistic": "mitjana horària amb IC 95% bootstrap per dies",
                      "groups":{"weekday":{"rows":len(weekday)},"weekend":{"rows":len(weekend)}}}
    if plan.get("aggregation") not in (None, "mean"):
        summaries.append(f"S'ha comparat la mitjana (no «{plan.get('aggregation')}») perquè els dos grups tenen un nombre de dies diferent.")
    for metric in plan["metrics"]:
        if metric not in df: continue
        r = eda.bootstrap_diff(weekday, weekend, metric)
        if not r: continue
        fig=go.Figure()
        for key,name,color in (("a","Laborables","#527b91"),("b","Caps de setmana","#d67b4e")):
            mean = r[f"mean_{key}"]
            fig.add_scatter(x=list(range(24)),y=[_num(v,5) for v in mean],mode="lines+markers",name=name,
                            line=dict(width=2.6,color=color),marker=dict(size=4),
                            error_y=dict(type="data",symmetric=False,array=[_num(v,5) for v in (r[f"{key}_hi"]-mean)],
                                         arrayminus=[_num(v,5) for v in (mean-r[f"{key}_lo"])],thickness=1,width=2,color=color),
                            hovertemplate="%{x}:00 · %{y:.4g} "+UNITS.get(metric,"")+"<extra>"+name+"</extra>")
        fig.update_xaxes(title="Hora del dia",dtick=2)
        fig.update_yaxes(title=UNITS.get(metric,""))
        charts.append({"title":FIELD_NAMES.get(metric,metric)+" · laborables vs cap de setmana (IC 95%)",
                       "unit":UNITS.get(metric,""),"figure":_fig_json(_style_fig(fig))})
        a=float(np.nanmean(r["mean_a"])); b=float(np.nanmean(r["mean_b"]))
        summaries.append(f"{FIELD_NAMES.get(metric,metric)}: laborables {eda.ca(a,3 if abs(a)<10 else 1)} {UNITS.get(metric,'')}, caps de setmana "
                         f"{eda.ca(b,3 if abs(b)<10 else 1)} {UNITS.get(metric,'')}; diferència significativa en {len(r['significant_hours'])} de 24 hores.")
        evidence[metric]={"weekday_mean":_num(a,5),"weekend_mean":_num(b,5),"significant_hours":r["significant_hours"],
                          "days":{"weekday":r["days_a"],"weekend":r["days_b"]}}
    summaries.append("Comparació descriptiva de períodes històrics; no prova causalitat ni determina per si sola una consigna de bateria.")
    return charts[:3], summaries, evidence


@app.post("/api/assistant")
async def assistant(req: AssistantReq):
    """Route a question: physics what-if, multi-step analyst with tools, or chart planner.

    Chart requests and explicit follow-ups ("ara només…", "compara-ho…") keep the
    verified chart-plan path; analytical and decision questions go to the
    tool-using analyst (Helmcode tool calling, JSON-plan fallback, or local).
    """
    ds=_get_dataset(req.dataset_id)
    available=_numeric_fields(ds.frame)
    plain=unicodedata.normalize("NFKD", req.message.casefold())
    plain="".join(c for c in plain if not unicodedata.combining(c))
    explicit,_=_query_requirements(req.message, available)
    followup=is_followup(plain, req.previous_plan if isinstance(req.previous_plan, dict) else None, explicit)
    intent="simulate_day" if copilot.is_what_if(req.message) else (None if followup else copilot.classify(req.message))
    if intent:
        if intent == "simulate_day" and not _energy_ready(ds.frame):
            raise HTTPException(400, "Per fer un what-if calen dades de temps i energia")
        history=[t for t in (req.history or []) if isinstance(t, dict)][-3:]
        return await copilot.answer(sys.modules[__name__], req.dataset_id, req.message, history, intent)
    return await _chart_assistant(req)


async def _chart_assistant(req: "AssistantReq"):
    ds=_get_dataset(req.dataset_id)
    available=_numeric_fields(ds.frame)
    has_time="hour" in ds.frame
    resolved_query, continued_plan, day_type, comparison = _continuation(
        req.message, req.previous_query, req.previous_plan, available, has_time)
    if continued_plan:
        # Follow-up inherits the verified plan: no extra model round-trip.
        plan, llm_warning, mode = continued_plan, None, "context"
    else:
        plan, llm_warning=await _llm_plan(resolved_query,available,has_time)
        mode="helmcode" if plan else "local"
        if not plan: plan=_local_plan(resolved_query,available,has_time)
    plan, intent_adjustments = _complete_user_intent(plan, resolved_query, available, has_time)
    safe={"dataset_id":req.dataset_id,"kind":str(plan.get("kind","line")).lower(),
          "metrics":[str(m) for m in plan.get("metrics",[]) if m in available][:3],
          "group":str(plan.get("group","hour")).lower(),
          "aggregation":str(plan.get("aggregation","mean")).lower()}
    if safe["kind"] not in {"heatmap","line","bar","scatter","histogram","box"}: safe["kind"]="line"
    if safe["kind"]=="heatmap" and not has_time: safe["kind"]="histogram"
    if not safe["metrics"]: safe["metrics"]=_local_plan(req.message,available,has_time)["metrics"]
    if safe["kind"]=="scatter" and len(safe["metrics"])<2:
        safe["kind"]="line" if has_time else "histogram"
    if day_type != "all" and "weekday" not in ds.frame:
        raise HTTPException(400,"Aquest dataset no permet filtrar per tipus de dia")
    plan_req=ChartReq(**safe, day_type=day_type)
    if comparison and "weekday" in ds.frame:
        charts, facts, evidence = _compare_daytypes(req.dataset_id, safe)
    else:
        charts=make_charts(plan_req)
        filtered=_filtered(ds.frame,day_type=day_type)
        facts,evidence=_grounded_insights(filtered,safe["metrics"],plan_req)
        if day_type != "all":
            facts.insert(0,"Filtre aplicat: "+("caps de setmana" if day_type=="weekend" else "dies laborables")+f" ({len(filtered)} registres).")
    if intent_adjustments and mode == "helmcode":
        facts = intent_adjustments + facts
    return {"mode":mode,"plan":safe,"charts":charts,"insights":facts,"evidence":evidence,
            "llm_warning":llm_warning, "day_type":day_type, "comparison":comparison,
            "intent_adjustments":intent_adjustments,
            "note":"Resultats calculats al servidor amb les dades seleccionades. L'LLM interpreta el pla, no calcula els resultats."}


@app.get("/api/radar")
def opportunity_radar(dataset_id: str = "campus", month: int | None = None):
    """Exploratory historical opportunity profile, not an executable dispatch.

    Signal = 0.7·CO₂/mean + 0.3·price/mean (dimensionless). A candidate pair is
    a hypothesis to validate in the Lab. Uses the same complete days as the Lab,
    and reports the per-day spread distribution and the best pair per month.
    """
    df=_get_dataset(dataset_id).frame
    need=["hour","date","precio_eur_per_kwh","co2_kg_per_kwh"]
    if any(c not in df for c in need):
        raise HTTPException(400,"Aquest dataset no disposa de preus i emissions horaris")
    days=set(arena.complete_days(df)) if _energy_ready(df) else set(df.dropna(subset=need).groupby("date")["hour"].nunique().loc[lambda x: x==24].index)
    clean=df[df["date"].isin(days)].dropna(subset=need).copy()
    if month is not None:
        clean=clean[clean["month"]==month]
    if len(clean)<24:raise HTTPException(400,"Falten hores suficients per al radar")
    pmean=float(df["precio_eur_per_kwh"].mean()); cmean=float(df["co2_kg_per_kwh"].mean())
    if pmean<=0 or cmean<=0:raise HTTPException(400,"No es pot normalitzar aquest mercat")
    clean["signal"]=0.7*clean["co2_kg_per_kwh"]/cmean + 0.3*clean["precio_eur_per_kwh"]/pmean
    profile=clean.groupby("hour")["signal"].mean().reindex(range(24))
    if profile.isna().any():raise HTTPException(400,"Calen dades representatives de les 24 hores")
    signal=profile.to_numpy(float)
    best,c,d=max(((0.9*signal[d]-signal[c],c,d) for c in range(24) for d in range(c+1,24)),key=lambda x:x[0])
    pivot=clean.pivot_table(index="date",columns="hour",values="signal",aggfunc="mean")
    spread=(0.9*pivot[d]-pivot[c]).dropna()
    count=int(len(spread)); viable=int((spread>0).sum())
    by_month=eda.signal_by_month(df[df["date"].isin(days)])
    matrix=by_month["matrix"]
    return {"method":"Indicador exploratori 70% CO₂ normalitzat + 30% preu normalitzat",
            "efficiency":0.90, "month":month, "profile":[{"hour":i,"signal":_num(signal[i],4)} for i in range(24)],
            "candidate":{"charge_hour":int(c),"discharge_hour":int(d),"spread_index":_num(best,4),
                         "viable_day_pct":_num(100*viable/count,1) if count else None,
                         "days_examined":count,
                         "spread_quartiles":[_num(spread.quantile(q),3) for q in (.1,.25,.5,.75,.9)] if count else []},
            "day_spreads":[_num(v,3) for v in spread.tolist()],
            "by_month":by_month["best"],
            "matrix":{"months":[int(m) for m in matrix.index],"z":[[_num(v,3) for v in row] for row in matrix.to_numpy()]},
            "disclaimer":"Perfils mitjans de train, no previsió de les pròximes 24 h ni score oficial. La parella òptima canvia segons el mes "
                         "i el marge diari varia molt: valida-ho al Decision Lab o, millor, a l'Agent Arena."}


def default_day(days: list[str]) -> str:
    """Most recent complete working day: a realistic, reproducible default for demos."""
    for d in reversed(days):
        if pd.Timestamp(d).dayofweek < 5:
            return d
    return days[-1]


@app.get("/api/days")
def days(dataset_id: str = "campus"):
    df=_get_dataset(dataset_id).frame
    if not _energy_ready(df):
        return {"dates":[],"reason":"Es necessiten dades horàries del campus, solar, preus i CO₂"}
    ds=arena.complete_days(df)
    return {"dates":ds,"default":default_day(ds) if ds else None,"reason":None}


class SimReq(BaseModel):
    dataset_id: str = "campus"
    date: str
    # Windows are inclusive and may wrap around midnight (e.g. 22 → 3).
    charge_start: int | None = Field(default=None,ge=0,le=23)
    charge_end: int | None = Field(default=None,ge=0,le=23)
    discharge_start: int | None = Field(default=None,ge=0,le=23)
    discharge_end: int | None = Field(default=None,ge=0,le=23)
    # Backwards compatible single-hour fields (v2).
    charge_hour: int = Field(default=14,ge=0,le=23)
    discharge_hour: int = Field(default=20,ge=0,le=23)
    power_kw: float = Field(default=500,ge=0,le=500)


def run_day_simulation(df: pd.DataFrame, date: str, charge: tuple[int,int], discharge: tuple[int,int], power: float) -> dict:
    day=df[df["date"]==date].sort_values("Fecha").dropna(subset=["load_kw","pv_kw","precio_eur_per_kwh","co2_kg_per_kwh"])
    if len(day)!=24 or set(day["hour"])!=set(range(24)):
        raise HTTPException(400,"Selecciona un dia amb les 24 hores completes")
    try:
        r=arena.simulate_day(day, charge, discharge, power)
    except RuntimeError as exc:
        raise HTTPException(500,"No s'ha pogut resoldre l'optimització") from exc
    m,o=r["manual"],r["optimal"]
    capture=_num(100*m["score_raw"]/o["score_raw"],1) if o["score_raw"]>0 else None
    return {"date":date,"baseline":{"cost":_num(m["cost_baseline_eur"],2),"co2":_num(m["co2_baseline_kg"],2)},
            "manual":arena.public(m),"optimal":arena.public(o),"capture_pct":capture,
            "hours":list(range(24)),"price":day["precio_eur_per_kwh"].round(5).tolist(),
            "carbon":day["co2_kg_per_kwh"].round(5).tolist(),
            "weights":[_num(v,5) for v in r["weights"]],
            "manual_actions":np.around(r["manual_actions"],2).tolist(),"manual_soc":np.around(m["_soc"],2).tolist(),
            "optimal_actions":np.around(r["optimal_actions"],2).tolist(),"optimal_soc":np.around(o["_soc"],2).tolist(),
            "charge_hours":r["charge_hours"],"discharge_hours":r["discharge_hours"],"overlap_hours":r["overlap_hours"],
            "notice":"Avaluat amb la física i la fórmula de starter_kit/environment.py. Cicle tancat: la bateria no baixa del SOC inicial "
                     "(1.000 kWh) i l'òptim acaba com a mínim al mateix SOC, perquè la comparació diària sigui justa. "
                     "L'òptim coneix tota la demanda del dia: és una cota superior, no un agent desplegable."}


@app.post("/api/simulate")
def simulate(req:SimReq):
    df=_get_dataset(req.dataset_id).frame
    if "date" not in df or not _energy_ready(df):raise HTTPException(400,"Calen sèries temporals d'energia")
    charge=(req.charge_start if req.charge_start is not None else req.charge_hour,
            req.charge_end if req.charge_end is not None else (req.charge_start if req.charge_start is not None else req.charge_hour))
    discharge=(req.discharge_start if req.discharge_start is not None else req.discharge_hour,
               req.discharge_end if req.discharge_end is not None else (req.discharge_start if req.discharge_start is not None else req.discharge_hour))
    return run_day_simulation(df, req.date, charge, discharge, req.power_kw)


# ---------------------------------------------------------------------------
# Data quality, Agent Arena and strategic studies
# ---------------------------------------------------------------------------
@app.get("/api/quality")
def quality(dataset_id: str = "campus"):
    return eda.data_quality(_get_dataset(dataset_id).frame)


def _official_frame() -> pd.DataFrame:
    ds = DATASETS.get("campus")
    if ds is None or not _energy_ready(ds.frame):
        raise HTTPException(400, "L'Arena necessita les dades oficials del campus")
    return ds.frame


@app.get("/api/arena")
def arena_status():
    df = _official_frame()
    arena.start_background(df)
    snap = arena.ARENA.snapshot()
    return {**snap, "result": arena.ARENA.result if snap["status"] == "ready" else None}


@app.post("/api/arena/rerun")
def arena_rerun():
    """Re-run after editing the agent file (results are cached per agent mtime)."""
    df = _official_frame()
    arena.reset_arena()
    arena.start_background(df)
    return arena.ARENA.snapshot()


@app.get("/api/studies")
def studies_status():
    df = _official_frame()
    arena.start_background(df)
    snap = arena.STUDIES.snapshot()
    return {**snap, "result": arena.STUDIES.result if snap["status"] == "ready" else None}
