"""
dashboard/app.py
Visualizador de datos estilo Power BI para el reto UPC Sostenible (Datathon 2026).

Ejecutar desde la raíz del repo:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim  # noqa: E402
from data import DIAS, MESES, load_hourly, sim_frame  # noqa: E402

# ----------------------------------------------------------------------------
# Paleta (validada para daltonismo; ver skill dataviz) y estilo
# ----------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
GRAY, GRAY_LIGHT = "#8a8984", "#d9d8d4"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
SEQ_BLUE = [[0.0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"], [0.5, "#3987e5"],
            [0.67, "#256abf"], [0.83, "#184f95"], [1.0, "#0d366b"]]

# Colores por entidad (fijos en todo el dashboard)
C_CONSUMO, C_SOLAR, C_PRECIO, C_CO2 = BLUE, ORANGE, AQUA, VIOLET
C_LAB, C_FINDE = BLUE, ORANGE
C_AGENTE, C_OPTIMO = BLUE, ORANGE

st.set_page_config(page_title="Campus Nord · Explorador de datos", page_icon="🔋", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #f3f2f1; }
      .block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1500px; }
      .pbi-header { background: linear-gradient(90deg, #06142b 0%, #0b2a52 70%, #0f4a5a 100%);
                    color: #fff; border-radius: 10px; padding: 18px 24px; margin-bottom: 14px; }
      .pbi-header h1 { color: #fff; font-size: 1.55rem; margin: 0; padding: 0; font-weight: 700; }
      .pbi-header p { color: #c9d6e8; margin: 4px 0 0 0; font-size: .92rem; }
      .pbi-header b { color: #4fd1a1; }
      .kpi { background: #fff; border-radius: 8px; padding: 12px 14px 10px 14px; height: 100%;
             box-shadow: 0 1px 2px rgba(0,0,0,.08); border-top: 3px solid var(--acc, #2a78d6); }
      .kpi .lbl { color: #52514e; font-size: .78rem; text-transform: uppercase; letter-spacing: .03em; }
      .kpi .val { color: #0b0b0b; font-size: 1.55rem; font-weight: 700; line-height: 1.25;
                  font-variant-numeric: tabular-nums; }
      .kpi .sub { color: #8a8984; font-size: .78rem; }
      .card-title { font-weight: 650; color: #0b0b0b; font-size: 1.0rem; margin: 2px 0 0 0; }
      .card-cap { color: #52514e; font-size: .83rem; margin: 0 0 4px 0; }
      .insight { background: #fff; border-left: 4px solid #1baf7a; border-radius: 6px;
                 padding: 10px 14px; margin-bottom: 8px; font-size: .9rem; color: #0b0b0b; }
      .warnbox { background: #fff; border-left: 4px solid #fab219; border-radius: 6px;
                 padding: 10px 14px; margin-bottom: 8px; font-size: .9rem; color: #0b0b0b; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 8px; }
      section[data-testid="stSidebar"] { background: #ffffff; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] { background: #e6e5e1; border-radius: 6px 6px 0 0; padding: 6px 14px; }
      .stTabs [aria-selected="true"] { background: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Segoe UI, Inter, system-ui, sans-serif", size=12, color=INK_2),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None,
                    font=dict(color=INK_2)) if legend else None,
        showlegend=legend,
        hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor=GRID),
        bargap=0.15,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def plot(fig: go.Figure):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def card(title: str, caption: str | None = None):
    box = st.container(border=True)
    box.markdown(f"<p class='card-title'>{title}</p>", unsafe_allow_html=True)
    if caption:
        box.markdown(f"<p class='card-cap'>{caption}</p>", unsafe_allow_html=True)
    return box


def kpi(col, label: str, value: str, sub: str = "", accent: str = BLUE):
    col.markdown(
        f"<div class='kpi' style='--acc:{accent}'><div class='lbl'>{label}</div>"
        f"<div class='val'>{value}</div><div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def fmt(x: float, dec: int = 0) -> str:
    """Formato español: 1.234,5"""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def heatmap(z: pd.DataFrame, title_z: str, fmt_z: str = ".0f", height: int = 300) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
        colorscale=SEQ_BLUE, xgap=1, ygap=1,
        colorbar=dict(title=dict(text=title_z, side="right"), thickness=10, outlinewidth=0),
        hovertemplate="%{y} · %{x}h<br>" + title_z + ": %{z:" + fmt_z + "}<extra></extra>",
    ))
    fig = style(fig, height, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(title="Hora del día", dtick=2)
    return fig


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando y agregando los CSV de data/train …")
def get_data() -> pd.DataFrame:
    return load_hourly()


df_all = get_data()
valid_days = df_all.loc[df_all["completa"], "fecha_dia"]
d_min, d_max = valid_days.min().date(), valid_days.max().date()

# ----------------------------------------------------------------------------
# Barra lateral: segmentadores (slicers)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Filtros")
    rango = st.date_input("Periodo", value=(d_min, d_max), min_value=d_min, max_value=d_max,
                          format="DD/MM/YYYY")
    if isinstance(rango, (tuple, list)) and len(rango) == 2:
        f_ini, f_fin = rango
    else:
        f_ini, f_fin = d_min, d_max
    meses_disp = [m for m in sorted(df_all["mes_num"].unique())]
    meses_sel = st.pills("Meses", options=meses_disp, default=meses_disp, selection_mode="multi",
                         format_func=lambda m: MESES[m - 1]) or meses_disp
    tipo_dia = st.segmented_control("Tipo de día", ["Todos", "Laborables", "Fin de semana"],
                                    default="Todos")
    st.divider()
    excl_cong = st.toggle("Excluir lecturas nocturnas congeladas", value=True,
                          help="El contador repite la misma lectura cada noche de ~22:00 a 06:00. "
                               "Activado: esas horas no cuentan en los análisis de demanda.")
    st.caption("Los KPIs de coste y CO₂ usan siempre todas las horas, igual que el simulador oficial.")
    st.divider()
    st.caption(f"Train: {fmt(int(df_all['completa'].sum()))} horas completas · "
               f"{d_min:%d/%m/%Y} – {d_max:%d/%m/%Y}")

mask = (df_all["fecha_dia"].dt.date >= f_ini) & (df_all["fecha_dia"].dt.date <= f_fin)
mask &= df_all["mes_num"].isin(meses_sel)
if tipo_dia == "Laborables":
    mask &= ~df_all["finde"]
elif tipo_dia == "Fin de semana":
    mask &= df_all["finde"]
df = df_all[mask]
dc = df[df["completa"]]                                  # horas con consumo + mercado
dd = dc[~dc["load_congelada"]] if excl_cong else dc      # horas para análisis de demanda

# ----------------------------------------------------------------------------
# Cabecera + KPIs
# ----------------------------------------------------------------------------
st.markdown(
    "<div class='pbi-header'><h1>🔋 Campus Nord UPC · Explorador de datos</h1>"
    "<p>Datathon 2026 · Batería <b>2.000 kWh · ±500 kW</b> · Objetivo "
    "<b>70 % CO₂</b> + <b>30 % coste</b> frente al campus sin batería</p></div>",
    unsafe_allow_html=True,
)

if dc.empty:
    st.warning("No hay horas con datos para los filtros seleccionados.")
    st.stop()

e_cons = dc["load_kw"].sum() / 1000
e_pv = dc["pv_kw_sim"].sum() / 1000
coste = dc["coste_base_eur"].sum()
emis = dc["co2_base_kg"].sum() / 1000
precio_pond = coste / dc["red_base_kw"].sum()
co2_pond = dc["co2_base_kg"].sum() / dc["red_base_kw"].sum()

k = st.columns(6)
kpi(k[0], "Consumo", f"{fmt(e_cons)} MWh", f"{fmt(len(dc))} h · media {fmt(dc['load_kw'].mean())} kW", C_CONSUMO)
kpi(k[1], "Pico de demanda", f"{fmt(dd['load_kw'].max())} kW", f"mín. {fmt(dd['load_kw'].min())} kW", C_CONSUMO)
kpi(k[2], "Solar FV", f"{fmt(e_pv, 1)} MWh", f"cubre el {fmt(100 * e_pv / e_cons, 1)} % del consumo", C_SOLAR)
kpi(k[3], "Coste red (sin batería)", f"{fmt(coste / 1000, 1)} k€", f"{fmt(precio_pond, 3)} €/kWh ponderado", C_PRECIO)
kpi(k[4], "Emisiones (sin batería)", f"{fmt(emis, 1)} t CO₂", f"{fmt(co2_pond, 3)} kg/kWh ponderado", C_CO2)
kpi(k[5], "Batería vs consumo diario", f"{fmt(2000 / (e_cons * 1000 / max(dc['fecha_dia'].nunique(), 1)) * 100, 1)} %",
    "2 MWh frente a la energía de un día medio", GRAY)

st.write("")
tabs = st.tabs(["📊 Resumen", "🏫 Demanda", "💶 Precio y CO₂", "☀️ Solar y meteo",
                "🔋 Simulador de batería", "🧪 Calidad de datos"])

# ============================================================================
# 1. RESUMEN
# ============================================================================
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        daily = dc.groupby("fecha_dia").agg(mwh=("load_kw", lambda s: s.sum() / 1000), finde=("finde", "first"),
                                            horas=("load_kw", "size")).reset_index()
        _r = daily.loc[daily["finde"], "mwh"].mean() / daily.loc[~daily["finde"], "mwh"].mean()
        box = card("Consumo diario del campus", "Energía importada por día (MWh)."
                   + (f" Un día de fin de semana consume un {fmt(100 * (1 - _r))} % menos que uno laborable." if _r == _r else ""))
        fig = go.Figure()
        for name, flag, col in [("Laborable", False, C_LAB), ("Fin de semana", True, C_FINDE)]:
            sub = daily[daily["finde"] == flag]
            fig.add_bar(x=sub["fecha_dia"], y=sub["mwh"], name=name, marker_color=col,
                        customdata=sub["horas"],
                        hovertemplate="%{x|%a %d/%m/%Y}<br>%{y:.1f} MWh (%{customdata} h)<extra></extra>")
        fig.update_layout(barmode="overlay")
        fig.update_yaxes(title="MWh/día")
        box.plotly_chart(style(fig, 300), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Lo que dicen los datos", "Se recalcula con los filtros.")
        prof = dc.groupby("hora")[["co2_kg_per_kwh", "precio_eur_per_kwh"]].mean()
        limpias = prof["co2_kg_per_kwh"].nsmallest(3).index.sort_values().tolist()
        sucias = prof["co2_kg_per_kwh"].nlargest(3).index.sort_values().tolist()
        baratas = prof["precio_eur_per_kwh"].nsmallest(3).index.sort_values().tolist()
        caras = prof["precio_eur_per_kwh"].nlargest(3).index.sort_values().tolist()
        corr = dc["precio_eur_per_kwh"].corr(dc["co2_kg_per_kwh"])
        cyc = sim.daily_cycle_value(df)
        lab = dd.loc[~dd["finde"], "load_kw"].mean()
        fin = dd.loc[dd["finde"], "load_kw"].mean()
        items = [
            f"🌱 Horas <b>más limpias</b> de media: <b>{', '.join(f'{h}h' for h in limpias)}</b>; "
            f"<b>más sucias</b>: <b>{', '.join(f'{h}h' for h in sucias)}</b>.",
            f"💶 Horas <b>más baratas</b>: <b>{', '.join(f'{h}h' for h in baratas)}</b>; "
            f"<b>más caras</b>: <b>{', '.join(f'{h}h' for h in caras)}</b>.",
            f"🔗 Correlación precio–CO₂: <b>{fmt(corr, 2)}</b>. Los dos objetivos van casi siempre en la misma dirección.",
            f"🔋 Un ciclo diario carga→descarga reduce CO₂ el <b>{fmt(100 * (cyc['margen_co2'] > 0).mean(), 0)} %</b> "
            f"de los días, contando el 90 % de rendimiento.",
        ]
        if not np.isnan(lab) and not np.isnan(fin):
            items.append(f"🏫 Laborable medio: <b>{fmt(lab)} kW</b> · fin de semana: <b>{fmt(fin)} kW</b>.")
        items.append(f"☀️ La solar solo cubre el <b>{fmt(100 * e_pv / e_cons, 1)} %</b>: "
                     "nunca sobra energía, así que <b>cada kWh que se carga sale de la red</b>.")
        for it in items:
            box.markdown(f"<div class='insight'>{it}</div>", unsafe_allow_html=True)

    box = card("Matriz mensual", "Como una matriz de Power BI. Ordena por cualquier columna.")
    m = dc.groupby("mes").agg(
        horas=("load_kw", "size"),
        consumo_mwh=("load_kw", lambda s: s.sum() / 1000),
        pico_kw=("load_kw", "max"),
        solar_mwh=("pv_kw_sim", lambda s: s.sum() / 1000),
        coste_keur=("coste_base_eur", lambda s: s.sum() / 1000),
        co2_t=("co2_base_kg", lambda s: s.sum() / 1000),
        precio_medio=("precio_eur_per_kwh", "mean"),
        co2_medio=("co2_kg_per_kwh", "mean"),
        temp_media=("temp_amb", "mean"),
    )
    m["solar_pct"] = 100 * m["solar_mwh"] / m["consumo_mwh"]
    box.dataframe(
        m[["horas", "consumo_mwh", "pico_kw", "solar_mwh", "solar_pct", "coste_keur", "co2_t",
           "precio_medio", "co2_medio", "temp_media"]],
        width="stretch",
        column_config={
            "horas": st.column_config.NumberColumn("Horas", format="%d"),
            "consumo_mwh": st.column_config.ProgressColumn("Consumo (MWh)", format="%.0f",
                                                           min_value=0, max_value=float(m["consumo_mwh"].max())),
            "pico_kw": st.column_config.NumberColumn("Pico (kW)", format="%.0f"),
            "solar_mwh": st.column_config.NumberColumn("Solar (MWh)", format="%.1f"),
            "solar_pct": st.column_config.NumberColumn("Solar (%)", format="%.1f %%"),
            "coste_keur": st.column_config.ProgressColumn("Coste (k€)", format="%.1f",
                                                          min_value=0, max_value=float(m["coste_keur"].max())),
            "co2_t": st.column_config.ProgressColumn("CO₂ (t)", format="%.1f",
                                                     min_value=0, max_value=float(m["co2_t"].max())),
            "precio_medio": st.column_config.NumberColumn("Precio medio (€/kWh)", format="%.3f"),
            "co2_medio": st.column_config.NumberColumn("CO₂ medio (kg/kWh)", format="%.3f"),
            "temp_media": st.column_config.NumberColumn("T. media (°C)", format="%.1f"),
        },
    )

# ============================================================================
# 2. DEMANDA
# ============================================================================
with tabs[1]:
    if excl_cong:
        st.caption("⚠️ Se excluyen las horas de 22:00 a 06:00 con lectura congelada (ver **Calidad de datos**). "
                   "Desactívalo en la barra lateral para ver los valores que usa el simulador.")
    c1, c2 = st.columns(2)
    with c1:
        box = card("Consumo medio por hora y día de la semana (kW)",
                   "Mapa de calor: cuándo consume el campus. Las celdas vacías son horas sin lectura real.")
        z = dd.pivot_table(index="dia_semana", columns="hora", values="load_kw", aggfunc="mean", observed=False)
        z = z.reindex(index=DIAS, columns=range(24))
        box.plotly_chart(heatmap(z, "kW"), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Perfil horario: laborable frente a fin de semana",
                   "La línea es la media y la banda va del percentil 10 al 90.")
        fig = go.Figure()
        for name, flag, col in [("Laborable", False, C_LAB), ("Fin de semana", True, C_FINDE)]:
            s = dd[dd["finde"] == flag].groupby("hora")["load_kw"]
            if s.size().sum() == 0:
                continue
            q = s.quantile([0.1, 0.9]).unstack()
            fig.add_scatter(x=q.index, y=q[0.9], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=q.index, y=q[0.1], fill="tonexty", mode="lines", line=dict(width=0),
                            fillcolor=f"rgba({int(col[1:3], 16)},{int(col[3:5], 16)},{int(col[5:7], 16)},0.15)",
                            showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=s.mean().index, y=s.mean().values, name=name, mode="lines+markers",
                            line=dict(color=col, width=2), marker=dict(size=6),
                            hovertemplate=name + " %{x}h: %{y:.0f} kW<extra></extra>")
        fig.update_xaxes(title="Hora del día", dtick=2, range=[-0.5, 23.5])
        fig.update_yaxes(title="kW")
        box.plotly_chart(style(fig, 300), width="stretch", config={"displaylogo": False})

    c1, c2 = st.columns(2)
    with c1:
        box = card("Consumo medio por mes y hora (kW)", "Estacionalidad: en verano la climatización sube el consumo de día.")
        z = dd.pivot_table(index="mes", columns="hora", values="load_kw", aggfunc="mean").reindex(columns=range(24))
        box.plotly_chart(heatmap(z, "kW", height=330), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Consumo frente a temperatura exterior",
                   "Cada punto es una hora. Con más de ~25 °C el consumo sube (refrigeración).")
        fig = go.Figure()
        for name, flag, col in [("Laborable", False, C_LAB), ("Fin de semana", True, C_FINDE)]:
            sub = dd[(dd["finde"] == flag)].dropna(subset=["temp_amb"])
            if len(sub) > 2500:
                sub = sub.sample(2500, random_state=0)
            fig.add_scattergl(x=sub["temp_amb"], y=sub["load_kw"], mode="markers", name=name,
                              marker=dict(color=col, size=5, opacity=0.45, line=dict(width=0)),
                              hovertemplate="%{x:.1f} °C → %{y:.0f} kW<extra>" + name + "</extra>")
        fig.update_xaxes(title="Temperatura ambiente (°C)", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(title="Consumo (kW)")
        box.plotly_chart(style(fig, 330), width="stretch", config={"displaylogo": False})

    c1, c2 = st.columns(2)
    with c1:
        box = card("Curva monótona de carga",
                   "Horas ordenadas de mayor a menor consumo. La línea discontinua marca 500 kW: "
                   "el consumo nunca baja de ahí, así que descargar a tope nunca exporta energía.")
        srt = np.sort(dd["load_kw"].dropna().values)[::-1]
        fig = go.Figure(go.Scatter(x=np.arange(len(srt)) / max(len(srt) - 1, 1) * 100, y=srt, mode="lines",
                                   line=dict(color=C_CONSUMO, width=2), name="Consumo",
                                   hovertemplate="%{x:.1f} % de las horas > %{y:.0f} kW<extra></extra>"))
        fig.add_hline(y=500, line=dict(color=GRAY, dash="dash", width=1),
                      annotation_text="500 kW (potencia de la batería)", annotation_font_color=INK_2)
        fig.update_xaxes(title="% de horas", ticksuffix=" %")
        fig.update_yaxes(title="kW", rangemode="tozero")
        box.plotly_chart(style(fig, 300, legend=False), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Serie temporal horaria del consumo", "Arrastra para hacer zoom; doble clic para volver.")
        fig = go.Figure(go.Scattergl(x=dc.index, y=dc["load_kw"].where(~dc["load_congelada"] | (not excl_cong)),
                                     mode="lines", line=dict(color=C_CONSUMO, width=1), name="Consumo",
                                     hovertemplate="%{x|%a %d/%m %H:%M}<br>%{y:.0f} kW<extra></extra>"))
        fig.update_yaxes(title="kW")
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08))
        box.plotly_chart(style(fig, 300, legend=False), width="stretch", config={"displaylogo": False})

# ============================================================================
# 3. PRECIO Y CO2
# ============================================================================
with tabs[2]:
    mk = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"])
    st.markdown(
        "<div class='insight'>📌 <b>La clave del reto:</b> el agente conoce <b>de verdad</b> el precio y el CO₂ de "
        "las próximas 24 h. La batería gana cuando carga en las horas limpias y baratas y descarga en las sucias "
        "y caras, siempre que la diferencia supere la pérdida del 10 % de cada ciclo.</div>",
        unsafe_allow_html=True,
    )

    # Señal combinada del objetivo
    w_co2 = 0.7 / mk["co2_kg_per_kwh"].mean()
    w_pr = 0.3 / mk["precio_eur_per_kwh"].mean()
    mk = mk.assign(senal=w_co2 * mk["co2_kg_per_kwh"] + w_pr * mk["precio_eur_per_kwh"])

    c1, c2, c3 = st.columns(3)
    for col_, var, name, color, unit, fmt_h in [
        (c1, "co2_kg_per_kwh", "Intensidad de CO₂ de la red", C_CO2, "kg CO₂/kWh", ".3f"),
        (c2, "precio_eur_per_kwh", "Precio PVPC", C_PRECIO, "€/kWh", ".3f"),
        (c3, "senal", "Señal combinada del score (0,7·CO₂ + 0,3·precio)", GRAY, "índice (1 = media)", ".2f"),
    ]:
        with col_:
            cap = ("Normalizada: 1 = hora media. <b>Por debajo de 1 conviene cargar y por encima, descargar.</b>"
                   if var == "senal" else f"Perfil horario medio ({unit}) con banda P10–P90.")
            box = card(name, cap)
            g = mk.groupby("hora")[var]
            q = g.quantile([0.1, 0.9]).unstack()
            rgb = f"{int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)}"
            fig = go.Figure()
            fig.add_scatter(x=q.index, y=q[0.9], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=q.index, y=q[0.1], fill="tonexty", fillcolor=f"rgba({rgb},0.15)",
                            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=g.mean().index, y=g.mean().values, mode="lines+markers",
                            line=dict(color=color, width=2), marker=dict(size=6), name=name,
                            hovertemplate="%{x}h: %{y:" + fmt_h + "}<extra></extra>")
            if var == "senal":
                fig.add_hline(y=1, line=dict(color=INK_3, dash="dot", width=1))
            fig.update_xaxes(title="Hora del día", dtick=3, range=[-0.5, 23.5])
            fig.update_yaxes(title=unit)
            box.plotly_chart(style(fig, 270, legend=False), width="stretch", config={"displaylogo": False})

    c1, c2 = st.columns(2)
    with c1:
        box = card("CO₂ medio por mes y hora (kg/kWh)", "El valle solar del mediodía se ve todo el año, más ancho en primavera.")
        z = mk.pivot_table(index="mes", columns="hora", values="co2_kg_per_kwh", aggfunc="mean").reindex(columns=range(24))
        box.plotly_chart(heatmap(z, "kg/kWh", ".3f", 330), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Precio medio por mes y hora (€/kWh)", "La punta de 19–22 h es la hora cara casi todos los meses.")
        z = mk.pivot_table(index="mes", columns="hora", values="precio_eur_per_kwh", aggfunc="mean").reindex(columns=range(24))
        box.plotly_chart(heatmap(z, "€/kWh", ".3f", 330), width="stretch", config={"displaylogo": False})

    c1, c2 = st.columns(2)
    with c1:
        box = card("Precio frente a CO₂ por franja horaria",
                   f"Cada punto es una hora. Correlación: {fmt(mk['precio_eur_per_kwh'].corr(mk['co2_kg_per_kwh']), 2)}.")
        franjas = [("Noche (22–7h)", lambda h: (h >= 22) | (h < 7), BLUE),
                   ("Solar (10–17h)", lambda h: (h >= 10) & (h < 17), ORANGE),
                   ("Punta (7–10h, 17–22h)", lambda h: ((h >= 7) & (h < 10)) | ((h >= 17) & (h < 22)), AQUA)]
        fig = go.Figure()
        for name, fn, col in franjas:
            sub = mk[fn(mk["hora"])]
            fig.add_scattergl(x=sub["co2_kg_per_kwh"], y=sub["precio_eur_per_kwh"], mode="markers", name=name,
                              marker=dict(color=col, size=5, opacity=0.45, line=dict(width=0)),
                              hovertemplate="%{x:.3f} kg/kWh · %{y:.3f} €/kWh<extra>" + name + "</extra>")
        fig.update_xaxes(title="CO₂ (kg/kWh)", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(title="Precio (€/kWh)")
        box.plotly_chart(style(fig, 330), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("¿A qué hora está el mínimo y el máximo de CO₂ cada día?",
                   "Nº de días en que cada hora es la más limpia o la más sucia. Dice cuándo cargar y cuándo descargar.")
        cyc = sim.daily_cycle_value(df)
        fig = go.Figure()
        fig.add_bar(x=list(range(24)), y=cyc["hora_co2_min"].value_counts().reindex(range(24), fill_value=0),
                    name="Hora más limpia (cargar)", marker_color=BLUE,
                    hovertemplate="%{x}h: %{y} días<extra>Más limpia</extra>")
        fig.add_bar(x=list(range(24)), y=cyc["hora_co2_max"].value_counts().reindex(range(24), fill_value=0),
                    name="Hora más sucia (descargar)", marker_color=ORANGE,
                    hovertemplate="%{x}h: %{y} días<extra>Más sucia</extra>")
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="Hora del día", dtick=2)
        fig.update_yaxes(title="días")
        box.plotly_chart(style(fig, 330), width="stretch", config={"displaylogo": False})

    c1, c2 = st.columns(2)
    with c1:
        box = card("Margen diario de un ciclo de CO₂",
                   "0,9 × CO₂ máximo − CO₂ mínimo del día. Por encima de 0, un ciclo completo reduce emisiones.")
        fig = go.Figure(go.Bar(x=cyc.index, y=cyc["margen_co2"], marker_color=C_CO2, name="Margen CO₂",
                               hovertemplate="%{x|%d/%m/%Y}: %{y:.3f} kg/kWh<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=INK_3, width=1))
        fig.update_yaxes(title="kg CO₂ por kWh cargado")
        box.plotly_chart(style(fig, 280, legend=False), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Mix de generación peninsular", "% de la generación que emite CO₂, media por hora (P10–P90).")
        g = mk.groupby("hora")["porcentaje_con_emisiones"]
        q = g.quantile([0.1, 0.9]).unstack() * 100
        fig = go.Figure()
        fig.add_scatter(x=q.index, y=q[0.9], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=q.index, y=q[0.1], fill="tonexty", fillcolor="rgba(98,80,214,0.15)",
                        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=g.mean().index, y=g.mean().values * 100, mode="lines+markers", name="% con emisiones",
                        line=dict(color=C_CO2, width=2), marker=dict(size=6),
                        hovertemplate="%{x}h: %{y:.1f} %<extra></extra>")
        fig.update_xaxes(title="Hora del día", dtick=2)
        fig.update_yaxes(title="% generación fósil", ticksuffix=" %")
        box.plotly_chart(style(fig, 280, legend=False), width="stretch", config={"displaylogo": False})

    box = card("Serie temporal de CO₂ y precio", "Dos paneles con el eje temporal compartido. Usa el zoom para ver días concretos.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_scattergl(x=mk.index, y=mk["co2_kg_per_kwh"], mode="lines", line=dict(color=C_CO2, width=1),
                      name="CO₂ (kg/kWh)", row=1, col=1, hovertemplate="%{x|%d/%m %H}h: %{y:.3f} kg/kWh<extra></extra>")
    fig.add_scattergl(x=mk.index, y=mk["precio_eur_per_kwh"], mode="lines", line=dict(color=C_PRECIO, width=1),
                      name="Precio (€/kWh)", row=2, col=1, hovertemplate="%{x|%d/%m %H}h: %{y:.3f} €/kWh<extra></extra>")
    fig.update_yaxes(title_text="kg CO₂/kWh", row=1, col=1)
    fig.update_yaxes(title_text="€/kWh", row=2, col=1)
    box.plotly_chart(style(fig, 380), width="stretch", config={"displaylogo": False})

# ============================================================================
# 4. SOLAR Y METEO
# ============================================================================
with tabs[3]:
    sp = df.dropna(subset=["pv_kw"])
    k = st.columns(4)
    kpi(k[0], "Potencia FV máxima", f"{fmt(sp['pv_kw'].max())} kW", "las diapositivas hablan de hasta 450 kW", C_SOLAR)
    kpi(k[1], "Correlación FV–irradiancia", fmt(sp["pv_kw"].corr(sp["irradiancia"]), 3), "casi perfecta", C_SOLAR)
    kpi(k[2], "Irradiancia máxima", f"{fmt(df['irradiancia'].max())} W/m²", "piranómetro del tejado", YELLOW)
    kpi(k[3], "Temperatura", f"{fmt(df['temp_amb'].min(), 1)} – {fmt(df['temp_amb'].max(), 1)} °C",
        "mín. – máx. horaria", GRAY)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        box = card("Generación FV frente a irradiancia",
                   "Relación casi lineal. Un modelo solar con la irradiancia sería trivial, pero el agente no recibe la irradiancia.")
        sub = sp.dropna(subset=["irradiancia"])
        sub = sub[sub["irradiancia"] > 5]
        fig = go.Figure(go.Scattergl(x=sub["irradiancia"], y=sub["pv_kw"], mode="markers", name="Hora",
                                     marker=dict(color=C_SOLAR, size=5, opacity=0.4, line=dict(width=0)),
                                     hovertemplate="%{x:.0f} W/m² → %{y:.1f} kW<extra></extra>"))
        fig.update_xaxes(title="Irradiancia (W/m²)", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(title="FV total (kW)")
        box.plotly_chart(style(fig, 320, legend=False), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Generación FV media por mes y hora (kW)", "Ventana solar útil: ~8–19 h en verano y ~9–17 h en invierno.")
        z = sp.pivot_table(index="mes", columns="hora", values="pv_kw", aggfunc="mean").reindex(columns=range(24))
        box.plotly_chart(heatmap(z, "kW", ".1f", 320), width="stretch", config={"displaylogo": False})
    c1, c2 = st.columns(2)
    with c1:
        box = card("Energía FV diaria por inversor (kWh)", "Los 3 inversores producen casi lo mismo, así que no parece haber ninguno averiado.")
        dpv = sp.groupby("fecha_dia")[["pv_inv1_kw", "pv_inv2_kw", "pv_inv3_kw"]].sum()
        fig = go.Figure()
        for i, col in zip((1, 2, 3), (BLUE, ORANGE, AQUA)):
            fig.add_bar(x=dpv.index, y=dpv[f"pv_inv{i}_kw"], name=f"Inversor {i}", marker_color=col,
                        marker_line=dict(color="#ffffff", width=0.5),
                        hovertemplate="%{x|%d/%m/%Y}: %{y:.0f} kWh<extra>Inversor " + str(i) + "</extra>")
        fig.update_layout(barmode="stack", bargap=0.05, legend_traceorder="normal")
        fig.update_yaxes(title="kWh/día")
        box.plotly_chart(style(fig, 300), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Temperatura ambiente frente a temperatura de panel",
                   "Los paneles se calientan hasta ~20 °C por encima del aire a mediodía. Rinden ~0,4 % menos por cada °C por encima de 25.")
        sub = df.dropna(subset=["temp_amb", "temp_panel"])
        fig = go.Figure(go.Scattergl(x=sub["temp_amb"], y=sub["temp_panel"], mode="markers", name="Hora",
                                     marker=dict(color=sub["irradiancia"], colorscale=SEQ_BLUE, size=5, opacity=0.6,
                                                 colorbar=dict(title="W/m²", thickness=10, outlinewidth=0)),
                                     hovertemplate="Aire %{x:.1f} °C · Panel %{y:.1f} °C<extra></extra>"))
        lo, hi = float(sub["temp_amb"].min()), float(sub["temp_amb"].max())
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=INK_3, dash="dot", width=1),
                        name="Panel = aire", hoverinfo="skip")
        fig.update_xaxes(title="T. ambiente (°C)", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(title="T. panel (°C)")
        box.plotly_chart(style(fig, 300, legend=False), width="stretch", config={"displaylogo": False})

# ============================================================================
# 5. SIMULADOR
# ============================================================================
@st.cache_data(show_spinner=False)
def run_agent_cached(path_str: str, mtime: float, ini, fin):
    frame = sim_frame(df_all[(df_all["fecha_dia"].dt.date >= ini) & (df_all["fecha_dia"].dt.date <= fin)])
    agent = sim.load_agent(Path(path_str))
    return sim.run_agent(agent, frame)


@st.cache_data(show_spinner=False)
def run_lp_cached(ini, fin):
    frame = sim_frame(df_all[(df_all["fecha_dia"].dt.date >= ini) & (df_all["fecha_dia"].dt.date <= fin)])
    return sim.optimal_lp(frame)


with tabs[4]:
    st.markdown(
        "<div class='insight'>Ejecuta cualquier <code>agent.py</code> del repo (cualquier archivo con "
        "<code>class BatteryAgent</code>) con el <b>simulador oficial</b> (<code>starter_kit/environment.py</code>) "
        "sobre el periodo elegido. Lo compara con el <b>óptimo teórico</b>: una programación lineal que conoce "
        "todo el futuro. Ningún agente real puede superarlo, así que sirve para saber cuánto margen queda.</div>",
        unsafe_allow_html=True,
    )
    agents = sim.find_agents()
    c1, c2, c3 = st.columns([2, 2, 1])
    sel = c1.selectbox("Agente", list(agents.keys()),
                       index=list(agents.keys()).index("starter_kit/agent.py") if "starter_kit/agent.py" in agents else 0)
    sim_rango = c2.date_input("Periodo de simulación", value=(f_ini, f_fin), min_value=d_min, max_value=d_max,
                              format="DD/MM/YYYY", key="sim_rango",
                              help="Solo se usa el rango de fechas: el agente necesita horas consecutivas, "
                                   "así que aquí no se aplican los filtros de mes ni de tipo de día.")
    s_ini, s_fin = (sim_rango if isinstance(sim_rango, (tuple, list)) and len(sim_rango) == 2 else (f_ini, f_fin))
    c3.write("")
    go_btn = c3.button("▶ Ejecutar", type="primary", width="stretch")

    if go_btn:
        st.session_state["sim_req"] = (sel, s_ini, s_fin)
    req = st.session_state.get("sim_req")
    if req is None:
        st.info("Elige un agente y un periodo y pulsa **Ejecutar**.")
    else:
        sel_r, ini_r, fin_r = req
        p = agents.get(sel_r)
        try:
            with st.spinner("Simulando hora a hora …"):
                res_a, tr_a = run_agent_cached(str(p), p.stat().st_mtime, ini_r, fin_r)
                res_o, tr_o = run_lp_cached(ini_r, fin_r)
        except Exception as e:  # el agente del usuario puede fallar
            st.error(f"El agente ha fallado: {type(e).__name__}: {e}")
            st.stop()

        st.caption(f"**{sel_r}** · {ini_r:%d/%m/%Y} – {fin_r:%d/%m/%Y} · {fmt(len(tr_a))} horas · "
                   f"{res_a['segundos']} s de ejecución (el límite oficial es 60 s para ~1.700 h)")
        k = st.columns(5)
        pct_opt = 100 * res_a["score"] / res_o["score"] if res_o["score"] > 0 else 0
        bruto = 0.7 * res_a["co2_estalviat_pct"] + 0.3 * res_a["cost_estalviat_pct"]
        kpi(k[0], "Score del agente", fmt(res_a["score"], 2),
            f"sin penalizar: {fmt(bruto, 2)} · óptimo: {fmt(res_o['score'], 2)}", C_AGENTE)
        kpi(k[1], "% del óptimo", f"{fmt(pct_opt, 0)} %", "score agente / score óptimo", C_AGENTE)
        kpi(k[2], "Reducción CO₂", f"{fmt(res_a['co2_estalviat_pct'], 2)} %",
            f"{fmt(res_a['co2_estalviat_kg'] / 1000, 1)} t · óptimo {fmt(res_o['co2_estalviat_pct'], 2)} %", C_CO2)
        kpi(k[3], "Reducción coste", f"{fmt(res_a['cost_estalviat_pct'], 2)} %",
            f"{fmt(res_a['cost_estalviat_eur'] / 1000, 1)} k€ · óptimo {fmt(res_o['cost_estalviat_pct'], 2)} %", C_PRECIO)
        kpi(k[4], "Infracciones", fmt(res_a["infraccions"]),
            f"−{fmt(res_a['infraccions'] * 0.2, 1)} pts en local · −{fmt(res_a['infraccions'] * 1.0, 0)} según el enunciado",
            RED if res_a["infraccions"] else GREEN)
        if res_a["infraccions"]:
            st.markdown(
                "<div class='warnbox'>⚠️ <b>Hay infracciones.</b> Cuidado: <code>environment.py</code> resta 0,2 pts "
                "por infracción, pero el enunciado y las diapositivas dicen <b>1 pt</b>. Si las infracciones aparecen "
                "con SOC ≈ 0 o ≈ 2.000 kWh, suele ser por redondeo: hay que usar <code>0.9 ** 0.5</code> exacto "
                "y no <code>0.9487</code>, y dejar un pequeño margen.</div>",
                unsafe_allow_html=True,
            )

        # Explorador semanal
        box = card("Explorador de una semana: qué hace el agente frente al óptimo",
                   "Arriba, las señales de la red. Abajo, la potencia de la batería (positiva = carga) y la energía almacenada.")
        days = sorted(tr_a.index.normalize().unique())
        if len(days) > 7:
            start_day = box.select_slider("Semana que empieza el", options=days, value=days[0],
                                          format_func=lambda d: f"{DIAS[pd.Timestamp(d).dayofweek]} {pd.Timestamp(d):%d/%m/%Y}")
        else:
            start_day = days[0]
        w0 = pd.Timestamp(start_day)
        w1 = w0 + pd.Timedelta(days=7)
        wa, wo = tr_a[(tr_a.index >= w0) & (tr_a.index < w1)], tr_o[(tr_o.index >= w0) & (tr_o.index < w1)]
        wm = df_all.loc[wa.index]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.2, 0.2, 0.3, 0.3])
        fig.add_scatter(x=wm.index, y=wm["co2_kg_per_kwh"], mode="lines", line=dict(color=C_CO2, width=2),
                        name="CO₂ red", row=1, col=1, hovertemplate="%{y:.3f} kg/kWh<extra>CO₂</extra>")
        fig.add_scatter(x=wm.index, y=wm["precio_eur_per_kwh"], mode="lines", line=dict(color=C_PRECIO, width=2),
                        name="Precio", row=2, col=1, hovertemplate="%{y:.3f} €/kWh<extra>Precio</extra>")
        fig.add_bar(x=wa.index, y=wa["bateria_kw"], name="Agente (kW)", marker_color=C_AGENTE,
                    row=3, col=1, hovertemplate="%{y:.0f} kW<extra>Agente</extra>")
        fig.add_scatter(x=wo.index, y=wo["bateria_kw"], mode="lines", line=dict(color=C_OPTIMO, width=2, shape="hv"),
                        name="Óptimo (kW)", row=3, col=1, hovertemplate="%{y:.0f} kW<extra>Óptimo</extra>")
        fig.add_scatter(x=wa.index, y=wa["soc_kwh"], mode="lines", line=dict(color=C_AGENTE, width=2),
                        name="SOC agente", row=4, col=1, showlegend=False,
                        hovertemplate="%{y:.0f} kWh<extra>SOC agente</extra>")
        fig.add_scatter(x=wo.index, y=wo["soc_kwh"], mode="lines", line=dict(color=C_OPTIMO, width=2),
                        name="SOC óptimo", row=4, col=1, showlegend=False,
                        hovertemplate="%{y:.0f} kWh<extra>SOC óptimo</extra>")
        inf = wa[wa["infracciones"] > 0]
        if len(inf):
            fig.add_scatter(x=inf.index, y=inf["soc_kwh"], mode="markers", name="Infracción",
                            marker=dict(symbol="x", size=10, color=RED), row=4, col=1)
        fig.update_yaxes(title_text="kg/kWh", row=1, col=1)
        fig.update_yaxes(title_text="€/kWh", row=2, col=1)
        fig.update_yaxes(title_text="kW", row=3, col=1, range=[-560, 560])
        fig.update_yaxes(title_text="kWh", row=4, col=1, range=[-50, 2050])
        fig.update_layout(hovermode="x unified")
        box.plotly_chart(style(fig, 560), width="stretch", config={"displaylogo": False})

        c1, c2 = st.columns(2)
        with c1:
            box = card("Estrategia media por hora", "Potencia media de la batería a cada hora del día (positiva = carga).")
            fig = go.Figure()
            for name, tr, col in [("Agente", tr_a, C_AGENTE), ("Óptimo teórico", tr_o, C_OPTIMO)]:
                g = tr.groupby(tr.index.hour)["bateria_kw"].mean()
                fig.add_scatter(x=g.index, y=g.values, mode="lines+markers", name=name,
                                line=dict(color=col, width=2), marker=dict(size=6),
                                hovertemplate="%{x}h: %{y:.0f} kW<extra>" + name + "</extra>")
            fig.add_hline(y=0, line=dict(color=INK_3, width=1))
            fig.update_xaxes(title="Hora del día", dtick=2)
            fig.update_yaxes(title="kW medio")
            box.plotly_chart(style(fig, 300), width="stretch", config={"displaylogo": False})
        with c2:
            box = card("Score acumulado por mes", "En qué meses gana más o menos puntos el agente frente al óptimo.")
            base_kw = df_all.loc[tr_a.index, "red_base_kw"]
            co2v = df_all.loc[tr_a.index, "co2_kg_per_kwh"]
            prv = df_all.loc[tr_a.index, "precio_eur_per_kwh"]
            co2_b, cost_b = float((base_kw * co2v).sum()), float((base_kw * prv).sum())
            fig = go.Figure()
            for name, tr, col in [("Agente", tr_a, C_AGENTE), ("Óptimo teórico", tr_o, C_OPTIMO)]:
                contrib = 100 * (0.7 * (base_kw - tr["red_kw"]) * co2v / co2_b
                                 + 0.3 * (base_kw - tr["red_kw"]) * prv / cost_b)
                g = contrib.groupby(tr.index.to_period("M").astype(str)).sum()
                fig.add_bar(x=g.index, y=g.values, name=name, marker_color=col,
                            hovertemplate="%{x}: %{y:.2f} pts<extra>" + name + "</extra>")
            fig.update_layout(barmode="group")
            fig.update_yaxes(title="puntos de score (sin penalizar)")
            box.plotly_chart(style(fig, 300), width="stretch", config={"displaylogo": False})

# ============================================================================
# 6. CALIDAD DE DATOS
# ============================================================================
with tabs[5]:
    full = df_all.copy()
    # Estado por hora: 0 sin dato · 1 congelado · 2 OK · 3 test oculto
    estado = np.where(full["es_test"], 3,
                      np.where(full["load_kw"].isna(), 0, np.where(full["load_congelada"], 1, 2)))
    full["estado"] = estado
    cal = full.pivot_table(index="fecha_dia", columns="hora", values="estado", aggfunc="first").reindex(columns=range(24))
    labels = {0: "Sin dato", 1: "Lectura congelada", 2: "Dato real", 3: "Test oculto (no disponible)"}
    colors = {0: GRAY_LIGHT, 1: ORANGE, 2: BLUE, 3: VIOLET}
    cs = []
    for v in range(4):
        cs += [[v / 4, colors[v]], [(v + 1) / 4, colors[v]]]
    text = np.vectorize(lambda v: labels.get(int(v), "") if not np.isnan(v) else "")(cal.values.astype(float))

    c1, c2 = st.columns([3, 2])
    with c1:
        box = card("Mapa de cobertura del contador (día × hora)",
                   "Cada celda es una hora. Muestra qué es dato real, qué se repite y qué falta.")
        fig = go.Figure(go.Heatmap(
            z=cal.values, x=list(range(24)), y=cal.index, zmin=-0.5, zmax=3.5, colorscale=cs, showscale=False,
            customdata=text, hovertemplate="%{y|%a %d/%m/%Y} %{x}h<br>%{customdata}<extra></extra>",
        ))
        for v in (2, 1, 0, 3):
            fig.add_scatter(x=[None], y=[None], mode="markers", name=labels[v],
                            marker=dict(symbol="square", size=12, color=colors[v]))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(title="Hora del día", dtick=2)
        box.plotly_chart(style(fig, 640), width="stretch", config={"displaylogo": False})
    with c2:
        box = card("Completitud por sensor", "% de horas del periodo de train con dato (sin contar la semana de test).")
        tr_mask = ~full["es_test"]
        comp = pd.DataFrame({
            "Sensor": ["Contador (consumo)", "Inversores (FV)", "Meteo (irradiancia)", "Precio PVPC", "CO₂ red"],
            "Horas con dato": [full.loc[tr_mask, c].notna().sum() for c in
                               ["load_kw", "pv_kw", "irradiancia", "precio_eur_per_kwh", "co2_kg_per_kwh"]],
        })
        comp["%"] = 100 * comp["Horas con dato"] / tr_mask.sum()
        box.dataframe(comp, hide_index=True, width="stretch",
                      column_config={"%": st.column_config.ProgressColumn("% cobertura", format="%.1f %%",
                                                                          min_value=0, max_value=100)})
        n_cong = int(full["load_congelada"].sum())
        n_ok = int((full["estado"] == 2).sum())
        box.markdown(
            f"<div class='warnbox'>🧊 <b>{fmt(n_cong)} horas</b> de consumo ({fmt(100 * n_cong / max(n_cong + n_ok, 1), 0)} %) "
            "son una lectura <b>congelada</b>: cada noche, de ~22:05 a 06:00, el contador repite la última "
            "lectura. El consumo nocturno del simulador es plano y artificial.</div>",
            unsafe_allow_html=True,
        )

        box = card("Problemas detectados y su impacto en el reto")
        issues = [
            ("🧊", "Consumo nocturno congelado", "El valor de 22–06 h es la lectura de las ~22 h repetida. "
             "No predigáis la noche a partir de la noche: el consumo de las 22 h ya la predice exactamente."),
            ("🕳️", "Noviembre 2025 vacío", "Falta casi todo el mes (contador, FV y meteo). El precio y el CO₂ sí están."),
            ("☀️", "Inversores sin datos desde el 16/08/2026", "El simulador del dashboard imputa la FV a 0 en esas horas."),
            ("📉", "FV mucho menor de lo anunciado", "Máximo de ~144 kW frente a los 450 kW de las diapositivas. "
             "La solar es ~2 % del consumo y nunca hay excedente."),
            ("➖", "Posible doble resta de la FV", "El tutorial usa el contador (neto en el PCC) como consumo y el "
             "simulador vuelve a restar la FV. El impacto es pequeño (~2 %), pero conviene saberlo."),
            ("⚖️", "Penalización inconsistente", "environment.py: −0,2 pts por infracción. Enunciado y diapositivas: −1 pt."),
            ("🎯", "Test = últimos 7 días de cada mes", "Train son los días 1 a ~24. Las semanas de test incluyen "
             "finales de mes y fiestas (p. ej. Navidad, 25–31/12)."),
        ]
        for ico, t, d in issues:
            box.markdown(f"<div class='insight'>{ico} <b>{t}.</b> {d}</div>", unsafe_allow_html=True)

st.caption("Datos: data/train/ del reto UPC Sostenible · Visualizador hecho con Streamlit + Plotly · "
           "`streamlit run dashboard/app.py`")
