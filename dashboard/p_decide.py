"""
Pestanya «Decidir»: eines per comparar-nos amb els altres equips (rànquing públic), veure el valor de la informació
perfecta (validació) i la relació entre el sol i el CO₂ (dades del campus). Els títols descriuen què es mostra; la lectura
la fa qui mira el gràfic. Tot surt de `live.load()`, de `model_data/` i de les dades del campus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import live
import models
import p_campus
import p_front
import ui
from ui import C_CO2, C_COST, C_SOLAR, GRAY, GRAY_LIGHT, INK, OURS, RIVAL, fmt

W_CO2, W_COST = 0.7, 0.3


def _gap_rows(stand: list, us: dict, leader: str) -> list[dict]:
    """Per a cada referent: punts que ens falten, repartits entre CO₂ (70 %) i factura (30 %)."""
    def mean(rows, k):
        return float(np.mean([r[k] for r in rows]))

    groups = [
        (f"{'Líder' if stand[0]['team'] == leader else 'Millor amb desglossament'} · {stand[0]['team']}", stand[0]),
        ("Mitjana del top 5", stand[:5]),
        ("Mitjana del top 10", stand[:10]),
    ]
    out = []
    for name, g in groups:
        c = g["co2"] if isinstance(g, dict) else mean(g, "co2")
        k = g["cost"] if isinstance(g, dict) else mean(g, "cost")
        out.append(dict(name=name, co2=W_CO2 * (c - us["co2"]), cost=W_COST * (k - us["cost"])))
    return out


def _rivals(stand: list, us: dict, L: live.Live, leader: str):
    rows = _gap_rows(stand, us, leader)
    t_ink = ui.DARK["ink"] if ui.is_dark() else ui.LIGHT["ink"]

    cl, cr = st.columns([1, 1], gap="small")
    with cl:
        box = ui.panel(
            "Diferència de nota amb el líder i el top, per component",
            "Punts de nota que separen la nostra entrada de cada referent (equips únics), repartits entre el CO₂ (pes 0,7) "
            "i la factura (pes 0,3). Positiu: el referent puntua més.", src="pub")
        fig = go.Figure()
        names = [r["name"] for r in rows]
        for key, label, col in (("co2", "CO₂", C_CO2), ("cost", "Factura", C_COST)):
            vals = [r[key] for r in rows]
            fig.add_bar(
                y=names, x=vals, orientation="h", name=label, marker_color=col,
                text=[fmt(v, 2, True) for v in vals], textposition="inside", insidetextanchor="middle",
                textfont=dict(color="#FFFFFF", size=12),
                hovertemplate=f"{label}: %{{x:+.2f}} punts<extra></extra>",
            )
        fig.update_layout(barmode="relative")
        fig = ui.style(fig, 250, legend=True)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=t_ink, size=12))
        fig.update_xaxes(tickformat="+.1f", title="punts de nota")
        ui.show(fig, box)
    with cr:
        box = ui.panel(
            "Cap a on millorar: amunt i a la dreta",
            "Cada punt és un equip únic. La línia és la nostra nota: tot el que hi ha per sobre i a la dreta puntua més.",
            src="pub")
        seen, xs, ys, cols, sizes, txt = set(), [], [], [], [], []
        for r in stand:
            key = (r["co2"], r["cost"])
            if key in seen:
                continue
            seen.add(key)
            me = r["team"] == L.us
            xs.append(r["co2"])
            ys.append(r["cost"])
            cols.append(OURS if me else RIVAL if r["rank"] == 1 else GRAY)
            sizes.append(15 if me else 11 if r["rank"] == 1 else 8)
            txt.append(f"{r['team']} · {fmt(r['score'], 2)}")
        fig = go.Figure()
        x0, x1 = min(xs) - 0.15, max(xs) + 0.15
        xl = np.array([x0, x1])
        fig.add_scatter(
            x=xl, y=(us["score"] - W_CO2 * xl) / W_COST, mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=OURS, width=1.5, dash="dot"),
        )
        fig.add_scatter(
            x=xs, y=ys, mode="markers", text=txt, showlegend=False, hovertemplate="%{text}<extra></extra>",
            marker=dict(color=cols, size=sizes, line=dict(color="#fff", width=1)),
        )
        fig = ui.style(fig, 250, legend=False)
        fig.update_xaxes(title="CO₂ evitat (%)", tickformat=".1f", range=[x0, x1])
        fig.update_yaxes(title="Factura estalviada (%)", tickformat=".1f", showgrid=False, range=[min(ys) - 0.4, max(ys) + 0.4])
        ui.show(fig, box)


def _boxplot(stand_all: list, stand: list, us: dict):
    box = ui.panel(
        "Distribució del rànquing públic",
        "Caixa: del primer al tercer quartil; ratlla: mediana; bigotis: valors no atípics; punts: la resta d'equips únics. "
        "El diamant és la nostra entrada.", src="pub")
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                        subplot_titles=("Nota", "CO₂ evitat (%)", "Factura estalviada (%)"))
    for i, (k, col) in enumerate((("score", INK), ("co2", C_CO2), ("cost", C_COST)), start=1):
        vals = [r[k] for r in (stand_all if k == "score" else stand)]
        fig.add_trace(go.Box(y=vals, name="", boxpoints="outliers", marker_color=col, line=dict(color=col, width=1.6),
                             fillcolor=ui.rgba(col, 0.14), showlegend=False,
                             hovertemplate="%{y:.2f}<extra>equips</extra>"), row=1, col=i)
        fig.add_trace(go.Scatter(x=[""], y=[us[k]], mode="markers", showlegend=False,
                                 marker=dict(symbol="diamond", size=12, color=OURS, line=dict(color="#fff", width=1.5)),
                                 hovertemplate=f"{us['team']} · %{{y:.2f}}<extra></extra>"), row=1, col=i)
    fig = ui.style(fig, 340, legend=False)
    fig.update_layout(margin=dict(l=2, r=14, t=56, b=8))
    fig.update_xaxes(showticklabels=False, showline=False, ticks="")
    fig.update_yaxes(tickformat=".1f")
    for a in fig.layout.annotations:
        a.font = dict(size=11.5, color=INK)
    ui.show(fig, box)


def _levers(reg: dict):
    iv = models.info_value(reg)
    t_ink = ui.DARK["ink"] if ui.is_dark() else ui.LIGHT["ink"]
    if iv.empty:
        return
    iv = iv.sort_values("gain")
    tot = iv["what"].str.startswith("Tot conegut")
    H = 300
    box = ui.panel(
        "Punts de validació amb informació perfecta, per entrada",
        "Punts de nota que guanyaria la versió de referència a la validació (mitjana dels 3 plecs) si una sola entrada "
        "fos perfecta i la resta es quedés igual. L'última barra suposa totes les entrades perfectes alhora.", src="val")
    fig = go.Figure(go.Bar(
        y=iv["what"], x=iv["gain"], orientation="h",
        marker_color=[t_ink if t else RIVAL for t in tot],
        text=[f"+{fmt(g, 2)}" for g in iv["gain"]], textposition="outside", textfont=dict(color=t_ink, size=12),
        cliponaxis=False, customdata=iv["note"], hovertemplate="%{y}: +%{x:.2f} punts<br>%{customdata}<extra></extra>",
    ))
    fig = ui.style(fig, H, legend=False)
    fig.update_layout(margin=dict(l=2, r=30, t=8, b=2))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=t_ink))
    fig.update_xaxes(range=[0, iv["gain"].max() * 1.2], tickformat=".1f", title="punts de nota")
    ui.show(fig, box)


def _sun(df: pd.DataFrame):
    t_ink = ui.DARK["ink"] if ui.is_dark() else ui.LIGHT["ink"]
    mode = st.segmented_control("Època", ["Tot l'any", "Abr–Set", "Oct–Mar"], default="Tot l'any", key="sun_season") or "Tot l'any"
    d = df[df["co2_kg_per_kwh"].notna() & df["pv_kw"].notna()]
    if mode == "Abr–Set":
        d = d[d["mes_num"].between(4, 9)]
    elif mode == "Oct–Mar":
        d = d[~d["mes_num"].between(4, 9)]
    if len(d) < 200:
        st.info("Poques dades per a aquesta època.")
        return
    prof = d.groupby("hora").agg(pv=("pv_kw", "mean"), co2=("co2_kg_per_kwh", "mean"))
    h_co2, h_pv = int(prof["co2"].idxmin()), int(prof["pv"].idxmax())
    noon = d[d["hora"].between(10, 16)].copy()
    noon["bin"] = pd.qcut(noon["pv_kw"].rank(method="first"), 4, labels=False)
    by = noon.groupby("bin", observed=True)["co2_kg_per_kwh"].mean()
    lim = noon.groupby("bin", observed=True)["pv_kw"].agg(["min", "max"])
    corr = float(noon["pv_kw"].corr(noon["co2_kg_per_kwh"]))

    cl, cr = st.columns([1.3, 1], gap="small")
    with cl:
        box = ui.panel(
            "CO₂ de la xarxa i producció solar del campus per hora del dia",
            f"Mitjana per hora del dia: producció solar (àrea) i CO₂ de la xarxa (línia). El CO₂ mitjà és mínim a les "
            f"{h_co2} h i la producció solar màxima a les {h_pv} h.", src="camp")
        view = box.segmented_control("Vista", ["Línies", "Mapa de calor"], default="Línies", key="sun_view") or "Línies"
        if view == "Mapa de calor":
            hm = d.pivot_table(index="mes_num", columns="hora", values=["co2_kg_per_kwh", "pv_kw"], aggfunc="mean")
            rows_ = [int(m) for m in hm.index]
            ylab = [ui.MESOS[m - 1] for m in rows_]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=("CO₂ de la xarxa (kg/kWh)", "Producció solar (kW)"))
            for r_, (col_, unit, fm, cb) in enumerate((("co2_kg_per_kwh", "kg/kWh", ".3f", 0.82), ("pv_kw", "kW", ".0f", 0.2)), start=1):
                z = hm[col_].reindex(columns=range(24))
                fig.add_trace(go.Heatmap(
                    z=z.values, x=list(range(24)), y=ylab, colorscale=ui.SEQ, xgap=1, ygap=1,
                    colorbar=dict(thickness=8, outlinewidth=0, len=0.4, y=cb, title=dict(text=unit, side="right")),
                    hovertemplate="%{y} · %{x}:00<br>%{z:" + fm + "} " + unit + "<extra></extra>"), row=r_, col=1)
            fig = ui.style(fig, 380, legend=False)
            fig.update_layout(margin=dict(l=2, r=14, t=30, b=2))
            fig.update_yaxes(autorange="reversed", showgrid=False, type="category", tickfont=dict(color=t_ink))
            fig.update_xaxes(dtick=3, showline=False, ticks="")
            fig.update_xaxes(title="hora del dia", row=2, col=1)
            for an in fig.layout.annotations:
                an.font = dict(size=11.5, color=t_ink)
            ui.show(fig, box)
        else:
            fig = go.Figure()
            fig.add_scatter(
                x=prof.index, y=prof["pv"], name="Solar (kW)", mode="lines", fill="tozeroy", yaxis="y2",
                line=dict(color="#C9B800", width=1), fillcolor=ui.rgba(C_SOLAR, 0.45),
                hovertemplate="%{x} h · %{y:.0f} kW<extra>Solar</extra>",
            )
            fig.add_scatter(
                x=prof.index, y=prof["co2"], name="CO₂ (kg/kWh)", mode="lines+markers",
                line=dict(color=C_CO2, width=2.5), marker=dict(size=5),
                hovertemplate="%{x} h · %{y:.3f} kg/kWh<extra>CO₂</extra>",
            )
            fig = ui.style(fig, 300, legend=True)
            fig.update_layout(
                yaxis2=dict(
                    overlaying="y", side="right", showgrid=False, tickformat=".0f", rangemode="tozero",
                    title=dict(text="solar (kW)", font=dict(size=12, color=t_ink)),
                    tickfont=dict(color=t_ink),
                )
            )
            fig.update_yaxes(showgrid=False, tickformat=".2f", title="CO₂ (kg/kWh)", rangemode="tozero", selector=dict(overlaying=None))
            fig.update_xaxes(title="hora del dia", dtick=3)
            ui.show(fig, box)
    with cr:
        box = ui.panel(
            "CO₂ mitjà de 10 a 16 h segons la producció solar",
            f"Les hores de 10 a 16 h s'ordenen per producció solar i es reparteixen en 4 grups iguals (quartils). "
            f"L'etiqueta de cada barra és l'interval de kW del grup. Correlació entre solar i CO₂: {fmt(corr, 2)} "
            "(una relació, no una prova de causalitat).", src="camp")
        names = ["Q1", "Q2", "Q3", "Q4"][: len(by)]
        labels = [f"{n}<br>{fmt(lim.loc[i, 'min'])}–{fmt(lim.loc[i, 'max'])} kW" for n, i in zip(names, by.index)]
        fig = go.Figure(
            go.Bar(
                x=labels, y=by.values, marker_color=[GRAY_LIGHT] * (len(by) - 1) + [C_CO2],
                text=[fmt(v, 3) for v in by.values], textposition="outside",
                textfont=dict(color=t_ink, size=12), cliponaxis=False,
                hovertemplate="%{x}<br>%{y:.3f} kg/kWh<extra></extra>",
            )
        )
        fig = ui.style(fig, 300, legend=False)
        fig.update_yaxes(showgrid=False, visible=False, range=[0, by.max() * 1.18])
        fig.update_xaxes(title="grup de producció solar (kW)", showline=False, ticks="")
        ui.show(fig, box)


def render():
    L = live.load()
    reg = L.registry
    stand_all = p_front.deduplicate_standings(L.standings)
    stand = [r for r in stand_all if r.get("co2") is not None and r.get("cost") is not None]
    leader = stand_all[0]["team"] if stand_all else ""
    us = next((r for r in stand if r["team"] == L.us), None)

    if not us or len(stand) < 5:
        ui.note("Sense rànquing.", "Cal una font a <code>dashboard/live.py</code> per comparar-nos amb els altres equips.")
    else:
        ui.section("Comparació amb els altres equips", src="pub")
        _rivals(stand, us, L, leader)
        _boxplot(stand_all, stand, us)
    ui.section("Informació perfecta", src="val")
    _levers(reg)
    ui.section("El sol i el CO₂", src="camp")
    _sun(p_campus.get_data())
