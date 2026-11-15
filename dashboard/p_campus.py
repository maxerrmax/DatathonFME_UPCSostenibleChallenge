"""Pàgina 4 - El campus i la xarxa: quan és car un kWh, com de gran ha de ser la bateria, qualitat de les dades."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import models
import sim
import ui
from data import load_hourly
from ui import GRAY, INK, INK_3, OURS, RIVAL, fmt


@st.cache_data(show_spinner="Agregant els CSV de 5 minuts …")
def get_data() -> pd.DataFrame:
    return load_hourly()


def heatmap(z: pd.DataFrame, unit: str, f: str = ".0f", height: int = 290) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index], colorscale=ui.SEQ,
        xgap=1, ygap=1, colorbar=dict(title=dict(text=unit, side="right"), thickness=8, outlinewidth=0, len=0.9),
        hovertemplate="%{y} · %{x}:00<br>%{z:" + f + "} " + unit + "<extra></extra>"))
    fig = ui.style(fig, height, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category", tickfont=dict(color=INK))
    fig.update_xaxes(title="hora del dia", dtick=2, showline=False, ticks="")
    return fig


def render():
    df_all = get_data()
    valid = df_all.loc[df_all["completa"], "fecha_dia"]
    d_min, d_max = valid.min().date(), valid.max().date()
    c1, c2, c3 = st.columns([3, 3, 5])
    rng = c1.date_input("Període", value=(d_min, d_max), min_value=d_min, max_value=d_max, format="DD/MM/YYYY")
    f_ini, f_fin = rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (d_min, d_max)
    kind = c2.segmented_control("Dies", ["Tots", "Feiners", "Cap de setmana"], default="Tots") or "Tots"
    c3.markdown("<p class='cc' style='margin-top:1.7rem'>Dades mesurades al campus (entrenament: de setembre de 2025 a "
                "setembre de 2026, sense els darrers 7 dies de cada mes). El Mode Difícil multiplica la planta "
                "solar per 12; aquí es mostra la planta real si no es diu el contrari.</p>", unsafe_allow_html=True)
    mask = (df_all["fecha_dia"].dt.date >= f_ini) & (df_all["fecha_dia"].dt.date <= f_fin)
    if kind != "Tots":
        mask &= df_all["finde"] == (kind == "Cap de setmana")
    df = df_all[mask]
    dc = df[df["completa"]]
    if dc.empty:
        st.warning("No hi ha hores amb dades per a aquesta selecció.")
        return
    dd = dc[~dc["load_congelada"]]
    mk = df.dropna(subset=["precio_eur_per_kwh", "co2_kg_per_kwh"])
    days = max(dc["fecha_dia"].nunique(), 1)
    e, pv = dc["load_kw"].sum() / 1000, dc["pv_kw_sim"].sum() / 1000
    ui.facts([
        ("Consum del campus", f"{fmt(e / days, 1)} <small>MWh al dia</small>",
         f"{fmt(e)} MWh en {fmt(days)} dies; punta de {fmt(dd['load_kw'].max())} kW"),
        ("Emissions sense bateria", f"{fmt(dc['co2_base_kg'].sum() / 1000, 0)} <small>t de CO₂</small>",
         f"{fmt(dc['co2_base_kg'].sum() / dc['red_base_kw'].sum(), 3)} kg per kWh importat"),
        ("Factura sense bateria", f"{fmt(dc['coste_base_eur'].sum() / 1000, 0)} <small>k€</small>",
         f"{fmt(dc['coste_base_eur'].sum() / dc['red_base_kw'].sum(), 3)} € per kWh importat"),
        ("Solar avui", f"{fmt(100 * pv / e, 1)} <small>% del consum</small>",
         f"planta real, màxim de {fmt(df['pv_kw'].max())} kW; ×12 al Mode Difícil"),
    ], cols=4)

    # ------------------------------------------------------------------ senyal
    ui.section("Quan fa més mal un kWh de la xarxa?")
    w_co2, w_pr = 0.7 / mk["co2_kg_per_kwh"].mean(), 0.3 / mk["precio_eur_per_kwh"].mean()
    mk = mk.assign(signal=w_co2 * mk["co2_kg_per_kwh"] + w_pr * mk["precio_eur_per_kwh"])
    g = mk.groupby("hora")["signal"]
    prof = g.mean()
    lo, hi = prof.nsmallest(3).index.sort_values().tolist(), prof.nlargest(3).index.sort_values().tolist()
    corr = mk["precio_eur_per_kwh"].corr(mk["co2_kg_per_kwh"])
    cyc = sim.daily_cycle_value(df)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        box = ui.panel(f"Carregar cap a les {lo[0]}–{lo[-1] + 1} h, descarregar cap a les {hi[0]}–{hi[-1] + 1} h",
                       "Cost en punts d'importar un kWh a cada hora (0,7 × CO₂ + 0,3 × preu, tots dos respecte a la "
                       "seva mitjana; 1 = hora mitjana). Línia: mitjana. Banda: del percentil 10 al 90 dels dies.")
        q = g.quantile([0.1, 0.9]).unstack()
        fig = go.Figure()
        fig.add_scatter(x=q.index, y=q[0.9], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=q.index, y=q[0.1], fill="tonexty", fillcolor="rgba(18,18,18,0.10)", mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=prof.index, y=prof.values, mode="lines", line=dict(color=INK, width=2.5),
                        hovertemplate="%{x}:00 · %{y:.2f}<extra></extra>")
        fig.add_hline(y=1, line=dict(color=GRAY, width=1, dash="dot"))
        fig = ui.style(fig, 290, legend=False)
        fig.update_xaxes(title="hora del dia", dtick=2, range=[-0.5, 23.5])
        fig.update_yaxes(side="right")
        ui.show(fig, box)
        ui.source(f"La banda és ampla: la millor hora canvia d'un dia a l'altre, i per això l'agent fa previsions "
                  f"en lloc de seguir un horari fix. Correlació preu–CO₂: {fmt(corr, 2)}.", box)
    with c2:
        share = 100 * (cyc["margen_co2"] > 0).mean()
        box = ui.panel(f"Un cicle diari de càrrega i descàrrega redueix CO₂ el {fmt(share, 0)} % dels dies",
                       "Dies en què cada hora ha estat la més neta (carregar) o la més bruta (descarregar). "
                       "El cicle compensa quan 0,9 × la més bruta > la més neta, perquè es perd un 10 % en l'anada i tornada.")
        fig = go.Figure()
        fig.add_bar(x=list(range(24)), y=cyc["hora_co2_min"].value_counts().reindex(range(24), fill_value=0),
                    name="hora més neta del dia", marker_color=OURS, hovertemplate="%{x}:00 · %{y} dies<extra>més neta</extra>")
        fig.add_bar(x=list(range(24)), y=cyc["hora_co2_max"].value_counts().reindex(range(24), fill_value=0),
                    name="hora més bruta del dia", marker_color=RIVAL, hovertemplate="%{x}:00 · %{y} dies<extra>més bruta</extra>")
        fig.update_layout(barmode="group", bargap=0.2)
        fig = ui.style(fig, 290)
        fig.update_xaxes(title="hora del dia", dtick=2)
        fig.update_yaxes(title="dies")
        ui.show(fig, box)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        box = ui.panel("El CO₂ de la xarxa té una vall al migdia tot l'any, més ampla a la primavera",
                       "Intensitat mitjana per mes i hora, kg de CO₂ per kWh. La vall és la solar peninsular.")
        z = mk.pivot_table(index="mes", columns="hora", values="co2_kg_per_kwh", aggfunc="mean").reindex(columns=range(24))
        z.index = [f"{ui.MESOS[int(m[5:]) - 1]} {m[2:4]}" for m in z.index]
        ui.show(heatmap(z, "kg/kWh", ".3f", 310), box)
    with c2:
        box = ui.panel("El campus consumeix de dia i entre setmana",
                       "Demanda mitjana en kW per dia de la setmana i hora. Les cel·les buides són hores de nit en "
                       "què el comptador repeteix l'última lectura (vegeu la qualitat de les dades, més avall).")
        z = dd.pivot_table(index="dow", columns="hora", values="load_kw", aggfunc="mean").reindex(index=range(7), columns=range(24))
        z.index = ui.DIES
        ui.show(heatmap(z, "kW", ".0f", 310), box)

    # ------------------------------------------------------------------ dimensionament
    sz = models.sizing()
    if sz is not None:
        ui.section("Com de gran ha de ser la bateria?",
                   "Una pregunta per a UPC Sostenible més que per al concurs: què es guanya amb cada MWh instal·lat.")
        c1, c2 = st.columns([6, 5], gap="large")
        s1, s12 = sz[sz.pv_scale == 1].set_index("cap_kwh"), sz[sz.pv_scale == 12].set_index("cap_kwh")
        with c1:
            box = ui.panel(f"Passar de 2 a 6 MWh gairebé triplica l'estalvi; la planta solar gran hi afegeix "
                           f"{fmt(s12.loc[6000, 'score'] - s1.loc[6000, 'score'], 1)} punts",
                           "Millor puntuació possible (previsió perfecta) per a cada mida de bateria, amb potència = "
                           "capacitat / 4 h, sobre tot l'any d'entrenament. Els quadrats són les dues configuracions del repte.")
            fig = go.Figure()
            for s, name, colr in ((s1, "solar actual", GRAY), (s12, "solar ×12 (Mode Difícil)", OURS)):
                fig.add_scatter(x=s.index / 1000, y=s["score"], mode="lines+markers", name=name,
                                line=dict(color=colr, width=2.5), marker=dict(size=7),
                                customdata=np.c_[s["co2_t_year"], s["eur_year"] / 1000],
                                hovertemplate=name + "<br>%{x:.0f} MWh · puntuació %{y:.1f}<br>%{customdata[0]:.0f} t de CO₂ "
                                              "i %{customdata[1]:.0f} k€ l'any<extra></extra>")
            for s, cap, txt, ax, ay in ((s1, 2000, "Mode Estàndard", 66, 26), (s12, 6000, "Mode Difícil", -50, -30)):
                fig.add_scatter(x=[cap / 1000], y=[s.loc[cap, "score"]], mode="markers", showlegend=False,
                                marker=dict(symbol="square-open", size=15, line=dict(color=INK, width=2)), hoverinfo="skip")
                fig.add_annotation(x=cap / 1000, y=s.loc[cap, "score"], text=f"{txt} · {fmt(s.loc[cap, 'score'], 1)}",
                                   ax=ax, ay=ay, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=INK_3,
                                   standoff=9, font=dict(color=INK, size=12))
            fig = ui.style(fig, 330)
            fig.update_xaxes(title="capacitat de la bateria, MWh", rangemode="tozero")
            fig.update_yaxes(rangemode="tozero", side="right")
            ui.show(fig, box)
        with c2:
            t = s12.reset_index()
            t["marg"] = t["co2_t_year"].diff() / (t["cap_kwh"].diff() / 1000)
            t.loc[0, "marg"] = t.loc[0, "co2_t_year"] / (t.loc[0, "cap_kwh"] / 1000)
            t["bat"] = [f"{fmt(c / 1000)} MWh / {fmt(c / 4000, 2).rstrip('0').rstrip(',')} MW" for c in t["cap_kwh"]]
            t["keur"] = t["eur_year"] / 1000
            box = ui.panel(f"Cada MWh de més evita menys CO₂: {fmt(t['marg'].iloc[0], 0)} t l'any el primer, "
                           f"{fmt(t['marg'].iloc[-1], 0)} t l'últim",
                           "Amb solar ×12 i previsió perfecta. El nostre agent arriba al 93 % de l'òptim a la validació.")
            ui.table(t, names={"bat": "Bateria", "score": "Puntuació", "co2_t_year": "CO₂ evitat, t/any",
                               "keur": "Factura, k€/any", "marg": "t per MWh de més"},
                     formats={"score": lambda v: fmt(v, 1), "co2_t_year": lambda v: fmt(v, 0), "keur": lambda v: fmt(v, 0),
                              "marg": lambda v: fmt(v, 0)},
                     left=("bat",), bars={"marg": float(t["marg"].max())}, highlight=lambda r: r["cap_kwh"] == 6000, where=box)
            ui.source(f"<br>Base amb solar ×12: {fmt(t['co2_base_t_year'].iloc[0], 0)} t de CO₂ i "
                      f"{fmt(t['eur_base_year'].iloc[0] / 1000, 0)} k€ l'any, anualitzat a partir de les hores d'entrenament. "
                      "En negreta, la bateria del Mode Difícil.", box)

    # ------------------------------------------------------------------ qualitat de dades
    ui.section("Ens podem refiar de les dades?")
    full = df_all.copy()
    full["estado"] = np.where(full["es_test"], 3, np.where(full["load_kw"].isna(), 0, np.where(full["load_congelada"], 1, 2)))
    cal = full.pivot_table(index="fecha_dia", columns="hora", values="estado", aggfunc="first").reindex(columns=range(24))
    labels = {0: "sense lectura", 1: "lectura repetida", 2: "lectura real", 3: "setmana de test (oculta)"}
    colors = {0: "#f1f1f1", 1: RIVAL, 2: OURS, 3: "#bdbdbd"}
    cs = []
    for v in range(4):
        cs += [[v / 4, colors[v]], [(v + 1) / 4, colors[v]]]
    text = np.vectorize(lambda v: labels.get(int(v), "") if v == v else "")(cal.values.astype(float))
    n_fz, n_ok = int(full["load_congelada"].sum()), int((full["estado"] == 2).sum())
    c1, c2 = st.columns([6, 5], gap="large")
    with c1:
        box = ui.panel(f"El {fmt(100 * n_fz / max(n_fz + n_ok, 1), 0)} % de les lectures de consum són un valor de nit "
                       "repetit, i falta el novembre",
                       "Cada cel·la és una hora del comptador del campus: els dies cap avall, les hores cap a la dreta.")
        fig = go.Figure(go.Heatmap(z=cal.values, x=list(range(24)), y=cal.index, zmin=-0.5, zmax=3.5, colorscale=cs,
                                   showscale=False, customdata=text,
                                   hovertemplate="%{y|%d/%m/%Y} %{x}:00<br>%{customdata}<extra></extra>"))
        for v in (2, 1, 0, 3):
            fig.add_scatter(x=[None], y=[None], mode="markers", name=labels[v], marker=dict(symbol="square", size=11, color=colors[v]))
        fig = ui.style(fig, 520)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickformat="%m/%Y", dtick="M2")
        fig.update_xaxes(title="hora del dia", dtick=2, showline=False, ticks="")
        ui.show(fig, box)
    with c2:
        trm = ~full["es_test"]
        comp = pd.DataFrame({"src": ["Comptador (consum)", "Inversors (solar)", "Estació meteorològica", "Preu PVPC", "CO₂ de la xarxa"],
                             "cov": [100 * full.loc[trm, c].notna().sum() / trm.sum() for c in
                                     ["load_kw", "pv_kw", "irradiancia", "precio_eur_per_kwh", "co2_kg_per_kwh"]]})
        box = ui.panel("Les dades de mercat són completes; les dels sensors del campus, no",
                       "Part de les hores d'entrenament que tenen valor.")
        ui.table(comp, names={"src": "Font", "cov": "Cobertura"}, formats={"cov": lambda v: f"{fmt(v, 1)} %"},
                 left=("src",), bars={"cov": 100}, where=box)
        ui.note("Què implica per a l'agent.",
                "<ul><li><b>Les nits són planes per construcció.</b> De les 22 h a les 6 h el comptador repeteix l'últim "
                "valor, de manera que la lectura de les 22 h ja diu tota la nit.</li>"
                "<li><b>El novembre de 2025 no té dades del campus</b> (comptador, inversors, meteo); el preu i el CO₂ sí.</li>"
                "<li><b>Els inversors s'aturen el 16 d'agost de 2026.</b> A partir d'aleshores la solar es pren com a zero.</li>"
                "<li><b>El test són els darrers 7 dies de cada mes</b>, amb la setmana de Nadal i el final d'agost: "
                "setmanes diferents de la major part de l'entrenament.</li>"
                "<li><b>Penalització incoherent:</b> el simulador local resta 0,2 punts per infracció i les bases en "
                "diuen 1,0. Tots els agents d'aquest dashboard en tenen zero.</li></ul>", where=box)
