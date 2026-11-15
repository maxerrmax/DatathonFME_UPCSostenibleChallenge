"""Pàgina 3 - D'on han sortit els punts, on són els que falten i què hem descartat."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import models
import ui
from ui import GRAY, GRAY_LIGHT, INK, OURS, fmt


def render():
    reg, raw = models.registry(), models.arena_raw()
    opt = raw["oracle"]["1"]["score"] if raw else 22.34
    lad = pd.DataFrame(reg["ladder"])
    lad["gain"] = lad["score"].diff()
    steps = lad.dropna(subset=["gain"])
    now = lad["score"].iloc[-1]
    left = opt - now

    ui.section("Com hem arribat fins aquí")
    c1, c2 = st.columns([6, 5], gap="large")
    with c1:
        big = steps.loc[steps["gain"].idxmax()]
        box = ui.panel(f"La previsió ha fet la feina grossa: {fmt(big['gain'], 2, True)} dels "
                       f"{fmt(now - lad['score'].iloc[0], 2, True)} punts guanyats des del primer agent",
                       f"Punts de validació que ha afegit cada canvi, del més antic al més recent (plec 1). "
                       "La barra grisa és el que encara hi afegiria una previsió perfecta.")
        y = list(steps["step"]) + ["Encara obert fins a l'òptim"]
        x = list(steps["gain"]) + [left]
        fig = go.Figure(go.Bar(
            x=x, y=y, orientation="h", marker_color=[OURS] * len(steps) + [GRAY_LIGHT],
            text=[fmt(v, 2, True) for v in x], textposition="outside", cliponaxis=False,
            textfont=dict(size=12.5, color=INK),
            customdata=list(steps["score"]) + [opt], hovertemplate="%{y}<br>%{x:+.2f} punts, fins a %{customdata:.2f}<extra></extra>"))
        fig = ui.style(fig, 360, legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12, color=INK))
        fig.update_xaxes(range=[0, max(x) * 1.14], showticklabels=False, showline=False, ticks="")
        ui.show(fig, box)
        ui.source(f"Inici: {fmt(lad['score'].iloc[0], 2)} → ara: {fmt(now, 2)} → òptim: {fmt(opt, 2)}.", box)
    with c2:
        pi = pd.DataFrame(reg["perfect_info"]).sort_values("gain", ascending=False)
        box = ui.panel("El punt següent és a la previsió del CO₂, no a la del preu",
                       "Punts de validació que guanyaríem si l'agent rebés el valor real d'una sola entrada i la "
                       "resta quedés igual (mesurat sobre la v8).")
        fig = go.Figure(go.Bar(
            x=pi["gain"], y=pi["what"], orientation="h", marker_color=[INK] + [GRAY] * (len(pi) - 1),
            text=[fmt(v, 1, True) for v in pi["gain"]], textposition="outside", cliponaxis=False,
            textfont=dict(size=12.5, color=INK),
            customdata=pi["note"], hovertemplate="%{y}<br>%{x:+.1f} punts<br>%{customdata}<extra></extra>"))
        fig = ui.style(fig, 170, legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12, color=INK))
        fig.update_xaxes(range=[0, pi["gain"].max() * 1.2], showticklabels=False, showline=False, ticks="")
        ui.show(fig, box)
        ui.note("Lectura.", "<ul>" + "".join(f"<li><b>{r.what}:</b> {r.note}.</li>" for r in pi.itertuples()) + "</ul>"
                f"Les tres xifres no se sumen exactament, però juntes expliquen la major part dels {fmt(left, 2)} punts "
                "que queden oberts. Els equips que tenim al davant també ens guanyen sobretot en CO₂.", where=box)

    ui.section("Què hem provat i hem descartat")
    c1, c2 = st.columns([5, 6], gap="large")
    with c1:
        rs = pd.DataFrame(reg["ratio_sweep"], columns=["ratio", "score"])
        spread = rs["score"].max() - rs["score"].min()
        box = ui.panel(f"Canviar el pes del CO₂ respecte al preu mou la puntuació {fmt(spread, 2)} punts com a molt",
                       "Puntuació de validació per a quatre valors del pes que el planificador dóna al CO₂ respecte "
                       "al preu. L'objectiu oficial 70/30 correspon a 0,62. L'eix comença a zero expressament: "
                       "la corba és plana.")
        fig = go.Figure(go.Scatter(x=rs["ratio"], y=rs["score"], mode="lines+markers+text",
                                   line=dict(color=OURS, width=2), marker=dict(size=9),
                                   text=[fmt(v, 2) for v in rs["score"]], textposition="top center",
                                   textfont=dict(size=12.5, color=INK),
                                   hovertemplate="pes %{x:.2f} · puntuació %{y:.2f}<extra></extra>"))
        fig = ui.style(fig, 260, legend=False)
        fig.update_yaxes(range=[0, opt * 1.1])
        fig.update_xaxes(title="pes del CO₂ al planificador", dtick=0.05)
        ui.show(fig, box)
        ui.source("El compromís entre els dos objectius no és una palanca: CO₂ i preu gairebé sempre "
                  "assenyalen les mateixes hores.", box)
    with c2:
        box = ui.panel(f"{len(reg['dropped'])} idees que no han sobreviscut",
                       "Efecte a la validació i, quan es va enviar, al rànquing públic. "
                       "Les deixem aquí perquè ningú hi torni a perdre un vespre.")
        ui.table(pd.DataFrame(reg["dropped"]),
                 names={"idea": "Idea", "validation": "Validació", "public": "Públic", "verdict": "Per què es descarta"},
                 left=("idea", "verdict"), where=box)

    ui.note("Què fem ara.",
            "<ol><li><b>CO₂ a curt termini</b> (hores 4 a 8): hi ha la meitat del marge del CO₂.</li>"
            "<li><b>Demanda neta de l'hora actual:</b> més senyals de curt termini, perquè l'agent decideix abans de veure l'hora.</li>"
            "<li><b>No tocar els pesos de l'objectiu ni l'horitzó:</b> tots dos són plans.</li>"
            "<li><b>Enviar qualsevol versió nova al test públic abans de refiar-nos-en:</b> vegeu la primera pestanya.</li></ol>")
