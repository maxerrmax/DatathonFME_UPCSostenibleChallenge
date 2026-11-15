"""Pàgina 3 - D'on han sortit els punts, on són els que falten i què hem descartat."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import models
import ui
from ui import GRAY, GRAY_LIGHT, INK, NAVY, OURS, fmt


def render():
    reg, raw = models.registry(), models.arena_raw()
    ch = models.choose(reg)
    opt = raw["oracle"]["1"]["score"] if raw else 22.34
    lad = pd.DataFrame(reg["ladder"])
    lad["gain"] = lad["score"].diff()
    steps = lad.dropna(subset=["gain"])
    now, first = lad["score"].iloc[-1], lad["score"].iloc[0]
    left = opt - now
    pi = pd.DataFrame(reg["perfect_info"]).sort_values("gain", ascending=False).reset_index(drop=True)
    top, last = pi.iloc[0], pi.iloc[-1]
    lead = reg["leaderboard"]["teams"][0]
    mine = ch["table"].loc[ch["chosen"], "public"]

    ui.intro("On són els punts que falten",
             "Què ha fet pujar l'agent fins ara, quina informació li falta per arribar a l'òptim i en què no val la pena "
             "insistir.")
    ui.tiles([
        ("Guanyat des del primer agent", fmt(now - first, 2, True), f"validació: de {fmt(first, 2)} a {fmt(now, 2)}", "navy"),
        ("Encara obert fins a l'òptim", fmt(left, 2), f"l'òptim amb previsió perfecta fa {fmt(opt, 2)}", ""),
        ("Distància al líder", fmt(lead["score"] - mine, 2), f"{lead['team']}, {fmt(lead['score'], 2)} al públic", "sun"),
        ("Idees descartades", str(len(reg["dropped"])), "amb xifres, al desplegable de sota", ""),
    ])

    c1, c2 = st.columns([6, 5], gap="large")
    with c1:
        big = steps.loc[steps["gain"].idxmax()]
        box = ui.panel(f"La previsió ha fet la feina grossa: {fmt(big['gain'], 2, True)} dels {fmt(now - first, 2, True)} punts",
                       "Punts de validació que ha afegit cada canvi, del més antic al més recent (plec 1). "
                       "La barra grisa és el que encara hi afegiria una previsió perfecta.")
        y = list(steps["step"]) + ["Encara obert fins a l'òptim"]
        x = list(steps["gain"]) + [left]
        fig = go.Figure(go.Bar(
            x=x, y=y, orientation="h", marker_color=[OURS] * len(steps) + [GRAY_LIGHT],
            text=[fmt(v, 2, True) for v in x], textposition="outside", cliponaxis=False,
            textfont=dict(size=12.5, color=INK),
            customdata=list(steps["score"]) + [opt], hovertemplate="%{y}<br>%{x:+.2f} punts, fins a %{customdata:.2f}<extra></extra>"))
        fig = ui.style(fig, 350, legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12, color=INK))
        fig.update_xaxes(range=[0, max(x) * 1.14], showticklabels=False, showline=False, ticks="")
        ui.show(fig, box)
    with c2:
        box = ui.panel(f"El que més val és saber: {top['what'][0].lower() + top['what'][1:]}",
                       "Punts de validació que guanyaríem si l'agent rebés el valor real d'una sola entrada i la resta "
                       "quedés igual. " + reg.get("perfect_info_note", ""))
        fig = go.Figure(go.Bar(
            x=pi["gain"], y=pi["what"], orientation="h", marker_color=[NAVY] + [GRAY] * (len(pi) - 1),
            text=[fmt(v, 2, True) for v in pi["gain"]], textposition="outside", cliponaxis=False,
            textfont=dict(size=12.5, color=INK),
            customdata=pi["note"], hovertemplate="%{y}<br>%{x:+.2f} punts<br>%{customdata}<extra></extra>"))
        fig = ui.style(fig, 170, legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12, color=INK))
        fig.update_xaxes(range=[0, pi["gain"].max() * 1.25], showticklabels=False, showline=False, ticks="")
        ui.show(fig, box)
        ui.source(f"De la segona: {pi.iloc[1]['note']}.", box)

    pend = "".join(f"<li><b>Enviar la {p['id']} un cop</b> i quedar-s'hi només si fa més de {fmt(p['threshold'], 2)} al "
                   "públic (regla de la pestanya Veredicte).</li>" for p in ch["pending"])
    ui.note("Què fem ara.",
            f"<ol><li><b>Atacar la previsió de: {top['what'][0].lower() + top['what'][1:]}</b> ({fmt(top['gain'], 2, True)} "
            f"punts possibles): {top['note']}.</li>"
            f"<li><b>No gastar més hores en: {last['what'][0].lower() + last['what'][1:]}</b> ({fmt(last['gain'], 2, True)}): "
            f"{last['note']}.</li>"
            "<li><b>No tocar el planificador:</b> unes 40 estratègies de decisió han mogut ±0,25 punts com a molt.</li>"
            f"{pend}</ol>Avís: fins ara, grans millores de l'error de previsió s'han traduït en guanys de puntuació molt petits.")

    with ui.more(f"Què hem provat i descartat ({len(reg['dropped'])} idees)"):
        ui.table(pd.DataFrame(reg["dropped"]),
                 names={"idea": "Idea", "validation": "Validació", "public": "Públic", "verdict": "Per què es descarta"},
                 left=("idea", "verdict"))
        rs = pd.DataFrame(reg["ratio_sweep"], columns=["ratio", "score"])
        ui.source(f"<br>Pes del CO₂ respecte al preu al planificador: amb {', '.join(fmt(r, 2) for r in rs['ratio'])} la validació "
                  f"fa {', '.join(fmt(s, 2) for s in rs['score'])}. La corba és plana: CO₂ i preu gairebé sempre assenyalen "
                  "les mateixes hores.")
