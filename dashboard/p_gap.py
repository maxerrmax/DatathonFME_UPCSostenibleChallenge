"""Pàgina «On són els punts»: evolució de la validació de l'agent, distància a l'òptim i al líder, idees provades."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live
import models
import ui
from ui import GRAY_LIGHT, INK, OURS, fmt


def render():
    L = live.load()
    reg, raw = L.registry, models.arena_raw()
    ch = models.choose(reg)
    opt = raw["oracle"]["1"]["score"] if raw else None
    lad = pd.DataFrame(reg["ladder"])
    lad["gain"] = lad["score"].diff()
    steps = lad.dropna(subset=["gain"])
    now, first = lad["score"].iloc[-1], lad["score"].iloc[0]
    lead = L.standings[0] if L.standings else reg["leaderboard"]["teams"][0]
    best_pub = max(s["public"] for s in reg["submissions"] if s.get("public") is not None)

    ui.intro("Evolució de l'agent i distància als referents",
             "Puntuació de validació (plec 1) després de cada canvi i distància a l'òptim amb previsió perfecta i al "
             "primer del rànquing públic.")
    tiles = [("Validació: primer agent → actual", f"{fmt(first, 2)} → {fmt(now, 2)}", f"{fmt(now - first, 2, True)} punts", "val"),
             ("Validació: distància a l'òptim", fmt(opt - now, 2) if opt else "—",
              f"òptim amb previsió perfecta: {fmt(opt, 2)}" if opt else "sense dades de l'arena", "val"),
             ("Públic: distància al primer", fmt(lead["score"] - best_pub, 2),
              f"{lead['team']}, {fmt(lead['score'], 2)}; la nostra millor, {fmt(best_pub, 2)}", "pub")]
    ui.tiles(tiles, cols=3)

    box = ui.panel("Punts de validació afegits per cada canvi",
                   "Punts de validació que ha afegit cada canvi, del més antic al més recent (plec 1). La barra grisa és "
                   "la distància que queda fins a l'òptim amb previsió perfecta.", src="val")
    y = list(steps["step"]) + (["Fins a l'òptim"] if opt else [])
    x = list(steps["gain"]) + ([opt - now] if opt else [])
    fig = go.Figure(go.Bar(
        x=x, y=y, orientation="h", marker_color=[OURS] * len(steps) + [GRAY_LIGHT] * (1 if opt else 0),
        text=[fmt(v, 2, True) for v in x], textposition="outside", cliponaxis=False,
        textfont=dict(size=12.5, color=INK),
        customdata=list(steps["score"]) + ([opt] if opt else []),
        hovertemplate="%{y}<br>%{x:+.2f} punts, fins a %{customdata:.2f}<extra></extra>"))
    fig = ui.style(fig, 380, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12, color=INK))
    fig.update_xaxes(range=[0, max(x) * 1.14], showticklabels=False, showline=False, ticks="")
    ui.show(fig, box)

    with ui.more(f"Idees provades ({len(reg['dropped'])})"):
        ui.table(pd.DataFrame(reg["dropped"]),
                 names={"idea": "Idea", "validation": "Validació", "public": "Públic", "verdict": "Resultat"},
                 left=("idea", "verdict"))
