"""Pàgina «Com triem»: el detall de la tria automàtica (models.choose). La portada és p_front.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live
import models
import ui
from ui import GRAY, GRAY_LIGHT, INK, INK_2, OURS, RIVAL, fmt


def _label(name: str, reg: dict) -> str:
    for s in reg["submissions"]:
        if s.get("arena") == name and s["id"] != name:
            return f"{name} (= {s['id']})"
    return name


def render():
    reg = live.load().registry
    ch = models.choose(reg)
    me, anchor, tab = ch["chosen"], ch["anchor"], ch["table"]
    subs = pd.DataFrame(reg["submissions"])
    sel = subs.set_index("id").loc[me]
    scored = subs.dropna(subset=["public"]).reset_index(drop=True)
    lead = reg["leaderboard"]["teams"][0]
    raw, runs = models.arena_raw(), models.runs()
    T = W = None
    if raw:
        T, W = models.arena_table(raw)
    pt, lim = reg["public_test"], reg["limits"]
    opt1 = raw["oracle"]["1"]["score"] if raw else None
    fooled = tab[(tab["pred"] > 0) & (tab["d_pub"] < 0)]

    wks = pt["hours"] // 168
    lb = reg["leaderboard"]

    # ------------------------------------------------------------------ com tria
    ui.section("Estimació de l'avantatge al test privat",
               "El rànquing públic només ensenya una part del test i una comparació hi té un soroll de ±"
               + fmt(ch["sd_pub"], 2) + " punts. L'estimació combina la diferència pública i la de validació respecte "
               "a la versió àncora (la de més nota pública amb validació). Està calibrada amb només "
               + str(len(ch["fit"])) + " lliuraments i suposa que el test privat es comporta com el públic: serveix per "
               "ordenar versions, no per predir la nota final.", src="calc")
    c1, c2 = st.columns([6, 5], gap="large")
    with c1:
        el = tab[tab["eligible"] & (tab["est"] > -1.2)].sort_values("est")
        n_out = int((tab["eligible"] & (tab["est"] <= -1.2)).sum())
        n_better = int((el["est"] > 0.005).sum())
        box = ui.panel(f"Avantatge estimat respecte a la {anchor}, per versió",
                       f"Punts estimats al test privat respecte a la {anchor}; la barra és l'interval del 95 %. "
                       f"Blau: versió amb millor estimació ({me}). Cercle taronja buit: encara no enviada (només "
                       "validació, no entra a l'estimació principal).", src="calc")
        fig = go.Figure()
        fig.add_scatter(x=el["est"], y=el.index, mode="markers", showlegend=False,
                        marker=dict(color=[OURS if i == me else GRAY for i in el.index],
                                    size=[15 if i == me else 10 for i in el.index]),
                        error_x=dict(type="data", array=1.96 * el["sd"], color=GRAY, thickness=1.3, width=0),
                        customdata=np.c_[el["public"], el["validation"].fillna(-1)],
                        hovertemplate="<b>%{y}</b> · %{x:+.2f} punts estimats<br>públic %{customdata[0]:.2f}<extra></extra>")
        for p in ch["pending"]:
            fig.add_scatter(x=[p["pred"]], y=[p["id"]], mode="markers", showlegend=False,
                            marker=dict(symbol="circle-open", size=12, color=RIVAL, line=dict(width=2.5)),
                            error_x=dict(type="data", array=[1.96 * ch["sd_val"]], color=RIVAL, thickness=1.3, width=0),
                            hovertemplate=f"<b>{p['id']}</b> · sense nota pública<br>%{{x:+.2f}} només per validació<extra></extra>")
        fig.add_vline(x=0, line=dict(color=INK, width=1.2, dash="dot"))
        fig = ui.style(fig, 44 + 30 * (len(el) + len(ch["pending"])), legend=False)
        fig.update_xaxes(tickformat="+.1f", showgrid=True, gridcolor=ui.GRID, showline=False, ticks="")
        fig.update_yaxes(type="category", showgrid=False, tickfont=dict(size=13, color=INK))
        ui.show(fig, box)
        if n_out:
            ui.source(f"Fora d'escala: {', '.join(tab.index[tab['eligible'] & (tab['est'] <= -1.2)])}.", box)
    with c2:
        fit = pd.DataFrame(ch["fit"], columns=["id", "dv", "dp"])
        box = ui.panel(f"Diferència de validació i diferència pública, per lliurament",
                       f"Cada punt és un lliurament: diferència amb la {anchor} a la validació (horitzontal) i al rànquing "
                       f"públic (vertical). Línia blava: pendent {fmt(ch['beta'], 2)} amb ±{fmt(ch['sd_val'], 2)} de soroll. "
                       "Línia grisa: si la validació es complís al peu de la lletra.", src="calc")
        lo, hi = fit["dv"].min() - 0.1, fit["dv"].max() + 0.15
        fig = go.Figure()
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GRAY_LIGHT, width=1.5, dash="dash"),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[lo, hi], y=[ch["beta"] * lo, ch["beta"] * hi], mode="lines", line=dict(color=OURS, width=2.5),
                        hoverinfo="skip", showlegend=False)
        pos = {"v8": "top left", "v9": "bottom right", "v13": "middle right", "v10": "middle left"}
        fig.add_scatter(x=fit["dv"], y=fit["dp"], mode="markers+text", text=fit["id"], showlegend=False,
                        textposition=[pos.get(i, "top center") for i in fit["id"]], textfont=dict(size=12.5, color=INK),
                        marker=dict(size=11, color=[RIVAL if (v > 0 and p < 0) else GRAY for v, p in zip(fit["dv"], fit["dp"])]),
                        hovertemplate="<b>%{text}</b><br>validació %{x:+.2f}<br>públic %{y:+.2f}<extra></extra>")
        fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dot"))
        fig.add_vline(x=0, line=dict(color=INK, width=1, dash="dot"))
        fig = ui.style(fig, 330, legend=False)
        fig.update_xaxes(title=f"validació respecte a la {anchor}", tickformat="+.1f")
        fig.update_yaxes(title=f"públic respecte a la {anchor}", tickformat="+.1f")
        ui.show(fig, box)

    # ------------------------------------------------------------------ detall
    if T is None:
        st.info("Executeu `python dashboard/build_models.py` per afegir l'anàlisi dels aspirants.")
        return

    main = T[(T["wk_delta"] > -0.6) & (T.index != models.REF)].sort_values("wk_delta")
    with ui.more(f"Estratègies: {len(T) - 1} provades contra la {models.REF} a la validació"):
        c1, c2 = st.columns([6, 5], gap="large")
        with c1:
            n_sig = int(((main["wk_delta"] > 0) & main["significant"]).sum())
            ui.heading(f"Diferència setmanal respecte a la {models.REF}",
                       f"Diferència mitjana de puntuació setmanal en {int(T['weeks'].max())} setmanes de validació; la barra "
                       f"és l'interval del 95 %. {n_sig} estratègies queden per sobre de la {models.REF} més enllà del soroll.",
                       "val")
            col = [RIVAL if (s and d > 0) else (INK_2 if s else GRAY_LIGHT) for s, d in zip(main["significant"], main["wk_delta"])]
            fig = go.Figure(go.Scatter(
                x=main["wk_delta"], y=[_label(n, reg) for n in main.index], mode="markers", showlegend=False,
                marker=dict(color=col, size=9), error_x=dict(type="data", array=main["ci"], color=GRAY, thickness=1.2, width=0),
                customdata=np.c_[main["wins"], main["weeks"]],
                hovertemplate="<b>%{y}</b><br>%{x:+.2f} punts per setmana<br>millor en %{customdata[0]} de "
                              "%{customdata[1]} setmanes<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=OURS, width=2.5))
            fig = ui.style(fig, max(380, 19 * len(main) + 70), legend=False)
            span = float((main["wk_delta"] + main["ci"]).max() - (main["wk_delta"] - main["ci"]).min())
            fig.update_xaxes(showgrid=True, gridcolor=ui.GRID, tickformat="+.1f", dtick=0.2 if span < 3 else 0.5,
                             side="top", showline=False, ticks="")
            fig.update_yaxes(showgrid=False, tickfont=dict(size=11.5, color=INK))
            ui.show(fig)
        with c2:
            opts = [n for n in T.index if n != models.REF]
            cand = st.selectbox("Estratègia a examinar", opts, format_func=lambda n: _label(n, reg))
            r = T.loc[cand]
            d = (W.loc[cand] - W.loc[models.REF]).dropna()
            dates = [pd.Timestamp(c.split("|")[1]).strftime("%d/%m/%y") for c in d.index]
            ui.heading(f"{_label(cand, reg)}: setmana a setmana",
                       f"Puntuació setmanal menys la de la {models.REF}; en taronja, setmanes en què guanya l'aspirant "
                       f"({int(r['wins'])} de {int(r['weeks'])}).", "val")
            fig = go.Figure(go.Bar(x=dates, y=d.values, marker_color=[RIVAL if v > 0 else OURS for v in d.values],
                                   hovertemplate="setmana del %{x}<br>%{y:+.2f} punts<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=INK, width=1))
            fig = ui.style(fig, 240, legend=False)
            fig.update_xaxes(type="category", showticklabels=False, showline=False, ticks="")
            fig.update_yaxes(tickformat="+.1f", side="right")
            ui.show(fig)
            gates, concl, sub = models.gates(reg, T, runs, cand)
            ui.heading("Condicions de la regla de decisió", "Cada condició es calcula amb les dades de l'aspirant "
                       "seleccionat; no n'hi ha cap d'escrita a mà.", "val")
            ui.checklist(gates)

    with ui.more("Taula de validació (3 plecs)"):
        t = T.reset_index()
        t["name"] = [_label(n, reg) for n in t["strategy"]]
        t["win_rate"] = 100 * t["wins"] / t["weeks"]
        f2, s2 = (lambda v: fmt(v, 2)), (lambda v: fmt(v, 2, True))
        ui.table(t, names={"name": "Estratègia", "mean": "Mitjana 3 plecs", "fold1": "Plec 1", "fold2": "Plec 2", "fold3": "Plec 3",
                           "delta": f"Resp. {models.REF}", "worst_fold": "Pitjor plec", "win_rate": "Setmanes guanyades",
                           "infr": "Infraccions"},
                 formats={"mean": f2, "fold1": f2, "fold2": f2, "fold3": f2, "delta": s2, "worst_fold": s2,
                          "win_rate": lambda v: f"{fmt(v, 0)} %", "infr": lambda v: str(int(v))},
                 left=("name",), bars={"win_rate": 100}, highlight=lambda r: r["strategy"] == models.REF, height=430)
        orc = raw["oracle"]
        ui.source("<br>El plec k aparta la k-èsima setmana comptant des del final de cada bloc mensual de les dades d'entrenament. "
                  "Òptim amb previsió perfecta: " + " · ".join(f"plec {k}, {fmt(orc[k]['score'], 2)}" for k in sorted(orc)) + ".")
