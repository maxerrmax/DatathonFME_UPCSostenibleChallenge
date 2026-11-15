"""Pàgina 2 - Dos agents cara a cara, hora a hora, contra l'òptim amb previsió perfecta."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import live
import models
import ui
from ui import C_CO2, C_SOLAR, GRAY, GRAY_LIGHT, INK, INK_2, OPT, OURS, RIVAL, fmt

CAP, PMAX = 6000, 1500
OPT_NAME = "òptim"


def render():
    tr, runs, reg = models.traces(), models.runs(), live.load().registry
    if tr is None or not runs:
        st.info("Executeu `python dashboard/build_models.py` un cop per simular els agents hora a hora.")
        return
    names = [n for n in runs if n != "optimum"]
    lim = reg["limits"]

    def lab(n):
        return n if runs[n]["label"] == n else f"{n} · {runs[n]['label']}"

    def nm(m):
        return OPT_NAME if m == "optimum" else m

    idx = tr[tr["model"] == "optimum"].index
    ui.intro("Dos agents, les mateixes hores",
             f"Validació, plec 1: {fmt(len(idx))} hores que els models no han vist, amb el simulador oficial. "
             "L'òptim és el pla que faria algú que conegués el futur.")
    c1, c2, _ = st.columns([3, 3, 4])
    a = c1.selectbox("Agent A (verd)", names, index=names.index(models.REF) if models.REF in names else 0, format_func=lab)
    rest = [n for n in names if n != a]
    b = c2.selectbox("Agent B (taronja)", rest, index=0, format_func=lab)
    T = {m: tr[tr["model"] == m] for m in (a, b, "optimum")}
    col = {a: OURS, b: RIVAL, "optimum": OPT}
    w = models.score_weights(tr)
    ra, rb, ro = runs[a], runs[b], runs["optimum"]
    hour = idx.hour
    imp = {m: T[m]["red_kw"].to_numpy() for m in T}
    lost = {m: pd.Series(w.to_numpy() * (imp[m] - imp["optimum"])).groupby(hour).sum() for m in (a, b)}

    # ------------------------------------------------------------------ marcador + on es perden els punts
    c1, c2 = st.columns([5, 6], gap="large")
    with c1:
        secs = lambda r: r["ms"] * lim["test_hours"] / 1000 if r.get("ms") else None
        spec = [("Puntuació", lambda r: r["score"], 2), ("CO₂ evitat, %", lambda r: r["co2"], 2),
                ("Factura estalviada, %", lambda r: r["cost"], 2), ("Infraccions", lambda r: r["infr"], 0),
                (f"Temps de test, s (límit {lim['timeout_s']})", secs, 0)]
        rows_ = []
        for name, f, dec in spec:
            va, vb, vo = f(ra), f(rb), f(ro)
            rows_.append({"k": name, "a": fmt(va, dec), "b": fmt(vb, dec),
                          "d": fmt(va - vb, dec, True) if va is not None and vb is not None else "—", "o": fmt(vo, dec)})
        win = a if ra["score"] >= rb["score"] else b
        box = ui.panel(f"Guanya la {win} per {fmt(abs(ra['score'] - rb['score']), 2)} punts; a l'òptim n'hi falten "
                       f"{fmt(ro['score'] - max(ra['score'], rb['score']), 2)}")
        ui.table(pd.DataFrame(rows_), names={"k": "", "a": f"A · {a}", "b": f"B · {b}", "d": "A − B", "o": "Òptim"},
                 left=("k",), highlight=lambda r: r["k"] == "Puntuació", where=box)
        ui.source("<br>El temps s'extrapola dels ms per pas mesurats en aquesta màquina: cal llegir-lo com una proporció "
                  "entre agents, no com el temps del servidor.", box)
    with c2:
        pk = int(lost[a].idxmax())
        box = ui.panel(f"La {a} perd més punts cap a les {pk}:00",
                       "Punts perduts respecte a l'òptim per hora del dia, sumats a tot el plec. La suma de les barres és "
                       "exactament la distància de puntuació.")
        fig = go.Figure()
        for m in (a, b):
            fig.add_bar(x=lost[m].index, y=lost[m].values, name=m, marker_color=col[m],
                        hovertemplate=m + " · %{x}:00 · %{y:+.2f} punts<extra></extra>")
        fig.add_hline(y=0, line=dict(color=INK, width=1))
        fig.update_layout(barmode="group", bargap=0.2)
        fig.update_yaxes(tickformat="+.1f")
        fig = ui.style(fig, 270)
        fig.update_xaxes(title="hora del dia", dtick=2, showline=False, ticks="")
        ui.show(fig, box)

    # ------------------------------------------------------------------ una setmana
    ui.section("Una setmana, hora a hora")
    gid = (pd.Series(idx).diff() != pd.Timedelta("1h")).cumsum().to_numpy()
    starts = {g: idx[gid == g][0] for g in np.unique(gid)}
    gap_w = pd.Series(runs["optimum"]["weeks"]) - pd.Series(runs[a]["weeks"])
    default = next(g for g, s in starts.items() if str(s.date()) == gap_w.idxmax())
    g_sel = st.select_slider("Setmana (per defecte, la que més s'allunya de l'òptim)", options=list(starts), value=default,
                             format_func=lambda g: ui.data(starts[g]))
    m_ = gid == g_sel
    ctx = T["optimum"][m_]
    net = ctx["load_kw"] - ctx["pv_kw"]
    ws = {m: runs[m]["weeks"][str(starts[g_sel].date())] for m in (a, b, "optimum")}
    box = ui.panel(f"Setmana del {ui.data(starts[g_sel])}: {a}, {fmt(ws[a], 2)} · {b}, {fmt(ws[b], 2)} · "
                   f"òptim, {fmt(ws['optimum'], 2)}",
                   "A dalt, què costa en punts importar un kWh (1 = hora mitjana) i la demanda del campus neta de solar "
                   "(groc: excedent solar). A baix, l'energia que cada agent té a la bateria.")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.22, 0.28, 0.5])
    fig.add_scatter(x=ctx.index, y=w[m_] / w.mean(), mode="lines", line=dict(color=C_CO2, width=2, shape="hv"),
                    showlegend=False, hovertemplate="valor d'un kWh · %{y:.2f}×<extra></extra>", row=1, col=1)
    fig.add_hline(y=1, line=dict(color=GRAY_LIGHT, width=1), row=1, col=1)
    fig.add_scatter(x=ctx.index, y=net, mode="lines", line=dict(color=INK_2, width=1.5), showlegend=False,
                    hovertemplate="demanda neta · %{y:.0f} kW<extra></extra>", row=2, col=1)
    fig.add_scatter(x=ctx.index, y=net.clip(upper=0), mode="lines", line=dict(width=0), fill="tozeroy",
                    fillcolor=ui.rgba(C_SOLAR, 0.6), showlegend=False, hoverinfo="skip", row=2, col=1)
    fig.add_hline(y=0, line=dict(color=GRAY, width=1), row=2, col=1)
    for m, dash in ((a, None), (b, None), ("optimum", "dot")):
        t_ = T[m][m_]
        fig.add_scatter(x=t_.index, y=t_["soc_kwh"], mode="lines", name=nm(m),
                        line=dict(color=col[m], width=2, dash=dash),
                        customdata=t_["bateria_kw"],
                        hovertemplate=nm(m) + " · %{y:.0f} kWh (%{customdata:+.0f} kW)<extra></extra>", row=3, col=1)
    fig = ui.style(fig, 520)
    fig.update_xaxes(showline=False, ticks="", tickformat="%d/%m", hoverformat="%d/%m %H h")
    fig.update_xaxes(showline=True, ticks="outside", row=3, col=1)
    fig.update_yaxes(title_text="valor kWh", row=1, col=1)
    fig.update_yaxes(title_text="kW nets", row=2, col=1)
    fig.update_yaxes(title_text="bateria, kWh", range=[-150, CAP + 150], row=3, col=1)
    fig.update_layout(hovermode="x unified")
    ui.show(fig, box)

    # ------------------------------------------------------------------ detall
    with ui.more("Més detall: totes les setmanes i el perfil mitjà del dia"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            wk = pd.DataFrame({m: runs[m]["weeks"] for m in (a, b, "optimum")})
            x = [pd.Timestamp(d).strftime("%d/%m/%y") for d in wk.index]
            st.markdown(f"<p class='ct'>La bateria val {wk['optimum'].max() / max(wk['optimum'].min(), 0.1):.0f} vegades més a la "
                        "millor setmana que a la pitjor</p><p class='cc'>Puntuació de cada setmana de validació. La ratlla "
                        "és el pla òptim per al conjunt del plec; en una setmana concreta un agent el pot superar per poc.</p>",
                        unsafe_allow_html=True)
            fig = go.Figure()
            for m in (a, b):
                fig.add_scatter(x=x, y=wk[m], mode="markers", name=m, marker=dict(color=col[m], size=9),
                                hovertemplate=m + " · %{y:.2f}<extra></extra>")
            fig.add_scatter(x=x, y=wk["optimum"], mode="markers", name=OPT_NAME,
                            marker=dict(symbol="line-ew", size=15, line=dict(color=OPT, width=2)),
                            hovertemplate=OPT_NAME + " · %{y:.2f}<extra></extra>")
            fig.update_xaxes(type="category", tickangle=-45)
            fig.update_yaxes(rangemode="tozero", side="right")
            fig.update_layout(hovermode="x unified")
            ui.show(ui.style(fig, 290))
        with c2:
            st.markdown("<p class='ct'>Tots carreguen amb l'excedent solar del migdia i descarreguen al matí i al vespre</p>"
                        "<p class='cc'>Potència mitjana de la bateria a cada hora del dia; per sobre de zero, carrega.</p>",
                        unsafe_allow_html=True)
            fig = go.Figure()
            for m, dash in ((a, None), (b, None), ("optimum", "dot")):
                g = T[m]["bateria_kw"].groupby(hour).mean()
                fig.add_scatter(x=g.index, y=g.values, mode="lines", name=nm(m), line=dict(color=col[m], width=2, dash=dash),
                                hovertemplate=nm(m) + " · %{y:+.0f} kW<extra></extra>")
            fig.add_hline(y=0, line=dict(color=INK, width=1))
            fig.update_layout(hovermode="x unified")
            fig = ui.style(fig, 290)
            fig.update_xaxes(title="hora del dia", dtick=2, showline=False, ticks="")
            fig.update_yaxes(title="kW")
            ui.show(fig)
