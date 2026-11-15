"""Pàgina 2 - Dos agents cara a cara, hora a hora, contra l'òptim amb previsió perfecta."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import models
import ui
from ui import C_CO2, C_SOLAR, GRAY, GRAY_LIGHT, INK, INK_2, OPT, OURS, RIVAL, fmt

CAP, PMAX = 6000, 1500
OPT_NAME = "òptim"


def render():
    tr, runs, reg = models.traces(), models.runs(), models.registry()
    if tr is None or not runs:
        st.info("Executeu `python dashboard/build_models.py` un cop per simular els agents hora a hora.")
        return
    names = [n for n in runs if n != "optimum"]
    lim = reg["limits"]

    def lab(n):
        return n if runs[n]["label"] == n else f"{n} · {runs[n]['label']}"

    def nm(m):
        return OPT_NAME if m == "optimum" else m

    c1, c2, c3 = st.columns([3, 3, 4])
    a = c1.selectbox("Agent A (blau)", names, index=names.index(models.REF) if models.REF in names else 0, format_func=lab)
    rest = [n for n in names if n != a]
    b = c2.selectbox("Agent B (vermell)", rest, index=0, format_func=lab)
    idx = tr[tr["model"] == "optimum"].index
    c3.markdown(f"<p class='cc' style='margin-top:1.7rem'>Validació, plec 1: els darrers 7 dies de cada bloc mensual de "
                f"les dades d'entrenament, {fmt(len(idx))} hores que els models no han vist. Bateria del Mode Difícil: "
                "6.000 kWh, ±1.500 kW, mercat conegut amb 3 h d'antelació.</p>", unsafe_allow_html=True)
    T = {m: tr[tr["model"] == m] for m in (a, b, "optimum")}
    col = {a: OURS, b: RIVAL, "optimum": OPT}
    w = models.score_weights(tr)
    ra, rb, ro = runs[a], runs[b], runs["optimum"]

    # ------------------------------------------------------------------ marcador
    ui.section("Marcador")
    c1, c2 = st.columns([5, 6], gap="large")
    secs = lambda r: r["ms"] * lim["test_hours"] / 1000 if r.get("ms") else None
    spec = [  # (nom, funció valor, decimals)
        ("Puntuació", lambda r: r["score"], 2),
        ("CO₂ evitat, %", lambda r: r["co2"], 2),
        ("Cost estalviat, %", lambda r: r["cost"], 2),
        ("CO₂ evitat, t", lambda r: r["co2_kg"] / 1000, 1),
        ("Factura estalviada, k€", lambda r: r["cost_eur"] / 1000, 1),
        ("Infraccions", lambda r: r["infr"], 0),
        ("Part de l'òptim, %", lambda r: 100 * r["score"] / ro["score"], 1),
        (f"Temps per a {fmt(lim['test_hours'])} h, s (límit {lim['timeout_s']})", secs, 0),
    ]
    rows_ = []
    for name, f, dec in spec:
        va, vb, vo = f(ra), f(rb), f(ro)
        rows_.append({"k": name, "a": fmt(va, dec), "b": fmt(vb, dec),
                      "d": fmt(va - vb, dec, True) if va is not None and vb is not None else "—",
                      "o": fmt(vo, dec)})
    with c1:
        ui.table(pd.DataFrame(rows_), names={"k": "", "a": f"A · {a}", "b": f"B · {b}", "d": "A − B", "o": "Òptim"},
                 left=("k",), highlight=lambda r: r["k"] == "Puntuació")
        ui.source("<br>El temps s'extrapola dels ms per pas mesurats en aquesta màquina amb un sol fil; el servidor pot "
                  "ser més ràpid o més lent, de manera que cal llegir-lo com una proporció entre agents.")
    with c2:
        wk = pd.DataFrame({m: runs[m]["weeks"] for m in (a, b, "optimum")})
        gap = wk["optimum"] - wk[a]
        x = [pd.Timestamp(d).strftime("%d/%m/%y") for d in wk.index]
        box = ui.panel(f"La bateria val {wk['optimum'].max() / max(wk['optimum'].min(), 0.1):.0f} vegades més a la millor "
                       "setmana que a la pitjor",
                       "Puntuació de cada setmana de validació per separat. La ratlla negra és el pla òptim per al "
                       "conjunt del plec, i per això en una setmana concreta un agent el pot superar per poc. "
                       "Al desembre gairebé no hi ha excedent solar per desplaçar.")
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
        ui.show(ui.style(fig, 290), box)
        ui.source(f"Setmana amb més distància a l'òptim per a la {a}: la del {ui.data(gap.idxmax())} "
                  f"({fmt(gap.max(), 1)} punts).", box)

    # ------------------------------------------------------------------ comportament
    ui.section("On actuen diferent els dos agents")
    hour = idx.hour
    imp = {m: T[m]["red_kw"].to_numpy() for m in T}
    lost = {m: pd.Series(w.to_numpy() * (imp[m] - imp["optimum"])).groupby(hour).sum() for m in (a, b)}
    c1, c2 = st.columns(2, gap="large")
    with c1:
        pk = int(lost[a].idxmax())
        box = ui.panel(f"La {a} perd més punts cap a les {pk}:00",
                       "Punts perduts respecte a l'òptim, per hora del dia, sumats a tot el plec. Una barra positiva és "
                       "el cost d'haver importat més que l'òptim en aquella hora; per sota de zero l'agent ha importat "
                       "menys (i ho ha pagat en una altra hora).")
        fig = go.Figure()
        for m in (a, b):
            fig.add_bar(x=lost[m].index, y=lost[m].values, name=m, marker_color=col[m],
                        hovertemplate=m + " · %{x}:00 · %{y:+.2f} punts<extra></extra>")
        fig.add_hline(y=0, line=dict(color=INK, width=1))
        fig.update_layout(barmode="group", bargap=0.2)
        fig.update_xaxes(title="hora del dia", dtick=2, showline=False, ticks="")
        fig.update_yaxes(tickformat="+.1f")
        fig = ui.style(fig, 300)
        fig.update_xaxes(showline=False, ticks="")
        ui.show(fig, box)
        ui.source(f"Total: la {a}, {fmt(lost[a].sum(), 2)} punts per sota de l'òptim ({fmt(ro['score'], 2)}); "
                  f"la {b}, {fmt(lost[b].sum(), 2)}. La suma de les barres és exactament la distància de puntuació.", box)
    with c2:
        box = ui.panel("Tots carreguen amb l'excedent solar del migdia i descarreguen a les puntes del matí i del vespre",
                       "Potència mitjana de la bateria a cada hora del dia; per sobre de zero, carrega. "
                       "L'òptim coneix el futur; els agents l'han de preveure.")
        fig = go.Figure()
        for m, dash in ((a, None), (b, None), ("optimum", "dot")):
            g = T[m]["bateria_kw"].groupby(hour).mean()
            fig.add_scatter(x=g.index, y=g.values, mode="lines", name=nm(m), line=dict(color=col[m], width=2, dash=dash),
                            hovertemplate=nm(m) + " · %{y:+.0f} kW<extra></extra>")
        fig.add_hline(y=0, line=dict(color=INK, width=1))
        fig.update_layout(hovermode="x unified")
        fig = ui.style(fig, 300)
        fig.update_xaxes(title="hora del dia", dtick=2, showline=False, ticks="")
        fig.update_yaxes(title="kW")
        ui.show(fig, box)

    # ------------------------------------------------------------------ una setmana
    ui.section("Una setmana, hora a hora")
    gid = (pd.Series(idx).diff() != pd.Timedelta("1h")).cumsum().to_numpy()
    starts = {g: idx[gid == g][0] for g in np.unique(gid)}
    gap_w = pd.Series(runs["optimum"]["weeks"]) - pd.Series(runs[a]["weeks"])
    default = next(g for g, s in starts.items() if str(s.date()) == gap_w.idxmax())
    g_sel = st.select_slider("Setmana", options=list(starts), value=default, format_func=lambda g: ui.data(starts[g]))
    m_ = gid == g_sel
    ctx = T["optimum"][m_]
    net = ctx["load_kw"] - ctx["pv_kw"]
    ws = {m: runs[m]["weeks"][str(starts[g_sel].date())] for m in (a, b, "optimum")}
    box = ui.panel(f"Setmana del {ui.data(starts[g_sel])}: {a}, {fmt(ws[a], 2)} · {b}, {fmt(ws[b], 2)} · "
                   f"òptim, {fmt(ws['optimum'], 2)}",
                   "A dalt, què costa en punts importar un kWh a cada hora (CO₂ i preu combinats; 1 = hora mitjana). "
                   "Després, la demanda del campus neta de solar, la potència de la bateria (positiva = càrrega) i "
                   "l'energia emmagatzemada.")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045, row_heights=[0.2, 0.22, 0.3, 0.28])
    sig = w[m_] / w.mean()
    fig.add_scatter(x=ctx.index, y=sig, mode="lines", line=dict(color=C_CO2, width=2, shape="hv"), showlegend=False,
                    hovertemplate="valor d'un kWh · %{y:.2f}×<extra></extra>", row=1, col=1)
    fig.add_hline(y=1, line=dict(color=GRAY_LIGHT, width=1), row=1, col=1)
    fig.add_scatter(x=ctx.index, y=net, mode="lines", line=dict(color=INK_2, width=1.5), showlegend=False,
                    hovertemplate="demanda neta · %{y:.0f} kW<extra></extra>", row=2, col=1)
    fig.add_scatter(x=ctx.index, y=net.clip(upper=0), mode="lines", line=dict(width=0), fill="tozeroy",
                    fillcolor=ui.rgba(C_SOLAR, 0.55), showlegend=False, hoverinfo="skip", row=2, col=1)
    fig.add_hline(y=0, line=dict(color=GRAY, width=1), row=2, col=1)
    for m, dash in ((a, None), (b, None), ("optimum", "dot")):
        t_ = T[m][m_]
        fig.add_scatter(x=t_.index, y=t_["bateria_kw"], mode="lines", name=nm(m), legendgroup=m,
                        line=dict(color=col[m], width=1.8, shape="hv", dash=dash),
                        hovertemplate=nm(m) + " · %{y:+.0f} kW<extra></extra>", row=3, col=1)
        fig.add_scatter(x=t_.index, y=t_["soc_kwh"], mode="lines", name=nm(m), legendgroup=m, showlegend=False,
                        line=dict(color=col[m], width=1.8, dash=dash),
                        hovertemplate=nm(m) + " · %{y:.0f} kWh<extra></extra>", row=4, col=1)
    fig = ui.style(fig, 640)
    fig.update_xaxes(showline=False, ticks="", tickformat="%d/%m", hoverformat="%d/%m %H h")
    fig.update_xaxes(showline=True, ticks="outside", row=4, col=1)
    fig.update_yaxes(title_text="valor kWh", row=1, col=1)
    fig.update_yaxes(title_text="kW nets", row=2, col=1)
    fig.update_yaxes(title_text="bateria, kW", range=[-PMAX * 1.08, PMAX * 1.08], row=3, col=1)
    fig.update_yaxes(title_text="energia, kWh", range=[-150, CAP + 150], row=4, col=1)
    fig.update_layout(hovermode="x unified")
    ui.show(fig, box)
    ui.source("Àrea groga: hores en què la solar supera la demanda del campus. Aquest excedent es perd si la bateria "
              "no té lloc, perquè l'energia abocada a la xarxa no es paga.", box)
