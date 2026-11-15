"""Pàgina 5 - Prova-ho tu: programa la bateria amb un horari fix i mira si guanyes l'agent."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import models
import ui
from ui import C_CO2, C_COST, GRAY, GRAY_LIGHT, INK, OPT, OURS, RIVAL, fmt

HOURS = list(range(24))
C_CH, C_DIS = C_CO2, C_COST           # càrrega (verd blavós) i descàrrega (ocre): colors propis d'aquesta pàgina


@st.cache_data(show_spinner=False)
def _ctx() -> pd.DataFrame:
    return models.fold_context(models.traces())


@st.cache_data(show_spinner=False)
def _run(charge: tuple, discharge: tuple) -> dict:
    r = models.timetable_run(_ctx(), charge, discharge)
    return {k: v for k, v in r.items() if k not in ("power", "soc")}


@st.cache_data(show_spinner="Cercant el millor horari fix …")
def best_timetable() -> tuple[tuple, tuple, float]:
    """Cerca local: canvia una hora cada cop (càrrega / repòs / descàrrega) mentre la puntuació millori."""
    state = {h: 0 for h in HOURS}

    def sc():
        return _run(tuple(h for h in HOURS if state[h] == 1), tuple(h for h in HOURS if state[h] == -1))["score"]

    cur, improved = sc(), True
    while improved:
        improved = False
        for h in HOURS:
            for v in (1, -1, 0):
                if v == state[h]:
                    continue
                old, state[h] = state[h], v
                s = sc()
                if s > cur + 1e-6:
                    cur, improved = s, True
                else:
                    state[h] = old
    return tuple(h for h in HOURS if state[h] == 1), tuple(h for h in HOURS if state[h] == -1), cur


def _optimum_profile() -> tuple[tuple, tuple]:
    tr = models.traces()
    o = tr[tr["model"] == "optimum"]
    g = o["bateria_kw"].groupby(o.index.hour).mean()
    return tuple(h for h in HOURS if g[h] > 250), tuple(h for h in HOURS if g[h] < -250)


def _set(charge, discharge):
    st.session_state["play_ch"], st.session_state["play_dis"] = list(charge), list(discharge)


def render():
    tr, runs = models.traces(), models.runs()
    if tr is None or not runs:
        st.info("Executeu `python dashboard/build_models.py` un cop per activar aquesta pàgina.")
        return
    ctx = _ctx()
    ref, opt = runs[models.REF], runs["optimum"]
    st.session_state.setdefault("play_ch", [12, 13, 14, 15])
    st.session_state.setdefault("play_dis", [19, 20, 21, 22])

    st.markdown(
        f"<div class='lead' style='margin-top:.8rem'><div><p class='h'>Creus que un simple temporitzador ho faria igual de bé?</p>"
        f"<p class='t' style='max-width:74ch'>Tria a quines hores carrega i a quines descarrega la bateria, cada dia igual. "
        f"El simulem sobre les mateixes {fmt(len(ctx))} hores de validació que l'agent i amb les regles oficials. "
        f"Et donem avantatge: el teu horari no comet mai infraccions ni malbarata energia exportant-la.</p></div></div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns([7, 5], gap="large")
    with c1:
        st.pills("Hores en què carrega (a 1.500 kW, fins que s'omple)", HOURS, selection_mode="multi", key="play_ch",
                 format_func=lambda h: f"{h}")
        st.pills("Hores en què descarrega (fins a cobrir la demanda del campus)", HOURS, selection_mode="multi",
                 key="play_dis", format_func=lambda h: f"{h}")
        b1, b2, b3, _ = st.columns([2, 3, 3, 2])
        b1.button("Buida", on_click=_set, args=((), ()), width="stretch")
        b2.button("Copia l'hora mitjana de l'òptim", on_click=_set, args=_optimum_profile(), width="stretch")
        bc, bd, _ = best_timetable()
        b3.button("El millor horari que hem trobat", on_click=_set, args=(bc, bd), width="stretch")
    ch = tuple(sorted(set(st.session_state["play_ch"]) - set(st.session_state["play_dis"])))
    dis = tuple(sorted(set(st.session_state["play_dis"]) - set(st.session_state["play_ch"])))
    both = sorted(set(st.session_state["play_ch"]) & set(st.session_state["play_dis"]))
    me = _run(ch, dis)
    diff = me["score"] - ref["score"]
    with c2:
        verdict = (f"Has guanyat la {models.REF} per {fmt(diff, 2)} punts" if diff > 0 else
                   f"El teu horari fa {fmt(me['score'], 2)} punts: {fmt(-diff, 2)} menys que la {models.REF}")
        box = ui.panel(verdict, "Puntuació sobre el plec 1 de validació (0 = campus sense bateria).")
        names = ["El teu horari", f"Agent {models.REF}", "Òptim (previsió perfecta)"]
        vals = [me["score"], ref["score"], opt["score"]]
        fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[RIVAL, OURS, OPT], width=0.62,
                               text=[fmt(v, 2) for v in vals], textposition="outside", cliponaxis=False,
                               textfont=dict(size=15, color=INK), hoverinfo="skip"))
        fig = ui.style(fig, 170, legend=False)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=13, color=INK))
        fig.update_xaxes(range=[0, opt["score"] * 1.16], showticklabels=False, showline=False, ticks="")
        ui.show(fig, box)
        ui.source(f"CO₂ evitat: {fmt(me['co2'], 2)} % (la {models.REF}, {fmt(ref['co2'], 2)} %). Factura estalviada: "
                  f"{fmt(me['cost'], 2)} % (la {models.REF}, {fmt(ref['cost'], 2)} %). Cicles complets de bateria: {fmt(me['cycles'], 0)}."
                  + (f" Hores marcades als dos llocs i ignorades: {', '.join(map(str, both))}." if both else ""), box)

    c1, c2 = st.columns([7, 5], gap="large")
    with c1:
        g = (ctx["w"] / ctx["w"].mean()).groupby(ctx.index.hour)
        prof, q = g.mean(), g.quantile([0.1, 0.9]).unstack()
        box = ui.panel("El teu horari sobre el valor d'un kWh al llarg del dia",
                       "Línia: quant costa en punts importar un kWh a cada hora, de mitjana (1 = hora mitjana); banda: "
                       "del percentil 10 al 90. Franges: verd, carregues; ocre, descarregues. Un bon horari carrega a la vall "
                       "i descarrega als cims, però la banda diu que la vall no cau cada dia a la mateixa hora.")
        fig = go.Figure()
        for h in ch:
            fig.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor=C_CH, opacity=0.28, line_width=0)
        for h in dis:
            fig.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor=C_DIS, opacity=0.32, line_width=0)
        fig.add_scatter(x=q.index, y=q[0.9], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=q.index, y=q[0.1], fill="tonexty", fillcolor="rgba(18,18,18,0.10)", mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=prof.index, y=prof.values, mode="lines", line=dict(color=INK, width=2.5), showlegend=False,
                        hovertemplate="%{x}:00 · %{y:.2f}<extra></extra>")
        fig.add_hline(y=1, line=dict(color=GRAY, width=1, dash="dot"))
        fig = ui.style(fig, 280, legend=False)
        fig.update_xaxes(title="hora del dia", dtick=1, range=[-0.5, 23.5], tickfont=dict(size=11))
        fig.update_yaxes(side="right")
        ui.show(fig, box)
    with c2:
        wk = pd.DataFrame({"tu": me["weeks"], "ref": ref["weeks"], "opt": opt["weeks"]})
        wins = int((wk["tu"] > wk["ref"]).sum())
        x = [pd.Timestamp(d).strftime("%d/%m/%y") for d in wk.index]
        box = ui.panel(f"Guanyes la {models.REF} en {wins} de {len(wk)} setmanes",
                       "Puntuació de cada setmana de validació. Un horari fix s'ajusta a unes estacions i falla a "
                       "les altres; l'agent s'adapta a cada dia.")
        fig = go.Figure()
        for xi, a, b in zip(x, wk["tu"], wk["ref"]):
            fig.add_scatter(x=[xi, xi], y=[a, b], mode="lines", line=dict(color=GRAY_LIGHT, width=2), showlegend=False,
                            hoverinfo="skip")
        fig.add_scatter(x=x, y=wk["tu"], mode="markers", name="el teu horari", marker=dict(color=RIVAL, size=10),
                        hovertemplate="el teu horari · %{y:.2f}<extra></extra>")
        fig.add_scatter(x=x, y=wk["ref"], mode="markers", name=models.REF, marker=dict(color=OURS, size=10),
                        hovertemplate=models.REF + " · %{y:.2f}<extra></extra>")
        fig = ui.style(fig, 280)
        fig.update_xaxes(type="category", tickangle=-45)
        fig.update_yaxes(rangemode="tozero", side="right")
        fig.update_layout(hovermode="x unified")
        ui.show(fig, box)

    _, _, best = best_timetable()
    ui.note("Què en traiem.",
            f"El millor horari fix que hem trobat, buscat directament sobre aquestes mateixes setmanes i amb els avantatges "
            f"que et donem, arriba a <b>{fmt(best, 2)}</b> punts. La {models.REF} en fa <b>{fmt(ref['score'], 2)}</b> sense haver "
            f"vist mai aquestes setmanes: preveure el CO₂, el preu, el consum i la solar de cada dia val, com a mínim, "
            f"<b>{fmt(ref['score'] - best, 2)} punts</b>. Per a UPC Sostenible la conclusió és pràctica: programar la "
            f"bateria amb un rellotge deixaria sobre la taula el {fmt(100 * (1 - best / ref['score']), 0)} % de l'estalvi "
            "que aconsegueix l'agent.")
