"""Pàgina 5 - Executa qualsevol agent.py del repositori amb el simulador oficial (Mode Estàndard) i el compara amb l'òptim."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

import sim
import ui
from data import load_hourly, sim_frame
from ui import BAD, C_CO2, C_COST, OPT, OURS, fmt


@st.cache_data(show_spinner=False)
def _frame(ini, fin) -> pd.DataFrame:
    df = load_hourly()
    return sim_frame(df[(df["fecha_dia"].dt.date >= ini) & (df["fecha_dia"].dt.date <= fin)])


@st.cache_data(show_spinner=False)
def run_agent_cached(path_str: str, mtime: float, ini, fin):
    return sim.run_agent(sim.load_agent(Path(path_str)), _frame(ini, fin))


@st.cache_data(show_spinner=False)
def run_lp_cached(ini, fin):
    return sim.optimal_lp(_frame(ini, fin))


def render():
    df_all = load_hourly()
    valid = df_all.loc[df_all["completa"], "fecha_dia"]
    d_min, d_max = valid.min().date(), valid.max().date()
    st.markdown("<p class='cc' style='margin-top:.6rem'>Mode Estàndard (2.000 kWh, ±500 kW, solar real, mercat conegut "
                "amb 24 h d'antelació). Qualsevol fitxer del repositori que defineixi <code>class BatteryAgent</code> es pot "
                "executar amb el <code>starter_kit/environment.py</code> oficial i comparar amb un programa lineal que "
                "coneix tot el període per endavant.</p>", unsafe_allow_html=True)
    agents = {k: v for k, v in sim.find_agents().items()
              if k.replace("\\", "/").startswith(("starter_kit/", "estandard/agents/"))}
    if not agents:
        st.info("No s'ha trobat cap agent del Mode Estàndard.")
        return
    keys = list(agents)
    dflt = next((k for k in keys if "mpc_lp" in k and "safe" not in k and "tuned" not in k and "v2" not in k), keys[0])
    c1, c2, c3 = st.columns([3, 3, 1])
    sel = c1.selectbox("Agent", keys, index=keys.index(dflt), format_func=lambda k: k.replace("\\", "/"))
    rng = c2.date_input("Període", value=(d_min, d_max), min_value=d_min, max_value=d_max, format="DD/MM/YYYY", key="sim_rng")
    s_ini, s_fin = rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (d_min, d_max)
    c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    if c3.button("Executa", type="primary", width="stretch"):
        st.session_state["sim_req"] = (sel, s_ini, s_fin)
    req = st.session_state.get("sim_req")
    if req is None:
        st.caption("Trieu un agent i un període i premeu Executa. Un any sencer triga uns segons.")
        return
    sel_r, ini_r, fin_r = req
    p = agents[sel_r]
    try:
        with st.spinner("Simulant hora a hora …"):
            res_a, tr_a = run_agent_cached(str(p), p.stat().st_mtime, ini_r, fin_r)
            res_o, tr_o = run_lp_cached(ini_r, fin_r)
    except Exception as e:  # l'agent de l'usuari pot fallar
        st.error(f"L'agent ha fallat: {type(e).__name__}: {e}")
        return

    web = max(0.0, 0.7 * res_a["co2_estalviat_pct"] + 0.3 * res_a["cost_estalviat_pct"] - res_a["infraccions"])
    ui.section(f"{sel_r.replace(chr(92), '/')}", f"{ui.data(ini_r)} – {ui.data(fin_r)} · {fmt(len(tr_a))} hores "
               f"simulades en {fmt(res_a['segundos'], 1)} s")
    ui.facts([
        ("Puntuació (regla oficial, −1 per infracció)", fmt(web, 2),
         f"òptim: {fmt(res_o['score'], 2)}; n'obté el {fmt(100 * web / res_o['score'] if res_o['score'] else 0, 0)} %"),
        ("CO₂ evitat", f"{fmt(res_a['co2_estalviat_pct'], 2)} <small>%</small>",
         f"{fmt(res_a['co2_estalviat_kg'] / 1000, 1)} t; òptim: {fmt(res_o['co2_estalviat_pct'], 2)} %"),
        ("Factura estalviada", f"{fmt(res_a['cost_estalviat_pct'], 2)} <small>%</small>",
         f"{fmt(res_a['cost_estalviat_eur'] / 1000, 1)} k€; òptim: {fmt(res_o['cost_estalviat_pct'], 2)} %"),
        ("Infraccions", f"<span style='color:{BAD if res_a['infraccions'] else ui.INK}'>{fmt(res_a['infraccions'])}</span>",
         "consignes fora dels límits de potència o de capacitat"),
    ], cols=4)

    days = sorted(tr_a.index.normalize().unique())
    start = st.select_slider("Setmana que comença el", options=days, value=days[0],
                             format_func=lambda d: f"{ui.DIES[pd.Timestamp(d).dayofweek]} {ui.data(d)}") if len(days) > 7 else days[0]
    w0 = pd.Timestamp(start)
    w1 = w0 + pd.Timedelta(days=7)
    wa, wo = tr_a[(tr_a.index >= w0) & (tr_a.index < w1)], tr_o[(tr_o.index >= w0) & (tr_o.index < w1)]
    wm = df_all.loc[wa.index]
    box = ui.panel("Què fa l'agent i què faria l'òptim",
                   "CO₂ i preu de la xarxa (a dalt), potència de la bateria (positiva = càrrega) i energia emmagatzemada.")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045, row_heights=[0.2, 0.2, 0.3, 0.3])
    fig.add_scatter(x=wm.index, y=wm["co2_kg_per_kwh"], mode="lines", line=dict(color=C_CO2, width=2), showlegend=False,
                    hovertemplate="CO₂ · %{y:.3f} kg/kWh<extra></extra>", row=1, col=1)
    fig.add_scatter(x=wm.index, y=wm["precio_eur_per_kwh"], mode="lines", line=dict(color=C_COST, width=2), showlegend=False,
                    hovertemplate="preu · %{y:.3f} €/kWh<extra></extra>", row=2, col=1)
    for name, t_, colr, dash in (("agent", wa, OURS, None), ("òptim", wo, OPT, "dot")):
        fig.add_scatter(x=t_.index, y=t_["bateria_kw"], mode="lines", name=name, legendgroup=name,
                        line=dict(color=colr, width=1.8, shape="hv", dash=dash),
                        hovertemplate=name + " · %{y:+.0f} kW<extra></extra>", row=3, col=1)
        fig.add_scatter(x=t_.index, y=t_["soc_kwh"], mode="lines", name=name, legendgroup=name, showlegend=False,
                        line=dict(color=colr, width=1.8, dash=dash),
                        hovertemplate=name + " · %{y:.0f} kWh<extra></extra>", row=4, col=1)
    inf = wa[wa["infracciones"] > 0]
    if len(inf):
        fig.add_scatter(x=inf.index, y=inf["soc_kwh"], mode="markers", name="infracció",
                        marker=dict(symbol="x", size=9, color=BAD), row=4, col=1)
    fig = ui.style(fig, 580)
    fig.update_xaxes(showline=False, ticks="", tickformat="%d/%m", hoverformat="%d/%m %H h")
    fig.update_xaxes(showline=True, ticks="outside", row=4, col=1)
    fig.update_yaxes(title_text="kg/kWh", row=1, col=1)
    fig.update_yaxes(title_text="€/kWh", row=2, col=1)
    fig.update_yaxes(title_text="bateria, kW", range=[-560, 560], row=3, col=1)
    fig.update_yaxes(title_text="energia, kWh", range=[-50, 2050], row=4, col=1)
    fig.update_layout(hovermode="x unified")
    ui.show(fig, box)
