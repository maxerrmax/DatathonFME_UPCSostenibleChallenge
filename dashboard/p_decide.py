"""
Pestanya «Decidir»: on perdem punts davant dels altres equips, quines palanques queden i com afecta el sol al CO₂.
Tot calculat a partir de `live.load()` (rànquing en directe) i de les dades del campus; els títols diuen la conclusió.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live
import models
import p_campus
import p_front
import ui
from p_front import SHORT
from ui import C_CO2, C_COST, C_SOLAR, GRAY, GRAY_LIGHT, INK, OURS, RIVAL, fmt

W_CO2, W_COST = 0.7, 0.3


def _gap_rows(stand: list, us: dict) -> list[dict]:
    """Per a cada referent: punts que ens falten, repartits entre CO₂ (70 %) i factura (30 %)."""
    def mean(rows, k):
        return float(np.mean([r[k] for r in rows]))

    groups = [
        (f"Líder · {stand[0]['team']}", stand[0]),
        ("Mitjana del top 5", stand[:5]),
        ("Mitjana del top 10", stand[:10]),
    ]
    out = []
    for name, g in groups:
        c = g["co2"] if isinstance(g, dict) else mean(g, "co2")
        k = g["cost"] if isinstance(g, dict) else mean(g, "cost")
        out.append(dict(name=name, co2=W_CO2 * (c - us["co2"]), cost=W_COST * (k - us["cost"])))
    return out


def _rivals(stand: list, us: dict, L: live.Live):
    rows = _gap_rows(stand, us)
    lead = rows[0]
    tot = lead["co2"] + lead["cost"]
    big_k, big_v = ("la factura", lead["cost"]) if lead["cost"] >= lead["co2"] else ("el CO₂", lead["co2"])
    t_ink = ui.DARK["ink"] if ui.is_dark() else ui.LIGHT["ink"]

    cl, cr = st.columns([1, 1], gap="small")
    with cl:
        box = ui.panel(
            f"{fmt(100 * big_v / tot, 0)} % de la distància al líder és {big_k}" if tot > 0 else "Anem per davant del líder",
            "Punts que ens falten (equips únics), repartits entre CO₂ (pes 0,7) i factura (pes 0,3).",
        )
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
        fig = ui.style(fig, 230, legend=True)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=t_ink, size=12))
        fig.update_xaxes(tickformat="+.1f", title="punts per recuperar")
        ui.show(fig, box)
    with cr:
        box = ui.panel(
            "Cap a on millorar: amunt i a la dreta",
            "Cada punt és un equip únic. La línia és la nostra nota: tot el que hi ha per sobre i a la dreta puntua més.",
        )
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
        fig = ui.style(fig, 230, legend=False)
        fig.update_xaxes(title="CO₂ evitat (%)", tickformat=".1f", range=[x0, x1])
        fig.update_yaxes(title="Factura estalviada (%)", tickformat=".1f", showgrid=False, range=[min(ys) - 0.4, max(ys) + 0.4])
        ui.show(fig, box)


def _levers(reg: dict, gap: float):
    pi = reg["perfect_info"]
    t_ink = ui.DARK["ink"] if ui.is_dark() else ui.LIGHT["ink"]
    items = [(SHORT[i] if i < len(SHORT) else p["what"][:24], p["gain"]) for i, p in enumerate(pi)]
    items.append(("Totes juntes (3 plecs)", 1.62))
    items.sort(key=lambda t: t[1])
    top = max(((n, g) for n, g in items if n != "Totes juntes (3 plecs)"), key=lambda t: t[1])

    cl, cr = st.columns([1.2, 1], gap="small")
    with cl:
        box = ui.panel(
            f"Prioritat: {top[0].lower()} (+{fmt(top[1], 2)} punts)",
            "Punts que guanyaríem a la validació (mitjana dels 3 plecs) amb informació perfecta. La línia puntejada és la distància al líder al públic.",
        )
        fig = go.Figure(
            go.Bar(
                y=[n for n, _ in items],
                x=[g for _, g in items],
                orientation="h",
                marker_color=[t_ink if n == "Totes juntes (3 plecs)" else RIVAL for n, _ in items],
                text=[f"+{fmt(g, 2)}" for _, g in items],
                textposition="outside",
                textfont=dict(color=t_ink, size=12),
                cliponaxis=False,
                hovertemplate="%{y}: +%{x:.2f} punts<extra></extra>",
            )
        )
        fig.add_vline(
            x=gap,
            line=dict(color=OURS, width=2, dash="dot"),
            annotation=dict(text=f"líder públic +{fmt(gap, 2)}", font=dict(color=OURS, size=11)),
            annotation_position="top",
        )
        fig = ui.style(fig, 240, legend=False)
        fig.update_layout(margin=dict(l=2, r=14, t=34, b=2))
        fig.update_yaxes(showgrid=False, tickfont=dict(color=t_ink))
        fig.update_xaxes(range=[0, max(g for _, g in items) * 1.22], tickformat=".1f")
        ui.show(fig, box)
    with cr:
        dropped = reg.get("dropped", [])
        box = ui.panel(f"{len(dropped)} idees ja provades sense èxit", "Resum ràpid; vegeu el detall a «On són els punts».")
        rows = pd.DataFrame(dropped)
        ui.table(rows, names={"idea": "Idea", "public": "Públic"}, left=("idea",), height=220, where=box)


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
    delta = 100 * (by.iloc[-1] / by.iloc[0] - 1)
    corr = float(noon["pv_kw"].corr(noon["co2_kg_per_kwh"]))

    cl, cr = st.columns([1.3, 1], gap="small")
    with cl:
        box = ui.panel(
            f"El CO₂ fa mínim a les {h_co2} h; el sol toca màxim a les {h_pv} h",
            "Mitjana per hora del dia: producció solar del campus (àrea) i CO₂ de la xarxa (línia).",
        )
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
        fig.add_vline(x=h_co2, line=dict(color=C_CO2, width=1, dash="dot"))
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
        word = "menys" if delta < 0 else "més"
        box = ui.panel(
            f"Amb molt sol la xarxa emet un {fmt(abs(delta), 0)} % {word} CO₂ al migdia",
            f"CO₂ mitjà de 10 a 16 h segons la producció solar (quartils). Correlació: {fmt(corr, 2)}.",
        )
        labels = ["Poc sol", "Mig-baix", "Mig-alt", "Molt sol"][: len(by)]
        fig = go.Figure(
            go.Bar(
                x=labels, y=by.values, marker_color=[GRAY_LIGHT] * (len(by) - 1) + [C_CO2],
                text=[fmt(v, 3) for v in by.values], textposition="outside",
                textfont=dict(color=t_ink, size=12), cliponaxis=False,
                hovertemplate="%{x}: %{y:.3f} kg/kWh<extra></extra>",
            )
        )
        fig = ui.style(fig, 300, legend=False)
        fig.update_yaxes(showgrid=False, visible=False, range=[0, by.max() * 1.18])
        fig.update_xaxes(title="producció solar", showline=False, ticks="")
        ui.show(fig, box)


def render():
    L = live.load()
    reg = L.registry
    stand_raw = [r for r in L.standings if r.get("co2") is not None and r.get("cost") is not None]
    stand = p_front.deduplicate_standings(stand_raw)
    us = next((r for r in stand if r["team"] == L.us), None)

    if not us or len(stand) < 5:
        ui.note("Sense rànquing en directe.", "Cal una font a <code>dashboard/live.py</code> per comparar-nos amb els altres equips.")
    else:
        ui.section("On perdem punts davant els altres")
        _rivals(stand, us, L)
        ui.section("Què millorar primer")
        _levers(reg, stand[0]["score"] - us["score"])
    ui.section("El sol i el CO₂")
    _sun(p_campus.get_data())