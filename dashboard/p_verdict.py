"""Pàgina 1 - Quin agent lliurem? Historial públic, validació contra públic i els aspirants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import models
import ui
from ui import BAD, GOOD, GRAY, GRAY_LIGHT, INK, INK_2, INK_3, OURS, RIVAL, fmt


def _label(name: str, reg: dict) -> str:
    for s in reg["submissions"]:
        if s.get("arena") == name and s["id"] != name:
            return f"{name} (= {s['id']})"
    return name


def render():
    reg = models.registry()
    subs = pd.DataFrame(reg["submissions"])
    me = reg["selected"]
    sel = subs.set_index("id").loc[me]
    scored = subs.dropna(subset=["public"]).reset_index(drop=True)
    pending = subs[subs["public"].isna()]
    lead = reg["leaderboard"]["teams"][0]
    prev_best = scored[scored["id"] != me]["public"].max()
    prev_id = scored.loc[scored["public"] == prev_best, "id"].iloc[-1]
    raw, runs = models.arena_raw(), models.runs()
    T = W = None
    if raw:
        T, W = models.arena_table(raw)
    pt, lim = reg["public_test"], reg["limits"]
    opt1 = raw["oracle"]["1"]["score"] if raw else None
    both = scored.dropna(subset=["validation"])
    fooled = both[(both["validation"] > sel["validation"] - 0.2) & (both["public"] < sel["public"] - 0.2)]

    # ------------------------------------------------------------------ entrada: la decisió
    c1, c2 = st.columns([7, 5], gap="large")
    n_f = len(fooled)
    c1.markdown(
        f"""<div class='lead'>
          <div><div class='n'>{fmt(sel['public'], 2)}</div><div class='u'>punts al rànquing públic</div></div>
          <div><p class='h'>Lliurem la {me}: és la puntuació més alta que hem obtingut al test real.</p>
          <p class='t'>{'Dues' if n_f == 2 else n_f} versions posteriors tenien millor validació i van puntuar
          menys un cop enviades. Per això decidim amb el resultat públic i fem servir la validació
          només per triar què val la pena enviar.</p></div></div>""",
        unsafe_allow_html=True,
    )
    co2_t = pt["co2_base_kg"] * sel["co2"] / 100 / 1000
    eur_k = pt["cost_base_eur"] * sel["cost"] / 100 / 1000
    yr = 8760 / pt["hours"]
    ms = (runs.get(me) or {}).get("ms")
    run_s = ms * lim["test_hours"] / 1000 if ms else None
    wks = pt["hours"] // 168
    ui.rows([
        ("CO₂ evitat", f"70 % de la nota · {fmt(co2_t, 1)} t en {wks} setmanes, unes {fmt(co2_t * yr, 0)} t l'any a aquest ritme",
         f"{fmt(sel['co2'], 2)} <small>%</small>", ""),
        ("Factura estalviada", f"30 % de la nota · {fmt(eur_k, 1)} k€ en {wks} setmanes, uns {fmt(eur_k * yr, 0)} k€ l'any",
         f"{fmt(sel['cost'], 2)} <small>%</small>", ""),
        (f"Respecte a la {prev_id}", "el nostre millor lliurament anterior", fmt(sel["public"] - prev_best, 2, True), ""),
        ("Distància al líder", f"{lead['team']}, {fmt(lead['score'], 2)} ({reg['leaderboard']['as_of']})",
         fmt(sel["public"] - lead["score"], 2, True), ""),
        ("Marge fins a l'òptim", f"validació: {fmt(sel['validation'], 2)} de {fmt(opt1, 2)} amb previsió perfecta",
         f"{fmt(opt1 - sel['validation'], 2)} <small>punts</small>" if opt1 else "—", ""),
        ("Temps d'execució", f"{fmt(lim['test_hours'])} hores de test, límit {lim['timeout_s']} s; mesurat en aquesta màquina",
         f"{fmt(run_s, 0)} <small>s</small>" if run_s else "—", "bad" if run_s and run_s > lim["timeout_s"] else ""),
        ("Infraccions", "cada una resta 1 punt", str(int(sel["infr"])), ""),
    ], where=c2)

    # ------------------------------------------------------------------ historial públic
    ui.section("Què ens ha dit el rànquing fins ara")
    c1, c2 = st.columns([6, 5], gap="large")
    with c1:
        regress = scored["public"] < scored["public"].cummax().shift(1)
        box = ui.panel(f"{len(scored)} lliuraments, {int(regress.sum())} passos enrere: la {me} és el màxim",
                       "Puntuació pública de cada versió del Mode Difícil, en l'ordre en què les vam enviar. "
                       "En vermell, les que van quedar per sota de la millor anterior.")
        colors = [OURS if i == me else (RIVAL if r else GRAY) for i, r in zip(scored["id"], regress)]
        fig = go.Figure()
        fig.add_scatter(x=scored["id"], y=scored["public"], mode="lines", line=dict(color=GRAY_LIGHT, width=1.5),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=scored["id"], y=scored["public"], mode="markers+text", showlegend=False,
                        marker=dict(color=colors, size=11), text=[fmt(v, 2) for v in scored["public"]],
                        textposition="top center", textfont=dict(size=12, color=INK),
                        customdata=scored["what"], hovertemplate="<b>%{x}</b> · %{y:.2f}<br>%{customdata}<extra></extra>")
        fig.add_hline(y=lead["score"], line=dict(color=INK, width=1, dash="dash"))
        fig.add_annotation(x=0, xref="paper", y=lead["score"], text=f"líder: {fmt(lead['score'], 2)}", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(color=INK, size=12))
        fig.update_yaxes(range=[scored["public"].min() - 0.5, lead["score"] + 0.6], side="right")
        fig.update_xaxes(type="category")
        ui.show(ui.style(fig, 330, legend=False), box)
        if len(pending):
            ui.source("Sense puntuació pública encara: " + "; ".join(f"{r.id} ({r.what})" for r in pending.itertuples()) + ".", box)
    with c2:
        box = ui.panel("Més validació no ha volgut dir més puntuació pública",
                       "Cada punt és una versió: on quedava a la nostra validació (horitzontal) i què li va "
                       "donar el test (vertical). Si la validació fos fiable, els punts pujarien cap a la dreta.")
        fig = go.Figure()
        for _, r in both.iterrows():
            col = OURS if r["id"] == me else (RIVAL if r["id"] in set(fooled["id"]) else GRAY)
            fig.add_scatter(x=[r["validation"]], y=[r["public"]], mode="markers+text", showlegend=False,
                            marker=dict(color=col, size=12), text=[r["id"]],
                            textposition={"v9": "top center", "v8": "bottom left"}.get(
                                r["id"], "middle left" if r["id"] == me else "middle right"),
                            textfont=dict(size=13, color=INK),
                            hovertemplate=f"<b>{r['id']}</b><br>validació {fmt(r['validation'], 2)}<br>públic "
                                          f"{fmt(r['public'], 2)}<br>{r['what']}<extra></extra>")
        for r in pending.dropna(subset=["validation"]).itertuples():
            fig.add_vline(x=r.validation, line=dict(color=INK_3, width=1, dash="dot"))
            fig.add_annotation(x=r.validation, y=0, yref="paper", text=f"{r.id}: sense puntuar", showarrow=False,
                               xanchor="right", yanchor="bottom", xshift=-4, font=dict(color=INK_2, size=11.5))
        fig.update_xaxes(title="validació (plec 1)", range=[both["validation"].min() - 0.15, both["validation"].max() + 0.2])
        fig.update_yaxes(title="públic")
        ui.show(ui.style(fig, 330, legend=False), box)
    if len(fooled):
        txt = " i ".join(f"la <b>{r.id}</b> (validació {fmt(r.validation - sel['validation'], 2, True)}, "
                         f"públic {fmt(r.public - sel['public'], 2, True)} respecte a la {me})"
                         for r in fooled.itertuples())
        ui.note("Què n'extraiem.",
                f"El cas de {txt} explica la regla d'aquesta pàgina: un guany d'unes dècimes a la validació cap "
                "dins del que el test pot treure. Les setmanes de test són els darrers set dies de cada mes, "
                "amb Nadal i finals d'agost inclosos, i cap plec de validació les reprodueix del tot.")

    if T is None:
        st.info("Executeu `python dashboard/build_models.py` per afegir l'anàlisi dels aspirants.")
        return

    # ------------------------------------------------------------------ aspirants
    ui.section(f"Hi ha cap aspirant prou bo per substituir la {models.REF}?",
               f"Totes les estratègies provades, comparades amb la {models.REF} sobre les mateixes setmanes de validació "
               "(tres plecs, models reentrenats sense aquestes setmanes).")
    n_weeks = int(T["weeks"].max())
    main = T[(T["wk_delta"] > -0.6) & (T.index != models.REF)].sort_values("wk_delta")
    out = [n for n in T.index if n not in main.index and n != models.REF]
    c1, c2 = st.columns([6, 5], gap="large")
    with c1:
        n_sig = int(((main["wk_delta"] > 0) & main["significant"]).sum())
        box = ui.panel(f"De {len(main)} estratègies, {n_sig} superen la {models.REF} més enllà del soroll; "
                       f"la millor, per {fmt(main['wk_delta'].max(), 2)} punts",
                       f"Diferència mitjana de puntuació setmanal en {n_weeks} setmanes. La barra és l'interval "
                       "del 95 %: si toca el zero, la diferència pot ser atzar.")
        col = [RIVAL if (s and d > 0) else (INK if s else GRAY_LIGHT) for s, d in zip(main["significant"], main["wk_delta"])]
        fig = go.Figure(go.Scatter(
            x=main["wk_delta"], y=[_label(n, reg) for n in main.index], mode="markers", showlegend=False,
            marker=dict(color=col, size=9), error_x=dict(type="data", array=main["ci"], color=GRAY, thickness=1.2, width=0),
            customdata=np.c_[main["wins"], main["weeks"], main["mean"]],
            hovertemplate="<b>%{y}</b><br>%{x:+.2f} punts per setmana respecte a la v12<br>millor en %{customdata[0]} de "
                          "%{customdata[1]} setmanes<br>mitjana de 3 plecs %{customdata[2]:.2f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=OURS, width=2.5))
        fig.add_annotation(x=0, y=0, yref="paper", text=f"<b>{models.REF}</b>", showarrow=False, yanchor="top",
                           font=dict(color=OURS, size=12))
        fig.update_xaxes(showgrid=True, gridcolor=ui.GRID, tickformat="+.1f", side="top", showline=False, ticks="")
        fig.update_yaxes(showgrid=False, tickfont=dict(size=11.5, color=INK))
        fig = ui.style(fig, max(380, 19 * len(main) + 70), legend=False)
        fig.update_xaxes(showgrid=True, showline=False, ticks="")
        fig.update_layout(margin=dict(b=22))
        ui.show(fig, box)
        if out:
            ui.source("Fora d'escala: " + ", ".join(f"{n} ({fmt(T.loc[n, 'wk_delta'], 1, True)})" for n in out) + ".", box)
    with c2:
        opts = [n for n in T.index if n != models.REF]
        dflt = next((s["arena"] for s in reg["submissions"] if s.get("public") is None and s.get("arena") in opts), opts[0])
        cand = st.selectbox("Aspirant que voleu examinar", opts, index=opts.index(dflt), format_func=lambda n: _label(n, reg))
        r = T.loc[cand]
        d = (W.loc[cand] - W.loc[models.REF]).dropna()
        dates = [pd.Timestamp(c.split("|")[1]).strftime("%d/%m/%y") for c in d.index]
        box = ui.panel(f"{_label(cand, reg)}: millor en {int(r['wins'])} de {int(r['weeks'])} setmanes, "
                       f"{fmt(r['wk_delta'], 2, True)} de mitjana",
                       f"Puntuació setmanal menys la de la {models.REF}. Vermell: guanya l'aspirant. Blau: guanya la {models.REF}.")
        fig = go.Figure(go.Bar(x=dates, y=d.values, marker_color=[RIVAL if v > 0 else OURS for v in d.values],
                               hovertemplate="setmana del %{x}<br>%{y:+.2f} punts<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=INK, width=1))
        fig = ui.style(fig, 290, legend=False)
        fig.update_xaxes(type="category", tickangle=-90, tickfont=dict(size=9.5), showline=False, ticks="")
        fig.update_yaxes(tickformat="+.1f", side="right")
        ui.show(fig, box)

        gates, concl, sub = models.gates(reg, T, runs, cand)
        lis = "".join(f"<li><span class='m' style='color:{GOOD if ok else BAD}'>{'sí' if ok else 'no'}</span>"
                      f"<span>{name}</span><span class='d'>{detail}</span></li>" for name, ok, detail in gates)
        st.markdown("<p class='ct' style='margin-top:.6rem'>Condicions perquè una versió substitueixi l'actual</p>"
                    f"<ul class='chk'>{lis}</ul>", unsafe_allow_html=True)
        ui.note(concl, sub["note"] if sub and sub.get("note") else "")

    # ------------------------------------------------------------------ taula
    ui.section("Tots els resultats de validació")
    tab = T.reset_index()
    tab["name"] = [_label(n, reg) for n in tab["strategy"]]
    tab["win_rate"] = 100 * tab["wins"] / tab["weeks"]

    def f2(v):
        return fmt(v, 2)

    def s2(v):
        return fmt(v, 2, True)

    ui.table(tab,
             names={"name": "Estratègia", "mean": "Mitjana 3 plecs", "fold1": "Plec 1", "fold2": "Plec 2", "fold3": "Plec 3",
                    "delta": f"Resp. {models.REF}", "worst_fold": "Pitjor plec", "win_rate": "Setmanes guanyades",
                    "co2": "CO₂ evitat %", "cost": "Cost estalviat %", "infr": "Infraccions"},
             formats={"mean": f2, "fold1": f2, "fold2": f2, "fold3": f2, "delta": s2, "worst_fold": s2,
                      "win_rate": lambda v: f"{fmt(v, 0)} %", "co2": f2, "cost": f2, "infr": lambda v: str(int(v))},
             left=("name",), bars={"win_rate": 100}, highlight=lambda r: r["strategy"] == models.REF, height=430)
    orc = raw["oracle"]
    ui.source("<br>El plec k aparta la k-èsima setmana comptant des del final de cada bloc mensual de les dades d'entrenament. "
              "Òptim amb previsió perfecta: " + " · ".join(f"plec {k}, {fmt(orc[k]['score'], 2)}" for k in sorted(orc)) + ".")
