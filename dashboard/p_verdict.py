"""Pàgina 1 - Quin agent lliurem? La tria és automàtica (models.choose) i pensa en el test privat."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import models
import ui
from ui import GRAY, GRAY_LIGHT, INK, INK_2, OURS, RIVAL, fmt


def _label(name: str, reg: dict) -> str:
    for s in reg["submissions"]:
        if s.get("arena") == name and s["id"] != name:
            return f"{name} (= {s['id']})"
    return name


def render():
    reg = models.registry()
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

    # ------------------------------------------------------------------ la decisió
    if ch["tie"]:
        other = ch["tie"][0]
        head = (f"Empata amb la {other} al rànquing públic ({fmt(sel['public'], 2)}) i hi afegeix un canvi que "
                "només pot sumar a la part privada del test, que és la que decideix.")
    else:
        head = f"És la versió amb la millor estimació per al test privat, amb {fmt(sel['public'], 2)} punts al públic."
    text = ("La tria la fa el dashboard, no nosaltres: per a cada versió estima l'avantatge al test privat combinant la "
            f"nota pública i la validació. <b>{len(fooled)} versions amb millor validació que la {anchor} van puntuar "
            "menys</b> un cop enviades, i per això cap de les dues evidències mana sola.")
    right = ui.battery(100 * sel["validation"] / opt1, f"de l'òptim amb previsió perfecta a la validació "
                       f"({fmt(sel['validation'], 2)} de {fmt(opt1, 2)} punts)") if opt1 else ""
    ui.hero(me, "versió que lliurem", "Tria automàtica", head, text, right)

    co2_t = pt["co2_base_kg"] * sel["co2"] / 100 / 1000
    eur_k = pt["cost_base_eur"] * sel["cost"] / 100 / 1000
    wks = pt["hours"] // 168
    ui.tiles([
        ("Rànquing públic", fmt(sel["public"], 2), f"sobre unes {wks} setmanes de test; la resta és privada", "navy"),
        ("CO₂ evitat · 70 % de la nota", f"{fmt(sel['co2'], 2)} <small>%</small>", f"{fmt(co2_t, 1)} t en {wks} setmanes", "green"),
        ("Factura estalviada · 30 %", f"{fmt(sel['cost'], 2)} <small>%</small>", f"{fmt(eur_k, 1)} k€ en {wks} setmanes", "ocre"),
        ("Distància al líder", fmt(sel["public"] - lead["score"], 2, True),
         f"{lead['team']}, {fmt(lead['score'], 2)} ({reg['leaderboard']['as_of']})", "sun"),
    ])
    src = me if me in runs else sel.get("extends") if isinstance(sel.get("extends"), str) else None
    ms = (runs.get(src) or {}).get("ms") if src else None
    if ms:
        ui.source(f"Temps d'execució: uns {fmt(ms * lim['test_hours'] / 1000, 0)} s per a les {fmt(lim['test_hours'])} hores "
                  f"de test (límit {lim['timeout_s']} s; mesurat en aquesta màquina"
                  + (f" amb la {src}, de la qual la {me} és una ampliació" if src != me else "") + f"). "
                  f"Infraccions: {int(sel['infr'])}.")

    # ------------------------------------------------------------------ com tria
    ui.section("Com tria el dashboard", "La nota pública més alta no és necessàriament el millor model: el rànquing només "
               "ensenya una part del test i una comparació hi té un soroll de ±" + fmt(ch["sd_pub"], 2) + " punts.")
    c1, c2 = st.columns([6, 5], gap="large")
    with c1:
        el = tab[tab["eligible"] & (tab["est"] > -1.2)].sort_values("est")
        n_out = int((tab["eligible"] & (tab["est"] <= -1.2)).sum())
        n_better = int((el["est"] > 0.005).sum())
        box = ui.panel((f"Cap versió enviada supera la {anchor}" if n_better == 0 else
                        f"{n_better} versions superen la {anchor}") + f"; triem la {me}",
                       f"Avantatge estimat al test privat respecte a la {anchor}, en punts. La barra és l'interval del 95 %. "
                       "Blau: la versió triada. Taronja buit: encara no enviada (només validació, no és elegible).")
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
        box = ui.panel(f"Un punt guanyat a la validació n'ha valgut {fmt(ch['beta'], 2)} al test, amb ±{fmt(ch['sd_val'], 2)} de soroll",
                       f"Cada punt és un lliurament: quant guanyava a la {anchor} a la validació (horitzontal) i quant li "
                       "va guanyar al rànquing (vertical). Línia blava: la relació que fem servir. Grisa: si la "
                       "validació es complís al peu de la lletra.")
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
        ui.source("En taronja, les versions que la validació donava per millors i el test va posar per sota.", box)

    for p in ch["pending"]:
        ui.note(f"Regla per a la {p['id']} (encara no enviada).",
                f"{p['what']}. Només amb la validació li estimem {fmt(p['pred'], 2, True)} punts, amb un interval de "
                f"±{fmt(1.96 * ch['sd_val'], 2)}: no n'hi ha prou per triar-la. <b>S'envia un cop i substitueix la {me} "
                f"només si fa més de {fmt(p['threshold'], 2)} al públic</b>; si no, ens quedem com estem. El llindar el "
                "calcula el dashboard i es mourà sol quan s'ompli la seva nota al registre.")
    ui.source("Què no està verificat: l'estimació es calibra amb només " + str(len(ch["fit"])) + " lliuraments i suposa que "
              "el test privat es comporta com el públic. Serveix per ordenar versions i fixar regles, no per predir la nota final.")

    # ------------------------------------------------------------------ detall
    with ui.more(f"Historial: els {len(scored)} lliuraments en ordre"):
        regress = scored["public"] < scored["public"].cummax().shift(1)
        st.markdown(f"<p class='cc'>Puntuació pública de cada versió del Mode Difícil. En taronja, els {int(regress.sum())} "
                    "passos enrere respecte a la millor anterior; la línia de punts és el líder.</p>", unsafe_allow_html=True)
        colors = [OURS if i == me else (RIVAL if r else GRAY) for i, r in zip(scored["id"], regress)]
        fig = go.Figure()
        fig.add_scatter(x=scored["id"], y=scored["public"], mode="lines", line=dict(color=GRAY_LIGHT, width=1.5),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=scored["id"], y=scored["public"], mode="markers+text", showlegend=False,
                        marker=dict(color=colors, size=12), text=[fmt(v, 2) for v in scored["public"]],
                        textposition="top center", textfont=dict(size=12, color=INK),
                        customdata=scored["what"], hovertemplate="<b>%{x}</b> · %{y:.2f}<br>%{customdata}<extra></extra>")
        fig.add_hline(y=lead["score"], line=dict(color=INK, width=1, dash="dot"))
        fig.update_yaxes(range=[scored["public"].min() - 0.5, lead["score"] + 0.6], side="right")
        fig.update_xaxes(type="category")
        ui.show(ui.style(fig, 300, legend=False))

    if T is None:
        st.info("Executeu `python dashboard/build_models.py` per afegir l'anàlisi dels aspirants.")
        return

    main = T[(T["wk_delta"] > -0.6) & (T.index != models.REF)].sort_values("wk_delta")
    with ui.more(f"Aspirants: {len(T) - 1} estratègies provades contra la {models.REF} a la validació"):
        c1, c2 = st.columns([6, 5], gap="large")
        with c1:
            n_sig = int(((main["wk_delta"] > 0) & main["significant"]).sum())
            st.markdown(f"<p class='ct'>{n_sig} superen la {models.REF} més enllà del soroll; la millor, per "
                        f"{fmt(main['wk_delta'].max(), 2)} punts</p><p class='cc'>Diferència mitjana de puntuació setmanal en "
                        f"{int(T['weeks'].max())} setmanes de validació; la barra és l'interval del 95 %.</p>", unsafe_allow_html=True)
            col = [RIVAL if (s and d > 0) else (INK_2 if s else GRAY_LIGHT) for s, d in zip(main["significant"], main["wk_delta"])]
            fig = go.Figure(go.Scatter(
                x=main["wk_delta"], y=[_label(n, reg) for n in main.index], mode="markers", showlegend=False,
                marker=dict(color=col, size=9), error_x=dict(type="data", array=main["ci"], color=GRAY, thickness=1.2, width=0),
                customdata=np.c_[main["wins"], main["weeks"]],
                hovertemplate="<b>%{y}</b><br>%{x:+.2f} punts per setmana<br>millor en %{customdata[0]} de "
                              "%{customdata[1]} setmanes<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=OURS, width=2.5))
            fig = ui.style(fig, max(380, 19 * len(main) + 70), legend=False)
            fig.update_xaxes(showgrid=True, gridcolor=ui.GRID, tickformat="+.1f", side="top", showline=False, ticks="")
            fig.update_yaxes(showgrid=False, tickfont=dict(size=11.5, color=INK))
            ui.show(fig)
        with c2:
            opts = [n for n in T.index if n != models.REF]
            cand = st.selectbox("Aspirant que voleu examinar", opts, format_func=lambda n: _label(n, reg))
            r = T.loc[cand]
            d = (W.loc[cand] - W.loc[models.REF]).dropna()
            dates = [pd.Timestamp(c.split("|")[1]).strftime("%d/%m/%y") for c in d.index]
            st.markdown(f"<p class='ct'>{_label(cand, reg)}: millor en {int(r['wins'])} de {int(r['weeks'])} setmanes</p>"
                        f"<p class='cc'>Puntuació setmanal menys la de la {models.REF}. Taronja: guanya l'aspirant.</p>",
                        unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=dates, y=d.values, marker_color=[RIVAL if v > 0 else OURS for v in d.values],
                                   hovertemplate="setmana del %{x}<br>%{y:+.2f} punts<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=INK, width=1))
            fig = ui.style(fig, 240, legend=False)
            fig.update_xaxes(type="category", showticklabels=False, showline=False, ticks="")
            fig.update_yaxes(tickformat="+.1f", side="right")
            ui.show(fig)
            gates, concl, sub = models.gates(reg, T, runs, cand)
            st.markdown("<p class='ct'>Condicions per substituir la versió triada</p>", unsafe_allow_html=True)
            ui.checklist(gates)
            ui.note(concl, sub["note"] if sub and sub.get("note") else "")

    with ui.more("Taula completa de validació (3 plecs)"):
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
