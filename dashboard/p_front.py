"""
Portada · «Quin agent lliurem?». Poca cosa, tot calculat, i en directe.

Ordre de lectura (patró en F): decisió (versió · nota · validació) → comparació amb el líder (CO₂, factura, impacte)
→ on som al rànquing i quin és el següent pas. Cap text fix: tot surt de `live.load()` i de `models.choose()`.
Tot el que no canvia una decisió és a les altres pestanyes. Les peces s'actualitzen soles (`st.fragment(run_every)`).
"""
from __future__ import annotations

from html import escape

import streamlit as st

import live
import models
import ui
from ui import fmt

SHORT = ["Demanda neta", "CO₂ a 3–11 h", "Preu a 3–11 h"]       # etiquetes curtes de `perfect_info`, per ordre


def _status(L: live.Live):
    if not L.ok:
        cls, txt = "bad", "Sense connexió"
    elif L.stale:
        cls, txt = "warn", f"Dades antigues · {live.ago(L.age_s)}"
    else:
        cls, txt = "ok", f"En directe · {live.ago(L.age_s)}"
    st.markdown(f"<div class='live {cls}'><i></i>{txt}<span>font: {escape(L.source)}</span></div>", unsafe_allow_html=True)


def _sparkline(hist, me, lead_score) -> str:
    W, H, p = 420, 74, 8
    lo, hi = min(hist["public"]), max(lead_score, max(hist["public"]))
    x = lambda i: p + i * (W - 2 * p) / max(1, len(hist) - 1)
    y = lambda v: H - p - (v - lo) / (hi - lo + 1e-9) * (H - 2 * p)
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(hist["public"]))
    dots = "".join(f"<circle class='{'ac' if i == me else 'mu'}' cx='{x(k):.1f}' cy='{y(v):.1f}' r='{4.5 if i == me else 2.6}'/>"
                   for k, (i, v) in enumerate(zip(hist["id"], hist["public"])))
    return (f"<svg class='ch spark' viewBox='0 0 {W} {H}'><line class='rv' x1='{p}' x2='{W - p}' y1='{y(lead_score):.1f}' y2='{y(lead_score):.1f}' "
            f"stroke-dasharray='3 4'/><polyline class='acs' stroke-width='1.8' points='{pts}'/>{dots}</svg>")


def _ring(pct: float) -> str:
    r, c = 38, 2 * 3.14159 * 38
    return (f"<svg class='ring' viewBox='0 0 100 100'><circle class='rbg' cx='50' cy='50' r='{r}'/>"
            f"<circle class='rfg' cx='50' cy='50' r='{r}' stroke-dasharray='{c * pct / 100:.1f} {c:.1f}' transform='rotate(-90 50 50)'/>"
            f"<text x='50' y='56' text-anchor='middle'>{fmt(pct, 0)}%</text></svg>")


def _bar(us: float, rival: float | None, vmax: float) -> str:
    tick = f"<i class='tick' style='left:{100 * rival / vmax:.1f}%'></i>" if rival else ""
    return f"<div class='cmp'><b style='width:{100 * us / vmax:.1f}%'></b>{tick}</div>"


def _delta(d: float) -> str:
    return ui.badge(fmt(abs(d), 2), "up" if d > 0 else "dn" if d < 0 else "nt")


@st.fragment(run_every=live.REFRESH_EVERY)
def render():
    L = live.load()
    reg = L.registry
    _status(L)
    ch = models.choose(reg)
    me = ch["chosen"]
    sel = {s["id"]: s for s in reg["submissions"]}[me]
    pt, raw = reg["public_test"], models.arena_raw()
    opt1 = raw["oracle"]["1"]["score"] if raw else None
    stand = L.standings
    lead = stand[0] if stand else reg["leaderboard"]["teams"][0]
    us = next((r for r in stand if r["team"] == L.us), None)
    hist = [s for s in reg["submissions"] if s.get("public") and s["public"] >= 22.5]
    import pandas as pd
    hist = pd.DataFrame(hist)
    gap = sel["public"] - lead["score"]
    top10 = stand[9]["score"] if len(stand) >= 10 else None

    # ---------------------------------------------------------------- 1 · decisió
    val_pct = 100 * sel["validation"] / opt1 if opt1 else None
    chips = ""
    if us:
        chips += f"<span class='chip'>{us['rank']}<small>/{L.n_teams or len(stand)}</small></span>"
    chips += f"<span class='chip'>{_delta(gap).replace('badge', 'badge big')}<small>líder</small></span>"
    if top10:
        chips += f"<span class='chip'>{_delta(sel['public'] - top10).replace('badge', 'badge big')}<small>top 10</small></span>"
    st.markdown(
        f"<div class='box hero'>"
        f"<div><div class='lab'>Lliurem</div><div class='ver'>{me}</div><div class='chips'>{chips}</div></div>"
        f"<div class='mid'><div class='lab'>Nota pública</div><div class='score'>{fmt(sel['public'], 2)}</div>"
        f"{_sparkline(hist, me, lead['score']) if len(hist) > 2 else ''}</div>"
        f"<div class='rg'>{_ring(val_pct) if val_pct else ''}<div class='lab'>Validació<br>vs òptim</div></div></div>",
        unsafe_allow_html=True)

    # ---------------------------------------------------------------- 2 · comparació amb el líder
    co2_t = pt["co2_base_kg"] * sel["co2"] / 100 / 1000
    eur_k = pt["cost_base_eur"] * sel["cost"] / 100 / 1000
    lco2, lcost = lead.get("co2"), lead.get("cost")
    cells = [
        ("CO₂ evitat · 70 %", fmt(sel["co2"], 2), "%", _bar(sel["co2"], lco2, (lco2 or sel["co2"]) * 1.06),
         _delta(sel["co2"] - lco2) if lco2 else "", f"líder {fmt(lco2, 2)}" if lco2 else ""),
        ("Factura estalviada · 30 %", fmt(sel["cost"], 2), "%", _bar(sel["cost"], lcost, (lcost or sel["cost"]) * 1.06),
         _delta(sel["cost"] - lcost) if lcost else "", f"líder {fmt(lcost, 2)}" if lcost else ""),
        ("Impacte al públic", fmt(co2_t, 1), "t CO₂", "", "", f"{fmt(eur_k, 1)} k€ de factura"),
    ]
    st.markdown("<div class='gap'></div><div class='box kp'>" + "".join(
        f"<div><div class='lab'>{a}</div><div class='v'>{b}<small> {u}</small>{d}</div>{bar}<div class='s'>{s}</div></div>"
        for a, b, u, bar, d, s in cells) + "</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------- 3 · rànquing + següent pas
    st.markdown("<div class='gap'></div>", unsafe_allow_html=True)
    cl, cr = st.columns([1.7, 1], gap="small")
    with cl:
        if not stand:
            ui.note("Sense rànquing.", "No hi ha cap font en directe. Vegeu <code>dashboard/live.py</code>.")
        else:
            n_ahead = sum(1 for r in stand if r["score"] > sel["public"] + 0.004)
            box = ui.panel(f"{n_ahead} per davant" + (f" · el top 10 és a {fmt(top10 - sel['public'], 2)}" if top10 else ""), key="lb")
            view = box.segmented_control("Equips", ["Al voltant nostre", "Top 10", f"Tots ({len(stand)})"], default="Al voltant nostre",
                                         label_visibility="collapsed", key="lb_view")
            idx = list(range(len(stand)))
            if view == "Top 10":
                idx = idx[:10]
            elif view in (None, "Al voltant nostre") and us:
                k = us["rank"] - 1
                idx = sorted(set(idx[:3]) | set(range(max(0, k - 4), min(len(stand), k + 5))))
            rows = [stand[i] for i in idx]
            lo, hi, X0, X1, rh = min(r["score"] for r in rows) - 0.1, lead["score"] + 0.06, 190, 610, 21
            xs = lambda v: X0 + (min(max(v, lo), hi) - lo) / (hi - lo) * (X1 - X0)
            g, y, prev = "", 12, None
            if us:
                g += f"<line class='me' x1='{xs(us['score'])}' x2='{xs(us['score'])}' y1='4' y2='{12 + rh * len(rows) - 6}' stroke-dasharray='2 4'/>"
            for r in rows:
                if prev is not None and r["rank"] - prev > 1:
                    g += f"<text x='{X0 - 12}' y='{y + 4}' text-anchor='end'>···</text>"; y += 12
                prev = r["rank"]
                m = r["team"] == L.us
                g += (f"<text class='{'hi' if m else ''}' x='0' y='{y + 4}'>{r['rank']:>2}  {escape(r['team'][:20])}</text>"
                      f"<line class='ln' x1='{X0}' x2='{xs(r['score']):.1f}' y1='{y}' y2='{y}'/>"
                      f"<circle class='{'ac' if m else 'rv' if r['rank'] == 1 else 'mu'}' cx='{xs(r['score']):.1f}' cy='{y}' r='{6 if m else 3.6}'/>"
                      f"<text class='{'hi' if m else ''}' x='{xs(r['score']) + 10:.1f}' y='{y + 4}'>{fmt(r['score'], 2)}</text>")
                y += rh
            box.markdown(f"<svg class='ch' viewBox='0 0 680 {y + 4}'>{g}</svg>", unsafe_allow_html=True)
    with cr:
        pend = ch["pending"][0] if ch["pending"] else None
        if pend:
            nxt = (f"<div class='lab'>Següent pas</div><div class='nx'><span class='chip2'>{pend['id']}</span>"
                   f"<span class='gt'>&gt; {fmt(pend['threshold'], 2)}</span></div>"
                   f"<div class='s'>Només substitueix la {me} si supera això al públic.</div>")
        else:
            nxt = f"<div class='lab'>Següent pas</div><div class='nx'><span class='gt'>—</span></div><div class='s'>Cap versió pendent.</div>"
        pot = reg["perfect_info"] + ([{"gain": opt1 - sel["validation"]}] if opt1 else [])
        vmax = max(p["gain"] for p in pot) if pot else 1
        bars = "".join(
            f"<div class='pb'><span>{SHORT[i] if i < len(reg['perfect_info']) else 'Fins a l’òptim'}</span>"
            f"<div><b class='{'full' if i >= len(reg['perfect_info']) else ''}' style='width:{100 * p['gain'] / vmax:.0f}%'></b></div>"
            f"<em>+{fmt(p['gain'], 2)}</em></div>" for i, p in enumerate(pot))
        st.markdown(f"<div class='box pad'>{nxt}</div><div class='gap'></div>"
                    f"<div class='box pad'><div class='lab'>Punts per guanyar</div><div class='pbs'>{bars}</div></div>",
                    unsafe_allow_html=True)
