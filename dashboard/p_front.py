"""
Portada «Quin agent lliurem?». Disseny editorial:
  * el veredicte és l'únic bloc fosc; la nota i la seva evolució van directament sobre el fons;
  * els indicadors són text tipogràfic amb una línia fina a sobre;
  * el rànquing és una taula densa amb opció de deduplicar equips i sense solapament entre barres i xifres.
"""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

import live
import models
import ui
from ui import fmt

SHORT = ["Demanda neta", "CO₂ a 3–11 h", "Preu a 3–11 h"]


def _base_team(name: str) -> str:
    """Agrupa entrades múltiples d'un mateix equip al rànquing públic (ex. SanHackobos_v7 -> SanHackobos)."""
    n = name.split("_v")[0].strip()
    if n == "SSA Gourme":
        return "SSA Gourmet"
    if n.lower() == "sanhackobos":
        return "SanHackobos"
    return n


def deduplicate_standings(stand: list[dict]) -> list[dict]:
    """Retorna la millor entrada per cada equip únic i en recalcula la posició real."""
    seen = set()
    out = []
    for r in stand:
        base = _base_team(r["team"])
        if base in seen:
            continue
        seen.add(base)
        out.append({**r, "team": base, "rank": len(out) + 1})
    return out


@st.fragment(run_every=live.REFRESH_EVERY)
def status():
    L = live.load()
    if not L.ok:
        cls, txt = "bad", "sense connexió amb el rànquing"
    elif L.stale:
        cls, txt = "warn", f"dades de {live.ago(L.age_s)}"
    else:
        cls, txt = "ok", f"en directe, actualitzat {live.ago(L.age_s)}"
    st.markdown(
        f"<div class='meta'>Equip JAMM · Repte UPC Sostenible · <span class='live {cls}'><i></i>{txt}"
        f" · {escape(L.source)}</span></div>",
        unsafe_allow_html=True,
    )


def _d(v: float, label: str = "") -> str:
    cls, arrow = ("up", "▲") if v > 0 else ("dn", "▼") if v < 0 else ("nt", "")
    return f"<span class='d {cls}'>{arrow} {fmt(abs(v), 2)}{' ' + label if label else ''}</span>"


def _trend(hist: pd.DataFrame, me: str, lead_score: float) -> str:
    W, H, px0, px1, top, bot = 640, 176, 18, 616, 26, 28
    lo, hi = min(hist["public"]) - 0.1, max(lead_score, max(hist["public"])) + 0.05
    x = lambda i: px0 + i * (px1 - px0) / max(1, len(hist) - 1)
    y = lambda v: H - bot - (v - lo) / (hi - lo) * (H - top - bot)
    ids, vals = list(hist["id"]), list(hist["public"])
    line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    g = (
        f"<line class='rv' x1='{px0}' x2='{px1}' y1='{y(lead_score):.1f}' y2='{y(lead_score):.1f}' stroke-dasharray='3 4' stroke-width='1'/>"
        f"<text class='rvl' x='{px1}' y='{y(lead_score) - 6:.1f}' text-anchor='end'>líder {fmt(lead_score, 2)}</text>"
        f"<polyline class='acs' stroke-width='1.6' stroke-linejoin='round' points='{line}'/>"
    )
    for i, (vid, v) in enumerate(zip(ids, vals)):
        cur = vid == me
        g += f"<circle class='{'ac' if cur else 'mu'}' cx='{x(i):.1f}' cy='{y(v):.1f}' r='{4.5 if cur else 2.5}'/>"
        g += f"<text class='{'hi' if cur else ''}' x='{x(i):.1f}' y='{H - 8}' text-anchor='middle'>{escape(str(vid))}</text>"
        if cur:
            cx = min(max(x(i), px0 + 48), px1 - 48)
            g += f"<text class='hi' x='{cx:.1f}' y='{y(v) - 11:.1f}' text-anchor='middle'>{escape(str(vid))} · {fmt(v, 2)}</text>"
    return f"<svg class='ch trend' viewBox='0 0 {W} {H}'>{g}</svg>"


def _bar(us: float, rival: float | None, vmax: float, tone: str = "") -> str:
    tick = f"<i class='tick' style='left:{100 * rival / vmax:.1f}%'></i>" if rival else ""
    return f"<div class='cmp {tone}'><b style='width:{100 * us / vmax:.1f}%'></b>{tick}</div>"


@st.fragment(run_every=live.REFRESH_EVERY)
def render():
    L = live.load()
    reg = L.registry
    ch = models.choose(reg)
    me = ch["chosen"]
    sel = {s["id"]: s for s in reg["submissions"]}[me]
    pt, raw = reg["public_test"], models.arena_raw()
    opt1 = raw["oracle"]["1"]["score"] if raw else None
    stand_raw = L.standings
    stand_dedup = deduplicate_standings(stand_raw)
    lead = stand_dedup[0] if stand_dedup else reg["leaderboard"]["teams"][0]
    us_dedup = next((r for r in stand_dedup if r["team"] == L.us), None)
    us_raw = next((r for r in stand_raw if r["team"] == L.us), None)

    all_pub = [s for s in reg["submissions"] if s.get("public") is not None]
    hist = pd.DataFrame([s for s in all_pub if s["public"] >= 22.5])
    omitted = [f"{s['id']} ({fmt(s['public'], 2)})" for s in all_pub if s["public"] < 22.5]

    gap = sel["public"] - lead["score"]
    top10 = stand_dedup[9]["score"] if len(stand_dedup) >= 10 else None
    pend = ch["pending"][0] if ch["pending"] else None

    # ------------------------------------------------------------ 1 · veredicte + nota sobre el fons
    where = (
        f"{us_dedup['rank']}è equip ({us_raw['rank']}a entrada de {L.n_teams or len(stand_raw)}) · {_d(gap, 'vs líder')}"
        if us_dedup and us_raw
        else ""
    )
    cand = (
        f"Candidat: <b>{pend['id']}</b> · ha de superar <b>{fmt(pend['threshold'], 2)}</b> per substituir la {me}"
        if pend
        else "Cap versió pendent de provar"
    )
    val = (
        f"Validació <b>{fmt(sel['validation'], 2)}</b> de {fmt(opt1, 2)} ({fmt(100 * sel['validation'] / opt1, 1)} % de l'òptim)"
        if opt1
        else ""
    )
    if omitted:
        val += f" · Fora d'escala al gràfic: {', '.join(omitted)}"

    st.markdown(
        f"<div class='ed-hero'>"
        f"<div class='verdict'><div class='lab'>Lliurem</div><div class='ver'>{me}</div>"
        f"<div class='line'>{where}</div><div class='cand'>{cand}</div></div>"
        f"<div class='scoreblk'><div class='lab'>Nota pública</div>"
        f"<div class='big'>{fmt(sel['public'], 2)}{('<span class=sub>' + _d(sel['public'] - top10, 'vs top 10') + '</span>') if top10 else ''}</div>"
        f"{_trend(hist, me, lead['score']) if len(hist) > 2 else ''}"
        f"<div class='note2'>{val}</div></div></div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------ 2 · comparació amb el líder
    co2_t = pt["co2_base_kg"] * sel["co2"] / 100 / 1000
    eur_k = pt["cost_base_eur"] * sel["cost"] / 100 / 1000
    lco2, lcost = lead.get("co2"), lead.get("cost")

    def kpi(lab, v, bar, delta, note):
        return (
            f"<div><div class='lab'>{lab}</div><div class='kv'>{v}<small>%</small></div>"
            f"<div class='kb'>{bar}{delta}</div><div class='kn'>{note}</div></div>"
        )

    k1 = kpi(
        "CO₂ evitat · pes 70 %",
        fmt(sel["co2"], 2),
        _bar(sel["co2"], lco2, (lco2 or sel["co2"]) * 1.06, "eco"),
        _d(sel["co2"] - lco2, "vs líder") if lco2 else "",
        f"líder {fmt(lco2, 2)} %" if lco2 else "",
    )
    k2 = kpi(
        "Factura estalviada · pes 30 %",
        fmt(sel["cost"], 2),
        _bar(sel["cost"], lcost, (lcost or sel["cost"]) * 1.06, "slate"),
        _d(sel["cost"] - lcost, "vs líder") if lcost else "",
        f"líder {fmt(lcost, 2)} %" if lcost else "",
    )
    k3 = (
        f"<div><div class='lab'>Impacte al públic (6 setmanes)</div><div class='kv'>{fmt(co2_t, 1)}<small>t CO₂</small></div>"
        f"<div class='kn' style='margin-top:14px'>evitades · {fmt(eur_k, 1)} k€ de factura estalviats</div></div>"
    )
    st.markdown(f"<div class='ed-kpis'>{k1}{k2}{k3}</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------ 3 · rànquing (taula) + punts per guanyar
    cl, cr = st.columns([7, 5], gap="large")
    with cl:
        if not stand_raw:
            ui.note("Sense rànquing.", "No hi ha cap font en directe. Vegeu <code>dashboard/live.py</code>.")
        else:
            view = st.segmented_control(
                "Equips",
                [
                    "Per equip (sense duplicats)",
                    "Al voltant nostre",
                    "Top 10",
                    f"Totes les entrades ({len(stand_raw)})",
                ],
                default="Per equip (sense duplicats)",
                label_visibility="collapsed",
                key="lb_view",
            )
            active_stand = stand_dedup if view in (None, "Per equip (sense duplicats)") else stand_raw
            active_us = us_dedup if view in (None, "Per equip (sense duplicats)") else us_raw
            n_ahead = sum(1 for r in active_stand if r["score"] > sel["public"] + 0.004)
            st.markdown(f"<div class='lab'>Rànquing públic · {n_ahead} per davant</div>", unsafe_allow_html=True)

            idx = list(range(len(active_stand)))
            if view == "Top 10":
                idx = idx[:10]
            elif view == "Al voltant nostre" and active_us:
                k = active_us["rank"] - 1
                idx = sorted(set(idx[:3]) | set(range(max(0, k - 4), min(len(active_stand), k + 5))))

            rows, prev, body = [active_stand[i] for i in idx], None, ""
            smin, smax = min(r["score"] for r in rows), lead["score"]
            for r in rows:
                if prev is not None and r["rank"] - prev > 1:
                    body += "<tr class='sep'><td colspan='4'></td></tr>"
                prev = r["rank"]
                m = r["team"] == L.us
                w = 100 * (r["score"] - smin + 0.05) / max(smax - smin + 0.05, 1e-6)
                d = "" if m or not active_us else f"{r['score'] - active_us['score']:+.2f}".replace(".", ",").replace("-", "−")
                body += (
                    f"<tr class='{'me' if m else ''}'><td class='r'>{r['rank']}</td>"
                    f"<td>{escape(r['team'])}{' · ' + me if m else ''}</td>"
                    f"<td class='g'><div class='gwrap'><i style='width:{w:.0f}%'></i><span>{d}</span></div></td>"
                    f"<td class='n'>{fmt(r['score'], 2)}</td></tr>"
                )
            st.markdown(f"<table class='lbt'><tbody>{body}</tbody></table>", unsafe_allow_html=True)
    with cr:
        # 1.62 és el guany conjunt de les 3 entrades sobre els 3 plecs (consistent amb perfect_info)
        pot = reg["perfect_info"] + [{"gain": 1.62}]
        vmax = max(p["gain"] for p in pot) if pot else 1
        bars = "".join(
            f"<div class='pbx'><span>{SHORT[i] if i < len(reg['perfect_info']) else 'Totes juntes (3 plecs)'}</span>"
            f"<div><b class='{'full' if i >= len(reg['perfect_info']) else ''}' style='width:{100 * p['gain'] / vmax:.0f}%'></b></div>"
            f"<em>+{fmt(p['gain'], 2)}</em></div>"
            for i, p in enumerate(pot)
        )
        st.markdown(f"<div class='lab'>Punts per guanyar (validació, mitjana 3 plecs)</div><div class='pbxs'>{bars}</div>", unsafe_allow_html=True)