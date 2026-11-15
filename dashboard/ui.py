"""
dashboard/ui.py
Aspecte comú del dashboard. Sistema de disseny: minimalista (Linear / Stripe / editorial suís).

  * Superfícies planes: llenç #FAFAFA, contenidors #FFFFFF, vora d'1 px #E4E4E7, cap ombra. Mode fosc amb les mateixes
    variables CSS (`LIGHT`/`DARK`); els gràfics de Plotly (transparents) s'inverteixen amb un filtre CSS.
  * Tipografia: Helvetica per a text i títols; xifres, percentatges i taules en monoespaiat amb `tabular-nums`.
    Etiquetes de 11 px en majúscules amb 0,05 em d'espaiat, en gris.
  * Visualització: sense reixeta ni degradats; un sol color de dades (verd blavós #0F766E, #2DD4BF en fosc).
    Vermell/verd només a les insígnies de tendència (`.badge`).
  * El títol de cada gràfic diu la conclusió; el subtítol diu com es llegeix. Poca cosa a la vista: la resta a `ui.more()`.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Entitats (fixes a tot el dashboard). Verd blavós = la versió que lliurem; ink = òptim/context fort;
# ambre = rival o aspirant; gris = context. Vermell/verd només en insígnies.
NAVY, SUN = "#18181B", "#E69F00"
OURS, RIVAL, OPT = "#0072B2", "#E69F00", NAVY          # paleta Okabe-Ito (segura per a daltonisme)
C_CO2, C_COST, C_SOLAR = "#009E73", "#E69F00", "#F0E442"
GRAY, GRAY_LIGHT = "#A1A1AA", "#D4D4D8"
GOOD, BAD = "#0072B2", "#D55E00"
INK, INK_2, INK_3 = "#18181B", "#52525B", "#71717A"
GRID, PAPER, CARD, LINE = "#F4F4F5", "#FAFAFA", "#FFFFFF", "#E4E4E7"
SEQ = [[0.0, "#EAF3FA"], [0.25, "#9CC9E8"], [0.5, "#4C9FD0"], [0.75, "#0072B2"], [1.0, "#0B3C5D"]]
FONT = HEAD = '"Helvetica Neue", Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace'
PFONT = "Helvetica Neue, Helvetica, Arial"

MESOS = ["gen.", "febr.", "març", "abr.", "maig", "juny", "jul.", "ag.", "set.", "oct.", "nov.", "des."]
DIES = ["dl.", "dt.", "dc.", "dj.", "dv.", "ds.", "dg."]

LIGHT = dict(bg="#FAFAFA", card="#FFFFFF", line="#E4E4E7", ink="#18181B", ink2="#52525B", mute="#71717A", acc="#0072B2",
             accbg="#EAF3FA", grid="#F4F4F5", dot="#A1A1AA", rival="#E69F00", up="#0072B2", upbg="#EAF3FA", upln="#B9D6EC",
             dn="#B84A00", dnbg="#FFF1E6", dnln="#F5C9A5")
DARK = dict(bg="#09090B", card="#111113", line="#27272A", ink="#FAFAFA", ink2="#A1A1AA", mute="#8B8B94", acc="#56B4E9",
            accbg="#0B1D2B", grid="#1C1C1F", dot="#52525B", rival="#F5B83D", up="#56B4E9", upbg="#0B1D2B", upln="#16405C",
            dn="#F59E0B", dnbg="#2A1A05", dnln="#6B4309")


def is_explain() -> bool:
    return bool(st.session_state.get("explain", False))


def is_dark() -> bool:
    return bool(st.session_state.get("dark", False))


CSS_TEMPLATE = """
<style>
:root { __VARS__; --font: __FONT__; --mono: __MONO__; }
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp div, .stApp span,
.stApp h1, .stApp h2, .stApp h3 { font-family: var(--font); }
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] { font-family: "Material Symbols Rounded", "Material Icons" !important; }
[data-testid="stButtonGroup"] button { background: var(--card) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; }
[data-testid="stButtonGroup"] button p { color: var(--ink) !important; }
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"], [data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--accbg) !important; border-color: var(--acc) !important; }
.stApp { background: var(--bg); color: var(--ink); }
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stMarkdownContainer"] { color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.1rem 2rem 3rem; max-width: 1240px; }
.stApp * { font-variant-numeric: tabular-nums lining-nums; }
[data-testid='stHeaderActionElements'] { display: none; }
.stApp .cc, .stApp .src, .stApp .top .meta, .stApp .kpis .s, .stApp .lab, .stApp .tile .l, .stApp .tile .s { color: var(--mute); }
.stApp .why, .stApp .act, .stApp .intro p, .stApp .sec p, .stApp .note { color: var(--ink2); }
.stApp .why b, .stApp .act b, .stApp .note b { color: var(--ink); }
.stApp .dec .score { color: var(--acc); }
.num, .mono { font-family: var(--mono); letter-spacing: -.01em; white-space: nowrap; }
.lab { font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--mute); font-weight: 500; }

/* capçalera */
.top { padding: .35rem 0 .5rem; }
.top h1 { font-size: 1.05rem; font-weight: 600; letter-spacing: -.01em; margin: 0; padding: 0; }
.top .meta { font-size: 12px; color: var(--mute); margin-top: .15rem; }
.top .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--acc); margin-right: 6px; }
div[data-testid="stToggle"] label p { font-size: 12px; color: var(--mute); }

/* pestanyes */
.stTabs [role="tablist"] { gap: 1.4rem; border: 0 !important; border-bottom: 1px solid var(--line) !important; box-shadow: none !important;
                           flex-wrap: wrap; margin-bottom: .9rem; }
.stTabs [role="tablist"]::before, .stTabs [role="tablist"]::after { display: none !important; }
.stTabs [data-testid="stTab"] { background: transparent; border: 0; border-bottom: 2px solid transparent; border-radius: 0;
                                padding: .5rem 0; height: auto; color: var(--mute); cursor: pointer; margin-bottom: -1px; }
.stTabs [data-testid="stTab"] p { font-size: 13px; font-weight: 500; color: inherit; }
.stTabs [data-testid="stTab"]:hover { color: var(--ink); }
.stTabs [data-testid="stTab"][aria-selected="true"] { color: var(--ink); border-bottom-color: var(--ink); }
.stTabs .react-aria-SelectionIndicator { display: none !important; }

/* seccions i contenidors: plans, vora d'1 px, sense ombres */
.sec { margin: 2.2rem 0 .8rem; }
.sec h2 { font-size: 1.05rem; font-weight: 600; letter-spacing: -.01em; margin: 0; padding: 0; }
.sec p { font-size: 13px; color: var(--ink2); margin: .3rem 0 0; max-width: 80ch; }
div[data-testid="stVerticalBlock"][class*="st-key-card"] {
    background: var(--card); border: 1px solid var(--line) !important; border-radius: 4px !important; padding: 1rem 1.1rem .5rem !important; box-shadow: none; }
.ct { font-size: 15px; line-height: 1.3; font-weight: 600; letter-spacing: -.01em; color: var(--ink); margin: 0 0 .25rem; }
.cc { font-size: 12.5px; line-height: 1.45; color: var(--mute); margin: 0 0 .4rem; max-width: 70ch; }
.src { font-size: 12px; line-height: 1.45; color: var(--mute); margin: -.1rem 0 .5rem; }
.box { background: var(--card); border: 1px solid var(--line); border-radius: 4px; }
.pad { padding: 1.05rem 1.25rem; }
.gap { height: .75rem; }

/* insígnies de tendència (l'únic lloc amb vermell/verd) */
.badge { display: inline-block; font: 500 11px var(--mono); padding: 1px 6px; border: 1px solid; border-radius: 3px; white-space: nowrap; }
.badge.up { color: var(--up); background: var(--upbg); border-color: var(--upln); }
.badge.dn { color: var(--dn); background: var(--dnbg); border-color: var(--dnln); }
.badge.nt { color: var(--mute); background: transparent; border-color: var(--line); }

/* decisió */
.dec { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); }
.dec > div { padding: 1.3rem 1.5rem; }
.dec > div + div { border-left: 1px solid var(--line); }
.dec .ver { font: 500 3.4rem/1 var(--mono); letter-spacing: -.04em; margin: .55rem 0 .7rem; }
.dec .ver small { font: 400 13px var(--font); letter-spacing: 0; color: var(--mute); margin-left: .6rem; }
.dec .why { font-size: 14px; line-height: 1.55; color: var(--ink2); max-width: 60ch; margin: 0; }
.dec .why b { color: var(--ink); font-weight: 600; }
.dec .score { font: 500 3.4rem/1 var(--mono); letter-spacing: -.04em; margin: .55rem 0 .55rem; color: var(--acc); }
.rank { display: flex; gap: 1.6rem; margin-top: .9rem; flex-wrap: wrap; }
.rank .v { font: 500 1.35rem/1.1 var(--mono); letter-spacing: -.02em; margin-top: .2rem; white-space: nowrap; }
.rank .v small { font: 400 12px var(--mono); color: var(--mute); }

/* franja d'indicadors: una sola peça, separadors fins */
.kpis { display: grid; grid-template-columns: repeat(var(--n, 5), minmax(0, 1fr)); }
.kpis > div { padding: .85rem 1.1rem; }
.kpis > div + div { border-left: 1px solid var(--line); }
.kpis .v { font: 500 1.5rem/1.15 var(--mono); letter-spacing: -.03em; margin: .3rem 0 .3rem; white-space: nowrap; }
.kpis .v small { font-size: 12px; color: var(--mute); letter-spacing: 0; }
.kpis .s { font-size: 12px; color: var(--mute); }

/* llista de xifres i accions */
.rows .r { display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; padding: .62rem 0; border-top: 1px solid var(--line); font-size: 13px; }
.rows .r:first-child { border-top: 0; }
.rows .r span:first-child small { color: var(--mute); font-size: 12px; margin-left: .3rem; }
.rows .r .num { font-size: 15px; }
.act { font-size: 13.5px; line-height: 1.5; color: var(--ink2); }
.act b { color: var(--ink); }
.act .rule { border: 1px solid var(--line); border-left: 2px solid var(--acc); padding: .65rem .8rem; margin-top: .55rem; border-radius: 4px; }
svg.ch { width: 100%; height: auto; display: block; }
svg.ch text { font-family: var(--mono); font-size: 11px; fill: var(--mute); }
svg.ch text.hi { fill: var(--ink); font-weight: 600; }
svg.ch .ln { stroke: var(--line); } svg.ch .ac { fill: var(--acc); } svg.ch .ik { fill: var(--ink); } svg.ch .mu { fill: var(--dot); }
svg.ch .acs { stroke: var(--acc); fill: none; } svg.ch .bg { fill: var(--grid); }

/* xifres (tiles de les altres pàgines): planes, d'una peça */
.tiles { display: grid; grid-template-columns: repeat(var(--n, 4), minmax(0, 1fr)); margin: .8rem 0 .3rem;
         background: var(--card); border: 1px solid var(--line); border-radius: 4px; }
.tile { padding: .8rem 1.05rem .85rem; }
.tile + .tile { border-left: 1px solid var(--line); }
.tile .l { font-size: 11px; letter-spacing: .05em; text-transform: uppercase; font-weight: 500; color: var(--mute); }
.tile .v { font: 500 1.6rem/1.2 var(--mono); letter-spacing: -.03em; color: var(--ink); margin: .2rem 0; white-space: nowrap; }
.tile .v small { font-size: 12px; color: var(--mute); letter-spacing: 0; }
.tile .s { font-size: 12px; line-height: 1.4; color: var(--mute); }

/* introducció de pàgina i notes */
.intro { margin: .9rem 0 .6rem; }
.intro h3 { font-size: 1.25rem; font-weight: 600; letter-spacing: -.02em; margin: 0 0 .25rem; padding: 0; }
.intro p { font-size: 14px; line-height: 1.5; color: var(--ink2); margin: 0; max-width: 80ch; }
.note { background: var(--card); border: 1px solid var(--line); border-left: 2px solid var(--acc); border-radius: 4px; padding: .8rem 1.05rem;
        font-size: 13.5px; line-height: 1.55; color: var(--ink2); margin: .7rem 0 .9rem; }
.note b { color: var(--ink); }
.note ol, .note ul { margin: .3rem 0 0; padding-left: 1.15rem; }
.note li { margin: .25rem 0; }
.chk { list-style: none; padding: 0 !important; margin: .2rem 0 .4rem !important; }
.chk li { display: flex; gap: .6rem; border-bottom: 1px solid var(--line); padding: .38rem 0; margin: 0; font-size: 13px; }
.chk .m { width: 2rem; font: 500 12px var(--mono); }
.chk .d { margin-left: auto; color: var(--mute); text-align: right; }

/* desplegables "més detall" */
[data-testid="stExpander"] details { background: var(--card); border: 1px solid var(--line); border-radius: 4px; }
[data-testid="stExpander"] summary { padding: .6rem 1rem; }
[data-testid="stExpander"] summary p { font-size: 13px; font-weight: 500; color: var(--ink); }

/* taules */
table.t { width: 100%; border-collapse: collapse; font-size: 12.5px; font-family: var(--mono); }
table.t, table.t th, table.t td { border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }
table.t th { text-align: right; font: 500 11px var(--font); letter-spacing: .05em; text-transform: uppercase; color: var(--mute);
             border-bottom: 1px solid var(--line); padding: .35rem .5rem .35rem 0; vertical-align: bottom; position: sticky; top: 0; background: var(--card); }
table.t td { text-align: right; padding: .4rem .5rem .4rem 0; border-bottom: 1px solid var(--line); white-space: nowrap; color: var(--ink); }
table.t th.L, table.t td.L { text-align: left; white-space: normal; font-family: var(--font); }
table.t tr.hl td { font-weight: 600; background: var(--accbg); }
table.t td.L:first-child { min-width: 11rem; padding-left: .3rem; }
table.t .bar { display: inline-block; width: 70px; height: 4px; background: var(--grid); margin-right: .45rem; vertical-align: middle; overflow: hidden; }
table.t .bar i { display: block; height: 100%; background: var(--acc); }
.tw { overflow: auto; }

/* controls */
[data-testid="stButtonGroup"] [role="toolbar"] { flex-wrap: wrap !important; overflow: visible !important; row-gap: .35rem; }
.stButton button { height: auto; min-height: 2.3rem; }
.stButton button p { white-space: normal; line-height: 1.25; font-size: 13px; }
.stButton button, .stFormSubmitButton button { border-radius: 4px; border: 1px solid var(--line); font-weight: 500; background: var(--card); color: var(--ink); }
.stButton button:hover { border-color: var(--ink); color: var(--ink); }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] { background: var(--ink); border-color: var(--ink); color: var(--bg); }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="textarea"], [data-testid="stChatInput"] > div {
    background: var(--card) !important; border-color: var(--line) !important; border-radius: 4px !important; color: var(--ink); }
[data-baseweb="select"] *, [data-baseweb="input"] input, [data-testid="stChatInput"] textarea { color: var(--ink) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] { background: var(--card) !important; color: var(--ink) !important; }
[data-testid="stChatMessage"] { background: var(--card); border: 1px solid var(--line); border-radius: 4px; }
.foot { font-size: 12px; line-height: 1.5; color: var(--mute); margin-top: 2.4rem; padding-top: .7rem; border-top: 1px solid var(--line); }
div[data-testid="stCaptionContainer"] p { font-size: 12px; color: var(--mute); }

/* portada: decisió */
.live { display: flex; align-items: center; gap: .5rem; font: 500 12px var(--mono); color: var(--ink2); margin: 0 0 .5rem; }
.live i { width: 8px; height: 8px; border-radius: 50%; background: var(--acc); display: inline-block; }
.live.ok i { animation: pulse 2s infinite; }
.live.warn i { background: var(--rival); border-radius: 1px; } .live.bad i { background: var(--dn); border-radius: 0; transform: rotate(45deg); }
.live span { color: var(--mute); margin-left: .5rem; font-weight: 400; }
@keyframes pulse { 0% { opacity: 1 } 50% { opacity: .35 } 100% { opacity: 1 } }
.hero { display: grid; grid-template-columns: .8fr 1.6fr .55fr; align-items: stretch; }
.hero > div { padding: 1.2rem 1.5rem; }
.hero > div + div { border-left: 1px solid var(--line); }
.hero .ver { font: 500 4.2rem/1 var(--mono); letter-spacing: -.05em; margin: .6rem 0 .8rem; }
.hero .score { font: 500 4.2rem/1 var(--mono); letter-spacing: -.05em; margin: .6rem 0 .3rem; color: var(--acc); }
.hero .chips { display: flex; gap: .6rem; flex-wrap: wrap; }
.chip { display: inline-flex; align-items: baseline; gap: .35rem; font: 500 1.05rem var(--mono); color: var(--ink); border: 1px solid var(--line); border-radius: 4px; padding: .2rem .55rem; }
.chip small { font: 400 11px var(--font); color: var(--mute); letter-spacing: .03em; }
.chip .badge { border: 0; background: transparent; padding: 0; font-size: 1.05rem; }
.hero .rg { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: .5rem; }
.ring { width: 118px; height: 118px; }
.ring .rbg { fill: none; stroke: var(--grid); stroke-width: 9; } .ring .rfg { fill: none; stroke: var(--acc); stroke-width: 9; stroke-linecap: butt; }
.ring text { font: 500 20px var(--mono); fill: var(--ink); }
svg.spark { width: 100%; height: 74px; margin-top: .3rem; }
svg.ch .rv { fill: var(--rival); stroke: var(--rival); } svg.ch .me { stroke: var(--acc); opacity: .5; } svg.ch circle.mu { fill: var(--dot); }
/* portada: indicadors amb comparació */
.kp { display: grid; grid-template-columns: 1fr 1fr .8fr; }
.kp > div { padding: 1rem 1.3rem; } .kp > div + div { border-left: 1px solid var(--line); }
.kp .v { font: 500 2rem/1.1 var(--mono); letter-spacing: -.04em; margin: .35rem 0 .6rem; white-space: nowrap; display: flex; align-items: baseline; gap: .15rem; }
.kp .v small { font-size: 13px; color: var(--mute); letter-spacing: 0; margin-right: .6rem; }
.kp .s { font: 12px var(--mono); color: var(--mute); margin-top: .4rem; }
.cmp { position: relative; height: 8px; background: var(--grid); border-radius: 2px; }
.cmp b { position: absolute; left: 0; top: 0; bottom: 0; background: var(--acc); border-radius: 2px; }
.cmp .tick { position: absolute; top: -4px; bottom: -4px; width: 3px; background: var(--rival); border-radius: 1px; }
.badge.big { font-size: 12px; }
/* següent pas i potencial */
.nx { display: flex; align-items: baseline; gap: .9rem; margin: .5rem 0 .4rem; }
.nx .gt { font: 500 2.4rem/1 var(--mono); letter-spacing: -.04em; }
.chip2 { font: 500 13px var(--mono); border: 1px solid var(--rival); color: var(--ink); border-radius: 4px; padding: .1rem .5rem; }
.box .s { font-size: 12px; color: var(--mute); }
.pbs { margin-top: .6rem; display: grid; gap: .55rem; }
.pb { display: grid; grid-template-columns: 8.5rem 1fr 3rem; align-items: center; gap: .6rem; font-size: 12px; color: var(--ink2); }
.pb > div { height: 8px; background: var(--grid); border-radius: 2px; } .pb b { display: block; height: 100%; background: var(--rival); border-radius: 2px; }
.pb b.full { background: var(--ink); } .pb em { font: normal 12px var(--mono); text-align: right; color: var(--ink); }
@media (max-width: 900px) { .hero, .kp { grid-template-columns: 1fr; } .hero > div + div, .kp > div + div { border-left: 0; border-top: 1px solid var(--line); } }
__DARKFIX__
__HIDE__
@media (max-width: 900px) { .dec, .kpis { grid-template-columns: 1fr 1fr; } .dec > div + div { border-left: 0; border-top: 1px solid var(--line); }
    .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .dec, .kpis { grid-template-columns: 1fr; } .kpis > div + div { border-left: 0; border-top: 1px solid var(--line); } }
</style>
"""


HIDE_RULES = """.cc, .src, .sec p, .intro p, .note, .foot, div[data-testid="stCaptionContainer"] { display: none !important; }"""


def _css(theme: dict, dark: bool) -> str:
    vars_ = ";".join(f"--{k}:{v}" for k, v in theme.items())
    fix = ".stPlotlyChart { filter: invert(1) hue-rotate(180deg); }" if dark else ""
    return CSS_TEMPLATE.replace("__VARS__", vars_).replace("__FONT__", FONT).replace("__MONO__", MONO).replace("__DARKFIX__", fix).replace("__HIDE__", "" if is_explain() else HIDE_RULES)


def setup(title: str):
    st.set_page_config(page_title=title, page_icon=None, layout="wide", initial_sidebar_state="collapsed")
    st.markdown(_css(DARK if is_dark() else LIGHT, is_dark()), unsafe_allow_html=True)


def fmt(x, dec: int = 0, sign: bool = False) -> str:
    """Format català: 1.848,5 (coma decimal, punt de milers)."""
    if x is None or x != x:
        return "—"
    s = f"{x:+,.{dec}f}" if sign else f"{x:,.{dec}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".").replace("-", "−")


def data(ts, any_: bool = True) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {MESOS[ts.month - 1]}" + (f" {ts.year}" if any_ else "")


def badge(text: str, kind: str = "nt") -> str:
    """Insígnia de tendència: kind 'up' (verd), 'dn' (vermell) o 'nt' (neutra)."""
    arrow = {"up": "▲ ", "dn": "▼ "}.get(kind, "")      # la forma també ho diu: no depèn només del color
    return f"<span class='badge {kind}'>{arrow}{text}</span>"


def section(title: str, sub: str | None = None):
    st.markdown(f"<div class='sec'><h2>{title}</h2>" + (f"<p>{sub}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


def intro(title: str, text: str):
    st.markdown(f"<div class='intro'><h3>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)


def panel(title: str, caption: str | None = None, key: str | None = None):
    """Contenidor pla per a un gràfic: títol = conclusió, subtítol = com es llegeix.

    La clau `st-key-card_…` és el que el CSS fa servir per pintar el contenidor; es deriva del títol, i cal passar
    `key` si dos panells de la mateixa pàgina poden tenir el mateix títol.
    """
    box = st.container(border=True, key="card_" + (key or hashlib.md5(title.encode()).hexdigest()[:10]))
    box.markdown(f"<p class='ct'>{title}</p>" + (f"<p class='cc'>{caption}</p>" if caption else ""), unsafe_allow_html=True)
    return box


def more(label: str):
    """Desplegable tancat per a tot el que no cal per decidir."""
    return st.expander(label, expanded=False)


def source(text: str, where=st):
    where.markdown(f"<p class='src'>{text}</p>", unsafe_allow_html=True)


def tiles(items: list[tuple], cols: int = 4, where=st):
    """Xifres en una sola peça: (etiqueta, valor_html, detall[, to ignorat]). Etiquetes en majúscules, valors en monoespaiat."""
    cells = "".join(f"<div class='tile'><div class='l'>{it[0]}</div>"
                    f"<div class='v'>{it[1]}</div><div class='s'>{it[2]}</div></div>" for it in items)
    where.markdown(f"<div class='tiles' style='--n:{cols}'>{cells}</div>", unsafe_allow_html=True)


facts = tiles


def note(head: str, body_html: str, where=st):
    where.markdown(f"<div class='note'><b>{head}</b> {body_html}</div>", unsafe_allow_html=True)


def checklist(items: list[tuple], where=st):
    """(nom, compleix, detall) -> llista de sí / no."""
    lis = "".join(f"<li><span class='m' style='color:{'var(--up)' if ok else 'var(--dn)'}'>{'sí' if ok else 'no'}</span>"
                  f"<span>{name}</span><span class='d'>{detail}</span></li>" for name, ok, detail in items)
    where.markdown(f"<ul class='chk'>{lis}</ul>", unsafe_allow_html=True)


def table(df: pd.DataFrame, names: dict, formats: dict | None = None, left: tuple = (), bars: dict | None = None,
          highlight=None, height: int | None = None, where=st):
    """Taula HTML: xifres a la dreta amb coma decimal i, si es vol, una barra dins la cel·la."""
    formats, bars = formats or {}, bars or {}
    head = "".join(f"<th class='{'L' if c in left else ''}'>{names[c]}</th>" for c in names)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in names:
            v = r[c]
            txt = formats[c](v) if c in formats else str(v)
            if c in bars:
                w = max(0.0, min(100.0, 100 * float(v) / bars[c])) if v == v else 0
                txt = f"<span class='bar'><i style='width:{w:.0f}%'></i></span>{txt}"
            tds.append(f"<td class='{'L' if c in left else ''}'>{txt}</td>")
        body.append(f"<tr class='{'hl' if highlight is not None and highlight(r) else ''}'>{''.join(tds)}</tr>")
    style_ = f" style='max-height:{height}px'" if height else ""
    where.markdown(f"<div class='tw'{style_}><table class='t'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}"
                   "</tbody></table></div>", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=2, r=14, t=26 if legend else 10, b=2), separators=",.",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PFONT, size=12, color=INK_2), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None, font=dict(color=INK)),
        hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor=INK, font_family=PFONT),
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRAY_LIGHT, linewidth=1, tickcolor=GRAY_LIGHT, zeroline=False, ticks="outside",
                     ticklen=4, title_font=dict(size=12, color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, title_font=dict(size=12, color=INK_2))
    return fig


def show(fig: go.Figure, where=st, key: str | None = None):
    where.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def rgba(hex_: str, a: float) -> str:
    return f"rgba({int(hex_[1:3], 16)},{int(hex_[3:5], 16)},{int(hex_[5:7], 16)},{a})"
