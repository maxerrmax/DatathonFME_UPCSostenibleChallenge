"""
dashboard/ui.py
Sistema de disseny d'enginyeria d'alta precisió: jerarquia clara, separació visual nítida entre seccions i panells,
descripcions micro-tipogràfiques (11 px) i xat compacte amb entrada sempre visible.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY, SUN = "#0F172A", "#D97706"
OURS, RIVAL, OPT = "#0284C7", "#D97706", "#0F172A"
C_CO2, C_COST, C_SOLAR = "#0D9488", "#D97706", "#EAB308"
GRAY, GRAY_LIGHT = "#94A3B8", "#CBD5E1"
GOOD, BAD = "#0284C7", "#DC2626"
INK, INK_2, INK_3 = "#0F172A", "#334155", "#64748B"
GRID, PAPER, CARD, LINE = "#E2E8F0", "#F1F5F9", "#FFFFFF", "#CBD5E1"

SEQ = [[0.0, "#E0F2FE"], [0.25, "#7DD3FC"], [0.5, "#0284C7"], [0.75, "#0369A1"], [1.0, "#0C4A6E"]]
SEQ_CO2 = [[0.0, "#F0FDFA"], [0.25, "#99F6E4"], [0.5, "#14B8A6"], [0.75, "#0F766E"], [1.0, "#134E4A"]]

FONT = HEAD = '-apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", Arial, sans-serif'
MONO = '"JetBrains Mono", "IBM Plex Mono", "SF Mono", ui-monospace, Consolas, monospace'
PFONT = "Helvetica Neue, Helvetica, Arial"
MESOS = ["gen.", "febr.", "març", "abr.", "maig", "juny", "jul.", "ag.", "set.", "oct.", "nov.", "des."]
DIES = ["dl.", "dt.", "dc.", "dj.", "dv.", "ds.", "dg."]

LIGHT = dict(
    bg="#F1F5F9", card="#FFFFFF", line="#CBD5E1", line_soft="#E2E8F0", ink="#0F172A", ink2="#334155", mute="#64748B",
    acc="#0284C7", accbg="#E0F2FE", eco="#0D9488", ecobg="#CCFBF1", grid="#E2E8F0", track="#E2E8F0", dot="#94A3B8",
    dash="#CBD5E1", rival="#D97706", up="#0369A1", upbg="#E0F2FE", upln="#BAE6FD", dn="#B45309", dnbg="#FEF3C7",
    dnln="#FDE68A", verdict="#0F172A", bubble="#F1F5F9", seg="#E2E8F0", outline="#CBD5E1", onacc="#FFFFFF"
)
DARK = dict(
    bg="#090D16", card="#111827", line="#334155", line_soft="#1E293B", ink="#F8FAFC", ink2="#CBD5E1", mute="#94A3B8",
    acc="#38BDF8", accbg="#082F49", eco="#2DD4BF", ecobg="#115E59", grid="#1E293B", track="#1E293B", dot="#64748B",
    dash="#334155", rival="#F59E0B", up="#7DD3FC", upbg="#082F49", upln="#0369A1", dn="#FBBF24", dnbg="#451A03",
    dnln="#78350F", verdict="#1E293B", bubble="#1E293B", seg="#1E293B", outline="#334155", onacc="#090D16"
)


def is_dark() -> bool:
    return bool(st.session_state.get("dark", False))


CSS_TEMPLATE = """
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap");
:root { __VARS__; --font: __FONT__; --mono: __MONO__; }

html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp div, .stApp span, .stApp h1, .stApp h2, .stApp h3, [data-baseweb] *, [data-testid="stChatInput"] * {
    font-family: var(--font);
}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}
.stApp { background: var(--bg); color: var(--ink); font-size: 14px; line-height: 1.45; }
.stApp * { font-variant-numeric: tabular-nums lining-nums; }
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stMarkdownContainer"] {
    color: var(--ink);
}
#MainMenu, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
footer, [data-testid="stStatusWidget"], [data-testid="stAppDeployButton"], [data-testid="stHeaderActionElements"] {
    display: none !important;
}

.block-container { padding: 20px 28px 48px; max-width: 1260px; }
[data-testid="stHorizontalBlock"] { gap: 20px; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 14px; }

/* Tipografia tècnica i descripcions molt més petites */
.num, .mono, .tile .v, table.t td { font-family: var(--mono); letter-spacing: -.015em; white-space: nowrap; }
.lab, .tile .l { font: 600 10.5px/14px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--mute); }
.ct { font: 600 14.5px/19px var(--font); letter-spacing: -.01em; color: var(--ink); margin: 0 0 3px; }
.cc {
    font: 400 10px/13px var(--font); color: var(--mute) !important; margin: 0 0 8px; max-width: 78ch;
    overflow-wrap: anywhere;
}
.src { font: 400 10.5px/14px var(--font); color: var(--mute) !important; margin: 2px 0 4px; }
code, .stApp code { font-family: var(--mono); font-size: 11px; background: var(--grid); color: var(--ink); border-radius: 4px; padding: 1px 5px; }

/* Capçalera + casella Explicacions + botó icona sol/lluna */
.top { padding: 0 0 8px; }
.top h1 { font: 700 22px/26px var(--font); letter-spacing: -.02em; margin: 0 0 2px; }
.top .meta { font: 400 11.5px/16px var(--mono); color: var(--mute); }
.live { display: inline; font: inherit; color: inherit; }
.live i { width: 6px; height: 6px; border-radius: 50%; background: var(--eco); display: inline-block; margin-right: 5px; vertical-align: 1px; }
.live.warn, .live.bad { color: var(--dn); }
.live.warn i { background: var(--rival); }
.live.bad i { background: var(--dn); border-radius: 1px; }

div[data-testid="stCheckbox"] { display: flex; justify-content: flex-end; align-items: center; }
div[data-testid="stCheckbox"] label { gap: 6px !important; cursor: pointer; }
div[data-testid="stCheckbox"] label p { font: 500 12px/16px var(--font) !important; color: var(--ink2) !important; white-space: nowrap; }
.st-key-dark_btn button {
    min-height: 34px !important; width: 38px !important; padding: 0 !important; border-radius: 8px !important;
    border: 1px solid var(--line) !important; background: var(--card) !important; color: var(--ink) !important;
    display: flex; align-items: center; justify-content: center;
}
.st-key-dark_btn button:hover { border-color: var(--acc) !important; background: var(--accbg) !important; }

/* Pestanyes amb separació estructural clara */
.stTabs [role="tablist"] {
    gap: 4px; background: var(--card); padding: 4px; border-radius: 8px;
    border: 1px solid var(--line) !important; width: fit-content; max-width: 100%; margin-bottom: 20px;
    position: sticky; top: 8px; z-index: 60; box-shadow: 0 2px 6px rgba(15,23,42,.06); flex-wrap: wrap;
}
.stTabs { margin-top: 14px; }
.stTabs [role="tablist"]::before, .stTabs [role="tablist"]::after, .stTabs .react-aria-SelectionIndicator { display: none !important; }
.stTabs [data-testid="stTab"] {
    min-height: 32px; padding: 0 12px; border: 0; border-radius: 6px; color: var(--mute); background: transparent;
    transition: background .12s, color .12s; cursor: pointer;
}
.stTabs [data-testid="stTab"] p { font: 500 12.5px/16px var(--font); color: inherit; }
.stTabs [data-testid="stTab"][aria-selected="false"]:hover { color: var(--ink); background: var(--bg); }
.stTabs [data-testid="stTab"][aria-selected="true"] {
    background: var(--ink); color: var(--card);
}
.stTabs [data-testid="stTab"][aria-selected="true"] p { font-weight: 600; color: var(--card) !important; }

/* Separació visual forta entre seccions i panells */
.sec {
    margin: 36px 0 12px; padding: 14px 0 6px;
    border-top: 2px solid var(--line); border-bottom: 1px solid var(--line_soft);
}
.sec:first-of-type { margin-top: 16px; }
.sec h2 { font: 700 15px/20px var(--font); letter-spacing: -.01em; text-transform: uppercase; color: var(--ink); margin: 0; }
.sec p { font: 400 10.5px/14px var(--font); color: var(--mute) !important; margin: 2px 0 0; max-width: 80ch; }

.intro {
    margin: 4px 0 16px; padding: 12px 16px; background: var(--card);
    border: 1px solid var(--line); border-left: 4px solid var(--ink); border-radius: 6px;
}
.intro h3 { font: 700 16px/20px var(--font); letter-spacing: -.01em; margin: 0 0 2px; }
.intro p { font: 400 10.5px/14px var(--font); color: var(--mute) !important; margin: 0; max-width: 85ch; }

div[data-testid="stVerticalBlock"][class*="st-key-card"], .box, .tiles, [data-testid="stExpander"] details {
    background: var(--card); border: 1px solid var(--line) !important;
    border-radius: 8px !important; box-shadow: 0 1px 2px rgba(15,23,42,.04) !important;
}
div[data-testid="stVerticalBlock"][class*="st-key-card"] { padding: 14px 18px 10px !important; }
[data-testid="stPlotlyChart"], .stPlotlyChart { margin: 0; overflow: hidden; }

/* Franja d'indicadors (tiles) amb divisors verticals nítids */
.tiles { display: grid; grid-template-columns: repeat(var(--n, 4), minmax(0, 1fr)); margin: 6px 0 10px; }
.tile { padding: 12px 16px; }
.tile + .tile { border-left: 1px solid var(--line); }
.tile .v { font: 600 22px/28px var(--mono); letter-spacing: -.03em; color: var(--ink); margin: 3px 0 2px; }
.tile .v small { font: 400 11px var(--font); color: var(--mute); letter-spacing: 0; }
.tile .s { font: 400 11px/14px var(--font); color: var(--mute) !important; }

/* Notes de decisió i desplegables */
.note {
    background: var(--card); border: 1px solid var(--line) !important; border-left: 4px solid var(--acc) !important;
    border-radius: 6px !important; padding: 12px 16px; font-size: 12.5px; line-height: 1.5; color: var(--ink2); margin: 8px 0 12px;
}
.note b { color: var(--ink); }
.note ol, .note ul { margin: 4px 0 0; padding-left: 1.15rem; }
.note li { margin: 2px 0; }
.chk { list-style: none; padding: 0 !important; margin: 4px 0 !important; }
.chk li { display: flex; gap: 8px; border-bottom: 1px solid var(--line_soft); padding: 7px 0; margin: 0; font-size: 12px; }
.chk .m { width: 2rem; font: 600 11px var(--mono); text-transform: uppercase; }
.chk .d { margin-left: auto; color: var(--mute); font: 11px var(--mono); text-align: right; }

[data-testid="stExpander"] details { overflow: hidden; margin-top: 8px; }
[data-testid="stExpander"] summary { padding: 10px 16px; background: color-mix(in srgb, var(--bg) 50%, var(--card)); }
[data-testid="stExpander"] summary:hover { background: var(--accbg); }
[data-testid="stExpander"] details[open] summary { border-bottom: 1px solid var(--line); }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 14px 18px 16px; }
[data-testid="stExpander"] summary p { font: 600 12.5px/18px var(--font); color: var(--ink); }

/* Taules d'enginyeria denses */
.tw { overflow: auto; border-radius: 6px; border: 1px solid var(--line_soft); }
table.t { width: 100%; border-collapse: collapse; font-size: 12px; font-family: var(--mono); }
table.t, table.t th, table.t td { border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }
table.t th {
    text-align: right; font: 600 10.5px var(--mono); letter-spacing: .05em; text-transform: uppercase; color: var(--mute);
    border-bottom: 1px solid var(--line); padding: 8px 12px 8px 0; background: color-mix(in srgb, var(--bg) 55%, var(--card));
    position: sticky; top: 0; z-index: 2;
}
table.t td { text-align: right; padding: 7px 12px 7px 0; border-bottom: 1px solid var(--line_soft); color: var(--ink); }
table.t th.L, table.t td.L { text-align: left; white-space: normal; font-family: var(--font); }
table.t td.L:first-child, table.t th.L:first-child { min-width: 10rem; padding-left: 10px; }
table.t tbody tr:hover td { background: color-mix(in srgb, var(--acc) 6%, transparent); }
table.t tr.hl td { font-weight: 600; background: var(--accbg); }
table.t .bar { display: inline-block; width: 60px; height: 4px; background: var(--grid); margin-right: 6px; vertical-align: middle; overflow: hidden; border-radius: 2px; }
table.t .bar i { display: block; height: 100%; background: var(--acc); }

/* Portada (p_front.py): panells ben delimitats */
.d { font: 500 12px var(--mono); white-space: nowrap; }
.d.up, .badge.up { color: var(--up); }
.d.dn, .badge.dn { color: var(--dn); }
.d.nt, .badge.nt { color: var(--mute); }
.badge { font: 500 12px var(--mono); }

.ed-hero { display: grid; grid-template-columns: 5fr 7fr; gap: 20px; align-items: stretch; margin-top: 4px; }
.verdict {
    background: var(--verdict); color: #F8FAFC; border: 1px solid var(--line); border-radius: 8px;
    padding: 22px 24px; display: flex; flex-direction: column; min-height: 260px;
}
.verdict .lab { color: #94A3B8; }
.verdict .ver { font: 600 72px/1 var(--mono); letter-spacing: -3px; margin: 12px 0 10px; color: #fff; }
.verdict .line { font-size: 13px; color: #CBD5E1; }
.verdict .line .d.dn { color: #FCD34D; }
.verdict .line .d.up { color: #7DD3FC; }
.verdict .cand { margin-top: auto; padding-top: 12px; border-top: 1px solid #334155; font-size: 12px; line-height: 18px; color: #CBD5E1; }
.verdict .cand b { font-family: var(--mono); color: #fff; font-weight: 600; }

.scoreblk {
    background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 22px 24px;
}
.scoreblk .big { font: 600 64px/1 var(--mono); letter-spacing: -2.5px; color: var(--acc); margin: 10px 0 4px; display: flex; align-items: baseline; gap: 16px; }
.scoreblk .big .sub { font-size: 13px; letter-spacing: 0; }
.scoreblk .note2 { font: 11px/15px var(--mono); color: var(--mute); margin-top: 6px; }
.scoreblk .note2 b { color: var(--ink); font-weight: 600; }
svg.ch { width: 100%; height: auto; display: block; }
svg.ch text { font-family: var(--mono); font-size: 10px; fill: var(--mute); }
svg.ch text.hi { fill: var(--ink); font-weight: 600; }
svg.ch text.rvl { fill: var(--rival); font-weight: 600; }
svg.ch .rv { stroke: var(--rival); }
svg.ch .acs { stroke: var(--acc); fill: none; }
svg.ch .ac { fill: var(--acc); }
svg.ch circle.mu { fill: var(--dot); }
svg.trend { margin-top: 8px; }

.ed-kpis {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin: 20px 0 24px;
    background: var(--card); border: 1px solid var(--line); border-radius: 8px; overflow: hidden;
}
.ed-kpis > div { padding: 14px 20px; }
.ed-kpis > div + div { border-left: 1px solid var(--line); }
.kv { font: 600 32px/1.1 var(--mono); letter-spacing: -1px; margin-top: 4px; }
.kv small { font: 500 12px var(--mono); color: var(--mute); margin-left: 4px; letter-spacing: 0; }
.kb { display: flex; align-items: center; gap: 12px; margin-top: 10px; }
.kb .cmp { flex: 1; }
.kn { font: 11px/14px var(--font); color: var(--mute); margin-top: 6px; }
.cmp { position: relative; height: 5px; border-radius: 2px; background: var(--grid); }
.cmp b { position: absolute; left: 0; top: 0; bottom: 0; background: var(--acc); border-radius: 2px; }
.cmp.eco b { background: var(--eco); }
.cmp.slate b { background: var(--ink2); }
.cmp .tick { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--rival); }

table.lbt { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 6px; background: var(--card); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
table.lbt td { padding: 6px 10px; border-bottom: 1px solid var(--line_soft); color: var(--ink); }
table.lbt td.r { width: 34px; text-align: right; font: 400 12px var(--mono); color: var(--mute); }
table.lbt td.n { text-align: right; font: 600 13px var(--mono); width: 68px; }
table.lbt td.g { width: 140px; font: 11.5px var(--mono); color: var(--mute); text-align: right; }
table.lbt td.g .gwrap { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
table.lbt td.g i { display: inline-block; height: 4px; background: var(--dash); border-radius: 2px; max-width: 70px; }
table.lbt td.g span { min-width: 42px; text-align: right; }
table.lbt tr:not(.me):not(.sep):hover td { background: color-mix(in srgb, var(--ink) 4%, transparent); }
table.lbt tr.me td { background: var(--accbg); font-weight: 600; }
table.lbt tr.me td.r { color: var(--acc); }
table.lbt tr.sep td { padding: 0; height: 8px; border-bottom: 1px dashed var(--dash); }

.pbxs { margin-top: 8px; background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 6px 14px; }
.pbx { display: grid; grid-template-columns: 8.5rem 1fr 3.4rem; gap: 12px; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--line_soft); font-size: 12px; color: var(--ink2); }
.pbx:last-child { border-bottom: 0; }
.pbx > div { height: 5px; background: var(--grid); border-radius: 2px; }
.pbx b { display: block; height: 100%; background: var(--rival); border-radius: 2px; }
.pbx b.full { background: var(--ink); }
.pbx em { font: normal 12px var(--mono); text-align: right; color: var(--ink); }

/* Controls i píndoles d'hores */
[data-testid="stWidgetLabel"] p, .stApp label p { font: 600 11px/15px var(--font) !important; color: var(--mute) !important; }
[data-testid="stButtonGroup"] [role="toolbar"] { flex-wrap: wrap !important; overflow: visible !important; row-gap: 4px; }
[data-testid="stButtonGroup"] button {
    border: 1px solid var(--line) !important; border-radius: 6px !important;
    background: var(--card) !important; margin: 0 4px 0 0 !important; padding: 2px 10px !important;
    min-height: 28px; color: var(--ink2) !important;
}
[data-testid="stButtonGroup"] button:hover { border-color: var(--ink) !important; color: var(--ink) !important; }
[data-testid="stButtonGroup"] button p { color: inherit !important; font: 500 12px/16px var(--font) !important; }
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
[data-testid="stButtonGroup"] button[aria-checked="true"] {
    background: var(--accbg) !important; color: var(--acc) !important; border-color: var(--acc) !important;
}

.st-key-play_ch [data-testid="stButtonGroup"] button,
.st-key-play_dis [data-testid="stButtonGroup"] button {
    padding: 3px 9px !important; margin: 2px !important; min-height: 30px !important;
}
.st-key-play_ch [data-testid="stButtonGroup"] button p,
.st-key-play_dis [data-testid="stButtonGroup"] button p { font-family: var(--mono) !important; font-size: 12px !important; }
.st-key-play_ch [data-testid="stButtonGroup"] button[aria-checked="true"] {
    background: var(--ecobg) !important; border-color: var(--eco) !important; color: var(--eco) !important;
}
.st-key-play_dis [data-testid="stButtonGroup"] button[aria-checked="true"] {
    background: var(--dnbg) !important; border-color: var(--rival) !important; color: var(--dn) !important;
}

.stButton button, .stFormSubmitButton button {
    min-height: 36px; border-radius: 6px; padding: 0 14px; border: 1px solid var(--outline) !important;
    background: var(--card); color: var(--ink);
}
.stButton button:hover { border-color: var(--ink) !important; background: var(--bg); }
.stButton button p { font: 500 12.5px/18px var(--font); }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
    background: var(--acc) !important; border-color: var(--acc) !important; color: var(--onacc) !important;
}
.stButton button[kind="primary"] p { color: var(--onacc) !important; }
.stButton button[kind="tertiary"] { min-height: 28px; border: 0 !important; background: transparent !important; color: var(--acc) !important; padding: 0; }
.stButton button[kind="tertiary"] p { color: var(--acc) !important; font: 500 12.5px var(--font); }

[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="textarea"], [data-testid="stDateInput"] > div > div {
    min-height: 38px; background: var(--card) !important; border: 1px solid var(--outline) !important; border-radius: 6px !important; color: var(--ink);
}
[data-baseweb="select"] *, [data-baseweb="input"] input { color: var(--ink) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] { background: var(--card) !important; color: var(--ink) !important; }

/* ===== Xat de l'Assistent (Pregunta-li): sempre mostra la caixa de text + suggeriments compactes a baix ===== */
.st-key-chat_fab { position: fixed; right: 24px; bottom: 24px; z-index: 1000; width: auto !important; }
.st-key-chat_fab button {
    min-height: 44px; padding: 0 18px; border-radius: 22px; background: var(--ink) !important; border: 1px solid var(--line) !important;
    color: var(--card) !important; box-shadow: 0 4px 12px rgba(0,0,0,.22);
}
.st-key-chat_fab button p, .st-key-chat_fab button span { color: var(--card) !important; font: 600 13px/18px var(--font); }

div[data-baseweb="popover"]:has([data-testid="stPopoverBody"]) {
    position: fixed !important; top: auto !important; left: auto !important; right: 24px !important; bottom: 78px !important; transform: none !important;
}
[data-testid="stPopoverBody"] {
    width: min(390px, calc(100vw - 32px)) !important; max-height: min(560px, calc(100vh - 110px)) !important;
    padding: 0 !important; display: flex; flex-direction: column; border-radius: 12px !important; overflow: hidden !important;
    background: var(--card) !important; border: 1px solid var(--line) !important; box-shadow: 0 12px 32px rgba(0,0,0,.25) !important;
}
[data-testid="stPopoverBody"] > div, [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"]:first-child { gap: 0 !important; }
[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"]:first-of-type {
    padding: 10px 14px; border-bottom: 1px solid var(--line); align-items: center; gap: 8px; background: var(--card);
}
[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"]:first-of-type .stButton { display: flex; justify-content: flex-end; }
.chat-h { display: flex; flex-direction: row; align-items: center; gap: 10px; }
.chat-h div { display: flex; flex-direction: column; }
.chat-h .av { width: 30px; height: 30px; border-radius: 50%; background: var(--acc); color: #fff; font: 600 13px/30px var(--mono); text-align: center; font-style: normal; flex: none; }
.chat-h b { font: 600 14px/17px var(--font); }
.chat-h span { font: 11px/14px var(--font); color: var(--mute); }

.st-key-chat_scroll { background: var(--bg); padding: 0 !important; border-radius: 0 !important; }
.st-key-chat_scroll > div { display: flex; flex-direction: column; gap: 8px; padding: 12px 14px !important; }
[class*="st-key-bu_"][class*="st-key-bu_"] {
    align-self: flex-end; width: fit-content !important; max-width: 84%; margin-left: auto;
    background: var(--acc); color: #fff; border-radius: 12px 12px 3px 12px; padding: 7px 11px !important;
}
[class*="st-key-bu_"] *, [class*="st-key-bu_"] p { color: #fff !important; margin: 0; font-size: 13px !important; line-height: 18px !important; }
[class*="st-key-ba_"][class*="st-key-ba_"] {
    width: fit-content !important; max-width: 90%; margin-right: auto;
    background: var(--card); border: 1px solid var(--line_soft); border-radius: 12px 12px 12px 3px; padding: 8px 12px !important;
    flex: 0 0 auto !important; height: auto !important; min-height: 0; overflow: visible !important;
}
[class*="st-key-bu_"] { flex: 0 0 auto !important; height: auto !important; overflow: visible !important; }
[class*="st-key-ba_"] > div, [class*="st-key-ba_"] [data-testid="stVerticalBlock"] { height: auto !important; overflow: visible !important; }
[class*="st-key-ba_"] p, [class*="st-key-ba_"] li { color: var(--ink) !important; margin: 0 0 3px; font-size: 13px !important; line-height: 18px !important; }
[class*="st-key-ba_"] p:last-child { margin-bottom: 0; }
[class*="st-key-ba_"] p.sim { font: 11px/14px var(--mono) !important; color: var(--mute) !important; margin-top: 4px; }

/* Safata de suggeriments compacta just a sobre de la caixa de text */
.st-key-chat_sugs {
    padding: 6px 10px !important; background: var(--card); border-top: 1px solid var(--line_soft);
}
.st-key-chat_sugs > div {
    display: flex !important; flex-direction: row !important; flex-wrap: wrap !important; gap: 4px !important;
}
.st-key-chat_sugs [data-testid="stElementContainer"] { width: auto !important; margin: 0 !important; }
.st-key-chat_sugs .stButton button {
    min-height: 24px !important; height: auto !important; padding: 3px 8px !important;
    border-radius: 6px !important; border: 1px solid var(--line) !important;
    background: var(--bg) !important; color: var(--ink2) !important; width: auto !important;
}
.st-key-chat_sugs .stButton button:hover {
    border-color: var(--acc) !important; color: var(--acc) !important; background: var(--accbg) !important;
}
.st-key-chat_sugs .stButton button p {
    font: 500 11px/14px var(--font) !important; color: inherit !important; text-align: left;
}

/* Caixa d'entrada sempre visible a baix */
[data-testid="stPopoverBody"] [data-testid="stChatInput"] {
    padding: 8px 10px 10px !important; border-top: 1px solid var(--line); background: var(--card) !important;
}
[data-testid="stChatInput"] { background: transparent !important; }
[data-testid="stChatInput"] > div > div, [data-testid="stChatInput"] > div > div > div { background: transparent !important; border: 0 !important; box-shadow: none !important; }
[data-testid="stChatInput"] > div {
    border-radius: 8px !important; background: var(--bg) !important; border: 1px solid var(--line) !important; min-height: 38px !important;
}
[data-testid="stChatInput"] textarea { font-size: 13px !important; line-height: 18px !important; }
[data-testid="stChatInput"] button { background: var(--acc) !important; border-radius: 6px !important; color: #fff !important; width: 28px; height: 28px; min-height: 0; }

.typing { display: inline-flex; gap: 4px; background: var(--card); border: 1px solid var(--line_soft); border-radius: 10px; padding: 8px 12px; }
.typing i { width: 6px; height: 6px; border-radius: 50%; background: var(--dot); animation: blink 1.2s infinite; }
.typing i:nth-child(2) { animation-delay: .2s; }
.typing i:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%, 60%, 100% { opacity: .3 } 30% { opacity: 1 } }


@media (max-width: 900px) {
    .ed-hero, .ed-kpis { grid-template-columns: 1fr; gap: 16px; }
    .ed-kpis > div + div { border-left: 0; border-top: 1px solid var(--line); }
    .verdict .ver, .scoreblk .big { font-size: 52px; }
    .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
/* ---- descripcions amb (?), etiquetes d'origen de la dada, desbordament, animacions ---- */
.ph { display: flex; align-items: flex-start; gap: 8px; flex-wrap: wrap; }
.ph .ct { margin: 0; overflow-wrap: anywhere; min-width: 0; flex: 1 1 12rem; }
.help { position: relative; display: inline-flex; flex: none; width: 16px; height: 16px; margin-top: 1px; align-items: center;
    justify-content: center; border: 1px solid var(--line); border-radius: 50%; font: 600 10px/1 var(--font);
    color: var(--mute); background: var(--card); cursor: help; }
.help:hover, .help:focus { color: var(--onacc); background: var(--acc); border-color: var(--acc); outline: none; }
.help .tip { display: none; position: absolute; z-index: 80; top: 22px; right: -8px; width: min(340px, 78vw);
    padding: 10px 12px; border-radius: 6px; background: var(--ink); color: var(--card); font: 400 11.5px/16px var(--font);
    text-align: left; box-shadow: 0 6px 18px rgba(15,23,42,.25); white-space: normal; }
.help:hover .tip, .help:focus .tip { display: block; animation: rise .15s ease-out; }
.chip { flex: none; font: 600 9.5px/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; padding: 3px 6px;
    border-radius: 4px; margin-top: 2px; white-space: nowrap; }
.chip.val { color: var(--acc); background: var(--accbg); border: 1px solid var(--acc); }
.chip.pub { color: var(--card); background: var(--rival); border: 1px solid var(--rival); }
.chip.camp { color: var(--mute); background: transparent; border: 1px solid var(--line); }
.chip.calc { color: var(--ink2); background: transparent; border: 1px dashed var(--mute); }
.tile .l, .lab, .sec h2, .intro h3 { overflow-wrap: anywhere; }
.tile .s, .kn, .note2, .scoreblk .note2, .verdict .line, .verdict .cand { overflow-wrap: anywhere; min-width: 0; }
.tile, .ed-kpis > div, .scoreblk, .verdict { min-width: 0; }
.verdict .chip.calc { color: #CBD5E1; border-color: #64748B; }
.verdict .ver { overflow-wrap: anywhere; }
.scoreblk .big { font-size: clamp(40px, 5.2vw, 64px); flex-wrap: wrap; }
.tile .v { overflow-wrap: anywhere; white-space: normal; }
div[data-testid="stVerticalBlock"][class*="st-key-card"] { overflow: visible; }
.tile .chip { display: inline-block; margin-top: 6px; }
.stApp p.src { font-size: 9px !important; line-height: 12px !important; }
.stApp p.cc, .stApp .sec p, .stApp .intro p { font-size: 10px !important; line-height: 13px !important; font-weight: 400 !important; }
.stApp p.ct { font-size: 14px !important; }
.lab + .chip, .lab .chip { margin-left: 6px; }
.ph .lab { margin: 0; }
table.lbt th { text-align: left; font: 600 10px var(--mono); letter-spacing: .05em; text-transform: uppercase;
    color: var(--mute); padding: 7px 10px; border-bottom: 1px solid var(--line); background: color-mix(in srgb, var(--bg) 55%, var(--card)); }
table.lbt th.n { text-align: right; }
table.lbt td.n { width: 90px; }
.kv, .scoreblk .big { min-width: 0; }
@keyframes rise { from { opacity: 0; transform: translateY(10px) } to { opacity: 1; transform: none } }
@keyframes grow { from { transform: scaleX(0) } to { transform: scaleX(1) } }
@media (prefers-reduced-motion: no-preference) {
    .ed-hero, .ed-kpis, .tiles, .note, .sec, .intro, .lbt, .tw, .pbxs,
    div[data-testid="stVerticalBlock"][class*="st-key-card"], [data-testid="stPlotlyChart"],
    [data-testid="stExpander"] details { animation: rise .45s ease-out both; }
    .cmp b, .pbx b, table.t .bar i, table.lbt td.g i { transform-origin: left; animation: grow .7s ease-out both; }
    @supports (animation-timeline: view()) {
        div[data-testid="stVerticalBlock"][class*="st-key-card"], .tw, .lbt, .ed-kpis, .tiles, .note, .sec,
        [data-testid="stExpander"] details {
            animation: rise linear both; animation-timeline: view(); animation-range: entry 0% entry 28%;
        }
    }
}

</style>
"""

def _css(theme: dict) -> str:
    vars_ = ";".join(f"--{k}:{v}" for k, v in theme.items())
    return (
        CSS_TEMPLATE.replace("__VARS__", vars_)
        .replace("__FONT__", FONT)
        .replace("__MONO__", MONO)
    )


def setup(title: str):
    st.set_page_config(page_title=title, page_icon=None, layout="wide", initial_sidebar_state="collapsed")
    st.markdown(_css(DARK if is_dark() else LIGHT), unsafe_allow_html=True)


def fmt(x, dec: int = 0, sign: bool = False) -> str:
    if x is None or x != x:
        return "—"
    s = f"{x:+,.{dec}f}" if sign else f"{x:,.{dec}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".").replace("-", "−")


def data(ts, any_: bool = True) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {MESOS[ts.month - 1]}" + (f" {ts.year}" if any_ else "")


def badge(text: str, kind: str = "nt") -> str:
    arrow = {"up": "▲ ", "dn": "▼ "}.get(kind, "")
    return f"<span class='badge {kind}'>{arrow}{text}</span>"


SRC = {"val": "Validació", "pub": "Públic", "camp": "Campus", "calc": "Calculat"}


def _short(text: str, n: int = 90) -> str:
    """Primera frase d'una descripció, sense etiquetes i retallada: és la línia petita sota el títol."""
    t = re.sub(r"<[^>]+>", "", text).strip()
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
    return first if len(first) <= n else first[: n - 1].rsplit(" ", 1)[0] + "…"


def tag(kind: str) -> str:
    """Etiqueta d'origen d'una xifra: val = validació (setmanes apartades de data/train), pub = rànquing públic,
    camp = dades del campus, calc = càlcul del dashboard a partir d'altres xifres."""
    return f"<span class='chip {kind}' title='{SRC[kind]}'>{SRC[kind]}</span>"


def helpicon(text: str) -> str:
    return f"<span class='help' tabindex='0'>?<span class='tip'>{text}</span></span>"


def head(title: str, caption: str | None = None, src: str | None = None) -> str:
    """Capçalera d'un gràfic o taula: títol, etiqueta d'origen, icona (?) amb l'explicació completa i, a sota,
    una descripció curta."""
    h = f"<div class='ph'><p class='ct'>{title}</p>{tag(src) if src else ''}{helpicon(caption) if caption else ''}</div>"
    return h + (f"<p class='cc'>{_short(caption)}</p>" if caption else "")


def heading(title: str, caption: str | None = None, src: str | None = None, where=st):
    where.markdown(head(title, caption, src), unsafe_allow_html=True)


def section(title: str, sub: str | None = None, src: str | None = None):
    st.markdown(
        f"<div class='sec'><div class='ph'><h2>{title}</h2>{tag(src) if src else ''}{helpicon(sub) if sub else ''}</div>"
        + (f"<p>{_short(sub, 120)}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


def intro(title: str, text: str, src: str | None = None):
    st.markdown(
        f"<div class='intro'><div class='ph'><h3>{title}</h3>{tag(src) if src else ''}{helpicon(text)}</div>"
        f"<p>{_short(text, 120)}</p></div>", unsafe_allow_html=True)


def panel(title: str, caption: str | None = None, key: str | None = None, src: str | None = None):
    box = st.container(border=True, key="card_" + (key or hashlib.md5(title.encode()).hexdigest()[:10]))
    box.markdown(head(title, caption, src), unsafe_allow_html=True)
    return box


def more(label: str):
    return st.expander(label, expanded=False)


def source(text: str, where=st):
    where.markdown(f"<p class='src'>{text}</p>", unsafe_allow_html=True)


def tiles(items: list[tuple], cols: int = 4, where=st, src: str | None = None):
    cells = "".join(
        f"<div class='tile'><div class='l'>{it[0]}</div>{tag(it[3] if len(it) > 3 and it[3] in SRC else src) if (src or (len(it) > 3 and it[3] in SRC)) else ''}"
        f"<div class='v'>{it[1]}</div><div class='s'>{it[2]}</div></div>"
        for it in items
    )
    where.markdown(f"<div class='tiles' style='--n:{cols}'>{cells}</div>", unsafe_allow_html=True)


facts = tiles


def note(head: str, body_html: str, where=st):
    where.markdown(f"<div class='note'><b>{head}</b> {body_html}</div>", unsafe_allow_html=True)


def checklist(items: list[tuple], where=st):
    lis = "".join(
        f"<li><span class='m' style='color:{'var(--up)' if ok else 'var(--dn)'}'>{'sí' if ok else 'no'}</span>"
        f"<span>{name}</span><span class='d'>{detail}</span></li>"
        for name, ok, detail in items
    )
    where.markdown(f"<ul class='chk'>{lis}</ul>", unsafe_allow_html=True)


def table(
    df: pd.DataFrame,
    names: dict,
    formats: dict | None = None,
    left: tuple = (),
    bars: dict | None = None,
    highlight=None,
    height: int | None = None,
    where=st,
):
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
    where.markdown(
        f"<div class='tw'{style_}><table class='t'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}"
        "</tbody></table></div>",
        unsafe_allow_html=True,
    )


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    t = DARK if is_dark() else LIGHT
    fig.update_layout(
        height=height,
        margin=dict(l=2, r=14, t=24 if legend else 8, b=2),
        separators=",.",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PFONT, size=11.5, color=t["ink2"]),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            title=None, font=dict(color=t["ink"], size=11)
        ),
        hoverlabel=dict(bgcolor=t["card"], font_color=t["ink"], bordercolor=t["line"], font_family=PFONT),
        bargap=0.3,
    )
    fig.update_xaxes(
        showgrid=False, linecolor=t["line"], linewidth=1, tickcolor=t["line"],
        zeroline=False, ticks="outside", ticklen=4,
        title_font=dict(size=11, color=t["mute"]), tickfont=dict(color=t["ink2"], size=11),
    )
    fig.update_yaxes(
        gridcolor=t["grid"], zeroline=False, showline=False,
        title_font=dict(size=11, color=t["mute"]), tickfont=dict(color=t["ink2"], size=11),
    )
    return fig


def show(fig: go.Figure, where=st, key: str | None = None):
    where.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def rgba(hex_: str, a: float) -> str:
    return f"rgba({int(hex_[1:3], 16)},{int(hex_[3:5], 16)},{int(hex_[5:7], 16)},{a})"