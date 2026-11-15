"""
dashboard/ui.py
Aspecte comú del dashboard: colors, CSS, estil dels gràfics i les poques peces HTML que es repeteixen.

Decisions de disseny (totes triades, cap per defecte):
  * Pàgina blanca i tinta negra, com una secció de dades d'un diari. Sense targetes, ombres, degradats ni
    cantonades arrodonides: el que separa les coses és l'espai i un filet negre.
  * Una sola família tipogràfica, Libre Franklin (la Franklin Gothic dels gràfics de premsa), amb contrast de pes.
  * El color vol dir alguna cosa: blau UPC = l'agent que lliurem, vermell = l'aspirant o l'avís,
    negre = l'òptim, gris = context.
  * El títol de cada gràfic diu la conclusió; el subtítol diu com es llegeix.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Entitats (fixes a tot el dashboard)
OURS, RIVAL, OPT = "#007bc0", "#d1352b", "#121212"       # blau corporatiu UPC · vermell · tinta
C_CO2, C_COST, C_SOLAR = "#0f6b6b", "#b07a12", "#e0a100"
GRAY, GRAY_LIGHT = "#8c8c8c", "#d4d4d4"
GOOD, BAD = "#1d7a46", "#d1352b"
INK, INK_2, INK_3 = "#121212", "#4a4a4a", "#8c8c8c"
GRID, PAPER = "#e4e4e4", "#ffffff"
SEQ = [[0.0, "#e6f2f9"], [0.25, "#a6d0e8"], [0.5, "#55a6d4"], [0.75, "#0f72ae"], [1.0, "#003f68"]]
FONT = "'Libre Franklin', 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif"
PFONT = "Libre Franklin, Franklin Gothic Medium, Arial"

MESOS = ["gen.", "febr.", "març", "abr.", "maig", "juny", "jul.", "ag.", "set.", "oct.", "nov.", "des."]
DIES = ["dl.", "dt.", "dc.", "dj.", "dv.", "ds.", "dg."]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;700;800;900&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp div {{ font-family: {FONT}; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding: 1.6rem 2.2rem 3rem; max-width: 1280px; }}
.stApp * {{ font-variant-numeric: tabular-nums lining-nums; }}

.mast {{ border-top: 6px solid {INK}; padding-top: .55rem; margin-bottom: .5rem; }}
.mast .dl {{ display: flex; justify-content: space-between; gap: 2rem; font-size: .8rem; color: {INK_2};
            border-bottom: 1px solid {INK}; padding-bottom: .45rem; }}
.mast .dl b {{ color: {INK}; font-weight: 700; }}
.mast h1 {{ font-size: 2.9rem; line-height: 1.02; font-weight: 900; letter-spacing: -.03em; color: {INK};
           margin: .7rem 0 .2rem; padding: 0; }}
.mast .sub {{ font-size: 1.02rem; line-height: 1.45; color: {INK_2}; max-width: 70ch; margin: 0; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {INK}; }}
.stTabs [data-baseweb="tab"] {{ background: none; padding: .5rem 0 .45rem; height: auto; color: {INK_3}; }}
.stTabs [data-baseweb="tab"] p {{ font-size: .95rem; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [aria-selected="true"] p {{ font-weight: 800; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {OURS}; height: 4px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

.sec {{ border-top: 3px solid {INK}; margin: 2.6rem 0 1.1rem; padding-top: .5rem; }}
[data-testid='stHeaderActionElements'] {{ display: none; }}
.sec h2 {{ font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; color: {INK}; margin: 0; padding: 0; }}
.sec p {{ font-size: .95rem; color: {INK_2}; margin: .2rem 0 0; max-width: 78ch; }}

.ct {{ border-top: 1px solid {INK}; padding-top: .45rem; font-size: 1.06rem; line-height: 1.25; font-weight: 700;
      color: {INK}; margin: 0 0 .2rem; }}
.cc {{ font-size: .86rem; line-height: 1.4; color: {INK_2}; margin: 0 0 .3rem; max-width: 64ch; }}
.src {{ font-size: .76rem; line-height: 1.4; color: {INK_3}; margin: -.3rem 0 .4rem; }}

.lead {{ display: flex; gap: 1.4rem; align-items: flex-start; margin-top: 1rem; }}
.lead > div:first-child {{ flex: 0 0 auto; }}
.lead .n {{ white-space: nowrap; font-size: 5.6rem; line-height: .86; font-weight: 900; letter-spacing: -.055em; color: {OURS}; }}
.lead .u {{ font-size: .8rem; color: {INK_2}; margin-top: .55rem; }}
.lead .h {{ font-size: 1.5rem; line-height: 1.2; font-weight: 800; letter-spacing: -.015em; margin: 0 0 .5rem; color: {INK}; }}
.lead .t {{ font-size: 1rem; line-height: 1.5; color: {INK_2}; margin: 0; max-width: 46ch; }}

.rows {{ border-top: 1px solid {INK}; margin-top: 1rem; }}
.rows .r {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
           border-bottom: 1px solid {GRID}; padding: .42rem 0; }}
.rows .l {{ font-size: .9rem; color: {INK}; }}
.rows .l small {{ display: block; font-size: .77rem; color: {INK_3}; }}
.rows .v {{ font-size: 1.25rem; font-weight: 800; letter-spacing: -.02em; white-space: nowrap; color: {INK}; }}
.rows .v small {{ font-size: .8rem; font-weight: 500; color: {INK_2}; letter-spacing: 0; }}
.rows .v.bad {{ color: {BAD}; }} .rows .v.good {{ color: {GOOD}; }}

.facts {{ display: grid; grid-template-columns: repeat(var(--n, 4), minmax(0, 1fr)); gap: 0 1.8rem; margin-top: .6rem; }}
.fact {{ border-top: 1px solid {INK}; padding: .45rem 0 .7rem; }}
.fact .l {{ font-size: .84rem; color: {INK_2}; }}
.fact .v {{ font-size: 2rem; line-height: 1.1; font-weight: 800; letter-spacing: -.03em; color: {INK}; }}
.fact .v small {{ font-size: .85rem; font-weight: 500; color: {INK_2}; letter-spacing: 0; }}
.fact .s {{ font-size: .78rem; line-height: 1.4; color: {INK_3}; }}

.note {{ font-size: .95rem; line-height: 1.55; color: {INK}; margin: .5rem 0 1rem; max-width: 92ch; }}
.note b.h {{ font-weight: 800; }}
.note ol, .note ul {{ margin: .3rem 0 0; padding-left: 1.15rem; }}
.note li {{ margin: .25rem 0; }}
.chk {{ list-style: none; padding: 0 !important; }}
.chk li {{ display: flex; gap: .6rem; border-bottom: 1px solid {GRID}; padding: .32rem 0; margin: 0; font-size: .9rem; }}
.chk .m {{ width: 1.9rem; font-weight: 800; }}
.chk .d {{ margin-left: auto; color: {INK_3}; text-align: right; }}

table.t {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
table.t, table.t th, table.t td {{ border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }}
table.t th {{ text-align: right; font-weight: 600; color: {INK_2}; font-size: .78rem; border-bottom: 1px solid {INK};
             padding: .3rem .5rem .3rem 0; vertical-align: bottom; position: sticky; top: 0; background: {PAPER}; }}
table.t td {{ text-align: right; padding: .34rem .5rem .34rem 0; border-bottom: 1px solid {GRID}; white-space: nowrap; }}
table.t th.L, table.t td.L {{ text-align: left; white-space: normal; }}
table.t tr.hl td {{ font-weight: 800; }}
table.t td.L:first-child {{ min-width: 13rem; }}
table.t .bar {{ display: inline-block; width: 70px; height: 7px; background: #efefef; margin-right: .45rem; vertical-align: middle; }}
table.t .bar i {{ display: block; height: 100%; background: {INK}; }}
.tw {{ overflow: auto; }}
.foot {{ font-size: .76rem; line-height: 1.5; color: {INK_3}; border-top: 1px solid {INK}; margin-top: 3rem; padding-top: .6rem; }}
div[data-testid="stCaptionContainer"] p {{ font-size: .78rem; color: {INK_3}; }}
</style>
"""


def setup(title: str):
    st.set_page_config(page_title=title, page_icon=None, layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)


def fmt(x, dec: int = 0, sign: bool = False) -> str:
    """Format català: 1.848,5 (coma decimal, punt de milers)."""
    if x is None or x != x:
        return "—"
    s = f"{x:+,.{dec}f}" if sign else f"{x:,.{dec}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".").replace("-", "−")


def data(ts, any_: bool = True) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {MESOS[ts.month - 1]}" + (f" {ts.year}" if any_ else "")


def section(title: str, sub: str | None = None):
    st.markdown(f"<div class='sec'><h2>{title}</h2>" + (f"<p>{sub}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


def panel(title: str, caption: str | None = None):
    box = st.container()
    box.markdown(f"<p class='ct'>{title}</p>" + (f"<p class='cc'>{caption}</p>" if caption else ""), unsafe_allow_html=True)
    return box


def source(text: str, where=st):
    where.markdown(f"<p class='src'>{text}</p>", unsafe_allow_html=True)


def facts(items: list[tuple], cols: int = 4, where=st):
    cells = "".join(f"<div class='fact'><div class='l'>{l}</div><div class='v'>{v}</div><div class='s'>{s}</div></div>"
                    for l, v, s in items)
    where.markdown(f"<div class='facts' style='--n:{cols}'>{cells}</div>", unsafe_allow_html=True)


def rows(items: list[tuple], where=st):
    """Llista d'indicadors: (etiqueta, detall, valor_html, to) amb to en {'', 'good', 'bad'}."""
    html = "".join(f"<div class='r'><div class='l'>{l}<small>{d}</small></div><div class='v {t}'>{v}</div></div>"
                   for l, d, v, t in items)
    where.markdown(f"<div class='rows'>{html}</div>", unsafe_allow_html=True)


def note(head: str, body_html: str, where=st):
    where.markdown(f"<div class='note'><b class='h'>{head}</b> {body_html}</div>", unsafe_allow_html=True)


def table(df: pd.DataFrame, names: dict, formats: dict | None = None, left: tuple = (), bars: dict | None = None,
          highlight=None, height: int | None = None, where=st):
    """Taula HTML: xifres a la dreta amb coma decimal, filets fins i, si es vol, una barra dins la cel·la."""
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
    style = f" style='max-height:{height}px'" if height else ""
    where.markdown(f"<div class='tw'{style}><table class='t'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}"
                   "</tbody></table></div>", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=2, r=14, t=26 if legend else 10, b=2), separators=",.",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PFONT, size=12.5, color=INK_2), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None, font=dict(color=INK)),
        hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor=INK, font_family=PFONT),
        bargap=0.22,
    )
    fig.update_xaxes(showgrid=False, linecolor=INK, linewidth=1, tickcolor=INK, zeroline=False, ticks="outside",
                     ticklen=4, title_font=dict(size=12, color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, title_font=dict(size=12, color=INK_2))
    return fig


def show(fig: go.Figure, where=st, key: str | None = None):
    where.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def rgba(hex_: str, a: float) -> str:
    return f"rgba({int(hex_[1:3], 16)},{int(hex_[3:5], 16)},{int(hex_[5:7], 16)},{a})"
