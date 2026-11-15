"""
dashboard/ui.py
Aspecte comú del dashboard: colors, CSS, estil dels gràfics i les peces HTML que es repeteixen.

Decisions de disseny (triades a partir del tema, no per defecte):
  * Paleta d'una instal·lació elèctrica: blau marí de quadre de comandament, blau UPC, groc de senyalització
    (el sol i l'avís) i un fons gris blavós. El color sempre vol dir alguna cosa: blau UPC = la versió que
    lliurem, taronja = l'aspirant o el rival, marí = l'òptim, gris = context.
  * Blocs amb cos: capçalera marina, pestanyes en forma de píndola, targetes blanques per a cada gràfic i un
    bloc blau gran per a la decisió, amb un indicador de bateria. Res de degradats ni ombres difuses.
  * Dues famílies: Bricolage Grotesque per a títols i xifres grans, IBM Plex Sans per al text.
  * Poca cosa a la vista: cada pàgina ensenya la decisió i dos o tres gràfics; la resta és a `ui.more()`
    (desplegables "Més detall").
  * El títol de cada gràfic diu la conclusió; el subtítol diu com es llegeix.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Entitats (fixes a tot el dashboard)
NAVY, SUN = "#0a2540", "#ffc233"
OURS, RIVAL, OPT = "#007bc0", "#e4572e", NAVY            # blau corporatiu UPC · taronja · marí
C_CO2, C_COST, C_SOLAR = "#12876f", "#c98a00", "#ffc233"
GRAY, GRAY_LIGHT = "#8593a3", "#cfd8e1"
GOOD, BAD = "#12876f", "#d93a2b"
INK, INK_2, INK_3 = "#0a2540", "#42566b", "#7b8a9a"
GRID, PAPER, CARD, LINE = "#e3e9ef", "#eef2f6", "#ffffff", "#d3dce6"
SEQ = [[0.0, "#e9f3fa"], [0.25, "#a9d3ec"], [0.5, "#55a6d4"], [0.75, "#0f72ae"], [1.0, "#0a2540"]]
HEAD = "'Bricolage Grotesque', 'IBM Plex Sans', 'Segoe UI', sans-serif"
FONT = "'IBM Plex Sans', 'Segoe UI', Arial, sans-serif"
PFONT = "IBM Plex Sans, Segoe UI, Arial"

MESOS = ["gen.", "febr.", "març", "abr.", "maig", "juny", "jul.", "ag.", "set.", "oct.", "nov.", "des."]
DIES = ["dl.", "dt.", "dc.", "dj.", "dv.", "ds.", "dg."]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp div {{ font-family: {FONT}; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding: 1.3rem 2.2rem 3rem; max-width: 1280px; }}
.stApp * {{ font-variant-numeric: tabular-nums lining-nums; }}
[data-testid='stHeaderActionElements'] {{ display: none; }}

/* capçalera */
.mast {{ position: relative; overflow: hidden; background: {NAVY}; color: #fff; border-radius: 22px;
        padding: 1.5rem 1.9rem 1.45rem; margin-bottom: 1rem; }}
.mast::after {{ content: ''; position: absolute; right: -70px; top: -90px; width: 290px; height: 290px; border-radius: 50%;
               border: 46px solid {SUN}; opacity: .95; }}
.mast .who {{ font-size: .8rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: {SUN}; }}
.mast h1 {{ font-family: {HEAD}; font-size: 2.7rem; line-height: 1.04; font-weight: 800; letter-spacing: -.025em;
           color: #fff; margin: .35rem 0 .45rem; padding: 0; max-width: 20ch; }}
.mast .sub {{ font-size: .98rem; line-height: 1.5; color: #c5d3e2; max-width: 62ch; margin: 0 0 .9rem; }}
.mast .chips {{ display: flex; flex-wrap: wrap; gap: .45rem; position: relative; z-index: 1; }}
.mast .chips span {{ border: 1.5px solid #3c5674; background: {NAVY}; border-radius: 999px; padding: .2rem .75rem;
                    font-size: .8rem; color: #dbe6f1; }}
.mast .chips b {{ color: #fff; font-weight: 700; }}

/* pestanyes */
.stTabs [role="tablist"] {{ gap: .45rem; border: 0 !important; box-shadow: none !important; flex-wrap: wrap; padding-bottom: .2rem; }}
.stTabs [role="tablist"]::before, .stTabs [role="tablist"]::after {{ display: none !important; }}
.stTabs [data-testid="stTab"] {{ background: {CARD}; border: 1.5px solid {LINE}; border-radius: 999px; padding: .42rem 1.1rem;
                                height: auto; color: {INK_2}; cursor: pointer; }}
.stTabs [data-testid="stTab"] p {{ font-size: .93rem; font-weight: 600; color: inherit; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{ background: {NAVY}; border-color: {NAVY}; color: #fff; }}
.stTabs .react-aria-SelectionIndicator {{ display: none !important; }}

/* seccions i targetes de gràfic */
.sec {{ margin: 2rem 0 .8rem; }}
.sec h2 {{ display: inline; font-family: {HEAD}; font-size: 1.5rem; font-weight: 800; letter-spacing: -.02em; color: {INK};
          margin: 0; padding: 0 .15rem; background: linear-gradient(transparent 62%, {SUN} 62%, {SUN} 92%, transparent 92%); }}
.sec p {{ font-size: .93rem; color: {INK_2}; margin: .45rem 0 0; max-width: 80ch; }}
div[data-testid="stVerticalBlock"][class*="st-key-card"] {{
    background: {CARD}; border: 1.5px solid {LINE} !important; border-radius: 18px !important; padding: 1rem 1.15rem .6rem !important; }}
.ct {{ font-family: {HEAD}; font-size: 1.1rem; line-height: 1.25; font-weight: 700; color: {INK}; margin: 0 0 .25rem; }}
.ct::before {{ content: ''; display: inline-block; width: .62rem; height: .62rem; background: {SUN}; border-radius: 3px;
              margin-right: .5rem; }}
.cc {{ font-size: .85rem; line-height: 1.45; color: {INK_2}; margin: 0 0 .3rem; max-width: 70ch; }}
.src {{ font-size: .77rem; line-height: 1.45; color: {INK_3}; margin: -.2rem 0 .5rem; }}

/* bloc de la decisió */
.hero {{ display: grid; grid-template-columns: auto minmax(0, 1fr) 250px; gap: 1.8rem; align-items: center;
        background: {OURS}; color: #fff; border-radius: 22px; padding: 1.5rem 1.8rem; margin-top: .9rem; }}
.hero .chip {{ display: inline-block; background: {SUN}; color: {NAVY}; font-size: .74rem; font-weight: 700; letter-spacing: .06em;
              text-transform: uppercase; border-radius: 999px; padding: .16rem .7rem; }}
.hero .big {{ font-family: {HEAD}; font-size: 6rem; line-height: .95; font-weight: 800; letter-spacing: -.05em; color: #fff; }}
.hero .cap {{ font-size: .82rem; color: #d6ecf9; }}
.hero .h {{ font-family: {HEAD}; font-size: 1.5rem; line-height: 1.2; font-weight: 700; margin: 0 0 .5rem; color: #fff; }}
.hero .t {{ font-size: .98rem; line-height: 1.5; color: #e6f3fb; margin: 0; max-width: 58ch; }}
.hero .t b {{ color: #fff; }}
.bat {{ display: flex; align-items: center; }}
.bat .body {{ flex: 1; height: 74px; border: 4px solid #fff; border-radius: 14px; padding: 5px; }}
.bat .fill {{ height: 100%; background: {SUN}; border-radius: 7px; display: flex; align-items: center; justify-content: center;
             font-family: {HEAD}; font-size: 1.7rem; font-weight: 800; color: {NAVY}; }}
.bat .nub {{ width: 9px; height: 30px; background: #fff; border-radius: 0 5px 5px 0; }}
.batcap {{ font-size: .8rem; line-height: 1.4; color: #e6f3fb; margin-top: .45rem; }}

/* xifres */
.tiles {{ display: grid; grid-template-columns: repeat(var(--n, 4), minmax(0, 1fr)); gap: .8rem; margin: .9rem 0 .3rem; }}
.tile {{ background: {CARD}; border: 1.5px solid {LINE}; border-radius: 18px; padding: .8rem 1rem .85rem; }}
.tile .l {{ font-size: .8rem; font-weight: 600; color: {INK_2}; }}
.tile .v {{ font-family: {HEAD}; font-size: 2.05rem; line-height: 1.12; font-weight: 800; letter-spacing: -.03em; color: {INK}; }}
.tile .v small {{ font-family: {FONT}; font-size: .85rem; font-weight: 500; color: {INK_2}; letter-spacing: 0; }}
.tile .s {{ font-size: .78rem; line-height: 1.4; color: {INK_3}; }}
.tile.navy {{ background: {NAVY}; border-color: {NAVY}; }}
.tile.navy .l, .tile.navy .s, .tile.navy .v small {{ color: #c5d3e2; }} .tile.navy .v {{ color: #fff; }}
.tile.sun {{ background: {SUN}; border-color: {SUN}; }}
.tile.sun .l, .tile.sun .s, .tile.sun .v small {{ color: #4d3d00; }}
.tile.green .v {{ color: {C_CO2}; }} .tile.ocre .v {{ color: {C_COST}; }} .tile.red .v {{ color: {BAD}; }}

/* introducció de pàgina i notes */
.intro {{ margin: .9rem 0 .6rem; }}
.intro h3 {{ font-family: {HEAD}; font-size: 1.6rem; font-weight: 800; letter-spacing: -.02em; color: {INK}; margin: 0 0 .25rem; padding: 0; }}
.intro p {{ font-size: .98rem; line-height: 1.5; color: {INK_2}; margin: 0; max-width: 80ch; }}
.note {{ background: #fff6d6; border: 1.5px solid {SUN}; border-radius: 16px; padding: .8rem 1.05rem; font-size: .93rem;
        line-height: 1.55; color: {INK}; margin: .7rem 0 .9rem; }}
.note b.h {{ font-family: {HEAD}; font-weight: 800; }}
.note ol, .note ul {{ margin: .3rem 0 0; padding-left: 1.15rem; }}
.note li {{ margin: .25rem 0; }}
.chk {{ list-style: none; padding: 0 !important; margin: .2rem 0 .4rem !important; }}
.chk li {{ display: flex; gap: .6rem; border-bottom: 1px solid {GRID}; padding: .34rem 0; margin: 0; font-size: .9rem; }}
.chk .m {{ width: 2rem; font-weight: 700; }}
.chk .d {{ margin-left: auto; color: {INK_3}; text-align: right; }}

/* desplegables "més detall" */
[data-testid="stExpander"] details {{ background: {CARD}; border: 1.5px solid {LINE}; border-radius: 16px; }}
[data-testid="stExpander"] summary {{ padding: .65rem 1rem; }}
[data-testid="stExpander"] summary p {{ font-size: .93rem; font-weight: 600; color: {INK}; }}

/* taules */
table.t {{ width: 100%; border-collapse: collapse; font-size: .87rem; }}
table.t, table.t th, table.t td {{ border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }}
table.t th {{ text-align: right; font-weight: 600; color: {INK_2}; font-size: .78rem; border-bottom: 2px solid {NAVY};
             padding: .3rem .5rem .3rem 0; vertical-align: bottom; position: sticky; top: 0; background: {CARD}; }}
table.t td {{ text-align: right; padding: .36rem .5rem .36rem 0; border-bottom: 1px solid {GRID}; white-space: nowrap; }}
table.t th.L, table.t td.L {{ text-align: left; white-space: normal; }}
table.t tr.hl td {{ font-weight: 700; background: #eaf5fb; }}
table.t td.L:first-child {{ min-width: 11rem; padding-left: .3rem; }}
table.t .bar {{ display: inline-block; width: 70px; height: 8px; background: {GRID}; border-radius: 4px; margin-right: .45rem;
               vertical-align: middle; overflow: hidden; }}
table.t .bar i {{ display: block; height: 100%; background: {OURS}; }}
.tw {{ overflow: auto; }}

/* controls */
[data-testid="stButtonGroup"] [role="toolbar"] {{ flex-wrap: wrap !important; overflow: visible !important; row-gap: .35rem; }}
.stButton button {{ height: auto; min-height: 2.4rem; }}
.stButton button p {{ white-space: normal; line-height: 1.25; font-size: .88rem; }}
.stButton button, .stFormSubmitButton button {{ border-radius: 999px; border: 1.5px solid {LINE}; font-weight: 600; }}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {{ background: {NAVY}; border-color: {NAVY}; color: #fff; }}
.foot {{ font-size: .77rem; line-height: 1.5; color: {INK_3}; margin-top: 2.4rem; padding-top: .7rem; border-top: 1.5px solid {LINE}; }}
div[data-testid="stCaptionContainer"] p {{ font-size: .78rem; color: {INK_3}; }}
@media (max-width: 900px) {{ .hero {{ grid-template-columns: 1fr; }} .tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
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
    return s.replace(",", " ").replace(".", ",").replace(" ", ".").replace("-", "−")


def data(ts, any_: bool = True) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {MESOS[ts.month - 1]}" + (f" {ts.year}" if any_ else "")


def section(title: str, sub: str | None = None):
    st.markdown(f"<div class='sec'><h2>{title}</h2>" + (f"<p>{sub}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


def intro(title: str, text: str):
    st.markdown(f"<div class='intro'><h3>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)


def panel(title: str, caption: str | None = None, key: str | None = None):
    """Targeta blanca per a un gràfic: títol = conclusió, subtítol = com es llegeix.

    La clau `st-key-card_…` és el que el CSS fa servir per pintar la targeta; es deriva del títol, i cal passar
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


def battery(pct: float, caption: str) -> str:
    """HTML d'una bateria plena al `pct` %, per al bloc de la decisió."""
    p = max(8.0, min(100.0, pct))
    return (f"<div><div class='bat'><div class='body'><div class='fill' style='width:{p:.0f}%'>{fmt(pct, 0)} %</div></div>"
            f"<div class='nub'></div></div><div class='batcap'>{caption}</div></div>")


def hero(big: str, cap: str, chip: str, head: str, text: str, right_html: str = ""):
    st.markdown(f"<div class='hero'><div><span class='chip'>{chip}</span><div class='big'>{big}</div>"
                f"<div class='cap'>{cap}</div></div><div><p class='h'>{head}</p><p class='t'>{text}</p></div>"
                f"{right_html or '<div></div>'}</div>", unsafe_allow_html=True)


def tiles(items: list[tuple], cols: int = 4, where=st):
    """Xifres: (etiqueta, valor_html, detall[, to]) amb to en {'', 'navy', 'sun', 'green', 'ocre', 'red'}."""
    cells = "".join(f"<div class='tile {(it[3] if len(it) > 3 else '')}'><div class='l'>{it[0]}</div>"
                    f"<div class='v'>{it[1]}</div><div class='s'>{it[2]}</div></div>" for it in items)
    where.markdown(f"<div class='tiles' style='--n:{cols}'>{cells}</div>", unsafe_allow_html=True)


facts = tiles


def note(head: str, body_html: str, where=st):
    where.markdown(f"<div class='note'><b class='h'>{head}</b> {body_html}</div>", unsafe_allow_html=True)


def checklist(items: list[tuple], where=st):
    """(nom, compleix, detall) -> llista de sí / no."""
    lis = "".join(f"<li><span class='m' style='color:{GOOD if ok else BAD}'>{'sí' if ok else 'no'}</span>"
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
        font=dict(family=PFONT, size=12.5, color=INK_2), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None, font=dict(color=INK)),
        hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor=NAVY, font_family=PFONT),
        bargap=0.22,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRAY, linewidth=1, tickcolor=GRAY, zeroline=False, ticks="outside",
                     ticklen=4, title_font=dict(size=12, color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, title_font=dict(size=12, color=INK_2))
    return fig


def show(fig: go.Figure, where=st, key: str | None = None):
    where.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def rgba(hex_: str, a: float) -> str:
    return f"rgba({int(hex_[1:3], 16)},{int(hex_[3:5], 16)},{int(hex_[5:7], 16)},{a})"
