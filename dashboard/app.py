"""
dashboard/app.py
Dashboard de decisió del repte UPC Sostenible (Datathon FME 2026):
quin agent de bateria lliurem, per què, i què queda per guanyar.

Des de l'arrel del repositori:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import live  # noqa: E402
import models  # noqa: E402
import p_ask  # noqa: E402
import p_campus  # noqa: E402
import p_decide  # noqa: E402
import p_front  # noqa: E402
import p_gap  # noqa: E402
import p_models  # noqa: E402
import p_play  # noqa: E402
import p_verdict  # noqa: E402
import ui  # noqa: E402

st.session_state.setdefault("explain", True)
st.session_state.setdefault("dark", False)


def _toggle_dark():
    st.session_state["dark"] = not st.session_state.get("dark", False)


ui.setup("Bateria del Campus Nord · quin agent lliurem?")
L = live.load()
reg = L.registry

c_title, c_exp, c_dark = st.columns([6.5, 1.3, 0.5], vertical_alignment="center")
with c_title:
    st.markdown("<div class='top'><h1>Quin agent de bateria lliurem?</h1></div>", unsafe_allow_html=True)
    p_front.status()
with c_exp:
    st.checkbox("Explicacions", key="explain")
with c_dark:
    with st.container(key="dark_btn"):
        st.button(
            "",
            icon=":material/light_mode:" if ui.is_dark() else ":material/dark_mode:",
            on_click=_toggle_dark,
            help="Canvia entre mode clar i mode fosc",
        )

tabs = st.tabs([
    "Veredicte",
    "Decidir",
    "Com triem",
    "Cara a cara",
    "On són els punts",
    "Campus i xarxa",
    "Prova-ho tu",
])
for tab, page in zip(tabs, (p_front, p_decide, p_verdict, p_models, p_gap, p_campus, p_play)):
    with tab:
        page.render()

with st.container(key="chat_fab"):
    with st.popover("Pregunta-li", icon=":material/chat:"):
        p_ask.render()

built = models.DATA / "runs.json"
stamp = ui.data(datetime.fromtimestamp(built.stat().st_mtime)) if built.exists() else "sense generar"
as_of_str = ui.data(L.as_of) if L.as_of else reg["leaderboard"]["as_of"]
st.markdown(
    f"<div class='foot'>Puntuacions públiques: rànquing del repte; rivals a {as_of_str} "
    f"({L.source}). Validació: simulador oficial sobre setmanes apartades de data/train, amb els models "
    f"reentrenats sense aquestes setmanes (generat el {stamp}). Dades del campus: data/train.</div>",
    unsafe_allow_html=True,
)