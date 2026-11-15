"""
dashboard/app.py
Dashboard del repte UPC Sostenible (Datathon FME 2026): eina per comparar els agents de bateria del Mode Difícil
amb dades de validació i del rànquing públic, i per prendre decisions amb elles.

Des de l'arrel del repositori:
    streamlit run dashboard/app.py
Per refrescar-ho tot amb l'estat actual del repo:  python dashboard/refresh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import live  # noqa: E402
import p_ask  # noqa: E402
import p_campus  # noqa: E402
import p_decide  # noqa: E402
import p_front  # noqa: E402
import p_gap  # noqa: E402
import p_models  # noqa: E402
import p_verdict  # noqa: E402
import ui  # noqa: E402

st.session_state.setdefault("dark", False)


def _toggle_dark():
    st.session_state["dark"] = not st.session_state.get("dark", False)


ui.setup("Bateria del Campus Nord · Mode Difícil")
L = live.load()

c_title, c_dark = st.columns([8, 0.5], vertical_alignment="center")
with c_title:
    st.markdown("<div class='top'><h1>Agents de bateria · Mode Difícil</h1></div>", unsafe_allow_html=True)
    p_front.status()
with c_dark:
    with st.container(key="dark_btn"):
        st.button(
            "",
            icon=":material/light_mode:" if ui.is_dark() else ":material/dark_mode:",
            on_click=_toggle_dark,
            help="Canvia entre mode clar i mode fosc",
        )

tabs = st.tabs(["Resum", "Decidir", "Com triem", "Cara a cara", "On són els punts", "Campus i xarxa"])
for tab, page in zip(tabs, (p_front, p_decide, p_verdict, p_models, p_gap, p_campus)):
    with tab:
        page.render()

with st.container(key="chat_fab"):
    with st.popover("Pregunta-li", icon=":material/chat:"):
        p_ask.render()
