"""
dashboard/app.py
Dashboard de decisió del repte UPC Sostenible (Datathon FME 2026): quin agent de bateria lliurem,
per què, i què queda per guanyar.

Des de l'arrel del repositori:
    streamlit run dashboard/app.py

Les pàgines de comparació de models llegeixen dashboard/submissions.json (mantingut a mà) i
dashboard/model_data/ (generat amb `python dashboard/build_models.py`).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models  # noqa: E402
import p_campus  # noqa: E402
import p_gap  # noqa: E402
import p_models  # noqa: E402
import p_ask  # noqa: E402
import p_play  # noqa: E402
import p_verdict  # noqa: E402
import ui  # noqa: E402

ui.setup("Bateria del Campus Nord · quin agent lliurem?")
reg = models.registry()

st.markdown(
    """<div class='mast'>
      <div class='dl'><span><b>Equip JAMM</b> · Repte UPC Sostenible · Datathon FME 2026</span>
        <span>Mode Difícil: bateria de <b>6.000 kWh</b> i <b>±1.500 kW</b>, solar ×12, mercat conegut amb 3 h</span></div>
      <h1>Quin agent de bateria lliurem?</h1>
      <p class='sub'>La bateria del Campus Nord ha de carregar quan l'electricitat és neta i barata i descarregar quan
        és bruta i cara. La nota és 0,7 × el CO₂ evitat (%) més 0,3 × la factura estalviada (%),
        menys un punt per infracció.</p>
    </div>""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Veredicte", "Cara a cara", "On són els punts", "Campus i xarxa", "Prova-ho tu", "Pregunta-li"])
for tab, page in zip(tabs, (p_verdict, p_models, p_gap, p_campus, p_play, p_ask)):
    with tab:
        page.render()

built = models.DATA / "runs.json"
stamp = ui.data(datetime.fromtimestamp(built.stat().st_mtime)) if built.exists() else "sense generar"
st.markdown(
    f"<div class='foot'>Puntuacions públiques: rànquing del repte; rivals a {reg['leaderboard']['as_of']} "
    f"(dashboard/submissions.json). Validació: simulador oficial sobre setmanes apartades de data/train, amb els models "
    f"reentrenats sense aquestes setmanes (generat el {stamp}). Dades del campus: data/train.</div>",
    unsafe_allow_html=True,
)
