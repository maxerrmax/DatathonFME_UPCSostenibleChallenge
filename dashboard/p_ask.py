"""Pàgina 6 - Pregunta-li: un assistent que respon amb les dades del dashboard i pot simular horaris.

Fa servir l'API de Helmcode (compatible amb OpenAI). La clau va a .streamlit/secrets.toml, que no es versiona:

    [helmcode]
    api_key = "sk-..."
    base_url = "https://api.helmcode.com/v1"
    model = "qwen3.6"          # opcional
"""
from __future__ import annotations

import json
import re

import requests
import streamlit as st

import models
import p_play
import ui
from ui import INK_3, fmt

DEFAULT_MODEL = "qwen3.6"
TABS = ("Veredicte", "Cara a cara", "On són els punts", "Campus i xarxa", "Prova-ho tu")
SUGGESTED = [
    "Per què lliurem la v12 i no la v13, que tenia millor validació?",
    "Val la pena enviar la v14?",
    "Prova un horari que carregui d'11 a 16 h i descarregui de 19 a 23 h",
    "Quanta CO₂ evitaria una bateria de 8 MWh en lloc de la de 6?",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "simula_horari",
        "description": "Simula un horari fix de la bateria sobre el plec 1 de validació (1.848 hores, Mode Difícil) amb les "
                       "regles oficials i torna la puntuació. Cada dia carrega a potència màxima a les hores de "
                       "càrrega i descarrega a les de descàrrega.",
        "parameters": {"type": "object", "properties": {
            "hores_carrega": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 23}},
            "hores_descarrega": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 23}}},
            "required": ["hores_carrega", "hores_descarrega"]},
    },
}]


@st.cache_data(show_spinner=False)
def fact_pack() -> str:
    """Tot el que l'assistent pot saber: exactament les mateixes xifres que mostren les altres pàgines."""
    reg, raw, runs, sz = models.registry(), models.arena_raw(), models.runs(), models.sizing()
    T, _ = models.arena_table(raw)
    r2 = lambda v: round(float(v), 2)
    bc, bd, best = p_play.best_timetable()
    mine = next(x["public"] for x in reg["submissions"] if x["id"] == reg["selected"])
    pack = {
        "repte": {"mode": "Difícil", "bateria_kwh": 6000, "potencia_kw": 1500, "rendiment_cicle": 0.9, "solar": "x12",
                  "mercat_conegut_h": 3, "nota": "0,7 x CO2 evitat % + 0,3 x factura estalviada % - 1 per infraccio",
                  "limit_temps_s": reg["limits"]["timeout_s"], "hores_test": reg["limits"]["test_hours"]},
        "versio_recomanada": reg["selected"],
        "lliuraments": [{**{k: s.get(k) for k in ("id", "what", "validation", "co2", "cost", "infr", "arena", "note")
                            if s.get(k) is not None},
                         "public": s["public"] if s.get("public") is not None else "ENCARA NO ENVIADA: no te puntuacio publica"}
                        for s in reg["submissions"]],
        "regla_de_decisio_per_aspirant": {
            s["id"]: {"condicions": [{"condicio": n, "compleix": ok, "detall": d} for n, ok, d in g], "conclusio": c}
            for s in reg["submissions"] if s.get("arena") in T.index and s["id"] != reg["selected"]
            for g, c, _ in [models.gates(reg, T, runs, s["arena"])]},
        "ranquing_rivals": reg["leaderboard"],
        "posicio_al_ranquing": {
            "nota": "Foto del ranquing anterior al lliurament de la v12; els rivals poden haver millorat despres.",
            "la_nostra_puntuacio": mine,
            "equips_per_sobre": [{"equip": t["team"], "puntuacio": t["score"], "distancia": r2(t["score"] - mine)}
                                 for t in reg["leaderboard"]["teams"] if t["score"] > mine],
            "nombre_equips_per_sobre": sum(t["score"] > mine for t in reg["leaderboard"]["teams"])},
        "test_public_base": reg["public_test"],
        "optim_validacio_per_plec": {k: v["score"] for k, v in raw["oracle"].items()},
        "aspirants_vs_v12": [{"estrategia": n, "mitjana_3_plecs": r2(r["mean"]), "dif_setmanal_vs_v12": r2(r["wk_delta"]),
                              "interval95": r2(r["ci"]), "setmanes_guanyades": f"{int(r['wins'])}/{int(r['weeks'])}",
                              "pitjor_plec_vs_v12": r2(r["worst_fold"]), "significatiu": bool(r["significant"])}
                             for n, r in T.iterrows()],
        "execucions_plec1": {n: {"etiqueta": r["label"], "puntuacio": r["score"], "co2_pct": r["co2"], "cost_pct": r["cost"],
                                 "co2_t": round(r["co2_kg"] / 1000, 1), "factura_keur": round(r["cost_eur"] / 1000, 1),
                                 "ms_per_pas": r["ms"],
                                 "segons_test_complet": round(r["ms"] * reg["limits"]["test_hours"] / 1000) if r["ms"] else None,
                                 "puntuacio_per_setmana": r["weeks"]} for n, r in runs.items()},
        "escala_de_millores_validacio": reg["ladder"],
        "valor_informacio_perfecta": reg["perfect_info"],
        "idees_descartades": reg["dropped"],
        "pes_co2_vs_puntuacio": reg["ratio_sweep"],
        "dimensionament_previsio_perfecta": [{"solar": f"x{int(r.pv_scale)}", "bateria_mwh": r.cap_kwh / 1000, "puntuacio": r.score,
                                              "co2_evitat_t_any": round(r.co2_t_year), "factura_keur_any": round(r.eur_year / 1000)}
                                             for r in sz.itertuples()],
        "millor_horari_fix_trobat": {"hores_carrega": list(bc), "hores_descarrega": list(bd), "puntuacio": r2(best)},
    }
    return json.dumps(pack, ensure_ascii=False, separators=(",", ":"))


SYSTEM = """Ets l'assistent del dashboard de l'equip JAMM per al repte UPC Sostenible (Datathon FME 2026): una bateria al \
Campus Nord de la UPC que ha de reduir CO2 i factura. Ajudes a prendre decisions: quin agent lliurar, què millorar, com \
dimensionar la bateria.

Regles:
- Respon només amb les DADES de sota o amb resultats de l'eina simula_horari. Si la resposta no hi és, digues que el \
dashboard no té aquesta dada; no inventis xifres ni facis servir coneixement extern per a xifres.
- Respon en català, en 2 a 5 frases, començant per la conclusió. Dona les xifres exactes amb coma decimal (23,53).
- Si et demanen provar o proposar un horari de càrrega i descàrrega, crida simula_horari i compara el resultat amb la v12 \
(20,74) i l'òptim (22,34) del plec 1.
- Per a "val la pena enviar / substituir per la versió X", fes servir regla_de_decisio_per_aspirant[X]: cita les condicions que falla i la conclusió. No atribueixis a una versió la puntuació d'una altra.
- "validation" és la validació del plec 1; "public" és el rànquing públic real. Quan es contradiuen, mana el públic.
- No facis càlculs propis si la xifra ja és a les dades (distàncies, recomptes, conclusions).
- Acaba amb una línia "On mirar-ho:" i el nom d'una d'aquestes pestanyes, cap altra: Veredicte, Cara a cara, On són els punts, Campus i xarxa \
o Prova-ho tu.

DADES (JSON):
"""


def _cfg() -> dict | None:
    try:
        c = dict(st.secrets["helmcode"])
    except Exception:
        return None
    return c if c.get("api_key") else None


def _chat(cfg: dict, messages: list) -> dict:
    r = requests.post(cfg.get("base_url", "https://api.helmcode.com/v1").rstrip("/") + "/chat/completions",
                      headers={"Authorization": f"Bearer {cfg['api_key']}"},
                      json={"model": cfg.get("model", DEFAULT_MODEL), "messages": messages, "tools": TOOLS,
                            "temperature": 0.1, "max_tokens": 4000,          # el model raona abans de respondre
                            "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]


def answer(cfg: dict, history: list, question: str) -> tuple[str, list]:
    """Torna (resposta, simulacions fetes). Com a màxim 3 rondes d'eina."""
    msgs = [{"role": "system", "content": SYSTEM + fact_pack()}]
    for h in history[-4:]:
        msgs += [{"role": "user", "content": h["q"]}, {"role": "assistant", "content": h["a"]}]
    msgs.append({"role": "user", "content": question})
    sims = []
    for _ in range(4):
        m = _chat(cfg, msgs)
        calls = m.get("tool_calls") or []
        if not calls:
            text = re.sub(r"<think>.*?</think>", "", m.get("content") or "", flags=re.S).strip()
            m_ = re.search(r"\s*On mirar-ho:\s*(.*)\s*$", text)
            if m_ and not any(t in m_.group(1) for t in TABS):      # pestanya inventada: es treu la línia
                text = text[:m_.start()].rstrip()
            return text or "No he pogut generar una resposta.", sims
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        for c in calls:
            try:
                a = json.loads(c["function"]["arguments"] or "{}")
                ch = sorted({int(h) % 24 for h in a.get("hores_carrega", [])})
                dis = sorted({int(h) % 24 for h in a.get("hores_descarrega", [])} - set(ch))
                r = p_play._run(tuple(ch), tuple(dis))
                out = {"puntuacio": round(r["score"], 2), "co2_evitat_pct": round(r["co2"], 2),
                       "factura_estalviada_pct": round(r["cost"], 2), "cicles": round(r["cycles"]),
                       "puntuacio_per_setmana": {k: round(v, 2) for k, v in r["weeks"].items()}}
                sims.append({"ch": ch, "dis": dis, "score": r["score"]})
            except Exception as e:
                out = {"error": str(e)}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
    return "He fet massa simulacions seguides sense arribar a una conclusió; reformula la pregunta, sisplau.", sims


def _ask(q: str):
    st.session_state["ask_pending"] = q


def render():
    cfg = _cfg()
    st.markdown(
        "<div class='lead' style='margin-top:.8rem'><div><p class='h'>Pregunta-li al dashboard</p>"
        "<p class='t' style='max-width:74ch'>Un assistent que només coneix les xifres d'aquestes pàgines: lliuraments, "
        "validació, aspirants, marge fins a l'òptim i dimensionament. També pot simular un horari de càrrega amb el "
        "mateix simulador de “Prova-ho tu”. Si una dada no hi és, ho diu.</p></div></div>", unsafe_allow_html=True)
    if cfg is None:
        st.info("Falta la clau de l'API. Afegiu a `.streamlit/secrets.toml` la secció `[helmcode]` amb `api_key` "
                "(vegeu la capçalera de `dashboard/p_ask.py`).")
        return
    if models.traces() is None:
        st.info("Executeu `python dashboard/build_models.py` un cop per activar aquesta pàgina.")
        return
    hist = st.session_state.setdefault("ask_hist", [])

    with st.form("ask_form", clear_on_submit=True, border=False):
        c1, c2 = st.columns([8, 1])
        q = c1.text_input("Pregunta", placeholder="Per exemple: on perdem més punts respecte a l'òptim?",
                          label_visibility="collapsed")
        if c2.form_submit_button("Pregunta", type="primary", width="stretch") and q.strip():
            _ask(q.strip())
    cols = st.columns(len(SUGGESTED))
    for c, s in zip(cols, SUGGESTED):
        c.button(s, on_click=_ask, args=(s,), width="stretch", key=f"sug_{hash(s)}")

    pending = st.session_state.pop("ask_pending", None)
    if pending:
        with st.spinner("Consultant les dades …"):
            try:
                text, sims = answer(cfg, hist, pending)
            except Exception as e:
                text, sims = f"No he pogut contactar amb el model ({type(e).__name__}: {e}).", []
        hist.append({"q": pending, "a": text, "sims": sims})

    for h in reversed(hist):
        st.markdown(f"<p class='ct' style='margin-top:1.4rem'>{h['q']}</p>", unsafe_allow_html=True)
        st.markdown(h["a"])
        for s in h["sims"]:
            ui.source(f"Simulació feta: càrrega a les {', '.join(map(str, s['ch'])) or '—'} h; descàrrega a les "
                      f"{', '.join(map(str, s['dis'])) or '—'} h → {fmt(s['score'], 2)} punts.")
    if hist:
        st.button("Esborra la conversa", on_click=lambda: st.session_state.update(ask_hist=[]))
    st.markdown(f"<p class='src' style='margin-top:1.6rem'>Model: {cfg.get('model', DEFAULT_MODEL)} via Helmcode. "
                f"L'assistent rep {fmt(len(fact_pack()) / 1000, 1)} kB de dades del dashboard a cada pregunta i cap altra font; "
                "les seves respostes poden contenir errors de redacció, però les xifres són les de les altres pestanyes.</p>",
                unsafe_allow_html=True)
