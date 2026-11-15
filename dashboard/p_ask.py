"""
Pàgina 6 - Pregunta-li: un assistent que respon amb les dades del dashboard i pot simular horaris.
Suggeriments compactes ancorats just a sobre del camp de text, que queda sempre visible en obrir el xat.
"""
from __future__ import annotations

import json
import re

import requests
import streamlit as st

import live
import models
import p_play
import ui
from ui import fmt

DEFAULT_MODEL = "qwen3.6"
TABS = (
    "Resum",
    "Decidir",
    "Com triem",
    "Cara a cara",
    "On són els punts",
    "Campus i xarxa",
)


def suggested() -> list[str]:
    ch = models.choose(live.load().registry)
    me, pend = ch["chosen"], ch["pending"]
    fooled = ch["table"][(ch["table"]["pred"] > 0) & (ch["table"]["d_pub"] < 0)]
    other = fooled.index[0] if len(fooled) else ch["anchor"]
    return [
        f"Per què {me} i no {other}?",
        f"Enviar la {pend[0]['id']}?" if pend else "On guanyar més punts?",
        "Simula càrrega 11–16h i descàrrega 19–23h",
        "8 MWh vs 6 MWh de bateria?",
    ]


TOOLS = [{
    "type": "function",
    "function": {
        "name": "simula_horari",
        "description": (
            "Simula un horari fix de la bateria sobre el plec 1 de validació (1.848 hores, Mode Difícil) amb les "
            "regles oficials i torna la puntuació. Cada dia carrega a potència màxima a les hores de "
            "càrrega i descarrega a les de descàrrega."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hores_carrega": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 23}},
                "hores_descarrega": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 23}},
            },
            "required": ["hores_carrega", "hores_descarrega"],
        },
    },
}]


@st.cache_data(ttl=30, show_spinner=False)
def fact_pack() -> str:
    reg, raw, runs, sz = live.load().registry, models.arena_raw(), models.runs(), models.sizing()
    T, _ = models.arena_table(raw)
    r2 = lambda v: round(float(v), 2)
    bc, bd, best = p_play.best_timetable()
    ch = models.choose(reg)
    me = ch["chosen"]
    mine = next(x["public"] for x in reg["submissions"] if x["id"] == me)
    pack = {
        "repte": {
            "mode": "Difícil", "bateria_kwh": 6000, "potencia_kw": 1500, "rendiment_cicle": 0.9, "solar": "x12",
            "mercat_conegut_h": 3, "nota": "0,7 x CO2 evitat % + 0,3 x factura estalviada % - 1 per infraccio",
            "limit_temps_s": reg["limits"]["timeout_s"], "hores_test": reg["limits"]["test_hours"],
        },
        "versio_recomanada": me,
        "com_es_tria_la_versio": {
            "nota": (
                "Tria automatica del dashboard. La classificacio final es fa al test PRIVAT, que el ranquing public no "
                "mostra; per aixo no es tria la nota publica mes alta sino la millor estimacio de l'avantatge al test "
                "privat, combinant la nota publica (soroll sd_public) i la validacio traduida a punts de test "
                "(pendent beta, soroll sd_validacio). Nomes son elegibles versions amb nota publica. En cas d'empat "
                "guanya la versio que n'amplia una altra amb un canvi que nomes actua a la part privada."
            ),
            "referencia": ch["anchor"], "beta": r2(ch["beta"]), "sd_validacio": r2(ch["sd_val"]), "sd_public": ch["sd_pub"],
            "empata_amb": ch["tie"],
            "avantatge_estimat_al_test_privat": {
                i: {"estimacio": r2(r["est"]), "interval95": r2(1.96 * r["sd"]), "elegible": bool(r["eligible"])}
                for i, r in ch["table"].iterrows()
            },
            "versions_no_enviades": [
                {
                    "id": p["id"], "estimacio_nomes_validacio": r2(p["pred"]),
                    "regla": f"enviar-la un cop i substituir la {me} nomes si fa mes de {p['threshold']:.2f} al public",
                }
                for p in ch["pending"]
            ],
        },
        "lliuraments": [
            {
                **{k: s.get(k) for k in ("id", "what", "validation", "co2", "cost", "infr", "arena", "note") if s.get(k) is not None},
                "public": s["public"] if s.get("public") is not None else "ENCARA NO ENVIADA: no te puntuacio publica",
            }
            for s in reg["submissions"]
        ],
        "regla_de_decisio_per_aspirant": {
            s["id"]: {"condicions": [{"condicio": n, "compleix": ok, "detall": d} for n, ok, d in g], "conclusio": c}
            for s in reg["submissions"]
            if s.get("arena") in T.index and s["id"] != me
            for g, c, _ in [models.gates(reg, T, runs, s["arena"])]
        },
        "ranquing_rivals": reg["leaderboard"],
        "posicio_al_ranquing": {
            "nota": f"Ranquing public a {reg['leaderboard']['as_of']}; la classificacio final es fara amb la part privada.",
            "la_nostra_puntuacio": mine,
            "equips_per_sobre": [
                {"equip": t["team"], "puntuacio": t["score"], "distancia": r2(t["score"] - mine)}
                for t in reg["leaderboard"]["teams"]
                if t["score"] > mine
            ],
            "nombre_equips_per_sobre": sum(t["score"] > mine for t in reg["leaderboard"]["teams"]),
        },
        "test_public_base": reg["public_test"],
        "optim_validacio_per_plec": {k: v["score"] for k, v in raw["oracle"].items()},
        "aspirants_vs_v12": [
            {
                "estrategia": n, "mitjana_3_plecs": r2(r["mean"]), "dif_setmanal_vs_v12": r2(r["wk_delta"]),
                "interval95": r2(r["ci"]), "setmanes_guanyades": f"{int(r['wins'])}/{int(r['weeks'])}",
                "pitjor_plec_vs_v12": r2(r["worst_fold"]), "significatiu": bool(r["significant"]),
            }
            for n, r in T.iterrows()
        ],
        "execucions_plec1": {
            n: {
                "etiqueta": r["label"], "puntuacio": r["score"], "co2_pct": r["co2"], "cost_pct": r["cost"],
                "co2_t": round(r["co2_kg"] / 1000, 1), "factura_keur": round(r["cost_eur"] / 1000, 1),
                "ms_per_pas": r["ms"],
                "segons_test_complet": round(r["ms"] * reg["limits"]["test_hours"] / 1000) if r["ms"] else None,
                "puntuacio_per_setmana": r["weeks"],
            }
            for n, r in runs.items()
        },
        "escala_de_millores_validacio": reg["ladder"],
        "valor_informacio_perfecta": reg["perfect_info"],
        "idees_descartades": reg["dropped"],
        "pes_co2_vs_puntuacio": reg["ratio_sweep"],
        "dimensionament_previsio_perfecta": [
            {
                "solar": f"x{int(r.pv_scale)}", "bateria_mwh": r.cap_kwh / 1000, "puntuacio": r.score,
                "co2_evitat_t_any": round(r.co2_t_year), "factura_keur_any": round(r.eur_year / 1000),
            }
            for r in sz.itertuples()
        ],
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
- Si et demanen provar o proposar un horari de càrrega i descàrrega, crida simula_horari i compara el resultat amb \
l'agent v12 i amb l'òptim del plec 1 (les seves puntuacions són a execucions_plec1).
- Per a "quina versió lliurem / per què aquesta", fes servir com_es_tria_la_versio: la tria és automàtica i pensa en el test privat; la nota pública més alta no és necessàriament el millor model.
- Per a "val la pena enviar / substituir per la versió X": si X és a versions_no_enviades, cita la seva regla; si no, fes servir regla_de_decisio_per_aspirant[X] (condicions que falla i conclusió). No atribueixis a una versió la puntuació d'una altra.
- "validation" és la validació del plec 1; "public" és el rànquing públic real (només una part del test). Cap de les dues mana sola: digues sempre el grau d'incertesa.
- No facis càlculs propis si la xifra ja és a les dades (distàncies, recomptes, conclusions).
- Acaba amb una línia "On mirar-ho:" i el nom d'una d'aquestes pestanyes, cap altra: Resum, Decidir, Com triem, Cara a cara, On són els punts o Campus i xarxa.

DADES (JSON):
"""


def _cfg() -> dict | None:
    try:
        c = dict(st.secrets["helmcode"])
    except Exception:
        return None
    return c if c.get("api_key") else None


def _chat(cfg: dict, messages: list) -> dict:
    r = requests.post(
        cfg.get("base_url", "https://api.helmcode.com/v1").rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
        json={
            "model": cfg.get("model", DEFAULT_MODEL),
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.1,
            "max_tokens": 4000,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]


def answer(cfg: dict, history: list, question: str) -> tuple[str, list]:
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
            if m_ and not any(t in m_.group(1) for t in TABS):
                text = text[: m_.start()].rstrip()
            return text or "No he pogut generar una resposta.", sims
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        for c in calls:
            try:
                a = json.loads(c["function"]["arguments"] or "{}")
                ch = sorted({int(h) % 24 for h in a.get("hores_carrega", [])})
                dis = sorted({int(h) % 24 for h in a.get("hores_descarrega", [])} - set(ch))
                r = p_play._run(tuple(ch), tuple(dis))
                out = {
                    "puntuacio": round(r["score"], 2),
                    "co2_evitat_pct": round(r["co2"], 2),
                    "factura_estalviada_pct": round(r["cost"], 2),
                    "cicles": round(r["cycles"]),
                    "puntuacio_per_setmana": {k: round(v, 2) for k, v in r["weeks"].items()},
                }
                sims.append({"ch": ch, "dis": dis, "score": r["score"]})
            except Exception as e:
                out = {"error": str(e)}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
    return "He fet massa simulacions seguides sense arribar a una conclusió; reformula la pregunta, sisplau.", sims


def _ask(q: str):
    st.session_state["ask_pending"] = q


def _user(text: str, i: int):
    with st.container(key=f"bu_{i}"):
        st.markdown(text)


def _bot(text: str, sims: list, key: str):
    with st.container(key=f"ba_{key}"):
        st.markdown(text)
        for s in sims:
            st.markdown(
                "<p class='sim'>"
                + f"Simulació: càrrega {', '.join(map(str, s['ch'])) or '—'}h · descàrrega "
                f"{', '.join(map(str, s['dis'])) or '—'}h → {fmt(s['score'], 2)} pts</p>",
                unsafe_allow_html=True,
            )


TYPING = "<div class='typing'><i></i><i></i><i></i></div>"

SCROLL_JS = """<script>
(function(){const r=document.querySelector('.st-key-chat_scroll');if(!r)return;
const el=[r,...r.querySelectorAll('*')].find(e=>e.scrollHeight>e.clientHeight+4&&getComputedStyle(e).overflowY!=='visible');
if(el)el.scrollTop=el.scrollHeight;})();
</script>"""


def render():
    """Xat compacte amb suggeriments petits just a sobre de l'entrada de text (sempre visible)."""
    cfg = _cfg()
    if cfg is None:
        st.info(
            "Falta la clau de l'API. Afegiu a `.streamlit/secrets.toml` la secció `[helmcode]` amb `api_key` "
            "(vegeu la capçalera de `dashboard/p_ask.py`)."
        )
        return
    if models.traces() is None:
        st.info("Executeu `python dashboard/build_models.py` un cop per activar l'assistent.")
        return
    hist = st.session_state.setdefault("ask_hist", [])

    h1, h2 = st.columns([3, 1], vertical_alignment="center")
    h1.markdown("<div class='chat-h'><i class='av'>A</i><div><b>Assistent</b><span>En línia</span></div></div>", unsafe_allow_html=True)
    h2.button("Esborra", key="chat_clear", type="tertiary", on_click=lambda: st.session_state.update(ask_hist=[]))

    # Alçada reduïda perquè tant la safata de suggeriments com el camp de text quedin sempre dins del popover
    box = st.container(height=290, border=False, key="chat_scroll")
    sugs_dock = st.container(border=False, key="chat_sugs")
    q = st.chat_input("Escriu un missatge…", key="chat_in")
    pending = st.session_state.pop("ask_pending", None) or q

    with sugs_dock:
        for idx_s, s in enumerate(suggested()):
            st.button(s, on_click=_ask, args=(s,), key=f"sug_{idx_s}")

    with box:
        with st.container(key="ba_greet"):
            st.markdown("Hola! Puc respondre amb les xifres del dashboard i simular horaris de càrrega.")
        for i, h in enumerate(hist):
            _user(h["q"], i)
            _bot(h["a"], h["sims"], str(i))
        if pending:
            _user(pending, len(hist))
            ph = st.empty()
            ph.markdown(TYPING, unsafe_allow_html=True)
            try:
                text, sims = answer(cfg, hist, pending)
            except Exception as e:
                text, sims = f"No he pogut contactar amb el model ({type(e).__name__}: {e}).", []
            hist.append({"q": pending, "a": text, "sims": sims})
            with ph.container():
                _bot(text, sims, str(len(hist) - 1))
    try:
        st.html(SCROLL_JS, unsafe_allow_javascript=True)
    except TypeError:
        pass