# Dashboard de decisió · Bateria del Campus Nord

Tot el dashboard respon una pregunta: **quin agent de bateria lliurem, i per què?**
Aquest fitxer és també el traspàs per a qui continuï la feina (persona o agent): llegiu-lo sencer abans de tocar res.

Criteris del jurat (són el que cal optimitzar): **utilitat per decidir**, **representacions adequades** i
**coherència** (indicadors, gràfics i conclusions alineats entre ells i amb les dades).

## Com s'engega

Des de l'arrel del repositori:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Per a la previsualització de Claude: `.claude/launch.json`, configuració `dashboard` (port 8511).

## Pàgines (pestanyes)

Cada pàgina ensenya la decisió i dos o tres gràfics; la resta és en desplegables tancats (`ui.more`).

| Pestanya | Fitxer | A la vista | Als desplegables |
|---|---|---|---|
| Veredicte | `p_verdict.py` | Versió triada automàticament, 4 xifres, estimació al test privat per versió, validació contra públic, regla per a les versions no enviades | Historial de lliuraments, aspirants de l'arena amb les condicions, taula de validació |
| Cara a cara | `p_models.py` | Marcador A/B/òptim, punts perduts per hora, una setmana hora a hora | Totes les setmanes, perfil mitjà del dia |
| On són els punts | `p_gap.py` | Escala de millores, valor de la informació perfecta, «Què fem ara» | Idees descartades |
| Campus i xarxa | `p_campus.py` | 4 xifres, valor d'un kWh per hora, dimensionament de la bateria | Taula de dimensionament, mapes per hora, qualitat de les dades |
| Prova-ho tu | `p_play.py` | L'usuari programa un horari fix i el compara amb l'agent i l'òptim | Setmana a setmana |
| Pregunta-li | `p_ask.py` | Assistent (Helmcode) que respon amb les xifres del dashboard i pot simular horaris | — |

`p_sim.py` (simulador del Mode Estàndard) existeix però no és a les pestanyes. `app.py` munta capçalera, pestanyes i peu.

## La tria automàtica (`models.choose`)

**No hi ha cap camp «versió seleccionada» escrit a mà.** La classificació final es fa sobre la part privada del
test, que el rànquing públic no mostra, i per tant la nota pública més alta no és necessàriament el millor model.
`models.choose(reg)` fa això amb `submissions.json`:

1. **Àncora**: la versió amb més nota pública entre les que tenen validació (ara, v12). Tot es mesura respecte a ella.
2. **Evidència pública**: diferència al rànquing, amb soroll `selection.public_noise_sd` (0,35 per comparació, mesurat per l'equip).
3. **Evidència de validació**: la diferència de validació es tradueix a punts de test amb `beta`, el pendent de la
   regressió per l'origen de (diferència pública) sobre (diferència de validació) de tots els lliuraments amb les dues
   xifres. El soroll és la desviació dels residus. Ara: beta ≈ 0,57, soroll ≈ 0,41, amb 6 lliuraments.
4. **Estimació** de l'avantatge al test privat = mitjana de 2 i 3 ponderada per l'invers de la variància. Les versions
   sense validació només tenen 2; les no enviades només tenen 3 i **no són elegibles**.
5. **Tria** = l'elegible amb més estimació. En cas d'empat guanya la que té `"extends": "<versió empatada>"`
   (és aquella versió més un canvi que només actua a la part privada). Així surt la v15 sobre la v12.
6. **Versions no enviades**: es calcula la nota pública que necessitarien per passar al davant, mai per sota de la de la
   versió triada (el públic ha de confirmar). Per a la v19 dona «més de 23,53», la mateixa regla de l'equip.

Fan servir `choose`: `p_verdict.py`, `p_gap.py`, `p_ask.py` (paquet de dades i preguntes suggerides) i `models.gates`
(condicions perquè una estratègia de l'arena substitueixi la triada).

**Què no està verificat (cal dir-ho si es presenta):** és una heurística calibrada amb 6 punts; suposa que el test
privat es comporta com el públic; el 0,35 és una estimació de l'equip; no es fa validació creuada del pendent. Serveix
per ordenar versions i fixar regles de decisió, no per predir la nota final. Les condicions de `gates` encara poden dir
«val la pena gastar-hi un lliurament» per a estratègies que guanyen a la validació, tot i que tres d'aquestes ja han
perdut al públic: és una invitació a provar, no una recomanació de lliurament.

## Dades

| Què | On | Com s'actualitza |
|---|---|---|
| Lliuraments, rànquing, escala, informació perfecta, idees descartades | `submissions.json` | **A mà.** Després de cada lliurament: afegir la fila o omplir `public`. La tria i els llindars es recalculen sols |
| Resultats de l'arena (3 plecs) | `model_data/arena.json` | `python dashboard/build_models.py --no-traces` (llegeix `dificil/arena/results/`) |
| Traces hora a hora, temps per pas, dimensionament | `model_data/traces.parquet`, `runs.json`, `sizing.json` | `python dashboard/build_models.py` (uns minuts; un sol procés) |
| Només les files de l'òptim | — | `python dashboard/build_models.py --optimum` |
| Dades del campus | `data/train` via `data.py` | — |

Camps de `submissions.json` que cal conèixer: `validation` (plec 1), `public` (`null` = no enviada), `arena`
(estratègia de `dificil/arena` que reprodueix la versió), `extends` (vegeu la tria), `co2`/`cost` (percentatges
públics). **Supòsits escrits a mà:** el CO₂ i el cost públics de la v15 es copien de la v12 (mateixa nota pública);
`public_test.hours = 840` (unes 5 setmanes) i les bases de CO₂ i cost es dedueixen dels percentatges del rànquing.

Restriccions del projecte (CLAUDE.md): només 4 nuclis, **mai dos processos pesats alhora**; les dades del període de
test i els pseudo-tests no entren mai al repositori.

## Assistent (Pregunta-li)

API de Helmcode, compatible amb OpenAI. La clau va a `.streamlit/secrets.toml` (no es versiona; és al `.gitignore`):

```toml
[helmcode]
api_key = "..."
base_url = "https://api.helmcode.com/v1"
model = "qwen3.6"
```

`qwen3.6` admet crida d'eines i respon en 2–4 s (`glm5.3` demana un complement de pagament). La petició desactiva el
raonament (`chat_template_kwargs.enable_thinking = false`) i posa `max_tokens = 4000`; sense això les respostes sortien
buides. L'assistent només rep `fact_pack()` (les mateixes xifres de les pàgines, incloent-hi `com_es_tria_la_versio`) i
l'eina `simula_horari`. Per evitar que inventi: les versions no enviades porten el text «ENCARA NO ENVIADA», les regles
venen calculades, i es treu la línia «On mirar-ho» si anomena una pestanya que no existeix.

## Disseny (`ui.py`)

Història: la primera versió es va rebutjar per «semblar feta per una IA» (tipografia i colors per defecte); la segona,
d'estil diari en blanc i negre, per «massa minimalista» i per tenir «massa informació a cada pàgina». L'actual:

- Paleta d'instal·lació elèctrica: marí `#0a2540`, blau UPC `#007bc0`, groc `#ffc233`, fons `#eef2f6`.
  El color vol dir sempre el mateix: **blau = la versió que lliurem, taronja = aspirant o rival, marí = òptim, gris = context**;
  verd = CO₂, ocre = factura.
- Bricolage Grotesque (títols i xifres) + IBM Plex Sans (text). Format català amb `ui.fmt` (coma decimal).
- Peces: `ui.hero` (bloc blau de la decisió, amb `ui.battery`), `ui.tiles`, `ui.panel` (targeta blanca per a un
  gràfic), `ui.note` (groc), `ui.more` (desplegable), `ui.table`, `ui.checklist`, `ui.intro`, `ui.section`.
- El títol de cada gràfic és la conclusió i es calcula amb les dades (no escriure xifres ni versions fixes als textos);
  el subtítol diu com es llegeix.
- Abans d'afegir res a una pàgina: si no canvia una decisió, va a un desplegable o no hi va.

Detalls de Streamlit (1.67) que costen de trobar: les targetes es pinten amb la classe `st-key-card_…` que posa
`ui.panel` (clau derivada del títol; passeu `key=` si dos panells poden compartir títol); les pestanyes són
`[data-testid="stTab"]`; les píndoles d'hores s'emboliquen amb `[data-testid="stButtonGroup"] [role="toolbar"]`.

## Com comprovar canvis

- Streamlit no sempre recarrega els mòduls importats: **atureu i torneu a engegar el servidor** després d'editar.
- Captures senceres: finestra de 1440×900 i desplaçament amb
  `document.querySelector('[data-testid="stMain"]').scrollTo(0, y)`; comprovar que no hi hagi cap `stException`.
- Comprovat el 15/11/2026: les sis pestanyes carreguen sense errors, els desplegables de Veredicte s'obren, i
  l'assistent respon una pregunta suggerida. No s'ha comprovat en pantalla estreta ni amb el tema fosc.
