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

El dashboard és una **eina per comparar i decidir**: els títols descriuen què es mostra i no donen conclusions. Cada gràfic té
una icona (?) amb l'explicació completa (passa-hi el ratolí) i una etiqueta d'origen de la dada:
**VALIDACIÓ** (setmanes apartades de `data/train`, models reentrenats sense elles), **PÚBLIC** (rànquing del repte),
**CAMPUS** (dades mesurades) i **CALCULAT** (càlcul del dashboard a partir de les anteriors).

| Pestanya | Fitxer | Contingut |
|---|---|---|
| Resum | `p_front.py` | Selector de versió (per defecte, la de més nota pública), nota pública i evolució, CO₂ i factura contra el primer, rànquing públic complet |
| Decidir | `p_decide.py` | Diferència amb el líder i el top per component, CO₂ contra factura, boxplot del rànquing, valor de la informació perfecta, idees provades, sol i CO₂ |
| Com triem | `p_verdict.py` | Estimació de l'avantatge al test privat per versió, relació validació–públic, historial, estratègies de l'arena |
| Cara a cara | `p_models.py` | Dos agents qualssevol (totes les estratègies de l'arena) hora a hora contra l'òptim, boxplot setmanal |
| On són els punts | `p_gap.py` | Evolució de la validació, distància a l'òptim i al primer, idees provades |
| Campus i xarxa | `p_campus.py` | Consum, solar, valor d'un kWh per hora, dimensionament de la bateria, qualitat de les dades |
| Pregunta-li | `p_ask.py` | Assistent que respon amb les xifres del dashboard (botó flotant) |

`p_play.py` ja no és una pestanya: només conserva les funcions d'horari fix que fa servir l'assistent (`simula_horari`).
La resta de la lògica és a `ui.py` (disseny, etiquetes, icones d'ajuda, animacions), `models.py` (dades i tria) i
`live.py` (rànquing en directe).

## Com s'actualitza tot (repo net)

```bash
python scripts/arena.py                # si hi ha estratègies noves a dificil/arena/ (resultats a results/store*.json)
python dashboard/refresh.py            # arena.json + sizing, traces de totes les estratègies, voi.json (un pas darrere l'altre)
python dashboard/refresh.py --only voi # o un sol pas: arena | traces | voi
```

**Rànquing públic automàtic:** `python dashboard/fetch_leaderboard.py` llegeix el rànquing del Mode Difícil de
https://repteupc.datathon.cat/ i actualitza `live_feed.json` (equips, lliuraments, nota, CO₂ i cost; conserva les files més enllà
de la taula de la web, que només n'ensenya els primers 12). `--loop 300` el repeteix cada 5 min i `--dry` només mostra el resultat.
Necessita un navegador sense pantalla: `pip install playwright && playwright install chromium` (dependència opcional).

Després només cal omplir, a mà o amb un script, **`submissions.json`** (un lliurament nou o la seva nota pública) i
**`live_feed.json`** (rànquing; vegeu `live.write_feed`). Tota la resta —tria, estimacions, llindars, taules, gràfics i
text— es recalcula sola; no hi ha xifres escrites als títols.

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
| Lliuraments (validació, nota pública, CO₂ i cost públics) | `submissions.json` | **A mà.** Afegir la fila o omplir `public` després de cada lliurament |
| Rànquing públic | `live_feed.json` (o URL, vegeu `live.py`) | Script o a mà; la pàgina el refresca cada 30 s |
| Resultats de l'arena (3 plecs) | `model_data/arena.json` | `python dashboard/refresh.py --only arena` |
| Traces hora a hora de totes les estratègies | `model_data/traces.parquet`, `runs.json` | `python dashboard/refresh.py --only traces` (reprèn on es va deixar; `_cache/`) |
| Valor de la informació perfecta | `model_data/voi.json` | `python dashboard/refresh.py --only voi` |
| Dimensionament de la bateria | `model_data/sizing.json` | `python dashboard/refresh.py --only arena` |
| Dades del campus | `data/train` via `data.py` | — |

Camps de `submissions.json` que cal conèixer: `validation` (plec 1), `public` (`null` = no enviada), `arena`
(estratègia de `dificil/arena` que reprodueix la versió), `extends` (vegeu la tria), `co2`/`cost` (percentatges
públics). **Supòsits escrits a mà:** el CO₂ i el cost públics de la v15 es copien de la v12 (mateixa nota pública);
`public_test.hours = 1008` (la meitat de cada setmana de test, uns 5 setmanes equivalents) i les bases de CO₂ i cost es dedueixen dels percentatges del rànquing.

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

Història: la primera versió es va rebutjar per «semblar feta per una IA»; la segona (diari en blanc i negre) per «massa
minimalista»; la tercera (blau/groc, Bricolage + Plex) es va substituir el 15/11/2026 per l'actual, a petició de l'equip:

- Estil Linear / Stripe / editorial suís: llenç `#FAFAFA`, contenidors `#FFFFFF`, vora d'1 px `#E4E4E7`, **cap ombra**.
  **Mode fosc** amb el commutador de la capçalera (`st.toggle(key="dark")`); els colors són variables CSS (`ui.LIGHT`/`ui.DARK`)
  i els gràfics de Plotly s'inverteixen amb un filtre CSS (`invert + hue-rotate`), així que no cal duplicar colors als gràfics.
- Helvetica per al text; xifres, percentatges i taules en monoespaiat (`.num`, `tabular-nums`). Etiquetes (`.lab`) de 11 px
  en majúscules, 0,05 em, grises. Sense reixetes ni degradats; un sol color de dades (verd blavós `#0F766E`, `#2DD4BF` en fosc).
  Vermell/verd **només** a les insígnies de tendència (`ui.badge`).
- Peces: `ui.tiles` (franja d'indicadors), `ui.panel` (targeta amb títol, etiqueta d'origen i icona d'ajuda), `ui.heading`
  (el mateix per a gràfics dins de desplegables), `ui.tag` (VALIDACIÓ / PÚBLIC / CAMPUS / CALCULAT), `ui.helpicon`, `ui.note`,
  `ui.more`, `ui.table`, `ui.checklist`, `ui.intro`, `ui.section`. Format català amb `ui.fmt`.
- **Els títols descriuen, no concluen.** La descripció llarga va a la icona (?) (`ui.helpicon`, només CSS, sense JS) i a sota
  del títol només en queda la primera frase, en petit. Qualsevol xifra d'un títol o d'una descripció surt de les dades.
- Animacions discretes: entrada suau de targetes i taules, creixement de les barres, i aparició en fer scroll
  (`animation-timeline: view()` on el navegador ho admet); desactivades amb `prefers-reduced-motion`.
- Abans d'afegir res a una pàgina: si no canvia una decisió, va a un desplegable o no hi va.

## Dades en directe (`live.py`) — la porta

Tot el que canvia durant el datathon entra per `live.load()`; les pàgines no llegeixen mai `submissions.json` directament
per a això. Fonts, per ordre: **HTTP** (`DASHBOARD_FEED_URL` o `[live] url` a `.streamlit/secrets.toml`, opcional `token`)
→ **fitxer** `dashboard/live_feed.json` → còpia estàtica de `submissions.json` (marcada «antiga»). Per connectar una font
nova: una funció a `live.SOURCES`. L'esquema del JSON és a la capçalera de `live.py`; el connector només ha de cridar
`live.write_feed(...)` (escriptura atòmica) o servir el mateix JSON per HTTP.

- La portada es refresca sola cada 30 s (`st.fragment(run_every=live.REFRESH_EVERY)`) i mostra «En directe / Dades antigues
  (>15 min) / Sense connexió».
- El camp `submissions` del feed fusiona `public`/`co2`/`cost` dels nostres lliuraments amb el registre: en omplir-hi la
  nota pública d'una versió, `models.choose` recalcula la tria i les regles sola (comprovat amb una v19 de 23,70).
- `live_feed.json` conté ara una **instantània** de https://repteupc.datathon.cat/ (Mode Difícil, públic, 1.008 h = 6
  setmanes, 15/11/2026). La web és una app Streamlit sense API (`/_stcore/agent` desactivat): llegir-la automàticament
  demana un navegador o un client de websocket; no s'ha fet. Les files 13–22 es van llegir d'una captura.

## Text i colors

No hi ha commutador d'explicacions ni peu de pàgina: l'explicació és a les icones (?). Colors: paleta Okabe-Ito (blau `#0072B2` = nosaltres, ambre `#E69F00` = líder/rival)
i les insígnies de tendència duen ▲/▼ perquè no depenguin només del color.

Detalls de Streamlit (1.67) que costen de trobar: les targetes es pinten amb la classe `st-key-card_…` que posa
`ui.panel` (clau derivada del títol; passeu `key=` si dos panells poden compartir títol); les pestanyes són
`[data-testid="stTab"]`; les píndoles d'hores s'emboliquen amb `[data-testid="stButtonGroup"] [role="toolbar"]`.

## Com comprovar canvis

- Streamlit no sempre recarrega els mòduls importats: **atureu i torneu a engegar el servidor** després d'editar.
- Captures senceres: finestra de 1440×900 i desplaçament amb
  `document.querySelector('[data-testid="stMain"]').scrollTo(0, y)`; comprovar que no hi hagi cap `stException`.
- Comprovat el 15/11/2026: les sis pestanyes carreguen sense errors, els desplegables s'obren, i
  l'assistent respon una pregunta suggerida. No s'ha comprovat en pantalla estreta ni amb el tema fosc.
