# UPC Energy Intelligence · v3.0

**Datathon FME 2026 — Side Challenge Tukey.** Una aplicació web orientada a l'anàlisi exploratòria i a les decisions energètiques, complementària (i independent) de `BatteryAgent`.

## Com començar des de zero

Requisits: Python 3.11–3.13, un navegador modern i el repositori [DatathonFME_UPCSostenibleChallenge](https://github.com/maxerrmax/DatathonFME_UPCSostenibleChallenge). Descomprimeix el paquet v2.0 a l'arrel del repositori: la carpeta `sidechallenge/` ha d'estar al costat de `dashboard/`, `data/` i `starter_kit/`.

**Windows / PowerShell**, executant des de l'arrel del repositori:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r sidechallenge/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn sidechallenge.service:app --host 127.0.0.1 --port 8502
```

**Linux / macOS**:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r sidechallenge/requirements.txt
.venv/bin/python -m uvicorn sidechallenge.service:app --host 127.0.0.1 --port 8502
```

Obre **http://127.0.0.1:8502** i recarrega amb Ctrl+Shift+R si venies d'una versió anterior. També es pot utilitzar el llançador `sidechallenge/start_windows.bat` si l'entorn ja té les dependències.

En tenir `data/train/` amb els CSV del repte, l'aplicació farà servir automàticament les dades oficials, a través de `dashboard/data.py`. Si falten, utilitza una mostra local (`sample_hourly.csv`) per poder demostrar la interfície. L'estat real és a **/api/status**.

## Connectar Helmcode

La resta de l'aplicació funciona sense API. Per activar la planificació de gràfics amb el model, al **mateix terminal** on executaràs el servidor:

```powershell
$env:HELMCODE_API_KEY="LA_TEVA_CLAU"
$env:HELMCODE_MODEL="qwen3.6"
.\.venv\Scripts\python.exe -m uvicorn sidechallenge.service:app --host 127.0.0.1 --port 8502
```

El model ha d'estar habilitat per la vostra clau. **No publiquis la clau a Git ni al frontend.** `mode: helmcode` a `/api/status` només significa que les variables estan configurades, **no** que el proveïdor hagi respost. La interfície distingeix entre planificador Helmcode verificat, context local i fallback local; no inventa visualitzacions si el proveïdor falla.

## Què fa la v3.0?

| Espai | Pregunta de decisió que respon | Com |
|---|---|---|
| **Vista general** | Què diuen les dades i quant me'n puc refiar? | KPI, perfils de **mediana + rang interquartílic** (un eix per unitat), panell **Data Quality** (lectures congelades 22–05 h, solar imputada, forats de test, absència d'excedents) |
| **Data Explorer** | Com es distribueix i es relaciona cada variable? | Heatmaps, perfils amb banda de quartils, *small multiples* quan les unitats difereixen, scatter acolorit per hora, agregació setmanal sense *aliasing* |
| **AI Copilot** | Quina decisió prenc i per què? | Analista **multi-step amb eines controlades** (perfil robust, millors hores, comparació amb IC bootstrap, correlació, simulació física, backtest, Pareto, qualitat). Helmcode tria les eines (*tool calling*; si no s'admet, pla JSON; si falla, analista local). **Verificador numèric**: cada xifra del text s'ha de trobar a les evidències `[E1]` |
| **Decision Lab** | Com hauria de carregar avui? | Finestres de càrrega/descàrrega, física i score de `environment.py`, òptim del dia, **valor capturat**, gràfic «per què decideix així l'òptim», Radar per mes amb distribució diària del marge |
| **Agent Arena** | Quina política presentem i on perdem punts? | Backtest al simulador **oficial** de BatteryAgent, regla horària, MPC 24 h (previsió ingènua i perfecta) i òptim setmanal sobre 11 setmanes pseudo-test; descomposició del regret; frontera CO₂–cost; dimensionat capacitat×potència; calendari de valor diari |
| **Decision Board** | Com ho expliquem al jurat? | Targetes amb especificació reproduïble, notes que s'imprimeixen, conclusions generades només amb les evidències fixades, import/export JSON, persistència a `localStorage` |

### Resultats clau (train, 11 setmanes pseudo-test, `environment.py`)

- MPC de 24 h amb les previsions de preu i CO₂: **6,45 punts**, igual que l'òptim amb informació perfecta (regret d'horitzó i de previsió: 0,00).
- Plantilla `starter_kit/agent.py`: **1,17 punts** i 4 infraccions (0,00 amb la penalització de l'ENUNCIAT).
- La previsió de consum no aporta valor: la demanda neta mínima (460 kW) gairebé mai limita una descàrrega de 500 kW.
- Doblar la capacitat (4.000 kWh) val més que doblar la potència (9,92 vs 7,33 punts òptims).

### Demo recomanada (3 minuts)

1. **Vista general → Data Quality** (20 s): «abans de concloure, sabem que el 33% de les hores de consum són lectures congelades».
2. **Copilot → «Quina estratègia de bateria funciona millor?»** (40 s): mostra la traça `backtest()`, les cites `[E1]` i el segell «10/10 xifres verificades».
3. **Agent Arena** (60 s): 1,17 → 6,45 punts; waterfall del regret; conclusió «no cal predir consum». Fixa el waterfall al board.
4. **Decision Lab** (40 s): aplica el parell del Radar (14 → 21 h) i observa que captura poc; amplia les finestres i mira «per què decideix així l'òptim».
5. **Arena → Pareto i dimensionat** (20 s): recomanació per a UPC Sostenible.
6. **Board → Generar conclusions → Imprimir/PDF**.

### Agent avaluat a l'Arena

Per defecte s'importa (només lectura) `src/agent.py` si existeix; si no, `starter_kit/agent.py`. Per apuntar a un altre fitxer:

```bash
UPC_AGENT_PATH=ruta/al/agent.py python -m uvicorn sidechallenge.service:app --port 8502
```

Després d'editar l'agent, prem **Reexecutar** a l'Arena. Els resultats es desen a `sidechallenge/.cache/` (claus per hash de dades i data de modificació de l'agent).

### Notes metodològiques essencials

- Tots els números de bateria surten de `arena.official_eval`, una rèplica pas a pas de `environment.py` validada per test, o directament de `CampusBatteryEnv`. Els programes lineals deixen un marge d'1e-3 kWh als límits del SOC: la v2 provocava infraccions de *clipping* en 191 de 245 dies.
- `environment.py` resta 0,2 punts per infracció i l'ENUNCIAT diu 1,00: l'Arena mostra totes dues puntuacions.
- Setmanes pseudo-test: per a cada mes, els 7 últims dies complets de train (just abans de la finestra oculta). Les polítiques causals es normalitzen i calibren amb els dies **fora** d'aquestes setmanes.
- L'òptim, el Pareto, el dimensionat i el calendari tenen informació perfecta: són **cotes superiors**, no previsions.
- El Lab diari treballa en cicle tancat (la bateria no baixa del SOC inicial) per comparar dies de manera justa.
- Comparacions de cohort: mitjanes horàries amb IC 95% bootstrap remostrejant **dies**, no hores.
- El Copilot no calcula: Helmcode només tria eines i redacta. Els arguments es validen contra un esquema tancat i les xifres es verifiquen.

## Guardrails del Copilot

| Capa | Què evita | Com |
|---|---|---|
| **Entrada** | Injecció de prompt, extracció de claus o del prompt de sistema | `guardrails.input_check` respon sense cridar el model; els noms de columna pujats (text no fiable) es filtren abans d'arribar a cap prompt |
| **Eines** | Que el model executi codi o toqui dades arbitràries | Esquemes tancats, llista blanca de variables, rangs limitats (p. ex. potència ≤ 500 kW), màxim 6 crides, crides repetides reaprofitades |
| **Evidència** | Respostes de memòria o reutilitzant l'historial | Per a preguntes reconegudes el servidor executa abans les eines i el model redacta a partir d'aquests resultats; per a preguntes obertes, `tool_choice: required` |
| **Sortida** | Xifres inventades, afirmacions sense suport, HTML | Cada frase amb una xifra que no és a les evidències d'aquest torn s'elimina (es mostra a la interfície); «N punts» ha de coincidir amb un score calculat; no es confia en xifres de la pregunta; es marquen les línies sense cita; text pla i longitud limitada |
| **Volum** | Consum de crèdit si la demo s'exposa | Límit de 20 preguntes per minut i sessió |
| **Privacitat** | Fuites de dades | El model només rep agregats de les eines i noms de columna, mai files de dades; la clau no surt del servidor |

Limitació coneguda: la verificació és numèrica. Una afirmació qualitativa sense xifres i sense cita no s'elimina, només es pot detectar llegint-la; el prompt prohibeix afegir matisos que no surtin de les evidències.

## Arquitectura

```text
sidechallenge/
  service.py       API FastAPI: dades, gràfics, planificador de gràfics, Lab, Radar, endpoints Arena
  arena.py         Física oficial, LP amb marge, MPC, backtest, Pareto, dimensionat, calendari (en segon pla, amb memòria cau)
  eda.py           Estadística robusta, bootstrap per dies, qualitat de dades, senyal 70/30 per mes
  guardrails.py    Entrada (injeccions, columnes no fiables), sortida (frases no verificables), límit de peticions
  copilot.py       Analista amb eines: classificació d'intenció, tool calling Helmcode, pla JSON, analista local, verificador
  static/          index.html · style.css · app.js · plotly.min.js
  tests/           test_service.py (v1) · test_v2.py · test_v3.py
```

## Tests

Des de l'arrel del repositori:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest sidechallenge/tests -q
```

El navegador ha estat provat amb el flux del Copilot, visualitzacions, fixació al Decision Board, seguiment contextual i el Decision Lab. La crida **real** a Helmcode no està verificada en aquesta construcció (requereix la vostra clau). En cas d'error, el planificador local preserva les visualitzacions essencials i mostra un avís de fallback.
