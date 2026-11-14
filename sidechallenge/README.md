# UPC Energy Intelligence · v2.0

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

## Què fa la v2.0?

| Espai | Funcions |
|---|---|
| **Vista general** | KPI de demanda, energia solar, emissions i cost sense bateria; gràfics i insights sobre train |
| **Data Explorer** | Importar CSV/ZIP, construir heatmaps, perfils, histogrames, boxplots, scatter; exportar PNG |
| **AI Copilot** | Preguntar en llenguatge natural, generar gràfics verificables, seguir una conversa amb context analític, filtrar per laborables/caps de setmana, canviar agregacions i comparar perfils |
| **Decision Lab** | Simular un cicle de càrrega i descàrrega amb física del campus, comparar-lo amb un òptim retrospectiu i explorar el *Historical Opportunity Radar* |
| **Decision Board** | Fixar figures de qualsevol anàlisi, canviar títols, reordenar amb drag-and-drop o fletxes, anotar decisions, exportar PNG o un JSON del board i imprimir a PDF |

### Demo recomanada (3 minuts)

1. **Overview:** verifica l'origen de dades i explica la prioritat del 70% sobre CO₂.
2. **AI Copilot:** «Fes-me un heatmap dels preus i les emissions per hora i dia de la setmana» → comprova els dos gràfics.
3. **Memòria:** «Ara només els caps de setmana» → els mateixos gràfics, però filtrats. Després «Compara-ho amb els laborables» → perfils amb dues línies, una per cohort.
4. **Fixar al board:** fes clic a `+ Fixar al board` en dos gràfics.
5. **Decision Lab:** mira l'Opportunity Radar, aplica el parell candidat al simulador i fixa l'escenari al board.
6. **Decision Board:** reordena les evidències, explica què justifica cada gràfic, exporta el JSON i imprimeix el dossier a PDF.

### Notes metodològiques essencials

- Tot el càlcul de visualitzacions i de KPIs es fa al servidor amb `pandas`, `Plotly` i, per a decisions, `SciPy`. Helmcode interpreta preguntes i proposa plans, però **no genera els valors estadístics**.
- Les consultes amb context reutilitzen l'últim pla verificat, que es transmet al servidor com a configuració de gràfics, **no** com a codi executable. El backend revalida columnes i operacions.
- L'Opportunity Radar és **descriptiu**: índex adimensional format pel 70% de CO₂ normalitzat i el 30% del preu normalitzat, ajustat a l'eficiència del 90% d'un cicle. És útil per trobar hipòtesis, **no és l'score oficial** ni garanteix una consigna òptima.
- El Decision Lab opera **retrospectivament** i per **un dia**. Compara una estratègia manual amb un òptim de coneixement perfecte per aquell dia. Les restriccions de SOC inicial/final poden diferir de l'avaluació oficial de 2.016 hores; per això no s'ha de presentar la puntuació com a leaderboard.
- Les dades d'entrenament poden tenir mancances i lectures congelades del comptador. L'EDA ha de documentar-les abans de formular conclusions causals.
- Els datasets pujats s'aïllen per cookie de sessió i es mantenen només a la memòria del servidor. El Decision Board es guarda al `sessionStorage` del navegador (durant la sessió d'aquella pestanya) i el JSON exportat és local. **No es guarden claus API al navegador.**
- El dashboard no altera `starter_kit/agent.py`, `dashboard/app.py` ni cap component de l'agent del challenge principal. S'executa amb el seu propi servei a port 8502.

## Arquitectura

```text
sidechallenge/
  service.py                API FastAPI, estadístiques, planificació, anàlisi contextual,
                            radar i simulador físic retrospectiu
  static/
    index.html              Interfície editorial en català
    style.css               Identitat visual, estats, responsive i impressió/PDF
    app.js                  Interaccions, memòria analítica i board fixat
    plotly.min.js           Gràfics interactius sense CDN
  tests/
    test_service.py         Tests de la v1.x
    test_v2.py              Tests de context, comparació i radar
  sample_hourly.csv         Mostra de train per demostracions independents
  requirements.txt
```

## Tests

Des de l'arrel del repositori:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest sidechallenge/tests -q
```

El navegador ha estat provat amb el flux del Copilot, visualitzacions, fixació al Decision Board, seguiment contextual i el Decision Lab. La crida **real** a Helmcode no està verificada en aquesta construcció (requereix la vostra clau). En cas d'error, el planificador local preserva les visualitzacions essencials i mostra un avís de fallback.
