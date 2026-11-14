# 🔋 Explorador de datos · Campus Nord UPC

Visualizador interactivo estilo Power BI para entender los datos del reto
UPC Sostenible (Datathon 2026) antes de diseñar el `BatteryAgent`.

## Cómo arrancarlo

Desde la raíz del repo:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Se abre en <http://localhost:8501>. La primera carga agrega los CSV de
`data/train/` (~2 s) y guarda una caché en `data/processed/hourly.parquet`.
La caché se regenera sola si cambia algún CSV.

## Qué hay en cada pestaña

| Pestaña | Para qué sirve |
|---|---|
| **📊 Resumen** | KPIs del periodo (consumo, pico, solar, coste y CO₂ sin batería), consumo diario, conclusiones automáticas y matriz mensual. |
| **🏫 Demanda** | Mapas de calor hora × día y mes × hora, perfil laborable/fin de semana, consumo frente a temperatura, curva monótona y serie temporal. |
| **💶 Precio y CO₂** | Perfiles horarios, **señal combinada del score** (cuándo cargar y cuándo descargar), mapas de calor, correlación, hora más limpia/sucia de cada día, margen diario de un ciclo y mix de generación. |
| **☀️ Solar y meteo** | FV frente a irradiancia, producción por inversor, temperatura de panel. |
| **🔋 Simulador de batería** | Ejecuta cualquier `agent.py` del repo con el simulador oficial y lo compara con el **óptimo teórico** (LP con previsión perfecta). Incluye explorador semanal, estrategia media por hora y puntos por mes. |
| **🧪 Calidad de datos** | Mapa de cobertura día × hora, completitud por sensor y lista de problemas detectados. |

Los filtros de la barra lateral (periodo, meses, tipo de día) afectan a todas
las pestañas. El simulador solo usa el rango de fechas, porque el agente
necesita horas consecutivas.

## Hallazgos principales (train)

- **Contador congelado cada noche (~22:05–06:00):** el 33 % de las horas de
  consumo repiten la última lectura. El consumo nocturno del simulador es plano.
- **Noviembre de 2025 no tiene datos** de contador, FV ni meteo.
- **La FV es pequeña:** máximo de ~144 kW, ~2 % del consumo. Nunca hay
  excedente, así que todo lo que se carga sale de la red.
- **El consumo nunca baja de ~590 kW:** descargar a 500 kW nunca exporta.
- **CO₂ más bajo de 12 a 14 h** (valle solar) y más alto de 6 a 7 h y de 20 a
  21 h. **Precio más bajo de 14 a 16 h** y más alto de 19 a 21 h.
  Correlación precio–CO₂: 0,58.
- **Agente plantilla:** score 0, con 46 infracciones por redondeo de
  `0.9487` frente a `0.9 ** 0.5`. **Óptimo teórico: ~5,8 puntos**, con unos
  dos ciclos al día.

## Archivos

- `data.py`: carga, limpieza y agregación horaria (`load_hourly()`, `sim_frame()`).
- `sim.py`: ejecución de agentes con `starter_kit/environment.py`, LP óptimo
  (`optimal_lp()`) y análisis de ciclos diarios.
- `app.py`: interfaz Streamlit + Plotly.

`data.py` y `sim.py` también se pueden usar desde un notebook:

```python
import sys; sys.path.insert(0, "dashboard")
from data import load_hourly, sim_frame
import sim
df = load_hourly()
res, traza = sim.optimal_lp(sim_frame(df))
```
