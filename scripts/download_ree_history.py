"""
Descarrega historial PÚBLIC de Red Eléctrica (REE) per pre-entrenar les xarxes de CO2/preu (Mode Difícil).

  - Mix de generació peninsular cada 5 min (demanda.ree.es, seguiment en temps real) -> horari (mitjana).
    El CO2 del repte és 0,37 * gen_amb_emissions / gen_total; aquí es reconstrueix i després es calibra
    linealment contra co2_red.csv al període que comparteixen.
  - PVPC horari (apidatos.ree.es, mercados/precios-mercados-tiempo-real), coincideix exactament amb precio_pvpc.csv.

ÚNICAMENT dades ANTERIORS al primer període de test (setembre de 2025) per al pre-entrenament; el període solapat
(set. 2025 - ago. 2026) només serveix per calibrar el CO2 i NO s'ha de fer servir per entrenar.

    python scripts/download_ree_history.py 2022-01-01 2025-08-31      # pre-entrenament
    python scripts/download_ree_history.py 2025-09-01 2025-11-30 --calib   # calibratge (solapament)
Sortida: data/external/ree_gen_5min/<dia>.json (cache) i data/external/ree_pvpc/<mes>.json.
"""
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
UA = {"User-Agent": "datathon-research-client"}


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            print("  retry", k + 1, e, flush=True)
            time.sleep(3 * (k + 1))
    return None


def gen_day(d):
    f = EXT / "ree_gen_5min" / f"{d}.json"
    if f.exists():
        return
    t = get("https://demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/demandaGeneracionPeninsula"
            f"?callback=cb&curva=DEMANDAAU&fecha={d}")
    if t and "(" in t:
        f.write_text(t[t.index("(") + 1:t.rindex(")")], encoding="utf-8")
    time.sleep(0.3)


def pvpc_month(y, m):
    f = EXT / "ree_pvpc" / f"{y}-{m:02d}.json"
    if f.exists():
        return
    a = date(y, m, 1); b = (a + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    t = get("https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
            f"?start_date={a}T00:00&end_date={b}T23:59&time_trunc=hour")
    if t:
        f.write_text(t, encoding="utf-8")
    time.sleep(0.5)


if __name__ == "__main__":
    a, b = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
    (EXT / "ree_gen_5min").mkdir(parents=True, exist_ok=True); (EXT / "ree_pvpc").mkdir(parents=True, exist_ok=True)
    if "--calib" not in sys.argv:
        y, m = a.year, a.month
        while (y, m) <= (b.year, b.month):
            pvpc_month(y, m); m += 1
            if m == 13:
                y, m = y + 1, 1
        print("pvpc fet", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    days = [a + timedelta(days=i) for i in range((b - a).days + 1)]
    with ThreadPoolExecutor(4) as ex:                  # 4 peticions simultànies com a màxim
        for i, _ in enumerate(ex.map(gen_day, days)):
            if (i + 1) % 100 == 0:
                print(days[i], flush=True)
    print("fet", len(days), "dies")
