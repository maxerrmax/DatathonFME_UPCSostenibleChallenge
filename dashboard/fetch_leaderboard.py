"""
dashboard/fetch_leaderboard.py  ·  Actualitza `live_feed.json` llegint el rànquing del Mode Difícil de la web del repte.

    python dashboard/fetch_leaderboard.py            # una lectura
    python dashboard/fetch_leaderboard.py --loop 300 # repeteix cada 300 s (Ctrl+C per aturar)
    python dashboard/fetch_leaderboard.py --dry      # mostra el que s'escriuria, sense desar

La web (https://repteupc.datathon.cat/) és una app Streamlit: el rànquing es dibuixa amb JavaScript, així que cal un
navegador sense pantalla. Dependència opcional (no és a requirements.txt):

    pip install playwright && playwright install chromium

La taula de la web només mostra els primers equips (ara 12). Les files següents que ja eren al feed es conserven
(amb la nota que tenien); les que ara surten a la taula les sobreescriuen. El dashboard recull el fitxer en uns segons
(vegeu `live.py`); no cal reiniciar res.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

URL = "https://repteupc.datathon.cat/"
TAB = "Mode Difícil"
FEED = HERE / "live_feed.json"
US = "JAMM"


def _num(s: str) -> float | None:
    try:
        return float(s.replace(".", "").replace(",", ".")) if re.search(r"\d,\d{1,2}$", s) and "." in s else float(s.replace(",", "."))
    except ValueError:
        return None


def parse_rows(rows: list[list[str]]) -> list[dict]:
    """Files de la taula (capçalera inclosa) → [{rank, team, score, co2, cost}].
    Columnes: Posició | Data | Equip | Score Oficial | Reducció CO2 | Estalvi Cost | CO2 Evitat | Estalvi Factura."""
    out = []
    for r in rows:
        if len(r) < 6 or not re.match(r"^\W*\d+", r[0].strip()):
            continue
        rank = int(re.search(r"\d+", r[0]).group())
        score, co2, cost = _num(r[3]), _num(r[4]), _num(r[5])
        if score is None:
            continue
        out.append({"team": r[2].strip(), "score": score, "co2": co2, "cost": cost, "rank": rank})
    return out


def parse_totals(text: str) -> tuple[int | None, int | None]:
    def grab(label):
        m = re.search(label + r"\s*\n?\s*([\d.,]+)", text)
        return int(re.sub(r"\D", "", m.group(1))) if m else None
    return grab("Equips Participants"), grab("Total de Lliuraments")


def fetch() -> tuple[list[dict], int | None, int | None]:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(URL, wait_until="networkidle", timeout=60000)
        pg.get_by_text(TAB, exact=False).first.click()
        pg.wait_for_selector("text=Equips Participants", timeout=30000)
        pg.wait_for_selector("table", timeout=30000)
        tables = pg.query_selector_all("table")
        rows = []
        for t in tables:
            head = (t.inner_text() or "")[:200]
            if "Score Oficial" in head:
                rows = [[c.inner_text() for c in tr.query_selector_all("th,td")] for tr in t.query_selector_all("tr")]
        n_teams, n_subs = parse_totals(pg.inner_text("main"))
        b.close()
    return parse_rows(rows), n_teams, n_subs


def merge(old: dict, new: list[dict], n_teams, n_subs) -> dict:
    """Les files noves manen; les antigues amb un lloc més enllà de la taula de la web es conserven."""
    seen = {r["team"] for r in new}
    last = max((r["rank"] for r in new), default=0)
    tail = [r for r in old.get("standings", []) if r["team"] not in seen and r["rank"] > last]
    standings = sorted(new + tail, key=lambda r: -r["score"])
    for i, r in enumerate(standings, 1):
        r["rank"] = i
    return {**old, "as_of": datetime.now().isoformat(timespec="seconds"), "us": old.get("us", US),
            "n_teams": n_teams or old.get("n_teams"), "n_submissions": n_subs or old.get("n_submissions"),
            "standings": standings}


def run(dry: bool = False) -> dict:
    old = json.loads(FEED.read_text(encoding="utf-8")) if FEED.exists() else {}
    new, n_teams, n_subs = fetch()
    if not new:
        raise RuntimeError("no s'ha trobat la taula del rànquing (la web ha canviat?)")
    feed = merge(old, new, n_teams, n_subs)
    if not dry:
        tmp = FEED.with_suffix(".tmp")
        tmp.write_text(json.dumps(feed, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(FEED)
    top = feed["standings"][0]
    print(f"{feed['as_of']}  {feed['n_teams']} equips, {feed['n_submissions']} lliuraments · líder {top['team']} {top['score']}"
          + ("  (dry)" if dry else ""))
    return feed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", type=int, metavar="S", help="repeteix cada S segons")
    ap.add_argument("--dry", action="store_true", help="no desa el fitxer")
    a = ap.parse_args()
    while True:
        try:
            run(a.dry)
        except Exception as e:                       # una lectura fallida no ha de tombar el bucle
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            if not a.loop:
                sys.exit(1)
        if not a.loop:
            break
        time.sleep(a.loop)
