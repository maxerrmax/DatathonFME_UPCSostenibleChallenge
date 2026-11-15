"""
dashboard/live.py  ·  LA PORTA de les dades en directe.

Tot el que canvia durant el datathon (rànquing de la web del repte i, si es vol, les notes públiques dels nostres
lliuraments) entra al dashboard per aquí i només per aquí. Per connectar una font nova no cal tocar cap pàgina:
afegiu una funció a `SOURCES` que retorni un dict amb l'esquema d'avall (o `None` si no pot respondre).

Ordre de prova (la primera que respon guanya):
  1. HTTP  : URL a l'entorn `DASHBOARD_FEED_URL` o a `.streamlit/secrets.toml` → `[live] url = "..."`
             (qualsevol servei que retorni el JSON de sota; `[live] token` s'envia com a Bearer, opcional).
  2. Fitxer: `dashboard/live_feed.json`. És el punt de connexió més simple: un script, un cron o una persona
             hi escriu el JSON (vegeu `write_feed`) i el dashboard el recull en uns segons (es mira la data del fitxer).
  3. Registre: `submissions.json → leaderboard` (còpia estàtica, només com a darrer recurs; es marca com a «antic»).

Esquema del feed (tots els camps són opcionals tret de `standings`):
  {
    "as_of": "2026-11-15T09:00:00",          # quan es va llegir el rànquing (ISO)
    "n_teams": 51, "n_submissions": 358,
    "us": "JAMM",                              # nom del nostre equip al rànquing
    "standings": [{"rank": 1, "team": "...", "score": 24.36, "co2": 22.94, "cost": 27.69}, ...],
    "submissions": {"v19": {"public": 23.61, "co2": 22.4, "cost": 26.5}}   # notes públiques dels NOSTRES lliuraments
  }
`submissions` es fusiona amb `submissions.json`: en omplir-hi `public` d'una versió, la tria automàtica
(`models.choose`) i les regles es recalculen soles.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import models

HERE = Path(__file__).resolve().parent
FEED_FILE = HERE / "live_feed.json"
STALE_AFTER_S = 15 * 60          # passat aquest temps, la font es mostra com a «antiga»
REFRESH_EVERY = "30s"            # cada quant es refresquen les peces en directe


@dataclass
class Live:
    registry: dict
    standings: list = field(default_factory=list)
    n_teams: int | None = None
    n_submissions: int | None = None
    us: str | None = None
    as_of: datetime | None = None
    source: str = "cap"
    ok: bool = False             # hi ha dades de rànquing fiables
    error: str | None = None

    @property
    def age_s(self) -> float | None:
        return None if self.as_of is None else max(0.0, (datetime.now() - self.as_of).total_seconds())

    @property
    def stale(self) -> bool:
        return self.age_s is None or self.age_s > STALE_AFTER_S


# ----------------------------------------------------------------------------------------------- fonts
def _secret(key: str):
    try:
        import streamlit as st
        return dict(st.secrets.get("live", {})).get(key)
    except Exception:
        return None


def _from_http():
    url = os.environ.get("DASHBOARD_FEED_URL") or _secret("url")
    if not url:
        return None
    import requests
    tok = os.environ.get("DASHBOARD_FEED_TOKEN") or _secret("token")
    r = requests.get(url, timeout=6, headers={"Authorization": f"Bearer {tok}"} if tok else {})
    r.raise_for_status()
    return r.json() | {"_source": "HTTP"}


def _from_file():
    if not FEED_FILE.exists():
        return None
    d = json.loads(FEED_FILE.read_text(encoding="utf-8"))
    d.setdefault("as_of", datetime.fromtimestamp(FEED_FILE.stat().st_mtime).isoformat())
    return d | {"_source": "fitxer"}


def _from_registry():
    lb = models.registry().get("leaderboard", {})
    if not lb.get("standings"):
        return None
    return {"standings": lb["standings"], "n_teams": lb.get("n_teams"), "n_submissions": lb.get("n_submissions"),
            "us": lb.get("us"), "as_of": None, "_source": "registre (antic)"}


SOURCES = [_from_http, _from_file, _from_registry]     # <- afegiu aquí una font nova


# ----------------------------------------------------------------------------------------------- API
def _parse_ts(v):
    try:
        return datetime.fromisoformat(v) if v else None
    except ValueError:
        return None


def load() -> Live:
    """Llegeix la primera font que respon i fusiona les notes dels nostres lliuraments al registre."""
    reg = models.registry()
    err = None
    for fn in SOURCES:
        try:
            d = fn()
        except Exception as e:                     # una font caiguda no ha de tombar el dashboard
            err = f"{fn.__name__[6:]}: {type(e).__name__}"
            continue
        if not d or not d.get("standings"):
            continue
        for sid, patch in (d.get("submissions") or {}).items():
            for s in reg["submissions"]:
                if s["id"] == sid:
                    s.update({k: v for k, v in patch.items() if v is not None})
        reg["leaderboard"] = {**reg.get("leaderboard", {}), "standings": d["standings"],
                              "teams": d["standings"][:5], "us": d.get("us"),
                              "n_teams": d.get("n_teams"), "n_submissions": d.get("n_submissions")}
        return Live(reg, d["standings"], d.get("n_teams"), d.get("n_submissions"), d.get("us"),
                    _parse_ts(d.get("as_of")), d["_source"], True, err)
    return Live(reg, error=err or "cap font disponible")


def write_feed(standings: list, *, us: str, n_teams: int | None = None, n_submissions: int | None = None,
               submissions: dict | None = None, path: Path = FEED_FILE) -> Path:
    """Escriu el feed (és el que ha de cridar el futur connector). Escriptura atòmica perquè el dashboard mai llegeixi mig fitxer."""
    payload = {"as_of": datetime.now().isoformat(timespec="seconds"), "us": us, "n_teams": n_teams,
               "n_submissions": n_submissions, "standings": standings}
    if submissions:
        payload["submissions"] = submissions
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def ago(seconds: float | None) -> str:
    if seconds is None:
        return "data desconeguda"
    s = int(seconds)
    return f"fa {s} s" if s < 90 else f"fa {s // 60} min" if s < 5400 else f"fa {s // 3600} h"


if __name__ == "__main__":                          # python dashboard/live.py  → mostra què es veuria
    L = load()
    print(L.source, L.ok, L.n_teams, len(L.standings), ago(L.age_s), L.error)
