"""
dashboard/refresh.py
Regenera, amb l'estat actual del repo, tot el que el dashboard llegeix de `model_data/`. Els passos són pesats i
s'executen UN darrere l'altre (CLAUDE.md: mai dos processos pesats alhora).

    python dashboard/refresh.py                  # arena + dimensionament, traces de tots els agents, valor de la informació
    python dashboard/refresh.py --only traces    # només un pas: arena | traces | voi
    python dashboard/refresh.py --procs 4        # processos de les traces (1 fil cadascun)

Què llegeix el dashboard i d'on surt:
    model_data/arena.json     <- dificil/arena/results/store*.json (les genera `python scripts/arena.py`) + l'òptim per plec
    model_data/sizing.json    <- òptim amb previsió perfecta per a diverses mides de bateria
    model_data/traces.parquet i runs.json  <- cada estratègia de l'arena executada hora a hora al plec 1
    model_data/voi.json       <- dificil/diag/voi_folds.py (valor de la informació perfecta per entrada)
    submissions.json          <- registre manual dels lliuraments (id, validació, nota pública, CO₂ i cost públics)
    live_feed.json            <- rànquing públic (el pot escriure `live.write_feed` o qualsevol script)
Només submissions.json i live_feed.json s'omplen a mà (o amb un script); la resta es recalcula d'aquí.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable


def run(*cmd: str) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run([PY, *cmd], cwd=ROOT, check=True)


if __name__ == "__main__":
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    procs = sys.argv[sys.argv.index("--procs") + 1] if "--procs" in sys.argv else "4"
    if only in (None, "arena"):
        run("dashboard/build_models.py", "--no-traces")
    if only in (None, "traces"):
        run("dashboard/build_traces.py", "--procs", procs)
    if only in (None, "voi"):
        run("dificil/diag/voi_folds.py", "v12", "net_h0", "net_all", "co2_all", "price_all")
    print("\nFet. Recorda actualitzar dashboard/submissions.json (lliuraments nous) i dashboard/live_feed.json (rànquing).")
