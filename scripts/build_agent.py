"""
Genera submission_hard/agent.py autocontingut (sense imports locals) a partir de:
  src/forecast.py  +  src/optim.py (solve_dispatch)  +  submission_hard/agent_core.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def strip_header(src):
    # treu docstring inicial i imports (es reafegeixen a dalt)
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)
    lines = [l for l in src.splitlines() if not re.match(r"^(import |from )", l)]
    return "\n".join(lines).strip() + "\n"


def main():
    forecast = strip_header((ROOT / "src/forecast.py").read_text())
    optim = strip_header((ROOT / "src/optim.py").read_text())
    optim = optim.split("def score_weights")[0]
    core = strip_header((ROOT / "submission_hard/agent_core.py").read_text())
    header = '''"""
agent.py — Mode Difícil (Repte UPC Sostenible, Datathon 2026)
MPC de 24 h amb previsions LightGBM de preu, CO2, consum i solar + blindatge de dades.
Fitxer generat per scripts/build_agent.py: no editar a mà.
"""
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

try:
    import lightgbm as lgb
except Exception:
    lgb = None
'''
    out = header + "\n\n" + forecast + "\n\n" + optim + "\n\n" + core
    (ROOT / "submission_hard/agent.py").write_text(out)
    print("escrit submission_hard/agent.py", len(out), "bytes")


if __name__ == "__main__":
    main()
