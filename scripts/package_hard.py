"""Crea submission_hard/submission.zip (agent.py + models + params) a l'arrel del zip."""
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "submission_hard"
files = ["agent.py", "params.json"] + [f"model_{t}.txt" for t in ("p", "c", "load", "pv")]
out = D / "submission.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(D / f, f)
print(out, f"{out.stat().st_size/1e6:.1f} MB", files)
