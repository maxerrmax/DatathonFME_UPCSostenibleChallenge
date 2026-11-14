"""Empaqueta qualsevol agent de estandard/agents/<nom>/ en submissions/<nom>.zip
(sense scripts train*.py) i el valida amb starter_kit/check_submission.py.
Ús: python estandard/package.py mpc_lp"""
import subprocess, sys, zipfile
from pathlib import Path
HERE = Path(__file__).parent
name = sys.argv[1]; src = HERE / "agents" / name
out = HERE / "submissions"; out.mkdir(exist_ok=True); z = out / f"{name}.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    for p in src.iterdir():
        if p.is_file() and not p.name.startswith("train") and p.suffix != ".pyc":
            f.write(p, p.name)
print("creat", z, [n for n in zipfile.ZipFile(z).namelist()])
subprocess.run([sys.executable, str(HERE.parent / "starter_kit" / "check_submission.py"), str(z)],
               cwd=HERE.parent / "starter_kit")
