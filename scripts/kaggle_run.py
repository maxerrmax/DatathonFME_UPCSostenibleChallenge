"""
Executa feina de l'arena a Kaggle (dataset privat jordixu/jamm-arena-bundle, creat amb scripts/kaggle_bundle.sh).

    python scripts/kaggle_run.py arena  stoch_mpc analog_co2 online_rls   # torneig a CPU (v12 s'hi afegeix sol)
    python scripts/kaggle_run.py train_nn                                   # entrenament de xarxes a GPU
    python scripts/kaggle_run.py fetch <slug>                               # espera i baixa la sortida

`arena`: els resultats es fusionen a dificil/arena/results/store_kaggle.json (plataforma separada de la local:
cada execució a Kaggle inclou el v12 de referència, i les comparacions es fan dins del mateix fitxer).
`train_nn`: els .npz es copien a dificil/arena/nn_co2/.
Cal tornar a pujar el bundle (bash scripts/kaggle_bundle.sh) quan canvia el codi d'una estratègia.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "jordixu"
DATASET = f"{USER}/jamm-arena-bundle"
KDIR = ROOT / ".kaggle_stage" / "kernels"

PRELUDE = r'''
import os, sys, glob, json, shutil, subprocess, tarfile, time
W = "/kaggle/working"; T = "/kaggle/tmp"; os.makedirs(T, exist_ok=True)
hits = glob.glob("/kaggle/input/**/bundle.tgz", recursive=True)
if hits:
    tarfile.open(hits[0]).extractall(T); REPO = os.path.join(T, "repo")
else:  # Kaggle pot haver descomprimit l'arxiu
    a = glob.glob("/kaggle/input/**/scripts/arena.py", recursive=True)[0]
    REPO = os.path.join(T, "repo"); shutil.copytree(os.path.dirname(os.path.dirname(a)), REPO)
import numpy, scipy, pandas, lightgbm
print("versions", sys.version, numpy.__version__, scipy.__version__, pandas.__version__, lightgbm.__version__, flush=True)
print("cpus", os.cpu_count(), flush=True)
os.chdir(REPO)
'''

ARENA = PRELUDE + r'''
NAMES = {names!r}
cmd = [sys.executable, "scripts/arena.py", *NAMES, "--procs", str(os.cpu_count() or 4), "--store", "store_kaggle.json"]
t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True); print(r.stdout[-20000:]); print(r.stderr[-5000:])
print("elapsed", round(time.time() - t0), "s")
for f in glob.glob("dificil/arena/results/*kaggle*"):
    shutil.copy(f, W)
'''

TRAIN = PRELUDE + r'''
subprocess.run("nvidia-smi", shell=True)
out = os.path.join(W, "nn"); os.makedirs(out, exist_ok=True)
cmd = [sys.executable, "dificil/arena/gpu/train_nn.py", "--root", REPO, "--out", out, *{extra!r}]
t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True); print(r.stdout[-30000:]); print(r.stderr[-8000:])
print("elapsed", round(time.time() - t0), "s")
'''


def push(slug, code, gpu):
    d = KDIR / slug
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "run.py").write_text(code, encoding="utf-8")
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": gpu, "enable_internet": False, "dataset_sources": [DATASET],
            "competition_sources": [], "kernel_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], cwd=d, check=True)
    print("pushed", slug)


def fetch(slug, kind):
    while True:
        r = subprocess.run(["kaggle", "kernels", "status", f"{USER}/{slug}"], capture_output=True, text=True)
        s = (r.stdout + r.stderr).lower(); print(time.strftime("%H:%M:%S"), s.strip()[-120:], flush=True)
        if any(x in s for x in ("complete", "error", "cancel")):
            break
        time.sleep(30)
    out = KDIR / slug / "output"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    subprocess.run(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", "."], cwd=out)
    log = list(out.glob("*.log"))
    if log:
        txt = log[0].read_text(encoding="utf-8", errors="replace")
        print(txt[-6000:])
    res = ROOT / "dificil" / "arena" / "results"; res.mkdir(parents=True, exist_ok=True)
    if kind == "arena":
        sf = res / "store_kaggle.json"
        store = json.loads(sf.read_text()) if sf.exists() else {}
        for f in out.rglob("store_kaggle.json"):
            store.update(json.loads(f.read_text()))
        sf.write_text(json.dumps(store))
        for f in out.rglob("leaderboard_kaggle.txt"):
            print(f.read_text(encoding="utf-8"))
    else:
        dst = ROOT / "dificil" / "arena" / "nn_co2"
        for f in out.rglob("*.npz"):
            shutil.copy(f, dst / f.name); print("copiat", f.name)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    kind = sys.argv[1]
    if kind == "arena":
        names = sys.argv[2:]
        slug = "jamm-arena-cpu"
        push(slug, ARENA.format(names=names), gpu=False); fetch(slug, "arena")
    elif kind == "train_nn":
        slug = "jamm-nn-train"
        push(slug, TRAIN.format(extra=sys.argv[2:]), gpu=True); fetch(slug, "train_nn")
    elif kind == "fetch":
        slug = sys.argv[2]
        fetch(slug, "arena" if "arena" in slug else "train_nn")
