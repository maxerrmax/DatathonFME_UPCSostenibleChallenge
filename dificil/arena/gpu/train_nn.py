"""Entrenament (PyTorch, GPU si n'hi ha) de les xarxes de previsió CO2/preu del Mode Difícil.

    python train_nn.py --root <arrel_repo> --out <dir> [--epochs 200] [--seeds 5] [--lam 0.5]
                       [--fams mlp,tcn] [--variants f1,f2,f3,all]

Per a cada variant (f1,f2,f3 = plec k sense la setmana de validació; all = tot el train) i cada tasca (co2, price)
entrena un ensemble de `seeds` llavors per família (mlp, tcn) i desa <out>/nn_<variant>.npz (només numpy).
Features: dificil/arena/nn_co2/nnfeat.py (compartit amb l'agent).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="nn_out")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--fams", default="mlp,tcn")
    ap.add_argument("--variants", default="f1,f2,f3,all")
    ap.add_argument("--max_samples", type=int, default=0, help="només proves ràpides")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pretrain", default="", help="CSV REE (Fecha,precio_eur_per_kwh,co2_kg_per_kwh); buit = sense preentrenament")
    ap.add_argument("--pre_epochs", type=int, default=60)
    ap.add_argument("--ft_lr_mult", type=float, default=0.3, help="lr del fine-tuning = lr * mult")
    ap.add_argument("--out_suffix", default="", help='p.ex. "_pre" -> nn_<variant>_pre.npz')
    return ap.parse_args()


ARCH = {
    "mlp": {"fam": "mlp", "hidden": [256, 128], "ln": True, "drop": 0.1},
    "tcn": {"fam": "tcn", "ch": [16, 16, 16, 16], "k": 3, "dil": [1, 2, 4, 8], "hidden": [128], "ln": True, "drop": 0.1},
}


def main():
    a = get_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(root / "dificil")); sys.path.insert(0, str(root / "dificil" / "arena" / "nn_co2"))
    sys.path.insert(0, str(root / "estandard"))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import nnfeat as nf
    from hdata import load_hard
    from folds import split_fold

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, "| torch", torch.__version__, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load_hard()
    full = nf.DenseSeries(df)

    class Net(nn.Module):
        def __init__(self, arch):
            super().__init__()
            self.arch = arch
            self.conv = nn.ModuleList()
            cin = nf.SEQ_C
            if arch["fam"] == "tcn":
                for c in arch["ch"]:
                    self.conv.append(nn.Conv1d(cin, c, arch["k"], dilation=arch["dil"][len(self.conv)]))
                    cin = c
                d = 2 * cin + nf.N_FLAT
            else:
                d = nf.N_IN
            self.lin = nn.ModuleList(); self.lns = nn.ModuleList()
            for hdim in arch["hidden"]:
                self.lin.append(nn.Linear(d, hdim))
                self.lns.append(nn.LayerNorm(hdim) if arch["ln"] else nn.Identity())
                d = hdim
            self.lin.append(nn.Linear(d, nf.NH))
            self.drop = nn.Dropout(arch["drop"])
            self.act = nn.GELU(approximate="tanh")

        def forward(self, x):
            if self.arch["fam"] == "tcn":
                h = x[:, :nf.SEQ_C * nf.L].reshape(-1, nf.SEQ_C, nf.L)
                for i, cv in enumerate(self.conv):
                    k, dl = cv.kernel_size[0], cv.dilation[0]
                    h = self.act(cv(F.pad(h, ((k - 1) * dl, 0))))
                z = torch.cat([h[:, :, -1], h.mean(2), x[:, nf.SEQ_C * nf.L:]], 1)
            else:
                z = x
            for i, lin in enumerate(self.lin[:-1]):
                z = self.drop(self.act(self.lns[i](lin(z))))
            return self.lin[-1](z)

    def export(model, P, pre):
        """Pesos -> dict numpy (Linear transposat a (in,out))."""
        arch = model.arch
        for i, cv in enumerate(model.conv):
            P[f"{pre}cw{i}"] = cv.weight.detach().cpu().numpy().astype(np.float64)
            P[f"{pre}cb{i}"] = cv.bias.detach().cpu().numpy().astype(np.float64)
        for i, lin in enumerate(model.lin):
            P[f"{pre}W{i}"] = lin.weight.detach().cpu().numpy().T.astype(np.float64)
            P[f"{pre}b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float64)
            if i < len(model.lin) - 1 and arch["ln"]:
                P[f"{pre}g{i}"] = model.lns[i].weight.detach().cpu().numpy().astype(np.float64)
                P[f"{pre}be{i}"] = model.lns[i].bias.detach().cpu().numpy().astype(np.float64)

    def samples(ds, ts, task):
        X, Y, T = [], [], []
        for t in ts:
            y = nf.targets(ds, t, task)
            if y is None:
                continue
            x = nf.build(ds, t, task)
            if x is None:
                continue
            X.append(x); Y.append(y); T.append(t)
        return np.array(X, np.float32), np.array(Y, np.float64), np.array(T)

    def epochs_of(d):
        return d.index.values.astype("datetime64[h]").astype(np.int64)

    def pearson_loss(p, y):
        pc = p - p.mean(1, keepdim=True); yc = y - y.mean(1, keepdim=True)
        r = (pc * yc).sum(1) / (pc.norm(dim=1) * yc.norm(dim=1) + 1e-8)
        return (1 - r).mean()

    def train_one(arch, Xtr, Ytr, Xho, Yho, ym, ys, seed, epochs=None, lr=None, init=None):
        epochs = a.epochs if epochs is None else epochs
        lr = a.lr if lr is None else lr
        torch.manual_seed(seed); np.random.seed(seed)
        m = Net(arch).to(dev)
        if init is not None:
            m.load_state_dict(init)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ymt, yst = torch.tensor(ym, dtype=torch.float32, device=dev), torch.tensor(ys, dtype=torch.float32, device=dev)

        def loss_fn(xb, yb):
            p = m(xb)
            hub = F.huber_loss(p, yb, delta=1.0)
            return hub + a.lam * pearson_loss(p * yst + ymt, yb * yst + ymt)

        n = len(Xtr); best, best_sd, bad = 1e9, None, 0
        for ep in range(epochs):
            m.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                if len(idx) < 8:
                    continue
                opt.zero_grad(); l = loss_fn(Xtr[idx], Ytr[idx]); l.backward(); opt.step()
            sch.step()
            m.eval()
            with torch.no_grad():
                hl = float(loss_fn(Xho, Yho))
            if hl < best - 1e-5:
                best, bad = hl, 0
                best_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        m.load_state_dict(best_sd); m.eval()
        return m, best, ep + 1, best_sd

    fams = [f for f in a.fams.split(",") if f]
    report = {}
    tt = lambda z: torch.tensor(z, device=dev)
    FLAT0 = nf.SEQ_C * nf.L
    METEO_COLS = np.arange(FLAT0 + 15, FLAT0 + 19)  # irr, temp, vent, flag faltant (cal 6 + onehot 7 + 2 nivells = 15)

    # ---- preentrenament (REE): dades i normalitzacio fixes, compartides per totes les variants
    PRE = {}
    pre_sd = {}
    if a.pretrain:
        import pandas as pd
        r = pd.read_csv(a.pretrain)
        r["Fecha"] = pd.to_datetime(r["Fecha"])
        if r["Fecha"].dt.tz is not None:
            r["Fecha"] = r["Fecha"].dt.tz_localize(None)
        r = r.drop_duplicates("Fecha").sort_values("Fecha").set_index("Fecha")
        assert r.index.max() < pd.Timestamp("2025-09-01"), "REE ha de ser < 2025-09-01 (abans del test)"
        for c in nf.MAP.values():
            if c not in r.columns:
                r[c] = np.nan  # meteo absent -> faltant, igual que veu l'agent
        r = r[list(nf.MAP.values())].astype(float)
        ds_pre = nf.DenseSeries(r)
        ok = np.isfinite(ds_pre.a["co2"]) & np.isfinite(ds_pre.a["price"])
        cs = np.concatenate([[0], np.cumsum(ok)])
        # finestra completa: hores t-69..t+2 (72 h) i objectius t+3..t+23 -> t-69..t+23 (93 h) totes presents
        cand = np.arange(ds_pre.e0 + 69, ds_pre.e1 - 22)
        full_ok = (cs[cand + 23 - ds_pre.e0 + 1] - cs[cand - 69 - ds_pre.e0]) == 93
        ts_pre = cand[full_ok]
        assert len(ts_pre) > 0, "cap mostra REE valida"
        if a.max_samples:
            ts_pre = ts_pre[:a.max_samples]
        cut_p = ts_pre[int(len(ts_pre) * 0.9)]
        for task in nf.TASKS:
            X, Y, T = samples(ds_pre, ts_pre, task)
            xm = X.mean(0).astype(np.float64); xs = np.maximum(X.std(0), 0.05).astype(np.float64)
            xm[METEO_COLS] = 0.0; xs[METEO_COLS] = 1.0  # meteo (ja escalada per nnfeat) sense info a REE
            ym, ys = Y.mean(0), np.maximum(Y.std(0), 1e-3)
            tr = T + 23 < cut_p; ho = T >= cut_p
            Xn = np.clip((X - xm) / xs, -10, 10).astype(np.float32); Yn = ((Y - ym) / ys).astype(np.float32)
            PRE[task] = dict(xm=xm, xs=xs, ym=ym, ys=ys, Xtr=tt(Xn[tr]), Ytr=tt(Yn[tr]), Xho=tt(Xn[ho]), Yho=tt(Yn[ho]))
            print(f"[pretrain/{task}] REE samples train={tr.sum()} holdout={ho.sum()}", flush=True)

    def pretrained(task, fam, s):
        key = (task, fam, s)
        if key not in pre_sd:
            d = PRE[task]
            _, best, ne, sd = train_one(ARCH[fam], d["Xtr"], d["Ytr"], d["Xho"], d["Yho"], d["ym"], d["ys"], 1000 * s + 7,
                                        epochs=a.pre_epochs)
            pre_sd[key] = {k: v.cpu().clone() for k, v in sd.items()}
            print(f"  [pretrain {task}/{fam} seed{s}] REE holdout loss {best:.4f} ({ne} ep)", flush=True)
        return {k: v.to(dev).clone() for k, v in pre_sd[key].items()}
    for var in a.variants.split(","):
        t0 = time.time()
        if var == "all":
            fit, val = df, None
        else:
            fit, val = split_fold(df, int(var[1:]))
        ds_fit = nf.DenseSeries(fit)
        ts_fit = np.sort(epochs_of(fit))
        P = {}; meta = {"version": 1, "variant": var, "tasks": {}, "price_s": nf.PRICE_S, "lam": a.lam,
                "pretrain": bool(a.pretrain)}
        vres = {}
        for task in nf.TASKS:
            X, Y, T = samples(ds_fit, ts_fit, task)
            if a.max_samples:
                X, Y, T = X[:a.max_samples], Y[:a.max_samples], T[:a.max_samples]
            if a.pretrain:  # mateixa normalitzacio que al preentrenament (fixa, independent de la variant)
                xm, xs, ym, ys = (PRE[task][k] for k in ("xm", "xs", "ym", "ys"))
            else:
                xm, xs = X.mean(0).astype(np.float64), np.maximum(X.std(0), 0.05).astype(np.float64)
                ym, ys = Y.mean(0), np.maximum(Y.std(0), 1e-3)
            # fraccions de temps: holdout = últim 15 % de les hores de fit (amb embargo de 24 h al train)
            cut = ts_fit[int(len(ts_fit) * 0.85)]
            tr = T + 23 < cut; ho = T >= cut
            Xn = np.clip((X - xm) / xs, -10, 10).astype(np.float32)
            Yn = ((Y - ym) / ys).astype(np.float32)
            Xtr, Ytr, Xho, Yho = tt(Xn[tr]), tt(Yn[tr]), tt(Xn[ho]), tt(Yn[ho])
            print(f"[{var}/{task}] samples train={tr.sum()} holdout={ho.sum()}", flush=True)
            P[f"{task}/xmean"], P[f"{task}/xstd"], P[f"{task}/ymean"], P[f"{task}/ystd"] = xm, xs, ym, ys
            mods = []
            for fam in fams:
                for s in range(a.seeds):
                    if a.pretrain:
                        mdl, best, ne, _ = train_one(ARCH[fam], Xtr, Ytr, Xho, Yho, ym, ys, 1000 * s + 7,
                                                     lr=a.lr * a.ft_lr_mult, init=pretrained(task, fam, s))
                    else:
                        mdl, best, ne, _ = train_one(ARCH[fam], Xtr, Ytr, Xho, Yho, ym, ys, 1000 * s + 7)
                    pre = f"{task}/{fam}{s}/"
                    export(mdl, P, pre)
                    mods.append({"fam": fam, "seed": s, "prefix": pre, "arch": ARCH[fam], "holdout": best})
                    print(f"  {fam} seed{s}: holdout loss {best:.4f} ({ne} ep)", flush=True)
            meta["tasks"][task] = {"models": mods}
            if val is not None:
                vres[task] = evaluate(nf, torch, dev, ds_fit, full, val, task, mods, P, xm, xs, ym, ys, fams, a, Net, samples, epochs_of)
        P["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(out / f"nn_{var}{a.out_suffix}.npz", **P)
        report[var] = vres
        print(f"== {var} listo en {time.time() - t0:.0f}s -> {out / ('nn_' + var + a.out_suffix + '.npz')}", flush=True)
    (out / f"report{a.out_suffix}.json").write_text(json.dumps(report, indent=1))


def evaluate(nf, torch, dev, ds_fit, full, val, task, mods, P, xm, xs, ym, ys, fams, a, Net, samples, epochs_of):
    """MAE per horitzó a la validació: naive (últim valor conegut), cada família i ensemble total (via forward numpy)."""
    X, Y, T = samples(full, np.sort(epochs_of(val)), task)
    last = np.array([nf.last_known(full, t, task) for t in T])
    unit = "kg/kWh" if task == "co2" else "EUR/kWh"
    # valors físics
    if task == "co2":
        truth = np.maximum(last[:, None], nf.EPS) * np.exp(Y)
    else:
        truth = last[:, None] + Y * nf.PRICE_S
    Xs = np.clip((X.astype(np.float64) - xm) / xs, -10, 10)
    preds = {}
    for fam in fams + ["ens"]:
        ms = [m for m in mods if fam == "ens" or m["fam"] == fam]
        outs = []
        for i in range(len(Xs)):
            o = [nf.decode(task, nf.forward(m["arch"], P, m["prefix"], Xs[i]) * ys + ym, last[i]) for m in ms]
            outs.append(np.mean(o, axis=0))
        preds[fam] = np.array(outs)
    naive = np.abs(last[:, None] - truth).mean(0)
    res = {"n": int(len(T)), "naive": naive.tolist()}
    print(f"  [validació {task} ({unit}), n={len(T)}] MAE per horitzó 3..23", flush=True)
    print("   h      " + " ".join(f"{h:6d}" for h in range(3, 24)), flush=True)
    print("   naive  " + " ".join(f"{v:6.4f}" for v in naive) + f"  | mitjana {naive.mean():.5f}", flush=True)
    for fam, p in preds.items():
        mae = np.abs(p - truth).mean(0)
        res[fam] = mae.tolist()
        print(f"   {fam:<6} " + " ".join(f"{v:6.4f}" for v in mae) + f"  | mitjana {mae.mean():.5f} ({100 * (1 - mae.mean() / naive.mean()):+.1f}% vs naive)", flush=True)
    return res


if __name__ == "__main__":
    main()
