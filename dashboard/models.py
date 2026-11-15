"""
dashboard/models.py
Loads the model-comparison data: the hand-kept submission record (submissions.json) and the files
written by build_models.py (arena results, hour-by-hour traces, battery sizing).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "model_data"
REF = "v12"


def registry() -> dict:
    return json.loads((HERE / "submissions.json").read_text(encoding="utf-8"))


def _json(name: str):
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def arena_raw():
    return _json("arena.json")


def runs():
    return _json("runs.json") or {}


def sizing() -> pd.DataFrame | None:
    d = _json("sizing.json")
    return pd.DataFrame(d) if d else None


def traces() -> pd.DataFrame | None:
    p = DATA / "traces.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).set_index("Fecha")


def arena_table(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (one row per strategy, weekly score table strategy x week) for the 3 validation folds.

    The comparison with the reference (v12) is paired by week: same weeks, same data, so the
    week-to-week noise of the season cancels out and only the difference between agents is left.
    """
    strat, folds = raw["strategies"], sorted(raw["oracle"])
    weekly = {}
    for n, d in strat.items():
        weekly[n] = {f"{f}|{w}": s for f in folds for w, s in d[f]["weeks"].items()}
    weekly["optimum"] = {f"{f}|{w}": s for f in folds for w, s in raw["oracle"][f]["weeks"].items()}
    W = pd.DataFrame(weekly).T
    W = W[sorted(W.columns, key=lambda c: c.split("|")[1])]
    ref = W.loc[REF]
    rows = []
    for n, d in strat.items():
        diff = (W.loc[n] - ref).dropna().to_numpy()
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 2 else np.nan
        sc = [d[f]["score"] for f in folds]
        rows.append({
            "strategy": n, "mean": float(np.mean(sc)),
            **{f"fold{f}": d[f]["score"] for f in folds},
            "delta": float(np.mean(sc) - np.mean([strat[REF][f]["score"] for f in folds])),
            "worst_fold": float(min(d[f]["score"] - strat[REF][f]["score"] for f in folds)),
            "wk_delta": float(diff.mean()), "ci": float(1.96 * se) if se == se else np.nan,
            "wins": int((diff > 1e-9).sum()), "weeks": int(len(diff)),
            "co2": float(np.mean([d[f]["co2"] for f in folds])),
            "cost": float(np.mean([d[f]["cost"] for f in folds])),
            "infr": int(sum(d[f]["infr"] for f in folds)),
        })
    T = pd.DataFrame(rows).set_index("strategy").sort_values("mean", ascending=False)
    T["significant"] = (T["wk_delta"].abs() > T["ci"]) & (T.index != REF)
    T["all_folds"] = T["worst_fold"] > 0
    return T, W


def score_weights(tr: pd.DataFrame) -> pd.Series:
    """Score points per kWh imported at each hour (the official objective, linearised).

    score = 100 * (0.7 * dCO2/CO2_base + 0.3 * dCost/Cost_base)  ->  every kWh not imported at hour t
    is worth w_t points. Summing w_t * (import_A - import_B) gives the exact score gap between A and B.
    """
    one = tr[tr["model"] == tr["model"].iloc[0]]
    base = (one["load_kw"] - one["pv_kw"]).clip(lower=0)
    return 100 * (0.7 * one["co2_kg_per_kwh"] / (base * one["co2_kg_per_kwh"]).sum()
                  + 0.3 * one["precio_eur_per_kwh"] / (base * one["precio_eur_per_kwh"]).sum())


# --------------------------------------------------------------------------- horari fix ("Prova-ho tu")
CAP, PMAX, ETA = 6000.0, 1500.0, 0.9 ** 0.5


def fold_context(tr: pd.DataFrame) -> pd.DataFrame:
    """Hores del plec 1 amb demanda, solar, preu, CO2 i el pes de cada kWh en punts."""
    ctx = tr[tr["model"] == "optimum"][["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]].copy()
    ctx["net"] = ctx["load_kw"] - ctx["pv_kw"]
    ctx["w"] = score_weights(tr).to_numpy()
    return ctx


def week_ids(idx: pd.DatetimeIndex) -> np.ndarray:
    return (pd.Series(idx).diff() != pd.Timedelta("1h")).cumsum().to_numpy()


def score_power(ctx: pd.DataFrame, power: np.ndarray) -> dict:
    """Puntuació oficial d'una sèrie de potències de bateria ja factibles (kW, + càrrega)."""
    net = ctx["net"].to_numpy()
    base, grid = np.maximum(net, 0), np.maximum(net + power, 0)
    co2, pr = ctx["co2_kg_per_kwh"].to_numpy(), ctx["precio_eur_per_kwh"].to_numpy()
    d_co2 = 1 - (grid * co2).sum() / (base * co2).sum()
    d_cost = 1 - (grid * pr).sum() / (base * pr).sum()
    gid = week_ids(ctx.index)
    weeks = {}
    for g in np.unique(gid):
        m = gid == g
        weeks[str(ctx.index[m][0].date())] = float(
            100 * (0.7 * (1 - (grid[m] * co2[m]).sum() / (base[m] * co2[m]).sum())
                   + 0.3 * (1 - (grid[m] * pr[m]).sum() / (base[m] * pr[m]).sum())))
    return {"score": float(100 * (0.7 * d_co2 + 0.3 * d_cost)), "co2": float(100 * d_co2), "cost": float(100 * d_cost),
            "weeks": weeks}


def timetable_run(ctx: pd.DataFrame, charge_hours, discharge_hours) -> dict:
    """Simula un horari fix: cada dia carrega a potència màxima a `charge_hours` i descarrega a `discharge_hours`.

    Amb avantatge per a l'horari: la descàrrega mai supera la demanda neta real de l'hora (no malbarata energia
    exportant) i mai comet infraccions (la potència s'ajusta al que cap a la bateria).
    """
    ch, dis = set(int(h) for h in charge_hours), set(int(h) for h in discharge_hours)
    net, hours = ctx["net"].to_numpy(), ctx.index.hour.to_numpy()
    soc, power, socs = CAP / 2, np.zeros(len(ctx)), np.zeros(len(ctx))
    for t in range(len(ctx)):
        h = hours[t]
        if h in ch and h not in dis:
            p = min(PMAX, (CAP - soc) / ETA)
            soc += p * ETA
            power[t] = p
        elif h in dis and h not in ch:
            p = min(PMAX, soc * ETA, max(net[t], 0.0))
            soc -= p / ETA
            power[t] = -p
        socs[t] = soc
    out = score_power(ctx, power)
    out["power"], out["soc"] = power, socs
    out["cycles"] = float(np.clip(-power, 0, None).sum() / ETA / CAP)
    return out


# --------------------------------------------------------------------------- condicions per substituir la versió actual
def _f(x, dec=2, sign=False):
    t = f"{x:+,.{dec}f}" if sign else f"{x:,.{dec}f}"
    return t.replace(",", " ").replace(".", ",").replace(" ", ".").replace("-", "−")


def gates(reg: dict, T: pd.DataFrame, runs_: dict, cand: str) -> tuple[list[tuple], str, dict | None]:
    """Condicions (nom, compleix, detall) perquè l'estratègia `cand` substitueixi la versió lliurada, i la conclusió.

    És l'única definició de la regla de decisió: la fan servir la pàgina Veredicte i l'assistent.
    """
    me, lim = reg["selected"], reg["limits"]
    sel = next(s for s in reg["submissions"] if s["id"] == me)
    sub = next((s for s in reg["submissions"] if s.get("arena") == cand), None)
    r = T.loc[cand]
    has_pub = bool(sub and sub.get("public"))
    pub = (f"públic {_f(sub['public'])}, {_f(sub['public'] - sel['public'], 2, True)} respecte a la {me}"
           if has_pub else "encara no s'ha enviat")
    cand_ms, ref_ms = (runs_.get(cand) or {}).get("ms"), (runs_.get(REF) or {}).get("ms")
    cand_s = cand_ms * lim["test_hours"] / 1000 if cand_ms else None
    ref_s = ref_ms * lim["test_hours"] / 1000 if ref_ms else None
    g = [
        (f"Supera la {REF} a tots els plecs", bool(r["all_folds"]), f"pitjor plec {_f(r['worst_fold'], 2, True)}"),
        ("Guanya almenys el 60 % de les setmanes", bool(r["wins"] / r["weeks"] >= 0.6), f"{int(r['wins'])} de {int(r['weeks'])}"),
        ("El guany supera l'interval del 95 %", bool(r["significant"] and r["wk_delta"] > 0),
         f"{_f(r['wk_delta'], 2, True)} ± {_f(r['ci'])}"),
        *([(f"Cap dins dels {lim['timeout_s']} s", bool(cand_s <= lim["timeout_s"]),
            f"{_f(cand_s, 0)} s aquí, {_f(cand_s / ref_s, 1)}× la {REF}")] if cand_s and ref_s else []),
        ("Confirmat al test públic", bool(has_pub and sub["public"] > sel["public"]), pub),
    ]
    n_ok = sum(x[1] for x in g)
    concl = (f"Substituïm la {me}." if n_ok == len(g) else
             "Val la pena gastar-hi un dels lliuraments del dia." if not has_pub and n_ok == len(g) - 1 else
             f"Encara no: mantenim la {me}." if not has_pub else f"Mantenim la {me}.")
    return g, concl, sub
