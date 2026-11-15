"""
Construeix data/external/ree_hourly.csv (Fecha, precio_eur_per_kwh, co2_kg_per_kwh) a partir de la descàrrega
de scripts/download_ree_history.py. El CO2 es reconstrueix del mix (0,37 * gen. amb emissions / gen. total) i es
calibra linealment contra data/train/co2_red.csv al període solapat. Només s'exporten hores < 2025-09-01.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
CON = ["cc", "car", "gf", "cogenResto", "vap"]
SIN = ["eol", "nuc", "hid", "sol", "bio"]
CUT = pd.Timestamp("2025-09-01")


def gen_hourly():
    rows = []
    for f in sorted((EXT / "ree_gen_5min").glob("*.json")):
        try:
            rows += json.loads(f.read_text(encoding="utf-8"))["valoresHorariosGeneracion"]
        except Exception:
            pass
    g = pd.DataFrame(rows)
    g = g[~g["ts"].str.contains("2B:")]                 # canvi d'hora d'octubre: es queda la primera 02:00
    g["ts"] = g["ts"].str.replace("2A:", "02:", regex=False)
    g = g.drop_duplicates("ts")
    g["ts"] = pd.to_datetime(g["ts"]); g = g.set_index("ts").sort_index()
    for c in CON + SIN:
        if c not in g:
            g[c] = 0.0
    h = g[CON + SIN].apply(pd.to_numeric, errors="coerce").resample("1h").mean()
    h = h[g[CON + SIN].resample("1h").count().min(axis=1) >= 6]           # hores amb prou mostres de 5 min
    con, sin = h[CON].sum(axis=1), h[SIN].sum(axis=1)
    return pd.DataFrame({"x": 0.37 * con / (con + sin), "con": con / 1e4, "sin": sin / 1e4})


def pvpc():
    out = []
    for f in sorted((EXT / "ree_pvpc").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for x in d.get("included", []):
            if x["type"] == "PVPC":
                for v in x["attributes"]["values"]:
                    out.append((pd.Timestamp(v["datetime"][:19]), v["value"] / 1000.0))
    return pd.Series(dict(out), name="precio_eur_per_kwh").sort_index()


if __name__ == "__main__":
    G = gen_hourly()
    c = pd.read_csv(ROOT / "data/train/co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")["co2_kg_per_kwh"]
    J = G.join(c, how="inner").dropna()
    X = np.column_stack([np.ones(len(J)), J.x, J.con, J.sin]); y = J.co2_kg_per_kwh.values
    n = len(J) // 2
    beta = np.linalg.lstsq(X[:n], y[:n], rcond=None)[0]
    e_raw, e_cal = np.abs(J.x.values[n:] - y[n:]), np.abs(X[n:] @ beta - y[n:])
    print(f"solapament {len(J)} h | MAE cru {e_raw.mean():.5f} -> calibrat (2a meitat) {e_cal.mean():.5f}"
          f" | corr {np.corrcoef(X[n:] @ beta, y[n:])[0, 1]:.4f} | mitjana co2 {y.mean():.4f}")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    Gp = G[G.index < CUT]
    co2 = np.clip(np.column_stack([np.ones(len(Gp)), Gp.x, Gp.con, Gp.sin]) @ beta, 0.0, 1.0)
    out = pd.DataFrame({"co2_kg_per_kwh": co2}, index=Gp.index).join(pvpc(), how="inner")
    out = out[out.index < CUT].dropna(); out.index.name = "Fecha"
    assert out.index.max() < CUT
    out.to_csv(EXT / "ree_hourly.csv")
    p = pd.read_csv(ROOT / "data/train/precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")["precio_eur_per_kwh"]
    chk = pvpc().to_frame("a").join(p.rename("b"), how="inner")
    print(f"PVPC solapat {len(chk)} h, error màx {np.abs(chk.a - chk.b).max():.6f}")
    print(f"desat {len(out)} h, {out.index.min()} .. {out.index.max()}")
