import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"dificil"), str(ROOT/"estandard"), str(ROOT/"dificil"/"arena"), str(ROOT/"scripts")]
from arena import fold_data
from horacle import oracle, ETA
for k in (1, 2, 3):
    val = fold_data(k); R = pickle.load(open(ROOT/f"dificil/diag/rec_f{k}.pkl", "rb"))
    net = (val.load_kw-val.pv_kw).values; co2 = val.co2_kg_per_kwh.values; pr = val.precio_eur_per_kwh.values
    base = np.maximum(net, 0); B = (base*co2).sum(); C = (base*pr).sum()
    w = 100*(0.7*co2/B+0.3*pr/C)
    o, Po = oracle(val)
    soc = R["soc"]; s_prev = np.concatenate([[3000], soc[:-1]]); ds = soc-s_prev
    Pa = np.where(ds > 0, ds/ETA, ds*ETA)
    ia = np.maximum(net+Pa, 0); io = np.maximum(net+Po, 0)
    ga = (base-ia)*w; go = (base-io)*w
    so = 3000+np.cumsum(np.where(Po > 0, Po*ETA, Po/ETA))
    print(f"fold{k}: agent {ga.sum():.2f} oracle {go.sum():.2f} gap {go.sum()-ga.sum():.2f}  endSOC agent {soc[-1]:.0f} oracle {so[-1]:.0f}")
    gid = (val.index.to_series().diff() != pd.Timedelta("1h")).cumsum().values
    ends = [np.where(gid == g)[0][-1] for g in np.unique(gid)]
    print("   agent  soc@week-end", soc[ends].round(-2).astype(int).tolist())
    print("   oracle soc@week-end", so[ends].round(-2).astype(int).tolist())
    gw = pd.Series(go-ga).groupby(gid).sum().round(2).tolist()
    print("   gap per week", gw)
    hod = val.index.hour
    df = pd.DataFrame({"h": hod, "gap": go-ga, "Pa": Pa, "Po": Po})
    print(df.groupby("h").agg(gap=("gap", "sum"), Pa=("Pa", "mean"), Po=("Po", "mean")).round(2).T.to_string())
    print("   discharge thru agent %.0f oracle %.0f; spilled agent %.0f oracle %.0f" % (
        (-Pa[Pa < 0]).sum(), (-Po[Po < 0]).sum(), np.maximum(-(net+Pa), 0)[Pa < 0].sum(), np.maximum(-(net+Po), 0)[Po < 0].sum()))
    print("   PV surplus %.0f; charged from surplus agent %.0f oracle %.0f; grid charge agent %.0f oracle %.0f" % (
        np.maximum(-net, 0).sum(), np.minimum(np.maximum(Pa, 0), np.maximum(-net, 0)).sum(),
        np.minimum(np.maximum(Po, 0), np.maximum(-net, 0)).sum(),
        np.maximum(np.maximum(Pa, 0)-np.maximum(-net, 0), 0).sum(), np.maximum(np.maximum(Po, 0)-np.maximum(-net, 0), 0).sum()))
