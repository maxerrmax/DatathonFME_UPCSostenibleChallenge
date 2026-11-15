"""Prova de fum amb els models de lliurament i dates de test (dades substitutes): errors, infraccions, temps per pas."""
from common import *
import pandas as pd, time, henv
d = sys.argv[1]; df = load_hard(); m = load_agent(S + "/" + d)
W = []
for y, mo in [(2025, 9), (2025, 10), (2025, 11), (2025, 12)] + [(2026, k) for k in range(1, 9)]:
    nxt = np.datetime64(f"{y + (mo == 12)}-{(mo % 12) + 1:02d}-01T00", "h"); e = int(nxt.astype(np.int64)) - 1; W.append((e - 167, e))
parts = []
for i, (a, b) in enumerate(W):
    idx = pd.date_range(np.datetime64(a, "h"), np.datetime64(b, "h"), freq="1h")
    src = df.iloc[(np.arange(168) + 24 * 30 * i) % len(df)]
    parts.append(pd.DataFrame(src.values, columns=df.columns, index=idx))
T = pd.concat(parts); T.index.name = "Fecha"
t0 = time.time(); ag = m.BatteryAgent(); ti = time.time() - t0
extra = {k: getattr(ag, k) is not None for k in ("_an",) if hasattr(ag, k)}
env = henv.HardEnv(T.reset_index()); o = env.reset(); dts = []; dn = False
while not dn:
    t1 = time.perf_counter(); a = ag.act(o); dts.append(time.perf_counter() - t1); o, dn = env.step(a)
r = env.evaluate_results(); dts = np.array(dts) * 1e3
print(f"{d}: score {r['score']} infr {r['infraccions']} init {ti:.2f}s ms/pas {dts.mean():.1f} p99 {np.percentile(dts,99):.1f} max {dts.max():.1f} total {dts.sum()/1e3:.1f}s {extra}")
